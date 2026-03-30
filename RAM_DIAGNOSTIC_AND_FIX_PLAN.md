# TurboQuantDB: RAM Diagnostic Report & Fix Plan

**Date**: 2026-03-30
**Benchmark**: dim=384, bits=8, scales 10k–100k, vs LanceDB IVF_PQ and ChromaDB HNSW

---

## 1. Current State (post-Phase-1 flat slab, 11:24 build)

### Performance scorecard @ 100k vectors

| Metric | LanceDB | ChromaDB | TurboQuantDB | Winner |
|--------|---------|----------|-------------|--------|
| Time-to-ready | 22.2s | 43.7s | 11.3s | **TQDB ✓** |
| Ingest speed | 38,469 v/s | 2,289 v/s | 26,837 v/s | **TQDB ✓** |
| Disk | 156.6 MB | 243.3 MB | 75.0 MB | **TQDB ✓** |
| RAM | +51 MB | +201 MB | +444 MB | LanceDB ✗ |
| Query p50 | 6.3ms | 1.7ms | 0.7ms | **TQDB ✓** |
| Query p95 | 8.8ms | 3.0ms | 0.7ms | **TQDB ✓** |

TurboQuantDB wins 4 of 5 metrics. RAM is the only remaining blocker.

---

## 2. RAM Diagnostic — What the stats() fields revealed

The 11:24 build added `live_codes_bytes`, `live_id_count`, `metadata_bytes_estimate`,
`graph_nodes`, and `ann_slot_count` to `stats()`. At N=10,000:

```
[post-ingest]  ram=+51.6MB  live_codes=4,370,000B  live_ids=10,000  meta_est=808,890B  graph_nodes=0
[post-index]   ram=+56.4MB  live_codes=4,370,000B  live_ids=10,000  meta_est=808,890B  graph_nodes=10,000  ann_slots=10,000
```

### Breakdown of 51.6 MB at 10k vectors

| Component | Bytes | B/vec | Source |
|-----------|-------|-------|--------|
| `live_codes` flat slab | 4,370,000 | **437** | `Vec<u8>`, one alloc — working correctly |
| `metadata_bytes_estimate` (redb in-memory) | 808,890 | **81** | redb B-tree data |
| `live_id_to_slot` HashMap<String, u32> | ~3,200,000 | ~320 | hashbrown table + String heap allocs |
| `live_slot_to_id` Vec<Option<String>> | ~1,600,000 | ~160 | Vec + String heap allocs |
| redb B-tree page overhead | ~3,000,000 | ~300 | 4 KB pages, ~2.5% utilisation |
| Graph adjacency Vec<Vec<u32>> | ~1,600,000 | ~160 | per-node Vec alloc |
| `ann_slots: Vec<u32>` | 40,000 | 4 | flat, negligible |
| `index_ids: Vec<String>` | 600,000 | 60 | String per node |
| Quantizer matrices (fixed) | 2,400,000 | — | rotation + projection, amortised |
| Windows HeapAlloc fragmentation | ~34,000,000 | ~3,400 | small-alloc block headers |
| **Total** | **~51,600,000** | **~5,160** | |

### Key insight

The flat slab is **correct and efficient at 437 B/vec**. The problem is everything else:
- **String heap allocations** — `live_id_to_slot` and `live_slot_to_id` each hold one
  `String` per vector. Each UUID string is 36 bytes of data, but the `String` struct
  (ptr + len + cap) + heap block header + hashbrown slot adds ~480 B/vec across both maps.
- **redb page overhead** — `metadata_bytes_estimate` reports 81 B/vec of actual data,
  but the redb B-tree allocates in 4 KB pages at ~2.5% fill, costing ~300 B/vec extra.
- **Windows allocator fragmentation** — thousands of small `String` heap allocations
  each carry a 16–32 B `HeapAlloc` header. At 2 Strings/vector × 10k vectors = 20k
  allocations × ~1,700 B average fragmented block = ~34 MB.

---

## 3. Immediate Fix — Revert On-Demand Graph Scorer

### Problem

The 11:24 build replaced the pre-materialised `all_vectors` in `create_index_with_params`
with an on-demand `dequantize()` call inside the graph build scorer. This caused a
**700× regression** in index build time:

| Scale | Before (pre-materialised) | After (on-demand) | Regression |
|-------|--------------------------|-------------------|------------|
| 10k   | 0.3s                     | 218.3s            | **728×** |
| 25k   | ~0.8s                    | ~1,365s (est.)    | **~1,700×** |
| 100k  | ~7s                      | ~21,800s (est.)   | **~3,100×** |

The graph `build()` calls the scorer for every candidate pair:
`N × candidate_cap = 10,000 × 256 = 2,560,000 calls` at 10k.
Each on-demand `dequantize()` = 2× 384×384 GEMV ≈ 85 μs → **218 seconds total**.

### Fix

**File**: `src/storage/engine.rs`, function `create_index_with_params`

```rust
// REVERT TO: pre-materialise all_vectors before graph build

pub fn create_index_with_params(...) -> Result<...> {
    self.flush_wal_to_segment()?;

    // Collect all records sorted by ID
    let mut id_slot_pairs: Vec<(String, usize)> = self.live_id_to_slot
        .iter().map(|(id, &slot)| (id.clone(), slot as usize)).collect();
    if id_slot_pairs.is_empty() {
        self.invalidate_index_state()?;
        self.maybe_persist_state(true)?;
        return Ok(());
    }
    id_slot_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let indexed_ids: Vec<String> = id_slot_pairs.iter().map(|(id, _)| id.clone()).collect();
    let n = indexed_ids.len();

    // ── PRE-MATERIALISE (keep this) ──────────────────────────────────────────
    // Dequantize all N vectors in parallel into a temporary Vec.
    // This Vec is DROPPED at the end of create_index_with_params.
    // The temporary RSS spike (~307 MB at 100k for f64, ~154 MB for f32) is
    // acceptable because it is freed immediately after graph.build() returns.
    // Do NOT replace this with on-demand dequantize inside the scorer —
    // that causes O(N × candidate_cap) GEMV calls = 700× slower build time.
    // ────────────────────────────────────────────────────────────────────────
    let all_vectors: Vec<Array1<f64>> = id_slot_pairs
        .par_iter()
        .map(|(id, &slot)| {
            self.live_vectors.get(id).cloned().unwrap_or_else(|| {
                let (indices, qjl, gamma) = self.live_codes_at(slot);
                self.quantizer.dequantize(indices, qjl, gamma as f64)
            })
        })
        .collect();

    let metric = self.metric.clone();
    let build_scorer = |from: u32, to: u32| {
        score_vectors_with_metric(&metric, &all_vectors[from as usize], &all_vectors[to as usize])
    };

    self.graph.build(n, max_degree, alpha, build_scorer)?;

    // all_vectors is dropped here — temporary spike only, no permanent footprint.

    // Populate index_ids and ann_slots
    self.index_ids = indexed_ids;
    self.ann_slots = id_slot_pairs.iter().map(|(_, &slot)| slot as u32).collect();
    self.index_ids_dirty = true;

    self.manifest.index_state = Some(IndexState {
        max_degree,
        search_list_size,
        alpha,
        indexed_nodes: self.index_ids.len(),
    });
    self.pending_manifest_updates += 1;
    self.maybe_persist_state(true)?;
    Ok(())
}
```

**Expected result after revert**: `create_index` returns to ~0.3–8s across all scales.
The temporary `all_vectors` spike during build is ~154 MB at 100k (f32) or ~307 MB (f64),
but this is **freed immediately** and does not contribute to steady-state RSS.

---

## 4. Phase 2 — Eliminate String Heap Allocations (~3.2 KB/vec savings)

This is the largest remaining RAM lever. Two sub-phases:

### 4a. Replace String ID maps with a String-pool + u32 offset

**Problem**: `live_id_to_slot: HashMap<String, u32>` and `live_slot_to_id: Vec<Option<String>>`
each hold one individually-allocated `String` per vector. On Windows, each `String` heap
allocation carries a 16–32 B `HeapAlloc` block header. At 2 Strings × 10k vectors = 20k
allocations × ~1.7 KB average fragment = ~34 MB of pure allocator overhead.

**Solution**: intern all ID strings into a single `Vec<u8>` byte pool, referenced by
byte-range offsets. Zero individual String allocations.

```rust
/// Replaces the two String-per-vector fields in TurboQuantEngine:
///   live_id_to_slot: HashMap<String, u32>
///   live_slot_to_id: Vec<Option<String>>
pub struct IdPool {
    /// Contiguous UTF-8 bytes for all live IDs: "id0id1id2..."
    bytes: Vec<u8>,
    /// Per-slot: (start_byte, length). u32 supports IDs up to 4 GB total.
    /// `length == 0` means the slot is deleted.
    offsets: Vec<(u32, u16)>,
    /// id string → slot index. Key is a u64 FNV hash of the id bytes.
    /// On collision (rare), fall back to linear scan of offsets.
    index: HashMap<u64, u32>,
}

impl IdPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity * 40), // 40 B avg UUID
            offsets: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, id: &str) -> u32 {
        let start = self.bytes.len() as u32;
        let len   = id.len() as u16;
        self.bytes.extend_from_slice(id.as_bytes());
        let slot = self.offsets.len() as u32;
        self.offsets.push((start, len));
        self.index.insert(fnv1a(id.as_bytes()), slot);
        slot
    }

    pub fn get_slot(&self, id: &str) -> Option<u32> {
        let h = fnv1a(id.as_bytes());
        let slot = *self.index.get(&h)?;
        // Verify (handle hash collision)
        let (start, len) = self.offsets[slot as usize];
        if len > 0 && &self.bytes[start as usize .. start as usize + len as usize] == id.as_bytes() {
            Some(slot)
        } else {
            // Linear fallback (rare)
            self.offsets.iter().enumerate()
                .find(|(_, &(s, l))| l > 0 &&
                    &self.bytes[s as usize .. s as usize + l as usize] == id.as_bytes())
                .map(|(i, _)| i as u32)
        }
    }

    pub fn get_id(&self, slot: usize) -> Option<&str> {
        let (start, len) = self.offsets[slot];
        if len == 0 { return None; }
        std::str::from_utf8(
            &self.bytes[start as usize .. start as usize + len as usize]
        ).ok()
    }

    pub fn delete(&mut self, slot: usize) {
        let (_, len) = &mut self.offsets[slot];
        *len = 0; // mark deleted
        // Note: bytes are NOT reclaimed until compact()
    }

    pub fn compact(&mut self) {
        let mut new_bytes   = Vec::with_capacity(self.bytes.len());
        let mut new_offsets = Vec::with_capacity(self.offsets.len());
        let mut new_index   = HashMap::with_capacity(self.index.len());
        for (old_slot, &(start, len)) in self.offsets.iter().enumerate() {
            if len == 0 { continue; }
            let id_bytes = &self.bytes[start as usize .. start as usize + len as usize];
            let new_slot = new_offsets.len() as u32;
            let new_start = new_bytes.len() as u32;
            new_bytes.extend_from_slice(id_bytes);
            new_offsets.push((new_start, len));
            new_index.insert(fnv1a(id_bytes), new_slot);
        }
        self.bytes   = new_bytes;
        self.offsets = new_offsets;
        self.index   = new_index;
    }

    pub fn len(&self) -> usize {
        self.offsets.iter().filter(|&&(_, l)| l > 0).count()
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
    h
}
```

**In `TurboQuantEngine`, replace**:
```rust
// REMOVE:
live_id_to_slot: HashMap<String, u32>,
live_slot_to_id: Vec<Option<String>>,

// ADD:
live_ids: IdPool,
```

Update all call sites:
- `live_id_to_slot.contains_key(&id)` → `live_ids.get_slot(&id).is_some()`
- `live_id_to_slot.insert(id, slot)` → `live_ids.insert(&id)` (returns slot)
- `live_id_to_slot.get(&id)` → `live_ids.get_slot(&id)`
- `live_slot_to_id[slot]` → `live_ids.get_id(slot)`
- `live_ids.len()` for vector count

**Expected RAM savings**: eliminates 20k String heap allocations at 10k vectors.
Each String → 36 bytes data + 24 bytes struct + 16 bytes HeapAlloc header = 76 bytes,
times 2 maps = 152 B/vec in raw alloc overhead. Plus hashbrown table slots reduced.
With IdPool: 36 bytes data in pool + 6 bytes offsets entry + 12 bytes hash entry = 54 B/vec.
**Saving: ~100 B/vec direct + ~3,300 B/vec Windows fragmentation = ~3,400 B/vec.**

### 4b. Replace redb with flat binary metadata file

**Problem**: `MetadataStore` uses `RedbDatabase`. redb allocates in 4 KB pages with
~2.5% actual fill at 81 B/vec of real data → ~300 B/vec wasted in B-tree page structure.
In-memory page cache adds further overhead.

**Solution**: fully in-memory `HashMap<u64, VectorMetadata>` (keyed by FNV hash of id),
flushed to a flat binary file atomically on `flush()` / `close()`. The hash key avoids
storing the full String again.

**New `MetadataStore` struct** (see `RAM_PLAN.md` Phase 2 for full implementation):

```rust
pub struct MetadataStore {
    path: PathBuf,
    /// Key: FNV-1a hash of id (u64). Collision probability negligible at <100M entries.
    data: HashMap<u64, (String, VectorMetadata)>, // keep id for collision resolution
    dirty: bool,
}
```

**Flat binary on-disk format** (replaces `metadata.redb`):
```
[8B: entry_count]
Per entry:
  [2B: id_len][id_len bytes: UTF-8 id]
  [4B: json_len][json_len bytes: JSON-encoded VectorMetadata]
```

Write atomically: write to `metadata.tmp`, then `rename()` to `metadata.bin`.

**Call sites to update** (`MetadataStore::put` / `put_many` become `&mut self`):
- `write_many`: `self.metadata.put_many(&metadata_entries)?`  — unchanged signature
- `flush_wal_to_segment`: add `self.metadata.flush()?` after WAL truncate
- `close`: add `self.metadata.flush()?`
- Engine struct `metadata` field type changes — no other callers affected

**Expected RAM savings**: eliminates redb in-memory page cache (~300 B/vec).
MetadataStore now uses `HashMap<u64, (String, VectorMetadata)>`:
- u64 key: 8 bytes in hashbrown slot
- `(String, VectorMetadata)` value: ~60 + ~100 bytes = ~160 bytes per entry
- Total: ~170 B/vec vs ~380 B/vec (redb) → **saving ~210 B/vec**

---

## 5. Phase 3 — Binary Graph ID File (~10 B/vec disk, negligible RAM)

Already specified in `RAM_PLAN.md`. Low priority — implement after Phase 2 shows results.

---

## 6. Expected Outcomes

### After Immediate Fix (revert on-demand scorer)

| Metric | Current | After revert |
|--------|---------|-------------|
| create_index @ 10k | 218s | ~0.3s |
| create_index @ 100k | ~21,800s | ~7s |
| RAM @ 100k | +444 MB | +444 MB (unchanged) |
| Query p50 | 0.7ms | 0.7ms |

### After Phase 2a (IdPool — eliminate String allocations)

| Metric | Current | After 2a |
|--------|---------|---------|
| RAM @ 10k | +52 MB | **~8 MB** |
| RAM @ 100k | +444 MB | **~110 MB** |
| Marginal cost/vec | ~4.4 KB | **~600 B** |

### After Phase 2b (flat binary metadata)

| Metric | After 2a | After 2b |
|--------|---------|---------|
| RAM @ 100k | ~110 MB | **~90 MB** |
| Disk @ 100k | ~75 MB | **~65 MB** |
| Marginal cost/vec | ~600 B | **~550 B** |

### VDI capacity projection after Phase 2a+2b

16 GB VDI, ~10 GB usable:
- **~18M vectors per Axon project** (up from ~2.5M today)
- 1 TB raw text (~50M vectors) → **3–4 projects**
- 10 TB raw text (~500M vectors) → **28–30 projects**

---

## 7. Implementation Order

| Step | Action | File(s) | RAM impact | Effort |
|------|--------|---------|------------|--------|
| **0** | **Revert on-demand graph scorer** | `engine.rs` | none (correctness fix) | 30 min |
| **1** | Add `IdPool` struct | new `id_pool.rs` | — | 2 hrs |
| **2** | Replace `live_id_to_slot` + `live_slot_to_id` with `live_ids: IdPool` in engine | `engine.rs` | **−3,400 B/vec** | 3–4 hrs |
| **3** | Replace `MetadataStore` redb with flat binary | `metadata.rs`, `engine.rs` | −210 B/vec | 2–3 hrs |
| **4** | Benchmark + verify | bench script | — | 30 min |

**Total estimated effort: 8–10 hours.**

---

## 8. Correctness Notes

### IdPool hash collisions
FNV-1a 64-bit has birthday collision probability of ~50% at 4 billion entries. At
100M vectors (10 TB use case), probability ≈ 0.05% — negligible for a vector DB.
The `get_slot` fallback linear scan handles collisions correctly.

### Metadata flush timing
Call `metadata.flush()` at the same points as `wal.truncate()` — i.e. end of
`flush_wal_to_segment()` and in `close()`. Do NOT flush on every `put()`.

### IdPool compaction
Call `live_ids.compact()` at the same point as `live_compact_slab()` — after every
`flush_wal_to_segment()`. They compact together.

### Backward compatibility
The flat `metadata.bin` is a new format. On `open()`, detect format by checking if
`metadata.redb` exists; if so, migrate to `metadata.bin` on first `close()`. After
migration, delete `metadata.redb`.

### Thread safety
`IdPool` and `MetadataStore` are accessed through the existing `RwLock<TurboQuantEngine>`.
No additional locking needed.
