# TurboQuantDB: RAM & Disk Reduction Plan

## Target

Reduce per-vector RAM from **~4 KB → ~1 KB** at runtime so that a 16 GB VDI can serve
~9–14M vectors per Axon project (currently ~2.5M). Disk stays at ~700 B/vec (already good).

### Use-case context

- Source dataset: 1 TB–10 TB of raw documents ingested through Axon
- Estimated vector count: ~50M–100M per TB of text (at ~2 KB/chunk)
- Query machine: 16 GB VDI, ~10 GB available for DB after OS + Python
- Strategy: split into per-project shards ≤ project RAM budget each

### Measured baseline (10:46 build, 100k vectors)

| Item | Measured |
|------|----------|
| Total process RAM delta | +400 MB |
| Marginal cost per vector | ~3.5–4.4 KB |
| Disk per vector | ~697 B |

---

## Root-cause breakdown (~4 KB/vector)

| Source | Estimated cost |
|--------|----------------|
| `live_records: HashMap<String, SegmentRecord>` — 4 separate heap allocs per record + Windows HeapAlloc fragmentation | **~2.5 KB** |
| `MetadataStore` redb B-tree in-memory pages | **~400 B** |
| `index_ids: Vec<String>` (one String per graph node) | **~60 B** |
| `graph adjacency: Vec<Vec<u32>>` per node | **~160 B** |
| Quantizer matrices, WAL, misc | **~100 B** |
| **Total** | **~3,200 B** (balance is allocator block headers + hashbrown table) |

---

## Phase 1 — Flat Code Array *(biggest win: ~2–2.5 KB/vector)*

### Problem

`live_records: HashMap<String, SegmentRecord>` makes **4 heap allocations per vector**:
1. `String` key in hashbrown table
2. `SegmentRecord.id: String` (duplicate of key)
3. `SegmentRecord.quantized_indices: Vec<u8>` (384 bytes)
4. `SegmentRecord.qjl_bits: Vec<u8>` (48 bytes)

On Windows, each small heap allocation carries ~16–32 B of `HeapAlloc` metadata overhead
plus alignment padding. At 100k records, that is 400k allocations. The heap fragmentation
alone accounts for ~1.5 KB/vector above the raw data size.

### Solution

Replace the four-alloc pattern with a single contiguous byte slab:

```rust
// BEFORE (engine.rs struct fields):
live_records: HashMap<String, SegmentRecord>,

// AFTER:
/// Flat code slab: slot `i` occupies bytes [i*STRIDE .. i*STRIDE+STRIDE].
/// Layout: [384 B quantized_indices | 48 B qjl_bits | 4 B gamma f32 | 1 B is_deleted]
live_codes: Vec<u8>,
/// Forward map: user-provided id → slot index in live_codes
live_id_to_slot: HashMap<String, u32>,
/// Reverse map: slot → id. `None` means the slot was deleted (tombstone).
live_slot_to_id: Vec<Option<String>>,
```

**Constants to add at top of engine.rs:**

```rust
const LIVE_INDICES_LEN: usize = 384;   // quantized_indices bytes per vector
const LIVE_QJL_LEN:     usize = 48;    // qjl_bits bytes per vector (d.div_ceil(8))
const LIVE_GAMMA_OFF:   usize = LIVE_INDICES_LEN + LIVE_QJL_LEN;          // 432
const LIVE_DELETED_OFF: usize = LIVE_GAMMA_OFF + 4;                        // 436
const LIVE_STRIDE:      usize = LIVE_DELETED_OFF + 1;                      // 437
```

> **Note**: `LIVE_INDICES_LEN` and `LIVE_QJL_LEN` depend on `d` and `b`.
> At runtime derive them as `self.d` and `self.d.div_ceil(8)`.
> Hard-coding is fine for the common case (d=384, b=8); add a runtime assert in `open()`.

### Helper methods to add to TurboQuantEngine

```rust
/// Append a new slot; returns the slot index.
fn live_alloc_slot(&mut self, id: String, indices: &[u8], qjl: &[u8], gamma: f32) -> u32 {
    let slot = self.live_slot_to_id.len() as u32;
    let qjl_len = self.d.div_ceil(8);
    // Extend slab
    self.live_codes.extend_from_slice(indices);
    self.live_codes.extend_from_slice(qjl);
    self.live_codes.extend_from_slice(&gamma.to_le_bytes());
    self.live_codes.push(0u8); // is_deleted = false
    // Update maps
    self.live_id_to_slot.insert(id.clone(), slot);
    self.live_slot_to_id.push(Some(id));
    slot
}

/// Soft-delete a slot (mark is_deleted = 1).
fn live_delete_slot(&mut self, id: &str) {
    if let Some(&slot) = self.live_id_to_slot.get(id) {
        let off = slot as usize * LIVE_STRIDE + LIVE_DELETED_OFF;
        self.live_codes[off] = 1;
        self.live_slot_to_id[slot as usize] = None;
        self.live_id_to_slot.remove(id);
    }
}

/// Borrow raw code slice for a slot (bounds-checked).
fn live_codes_at(&self, slot: usize) -> (&[u8], &[u8], f32) {
    let qjl_len = self.d.div_ceil(8);
    let base = slot * LIVE_STRIDE;
    let indices = &self.live_codes[base .. base + self.d];
    let qjl    = &self.live_codes[base + self.d .. base + self.d + qjl_len];
    let gamma  = f32::from_le_bytes(
        self.live_codes[base + LIVE_GAMMA_OFF .. base + LIVE_GAMMA_OFF + 4]
            .try_into().unwrap()
    );
    (indices, qjl, gamma)
}

/// Compact: rebuild slab dropping all soft-deleted slots (call after flush).
fn live_compact_slab(&mut self) {
    let mut new_codes   = Vec::with_capacity(self.live_id_to_slot.len() * LIVE_STRIDE);
    let mut new_ids     = Vec::with_capacity(self.live_id_to_slot.len());
    let mut new_id_map  = HashMap::with_capacity(self.live_id_to_slot.len());

    for (old_slot, maybe_id) in self.live_slot_to_id.iter().enumerate() {
        if let Some(id) = maybe_id {
            let new_slot = new_ids.len() as u32;
            new_id_map.insert(id.clone(), new_slot);
            new_ids.push(Some(id.clone()));
            let base = old_slot * LIVE_STRIDE;
            new_codes.extend_from_slice(&self.live_codes[base .. base + LIVE_STRIDE]);
        }
    }
    self.live_codes      = new_codes;
    self.live_slot_to_id = new_ids;
    self.live_id_to_slot = new_id_map;
}
```

### Call-site changes (engine.rs)

**`write_many` (line ~688)** — replace `live_records.insert`:
```rust
// BEFORE:
for (id, record) in live_updates {
    self.live_records.insert(id, record);
}

// AFTER:
for (id, indices, qjl, gamma) in live_updates {
    if self.live_id_to_slot.contains_key(&id) {
        // upsert: overwrite existing slot in-place
        if let Some(&slot) = self.live_id_to_slot.get(&id) {
            let base = slot as usize * LIVE_STRIDE;
            self.live_codes[base..base + self.d].copy_from_slice(&indices);
            self.live_codes[base + self.d..base + self.d + qjl.len()].copy_from_slice(&qjl);
            self.live_codes[base + LIVE_GAMMA_OFF..base + LIVE_GAMMA_OFF + 4]
                .copy_from_slice(&gamma.to_le_bytes());
            self.live_codes[base + LIVE_DELETED_OFF] = 0;
        }
    } else {
        self.live_alloc_slot(id, &indices, &qjl, gamma);
    }
}
```

**`flush_wal_to_segment` (line ~571)** — replace HashMap insert/remove:
```rust
// BEFORE:
for record in &records {
    if record.is_deleted {
        self.live_records.remove(&record.id);
        ...
    }
    self.live_records.insert(record.id.clone(), record.clone());
}

// AFTER:
for record in &records {
    if record.is_deleted {
        self.live_delete_slot(&record.id);
        self.live_vectors.remove(&record.id);
        continue;
    }
    // alloc or overwrite
    self.live_alloc_slot(record.id.clone(), &record.quantized_indices,
                         &record.qjl_bits, record.gamma);
    self.live_vectors.remove(&record.id);
}
// Compact after every flush to reclaim soft-deleted slots
self.live_compact_slab();
```

**`delete` (line ~360)** — replace `live_records.remove`:
```rust
self.live_delete_slot(&id);
```

**`vector_count`** — replace `live_records.len()`:
```rust
self.live_id_to_slot.len() as u64
```

**`create_index_with_params` (line ~393)** — replace records iteration:
```rust
// BEFORE:
let mut records: Vec<SegmentRecord> = self.live_records.values().cloned().collect();
records.sort_by(|a, b| a.id.cmp(&b.id));
let indexed_ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
let all_vectors: Vec<Array1<f64>> = indexed_ids.par_iter().map(|id| {
    self.live_vectors.get(id).cloned().unwrap_or_else(|| {
        let r = &self.live_records[id];
        self.quantizer.dequantize(&r.quantized_indices, &r.qjl_bits, r.gamma as f64)
    })
}).collect();

// AFTER:
let mut id_slot_pairs: Vec<(String, usize)> = self.live_id_to_slot
    .iter().map(|(id, &slot)| (id.clone(), slot as usize)).collect();
id_slot_pairs.sort_by(|a, b| a.0.cmp(&b.0));

let indexed_ids: Vec<String> = id_slot_pairs.iter().map(|(id, _)| id.clone()).collect();
let all_vectors: Vec<Array1<f64>> = id_slot_pairs.par_iter().map(|(id, &slot)| {
    self.live_vectors.get(id).cloned().unwrap_or_else(|| {
        let (indices, qjl, gamma) = self.live_codes_at(slot);
        self.quantizer.dequantize(indices, qjl, gamma as f64)
    })
}).collect();
```

**`search_with_filter_and_ann` ADC branch (line ~483)** — replace HashMap lookup:
```rust
// BEFORE:
let r = &self.live_records[&self.index_ids[node as usize]];
self.quantizer.score_ip_encoded(&prep, &r.quantized_indices, &r.qjl_bits, r.gamma as f64)

// AFTER (requires ann_slots: Vec<u32> built during create_index, same order as index_ids):
let slot = self.ann_slots[node as usize];
let (indices, qjl, gamma) = self.live_codes_at(slot as usize);
self.quantizer.score_ip_encoded(&prep, indices, qjl, gamma as f64)
```

Add `ann_slots: Vec<u32>` to the engine struct — populated in `create_index_with_params`
alongside `index_ids` by recording the slot for each indexed ID.

**`rebuild_live_records_cache`** — replace with equivalent that builds the flat slab:
```rust
fn rebuild_live_codes(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    self.live_codes.clear();
    self.live_id_to_slot.clear();
    self.live_slot_to_id.clear();
    self.live_vectors.clear();

    let mut by_id: HashMap<String, SegmentRecord> = HashMap::new();
    for record in self.segments.iter_all_records()? {
        by_id.insert(record.id.clone(), record);
    }
    for entry in &self.wal_buffer {
        by_id.insert(entry.id.clone(), SegmentRecord {
            id: entry.id.clone(),
            quantized_indices: entry.quantized_indices.clone(),
            qjl_bits: entry.qjl_bits.clone(),
            gamma: entry.gamma,
            is_deleted: entry.is_deleted,
        });
    }
    // Insert non-deleted records in sorted order (deterministic slot assignment)
    let mut records: Vec<SegmentRecord> = by_id.into_values()
        .filter(|r| !r.is_deleted).collect();
    records.sort_by(|a, b| a.id.cmp(&b.id));

    for r in records {
        self.live_alloc_slot(r.id, &r.quantized_indices, &r.qjl_bits, r.gamma);
    }
    Ok(())
}
```

### Expected RAM after Phase 1

| Item | Before | After |
|------|--------|-------|
| Per-vector in slab | — | 437 B (one big alloc) |
| `live_id_to_slot` entry | 60 B key + 4 B value + hashbrown overhead | ~80 B |
| `live_slot_to_id` entry | — | `Option<String>` ~60 B |
| Eliminated Vec allocs | 4 allocs × ~400 B avg | 0 |
| **Marginal per vector** | **~2,500 B** | **~577 B** |

---

## Phase 2 — Replace redb with Flat Binary Metadata *(~400 B/vector)*

### Problem

`MetadataStore` uses `RedbDatabase` (an embedded B-tree). B-trees allocate in 4 KB
pages. At 100 B of metadata per record, only 2.5% of each page is payload; the rest is
B-tree structural overhead. redb also maintains an in-memory page cache.

### Solution

Replace redb with a fully in-memory `HashMap<String, VectorMetadata>` that flushes to a
flat length-prefixed binary file. Metadata is only needed at query time for the top-k
results (typically 10), so full in-memory residence is fine and O(1) lookup is fast.

**New `MetadataStore` (metadata.rs):**

```rust
pub struct MetadataStore {
    path: PathBuf,
    data: HashMap<String, VectorMetadata>,
    dirty: bool,
}

impl MetadataStore {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = PathBuf::from(path);
        let data = if path.exists() {
            Self::load_from_file(&path)?
        } else {
            HashMap::new()
        };
        Ok(Self { path, data, dirty: false })
    }

    pub fn put(&mut self, id: &str, meta: &VectorMetadata) -> Result<...> {
        self.data.insert(id.to_string(), meta.clone());
        self.dirty = true;
        Ok(())
    }

    pub fn put_many(&mut self, entries: &[(String, VectorMetadata)]) -> Result<...> {
        for (id, meta) in entries {
            self.data.insert(id.clone(), meta.clone());
        }
        self.dirty = true;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<VectorMetadata>, ...> {
        Ok(self.data.get(id).cloned())
    }

    pub fn get_many(&self, ids: &[String]) -> Result<HashMap<String, VectorMetadata>, ...> {
        Ok(ids.iter().filter_map(|id| {
            self.data.get(id).map(|m| (id.clone(), m.clone()))
        }).collect())
    }

    pub fn delete(&mut self, id: &str) -> Result<...> {
        self.data.remove(id);
        self.dirty = true;
        Ok(())
    }

    /// Flush to disk atomically (write to .tmp, then rename).
    pub fn flush(&mut self) -> Result<...> {
        if !self.dirty { return Ok(()); }
        let tmp = self.path.with_extension("tmp");
        let mut f = BufWriter::new(File::create(&tmp)?);
        // Header: entry count
        f.write_all(&(self.data.len() as u64).to_le_bytes())?;
        for (id, meta) in &self.data {
            let id_bytes = id.as_bytes();
            let json = serde_json::to_vec(meta)?;
            f.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
            f.write_all(id_bytes)?;
            f.write_all(&(json.len() as u32).to_le_bytes())?;
            f.write_all(&json)?;
        }
        f.flush()?;
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    fn load_from_file(path: &Path) -> Result<HashMap<String, VectorMetadata>, ...> {
        let bytes = std::fs::read(path)?;
        let mut cur = Cursor::new(bytes);
        let count = u64::from_le_bytes(read_bytes::<8>(&mut cur)?) as usize;
        let mut map = HashMap::with_capacity(count);
        for _ in 0..count {
            let id_len = u32::from_le_bytes(read_bytes::<4>(&mut cur)?) as usize;
            let mut id_bytes = vec![0u8; id_len];
            cur.read_exact(&mut id_bytes)?;
            let id = String::from_utf8(id_bytes)?;
            let json_len = u32::from_le_bytes(read_bytes::<4>(&mut cur)?) as usize;
            let mut json = vec![0u8; json_len];
            cur.read_exact(&mut json)?;
            let meta: VectorMetadata = serde_json::from_slice(&json)?;
            map.insert(id, meta);
        }
        Ok(map)
    }
}
```

**Call `metadata.flush()` in `engine.flush_wal_to_segment()` and `engine.close()`.**

Change `MetadataStore::put` / `put_many` signature from `&self` to `&mut self` throughout
engine.rs (currently redb takes `&self` because redb handles its own interior mutability).

### Expected savings

- RAM: eliminates B-tree page overhead (~400 B/vector at 100k scale)
- Disk: replaces `metadata.redb` (B-tree file with 4 KB page granularity) with a flat
  binary file; saves ~300–500 B/vector

---

## Phase 3 — Binary Graph ID File *(disk savings, minor RAM)*

### Problem

`index_ids: Vec<String>` is persisted as `graph_ids.json` — a JSON array of UUID strings.
Each UUID as JSON string: `"550e8400-e29b-41d4-a716-446655440000"` = 38 bytes + JSON
overhead = ~42 bytes per entry. At 100k nodes: 4.2 MB on disk, loaded into memory as
`Vec<String>` (100k × ~60 B heap alloc = 6 MB RAM).

### Solution

**On-disk format** (`graph_ids.bin`, replaces `graph_ids.json`):

```
[8B: node_count]
For each node in order:
  [2B: id_len][id_len bytes: UTF-8 id]
```

**Serialize / deserialize helpers (engine.rs):**

```rust
const INDEX_IDS_FILE: &str = "graph_ids.bin"; // was "graph_ids.json"

fn serialize_index_ids(ids: &[String]) -> Result<Vec<u8>, ...> {
    let mut out = Vec::with_capacity(8 + ids.iter().map(|s| 2 + s.len()).sum::<usize>());
    out.extend_from_slice(&(ids.len() as u64).to_le_bytes());
    for id in ids {
        let bytes = id.as_bytes();
        out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

fn deserialize_index_ids(data: &[u8]) -> Result<Vec<String>, ...> {
    let mut cur = Cursor::new(data);
    let count = u64::from_le_bytes(read_bytes::<8>(&mut cur)?) as usize;
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u16::from_le_bytes(read_bytes::<2>(&mut cur)?) as usize;
        let mut buf = vec![0u8; len];
        cur.read_exact(&mut buf)?;
        ids.push(String::from_utf8(buf)?);
    }
    Ok(ids)
}
```

Also change `maybe_persist_state` to call `serde_json::to_vec_pretty(&self.index_ids)` →
`serialize_index_ids(&self.index_ids)` and write to `graph_ids.bin`.

For backward compatibility on `open()`: attempt to read `graph_ids.bin` first, fall back
to `graph_ids.json` if not present.

### Expected savings

- Disk: UUID format (36 chars) → 38 bytes raw vs 42 B JSON = ~10% per entry. More
  importantly: for non-UUID string IDs with short names, savings are larger.
- RAM: `Vec<String>` unchanged; no RAM benefit unless IDs are replaced with `Box<str>`
  (saves 8 B/entry vs String — worthwhile for > 10M entries: saves 80 MB at 10M).

---

## Phase 4 — Fixed-stride Segment File *(disk savings ~100 B/vector)*

### Problem

`segment.rs` uses `bincode::serialize(record)` per record. Each `Vec<u8>` field is
prefixed with an 8-byte length (bincode default). For `quantized_indices` (384 B) and
`qjl_bits` (48 B), that adds 16 bytes overhead per record. `String id` adds another 8 B.
Total bincode overhead: ~30–40 B per record.

### Solution

Replace bincode with a fixed-stride binary format:

**Segment file layout:**
```
Header (16 bytes):
  [8B: record_count]
  [2B: d (dimension)]
  [2B: qjl_len = d.div_ceil(8)]
  [4B: reserved / version]

Per record (variable only for id):
  [2B: id_len]
  [id_len bytes: UTF-8 id]
  [d bytes: quantized_indices]
  [qjl_len bytes: qjl_bits]
  [4B: gamma f32 LE]
  [1B: is_deleted]
```

**New `Segment::write_batch` (segment.rs):**

```rust
pub fn write_batch(backend: &StorageBackend, name: &str, records: &[SegmentRecord])
    -> Result<Self, ...>
{
    if records.is_empty() { ... }
    let d       = records[0].quantized_indices.len();
    let qjl_len = records[0].qjl_bits.len();

    let mut buf = Vec::new();
    buf.extend_from_slice(&(records.len() as u64).to_le_bytes()); // record_count
    buf.extend_from_slice(&(d as u16).to_le_bytes());              // dimension
    buf.extend_from_slice(&(qjl_len as u16).to_le_bytes());        // qjl_len
    buf.extend_from_slice(&0u32.to_le_bytes());                    // reserved

    for r in records {
        let id_bytes = r.id.as_bytes();
        buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(id_bytes);
        buf.extend_from_slice(&r.quantized_indices);
        buf.extend_from_slice(&r.qjl_bits);
        buf.extend_from_slice(&r.gamma.to_le_bytes());
        buf.push(r.is_deleted as u8);
    }
    backend.write(name, buf)?;
    Ok(Self { name: name.to_string(), record_count: records.len() })
}
```

Add a version byte to the header and keep the bincode reader as a fallback for opening
old segment files, removing it once all data is migrated.

### Expected savings

- ~30–40 B/vector on disk (~4% of current 700 B/vec)
- Faster segment reads: no bincode allocation per record

---

## Implementation Order & Expected Outcome

| Phase | Change | RAM saving/vec | Disk saving/vec | Effort |
|-------|--------|---------------|-----------------|--------|
| **1** | Flat code slab | **~2.2 KB** | — | 4–8 hrs |
| **2** | Flat binary metadata | **~400 B** | ~400 B | 2–4 hrs |
| **3** | Binary graph IDs | ~0 | ~10 B | 1 hr |
| **4** | Fixed-stride segment | — | ~35 B | 2–3 hrs |
| **Total** | | **~2.6 KB/vec** | **~445 B/vec** | ~10–15 hrs |

### Projected per-vector cost after all phases

| Item | After all phases |
|------|-----------------|
| Flat code slab | 437 B |
| `live_id_to_slot` entry | ~80 B |
| `live_slot_to_id` entry | ~60 B |
| `ann_slots: Vec<u32>` | 4 B |
| `index_ids: Vec<String>` | ~60 B |
| Graph adjacency (unchanged) | ~160 B |
| MetadataStore HashMap entry | ~150 B |
| **Total** | **~951 B ≈ ~1 KB/vector** |

### VDI capacity after all phases (16 GB, ~10 GB usable)

| Scale | RAM (after) | Fits in 16 GB VDI? |
|-------|------------|-------------------|
| 1M    | ~1 GB      | ✓ |
| 5M    | ~5 GB      | ✓ |
| 10M   | ~10 GB     | ✓ (tight) |
| 14M   | ~13.5 GB   | ✓ (max safe) |
| 25M   | ~24 GB     | ✗ |

### Sharding strategy for 1 TB–10 TB in Axon

At ~1 KB/vector after refactor and ~14M vectors/project max on a 16 GB VDI:

| Raw data | Approx vectors | Axon projects needed |
|----------|---------------|----------------------|
| 1 TB     | ~50M          | 4–5 projects |
| 5 TB     | ~250M         | 18–20 projects |
| 10 TB    | ~500M         | 36–40 projects |

Disk per project: ~14M × 700 B ≈ 10 GB. All projects for 1 TB: ~50 GB disk total.

---

## Correctness Notes

1. **Soft-delete + compact**: `live_delete_slot` marks the slot without shifting indices.
   `live_compact_slab()` must be called after every `flush_wal_to_segment` to reclaim
   memory. Between flush and compact, `live_id_to_slot.len()` is authoritative for
   live vector count.

2. **`ann_slots` invalidation**: `ann_slots` must be cleared in `invalidate_index_state()`
   alongside `index_ids`. They go stale together.

3. **MetadataStore flush timing**: call `metadata.flush()` in `flush_wal_to_segment()`
   and `close()` — the same places WAL truncation happens. Do NOT flush metadata on
   every `put()` (too expensive for large batch inserts).

4. **Segment backward compatibility**: keep the bincode reader in `Segment::read_all`
   behind a header version check. Detect old format by checking if byte 8 (first byte
   after the 8-byte count) is a valid `d` value (e.g. ≤ 4096) vs a bincode-encoded
   string length. On first compaction after upgrade, old segments are automatically
   rewritten in the new format.

5. **Thread safety**: `MetadataStore.data: HashMap` is `&mut self` — the engine's
   existing `RwLock<TurboQuantEngine>` already serializes access. No additional locking
   needed.

---

## Benchmarking After Each Phase

```
cd C:\dev\TurboQuantDB
cargo build --release
pip install --force-reinstall target_py\wheels\turboquantdb-*.whl
python Qualification\ClaudeQual\tqdb_comparison_bench.py
```

Target after Phase 1+2: RAM @ 100k ≤ 80 MB, query p50 ≤ 1 ms.
Target after all phases: RAM @ 100k ≤ 60 MB, disk @ 100k ≤ 65 MB.
