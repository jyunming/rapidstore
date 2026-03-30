# TurboQuantDB: Rearchitecting Plan to 30k v/s

## Current Performance Diagnosis

At `bits=8`, `dim=384`, each call to `ProdQuantizer::quantize()` does:

| Step | File | Cost per vector |
|------|------|----------------|
| `DMatrix` allocation + memcpy | `mse.rs:45-48` | 384 copies |
| Rotation multiply `Π × x` | `mse.rs:51` | 384² = **147k scalar MADs** (no BLAS) |
| Centroid search | `mse.rs:55-68` | 384 × 128 = **49k comparisons** (linear scan) |
| Residual subtraction | `prod.rs:39-42` | 384 scalar subs |
| Gamma L2 norm | `prod.rs:45-49` | 384 scalar ops |
| `DMatrix` alloc + memcpy again | `qjl.rs:29-32` | 384 copies |
| QJL projection `S × r` | `qjl.rs:35` | 384² = **147k scalar MADs** (no BLAS) |

Total: ~300k multiply-adds per vector, **all scalar, no SIMD, no BLAS**. At 500 v/s the CPU
is running at ~0.15 GFLOPS — modern CPUs do 50–100 GFLOPS. The gap is entirely architecture,
not hardware.

The entire batch of 5000 vectors goes through this per-vector loop sequentially in
`engine.rs:622`. There is no batching anywhere.

---

## Target: 30k v/s

30k v/s requires ~600M FLOPs/s. With 8-core CPU + BLAS (50+ GFLOPS available), this is well
within reach.

---

## Implementation Order & Expected v/s at Each Stage

| Phase | Change | File(s) | Expected v/s | Effort |
|-------|--------|---------|-------------|--------|
| Start | Baseline | — | ~500 v/s | — |
| 1 | `par_iter` in `write_many` | `engine.rs` | ~2,000–4,000 v/s | 1–2 hrs |
| 2 | Enable OpenBLAS | `Cargo.toml` | ~5,000–8,000 v/s | 2–4 hrs |
| 3a | Binary search in centroid scan | `mse.rs` | ~8,000–12,000 v/s | 1 hr |
| 3b–d | `quantize_batch` (batch DGEMM) | `mse.rs`, `qjl.rs`, `prod.rs`, `engine.rs` | **25,000–45,000 v/s** | 4–8 hrs |
| 4 | WAL deferred fsync | `wal.rs` | +5–10% | 30 min |
| 5 | `par_iter` + `use rayon` for `create_index` | `engine.rs` | fixes hang | 15 min |

**Start with Phase 5 + Phase 4** (isolated, no correctness risk, done in <1 hour).
**Phase 3b–d is the core work** that crosses the 30k threshold.

---

## Phase 1 — Parallelize `write_many` with Rayon *(est. 2,000–4,000 v/s)*

`ProdQuantizer::quantize()` takes `&self` (immutable) — the quantizer holds no mutable state
after construction, so it is already `Sync`. The only change needed is in `engine.rs`.

**File**: `src/storage/engine.rs`

Add `use rayon::prelude::*;` at the top of the file.

Replace the sequential `for item in items` loop at line 622:

```rust
// BEFORE (sequential):
for item in items {
    let (indices, qjl, gamma) = self.quantizer.quantize(&item.vector);
    // ...build wal_entries, metadata_entries, live_updates...
}

// AFTER: parallel quantization, then sequential I/O
use rayon::prelude::*;

// Step 1: quantize all vectors in parallel
let quantized: Vec<_> = items.par_iter()
    .map(|item| self.quantizer.quantize(&item.vector))
    .collect();

// Step 2: sequential I/O (WAL, metadata, live_records) — unchanged logic
for (item, (indices, qjl, gamma)) in items.iter().zip(quantized.into_iter()) {
    // ...same as before, but already quantized...
}
```

---

## Phase 2 — Enable BLAS for nalgebra *(est. 5,000–8,000 v/s combined with Phase 1)*

The two 384×384 matrix multiplies (rotation in `mse.rs:51`, projection in `qjl.rs:35`) use
nalgebra's scalar fallback — triple-nested loops with no SIMD. Enabling BLAS turns these into
a single `DGEMM` call using CPU SIMD and cache-optimized tiling.

**File**: `Cargo.toml`

```toml
# Replace:
nalgebra = { version = "0.32", features = ["serde-serialize"] }
ndarray = "0.15"

# With:
nalgebra = { version = "0.32", features = ["serde-serialize", "blas"] }
ndarray = { version = "0.15", features = ["blas"] }
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }

[target.'cfg(windows)'.dependencies]
blas-src = { version = "0.9", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["static", "system"] }
```

On Windows, install OpenBLAS first: `vcpkg install openblas:x64-windows-static`
Alternatively use Intel MKL via `blas-src = { features = ["intel-mkl"] }`.

The 384×384 matmul goes from ~1.5ms (scalar) to ~0.05ms (BLAS) — a ~30× speedup for that step.

---

## Phase 3a — Binary Search in Centroid Scan *(est. 8,000–12,000 v/s)*

Lloyd-Max centroids are always in **sorted ascending order** (the algorithm guarantees this).
Replace the O(2^b) = O(128) linear scan in `mse.rs:55-68` with O(b) = O(8) binary search.
This is a 16× speedup on the centroid search portion.

**File**: `src/quantizer/mse.rs`

```rust
// BEFORE (lines 55-68): O(128) linear scan per dimension
for i in 0..self.d {
    let val = y[(i, 0)];
    let mut min_dist = f64::MAX;
    let mut min_idx = 0;
    for (idx, &c) in self.centroids.iter().enumerate() {
        let dist = (val - c).abs();
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx;
        }
    }
    indices.push(min_idx);
}

// AFTER: O(log 128) = O(7) binary search per dimension
let num_centroids = self.centroids.len();
for i in 0..self.d {
    let val = y[(i, 0)];
    let idx = match self.centroids.partition_point(|&c| c < val) {
        0 => 0,
        n if n >= num_centroids => num_centroids - 1,
        n => {
            let lo = n - 1;
            if (val - self.centroids[lo]).abs() <= (self.centroids[n] - val).abs() { lo } else { n }
        }
    };
    indices.push(idx);
}
```

---

## Phase 3b–d — Batch Quantization API *(the main 30k v/s change)*

Instead of calling `quantize(single_vec)` N=5000 times, add `quantize_batch(N vecs)` that
performs the rotation and projection as **single DGEMM calls** for all N vectors simultaneously.
2 DGEMMs for 5000 vectors instead of 10,000 individual matrix multiplies.

### 3b. Add `quantize_batch` to `MseQuantizer`

**File**: `src/quantizer/mse.rs`

```rust
/// Quantize a batch of N vectors. xs columns are vectors: shape (d, N).
/// Returns N index vectors, each of length d.
pub fn quantize_batch(&self, xs: &DMatrix<f64>) -> Vec<Vec<usize>> {
    // ONE DGEMM: rotation (d×d) × xs (d×N) = y_batch (d×N)
    let y_batch = &self.rotation * xs;

    let n = xs.ncols();
    let num_centroids = self.centroids.len();
    let mut all_indices = vec![vec![0usize; self.d]; n];

    for col in 0..n {
        for row in 0..self.d {
            let val = y_batch[(row, col)];
            let idx = match self.centroids.partition_point(|&c| c < val) {
                0 => 0,
                k if k >= num_centroids => num_centroids - 1,
                k => {
                    let lo = k - 1;
                    if (val - self.centroids[lo]).abs() <= (self.centroids[k] - val).abs() { lo } else { k }
                }
            };
            all_indices[col][row] = idx;
        }
    }
    all_indices
}

/// Dequantize a batch of N index vectors. Returns (d, N) DMatrix.
pub fn dequantize_batch(&self, all_indices: &[Vec<usize>]) -> DMatrix<f64> {
    let n = all_indices.len();
    let mut y_tilde = DMatrix::zeros(self.d, n);
    for (col, indices) in all_indices.iter().enumerate() {
        for row in 0..self.d {
            y_tilde[(row, col)] = self.centroids[indices[row]];
        }
    }
    // ONE DGEMM: rotation^T (d×d) × y_tilde (d×N) = x_tilde (d×N)
    self.rotation.transpose() * y_tilde
}
```

### 3c. Add `quantize_batch` to `QjlQuantizer`

**File**: `src/quantizer/qjl.rs`

```rust
/// Quantize a batch of N residual vectors. rs columns are residuals: shape (d, N).
/// Returns N QJL bit vectors, each of length d, values ±1.
pub fn quantize_batch(&self, rs: &DMatrix<f64>) -> Vec<Vec<i8>> {
    // ONE DGEMM: projection (d×d) × rs (d×N) = s_r_batch (d×N)
    let s_r_batch = &self.projection * rs;

    let n = rs.ncols();
    let mut all_qjl = vec![vec![0i8; self.d]; n];
    for col in 0..n {
        for row in 0..self.d {
            all_qjl[col][row] = if s_r_batch[(row, col)] >= 0.0 { 1 } else { -1 };
        }
    }
    all_qjl
}
```

### 3d. Add `quantize_batch` to `ProdQuantizer`

**File**: `src/quantizer/prod.rs`

```rust
/// Quantize a batch of N vectors in two DGEMM calls total.
pub fn quantize_batch(&self, xs: &[Array1<f64>]) -> Vec<(Vec<usize>, Vec<i8>, f64)> {
    let n = xs.len();
    let d = self.d;

    // Build (d, N) DMatrix from input slice — one allocation
    let mut x_mat = DMatrix::zeros(d, n);
    for (col, x) in xs.iter().enumerate() {
        for row in 0..d {
            x_mat[(row, col)] = x[row];
        }
    }

    // MSE batch: ONE DGEMM for all N vectors
    let mse_indices = self.mse_quantizer.quantize_batch(&x_mat);

    // Dequantize batch: ONE DGEMM (rotation^T × y_tilde_batch)
    let x_tilde_mat = self.mse_quantizer.dequantize_batch(&mse_indices);

    // Residuals: r_mat = x_mat - x_tilde_mat  (element-wise, no DGEMM needed)
    let r_mat = x_mat - x_tilde_mat;

    // QJL batch: ONE DGEMM for all N residuals
    let qjl_all = self.qjl_quantizer.quantize_batch(&r_mat);

    // Gamma per vector: L2 norm of each column of r_mat (scalar, fast)
    let mut results = Vec::with_capacity(n);
    for (col, (indices, qjl)) in mse_indices.into_iter().zip(qjl_all.into_iter()).enumerate() {
        let mut gamma = 0.0f64;
        for row in 0..d {
            let v = r_mat[(row, col)];
            gamma += v * v;
        }
        results.push((indices, qjl, gamma.sqrt()));
    }
    results
}
```

### 3e. Wire `quantize_batch` into `write_many`

**File**: `src/storage/engine.rs`

```rust
fn write_many(&mut self, items: Vec<BatchWriteItem>, mode: BatchWriteMode) -> Result<...> {
    // ... validation loop unchanged ...

    // Extract vectors for batch quantization
    let vectors: Vec<Array1<f64>> = items.iter().map(|item| item.vector.clone()).collect();

    // ONE batch call — 3 DGEMMs total for all N vectors (was N×2 DGEMMs)
    let quantized = self.quantizer.quantize_batch(&vectors);

    // Build WAL/metadata entries — same logic as before
    let mut wal_entries    = Vec::with_capacity(items.len());
    let mut metadata_entries = Vec::with_capacity(items.len());
    let mut live_updates   = Vec::with_capacity(items.len());

    for (item, (indices, qjl, gamma)) in items.into_iter().zip(quantized.into_iter()) {
        // ...unchanged WAL/metadata construction...
    }

    self.wal.append_batch(&wal_entries)?;
    self.metadata.put_many(&metadata_entries)?;
    // ...
}
```

---

## Phase 4 — WAL Deferred fsync *(+5–10%)*

**File**: `src/storage/wal.rs`, line 56

`sync_data()` currently fsyncs on every `append_batch()` call. Move it to explicit flush only.

```rust
// Change append_batch to accept a sync flag:
pub fn append_batch(&mut self, entries: &[WalEntry], force_sync: bool) -> Result<...> {
    for entry in entries {
        let encoded = bincode::serialize(entry)?;
        let len = encoded.len() as u64;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&encoded)?;
        self.entry_count += 1;
    }
    self.writer.flush()?;
    if force_sync {
        self.writer.get_ref().sync_data()?;  // only when explicitly requested
    }
    Ok(())
}
```

- `write_many` calls: `self.wal.append_batch(&wal_entries, false)`
- `flush()` / `close()` calls: `self.wal.append_batch(&[], true)` or add a dedicated `sync()` method

---

## Phase 5 — Fix `create_index` Hang *(15 min, do first)*

**File**: `src/storage/engine.rs`

```rust
// 1. Add at top of engine.rs (after existing use statements):
use rayon::prelude::*;

// 2. Line 393-401: change .iter() to .par_iter():
let all_vectors: Vec<Array1<f64>> = indexed_ids
    .par_iter()          // was .iter()
    .map(|id| {
        self.live_vectors.get(id).cloned().unwrap_or_else(|| {
            let r = &self.live_records[id];
            self.quantizer.dequantize(&r.quantized_indices, &r.qjl_bits, r.gamma as f64)
        })
    })
    .collect();
```

Impact: `create_index` on 16k vectors: >60s timeout → ~2–5s.

---

## Correctness Notes

1. **Lloyd-Max centroids are sorted**: The `codebook.rs` implementation initializes centroids
   uniformly and iterates toward optimal boundaries. The final centroids are always in ascending
   order, making `partition_point` (binary search) valid.

2. **`dequantize_batch` must match `quantize_batch`**: The residual `r = x - dequant(mse_indices)`
   must use the same rotation matrix. Phase 3d's `dequantize_batch` uses `rotation.transpose() *
   y_tilde_batch` — exactly the inverse of `rotation * x_batch`. Verify round-trip accuracy with
   a unit test before wiring into production.

3. **WAL durability contract**: Phase 4 changes the durability guarantee from "every batch is
   durable" to "batches are durable after explicit flush/close". Document this in the API and
   ensure the Python bindings call `flush()` before considering an `insert_batch` permanent.

4. **Thread safety of quantizer in Phase 1**: `ProdQuantizer::quantize()` takes `&self` and
   reads only `self.mse_quantizer` and `self.qjl_quantizer` (both immutable after construction).
   No interior mutability — safe to call from multiple rayon threads without any locking.

---

## Benchmarking After Each Phase

Use `Qualification/ClaudeQual/tqdb_probe2.py` after each phase rebuild:

```
cd C:\dev\turboquantDB
cargo build --release
pip install --force-reinstall target_py\wheels\turboquantdb-*.whl
python Qualification\ClaudeQual\tqdb_probe2.py
```

Once insert_batch exceeds ~5k v/s, run the full comparison:
```
python Qualification\ClaudeQual\tqdb_comparison_bench.py
```
