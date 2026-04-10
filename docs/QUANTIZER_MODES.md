# Quantizer Modes: SRHT vs. Dense

TurboQuantDB supports two quantizer families, selected via `quantizer_type` when opening a
database. Both use the same MSE (Lloyd-Max) codebook and produce identical on-disk code
layouts — the only difference is the rotation applied before quantization.

```python
db = Database.open(path, dimension=DIM, bits=4)                           # default = dense
db = Database.open(path, dimension=DIM, bits=4, quantizer_type="dense")   # explicit dense
db = Database.open(path, dimension=DIM, bits=4, quantizer_type="srht")    # fast-ingest path
# "exact" is accepted as a backward-compatible alias for "dense"
```

---

## What Each Mode Does

### `"srht"`

Applies a **Structured Random Hadamard Transform**: a random diagonal sign-flip matrix `D`
followed by the Walsh-Hadamard transform `H`. Zero-pads the input to the next power of two
so `n = next_power_of_two(d)`.

- Rotation cost: O(d log d) per vector
- Rotation state: a random sign vector — O(d) storage, O(d) RAM
- Subspace count (b=4): `n / 8 = next_power_of_two(d) / 8`

### `"dense"` (default)

Applies a **Haar-uniform QR rotation**: samples a random Gaussian matrix and takes its QR
factorization to get a Haar-distributed orthogonal matrix. No zero-padding: `n = d`.

- Rotation cost: O(d²) per vector (dense d×d matrix-vector multiply)
- Rotation state: full d×d float32 matrix — O(d²) storage and RAM
- Subspace count (b=4): `d / 8`

---

## Resource Comparison

### CPU

| Operation | SRHT | Dense | Notes |
|-----------|------|-------|-------|
| Ingest per vector | O(d log d) | O(d²) | Dense is ~140× more ops at d=1536 |
| DB open (init) | Negligible | O(d²) one-time QR | Paid once at first `Database.open()` |
| Search scoring per vector | `n/8` subspaces | `d/8` subspaces | Dense is ~25% fewer at d=1536 |

**Example at d=1536:** SRHT ingest ≈ 16k multiply-adds per vector; Dense ≈ 2.4M. In
practice the dense mat-vec is BLAS-optimized (cache-friendly), so the wall-clock ratio is
10–30× rather than 140×, but still significant for large collections.

### Memory (RAM)

| Component | SRHT | Dense |
|-----------|------|-------|
| Rotation matrix | ~0 (seed only) | d² × 4 bytes — **18.9 MB at d=1536, 75.5 MB at d=3072** |
| `live_codes.bin` (mmap) | `n × b/8` B/vec, n padded | `d × b/8` B/vec — **~25% less** |
| MSE codebook | 256 centroids × `n/8` subspaces | 256 × `d/8` — smaller |
| `live_vectors.bin` (rerank) | `d × 4` B/vec | Same |

At very high d (≥3072), the Dense rotation matrix (75.5 MB) can exceed the savings on
codes — net RAM advantage shifts back to SRHT.

### Disk

| File | SRHT | Dense |
|------|------|-------|
| `live_codes.bin` | ~33% larger (padding) | Smaller |
| Rotation matrix (in `quantizer.bin`) | Negligible | 18.9 MB at d=1536 |
| `live_vectors.bin` (rerank=True) | `d × 4` B/vec | Same |

At d=1536, 100k vectors, `rerank=False` (codes only):
- SRHT: ~103 MB (codes only)
- Dense: ~78 MB (codes) + 19 MB (rotation matrix) ≈ **97 MB** — roughly similar total

### Recall

| Setting | SRHT | Dense |
|---------|------|-------|
| `fast_mode=True`, b=4 | Slightly lower | Marginally better |
| `fast_mode=True`, b=2 | More visible gap | Better |
| `fast_mode=False` | Similar | Similar |

The recall gap comes from zero-padding: SRHT allocates `b` bits across `n > d` dimensions,
wasting some budget on the padded zeros. Dense uses exactly `d` dimensions so no bits are
wasted. At b=4 the gap is small (< 1pp typically); at b=2 it is more noticeable.

---

## Benchmark Summary (d=1536, 100k vectors, DBpedia OpenAI3, brute-force)

| Config | Recall@1 | Recall@4 | Ingest | p50 search | Disk |
|--------|----------|----------|--------|------------|------|
| SRHT b=4 rerank=F | ~96.2% | ~100% | ~70k vps | ~39ms | ~103 MB |
| Dense b=4 rerank=F | ~96.8% | ~100% | ~5–8k vps | ~30ms | ~97 MB |
| SRHT b=2 rerank=F | ~79.2% | ~99.7% | ~70k vps | ~39ms | ~59 MB |
| Dense b=2 rerank=F | ~80.5% | ~99.8% | ~5–8k vps | ~30ms | ~48 MB |

*Ingest vps for Dense is heavily hardware-dependent (BLAS threadcount). Search p50 for
Dense is faster because it scores fewer subspaces (192 vs 256 at d=1536).*

---

## Decision Guide

| Scenario | Recommended |
|----------|-------------|
| Default / don't care | **Dense** — default, best recall, paper-faithful |
| Frequent ingestion (streaming, live updates) at d ≥ 512 | **SRHT** — ingest is 10–30× faster |
| Memory-constrained with very high d (≥ 3072) | **SRHT** — Dense rotation matrix costs 75 MB |
| Maximum recall at low bit budget (b=2) | **Dense** — no wasted bits on padding |
| Reproducing paper figures exactly | **Dense** — matches the paper's QR + Gaussian formulation |
| One-time bulk load, heavy read workload | **Dense** (default) — 25% faster search |

---

## ANN + Rerank Interaction

Both modes produce the same HNSW graph structure (indexed from the MSE codes). The rerank
path reads `live_vectors.bin` (float32 originals), which is identical for both modes.
The ANN recall difference between SRHT and Dense therefore comes only from the pre-rerank
candidate selection quality, not from the rerank itself.

For `rerank=True`, the recall gap between modes narrows further — the float32 rerank
compensates for any coding error in candidate retrieval.

---

## Backward Compatibility

`quantizer_type="exact"` is accepted as an alias for `"dense"` and will continue to work
in all future versions. Databases written with `quantizer_type="exact"` reopen correctly
without any migration step — the `quantizer.bin` is format-identical.
