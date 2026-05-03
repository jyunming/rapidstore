# Quantizer Modes: SRHT vs Dense Haar QR

TurboQuantDB supports two quantizer families, selected via `quantizer_type` when opening
a database. Both use the same Lloyd-Max codebook and produce identical on-disk *code*
layouts — the only difference is the rotation applied before quantization.

```python
db = Database.open(path, dimension=DIM, bits=4)                           # auto: dense if d<1024, srht if d>=1024
db = Database.open(path, dimension=DIM, bits=4, quantizer_type="dense")   # force dense
db = Database.open(path, dimension=DIM, bits=4, quantizer_type="srht")    # force srht
# "exact" is accepted as a backward-compatible alias for "dense"
```

## Default at a glance

| Dimension | Default | Why |
|---|---|---|
| `d < 1024` | `"dense"` | SRHT's pow2-padding tax outweighs its rotation-matrix savings; dense is competitive on every axis. |
| `d >= 1024` | `"srht"` | SRHT delivers 2–3× ingest speedup, 1.5–3× lower p50, equal or slightly better recall, and lower RAM. The pow2-padding overhead becomes acceptable relative to total disk. |

The threshold is conservative — at d ≈ 768 SRHT already wins on most axes. If you
care about ingest throughput at d=768 or below, override with `quantizer_type="srht"`
explicitly.

## What each mode does

### `"srht"` — Structured Random Hadamard Transform

Applies a random diagonal sign-flip matrix `D` followed by the Walsh–Hadamard transform
`H`. Zero-pads the input to the next power of two so `n = next_power_of_two(d)`.

- Rotation cost: O(d log d) per vector
- Rotation state on disk: a length-n random sign vector — **negligible**
- Subspace count (b=4): `n / 8 = next_power_of_two(d) / 8`

### `"dense"` — Haar-uniform QR rotation

Samples a random Gaussian d×d matrix and takes its QR factorization to get a
Haar-distributed orthogonal matrix. No zero-padding: `n = d`.

- Rotation cost: O(d²) per vector (dense d×d matrix-vector multiply)
- Rotation state on disk: full d×d matrix stored as **bfloat16** (since v0.9) — `d² × 2` bytes
- Subspace count (b=4): `d / 8`

> The rotation matrix is bf16 on disk but rehydrated to f32 in memory at load time, so
> the rotation hot path is unchanged. Recall is unchanged within sample noise (verified
> on GloVe d=200 b=4 rerank=F at 10k queries: ΔR@1 = +0.0002).

## Resource comparison

### Ingest throughput

Measured on Windows 11, Python 3.11, single-process, n=100k vectors:

| dim | dense (vps) | srht (vps) | speedup |
|---:|---:|---:|---:|
| 200 | 54,577 | 59,891 | 1.1× |
| 1536 | 6,361 | 12,207 | **1.9×** |
| 3072 | 2,140 | 6,348 | **3.0×** |

Below d=512 the speedup is in the noise; from d=1024 it grows quickly because the
dense `d²` rotation becomes the bottleneck.

### Search latency (brute-force, b=4, rerank=False, 1k queries)

| dim | dense p50 | srht p50 | dense p99 | srht p99 |
|---:|---:|---:|---:|---:|
| 200 | 1.15 ms | 1.30 ms | 1.48 ms | 1.81 ms |
| 1536 | 7.44 ms | **6.02 ms** | 8.66 ms | **6.81 ms** |
| 3072 | 19.14 ms | **10.74 ms** | 24.13 ms | **11.86 ms** |

Dense scores fewer subspaces at low d (no padding), so p50 is competitive there. At
d ≥ 1536 the per-vector rotation cost dominates and SRHT pulls ahead.

### Disk

Two competing taxes:

| Source | Dense | SRHT |
|---|---|---|
| Rotation state (`quantizer.bin`) | `d² × 2` bytes (bf16, **was f32 pre-v0.9**) | length-`n` sign vector — `~0` |
| Code padding (`live_codes.bin`) | None — exact `d` slots | Pads `d → next_power_of_two(d)` codes |

The padding tax matters at non-pow2 dims. At d=3072, b=4, n=100k:

|  | code data | rotation matrix | total |
|---|---:|---:|---:|
| dense (bf16 rotation) | 153.6 MB | 18.0 MB | **171.6 MB** |
| dense (f32 rotation, pre-v0.9) | 153.6 MB | 36.0 MB | 189.6 MB |
| srht (pads d to 4096) | 204.8 MB | ~0 | 204.8 MB |

Dense wins disk at non-pow2 d once bf16 rotation is in. At pow2 dims (1024, 2048,
4096) SRHT has no padding tax and wins disk.

### RAM (delta during ingest, brute search)

| dim | dense | srht |
|---:|---:|---:|
| 200 | 15.7 MB | 18.2 MB |
| 1536 | 169.5 MB | **126.2 MB** |
| 3072 | 372.4 MB | **294.9 MB** |

SRHT's RAM advantage at high d comes from the absent in-memory rotation matrix
(36 MB at d=3072 even after bf16 disk encoding).

### Recall (b=4, rerank=False, R@1)

Measured on the public benchmark datasets, n=100k:

| dataset | dim | dense | srht | Δ (srht − dense) |
|---|---:|---:|---:|---:|
| GloVe-200 | 200 | 0.819 | **0.841** | +0.022 |
| DBpedia-1536 | 1536 | 0.958 | **0.962** | +0.004 |
| DBpedia-3072 | 3072 | 0.963 | **0.980** | +0.017 |

SRHT meets or beats dense on R@1 at every measured dim. Earlier guidance suggesting
SRHT loses recall was based on synthetic tests at low d; on the public datasets SRHT
is at least as accurate.

At b=2 the picture is similar: SRHT b=2 rerank=F R@1 ≈ 0.55 / 0.86 / 0.90 vs dense
0.51 / 0.84 / 0.90 (d=200/1536/3072).

## Decision guide

| Scenario | Recommended |
|---|---|
| Default — let the auto-picker decide | `quantizer_type=None` (default) |
| You're at d ≥ 1024 and care about ingest, latency, or RAM | `"srht"` (this is now the default) |
| You're at d ≤ 768 and care about minimal disk overhead | `"dense"` (this is now the default) |
| Reproducing paper figures exactly | `"dense"` (matches the QR + Gaussian formulation in the TurboQuant paper) |
| Heavy frequent-ingestion workload at any d | `"srht"` |
| Maximum recall at b=2 with `rerank=True` | Either — at b=2 with rerank the gap is well within 0.5 pp |
| Recovering from format break in v0.9 | Old DBs with f32-stored rotation must be rebuilt; bf16 is bincode-incompatible with v0.8 |

## ANN + rerank interaction

Both modes produce the same HNSW graph structure (indexed from the MSE codes) and the
same rerank path (`live_vectors.bin` is identical). Their differences are confined to
the pre-rerank candidate selection. With `rerank=True` and `rerank_precision="int8"`,
the recall gap between modes is sub-0.5 pp at every measured cell.

## Backward compatibility

- `quantizer_type="exact"` remains an alias for `"dense"`.
- Setting `quantizer_type="dense"` or `"srht"` explicitly always wins over the
  auto-default — existing code paths are unaffected if you pass an explicit string.
- **Format break in v0.9**: bf16 rotation storage is bincode-incompatible with v0.8.x
  databases. Old databases must be rebuilt or migrated. SRHT-mode databases reopen
  cleanly — only `dense` mode is affected.
