# TurboQuantDB

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jyunming/TurboQuantDB/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tqdb)](https://pypi.org/project/tqdb/)

An embedded vector database written in Rust with Python bindings, implementing the **TurboQuant** algorithm ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) — zero training time, 2–4 bit compression, and provably unbiased inner product estimation.

**Goal:** make massive embedding datasets practical on lightweight hardware. A 100k-vector, 1536-dim collection that would occupy 586 MB as raw float32 fits in **108 MB on disk** with TQDB b=4, or just **59 MB** with b=2 — enabling laptop-scale RAG over millions of documents without a dedicated server.

Two deployment modes:
- **Embedded** — `tqdb` Python package (`pip install tqdb`), runs in-process (no daemon)
- **Server** — Axum HTTP service in `server/`, with multi-tenancy, RBAC, quotas, and async jobs

---

## Key Properties

- **Zero training** — No `train()` step. Vectors are quantized and stored immediately on insert.
- **5–10× compression** — b=4 reduces 1536-dim float32 embeddings from 586 MB to 108 MB (5.4×); b=2 reaches 59 MB (9.9×) at 100k vectors.
- **Unbiased scoring** — QJL transform guarantees unbiased inner product estimation.
- **Optional ANN index** — Build an HNSW graph after loading data for fast approximate search.
- **Metadata filtering** — MongoDB-style filter operators on any metadata field.
- **Crash recovery** — Write-ahead log (WAL) ensures durability without explicit flushing.
- **Python native** — Built with PyO3 and Maturin; no server or sidecar required.

---

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) stable toolchain
- Python 3.10+
- C++ compiler: Visual Studio Build Tools (Windows) · `xcode-select --install` (macOS) · `build-essential` (Linux)

### Build from source

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install maturin
maturin develop --release
```

### Install pre-built wheel

```bash
pip install tqdb
```

---

## Recommended Setup

Two presets cover most use cases — no indexing required to get started:

```python
from tqdb import Database

# Recommended — brute-force with dequantization reranking
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
results = db.search(query, top_k=10)
# 95.5% Recall@1, 100% Recall@4 at 100k×1536  |  108 MB disk  |  ~50ms p50

# Minimum disk — 9.9× compression, still excellent recall
db = Database.open(path, dimension=DIM, bits=2, rerank=True)
results = db.search(query, top_k=10)
# 86.8% Recall@1, 99.3% Recall@4 at 100k×1536  |  60 MB disk  |  ~43ms p50

# Optional: build an HNSW index after bulk load for sub-10ms queries
db.create_index()
results = db.search(query, top_k=10, _use_ann=True)
```

Full parameter reference: [`docs/PYTHON_API.md`](https://github.com/jyunming/TurboQuantDB/blob/main/docs/PYTHON_API.md)

---

## Quick Start

```python
import numpy as np
from tqdb import Database

db = Database.open("./my_db", dimension=1536, bits=4, metric="ip", rerank=True)

db.insert("doc-1", np.random.randn(1536).astype("f4"), metadata={"topic": "ml"}, document="Machine learning intro")
db.insert("doc-2", np.random.randn(1536).astype("f4"), metadata={"topic": "systems"}, document="Rust memory model")

results = db.search(np.random.randn(1536).astype("f4"), top_k=5)
for r in results:
    print(r["id"], r["score"], r["document"])
```

---

## Python API

> Full reference: **[`docs/PYTHON_API.md`](https://github.com/jyunming/TurboQuantDB/blob/main/docs/PYTHON_API.md)**

```python
# Open / create
db = Database.open(path, dimension, bits=4, seed=42, metric="ip",
                   rerank=True, fast_mode=False, rerank_precision=None,
                   collection=None, wal_flush_threshold=None,
                   quantizer_type=None)  # None/"srht" (default) or "exact" (paper-exact QR + Gaussian, O(d²))

# Write
db.insert(id, vector, metadata=None, document=None)
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")  # "insert"|"upsert"|"update"
db.upsert(id, vector, metadata=None, document=None)
db.update(id, vector, metadata=None, document=None)        # RuntimeError if not found
db.update_metadata(id, metadata=None, document=None)       # RuntimeError if not found

# Delete & retrieve
db.delete(id)                        # → bool
db.delete_batch(ids)                 # → int (count deleted)
db.get(id)                           # → {id, metadata, document} | None
db.get_many(ids)                     # → list[dict | None]
db.list_all()                        # → list[str]
db.list_ids(where_filter=None, limit=None, offset=0)       # paginated
db.count(filter=None)                # → int
db.stats()                           # → dict
len(db) / "id" in db                 # container protocol

# Search — brute-force by default; pass _use_ann=True to use HNSW index
results = db.search(query, top_k=10, filter=None, _use_ann=False,
                    ann_search_list_size=None, include=None)
# include: list of "id"|"score"|"metadata"|"document" (default all)
# ann_search_list_size: HNSW ef_search override (only used when _use_ann=True)

all_results = db.query(query_embeddings, n_results=10, where_filter=None)
# query_embeddings: np.ndarray (N, D) — returns list[list[dict]]

# Index
db.create_index(max_degree=32, ef_construction=200, n_refinements=5,
                search_list_size=128, alpha=1.2)

# Metadata filter operators
# $eq $ne $gt $gte $lt $lte $in $nin $exists $and $or
db.search(query, top_k=5, filter={"year": {"$gte": 2023}})
db.search(query, top_k=5, filter={"$and": [{"topic": "ml"}, {"year": {"$gte": 2023}}]})
```

---

## Recommended Presets

### Recommended — brute-force + reranking

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
results = db.search(query, top_k=10, _use_ann=False)
# 95.5% Recall@1, 100% Recall@4 at 100k×1536  |  108 MB disk  |  ~50ms p50 (brute-force)
```

### Minimum Disk — compress aggressively

```python
db = Database.open(path, dimension=DIM, bits=2, rerank=True)
results = db.search(query, top_k=10, _use_ann=False)
# 86.8% Recall@1, 99.3% Recall@4 at 100k×1536  |  60 MB disk (9.9× smaller)  |  ~43ms p50
```

### Optional — ANN index for lower latency

```python
# Build once after inserting data; recall scales with ann_search_list_size
db.create_index()
results = db.search(query, top_k=10, _use_ann=True, ann_search_list_size=200)
```

---

## Benchmarks

Three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — n=100k vectors each. Full script: [`benchmarks/paper_recall_bench.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/paper_recall_bench.py).

<!-- PAPER_BENCH_START -->
### Algorithm Validation — Recall vs Paper

![Benchmark recall curves — TQDB vs paper](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots.png)

Brute-force recall across all three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5 — n=100k vectors, paper values read visually from plots (approximate).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5a) | ≈55.0% | ≈70.0% | ≈83.0% | ≈91.0% | ≈96.0% | ≈99.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 37.1% | 50.0% | 62.0% | 73.0% | 82.0% | 88.9% | 93.5% |
| **TQDB b=2 rerank=T** | 52.8% | 68.4% | 81.1% | 90.3% | 95.5% | 98.4% | 99.5% |
| TurboQuant 4-bit (paper Fig. 5a) | ≈86.0% | ≈96.0% | ≈99.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 73.9% | 88.3% | 96.4% | 99.2% | 99.9% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 82.6% | 94.2% | 98.7% | 99.9% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5b) | ≈89.5% | ≈98.0% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 79.7% | 93.3% | 98.3% | 99.7% | 99.9% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 86.8% | 96.2% | 99.3% | 99.9% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5b) | ≈97.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 92.6% | 99.1% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 95.5% | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5c) | ≈90.5% | ≈98.5% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 84.6% | 95.1% | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 89.2% | 98.6% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5c) | ≈97.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 94.8% | 99.1% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 96.0% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

The GloVe gap (~12–18% at k=1) is expected: d=200 is the hardest case (fewest bits per dimension), and we evaluate on the first 100k vectors from a 1.18M corpus while the paper used a random sample. From k=4 onward the gap is ≤2.6% on GloVe and ≤1% on DBpedia. For high-dimensional embeddings (d≥1536), TQDB matches the paper within ~5% at k=1 and within 1% from k=4.

### Performance & Config Trade-offs

![Config trade-off overview — latency, disk, RAM, CPU](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots_perf.png)

All 8 configs — brute-force and ANN (HNSW md=32, ef=128). Disk MB for ANN includes `graph.bin`. RAM = peak RSS during query phase. Index = HNSW build time (ANN only).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 1.1s | — | 16.4 | 208 | 12.71 | 21.72 | 37.1% | 0.502 |
| b=2 rerank=T | Brute | 1.2s | — | 16.4 | 207 | 16.57 | 24.70 | 52.8% | 0.666 |
| b=4 rerank=F | Brute | 2.3s | — | 22.5 | 214 | 15.36 | 37.02 | 73.9% | 0.842 |
| b=4 rerank=T | Brute | 1.6s | — | 22.5 | 213 | 17.96 | 27.60 | 82.6% | 0.900 |
| b=2 rerank=F | ANN | 1.3s | 30.2s | 25.0 | 238 | 6.43 | 9.47 | 21.5% | 0.283 |
| b=2 rerank=T | ANN | 1.2s | 28.3s | 25.0 | 238 | 11.03 | 14.57 | 37.6% | 0.464 |
| b=4 rerank=F | ANN | 1.6s | 19.8s | 31.1 | 243 | 6.34 | 9.24 | 44.5% | 0.495 |
| b=4 rerank=T | ANN | 1.5s | 18.5s | 31.1 | 246 | 11.10 | 15.07 | 61.1% | 0.656 |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 5.5s | — | 59.1 | 755 | 45.97 | 70.70 | 79.7% | 0.882 |
| b=2 rerank=T | Brute | 4.4s | — | 59.1 | 805 | 63.93 | 99.16 | 86.8% | 0.926 |
| b=4 rerank=F | Brute | 7.4s | — | 108.0 | 854 | 66.77 | 110.56 | 92.6% | 0.961 |
| b=4 rerank=T | Brute | 9.2s | — | 108.0 | 858 | 71.44 | 121.88 | 95.5% | 0.977 |
| b=2 rerank=F | ANN | 4.6s | 99.7s | 67.7 | 772 | 10.15 | 14.61 | 75.2% | 0.829 |
| b=2 rerank=T | ANN | 5.2s | 99.6s | 67.7 | 772 | 38.29 | 53.88 | 84.9% | 0.904 |
| b=4 rerank=F | ANN | 7.9s | 98.2s | 116.5 | 823 | 11.47 | 15.83 | 87.9% | 0.907 |
| b=4 rerank=T | ANN | 10.0s | 107.0s | 116.5 | 822 | 39.30 | 70.66 | 93.8% | 0.959 |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 7.8s | — | 108.0 | 1401 | 110.82 | 398.86 | 84.6% | 0.913 |
| b=2 rerank=T | Brute | 11.3s | — | 108.0 | 1418 | 104.77 | 175.07 | 89.2% | 0.943 |
| b=4 rerank=F | Brute | 14.3s | — | 205.6 | 1497 | 119.48 | 191.95 | 94.8% | 0.972 |
| b=4 rerank=T | Brute | 15.7s | — | 205.6 | 1513 | 120.57 | 211.47 | 96.0% | 0.980 |
| b=2 rerank=F | ANN | 10.4s | 188.1s | 116.6 | 1412 | 16.03 | 27.62 | 81.8% | 0.879 |
| b=2 rerank=T | ANN | 7.9s | 209.2s | 116.6 | 1413 | 62.88 | 129.03 | 87.7% | 0.924 |
| b=4 rerank=F | ANN | 17.3s | 192.9s | 214.2 | 1508 | 36.90 | 62.74 | 90.6% | 0.929 |
| b=4 rerank=T | ANN | 31.6s | 209.0s | 214.2 | 1510 | 152.35 | 282.61 | 94.8% | 0.967 |

**Reproduction:** `maturin develop --release && python benchmarks/paper_recall_bench.py --update-readme --track`  (requires `pip install datasets psutil matplotlib`)

### SRHT vs. Exact Mode — Fair Comparison

The default SRHT mode pads d to the next power of two, producing n=next_power_of_two(d) codes vs. n=d for exact. Comparing at the same bit-width (b) gives SRHT more stored bits — inflating recall numbers unfairly. The only methodology that truly decouples rotation quality from code count is to run both modes at dimensions that are already powers of two, where `next_power_of_two(d) = d` and both modes produce exactly n=d codes. At that point the **only** difference is the rotation matrix: Walsh-Hadamard × ±1 diagonal (SRHT) vs. Haar-random QR (exact). Script: [`benchmarks/compare_quantizers_fair.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/compare_quantizers_fair.py) — 100k synthetic Gaussian vectors, 1k queries, direct delta (no interpolation).

| d | Bits | SRHT R@1 | Exact R@1 | Delta R@1 | SRHT MRR | Exact MRR | Delta MRR | Speed ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 9.7% | 10.4% | +0.7 pp | 0.181 | 0.190 | +0.009 | 1.1x |
| 128 | 4 | 40.7% | 39.9% | -0.8 pp | 0.560 | 0.552 | -0.008 | 1.4x |
| 128 | 8 | 49.8% | 49.2% | -0.6 pp | 0.643 | 0.640 | -0.004 | 1.1x |
| 256 | 2 | 11.7% | 10.9% | -0.8 pp | 0.203 | 0.199 | -0.005 | 3.2x |
| 256 | 4 | 47.8% | 46.6% | -1.2 pp | 0.633 | 0.631 | -0.003 | 1.8x |
| 256 | 8 | 59.4% | 59.9% | +0.5 pp | 0.742 | 0.744 | +0.003 | 1.7x |
| 512 | 2 | 11.1% | 11.3% | +0.2 pp | 0.200 | 0.202 | +0.002 | 5.1x |
| 512 | 4 | 52.7% | 53.6% | **+0.9 pp** | 0.687 | 0.691 | +0.004 | 2.8x |
| 512 | 8 | 72.6% | 73.2% | **+0.6 pp** | 0.838 | 0.840 | +0.002 | 2.8x |
| 1024 | 2 | 11.5% | 10.8% | -0.7 pp | 0.208 | 0.203 | -0.005 | 17.9x |
| 1024 | 4 | 56.7% | 60.5% | **+3.8 pp** | 0.719 | 0.741 | **+0.022** | 12.7x |
| 1024 | 8 | 79.6% | 79.6% | ~0.0 pp | 0.885 | 0.884 | ~0.000 | 6.6x |

*Positive delta = exact better. Speed ratio = SRHT ingest vps / Exact ingest vps. Disk difference (~4–8 MB) is the rotation matrix storage (d×d float32 vs d signs); vector code storage is identical.*

**Interpretation:**
- **d <= 256:** Differences are within noise (±0.012), with no consistent winner. At small dimensions both rotations mix coordinates equally well.
- **d = 512:** Exact is consistently slightly better (+0.2–0.9 pp R@1), consistent with QR's stronger theoretical guarantee (Haar-uniform distribution).
- **d = 1024, b=4:** Exact wins by +3.8 pp R@1 — a meaningful gap for precision-sensitive applications. The Haar-random matrix becomes a noticeably better mixing operator as d grows.
- **d = 1024, b=8:** Both modes reach near-ceiling recall (79.6%) and are indistinguishable.
- **Speed:** SRHT ingest is 1.1–17.9x faster depending on d. The gap widens with d because exact applies an O(d²) matrix multiply vs SRHT's O(d log d) WHT. At d=1024, exact is 7–18x slower to ingest.

**Recommendation:** SRHT is the right default for most workloads — it matches or nearly matches exact quality at d<=256, is only slightly behind at d=512, and is dramatically faster at all dimensions. Use `quantizer_type="exact"` when d>=512 and recall quality is more important than ingest throughput.

<!-- PAPER_BENCH_END -->

### When to use brute-force vs. ANN

The benchmark tables above show a clear pattern: the best search mode depends on vector dimensionality.

**Use brute-force (`_use_ann=False`, the default) when d <= 256**

At low dimensionality, TurboQuant's quantization must pack each dimension into very few bits relative to the total information content. On GloVe-200 (d=200), the gap between brute-force and ANN is stark:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | **82.6%** | 60.5% | ~2x faster p50 |
| b=2, rerank=T | **52.8%** | 37.5% | ~1x (no gain) |

ANN at d=200 loses 22 percentage points of recall versus brute-force because the HNSW graph is built on quantized distances, which are less accurate at low dimension. The latency advantage does not compensate for this recall collapse. Use brute-force for d <= 256.

**Use ANN (`_use_ann=True`) when d >= 512**

At high dimensionality, quantization is more accurate and the ANN approximation is much tighter. On DBpedia d=1536:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | 95.5% | **93.5%** | ~5x faster p50 |
| b=4, rerank=F | 92.6% | **87.9%** | ~3x faster p50 |

ANN costs only ~2 points of recall while cutting latency from ~48ms to ~14ms p50. For production RAG at d=1536 or d=3072, ANN is the right default — build the index once after initial load, then queries are sub-linear.

**Summary:**

| Dimension range | Recommended mode | Reason |
|-----------------|-----------------|--------|
| d <= 256 | Brute-force | Quantization noise at low-d collapses ANN recall |
| d = 512–1024 | Either (test both) | Moderate quantization quality; ANN gain is partial |
| d >= 1536 | ANN | High-d quantization is accurate; ANN gives 3–5x latency speedup with <3% recall cost |

---

## Release Comparison

Performance delta vs. previous version across key metrics (100k vectors, best config per mode).

### v0.4.0 vs v0.3.0 (2026-04-08)

**DBpedia d=1536, brute-force b=4 rerank=T**

| Metric | v0.3.0 | v0.4.0 | Delta |
|--------|--------|--------|-------|
| Ingest | — | ~7s | — |
| p50 query | — | 47.7ms | — |
| p99 query | — | 50.9ms | — |
| R@1 | — | 95.5% | — |
| Disk | — | 108.3 MB | — |
| RAM | — | 860 MB | — |

> v0.3.0 benchmark data was not tracked. Tracking started at v0.4.0 via `benchmarks/perf_history.json`. From v0.5.0 onward, each release row will include a delta column against the prior release.

---

## RAG Integration

```python
from tqdb.rag import TurboQuantRetriever

retriever = TurboQuantRetriever(db_path="./rag_db", dimension=1536, bits=4)
retriever.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

results = retriever.similarity_search(query_embedding=query_vec, k=5)
for r in results:
    print(r["score"], r["text"])
```

---

## Architecture

TurboQuantDB is an embedded database — it runs in-process with no daemon.

```
./my_db/
├── manifest.json        — DB config (dimension, bits, seed, metric)
├── quantizer.bin        — Serialized quantizer state
├── live_codes.bin       — Memory-mapped quantized vectors (hot path)
├── live_vectors.bin     — Raw vectors for exact reranking (only if rerank_precision="f16" or "f32")
├── wal.log              — Write-ahead log
├── metadata.bin         — Per-vector metadata and documents
├── live_ids.bin         — ID → slot index
├── graph.bin            — HNSW adjacency list (if index built)
└── seg-XXXXXXXX.bin     — Immutable flushed segment files
```

**Write path:** `insert()` → quantize (SRHT rotation → MSE → SRHT QJL) → WAL → `live_codes.bin` → flush to segment

**Search (brute-force):** query → precompute lookup tables → score all live vectors → top-k

**Search (ANN):** query → HNSW beam search → rerank → top-k

**Quantization:** Two-stage pipeline as specified in the paper:
1. **MSE** — random orthogonal rotation (QR) + Lloyd-Max scalar quantization to `bits` per coordinate
2. **QJL** — dense i.i.d. N(0,1) Gaussian projection, 1-bit quantized, bit-packed

The combination gives unbiased inner product estimates with near-optimal distortion, requiring no training data.

**SRHT approximation (default):** The default implementation substitutes SRHT (Walsh-Hadamard × random ±1 diagonal) for both the QR rotation and the Gaussian projection. SRHT runs in O(d log d) vs the paper's O(d²) and uses O(d) memory vs O(d²). It pads d to the next power of two, producing more stored codes than the exact mode at the same bit-width setting — which inflates raw recall numbers in a direct comparison. When compared at **equal disk usage**, the exact QR rotation achieves equal or slightly better recall per stored byte, consistent with its stronger theoretical guarantees (Haar-uniform distribution). SRHT's practical advantage is ingest throughput: 15–23× faster at d=1536. Use `quantizer_type="exact"` when storage efficiency matters more than ingest speed.

**Note on "zero indexing time":** The paper's claim (Table 2: TurboQuant 0.0013 s vs PQ 239 s vs RabitQ 2268 s at d=1536) measures **codebook/matrix construction time only** — i.e., how long to set up the quantizer before inserting any data. TurboQuant is near-zero because the quantizer is constructed analytically from a random seed with no data dependency; PQ and RabitQ require expensive training passes (k-means, SVD) over the corpus. The per-vector encoding cost (applying the rotation and projection to each inserted vector) still exists and scales as O(d log d) for SRHT or O(d²) for the exact mode.

### What comes from the paper vs. what is added here

The TurboQuant paper contributes the **quantization algorithm** — how to compress vectors and estimate inner products accurately. Its experiments use flat (exhaustive) search: all database vectors are scored against every query using the LUT-based asymmetric scorer. The paper's "indexing time virtually zero" claim refers to the quantizer requiring no training data, not to graph construction.

**From the paper:** two-stage MSE + QJL quantization, Lloyd-Max codebook, asymmetric LUT scoring, unbiased inner product estimation. The paper specifies QR-random orthogonal rotation and dense Gaussian projection — available as `quantizer_type="exact"`. The default implementation uses SRHT (O(d log d)) as a practical approximation; at equal storage budget, exact mode achieves equal or slightly better recall per byte, while SRHT is 15–23× faster to ingest at high dimensions.

**Added by TurboQuantDB (not in the paper):** WAL persistence, memory-mapped storage, metadata/documents, HNSW graph index, reranking, Python bindings, and the HTTP server.

The brute-force search path (`_use_ann=False`, the default) is the paper-conformant mode — it scores all vectors using TurboQuant's LUT scorer, matching the paper's experimental setup exactly. The HNSW index is a practical engineering addition that reduces the candidate set before scoring, enabling sub-linear search at the cost of approximate recall. Pass `_use_ann=True` to engage the HNSW index (requires `create_index()` to have been called first).

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/python/mod.rs` | `Database` class — Python-facing API |
| `src/storage/engine.rs` | `TurboQuantEngine` — insert/search/delete orchestration |
| `src/storage/wal.rs` | Write-ahead log |
| `src/storage/segment.rs` | Immutable append-only segments |
| `src/storage/live_codes.rs` | Memory-mapped hot vector cache |
| `src/storage/graph.rs` | HNSW graph index |
| `src/quantizer/prod.rs` | `ProdQuantizer` — MSE + QJL orchestrator |
| `src/quantizer/mse.rs` | `MseQuantizer` — SRHT rotation + Lloyd-Max codebook (exact: QR via `quantizer_type="exact"`) |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit SRHT projection, bit-packed (exact: dense Gaussian) |
| `python/tqdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |
| `server/` | Optional Axum HTTP service (separate Cargo workspace) |

---

## Server Mode

> **Status: experimental.** The server crate compiles and the core endpoints work, but it has not been hardened for production use. The embedded library (`tqdb` Python package, `from tqdb import Database`) is the primary supported interface.

An optional Axum-based HTTP server is available in `server/` for multi-tenant deployments. It adds API key authentication, quota enforcement, and async job management (compaction, index building, snapshots).

```bash
cd server && cargo build --release
TQ_SERVER_ADDR=0.0.0.0:8080 TQ_LOCAL_ROOT=./data ./target/release/tqdb-server
```

See [`server/README.md`](https://github.com/jyunming/TurboQuantDB/blob/main/server/README.md) for the full endpoint reference. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_SERVER_ADDR` | `127.0.0.1:8080` | Bind address |
| `TQ_LOCAL_ROOT` | `./data` | Storage root |
| `TQ_JOB_WORKERS` | `2` | Async job thread count |

---

## Performance Roadmap

The current implementation uses SIMD-accelerated scoring (AVX2) for the brute-force search inner loop, the MSE centroid scan,
and the QJL bit-unpack inner product. The FWHT transform (legacy SRHT path) also has an AVX2 fast path.

**GPU acceleration** — batch ingest would benefit from cuBLAS GEMM (~3–5× for
large batches on high-end cards). The ANN search path is memory-bound, not
compute-bound, so GPU benefit there is minimal; the bottleneck is random cache
misses during HNSW graph traversal rather than floating-point throughput.

**AVX-512 codebook scan** — on modern Intel CPUs the MSE centroid lookup can be
vectorised 2× wider with AVX-512, potentially halving scoring latency per batch.

**Persistent HNSW** — incremental graph updates (no full rebuild after each ingest
batch) would allow streaming use cases without periodic `create_index()` calls.

---

## Research Basis

This is an independent implementation of ideas from the TurboQuant paper. The algorithm itself was authored by the original researchers.

> Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

---

## License

Apache License 2.0 — see [LICENSE](https://github.com/jyunming/TurboQuantDB/blob/main/LICENSE).
