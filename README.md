# <img src="https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/website/favicon.svg" width="32" height="32" alt="" style="vertical-align:middle;margin-right:6px"> TurboQuantDB

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jyunming/TurboQuantDB/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tqdb)](https://pypi.org/project/tqdb/)

An embedded vector database with a Python API, built around the **TurboQuant** algorithm ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) — two-stage quantization that achieves near-optimal vector compression with zero training time.

**Goal:** make massive embedding datasets practical on lightweight hardware. A 100k-vector, 1536-dim collection that would occupy 586 MB as raw float32 fits in **108 MB on disk** with TQDB b=4, or just **59 MB** with b=2 — enabling laptop-scale RAG over millions of documents without a dedicated server.

Two deployment modes:
- **Embedded** — `tqdb` Python package (`pip install tqdb`), runs in-process (no daemon)
- **Server** — Axum HTTP service in `server/`, with multi-tenancy, RBAC, quotas, and async jobs

---

## Key Properties

- **Zero training** — No `train()` step. Vectors are quantized and stored immediately on insert.
- **5–10× compression** — b=4 reduces 1536-dim float32 embeddings from 586 MB to 108 MB (5.4×); b=2 reaches 59 MB (9.9×) at 100k vectors.
- **Two quantizer modes** — default (`dense`, best recall) and a faster ingest variant (`srht`) for streaming/high-d workloads. See [docs/QUANTIZER_MODES.md](https://github.com/jyunming/TurboQuantDB/blob/main/docs/QUANTIZER_MODES.md) for a full breakdown.
- **Optional ANN index** — Build an HNSW graph after loading data for fast approximate search.
- **Metadata filtering** — MongoDB-style filter operators on any metadata field.
- **Crash recovery** — Write-ahead log (WAL) ensures durability without explicit flushing.
- **Python native** — `pip install tqdb`; no server or sidecar required.

---

## Installation

```bash
pip install tqdb
```

Building from source (Rust toolchain required): see [`DEVELOPMENT.md`](https://github.com/jyunming/TurboQuantDB/blob/main/DEVELOPMENT.md).

---

## Config Advisor

The **[interactive Config Advisor](https://jyunming.github.io/TurboQuantDB/advisor.html)** selects the best configuration for your embedding dimension and use case (RAG, search-at-scale, edge deployment, etc.), scored against real benchmark data with adjustable priority weights for recall, compression, and speed.

[![Config Advisor](https://img.shields.io/badge/Config%20Advisor-Try%20it-4f46e5)](https://jyunming.github.io/TurboQuantDB/advisor.html)

---

## Recommended Setup

`rerank=True` stores raw INT8 vectors alongside compressed codes for exact second-pass rescoring. `fast_mode=True` (default) uses MSE-only quantization — optimal for d < 1536.

```python
from tqdb import Database

# Best recall, any dimension — brute-force
db = Database.open(path, dimension=DIM, bits=4, rerank=True)   # INT8 rerank storage
results = db.search(query, top_k=10)
# GloVe-200 (d=200):     R@1 ≈ 1.00  |  ~30 MB disk
# arXiv-768 (d=768):     R@1 ≈ 0.98  |  ~116 MB disk
# DBpedia-1536 (d=1536): R@1 ≈ 0.95  |  ~231 MB disk

# Best recall, high-d (d ≥ 1536) — also enable QJL residuals
db = Database.open(path, dimension=1536, bits=4, rerank=True, fast_mode=False)

# Minimum disk — MSE codes only (library default, no extra vector storage)
db = Database.open(path, dimension=DIM, bits=4)

# Low latency at N ≥ 100k — HNSW index
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
db.create_index()
results = db.search(query, top_k=10, _use_ann=True)       # p50 < 10ms

# Tune rerank oversampling at query time (default 10×)
results = db.search(query, top_k=10, rerank_factor=20)    # higher recall, higher latency
```

Full configuration guide: [`docs/CONFIGURATION.md`](https://github.com/jyunming/TurboQuantDB/blob/main/docs/CONFIGURATION.md) | Python API: [`docs/PYTHON_API.md`](https://github.com/jyunming/TurboQuantDB/blob/main/docs/PYTHON_API.md)

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
                   quantizer_type=None)  # None/"dense" = default (Haar QR + Gaussian); "srht" = fast O(d log d) ingest
# NOTE: rerank=True stores raw f16 vectors for exact second-pass rescoring.
#       rerank_factor (default 10× brute / 20× ANN) controls oversampling.

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

## Benchmarks

Three datasets, 100k vectors each, matching [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5. Benchmark config: `quantizer_type=None` (dense), `fast_mode=True, rerank=True` (MSE-only, matching paper Figure 5 bit allocation).

![Benchmark recall curves — TQDB vs paper](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots.png)

Key results at 100k × d=1536 (DBpedia), brute-force, b=4, rerank=True:

| Metric | Value |
|--------|-------|
| Recall@1 | 92.2% |
| Recall@4 | 99.9% |
| Disk | 108 MB (5.4× compression) |
| p50 latency | ~51ms |

Full tables (all 8 configs × 3 datasets), ANN guidance, and reproduction steps: **[docs/BENCHMARKS.md](https://github.com/jyunming/TurboQuantDB/blob/main/docs/BENCHMARKS.md)**

### Rerank unlocks recall at any bit depth

`bits=2, rerank=True` matches `bits=4, rerank=True` recall while using ~10% less disk, and outperforms `bits=4, rerank=False` at lower disk cost. (bit_sweep, n=10k, brute-force, fast_mode=True)

| Dataset | b=2, no rerank | b=4, no rerank | b=2 + rerank | b=4 + rerank |
|---------|---------------|---------------|-------------|-------------|
| GloVe-200 (d=200) | 0.528 (1.8 MB) | 0.822 (2.3 MB) | **0.992** (3.8 MB) | **0.992** (4.2 MB) |
| arXiv-768 (d=768) | 0.426 (7.4 MB) | 0.696 (9.2 MB) | **0.978** (14.7 MB) | **0.978** (16.6 MB) |
| GIST-960 (d=960)  | 0.294 (10.4 MB) | 0.566 (12.7 MB) | **0.974** (19.6 MB) | **0.974** (21.9 MB) |

### Coverage across d=65–3072

R@1 ≥ 0.87 across all 9 benchmark datasets at b=4, rerank=True, brute-force, fast_mode=True, n=10k:

| Dataset | d | R@1 | Disk | p50 |
|---------|---|-----|------|-----|
| lastfm-64 | 65 | 0.874 | 2.0 MB | 1.1 ms |
| deep-96 | 96 | 0.980 | 2.5 MB | 1.2 ms |
| glove-100 | 100 | 0.990 | 2.6 MB | 1.4 ms |
| glove-200 | 200 | 0.992 | 4.2 MB | 1.7 ms |
| nytimes-256 | 256 | 0.992 | 5.2 MB | 2.0 ms |
| arXiv-768 | 768 | 0.978 | 16.6 MB | 7.6 ms |
| GIST-960 | 960 | 0.974 | 21.9 MB | 7.3 ms |
| DBpedia-1536 | 1536 | 0.998 | 41.1 MB | 10.3 ms |
| DBpedia-3072 | 3072 | 1.000 | 117.0 MB | 46.8 ms |

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

## Server Mode

An optional Axum HTTP server in `server/` adds multi-tenancy, RBAC, and async jobs. See [`server/README.md`](https://github.com/jyunming/TurboQuantDB/blob/main/server/README.md) for setup and endpoint reference.

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
