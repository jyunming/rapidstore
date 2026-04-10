# TurboQuantDB

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

## Recommended Setup

Default config: `fast_mode=False, rerank=True` — QJL residual stored and used during reranking for best recall **at d ≥ 1536**.

> **Note:** At d < 512, QJL projections are too noisy and `fast_mode=False` reduces recall below the MSE-only baseline. Use `fast_mode=True, rerank=False` for d < 512.

```python
from tqdb import Database

# High-d (d ≥ 1536) — default config, QJL reranking enabled
db = Database.open(path, dimension=DIM, bits=4)
results = db.search(query, top_k=10)
# 92.2% Recall@1, 99.9% Recall@4 at 100k×1536  |  108 MB disk

# Low-d (d < 512) — MSE-only for best recall at low dimensions
db = Database.open(path, dimension=DIM, bits=4, fast_mode=True, rerank=False)
results = db.search(query, top_k=10)

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
                   quantizer_type=None)  # None/"dense" = default (Haar QR + Gaussian); "srht" = fast O(d log d) ingest
# NOTE: rerank=True only improves recall when fast_mode=False (the default).
#       With fast_mode=True, rerank=True adds latency but no recall gain.

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

Three datasets, 100k vectors each, matching [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5. Default config: `quantizer_type=None` (dense), `fast_mode=False, rerank=True` (QJL reranking enabled — best recall).

![Benchmark recall curves — TQDB vs paper](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots.png)

Key results at 100k × d=1536 (DBpedia), brute-force, b=4, rerank=True:

| Metric | Value |
|--------|-------|
| Recall@1 | 92.2% |
| Recall@4 | 99.9% |
| Disk | 108 MB (5.4× compression) |
| p50 latency | ~51ms |

Full tables (all 8 configs × 3 datasets), ANN guidance, and reproduction steps: **[docs/BENCHMARKS.md](https://github.com/jyunming/TurboQuantDB/blob/main/docs/BENCHMARKS.md)**

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
