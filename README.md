# TurboQuantDB

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/turboquantdb)](https://pypi.org/project/turboquantdb/)

An embedded vector database written in Rust with Python bindings, implementing the **TurboQuant** algorithm ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) — zero training time, 2–4 bit compression, and provably unbiased inner product estimation.

**Goal:** make massive embedding datasets practical on lightweight hardware. A 50k-vector, 1536-dim collection that would occupy 293 MB as raw float32 fits in **70 MB on disk and 488 MB of RAM** with TQDB b=4 — enabling laptop-scale RAG over millions of documents without a dedicated server.

Two deployment modes:
- **Embedded** — `turboquantdb` Python package, runs in-process (no daemon)
- **Server** — Axum HTTP service in `server/`, with multi-tenancy, RBAC, quotas, and async jobs

---

## Key Properties

- **Zero training** — No `train()` step. Vectors are quantized and stored immediately on insert.
- **4.2× compression** — Reduces float32 embeddings to 2–4 bits per coordinate; b=4 stores each vector in 1,466 bytes vs 6,144 bytes for float32.
- **Unbiased scoring** — QJL transform guarantees unbiased inner product estimation.
- **Optional ANN index** — Build an HNSW graph after loading data for fast approximate search.
- **Metadata filtering** — MongoDB-style filter operators on any metadata field.
- **Crash recovery** — Write-ahead log (WAL) ensures durability without explicit flushing.
- **Python native** — Built with PyO3 and Maturin; no server or sidecar required.

---

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) stable toolchain
- Python 3.8+
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
pip install turboquantdb
```

---

## Quick Start

```python
import numpy as np
from turboquantdb import Database

db = Database.open("./my_db", dimension=1536, bits=4, metric="ip", rerank=True)

db.insert("doc-1", np.random.randn(1536).astype("f4"), metadata={"topic": "ml"}, document="Machine learning intro")
db.insert("doc-2", np.random.randn(1536).astype("f4"), metadata={"topic": "systems"}, document="Rust memory model")

results = db.search(np.random.randn(1536).astype("f4"), top_k=5)
for r in results:
    print(r["id"], r["score"], r["document"])
```

---

## Python API

### `Database.open(path, dimension, bits=4, seed=42, metric="ip", rerank=True, fast_mode=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Directory path for database files |
| `dimension` | `int` | Vector dimension (must match on reopen) |
| `bits` | `int` | Quantization bits: `4` (4.2× compression) or `8` (2.5× compression, higher recall) |
| `seed` | `int` | RNG seed for quantizer — must match across sessions |
| `metric` | `str` | `"ip"`, `"cosine"`, or `"l2"` |
| `rerank` | `bool` | Store dequantized vectors for final reranking; improves recall |
| `fast_mode` | `bool` | Skip QJL stage — ~2× faster ingest, ~3pp recall loss |

### Insert / Update / Delete

```python
db.insert(id, vector, metadata=None, document=None)
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")
# mode: "insert" | "upsert" | "update"

db.upsert(id, vector, metadata=None, document=None)
db.update(id, vector, metadata=None, document=None)
db.delete(id)
```

### Retrieve

```python
db.get(id)             # → {id, metadata, document} or None
db.get_many(ids)       # → list (None for missing)
db.list_all()          # → list of all ids
db.stats()             # → {vector_count, disk_bytes, has_index, …}
```

### Search

```python
results = db.search(
    query,                      # np.ndarray or list[float]
    top_k=10,
    filter=None,                # metadata filter (see below)
    _use_ann=True,              # use HNSW index if available
    ann_search_list_size=None,  # HNSW ef_search (default: max_degree × 2)
)
# Each result: {"id": str, "score": float, "metadata": dict, "document": str | None}
```

### Index (HNSW)

Build **after** loading your data. Not updated incrementally — rebuild after large inserts.

```python
db.create_index(
    max_degree=32,          # neighbors per node — higher = better recall, larger graph
    ef_construction=200,    # beam size during build — higher = better quality
    n_refinements=5,        # refinement passes — higher = better graph
    search_list_size=128,   # alias for ef_construction
    alpha=1.2,              # pruning factor
)
```

### Metadata Filtering

```python
# Simple equality
db.search(query, top_k=5, filter={"topic": "ml"})

# Comparison operators: $eq $ne $gt $gte $lt $lte
db.search(query, top_k=5, filter={"year": {"$gte": 2023}})

# Logical: $and $or
db.search(query, top_k=5, filter={
    "$and": [{"topic": "ml"}, {"year": {"$gte": 2023}}]
})
```

---

## Recommended Presets

### High Quality — recall matters most

```python
db = Database.open(path, dimension=DIM, bits=8, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=8)
results = db.search(query, top_k=10, ann_search_list_size=200)
# ~97% Recall@10 at 50k×1536  |  ~38s ingest  |  119 MB disk
```

### Balanced — default recommendation

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=128)
# ~89% Recall@10 at 50k×1536  |  ~26s ingest  |  70 MB disk
```

### Fast Build — ingest speed is priority

```python
db = Database.open(path, dimension=DIM, bits=4, fast_mode=True, rerank=False)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=128)
# ~83% Recall@10 at 50k×1536  |  ~22s ingest  |  70 MB disk
```

---

## Benchmarks

Full results: **[BENCHMARKS.md](BENCHMARKS.md)**

**Windows 11 highlights (50k × 1536, top_k=10, DBpedia OpenAI embeddings):**

| Engine | Ingest | Disk | RAM | p50 | Recall@10 |
|--------|--------|------|-----|-----|-----------|
| ChromaDB (HNSW) | 30.6s | 398 MB | 865 MB | 1.73ms | 99.75% |
| LanceDB (IVF_PQ) | 77.6s | 318 MB | 526 MB | 9.17ms | 79.50% |
| **TQDB b=8 HQ** | **59.2s** | **119 MB** | **537 MB** | 12.42ms | **97.25%** |
| **TQDB b=4 Balanced** | **37.8s** | **70 MB** | **488 MB** | 9.98ms | **89.15%** |
| **TQDB b=4 FastBuild** | **28.2s** | **70 MB** | **487 MB** | 5.30ms | **83.35%** |

TQDB b=4 is **5.7× smaller** than ChromaDB and uses **44% less RAM**, within 11pp of recall.

---

## RAG Integration

```python
from turboquantdb.rag import TurboQuantRetriever

retriever = TurboQuantRetriever(db_path="./rag_db", dimension=1536, bits=4)
retriever.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

results = retriever.similarity_search(query_embedding=query_vec, k=5)
for r in results:
    print(r["score"], r["text"])
```

---

## Architecture

TurboQuantDB is an embedded database (like DuckDB) — it runs in-process with no daemon.

```
./my_db/
├── manifest.json        — DB config (dimension, bits, seed, metric)
├── quantizer.bin        — Serialized quantizer state
├── live_codes.bin       — Memory-mapped quantized vectors (hot path)
├── live_vectors.bin     — Raw float32 vectors (only if rerank=True)
├── wal.log              — Write-ahead log
├── metadata.bin         — Per-vector metadata and documents
├── live_ids.bin         — ID → slot index
├── graph.bin            — HNSW adjacency list (if index built)
└── seg-XXXXXXXX.bin     — Immutable flushed segment files
```

**Write path:** `insert()` → quantize (QR rotation → MSE → Gaussian QJL) → WAL → `live_codes.bin` → flush to segment

**Search (brute-force):** query → precompute lookup tables → score all live vectors → top-k

**Search (ANN):** query → HNSW beam search → rerank → top-k

**Quantization:** Two-stage pipeline:
1. **MSE** — QR rotation + Lloyd-Max scalar quantization to `bits` per coordinate
2. **QJL** — Dense Gaussian projection, 1-bit quantized, bit-packed

The combination gives unbiased inner product estimates with near-optimal distortion, requiring no training data.

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
| `src/quantizer/mse.rs` | `MseQuantizer` — QR rotation + Lloyd-Max codebook |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit Gaussian projection, bit-packed |
| `python/turboquantdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |
| `server/` | Optional Axum HTTP service (separate Cargo workspace) |

---

## Server Mode

```bash
cd server && cargo build --release
TQ_SERVER_ADDR=0.0.0.0:8080 TQ_LOCAL_ROOT=./data ./target/release/turboquantdb-server
```

See [`server/README.md`](server/README.md) for endpoints and environment variables. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_SERVER_ADDR` | `127.0.0.1:8080` | Bind address |
| `TQ_LOCAL_ROOT` | `./data` | Storage root |
| `TQ_JOB_WORKERS` | `2` | Async job thread count |

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

Apache License 2.0 — see [LICENSE](LICENSE).
