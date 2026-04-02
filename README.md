# RapidStore

A high-performance, embedded vector database written in Rust with Python bindings. It implements Google DeepMind's **TurboQuant** algorithm (arXiv:2504.19874) for data-oblivious vector quantization — zero training time, 8–16x memory compression, and provably unbiased inner product estimation. It supports both:

- Embedded mode (library + Python bindings)
- Service mode (multi-tenant HTTP API with background jobs)

The quantization approach is inspired by TurboQuant, with practical storage/indexing components implemented in this repository.

## Current Capabilities

- **Zero training time** — No `train()` step. Vectors are quantized and stored immediately on insert.
- **Extreme compression** — Reduces float32 embeddings to 2–4 bits per coordinate (8–16x less RAM).
- **Unbiased scoring** — Quantized Johnson-Lindenstrauss (QJL) transforms guarantee unbiased inner product estimation.
- **Optional ANN index** — Build an HNSW graph after loading data for sub-millisecond search on large collections.
- **Metadata filtering** — Filter search results by metadata fields, with support for MongoDB-style operators.
- **Crash recovery** — Write-ahead log (WAL) ensures inserts survive process crashes without explicit flushing.
- **Python native** — Built with PyO3 and Maturin; runs in-process, no server required.
- **Server mode** — Optional Axum HTTP service with multi-tenancy, RBAC, quotas, and async jobs.

- Collection-scoped storage model (`tenant/database/collection` in service mode)
- Batch write APIs: add, upsert, update, delete
- Query APIs with include controls and pagination (`offset`, `limit`)
- Filter support (`filter` / `where_filter`) for query/delete/get paths
- Async maintenance jobs: compact, index build, snapshot
- Persisted auth/RBAC, quotas, and job store in server mode
- Quota admission controls for vectors, disk usage, and concurrent jobs

<<<<<<< HEAD
## Installation

### Prerequisites

TurboQuantDB is a Rust extension that compiles into Python. You need:

- [Rust](https://rustup.rs/) (stable toolchain)
- Python 3.8+
- A C++ compiler:
  - **Windows**: [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) — select **"Desktop development with C++"** workload
  - **macOS**: `xcode-select --install`
  - **Linux**: `sudo apt-get install build-essential`

### Build

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install maturin
maturin develop --release       # Compiles Rust and installs into venv
```

---

## Quick Start

```python
import numpy as np
from turboquantdb import Database

# Open (or create) a database
db = Database.open("./my_db", dimension=128, bits=4, metric="cosine")

# Insert vectors
db.insert("doc-1", np.random.randn(128).tolist(), metadata={"topic": "ml"}, document="Machine learning intro")
db.insert("doc-2", np.random.randn(128).tolist(), metadata={"topic": "systems"}, document="Rust memory model")

# Search
query = np.random.randn(128).tolist()
results = db.search(query, top_k=5)

for r in results:
    print(r["id"], r["score"], r["document"])
```

---

## Python API Reference

### `Database.open(path, dimension, bits=4, seed=42, metric="ip", rerank=True)`

Opens an existing database or creates a new one.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Directory path for database files |
| `dimension` | `int` | Vector dimension (must match on reopen) |
| `bits` | `int` | Quantization bits — `2` (16x compression) or `4` (8x compression, higher recall) |
| `seed` | `int` | Random seed for quantizer initialization (must match on reopen) |
| `metric` | `str` | `"ip"` (inner product), `"cosine"`, or `"l2"` |
| `rerank` | `bool` | Store raw vectors for reranking; improves recall at the cost of extra disk/RAM |

### Insert & Update

```python
# Single insert
db.insert(id, vector, metadata=None, document=None)

# Batch insert — chunked internally at 2000 vectors for memory efficiency
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")
# mode: "insert" | "upsert" | "update"

# Upsert / update
db.upsert(id, vector, metadata=None, document=None)   # insert or replace
db.update(id, vector, metadata=None, document=None)   # error if id not found
```

### Delete & Retrieve

```python
db.delete(id)                     # Returns True if id existed
db.get(id)                        # Returns {id, metadata, document} or None
db.get_many(ids)                  # Returns list; None for missing ids
db.list_all()                     # Returns all ids as a list
```

### Search

```python
results = db.search(
    query,                        # list[float] or np.ndarray
    top_k=10,
    filter=None,                  # metadata filter dict (see Filtering section)
    _use_ann=True,                # use HNSW index if available
    ann_search_list_size=None,    # HNSW ef_search parameter (default: max_degree * 2)
)
# Each result: {"id": str, "score": float, "metadata": dict, "document": str | None}
```

### Index (HNSW)

Building an HNSW index is optional but dramatically improves search latency on large collections.

```python
db.create_index(
    max_degree=16,          # Max neighbors per node (higher = better recall, larger graph)
    search_list_size=64,    # Beam size during construction (higher = better quality)
    alpha=1.2,              # Pruning factor
)
```

Build the index **after** loading your data. It is not updated incrementally — rebuild after large batches of inserts.

### Stats

```python
stats = db.stats()
# Returns: vector_count, segment_count, total_disk_bytes, index_nodes,
#          live_slots, has_index, ram_estimate_bytes, etc.
```

---

## Metadata Filtering

Filters are applied server-side during search. Two syntaxes are supported:

**Simple equality** (shorthand):
```python
results = db.search(query, top_k=5, filter={"topic": "ml", "year": 2024})
```

**MongoDB-style operators**:
```python
# Comparison: $eq, $ne, $gt, $gte, $lt, $lte
filter = {"year": {"$gte": 2023}}

# Logical: $and, $or
filter = {
    "$and": [
        {"topic": {"$eq": "ml"}},
        {"year": {"$gte": 2023}}
    ]
}
```

---

## Distance Metrics

| Metric | `metric=` | Best for |
|--------|-----------|----------|
| Inner product | `"ip"` | Pre-normalized embeddings (fastest) |
| Cosine similarity | `"cosine"` | Text embeddings where magnitude varies |
| Euclidean (L2) | `"l2"` | Image features, coordinate spaces |

The metric is fixed at database creation time.

---

## RAG Integration

```python
import numpy as np
from turboquantdb.rag import TurboQuantRetriever

retriever = TurboQuantRetriever(db_path="./rag_db", dimension=1536, bits=4, metric="cosine")

texts = [
    "TurboQuant is a near-optimal quantization algorithm.",
    "Rust provides memory safety without garbage collection.",
    "Vector databases power modern RAG pipelines.",
]
embeddings = [np.random.randn(1536).tolist() for _ in texts]
metadatas = [{"source": "paper"}, {"source": "docs"}, {"source": "blog"}]

retriever.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

results = retriever.similarity_search(query_embedding=np.random.randn(1536).tolist(), k=2)
for r in results:
    print(r["score"], r["text"])
```

---

## Server Mode

An optional Axum-based HTTP server is available in `server/`. It adds multi-tenancy, API key authentication, quota enforcement, and async job management (compaction, index building, snapshots).

```bash
cd server && cargo build --release
TQ_SERVER_ADDR=0.0.0.0:8080 TQ_LOCAL_ROOT=./data ./target/release/turboquantdb-server
```

See [`server/README.md`](server/README.md) for the full API reference and environment variable documentation.

---

## Architecture

TurboQuantDB is an embedded database (like DuckDB or LanceDB) — it runs in-process with no daemon required. The storage layout on disk:

```
./my_db/
├── manifest.json        — DB config (dimension, bits, seed, metric)
├── quantizer.bin        — Serialized quantizer state
├── live_codes.bin       — Memory-mapped quantized vectors (hot path)
├── live_vectors.bin     — Raw float32 vectors (only if rerank=True)
├── wal.log              — Write-ahead log (crash recovery)
├── metadata.bin         — Vector metadata and documents
├── live_ids.bin         — ID → slot index
├── graph.bin            — HNSW adjacency list (if index built)
├── graph_ids.json       — Slot list for index nodes
└── seg-00000001.bin     — Immutable segment files (flushed from WAL)
```

**Write path:** `insert()` → quantize (SRHT → MSE → QJL) → WAL entry → `live_codes.bin` → periodic flush to segment

**Search path (brute-force):** query → precompute lookup tables → score all live vectors → top-k

**Search path (ANN):** query → HNSW beam search → candidate rerank (if `rerank=True`) → top-k

**Quantization pipeline:** Each vector goes through two stages:
1. **MSE stage** — Structured Random Hadamard Transform (SRHT) rotation, then Lloyd-Max scalar quantization to `b-1` bits per dimension
2. **QJL stage** — A second independent SRHT projection, 1-bit quantized (sign), packed into bytes

The combination gives unbiased inner product estimates with provably near-optimal distortion.

---

## Benchmarks

Full results across Windows / WSL2 / Linux (50k vectors, dim=1536, cosine): see **[BENCHMARKS.md](BENCHMARKS.md)**.

**Linux native highlights (50k × 1536, top_k=10):**

| Engine | Ingest | Disk | RAM | p50 | Recall@10 |
|--------|--------|------|-----|-----|----------|
| FAISS HNSW *(ceiling)* | 15.1s | 306 MB | 698 MB | 0.96ms | 99.75% |
| ChromaDB | 33.5s | 398 MB | 924 MB | 2.41ms | 99.75% |
| **TQDB b=4 FastBuild** | **22.7s** | **70 MB** | **506 MB** | **4.00ms** | **83.1%** |
| **TQDB b=4 Balanced** | 26.0s | 70 MB | 507 MB | 7.46ms | 88.7% |
| **TQDB b=8 HQ** | 38.0s | 119 MB | 554 MB | 8.73ms | 97.9% |
| LanceDB | 89.8s | 318 MB | 1,722 MB | 8.10ms | 79.9% |

TQDB b=4 stores each vector in **1,466 bytes** vs float32's 6,144 bytes — **4.2× compression** with no training.

---

## Performance Roadmap

### Current bottlenecks (profiled at n=50k, dim=1536)

| Phase | Dominant cost | Parallelism |
|-------|--------------|-------------|
| Ingest: SRHT rotation | 2 × d×d matmul (MSE + QJL) | Rayon per-batch |
| Ingest: WAL write | Sequential I/O (fsync avoided) | — |
| Search: brute-force scan | `n × score_ip_encoded_lite` | Rayon par_chunks |
| Search: HNSW beam | Sequential graph traversal | — |

### Near-term (CPU)

- **AVX-512 codebook scan** — widen the inner-product accumulator from 256-bit to 512-bit (2× throughput on supported CPUs)
- **Cached rotation matrices** — precompute MSE/QJL rotation as a contiguous f32 slab to improve BLAS locality
- **HNSW parallel construction** — build candidate lists concurrently (requires slot-level locking)

### Long-term (GPU)

GPU acceleration is technically viable **only for batch ingest** where the two d×d SRHT matrix multiplies dominate:

| Scenario | CPU (16 cores) | GPU (RTX class) | Whole-system gain | Effort |
|----------|---------------|----------------|------------------|--------|
| Ingest matmul (n=50k, d=1536) | ~26–38s | ~0.05s (cuBLAS) | ~3–5× end-to-end | 6–8 weeks |
| Search / HNSW beam | ~4–11ms | worse (kernel launch overhead) | N/A | — |

**Why not now:** current CPU ingest (22–38s at 50k/1536) is already faster than ChromaDB; the search path does not benefit from GPU; and the embedded in-process design means GPU memory must be explicitly managed. A `gpu` feature flag is reserved for future work.

---

## Reference

TurboQuant algorithm: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — *TurboQuant: Near-Optimal Data-Oblivious Vector Quantization*

---

## Repository Layout

- `src/`: core Rust engine, quantization, storage, indexing, Python bindings
- `server/`: Axum HTTP service crate (`turboquantdb-server`)
- `python/`: Python package helpers
- `tests/`: Rust integration and benchmark-style tests
- `docs/`: design docs, roadmap, API notes
- `benchmarks/`: recall/latency scripts and CI gates

## Service Mode

See [server/README.md](server/README.md) for endpoints and env vars.

Common env vars:

- `TQ_SERVER_ADDR` (default `127.0.0.1:8080`)
- `TQ_LOCAL_ROOT` (default `./data`)
- `TQ_JOB_WORKERS` (default `2`)

## Documentation

- [Python Migration Guide](docs/PYTHON_MIGRATION.md)
- [M5 Multi-Tenant Service Design](docs/M5_MULTITENANT_SERVICE_DESIGN.md)
- [M5 API Spec](docs/M5_API_SPEC.md)
- [Compatibility Matrix](docs/COMPATIBILITY_MATRIX.md)
- [Roadmap Backlog](docs/ROADMAP_BACKLOG.md)

## Research Basis

This repository is an independent implementation that uses ideas described in the TurboQuant paper; the paper itself was authored by the original researchers.

Reference:

Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. arXiv:2504.19874.

- arXiv: https://arxiv.org/abs/2504.19874
- Local copy in this repo: [2504.19874v1.pdf](2504.19874v1.pdf)

If your academic work depends on the TurboQuant theory, please cite the original paper:

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```
