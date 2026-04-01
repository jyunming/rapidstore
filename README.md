# TurboQuantDB

A high-performance, embedded vector database written in Rust with Python bindings. It implements Google DeepMind's **TurboQuant** algorithm (arXiv:2504.19874) for data-oblivious vector quantization — zero training time, 8–16x memory compression, and provably unbiased inner product estimation.

---

## Key Features

- **Zero training time** — No `train()` step. Vectors are quantized and stored immediately on insert.
- **Extreme compression** — Reduces float32 embeddings to 2–4 bits per coordinate (8–16x less RAM).
- **Unbiased scoring** — Quantized Johnson-Lindenstrauss (QJL) transforms guarantee unbiased inner product estimation.
- **Optional ANN index** — Build an HNSW graph after loading data for sub-millisecond search on large collections.
- **Metadata filtering** — Filter search results by metadata fields, with support for MongoDB-style operators.
- **Crash recovery** — Write-ahead log (WAL) ensures inserts survive process crashes without explicit flushing.
- **Python native** — Built with PyO3 and Maturin; runs in-process, no server required.
- **Server mode** — Optional Axum HTTP service with multi-tenancy, RBAC, quotas, and async jobs.

---

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

## Reference

TurboQuant algorithm: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — *TurboQuant: Near-Optimal Data-Oblivious Vector Quantization*
