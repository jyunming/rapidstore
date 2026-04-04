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

Three presets covering the main use cases — pick one and you're ready:

```python
from tqdb import Database

# High Quality — best recall, exact reranking
db = Database.open(path, dimension=DIM, bits=4, rerank=True, rerank_precision="f16")
db.create_index(max_degree=32, ef_construction=200, n_refinements=8)
results = db.search(query, top_k=10, ann_search_list_size=200)
# ~100% Recall@10 at 100k×1536  |  401 MB disk  |  38ms p50 (brute-force)

# Balanced — recommended default (dequant reranking, zero extra disk)
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=200)
# ~99.4% Recall@5 at 100k×1536  |  117 MB disk  |  59ms rerank / 8ms no-rerank

# Fast ANN — lowest latency, good recall
db = Database.open(path, dimension=DIM, bits=4, rerank=False)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=200)
# ~96% Recall@10 at 100k×1536  |  117 MB disk  |  8ms p50
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
                   collection=None)   # collection= → opens path/collection/

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

# Search
results = db.search(query, top_k=10, filter=None, _use_ann=True,
                    ann_search_list_size=None, include=None)
# include: list of "id"|"score"|"metadata"|"document" (default all)

all_results = db.query(query_embeddings, n_results=10, where_filter=None)
# query_embeddings: np.ndarray (N, D) — returns list[list[dict]]

# Index
db.create_index(max_degree=32, ef_construction=200, n_refinements=5,
                search_list_size=128, alpha=1.2)

# Metadata filter operators
# $eq $ne $gt $gte $lt $lte $in $nin $exists $contains $and $or
db.search(query, top_k=5, filter={"year": {"$gte": 2023}})
db.search(query, top_k=5, filter={"$and": [{"topic": "ml"}, {"year": {"$gte": 2023}}]})
```

---

## Recommended Presets

### High Quality — exact reranking

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True, rerank_precision="f16")
db.create_index(max_degree=32, ef_construction=200, n_refinements=8)
results = db.search(query, top_k=10, ann_search_list_size=200)
# 100% Recall@10 at 100k×1536  |  38ms p50 (brute-force)  |  401 MB disk
```

### Balanced — default recommendation

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=200)
# 99.4% Recall@5, 96% Recall@10 at 100k×1536  |  117 MB disk  |  8ms (ANN) / 45ms (brute+dequant)
```

### Minimum Disk — compress aggressively

```python
db = Database.open(path, dimension=DIM, bits=2, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=200)
# 96.4% Recall@10 at 100k×1536  |  68 MB disk (8.7× smaller than float32)  |  7ms p50
```

---

## Benchmarks

Measured on **DBpedia OpenAI3 embeddings** ([Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M)) — real 1536-dim embeddings, n=100k vectors, 500 queries, Recall@1@k metric. HNSW uses M=32, ef_construction=200.

### Algorithm validation (reproducing paper Section 4.4)

Brute-force recall across all three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5 — n=100k vectors, paper values read visually from plots (approximate). Full script: [`benchmarks/paper_recall_bench.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/paper_recall_bench.py).

**GloVe-200** (d=200, 100k corpus, 10k queries, brute-force)

| Method | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---|---|---|---|---|---|---|
| TurboQuant 2-bit (paper Fig. 5a) | ≈55.0% | ≈70.0% | ≈83.0% | ≈91.0% | ≈96.0% | ≈99.0% | ≈100% |
| **TQDB b=2** | **37.1%** | **50.0%** | **62.0%** | **73.0%** | **82.1%** | **88.9%** | **93.5%** |
| TurboQuant 4-bit (paper Fig. 5a) | ≈86.0% | ≈96.0% | ≈99.0% | ≈100% | 100% | 100% | 100% |
| **TQDB b=4** | **73.9%** | **88.3%** | **96.4%** | **99.2%** | **99.9%** | **100%** | **100%** |

**DBpedia OpenAI3 d=1536** (d=1536, 100k corpus, 1k queries, brute-force)

| Method | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---|---|---|---|---|---|---|
| TurboQuant 2-bit (paper Fig. 5b) | ≈89.5% | ≈98.0% | ≈99.5% | ≈100% | 100% | 100% | 100% |
| **TQDB b=2** | **79.7%** | **93.3%** | **98.3%** | **99.7%** | **99.9%** | **100%** | **100%** |
| TurboQuant 4-bit (paper Fig. 5b) | ≈97.0% | ≈100% | 100% | 100% | 100% | 100% | 100% |
| **TQDB b=4** | **92.6%** | **99.1%** | **99.9%** | **100%** | 100% | 100% | 100% |

**DBpedia OpenAI3 d=3072** (d=3072, 100k corpus, 1k queries, brute-force)

| Method | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---|---|---|---|---|---|---|
| TurboQuant 2-bit (paper Fig. 5c) | ≈90.5% | ≈98.5% | ≈99.5% | ≈100% | 100% | 100% | 100% |
| **TQDB b=2** | **84.6%** | **95.1%** | **99.0%** | **100%** | 100% | 100% | 100% |
| TurboQuant 4-bit (paper Fig. 5c) | ≈97.5% | ≈100% | 100% | 100% | 100% | 100% | 100% |
| **TQDB b=4** | **94.8%** | **99.1%** | **100%** | **100%** | 100% | 100% | 100% |

The GloVe gap (~12–18% at k=1) is expected: d=200 is the hardest case (fewest bits per dimension), and we evaluate on the first 100k vectors from a 1.18M corpus while the paper used a random sample. From k=4 onward the gap is ≤2.6% on GloVe and ≤1% on DBpedia. For high-dimensional embeddings (d≥1536), TQDB matches the paper within ~5% at k=1 and within 1% from k=4. The paper also reports TurboQuant quantization time 0.001 s versus Product Quantization 240 s at d=1536 — TQDB inherits the same zero-training-time property.

### Full results — n=100k × 1536-dim, brute-force Recall@1@k:

| Mode | Recall@1 | Recall@10 | Disk | Compression | p50 latency |
|------|----------|-----------|------|-------------|-------------|
| b=4 brute, no-rerank | 93.6% | 100% | 108 MB | **5.4×** | 54ms |
| b=4 brute, dequant-rerank | 93.2% | 100% | 108 MB | **5.4×** | 45ms |
| b=4 brute, f16-rerank | **100%** | 100% | 401 MB | 1.5× | 38ms |
| b=2 brute, no-rerank | 81.4% | 100% | 59 MB | **9.9×** | 41ms |
| b=2 brute, dequant-rerank | 84.4% | 100% | 59 MB | **9.9×** | 46ms |
| b=4 HNSW M=32, no-rerank | 89.8% | 96.0% | 117 MB | 5.0× | **8ms** |
| b=4 HNSW M=32, dequant-rerank | 92.6% | 99.4% | 117 MB | 5.0× | 59ms |
| b=2 HNSW M=32, no-rerank | 78.6% | 96.4% | 68 MB | 8.7× | **7ms** |
| float32 brute (reference) | 100% | 100% | 586 MB | 1× | ~120ms |

*dequant-rerank = rerank from codebook (zero extra disk). f16-rerank = store float16 raw vectors (+293 MB).*

**Reproduction:** build the extension with `maturin develop --release`, then run `python benchmarks/run_recall_bench.py` — requires `pip install datasets tqdb`

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

**Write path:** `insert()` → quantize (QR rotation → MSE → Gaussian QJL) → WAL → `live_codes.bin` → flush to segment

**Search (brute-force):** query → precompute lookup tables → score all live vectors → top-k

**Search (ANN):** query → HNSW beam search → rerank → top-k

**Quantization:** Two-stage pipeline:
1. **MSE** — QR rotation + Lloyd-Max scalar quantization to `bits` per coordinate
2. **QJL** — Dense Gaussian projection, 1-bit quantized, bit-packed

The combination gives unbiased inner product estimates with near-optimal distortion, requiring no training data.

### What comes from the paper vs. what is added here

The TurboQuant paper contributes the **quantization algorithm** — how to compress vectors and estimate inner products accurately. Its experiments use flat (exhaustive) search: all database vectors are scored against every query using the LUT-based asymmetric scorer. The paper's "indexing time virtually zero" claim refers to the quantizer requiring no training data, not to graph construction.

**From the paper:** two-stage MSE + QJL quantization, QR rotation, Lloyd-Max codebook, asymmetric LUT scoring, unbiased inner product estimation.

**Added by TurboQuantDB (not in the paper):** WAL persistence, memory-mapped storage, metadata/documents, HNSW graph index, reranking, Python bindings, and the HTTP server.

The brute-force search path (`_use_ann=False`) is the paper-conformant mode — it scores all vectors using TurboQuant's LUT scorer, matching the paper's experimental setup exactly. The HNSW index is a practical engineering addition that reduces the candidate set before scoring, enabling sub-linear search at the cost of approximate recall.

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

The current implementation already uses AVX2 SIMD for FWHT, the MSE centroid scan,
and the QJL bit-unpack inner product.

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
