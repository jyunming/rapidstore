# TurboQuantDB — Gemini Context

## What This Is

TurboQuantDB is a high-performance, embedded vector database written in Rust with Python bindings (PyO3/Maturin). It implements the **TurboQuant** algorithm (arXiv:2504.19874) for data-oblivious vector quantization:

- **Zero training time** — No codebook fitting, no training phase. Quantizer state is seeded from a random number only.
- **2–4 bit compression** — 8–16x RAM reduction versus float32, with provably near-optimal distortion.
- **Unbiased inner products** — QJL (Quantized Johnson-Lindenstrauss) transform guarantees unbiased scoring.
- **Optional ANN index** — HNSW graph for sub-millisecond search on large collections.
- **Server mode** — Axum HTTP service with multi-tenancy, RBAC, quotas, and async jobs (`server/` workspace).

## Build & Development Commands

```bash
# Build Python extension into active venv (primary workflow)
maturin develop --release

# Fast Rust type check
cargo check -q

# Format all Rust code
cargo fmt --all

# Run Rust unit tests
cargo test -q --lib

# Run integration tests
cargo test --test integration_tests

# Python benchmarks
python benchmarks/run_recall_bench.py
python benchmarks/ci_quality_gate.py

# Build optional HTTP server
cd server && cargo build --release
cd server && cargo test -q
```

## Architecture

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/python/mod.rs` | `Database` class — full public Python API |
| `src/storage/engine.rs` | `TurboQuantEngine` — insert/search/delete/flush orchestration |
| `src/storage/wal.rs` | Write-ahead log — crash recovery |
| `src/storage/segment.rs` | Immutable append-only segment files |
| `src/storage/live_codes.rs` | Memory-mapped hot vector cache (mmap'd) |
| `src/storage/graph.rs` | HNSW graph index — ANN search |
| `src/storage/id_pool.rs` | ID ↔ slot mapping (FNV-1a hash table) |
| `src/storage/metadata.rs` | Per-vector metadata and documents |
| `src/storage/backend.rs` | `StorageProvider` trait (local; extensible to cloud) |
| `src/storage/compaction.rs` | Segment merging |
| `src/quantizer/prod.rs` | `ProdQuantizer` — orchestrates MSE + QJL stages |
| `src/quantizer/mse.rs` | `MseQuantizer` — SRHT + Lloyd-Max codebook |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit random projection, bit-packed |
| `src/linalg/hadamard.rs` | In-place O(d log d) FWHT and SRHT |
| `src/linalg/matmul.rs` | GEMM/SGEMM via matrixmultiply crate |
| `python/turboquantdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |
| `server/` | Separate Cargo workspace — Axum HTTP service |

### Data Flow

**Write path:**
`insert_batch()` → quantize (SRHT → MSE centroids + QJL bits) → WAL entry → `live_codes.bin` → periodic flush to immutable segment

**Search path (brute-force):**
query → precompute MSE lookup table + QJL scale → score all live vectors → filter by metadata → top-k

**Search path (ANN):**
query → HNSW beam search → optional float32 rerank → filter by metadata → top-k

**Index build:**
`create_index()` → read all live vectors → HNSW graph construction → write `graph.bin` (memory-mapped)

### Storage Files

```
<db_path>/
├── manifest.json       — Config: dimension, bits, seed, metric
├── quantizer.bin       — Serialized ProdQuantizer
├── live_codes.bin      — Mmap'd quantized vectors (MSE codes + QJL bits + gamma + norm + deleted)
├── live_vectors.bin    — Raw float32 vectors (only if rerank=True)
├── wal.log             — Write-ahead log
├── metadata.bin        — Metadata and documents per slot
├── live_ids.bin        — Serialized IdPool
├── graph.bin           — HNSW adjacency (memory-mapped)
├── graph_ids.json      — Slot list for indexed nodes
└── seg-XXXXXXXX.bin    — Immutable segment files
```

## Python API Surface

```python
from turboquantdb import Database

db = Database.open(
    path,           # str — directory path
    dimension,      # int — vector size
    bits=4,         # int — 2 (16x compression) or 4 (8x compression, better recall)
    seed=42,        # int — quantizer seed (must match on reopen)
    metric="ip",    # "ip" | "cosine" | "l2"
    rerank=True,    # bool — store raw vectors for reranking
)

db.insert(id, vector, metadata=None, document=None)
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")
db.upsert(id, vector, metadata=None, document=None)
db.update(id, vector, metadata=None, document=None)
db.delete(id)                           # → bool
db.get(id)                              # → dict | None
db.get_many(ids)                        # → list[dict | None]
db.list_all()                           # → list[str]
db.search(query, top_k, filter=None, _use_ann=True, ann_search_list_size=None)
db.create_index(max_degree=16, search_list_size=64, alpha=1.2)
db.stats()                              # → dict with 16+ fields
```

**Search result fields:** `{id, score, metadata, document}`

### Metadata Filtering

```python
# Simple equality
db.search(q, 5, filter={"topic": "ml"})

# Comparison: $eq, $ne, $gt, $gte, $lt, $lte
db.search(q, 5, filter={"year": {"$gte": 2023}})

# Logical: $and, $or
db.search(q, 5, filter={"$and": [{"topic": {"$eq": "ml"}}, {"year": {"$gte": 2023}}]})
```

### Distance Metrics

| `metric=` | Formula | Use when |
|-----------|---------|----------|
| `"ip"` | `a · b` | Embeddings are pre-normalized |
| `"cosine"` | `(a·b) / (‖a‖‖b‖)` | Magnitude varies (text embeddings) |
| `"l2"` | `-‖a-b‖` | Euclidean space (image features) |

Metric is fixed at database creation and cannot be changed without recreating.

## Development Conventions

### Rust Style
- Follow `rustfmt` defaults (4-space indentation)
- Idiomatic ownership/borrowing; explicit error propagation via `Result` and `?`
- `snake_case` for functions/variables, `CamelCase` for structs/enums
- Hot paths (ingest/search/index build): avoid unnecessary allocations, prefer small focused functions

### Testing
- Add/adjust tests for every behavior change touching quantization, indexing, WAL/segment persistence, or Windows mmap
- Benchmark smoke checks at multiple scales (10k, 25k, 50k, 100k) for performance changes
- Test names must describe expected behavior explicitly

### Windows-Specific
- mmap/rename semantics require explicit file handle release before overwrite — critical for all storage layer changes
- Use forward slashes or raw string paths in documentation examples

### Versioning Policy

Package version is defined in `pyproject.toml`. Follow [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). The project is pre-1.0 (`0.x.y`), meaning the public API is not yet stable.

**Bump rules on every merge to main:**

| Commits present in the merge | Bump |
|-------------------------------|------|
| Any `feat:` | `MINOR`, reset PATCH to 0 |
| Only `fix:`, `perf:`, `refactor:` | `PATCH` |
| Any breaking API change (`feat!:` or `BREAKING CHANGE:` in body) | `MAJOR` |

- Never bump the version on a feature branch — only in the merge commit to main.
- Never reuse a version; if a bad release goes out, yank it on PyPI and cut a new patch.
- The git tag (e.g. `v0.1.0`) must always match the version in `pyproject.toml`. Create the tag after the version bump commit lands on main.
- **Single source of truth:** version is defined only in `pyproject.toml`. `Cargo.toml` must be kept manually in sync. The Python package reads it dynamically via `importlib.metadata`.

### Security & Configuration
- Use environment variables for service runtime config (`TQ_SERVER_ADDR`, `TQ_LOCAL_ROOT`, `TQ_STORAGE_URI`, `TQ_JOB_WORKERS`)
- Never commit secrets, API keys, or locally generated data (`server/tenants/`, temp directories, `*.bin` databases)
