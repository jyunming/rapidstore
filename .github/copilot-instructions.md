# TurboQuantDB — Copilot Instructions

TurboQuantDB is a high-performance, embedded vector database written in Rust with Python bindings (PyO3/Maturin). It implements the TurboQuant algorithm (arXiv:2504.19874) for data-oblivious vector quantization — zero training time, 2–4 bit compression, unbiased inner product estimation via QJL transforms.

Two deployment modes:
- **Embedded** — `tqdb` Python package, runs in-process (like DuckDB)
- **Server** — Axum HTTP service in `server/`, with multi-tenancy, RBAC, quotas, and async jobs

---

## Build & Test Commands

```bash
# Build Python extension into active venv (primary workflow)
maturin develop --release

# Fast compile/type validation
cargo check -q

# Format all Rust code
cargo fmt --all

# All Rust unit tests
cargo test -q --lib

# Integration test suites
cargo test --test integration_tests
cargo test --test bench_search
cargo test --test bench_batch_crud

# Python benchmarks (requires maturin develop first)
python benchmarks/run_recall_bench.py
python benchmarks/ci_quality_gate.py     # CI gates: min recall 0.60, max latency 100ms

# Build distributable wheel
maturin build --release --locked

# HTTP server (separate workspace)
cd server && cargo build --release
cd server && cargo test -q
```

---

## Architecture Overview

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/python/mod.rs` | `Database` class — full Python API surface |
| `src/storage/engine.rs` | `TurboQuantEngine` — insert/search/delete/flush |
| `src/storage/wal.rs` | Write-ahead log for crash recovery |
| `src/storage/segment.rs` | Immutable append-only segment files (binary) |
| `src/storage/live_codes.rs` | Memory-mapped hot vector cache (mmap) |
| `src/storage/graph.rs` | HNSW graph index for ANN search (mmap) |
| `src/storage/id_pool.rs` | ID ↔ slot hash table (FNV-1a) |
| `src/storage/metadata.rs` | Per-vector metadata and documents |
| `src/storage/backend.rs` | `StorageProvider` trait (local; extensible) |
| `src/storage/compaction.rs` | Segment merging |
| `src/quantizer/prod.rs` | `ProdQuantizer` — MSE + QJL orchestrator |
| `src/quantizer/mse.rs` | `MseQuantizer` — QR rotation + Lloyd-Max codebook |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — dense Gaussian projection, 1-bit sign bit-packed |
| `src/linalg/hadamard.rs` | In-place O(d log d) FWHT and SRHT (legacy path) |
| `src/linalg/matmul.rs` | GEMM/SGEMM via matrixmultiply crate |
| `python/tqdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |
| `server/` | Axum HTTP service — separate Cargo workspace |

### Data Flow

**Write:** `insert_batch()` → quantize (QR→MSE→Gaussian QJL) → WAL entry → `live_codes.bin` → periodic flush to immutable segment

**Search (brute-force):** query → precompute MSE lookup table + QJL scale → score all live vectors → filter → top-k

**Search (ANN):** query → HNSW beam search → optional float32 rerank → filter → top-k

**Index:** `create_index()` builds HNSW graph from all live data, stored as memory-mapped `graph.bin`. Build after loading data; rebuild after large inserts.

### Storage Layout on Disk

```
<db_path>/
├── manifest.json       — Config: dimension, bits, seed, metric
├── quantizer.bin       — Serialized ProdQuantizer
├── live_codes.bin      — Mmap'd quantized vectors (MSE + QJL + gamma + norm + deleted flag)
├── live_vectors.bin    — Raw float32 vectors (only if rerank=True)
├── wal.log             — Write-ahead log
├── metadata.bin        — Per-slot metadata and documents
├── live_ids.bin        — Serialized IdPool
├── graph.bin           — HNSW adjacency (mmap'd)
├── graph_ids.json      — Index slot list
└── seg-XXXXXXXX.bin    — Immutable segment files
```

---

## Python API Reference

```python
from tqdb import Database

db = Database.open(
    path,               # str — directory for database files
    dimension,          # int — vector size; must match on reopen
    bits=4,             # int — 2 (16x compression) or 4 (8x compression, better recall)
    seed=42,            # int — quantizer seed; must match on reopen
    metric="ip",        # "ip" | "cosine" | "l2" — fixed at creation
    rerank=True,        # bool — store raw vectors; improves recall at cost of extra storage
)

# Insert / Update
db.insert(id, vector, metadata=None, document=None)
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")
# mode: "insert" | "upsert" | "update"
db.upsert(id, vector, metadata=None, document=None)   # insert or replace
db.update(id, vector, metadata=None, document=None)   # error if id not found

# Delete & Retrieve
db.delete(id)                   # → bool (True if existed)
db.get(id)                      # → {id, metadata, document} | None
db.get_many(ids)                # → list[dict | None]
db.list_all()                   # → list[str]

# Search
results = db.search(
    query,                      # list[float] or np.ndarray
    top_k=10,
    filter=None,                # metadata filter (see below)
    _use_ann=True,              # use HNSW index if available
    ann_search_list_size=None,  # HNSW ef_search (default: max_degree * 2)
)
# result fields: {id, score, metadata, document}

# Index (optional, builds HNSW graph)
db.create_index(max_degree=16, search_list_size=64, alpha=1.2)

# Stats
db.stats()   # → dict: vector_count, segment_count, total_disk_bytes, has_index, ram_estimate_bytes, ...
```

### Metadata Filter Syntax

```python
# Simple equality
db.search(q, 5, filter={"topic": "ml"})

# Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte
db.search(q, 5, filter={"year": {"$gte": 2023}})

# Logical operators: $and, $or
db.search(q, 5, filter={"$and": [
    {"topic": {"$eq": "ml"}},
    {"year": {"$gte": 2023}}
]})
```

### Distance Metrics

| `metric=` | Formula | Use when |
|-----------|---------|----------|
| `"ip"` | `a · b` | Embeddings are pre-normalized (fastest) |
| `"cosine"` | `(a·b)/(‖a‖‖b‖)` | Magnitude varies (text embeddings) |
| `"l2"` | `-‖a-b‖` | Euclidean space (image features) |

---

## Key Conventions

### Rust Coding Style
- `rustfmt` defaults (4-space indentation)
- Idiomatic ownership/borrowing; explicit error propagation via `Result` and `?`
- `snake_case` functions/variables, `CamelCase` structs/enums
- Hot paths: small focused functions, avoid unnecessary allocations

### Quantization & Scoring
- Quantization is **data-oblivious** — seed-deterministic QR rotation (MSE) and dense Gaussian projection (QJL), no training
- `ProdQuantizer` encodes as: `[MSE centroid indices (d × log₂b bits)] + [QJL bit-pack (⌈d/8⌉ bytes)]`
- Scoring uses precomputed lookup tables — no repeated centroid arithmetic at query time
- Thread safety: `Arc<RwLock<TurboQuantEngine>>` — concurrent reads, serialized writes

### Testing
- Add/adjust tests for every behavior change touching quantization, indexing, WAL/segment persistence, or Windows mmap
- Performance changes require benchmark smoke checks at multiple scales (10k, 25k, 50k, 100k)
- Test names must describe expected behavior explicitly (e.g., `upsert_replaces_existing_vector`)

### Windows Specific
- mmap/rename semantics require explicit file handle release before overwrite — critical for all storage layer changes
- Use `.\` and `\\` paths in PowerShell documentation examples

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

### Commits & PRs
- **Conventional commits:** `type(scope): summary` (e.g., `fix(storage): release mmap before rename`, `perf(quantizer): faster bit-unpack`)
- One behavior change per commit
- PR descriptions must include: what changed and why, before/after benchmark highlights, platform notes, validation commands run

### Python Bindings
- Binding signatures must remain stable unless intentionally versioned
- Implementation belongs in Rust modules; expose only necessary API updates via `src/python/mod.rs`

### Configuration & Secrets
- Use env vars for server runtime config (`TQ_SERVER_ADDR`, `TQ_LOCAL_ROOT`, `TQ_STORAGE_URI`, `TQ_JOB_WORKERS`)
- Never commit secrets, API keys, or generated data (`server/tenants/`, `*.bin` databases, temp dirs)

---

## Server Mode (Optional)

The `server/` workspace provides an Axum HTTP service for production multi-tenant deployments.

**Endpoints:**
- `GET /healthz`
- `GET|POST /v1/tenants/:tenant/databases/:database/collections`
- `DELETE .../collections/:collection`
- `POST .../add` | `.../upsert` | `.../delete` | `.../get` | `.../query`
- `POST .../compact` | `.../index` | `.../snapshot` (async jobs)
- `GET .../jobs` | `GET /v1/jobs/:job_id` | `POST .../cancel` | `.../retry`

**Features:** Persisted API key auth + RBAC, quota enforcement (vector count, disk bytes, concurrent jobs), restart-safe async job lifecycle.

**Env vars:** `TQ_SERVER_ADDR`, `TQ_LOCAL_ROOT`, `TQ_STORAGE_URI`, `TQ_AUTH_STORE_PATH`, `TQ_QUOTA_STORE_PATH`, `TQ_JOB_STORE_PATH`, `TQ_JOB_WORKERS` (default 2)

See [`server/README.md`](server/README.md) for full details.
