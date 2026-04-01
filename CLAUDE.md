# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

TurboQuantDB is an embedded vector database written in Rust with Python bindings (PyO3/Maturin). It implements the TurboQuant algorithm (arXiv:2504.19874): two-stage quantization (MSE + QJL via SRHT) achieving near-optimal vector compression with zero training time.

Two deployment modes:
- **Embedded** — `turboquantdb` Python package, runs in-process
- **Server** — Axum HTTP service in `server/` workspace, with multi-tenancy, RBAC, quotas, async jobs

## Build & Development Commands

```bash
# Build Python extension and install into active venv (primary workflow)
maturin develop --release

# Fast Rust type-check (no codegen)
cargo check -q

# Format code
cargo fmt

# Build distributable wheel
maturin build --release --locked

# Build optional HTTP server
cd server && cargo build --release
```

## Testing

```bash
# All Rust unit tests
cargo test -q --lib

# Individual integration test suites
cargo test --test integration_tests
cargo test --test bench_search
cargo test --test bench_batch_crud

# Python benchmarks (requires maturin develop first)
python benchmarks/run_recall_bench.py
python benchmarks/ci_quality_gate.py   # CI gates: min recall 0.60, max latency 100ms
```

## Architecture

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/python/mod.rs` | `Database` class — full Python-facing API |
| `src/storage/engine.rs` | `TurboQuantEngine` — insert/search/delete/flush orchestration |
| `src/storage/wal.rs` | Write-ahead log for crash recovery |
| `src/storage/segment.rs` | Immutable append-only segment files |
| `src/storage/live_codes.rs` | Memory-mapped hot vector cache |
| `src/storage/graph.rs` | HNSW graph index for ANN search |
| `src/storage/id_pool.rs` | ID ↔ slot hash table (FNV-1a) |
| `src/storage/metadata.rs` | Per-vector metadata and documents |
| `src/storage/backend.rs` | `StorageProvider` trait (local; extensible to cloud) |
| `src/storage/compaction.rs` | Segment merging |
| `src/quantizer/prod.rs` | `ProdQuantizer` — two-stage MSE + QJL orchestrator |
| `src/quantizer/mse.rs` | `MseQuantizer` — SRHT rotation + Lloyd-Max codebook |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit random projection, bit-packed |
| `src/linalg/hadamard.rs` | In-place O(d log d) Walsh-Hadamard / SRHT |
| `python/turboquantdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |
| `server/` | Separate Cargo workspace — optional HTTP service |

### Data Flow

**Write:** `insert_batch()` → quantize via SRHT→MSE→QJL → WAL entry → `live_codes.bin` → periodic flush to immutable segment

**Search (brute-force):** query → precompute MSE lookup table + QJL scale → score all live vectors → top-k

**Search (ANN):** query → HNSW beam search → optional float32 rerank → top-k

**Index:** `create_index()` builds HNSW from existing segment data, stored as memory-mapped `graph.bin`

### Python API

The `Database` class (thread-safe via `Arc<RwLock<TurboQuantEngine>>`) exposes:

```python
Database.open(path, dimension, bits=4, seed=42, metric="ip", rerank=True)

db.insert(id, vector, metadata=None, document=None)
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")
db.upsert(id, vector, ...) / db.update(id, vector, ...)
db.delete(id)
db.get(id) / db.get_many(ids) / db.list_all()
db.search(query, top_k, filter=None, _use_ann=True, ann_search_list_size=None)
db.create_index(max_degree=16, search_list_size=64, alpha=1.2)
db.stats()
```

**Metadata filter syntax:**
```python
{"field": "value"}                                  # simple equality
{"field": {"$gte": 2023}}                           # comparison: $eq $ne $gt $gte $lt $lte
{"$and": [{"topic": "ml"}, {"year": {"$gte": 2023}}]}  # logical: $and $or
```

### Key Design Points

- Quantization is **data-oblivious** (seed-deterministic rotations, no training phase)
- `ProdQuantizer` encodes as: `[MSE centroid indices (d × log₂b bits)] + [QJL bit-pack (⌈d/8⌉ bytes)]`
- Scoring uses precomputed lookup tables — no repeated centroid arithmetic at query time
- Thread safety: `Arc<RwLock<TurboQuantEngine>>` — concurrent reads, serialized writes
- Windows mmap/rename semantics require explicit file handle release before overwrite — be careful with any storage layer changes

## Versioning Policy

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

## Commit Style

`type(scope): summary` — e.g., `fix(storage): release mmap before rename on Windows`, `perf(quantizer): faster bit-unpack`
