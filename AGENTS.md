# Repository Guidelines

## What This Is

TurboQuantDB is an embedded vector database written in Rust with Python bindings (PyO3/Maturin). It implements the TurboQuant algorithm (arXiv:2504.19874): two-stage quantization (MSE + QJL) that achieves near-optimal vector compression with zero training time.

Two deployment modes:
- **Embedded** (`turboquantdb` Python package) — runs in-process, no server needed
- **Server** (`server/` workspace) — Axum HTTP service with multi-tenancy, RBAC, quotas, async jobs

## Project Structure

```
src/
├── lib.rs                    # PyO3 module entry point
├── python/mod.rs             # Database class (public Python API)
├── quantizer/
│   ├── prod.rs               # ProdQuantizer — orchestrates MSE + QJL
│   ├── mse.rs                # MseQuantizer — SRHT + Lloyd-Max codebook
│   ├── qjl.rs                # QjlQuantizer — 1-bit random projection
│   └── codebook.rs           # lloyd_max() — optimal scalar quantizer
├── linalg/
│   ├── hadamard.rs           # fwht(), srht() — O(d log d) transforms
│   ├── matmul.rs             # gemm(), sgemm() — matrix multiply
│   └── rotation.rs           # Random rotation/projection matrices
└── storage/
    ├── engine.rs             # TurboQuantEngine — main orchestrator
    ├── wal.rs                # Write-ahead log
    ├── segment.rs            # Immutable append-only segment files
    ├── live_codes.rs         # Memory-mapped hot vector cache
    ├── graph.rs              # HNSW graph index
    ├── id_pool.rs            # ID ↔ slot hash table (FNV-1a)
    ├── metadata.rs           # Per-vector metadata and documents
    ├── backend.rs            # StorageProvider trait (local / extensible)
    └── compaction.rs         # Segment merging

python/turboquantdb/
├── __init__.py               # Re-exports Database and TurboQuantDB
└── rag.py                    # TurboQuantRetriever (LangChain-style wrapper)

server/                       # Separate Cargo workspace — HTTP service
benchmarks/                   # Recall and latency benchmark scripts
tests/                        # Rust integration tests
```

## Build, Test, and Development Commands

```bash
# Compile and install Python extension into active venv (primary workflow)
maturin develop --release

# Fast Rust type/compile check (no output artifact)
cargo check -q

# Format all Rust code
cargo fmt

# Build distributable wheel
maturin build --release --locked

# Build HTTP server
cd server && cargo build --release
```

```bash
# All Rust unit tests
cargo test -q --lib

# Integration tests
cargo test --test integration_tests
cargo test --test bench_search
cargo test --test bench_batch_crud

# Python benchmarks (requires maturin develop first)
python benchmarks/run_recall_bench.py
python benchmarks/ci_quality_gate.py    # CI quality gates
```

CI quality gates: min recall 0.60, max latency 100ms, min speedup vs NumPy 0.20x.

## Python API Surface

The public API is the `Database` class (exposed via `src/python/mod.rs`):

| Method | Purpose |
|--------|---------|
| `Database.open(path, dimension, bits, seed, metric, rerank)` | Open or create a database |
| `insert(id, vector, metadata, document)` | Insert a single vector |
| `insert_batch(ids, vectors, metadatas, documents, mode)` | Batch insert (chunked at 2000) |
| `upsert(id, vector, metadata, document)` | Insert or replace |
| `update(id, vector, metadata, document)` | Update existing (error if missing) |
| `delete(id)` | Mark as deleted |
| `get(id)` / `get_many(ids)` | Retrieve metadata and document |
| `list_all()` | Return all IDs |
| `search(query, top_k, filter, _use_ann, ann_search_list_size)` | Similarity search |
| `create_index(max_degree, search_list_size, alpha)` | Build HNSW graph |
| `stats()` | Return DB statistics |

**Metadata filter syntax** (passed to `search(filter=...)`):
```python
# Simple equality
{"topic": "ml", "year": 2024}

# Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte
{"year": {"$gte": 2023}}

# Logical operators: $and, $or
{"$and": [{"topic": {"$eq": "ml"}}, {"year": {"$gte": 2023}}]}
```

## Coding Style & Naming Conventions

- Rust: `rustfmt` defaults (4-space indentation), idiomatic ownership, explicit error propagation via `Result` and `?`
- Naming: `snake_case` functions/variables, `CamelCase` structs/enums
- Hot paths (ingest/search/index build): small focused functions, no unnecessary allocations
- Python binding signatures must remain stable unless intentionally versioned; expose only necessary updates

## Testing Guidelines

- Framework: Rust `#[test]` attribute (`cargo test`)
- Add or adjust tests when modifying quantization, indexing, WAL/segment persistence, or Windows mmap behavior
- Performance changes require benchmark smoke checks at multiple scales (10k, 25k, 50k, 100k vectors)
- Test names must describe expected behavior explicitly (e.g., `delete_persists_across_reload`)

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
- The git tag (`v0.3.0`) must always match `pyproject.toml`. Create the tag after the version bump commit lands on main.
- Current version: `0.3.0`

## Commit & Pull Request Guidelines

- Commit style: `type(scope): concise summary` (e.g., `fix(storage): release mmap before overwrite`, `perf(quantizer): faster bit-unpack`)
- Keep commits focused — one behavior change per commit when possible
- PRs must include:
  - What changed and why
  - Before/after benchmark highlights (ingest throughput, ready time, disk, RAM, latency)
  - Platform notes if relevant (especially Windows mmap/rename semantics)
  - Validation commands run
