# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

TurboquantDB is an embedded vector database written in Rust with Python bindings (via PyO3/Maturin). It implements the TurboQuant algorithm (arXiv:2504.19874) for data-oblivious vector quantization — zero training time, 2–4 bit compression, unbiased inner product estimation. Designed for RAG pipelines up to ~1M vectors.

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

# Run a single integration test
cargo test --test integration_tests

# Other integration test suites
cargo test --test bench_search
cargo test --test bench_batch_crud

# Python benchmarks (requires maturin develop first)
python benchmarks/run_recall_bench.py
python benchmarks/ci_quality_gate.py   # CI quality gates (recall, latency)
```

CI quality gates: min recall 0.60, max latency 100ms, min speedup vs NumPy 0.20x.

## Architecture

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/storage/engine.rs` | `TurboQuantEngine` — main orchestrator; insert/search/delete/flush |
| `src/storage/segment.rs` | Immutable append-only segment files (binary format) |
| `src/storage/wal.rs` | Write-ahead log for crash recovery |
| `src/storage/live_codes.rs` | Hot in-memory buffer before segment flush |
| `src/storage/graph.rs` | HNSW graph index (memory-mapped binary) |
| `src/storage/compaction.rs` | Segment merging and cleanup |
| `src/storage/backend.rs` | `StorageProvider` trait (local; extensible to S3/GCS) |
| `src/quantizer/prod.rs` | `ProdQuantizer` — two-stage MSE + QJL orchestrator |
| `src/quantizer/mse.rs` | `MseQuantizer` — SRHT rotation + Lloyd-Max codebook |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit random projection, bit-packed |
| `src/linalg/hadamard.rs` | In-place O(d log d) Walsh-Hadamard / SRHT |
| `src/python/mod.rs` | PyO3 `Database` class (public API surface) |
| `python/turboquantdb/rag.py` | LangChain-style `TurboQuantRetriever` wrapper |
| `benchmarks/` | Recall and latency benchmarks vs FAISS/NumPy |
| `tests/` | Rust integration tests |
| `server/` | Optional standalone HTTP server (separate Cargo workspace) |

### Data Flow

**Write:** `insert_batch()` → quantize via SRHT→MSE→QJL → WAL entry → LiveCodes buffer → flush to immutable Segment

**Search:** query → prepare lookup tables (MSE LUT + QJL scale) → brute-force scan OR HNSW graph traversal → optional float32 rerank → top-k results

**Index:** `create_index()` builds an HNSW graph from existing segment data; graph stored as memory-mapped `graph.bin`

### Key Design Points

- Quantization is **data-oblivious** (seed-deterministic rotations, no training phase)
- `ProdQuantizer` encodes as: `[MSE centroid indices (d × log₂b bits)] + [QJL bit-pack (⌈d/8⌉ bytes)]`
- Scoring uses precomputed lookup tables to avoid repeated centroid arithmetic
- Thread safety: `Arc<RwLock<TurboQuantEngine>>` — concurrent reads, serialized writes
- Windows mmap/rename semantics require explicit file handle release before overwrite — be careful with storage layer changes on Windows

## Commit Style

`type(scope): summary` — e.g., `fix(storage): release mmap before rename on Windows`, `perf(quantizer): faster bit-unpack`
