# Changelog

All notable changes to TurboQuantDB are documented in this file.

Format: `[version] — type(scope): summary`. Commits use [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

---

## [0.2.0] — 2026-04-15

### Changed

- **Package renamed from `turboquantdb` to `tqdb`** — `import tqdb` replaces `import turboquantdb`; the `Database` class is the same
- `src/lib.rs` doc comment updated to reference `tqdb` Python package

---

## [0.1.1] — 2026-04-10

### Fixed

- Release CI: replaced `--find-interpreter` with `-i python` in Windows and macOS matrix jobs to prevent duplicate wheels and "ZIP archive: Trailing data" PyPI upload failures

---

## [0.1.0] — 2026-04-03

### Added

- `rerank_precision` parameter on `Database.open()` — opt-in raw-vector reranking (`"f16"` / `"f32"`, default `None` = dequantization)
- `fast_mode` parameter on `Database.open()` — skip QJL stage for ~30% faster ingest at ~5pp recall cost
- `collection` parameter on `Database.open()` — opens `path/collection/` sub-directory for multi-namespace support
- `delete_batch(ids)` — delete multiple vectors in one call, returns count deleted
- `count(filter=None)` — count active vectors, optionally filtered
- `list_ids(where_filter, limit, offset)` — paginated, filtered ID listing
- `update_metadata(id, metadata, document)` — metadata/document-only update without re-uploading vector
- `query(query_embeddings, n_results, where_filter)` — batch multi-query accepting a 2-D numpy array
- `include=` parameter on `search()` — control which fields are returned (`"id"`, `"score"`, `"metadata"`, `"document"`)
- New metadata filter operators: `$in`, `$nin`, `$exists`, `$contains`
- Python container protocol: `len(db)` and `"id" in db`
- f16 HNSW build scorer — construction scorer reads f16 raw vectors when `rerank_precision="f16"`
- `half` crate dependency for f16 encoding/decoding

### Fixed

- HNSW build scorer hardcoded f32 reads — now branches on manifest `rerank_precision`
- `bench_batch_crud` integration test: updated to `delete_batch` API
- `cloud_tests` integration test: updated to `create_index_with_params` API

### Performance

- **perf(hnsw)**: skip QJL scoring loop when `gamma=0` in fast_mode
- **perf(search)**: reuse index buffer in search closures + thread_local SRHT temp
- **perf(search)**: eliminate per-scoring-call allocations, fix EF_UPPER cap
- **perf(hnsw)**: AVX2+FMA SIMD for `score_ip_encoded_lite` construction scorer
- **perf(hnsw)**: pre-cache encoded vecs, remove `cand_pool` floor, tunable `n_refinements`
- **perf(ingest)**: f32 hot path, AVX2 FWHT, WAL V3 packing, zero-copy batch quantize
- **perf(quantizer, engine)**: parallel quantize/dequantize, fast_mode QJL skip, centroid-lookup HNSW build
- **perf(search)**: parallel brute-force scan via rayon `par_chunks`
- **perf(quantizer)**: SGEMM acceleration + O(d log d) SRHT fast-path
- **perf(quantizer)**: faster bit-unpack for packed MSE codes

### Features (algorithm)

- **feat(quantizer)**: switch to paper-conformant QR rotation and Gaussian QJL projection
- **feat(storage)**: de-duplicate codes and persist id pool
- **feat(storage)**: dequantization-based reranking — no `live_vectors.bin` overhead by default

### Infrastructure

- Python test suite: 95 tests covering all API methods, filter operators, batch ops, RAG wrapper
- CI: release workflow builds wheels for Python 3.10 / 3.11 / 3.12 / 3.13 on Linux, Windows, macOS
- **docs**: complete Python API reference (`docs/PYTHON_API.md`) with all methods, parameters, error types
- **chore(version)**: single source of truth in `pyproject.toml`; `Cargo.toml` kept in sync manually

---

## Notes

- This project follows [Semantic Versioning](https://semver.org/) with `MAJOR.MINOR.PATCH`.
- It is pre-1.0 (`0.x.y`); public API may change between minor versions.
- Version is defined in `pyproject.toml` (single source of truth). `Cargo.toml` is kept in sync manually.
- Git tags (e.g., `v0.1.0`) must match `pyproject.toml` versions.
