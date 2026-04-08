# Changelog

All notable changes to TurboQuantDB are documented in this file.

Format: `[version] — type(scope): summary`. Commits use [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

---

## [0.4.0] — 2026-04-08

### Added
- Delta overlay: vectors inserted after `create_index()` are tracked in a persisted `delta_ids.json` and merged into ANN search results without requiring a graph rebuild
- Parallel ingest: Rayon-based quantization/normalization for large batches with sequential fallback for small ones

### Fixed
- Compaction crash recovery: segments written to `.tmp` then atomically renamed; orphan `.tmp` files cleaned on startup
- ANN delta filter: replaced per-query `HashSet` allocation with `is_slot_alive()` O(1) check
- `maybe_persist_state`: delta slots written to both `local_dir` and backend to prevent stale local file winning on reopen
- S3 `rename`: switched to server-side `store.copy()` (no RAM spike) with crash-safe ordering
- `release.yml`: `id-token: write` added to `release` job — was overridden by job-level permissions block, breaking OIDC publish

### Performance
- Batch insert: `indexed_set` built once per batch (not per chunk) — O(batch + indexed) instead of O(chunks × indexed)
- Single-vector insert: `binary_search` O(log n) instead of `Vec::contains` O(n); `delta_slots` maintained sorted

### Changed
- `.unwrap()` audit: replaced with `.expect()` on all write/search paths for clearer panic messages

---

## [0.3.0] — 2026-04-07

### Added

- **`Database.open(path)` parameterless reopen** — `dimension` is now optional. When omitted, all fixed parameters (`dimension`, `bits`, `seed`, `metric`) are loaded automatically from the existing `manifest.json`. A `ValueError` is raised only if the database does not yet exist.
- **`delete_batch(where_filter=...)` filter-based bulk delete** — `delete_batch` now accepts an optional `where_filter` dict (same syntax as `search`). All matching vectors are deleted atomically in addition to any explicitly listed IDs; overlapping entries are not double-counted.
- **`list_metadata_values(field)`** — enumerate all distinct values stored for a metadata field across active vectors; useful for building filter UIs.
- **`normalize=True` on `Database.open()`** — automatically L2-normalizes all inserted vectors and queries at write time, making inner-product scoring equivalent to cosine similarity without changing the metric.
- **Hybrid ANN + brute-force search** — vectors inserted *after* `create_index()` are no longer silently missed. The engine detects "dark slots" (active but unindexed vectors), runs a targeted brute-force scan, and merges results with HNSW candidates before returning top-k.
- **ChromaDB compatibility shim** (`tqdb.chroma_compat`) — drop-in `PersistentClient(path)` backed by `tqdb.Database`; supports `get_or_create_collection`, `add`/`upsert`/`update`/`delete`/`get`/`query`/`peek`/`count`/`modify`; metric parsed from `{"hnsw:space": "cosine"}`; where-filter operators `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$and/$or/$exists/$contains`.
- **LanceDB compatibility shim** (`tqdb.lancedb_compat`) — `connect(uri)` factory with `create_table`/`open_table`/`drop_table`; fluent `CompatQuery` builder; PyArrow and `list[dict]` ingestion; SQL WHERE parser supporting `field = 'val'`, `field != 'val'`, `field IN (...)`, and numeric comparisons (`>`, `>=`, `<`, `<=`).
- **S3 segment backend** (`--features cloud`) — `StorageProvider` implementation backed by `object_store`; write-through local cache; configured via `TQDB_S3_BUCKET` / `TQDB_S3_PREFIX` env vars.
- **Server restore endpoint** — `POST .../restore` atomically copies a snapshot back into the live collection directory.
- **Prometheus `/metrics` endpoint** — per-tenant vector count, WAL buffer size, and index node gauges.
- **`.pyi` type stubs** — shipped in the wheel; enables IDE autocomplete and mypy for all `Database` methods including `normalize` and `list_metadata_values`.

### Fixed

- **QJL-Hamming HNSW recall** (0.164 → 0.831) — `prepare_ip_query_from_codes` set `sq=0`, zeroing the QJL component during graph construction while search used the full LUT. Fix: blend MSE score with `hamming_score(from_bits, to_bits) − 0.5` as a sign-code proximity proxy during construction.
- **Brute-force P95 latency on Windows** (130 ms → 2.2 ms) — Rayon `par_chunks` thread park/unpark overhead (~15–40 ms) dominated sub-millisecond scoring work for small corpora. Fix: sequential path for N ≤ 20 k, parallel above.
- **WAL CRC32 integrity** — WAL v5 adds per-entry CRC32; corrupted entries are detected and rejected on replay. Legacy v4 WALs remain readable.
- **Segment CRC32 integrity** — segment records now include CRC32 + format sentinel; malformed records detected at read time.
- **ChromaDB shim correctness** — float64→float32 dtype, `update()` batch, `get`/`delete` where-filter via `list_ids()`, empty-string auto-embed removed, `$exists`/`$contains` operators added.
- **LanceDB shim correctness** — float64→float32 dtype, `to_arrow()`/`to_pandas()` record fetch, `count_rows(filter=)` for `id IN (...)`, SQL parser gaps, metric mismatch warning, dim fallback from `manifest.json`.
- **Server compilation** — added missing engine stubs.

### Performance

- **Sequential brute-force for N ≤ 20 k** — avoids Rayon thread scheduling overhead on Windows; parallel path preserved for larger corpora.
- **Pre-commit hook** — paper benchmark (N=100 k, ~20 min) is now opt-in via `TQDB_TRACK=1`; hook finishes in ~2 min per commit.

---

## [0.2.1] — 2026-04-05

### Fixed

- **`_use_ann` flag now works** — previously the parameter was silently ignored (Rust `_` prefix convention); the engine always used HNSW when an index existed. Now `_use_ann=False` (the default) always uses brute-force scoring regardless of whether an index has been built. Pass `_use_ann=True` to engage the HNSW index.
- **Disk measurement inflation in ingest benchmarks** — `GROW_SLOTS` pre-allocation inflated reported disk sizes by ~18%. Fixed by calling `db.close()` (triggers `truncate_to(slot_count)`) before measuring file sizes in `precommit_perf_check.py` and `paper_recall_bench.py`.
- **UnicodeEncodeError on Windows** — benchmark scripts now reconfigure stdout to UTF-8 on startup, fixing crashes on cp1252 consoles.

### Performance

- **Skip compaction on pure inserts** — compaction is now skipped when no deletes are pending, eliminating unnecessary segment merges during bulk ingest. Throughput improvement: 2–3×.

### Infrastructure

- Benchmark scripts auto-regenerate `_perf_history.html` after every `--track` run.
- README ANN search examples updated to include `_use_ann=True` where applicable.

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
