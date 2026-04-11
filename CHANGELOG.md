# Changelog

All notable changes to TurboQuantDB are documented in this file.

Format: `[version] — type(scope): summary`. Commits use [Conventional Commits](https://www.conventionalcommits.org/).

---

## [0.5.1] — 2026-04-11

### Added

- **`rerank_factor` at search time** — `db.search()` and `db.query()` now accept a `rerank_factor` parameter (integer multiplier). Controls how many over-sampled candidates are re-scored when `rerank=True`. Defaults: 10× for brute-force, 20× for ANN. Follows the industry pattern of Qdrant's `oversampling` and LanceDB's `refine_factor`.
- **`rerank_precision` defaults to `"f16"`** — When `rerank=True` and no explicit `rerank_precision` is provided, raw vectors are now stored as float16 (half of float32 disk usage) for exact re-scoring. Previously defaulted to dequantization-only, which produced zero recall improvement for the inner-product metric.
- **`docs/CONFIGURATION.md`** — new comprehensive configuration guide covering all parameter dimensions (`bits`, `fast_mode`, `rerank`, `rerank_factor`, `quantizer_type`, ANN vs brute-force), recommended presets for 6 common scenarios, storage estimation formulas, and a decision flowchart.
- **`benchmarks/full_config_bench.py`** — exhaustive 32-config × 4-dataset benchmark script. Runs all combinations of bits × rerank × ann × fast_mode × quantizer_type across GloVe-200, arXiv-768, DBpedia-1536, and DBpedia-3072. Generates recall curves, trade-off scatter plots, and a data-driven guidance report (`benchmarks/_full_config_report.md`, gitignored).

### Fixed

- **Rerank no-op bug** — `rerank=True` with `rerank_precision=None` previously resolved to `Disabled` (dequantization-only). For the IP metric, dequantized scores are mathematically identical to the LUT scores, so rerank had zero effect. Now defaults to `F16` exact re-scoring, giving +5–25 pp R@1 depending on dataset and bits.
- **`release.yml` update-docs job** — replaced branch+PR dance with a direct `git push origin HEAD:main`. GitHub Actions cannot create pull requests in this repository (`Allow GitHub Actions to create and approve pull requests` is off), causing the previous job to fail on every release.

---

## [0.5.0] — 2026-04-10

### Added

- **`quantizer_type="dense"` is now the default** — the Haar-uniform QR + dense Gaussian quantizer (paper-faithful) replaced `"srht"` as the default. `"srht"` remains available for streaming/high-d ingest workloads. `"exact"` is accepted as a backward-compatible alias for `"dense"`.
- **`fast_mode=False` is now the default** — QJL residual is stored and used during `rerank=True` dequantization, giving +9–15 pp R@1 over `fast_mode=True` at d ≥ 1536. Set `fast_mode=True, rerank=False` for d < 512 (QJL projections are too noisy at low d and reduce recall below the MSE-only baseline) or to reproduce paper Figure 5 recall numbers.
- **Auto query planner** — `_use_ann` now accepts `None` (the new default). When `None`, the engine automatically selects HNSW search when an index exists, N ≥ 10,000, and the unindexed delta is ≤ 20% of the corpus. Pass `True`/`False` to force a mode.
- **Range index for numeric metadata** — `$gt`/`$gte`/`$lt`/`$lte` filters now use a per-field BTreeMap index (IEEE-754 ordered keys) instead of a full scan, updated incrementally on insert/delete.
- **Equality index for metadata** — `$eq` filters resolved via an in-memory inverted index (O(1) candidate lookup), removing the need to scan all vectors on selective equality filters.
- **Filter pushdown** — the query planner resolves selective `$eq` filters to a candidate slot list before entering the scoring loop, avoiding full-corpus scans when filters are highly selective.
- **Incremental HNSW build** — `create_index()` can now build the graph layer-by-layer from existing segment data without reloading all raw vectors.
- **AVX2 SIMD paths** — `unpack_mse_indices` (b=4: 16 bytes → 32 u16 per AVX2 iteration) and float32 exact-rerank dot-product now have AVX2 fast paths.
- **`DEVELOPMENT.md`** — new contributor guide with prerequisites, build/test/benchmark commands, and sprint workflow.

### Fixed

- **`fast_mode=True` dequantize panic** — `dequantize()` now short-circuits to MSE-only in fast mode, preventing a zero-length QJL slice panic during rerank.
- **`live_codes` stride correctness** — stride now computed from `quantizer.n` instead of `next_power_of_two(d)`, so dense mode (n=d) and srht mode (n=next_power_of_two(d)) both get correct slot offsets on insert and search.
- **Delete-reinsert correctness** — WAL entries applied in insertion order so a delete-then-reinsert sequence preserves the latest slot across flush and reopen.
- **Python boundary hardening** — NaN/Inf rejection in insert/search vectors; dimension mismatch, invalid `bits`/`dimension`, negative `top_k`/`offset`/`limit` all raise `ValueError` instead of `PanicException`.
- **Unknown filter operators** — `search()`, `query()`, `list_ids()`, `count()`, and `delete_batch()` now raise `ValueError` on unrecognised `$`-prefixed operators.
- **`include=` validation** — unknown field names in the `include` parameter raise `ValueError` instead of silently returning empty dicts.
- **Collection path traversal** — collection names containing `..`, `/`, or `\` raise `ValueError` at the Python layer.
- **Server: concurrent create race** — `create_collection` serialised with a per-state mutex, preventing TOCTOU corruption when two requests both see "not found" and write the manifest simultaneously.
- **Server: path traversal** — all route handlers validate tenant/database/collection path components, rejecting `..` and separator characters.
- **Server: lock ordering** — jobs lock released before `dispatch_queued_jobs` to eliminate self-deadlock.
- **Server: scoped URI** — `open_collection_scoped` now uses the flat storage path, fixing 500 errors from manifest path mismatches in tests.
- **ChromaDB/LanceDB compat** — threading locks on `ChromaClient` and `LanceDBConnection` create paths; unknown operator rejection; SQL `IN` clause trailing-comma fix; `limit(-1)` raises `ValueError`; duplicate `create_table` raises `ValueError` in create mode.
- **`rag.py`** — `float64→float32` dtype cast; `similarity_search` returns dict results correctly; class/method docstrings added.
- **QA pass** — 381/381 tests passing (adversarial, market simulation, server blackbox suites added).

### Performance

- **WAL write coalescing** — `append_batch` pre-builds the full byte buffer and calls `write_all` + `flush` once per batch, eliminating per-entry syscall overhead.
- **WAL `BufWriter` increased to 4 MB** — reduces system calls per `append_batch` from ~5,000 to ~8 for 1536-dim entries.
- **ANN `search_batch` parallelised** — Rayon `par_iter` across queries for the ANN path; 1.46× throughput improvement at batch=8.
- **Brute-force batch queries always parallelised** — removed the large-N sequential guard; Rayon work-stealing handles nested `par_iter` + `par_chunks` without over-subscribing the thread pool.

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

- **feat(quantizer)**: switch quantizer internals to the paper-faithful QR rotation + Gaussian QJL formulation (the current docs refer to this formulation as the `exact` path)
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
