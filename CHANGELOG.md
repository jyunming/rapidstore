# Changelog

All notable changes to TurboQuantDB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`quantizer_type=None` default is now dimension-aware.** Picks `"dense"` (Haar QR) for `d < 1024` and `"srht"` (Walsh–Hadamard) for `d >= 1024`. Empirical wins for SRHT once the rotation dominates ingest/latency cost: 2–3× ingest throughput, 1.5–3× lower p50 latency, recall equal or slightly better than dense in the public bench (R@1 0.962 vs 0.958 at d=1536 b=4 rerank=F; 0.980 vs 0.963 at d=3072 b=4 rerank=F). Below d=1024 the SRHT pow2-padding tax wipes the win so dense remains the default. To restore explicit control, pass `quantizer_type="dense"` or `quantizer_type="srht"`.

### Performance

- **bf16 storage for the dense Haar QR rotation matrix.** `quantizer.bin` is now ~50% smaller for `quantizer_type="dense"` databases: 9 MB → 4.5 MB at d=1536, 36 MB → 18 MB at d=3072. The matrix is rehydrated to f32 in memory at load time, so the rotation hot path is unchanged. Recall verified unchanged on GloVe-200 with 10k queries (ΔR@1 = +0.0002).

### Migration

- **v0.8 → v0.9 dense-mode databases need to be rebuilt.** The bf16 rotation storage is bincode-incompatible with v0.8.x format. SRHT-mode databases reopen cleanly; only `quantizer_type="dense"` (or the legacy `"exact"` alias) is affected. There is no automatic migration in v0.9 — re-quantize from source vectors. A polish migration tool may follow in v0.10+.

---

## [0.8.3] — 2026-05-03

### Added

- **`rerank_precision="residual_int4"`** — a new compression-first rerank option. Stores the quantization residual at 4-bit precision instead of the full vector, cutting rerank storage by ~31% compared to `int8` while keeping recall within 1 percentage point. Ingest is ~1.4–1.5× slower than `int8`. Opt-in; `int8` remains the default.
- CI now verifies builds on macOS Apple Silicon and Linux ARM64 in addition to x86_64.

### Performance

- Brute-force search is up to **72% faster** on high-dimensional embeddings (d=1536, d=3072) with no recall change. All bit-rates (1, 2, 4, 8 bits) now benefit from the same SIMD optimizations; previously only 4-bit databases were optimized.
- Batch ingest at d=1536 and d=3072 is **7–8% faster**.
- ANN index quality improved for low-dimensional databases (d ≤ 384): R@1 and R@10 each increase by up to 6 percentage points with no rebuild required.
- ANN search is ~5% faster on x86_64 hardware.

### Deprecated

- **`rerank_precision="int4"`** — raw INT4 reranking is deprecated. It was outperformed by `rerank=False` at every tested configuration and actively hurts ranking. The option is kept for backward compatibility; use `rerank_precision="residual_int4"` for compact reranking, or omit `rerank_precision` entirely.

### Fixed

- Multi-query batch search (`db.query()`) returned incorrect scores when using the cosine metric with `normalize=False`. Scores are now correct in all configurations.
- A shutdown sequence bug could leave remote storage in an inconsistent state. Database close now ensures vectors are written before the ID index, so the database is always recoverable.

---

## [0.8.2] — 2026-04-30

### Fixed

- The IVF coarse index (created with `create_coarse_index()`) became silently stale after compaction, returning incorrect search results. Compaction now invalidates the IVF index; subsequent searches fall back to brute-force until `create_coarse_index()` is called again.
- Vectors inserted after `create_index()` could be tracked with stale slot references after compaction, causing incorrect search results. Slot tracking is now reset correctly after every compaction.
- **Security (server only):** A path traversal vulnerability in the snapshot create/restore endpoints could allow requests with `../../`-style names to access files outside the snapshots directory. Snapshot names are now validated to reject traversal sequences.

---

## [0.8.1] — 2026-04-30

### Performance

- ANN (HNSW) search is up to **17× faster** with recall improvements of up to **19 percentage points** across all dimensions. The fix uses dimension-aware candidate pool sizes: the previous fixed default over-expanded the search beam at high dimensions, making ANN *slower* than brute-force at d=1536. The new defaults step down with dimension automatically.
- Rerank candidate pool sizes now scale with embedding dimension, reducing unnecessary work at high dimensions.

### Fixed

- `top_k * rerank_factor` could overflow on pathological inputs; both ANN and brute-force paths now use safe multiplication with a sensible cap.
- WAL replay silently skipped truncated entries; they are now logged as warnings so partial writes are visible in production logs.
- A corrupt WAL length field could cause an out-of-memory error on replay; now rejected with a clear error message.
- HNSW search with a scorer returning NaN produced non-deterministic rankings; NaN scores are now treated as the lowest possible rank.
- Deeply nested metadata filters (e.g. `{"$and": [{"$and": [...]}]}`) could cause a stack overflow on adversarial input; nesting is now capped at 32 levels.
- `create_coarse_index(n_clusters=0)` and `search(..., nprobe=0)` now raise clear errors instead of producing undefined behavior.
- The benchmark version reporter was reading a stale cached version after `maturin develop`; it now reads `pyproject.toml` directly.

### Documentation

- `Database.search()` docstring updated to reflect the new dimension-aware `rerank_factor` defaults.

---

## [0.8.0] — 2026-04-28

### Added

- **`MultiVectorStore`** — ColBERT-style late-interaction retrieval. Each document can store multiple token vectors; queries score via MaxSim (`Σ_i max_j <q_i, d_j>`). See `docs/MULTI_VECTOR.md` for the quickstart and API reference.
- **LangChain v2 `VectorStore` integration** (`tqdb.vectorstore.TurboQuantVectorStore`) — implements the full LangChain v2 `VectorStore` ABC, compatible with LCEL pipelines and `as_retriever()`. Install with `pip install tqdb[langchain]`.
- **LlamaIndex `BasePydanticVectorStore` integration** (`tqdb.llama_index.TurboQuantVectorStore`) — supports `add`, `query`, `delete`, and `MetadataFilters`. Install with `pip install tqdb[llamaindex]`.
- **`AsyncDatabase`** (`tqdb.aio`) — asyncio-friendly wrapper for all `Database` methods. Long-running calls dispatch to a thread-pool executor and run in parallel. Supports async context manager and a `.sync` escape hatch for cheap operations.
- **Chroma / LanceDB migration toolkit** (`tqdb.migrate`) — migrate an existing ChromaDB or LanceDB collection to TurboQuantDB in one call, preserving IDs, vectors, metadata, and document text. CLI: `python -m tqdb.migrate` or the `tqdb-migrate` console script. Install with `pip install tqdb[migrate]`.

### Fixed

- `insert_batch()` now accepts `None` entries in the `metadatas` list (e.g. as returned by `chromadb.collection.get()` on collections without metadata). They are treated as empty dicts, matching the documented contract.

### Documentation

- `docs/MULTI_VECTOR.md` — MultiVectorStore quickstart, API reference, and recommended ColBERT configuration.
- `docs/MIGRATION.md` — migration toolkit install, CLI, and API reference.
- `docs/PYTHON_API.md` — new Async API section.
- New optional-dependency extras in `pyproject.toml`: `migrate`, `migrate-chroma`, `migrate-lancedb`, `langchain`, `llamaindex`.

---

## [0.7.0] — 2026-04-28

### Added

- **Hybrid sparse + dense search** — `db.search()` and `db.query()` now accept a `hybrid={"text": str, "weight": float}` argument. BM25 keyword results are fused with dense vector results via Reciprocal Rank Fusion (RRF). Useful when queries mix semantic meaning and exact keywords (e.g. product names, entity names). `TurboQuantRetriever.similarity_search()` also supports `hybrid`.
- **BM25 full-text index** — automatically maintained from the `document` field on every insert, upsert, and delete. No separate setup required; used automatically when `hybrid=` is passed to `search()`.
- **`benchmarks/retrieval_eval.py`** — repeatable retrieval evaluation harness reporting R@1, R@10, MRR@10, NDCG@10, and latency on semantic, lexical, and mixed query sets.

### Performance

- Dense and ANN search performance is unchanged from v0.6.0.
- Hybrid search adds ≤ 1.5× overhead over dense-only at default settings.

### Documentation

- `docs/BENCHMARKS.md` — new "Hybrid retrieval evaluation" section.
- `docs/PYTHON_API.md` — `search()` / `query()` updated with the `hybrid` parameter.

---

## [0.6.0] — 2026-04-23

### Added

- **IVF coarse index** — `db.create_coarse_index(n_clusters=256)` builds an Inverted File Index for 2–4× faster search at N ≥ 50k. Activate per-query with `db.search(..., nprobe=N)`. Persisted automatically; loaded on re-open.
- **`$in` / `$nin` filter fast paths** — these operators (and single-field `$or`) now use index lookups instead of a full scan, up to 11× faster on selective filters.

### Performance

- The metadata index is now persisted to disk; `Database.open()` loads it directly instead of scanning all vectors (1.7× faster startup at N=100k, scales with corpus size).
- Metadata writes are journaled, deferring the full rewrite to checkpoint/close.
- Dense-mode (`quantizer_type="dense"`) ingest now matches or exceeds SRHT throughput at d ≥ 512.
- Multi-query `db.query()` at very large N (≥ 500k) reads live codes once and scores all queries simultaneously, reducing memory traffic.

### Fixed

- ChromaDB compat: `list_collections()` updated for ChromaDB ≥ 1.5 (now returns strings instead of `CollectionInfo` objects).
- RAG hybrid results now support both `doc.page_content` / `doc.metadata` (LangChain Document) and `doc["id"]` / `doc["score"]` (dict) access patterns.

---

## [0.5.2] — 2026-04-13

### Added

- **`db.checkpoint()`** — explicitly flush the WAL and trigger segment compaction. Useful after large bulk loads.
- **`include=` for `db.query()`** — control which fields appear in results (same as `db.search()`). Reduces overhead when you only need scores or IDs.
- **`rerank_factor` for `db.query()`** — consistent with `db.search()`.

### Performance

- `fast_mode=True` databases no longer allocate or store the QJL projection, reducing memory and disk footprint.
- Segment files are deleted on clean close, saving 2–4 MB per database (WAL and live codes are sufficient for recovery).
- Small search batches (< 4 queries) use a sequential path, avoiding thread scheduling overhead.
- Compaction is automatically triggered when segment count exceeds the internal threshold.

### Documentation

- Server documentation moved to `docs/SERVER_API.md`.

---

## [0.5.1] — 2026-04-12

### Added

- **INT8 / INT4 quantized rerank** — `rerank=True` now stores compressed INT8 (default) or INT4 vectors for exact second-pass rescoring. INT8 uses ~75% less disk than float32; INT4 uses ~87% less. Select via `rerank_precision="int8"` (default) or `"int4"`. For unquantized reranking use `rerank_precision="f16"`.
- **`rerank_factor` at search time** — `db.search()` and `db.query()` now accept `rerank_factor` to control how many over-sampled candidates are re-scored. Defaults: 10× for brute-force, 20× for ANN.
- **Interactive Config Advisor** — [jyunming.github.io/TurboQuantDB/advisor.html](https://jyunming.github.io/TurboQuantDB/advisor.html) — select the best `bits`, `rerank`, `fast_mode`, and ANN settings for your embedding size and use case.
- **`tqdb-server` bundled in wheel** — `pip install tqdb` now includes the pre-built server binary. Launch with the `tqdb-server` console script.
- **ChromaDB compat** — `collection.get(include=["embeddings"])` returns original float32 vectors; `collection.id`, `collection.metadata`, `client.heartbeat()`, and `list_collections()` returning objects are all now supported.
- **LanceDB compat** — `len(tbl)`, `.schema`, `.head(n)`, `.to_list()`, `tbl.search(None)` full-table scan, `update()`, `merge_insert()`, and vector column in `to_pandas()` / `to_arrow()` are all now supported.
- **LangChain RAG** — `TurboQuantRetriever` now implements the full LangChain v2 interface: `invoke()`, `similarity_search_with_score()`, `filter=` kwarg, `from_texts()`, `delete()`, `as_retriever()`, `add_documents()`.
- `docs/CONFIGURATION.md` — comprehensive parameter guide with recommended presets for 6 common scenarios and a decision flowchart.

### Fixed

- `rerank=True` with `rerank_precision=None` previously had no effect (rerank scores were identical to the LUT scores). Now defaults to INT8 exact re-scoring, giving +5–25 percentage point R@1 improvement depending on dataset and bit-rate.
- Server: collection paths were constructed incorrectly, causing 404/500 errors in multi-tenant operations.
- Server: L2 distances were returned as positive values; now correctly negated so lower-is-better semantics are preserved.
- ChromaDB compat: `get(ids=[])` with an empty list now returns empty instead of triggering a full-table scan.
- ChromaDB compat: `modify(name=...)` now physically renames the collection directory, making the new name visible to `list_collections()`.

### Documentation

- README: new Config Advisor section, benchmark tables, and updated disk estimates.

---

## [0.5.0] — 2026-04-10

### Added

- **`quantizer_type="dense"` is now the default** — the paper-faithful QR + dense Gaussian quantizer replaced `"srht"` as the default. `"srht"` remains available for streaming or high-dimension ingest. `"exact"` is accepted as an alias for `"dense"`.
- **`fast_mode=True` is now the default** — MSE-only quantization (fastest ingest, smallest disk). Use `fast_mode=False` for +5–15 percentage point recall improvement at d ≥ 1536.
- **Auto query planner** — `_use_ann=None` (new default) automatically selects HNSW when an index exists and N ≥ 10,000 with ≤ 20% unindexed vectors. Pass `True` or `False` to force a mode.
- **Range index for numeric metadata** — `$gt` / `$gte` / `$lt` / `$lte` filters use a sorted index instead of a full scan.
- **Equality index for metadata** — `$eq` filters use an in-memory inverted index for O(1) lookup.
- **Filter pushdown** — selective `$eq` filters resolve to a candidate list before scoring, avoiding full-corpus scans.
- **Incremental HNSW build** — `create_index()` builds the graph from existing data without reloading all raw vectors into memory.
- `DEVELOPMENT.md` — new contributor guide with prerequisites, build/test/benchmark commands, and sprint workflow.

### Fixed

- `fast_mode=True` + `rerank=True` could panic during dequantization; now short-circuits correctly in fast mode.
- Live codes stride calculation was wrong for dense mode, causing incorrect slot offsets on insert and search.
- Delete-then-reinsert of the same ID could lose the reinserted vector across flush/reopen; insertion order is now preserved correctly.
- NaN/Inf vectors, dimension mismatches, invalid `bits`, and negative `top_k`/`offset`/`limit` now raise `ValueError` instead of a Rust panic.
- Unknown `$`-prefixed filter operators now raise `ValueError` instead of silently passing.
- Unknown field names in `include=` now raise `ValueError`.
- Collection names containing `..`, `/`, or `\` now raise `ValueError`.
- Server: concurrent collection creation could corrupt the manifest; now serialized with a per-state lock.
- Server: path traversal in route handlers is now rejected.
- Server: a self-deadlock in job dispatch has been eliminated.

### Performance

- WAL batch writes now use a single system call per batch, eliminating per-entry overhead.
- ANN batch search parallelized with Rayon.

---

## [0.4.0] — 2026-04-08

### Added

- **Delta overlay for ANN** — vectors inserted after `create_index()` are tracked and automatically merged into ANN search results without requiring a full graph rebuild.
- **Parallel ingest** — quantization and normalization now run in parallel for large batches, with a sequential fallback for small ones.

### Fixed

- Compaction crash recovery: segment files are now written to a temporary file and atomically renamed. Orphan temp files are cleaned on startup.
- Release CI: OIDC publish was broken due to a missing permission; now fixed.

### Performance

- Batch insert builds the indexed-ID set once per batch instead of once per chunk.
- Single-vector insert uses binary search instead of a linear scan.

---

## [0.3.0] — 2026-04-07

### Added

- **`Database.open(path)` without required parameters** — `dimension` is now optional on reopen. All parameters are loaded automatically from `manifest.json`; a `ValueError` is raised only if the database does not yet exist.
- **`delete_batch(where_filter=...)`** — filter-based bulk delete; accepts the same filter syntax as `search()`.
- **`list_metadata_values(field)`** — enumerate all distinct values stored for a metadata field across active vectors; useful for building filter UIs.
- **`normalize=True` on `Database.open()`** — automatically L2-normalizes all inserted vectors and queries at write time, making inner-product scoring equivalent to cosine similarity.
- **Hybrid ANN + brute-force search** — vectors inserted after `create_index()` are now automatically included in search results via a targeted brute-force pass merged with HNSW candidates. No more silent misses.
- **ChromaDB compatibility shim** (`tqdb.chroma_compat`) — `PersistentClient(path)` backed by TurboQuantDB, supporting `add` / `upsert` / `update` / `delete` / `get` / `query` / `peek` / `count` / `modify` and all standard where-filter operators.
- **LanceDB compatibility shim** (`tqdb.lancedb_compat`) — `connect(uri)` factory with `create_table` / `open_table` / `drop_table`; fluent query builder; PyArrow and `list[dict]` ingestion; SQL WHERE parser.
- **S3 segment backend** (`--features cloud`) — persist vector data to S3-compatible storage via `TQDB_S3_BUCKET` / `TQDB_S3_PREFIX` env vars.
- **Server: snapshot restore** — `POST .../restore` atomically restores a snapshot to the live collection directory.
- **Server: Prometheus `/metrics` endpoint** — per-tenant vector count, WAL buffer size, and index node gauges.
- **`.pyi` type stubs** — shipped in the wheel for IDE autocomplete and mypy support.

### Fixed

- ANN recall was extremely poor (R@1 ≈ 0.16 → 0.83) due to a bug in HNSW graph construction that zeroed out the QJL scoring component during index build.
- Brute-force P95 latency on Windows was ~130 ms at small N due to Rayon thread scheduling overhead; a sequential path for N ≤ 20k brings it down to ~2 ms.
- WAL and segment files now include CRC32 integrity checks; corrupt entries are detected and rejected on replay.
- ChromaDB and LanceDB shim: float64→float32 dtype conversion, various missing operators and methods.

---

## [0.2.1] — 2026-04-05

### Fixed

- `_use_ann` parameter was silently ignored; the engine always used HNSW when an index existed. Now `_use_ann=False` (the default) correctly uses brute-force regardless of whether an index has been built.
- Disk sizes reported by benchmarks were inflated by pre-allocation padding. Now measured after a clean close.
- `UnicodeEncodeError` on Windows cp1252 consoles in benchmark scripts; stdout is now set to UTF-8 on startup.

### Performance

- Compaction is now skipped when no deletes are pending, giving 2–3× throughput improvement during pure-insert bulk loads.

---

## [0.2.0] — 2026-04-15

### Changed

- **Package renamed from `turboquantdb` to `tqdb`** — use `import tqdb` and `pip install tqdb`. The `Database` class is otherwise unchanged.

---

## [0.1.1] — 2026-04-10

### Fixed

- Release CI produced duplicate wheels and PyPI upload failures on Windows and macOS; fixed.

---

## [0.1.0] — 2026-04-03

### Added

- `rerank_precision` parameter on `Database.open()` — opt-in raw-vector reranking (`"f16"` / `"f32"`, default `None` = dequantization reranking).
- `fast_mode` parameter on `Database.open()` — skip the QJL stage for faster ingest with a minor recall tradeoff at high dimensions.
- `collection` parameter on `Database.open()` — open a named sub-directory for multi-namespace support.
- `delete_batch(ids)` — delete multiple vectors in one call; returns the count of deleted vectors.
- `count(filter=None)` — count active vectors, optionally filtered.
- `list_ids(where_filter, limit, offset)` — paginated, filtered ID listing.
- `update_metadata(id, metadata, document)` — update metadata or document text without re-uploading the vector.
- `query(query_embeddings, n_results, where_filter)` — batch multi-query accepting a 2D numpy array.
- `include=` parameter on `search()` — control which fields are returned (`"id"`, `"score"`, `"metadata"`, `"document"`).
- New metadata filter operators: `$in`, `$nin`, `$exists`, `$contains`.
- Python container protocol: `len(db)` and `"id" in db`.

---

## Notes

- This project follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`) and is pre-1.0; public API may change between minor versions.
- Version is defined in `pyproject.toml` (single source of truth). `Cargo.toml` is kept in sync manually.
