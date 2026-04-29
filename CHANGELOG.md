# Changelog

All notable changes to TurboQuantDB are documented in this file.

Format: `[version] — type(scope): summary`. Commits use [Conventional Commits](https://www.conventionalcommits.org/).

---

## [0.8.2] — 2026-04-30

Audit-driven bug-fix + test-coverage release. A 3-agent code audit surfaced ~80 candidate findings; the 9 verified critical/high items are fixed here, and 5 previously untested modules get unit-test suites. No API changes; all behavior is either an opt-in safety improvement or a contract pin.

### Fixed

- **WAL truncation logging (A1)** — `wal::replay()` previously dropped truncated payloads and missing CRC bytes silently (only CRC *mismatch* logged). Now emits `eprintln!` warnings at every truncation point with the entry index, so partial writes are observable in production logs.
- **WAL oversized-payload guard (B1)** — corrupted length field claiming a multi-GB payload no longer triggers a giant `vec![0u8; len]` allocation. New `MAX_REASONABLE_PAYLOAD = 10 MiB` cap stops replay with a clear log message instead of OOM.
- **Graph NaN-score sanitization (A2)** — a user-supplied scorer returning `f64::NAN` previously yielded undefined HNSW ranking via the `partial_cmp().unwrap_or(Equal)` fallback. Non-finite scores are now coerced to `f64::NEG_INFINITY` at every `scorer()` call site (entry point, upper-level beam, level-0 beam) — those nodes effectively rank last and ranking is deterministic.
- **Filter recursion-depth DoS guard (A3)** — `metadata_matches_filter` previously recursed without bound on `{"$and": [{"$and": [...]}]}` nesting, risking stack overflow on adversarial input. New `MAX_FILTER_DEPTH = 32` is enforced both at validation time (`validate_filter_operators`) and at evaluation time (defense-in-depth).
- **IVF zero-cluster / zero-nprobe validation (A4)** — `create_coarse_index(n_clusters=0)` now errors early with a clear message. `IvfIndex::probe(query, nprobe=0)` now clamps nprobe to at least 1 (was: called `select_nth_unstable_by(0, ...)` which is defined but useless).
- **Quantizer dimension assertions (A5)** — `score_ip_encoded` and `score_ip_encoded_lite` previously called `idx.get_unchecked(i)` for `i in 0..self.n` with no length check; a too-short `idx` was undefined behavior. Now panics with a clear message at the safe wrapper.

### Tests

- **Filter NaN-coercion behavior pin (B2)** — pinned current behavior of `as_f64().unwrap_or(NAN)` for big-int metadata so future refactors don't silently change semantics. Documented inline.
- **BM25 empty-document contract (B3)** — audit flagged this as a bug; verified it's intentional (per `put` docstring). Added 5 tests pinning the contract: empty docs are excluded from `n_docs`/`avgdl`, mixed empty+real corpora compute correctly, single-doc corpora don't div-by-zero, empty queries return empty.
- **`rerank_factor` boundary recall continuity (B4)** — new E2E tests at d=384/385 and d=1024/1025 verify recall doesn't cliff across the v0.8.1 dimension cutoffs. Marked `#[ignore]` (~30s each) — run with `cargo test -- --ignored boundary`.
- **IVF unit-test suite (C1)** — added 9 tests for the previously untested `ivf.rs` module: build assigns all slots, probe is unique+sorted, save/load roundtrip, magic validation, etc.
- **WAL replay edge cases (C2)** — added tests for oversized length field, truncated payload, truncated CRC, duplicate id entries, corrupted middle entry.
- **Filter operator coverage (C3)** — added tests for empty `$and`/`$or` arrays (vacuous truth), `$contains` on non-string, `$exists` null vs missing, single-bound range extraction.
- **Quantizer numeric edges (C4)** — added tests for zero-vector quantize, b=1 (1-bit) round-trip, `dequantize_then_score` consistency with `score_ip_encoded`.
- **Graph beam-search edge cases (C5)** — added tests for top_k=0, single-node graph, NaN/+Inf scorers (determinism + non-finite output rejection).

Test count: 387 → **425** (+38 new; 2 ignored long-running boundary tests).

---

## [0.8.1] — 2026-04-29

### Performance

- **Dimension-aware default `rerank_factor`** — at high `d`, the prior fixed default of 20 (ANN) / 10 (brute) caused the level-0 HNSW search to bloat its candidate pool and visit far more nodes than needed, making ANN actively slower than brute-force at d=1536 (~90 ms vs ~56 ms p50 on the private bench). Defaults now step down with vector dimension: ANN uses **20 / 8 / 4** for `d ≤ 384` / `d ≤ 1024` / `d > 1024` respectively; brute uses **10 / 6 / 4**. User-supplied `rerank_factor=` continues to override. No API change; pure default tuning. Results on the private bench at n=100k (ANN, rerank=True):

  | Dataset (dim) | b=2 p50 before → after | b=2 R@1 before → after | b=4 p50 before → after | b=4 R@1 before → after |
  |---|---|---|---|---|
  | GloVe-200 | 19.5 ms → **2.4 ms** | 0.211 → **0.403** | 21.6 ms → **1.3 ms** | 0.366 → **0.422** |
  | DBpedia-1536 | 90.0 ms → **13.3 ms** | 0.738 → **0.836** | 97.9 ms → **8.5 ms** | 0.822 → **0.850** |
  | DBpedia-3072 | 43.2 ms → **23.8 ms** | 0.766 → **0.849** | 51.0 ms → **15.4 ms** | 0.816 → **0.861** |

  Speed improved 1.8–17× and recall improved 2–19% across all dims. The recall gain is from a tighter `search_list_size`: the prior factor of 20 inflated `ef_search` to ~200 nodes which over-spread the level-0 beam; the smaller pool keeps the search focused on better-scoring candidates.

### Tests

- Two new unit tests in `src/storage/engine/tests.rs` cover the `default_rerank_factor` helper across the dimension cutoffs for both ANN and brute paths.

---

## [0.8.0] — 2026-04-28

> Release overview + upgrade notes: [`docs/WHAT_S_NEW_0_8.md`](docs/WHAT_S_NEW_0_8.md).

### Added

- **`MultiVectorStore` (ColBERT-style late-interaction retrieval)** — new `python/tqdb/multivector.py`. Each document gets N token vectors; queries score via MaxSim (`Σ_i max_j <q_i, d_j>`). Python-layer wrapper over the existing single-vector engine: token vectors are stored as regular slots, a JSON sidecar maps `doc_id → [token_id]`, raw float32 token vectors live in a `.npz` sidecar for exact MaxSim. Public API: `MultiVectorStore.open(path, dimension, bits=4, metric="cosine")`, `insert(doc_id, vectors, document, metadata)`, `search(query_vectors, top_k, oversample=4, candidate_filter=None)`, `delete(doc_id)`, `get(doc_id)`. A future v0.9 native engine path will replace the wrapper while keeping the public API stable. Documented in `docs/MULTI_VECTOR.md`.
- **LangChain v2 `VectorStore` integration** — new `python/tqdb/vectorstore.py` exporting `TurboQuantVectorStore(VectorStore)` with the full v2 ABC: `add_texts`, `add_documents`, `similarity_search`, `similarity_search_with_score`, `similarity_search_by_vector`, `delete`, `get_by_ids`, `from_texts`, `from_documents`, `as_retriever`, `_select_relevance_score_fn`, `embeddings`. Lazy class build via PEP 562 `__getattr__`; LangChain itself is an optional dep (`tqdb[langchain]`).
- **LlamaIndex `BasePydanticVectorStore` integration** — new `python/tqdb/llama_index.py` exporting `TurboQuantVectorStore` with `add(nodes)`, `query(VectorStoreQuery)`, `delete(ref_doc_id)`, `delete_nodes`, `clear`, `persist`. `MetadataFilters` / `MetadataFilter` / `FilterOperator` / `FilterCondition` are translated to TQDB's MongoDB-style dialect. Lazy import; optional dep `tqdb[llamaindex]`.
- **`AsyncDatabase` — asyncio-friendly facade** — new `python/tqdb/aio.py`. Every long-running `Database` method has an awaitable counterpart that dispatches to a thread-pool executor. PyO3 already releases the GIL inside, so 50 concurrent `await db.search(...)` calls genuinely run in parallel (verified by test). Auto-created executor sized to `min(32, cpu_count + 4)`; user can supply their own. Async context manager support. `db.sync` escape hatch for cheap O(1) operations.
- **Chroma / LanceDB migration toolkit** — new `python/tqdb/migrate.py` with `migrate_chroma(src, dst, collection=None)` and `migrate_lancedb(src, dst, table_name)`. Reads each source library's native on-disk format (via the source library itself) and bulk-inserts into a fresh TQDB. CLI: `python -m tqdb.migrate {chroma|lancedb} <src> <dst>` and the `tqdb-migrate` console script. Preserves IDs, vectors, metadata, document text. Optional deps: `tqdb[migrate]` (both), `tqdb[migrate-chroma]`, `tqdb[migrate-lancedb]`.

### Fixed

- **`parse_metadata_rows` accepts `None` per-row entries** — the public `.pyi` stub for `Database.insert_batch` documents `metadatas: list[dict | None] | None`, but the Rust side previously rejected `None` entries with a TypeError. Chroma's `collection.get(include=["metadatas"])` returns `[None, None, ...]` for collections without metadata, which broke migration. Fixed to treat `None` as an empty dict per row, matching the documented contract.

### Documentation

- **`docs/MULTI_VECTOR.md`** — quickstart, API reference, knobs, recommended ColBERTv2 config, and a "limitations until v0.9" note.
- **`docs/MIGRATION.md`** — install / CLI / API / preserved-fields reference for the migration toolkit.
- **`docs/PYTHON_API.md`** — new "Async API" section with quickstart and constructor reference.
- **`pyproject.toml`** — five optional-deps extras: `migrate`, `migrate-chroma`, `migrate-lancedb`, `langchain`, `llamaindex`.

### Tests

- 9 tests for the LangChain integration (`tests/test_langchain_compat.py`)
- 9 tests for the LlamaIndex integration (`tests/test_llama_index.py`)
- 9 tests for `AsyncDatabase` including a 50-concurrent-search proof of parallelism (`tests/test_async_api.py`)
- 8 tests for the migration toolkit covering Chroma + LanceDB round-trip + CLI smoke (`tests/test_migrate.py`)
- 10 tests for the multi-vector store including a MaxSim correctness check (`tests/test_multivector.py`)
- All pre-existing suites (Rust 422, Python 14) continue to pass unchanged.

---

## [0.7.0] — 2026-04-28

### Added

- **BM25 sparse retrieval foundation** — new `Bm25Index` sidecar (`bm25.idx`) maintained alongside the dense store. Indexes the `document` field of every vector via a deterministic Unicode-word tokenizer (lowercase + FNV-1a hashing) and Okapi BM25 (k1=1.2, b=0.75). Persisted on every flush via tmp+rename, rebuilt from `metadata.iter_docs()` when the sidecar is missing or corrupt. Insert/upsert/delete/compaction paths all keep BM25 in sync. Engine API: `engine.search_bm25(query_text, top_k, filter)`.
- **`db.search(…, hybrid={…})` and `db.query(…, hybrid={…})`** — hybrid sparse+dense retrieval via Reciprocal Rank Fusion (RRF). Shape: `{"text": str, "weight": float, "rrf_k": float, "oversample": int}`. `weight` ∈ [0, 1] (default 0.5) controls the BM25 contribution; `rrf_k` (default 60) is the smoothing constant; `oversample` (default 4) is the per-list candidate multiplier. Dense and BM25 legs run in parallel via `rayon::join`. `query()` accepts `texts: [str]` for per-row text in batch mode. Empty text collapses to dense-only fast path.
- **`TurboQuantRetriever.similarity_search(..., hybrid={…})`** — the LangChain wrapper at `python/tqdb/rag.py` passes `hybrid` through to the underlying `Database.search` so RAG users can opt in without touching the lower-level API.
- **`benchmarks/retrieval_eval.py`** — repeatable retrieval-eval harness that scores dense, BM25, and hybrid paths on three query sets (semantic, lexical, mixed) over a synthetic self-contained corpus. Reports R@1, R@10, MRR@10, NDCG@10, and p50/p95 latency in one markdown table; appends to `benchmarks/retrieval_eval_history.json`. Documented in `docs/BENCHMARKS.md`.

### Performance

- No change to dense or ANN paths — all 24 paper-bench configs (3 datasets × 8 configs) at v0.7.0 are within ±1pp R@1 of v0.6.0 and ≤ baseline on every latency/ingest metric. Hybrid `search_hybrid()` is amortized O(K') per query where K' = oversample × top_k, i.e. ≤ 1.5× the dense-only latency at default settings.
- Sparse index lookup (`engine.search_bm25`) ≤ 0.5 ms p95 at N = 5k on the harness's synthetic corpus.

### Documentation

- `docs/BENCHMARKS.md` — new "Hybrid retrieval evaluation" section explaining the harness, the three query sets, and the expected outcome (hybrid raises mixed R@10 from 0.500 to 1.000 over either path alone).
- `docs/PYTHON_API.md` — `search()` / `query()` signatures updated with the `hybrid` kwarg.
- `python/tqdb/tqdb.pyi` — type stubs reflect the new kwarg on both methods.

### Tests

- 7 tokenizer unit tests + 12 BM25 unit tests + 8 RRF unit tests (`src/storage/{tokenizer,bm25,rrf}.rs`).
- 12 engine integration tests in `tests/test_bm25.rs`: keyword match, delete, upsert, persistence round-trip, hybrid recovers keyword-dense-misses, hybrid weight=0 collapses to dense, slot-reuse identity (no doc leak across deleted IDs), cold-start rebuild matches persisted, compaction with documents preserves search, hybrid with empty BM25 falls back to dense.
- 14 Python boundary tests in `tests/test_python_hybrid.py`: malformed dict shapes, wrong types, weight/rrf_k/oversample out-of-range, both `text` and `texts` set, `texts` length mismatch, Unicode round-trip.

---

## [0.6.0] — 2026-04-23

### Added

- **`db.create_coarse_index(n_clusters=256)`** — builds an IVF (Inverted File Index) coarse routing index by running k-means in the MSE-rotated SRHT space. Persisted as `ivf.bin`; loaded automatically on re-open.
- **`db.search(…, nprobe=N)`** — activates IVF coarse routing when an IVF index exists. Scores only the top-nprobe clusters (≈ nprobe/k of the corpus). 2–4× speedup at N ≥ 50k; recall degrades gracefully with decreasing nprobe.
- **`$in` / `$nin` / single-field `$or` filter fast paths** — filters of these patterns now use `eq_index` lookups instead of an O(N) full scan. `$eq` (5k/20k) is 11.7×, `$in`/`$or` (10k/20k) 6–7× faster than baseline scan.

### Performance

- **Dense-mode GEMM ingest** — `quantizer_type="dense"` now rotates all B vectors via a single SGEMM call (nalgebra/matrixmultiply) instead of B separate d×d matrix-vector multiplies. Dense ingest at d ≥ 512 now matches or exceeds SRHT throughput.
- **Blocked batch scorer** — new `score_batch_brute()` kernel for `db.query()` at very large N (≥ 500k) reads `live_codes.bin` once and scores all Q queries simultaneously; reduces memory traffic from Q×N to N code-reads.
- **Metadata index persistence** — `eq_index` and `range_index` now persisted as `metadata.idx` on every flush. `Database.open()` loads the persisted indexes directly, skipping the O(N) rebuild (1.7× faster startup at N=100k, scales with corpus size).
- **Metadata WAL** — `put_many()` appends O(batch_size) WAL entries to `metadata.wal` instead of touching `metadata.bin`; the O(N) full rewrite is deferred to `checkpoint()`/close.
- **Exact-index post-scan skip** — `$in`, `$nin`, single-field `$or`, and range conditions now short-circuit the per-candidate metadata re-evaluation step (was previously only done for pure-`$eq`). Measured ~20% end-to-end latency reduction on GloVe-200.

### Fixed

- **HNSW neighbor-selection deduplication** — `choose_top_ids()` helper replaces 4 identical 11-line blocks in the graph build/insert paths.
- **Integration test arg count** — `search_with_filter_and_ann` call sites updated to match the 6-argument signature added in v0.5.2.
- **ChromaDB ≥ 1.5 compat** — `list_collections()` now returns `List[str]` (was `List[CollectionInfo]`). Rename keeps the directory name stable (logical-only metadata rename).
- **RAG hybrid results** — `SearchResultDocument` hybrid object supports both `doc.page_content` / `doc.metadata` (LangChain Document style) and `doc["id"]` / `doc["score"]` (dict style).

---

## [0.5.2] — 2026-04-13

### Added

- **`db.checkpoint()`** — new public method; triggers an immediate WAL flush and segment compaction when the compaction threshold is reached. Useful for explicit maintenance after large bulk loads.
- **`include=` for `db.query()`** — `query()` now accepts the same `include` parameter as `search()`. Pass a subset of `["id", "score", "metadata", "document"]` to control which fields appear in each result dict. Defaults to all four. Reduces Python-side allocation and serialization overhead for callers that only need scores or IDs.
- **`rerank_factor` for `db.query()`** — `query()` now accepts `rerank_factor` (previously only on `search()`). Consistent with the single-query API.

### Performance

- **QJL no-op in `fast_mode`** — `fast_mode=True` databases now use a zero-allocation placeholder instead of the QJL projection (dense mode: saves D²×4 bytes per database; SRHT mode: saves 4D bytes). The projection is never used in fast_mode, so this reduces both disk footprint and open-time memory.
- **Sparse ID pool on disk** — the ID pool is now serialized without the redundant `hashes` array; hashes are recomputed on load. Saves 8 bytes × slot_count per database.
- **Dense alive-bitmap encoding** — when all IDs match the `id-{slot}` pattern (common in benchmarks and migrations), the ID pool is stored as a compact bit-array. Reduces live_ids.bin size by ~10× for these workloads.
- **Segment files deleted on clean close** — immutable segment files (crash-recovery fallbacks) are deleted when the database is closed cleanly. Saves 2–4 MB per database; state is recovered from live_codes.bin + WAL on reopen.
- **Sequential path for small search batches** — `search_batch` / `query()` now uses a sequential loop for batches with < 4 queries, avoiding Rayon scheduling overhead on interactive RAG calls.
- **Adaptive parallel threshold by dimension** — brute-force scoring switches to inner parallelism only when the candidate pool exceeds a per-dimension size threshold, reducing thread contention for small corpora.
- **Auto-compaction on WAL flush** — when the segment count exceeds the `AUTO_CHECKPOINT_SEGMENTS_THRESHOLD` (64), a compaction checkpoint is automatically triggered on the next WAL flush.
- **`q_norm_inv` precomputation** — query norm inverse precomputed once per search call instead of per-candidate, reducing redundant divisions in the scoring loop.
- **Zero-copy metadata filter scan** — `get_many_properties()` returns only the properties map (not the full `VectorMetadata` struct), reducing allocations in hot filter and list_ids paths.
- **Empty metadata skip** — `update_metadata()` now deletes the metadata entry when both properties and document are empty/None, avoiding writes of zero-content rows.

### Documentation

- **`server/README.md` → `docs/SERVER_API.md`** — server documentation consolidated into the `docs/` tree. README link updated.

---

## [0.5.1] — 2026-04-12

### Added

- **INT8/INT4 quantized rerank** — `rerank=True` now stores compressed INT8 (default) or INT4 raw vectors for exact second-pass rescoring. INT8 uses per-vector scale factors (~75% less disk than f32); INT4 packs two values per byte (~87% less disk). Select via `rerank_precision="int8"` (default) or `"int4"`. For exact rescoring without quantization use `rerank_precision="f16"`.
- **`rerank_factor` at search time** — `db.search()` and `db.query()` now accept a `rerank_factor` parameter (integer multiplier). Controls how many over-sampled candidates are re-scored when `rerank=True`. Defaults: 10× for brute-force, 20× for ANN. Follows the industry pattern of Qdrant's `oversampling` and LanceDB's `refine_factor`.
- **`rerank_precision` defaults to `"int8"`** — When `rerank=True` and no explicit `rerank_precision` is provided, raw vectors are stored as per-vector-scaled INT8 (~75% less disk than f32, same R@1 as f16 for inner-product search). Use `rerank_precision="f16"` for exact rescoring without quantization.
- **Config Advisor** — interactive web tool at [jyunming.github.io/TurboQuantDB/advisor.html](https://jyunming.github.io/TurboQuantDB/advisor.html). Selects the best `bits` / `rerank` / `fast_mode` / ANN combination for a given embedding dimension and use case (RAG, search-at-scale, edge deployment, etc.). Scored against real benchmark data with adjustable priority weights for recall, compression, and speed.
- **`tqdb-server` bundled in wheel** — `pip install tqdb` now ships the pre-built server binary at `tqdb/_bin/tqdb-server[.exe]`. The `tqdb-server` console script launches it directly. CI builds and embeds the binary for Linux x86_64, Windows x86_64, and macOS (x86_64 + arm64).
- **`docs/CONFIGURATION.md`** — new comprehensive configuration guide covering all parameter dimensions (`bits`, `fast_mode`, `rerank`, `rerank_factor`, `quantizer_type`, ANN vs brute-force), recommended presets for 6 common scenarios, storage estimation formulas, and a decision flowchart.
- **`benchmarks/full_config_bench.py`** — exhaustive 32-config × 4-dataset benchmark script. Runs all combinations of bits × rerank × ann × fast_mode × quantizer_type across GloVe-200, arXiv-768, DBpedia-1536, and DBpedia-3072. Generates recall curves, trade-off scatter plots, and a data-driven guidance report.
- **ChromaDB compat — embeddings retrieval** — `collection.get(include=["embeddings"])` and `collection.query(include=["embeddings"])` now return the original float32 vectors, stored in a thread-safe side-car `.npz` file alongside the tqdb database.
- **ChromaDB compat — `collection.id` / `collection.metadata`** — `CompatCollection` now exposes a stable UUID5 `id` property and a `metadata` property loaded from `_chroma_meta.json`.
- **ChromaDB compat — `client.heartbeat()`** — returns current time in nanoseconds, matching `chromadb.PersistentClient.heartbeat()`.
- **ChromaDB compat — `list_collections()` returns objects** — now returns `CollectionInfo` objects with `.name`, `.id`, and `.metadata` attributes instead of plain strings.
- **LanceDB compat — `__len__`, `schema`, `head(n)`, `to_list()`** — `CompatTable` now supports `len(tbl)`, `.schema` (PyArrow schema inferred from stored data), `.head(n)` (first n rows as Arrow Table), and `.to_list()`.
- **LanceDB compat — `search(None)` full-table scan** — `tbl.search(None).to_list()` performs a full-table scan, matching real LanceDB behaviour.
- **LanceDB compat — `update(where, values)`** — updates metadata/vector/document for rows matching a SQL WHERE clause. Handles `id = 'x'` as a direct primary-key lookup.
- **LanceDB compat — `merge_insert()`** — fluent builder supporting `when_matched_update_all()` / `when_not_matched_insert_all()` / `execute(data)`.
- **LanceDB compat — vector column in `to_pandas()` / `to_arrow()`** — original float32 vectors are now included via thread-safe `_VecStore` side-car.
- **LangChain RAG — full interface** — `TurboQuantRetriever` now implements: `get_relevant_documents()` (legacy BaseRetriever), `invoke()` (LCEL Runnable), `similarity_search_with_score()`, `filter=` kwarg on `similarity_search()`, `from_texts()` classmethod (accepts callable or pre-computed vectors), `delete(ids)`, `as_retriever()`, and `add_documents(List[Document])`.
- **LangChain RAG — `Document` return type** — `similarity_search()` now returns `List[Document]` with `.page_content` and `.metadata` attributes. `Document` is imported from `langchain_core` / `langchain` when available, or defined inline as a stub.

### Fixed

- **Rerank no-op bug** — `rerank=True` with `rerank_precision=None` previously resolved to `Disabled` (dequantization-only). For the IP metric, dequantized scores are mathematically identical to the LUT scores, so rerank had zero effect. Now defaults to `INT8` exact re-scoring, giving +5–25 pp R@1 depending on dataset and bits.
- **Server: `scoped_collection_dir` wrong path** — the server was prepending `tenants/.../databases/.../collections/` to collection paths; actual storage is flat under `{root}/{tenant}/{database}/{collection}`. Fixed to use the correct path, resolving 404/500 errors in multi-tenant collection operations.
- **Server: L2 score sign at API boundary** — L2 distances were returned as positive values; now negated at the response boundary so lower-is-better semantics are preserved in the JSON response.
- **ChromaDB compat — BUG-C7** — `collection.get(ids=[])` now returns empty (explicit empty list = no results). Previously, an empty list was falsy and triggered a full-table scan.
- **ChromaDB compat — BUG-C8** — `collection.modify(name=...)` now physically renames the collection directory, making the new name visible to `list_collections()`. Previously only updated `self._name` in memory.
- **`release.yml` update-docs job** — replaced branch+PR dance with a direct `git push origin HEAD:main`. GitHub Actions cannot create pull requests in this repository, causing the previous job to fail on every release.
- **Docs: stale defaults** — CHANGELOG v0.5.0 incorrectly stated `fast_mode=False` as the default; v0.5.1 incorrectly stated `rerank_precision` defaults to `"f16"`. Both corrected to match the actual code defaults (`fast_mode=True`, `rerank_precision="int8"`). `src/python/mod.rs` docstring updated to reflect `int8` (was `f32`).
- **README: benchmark section** — "Default config" label replaced with "Benchmark config" and `fast_mode` corrected to `True`, matching `docs/BENCHMARKS.md` and the actual bench runner.

### Documentation

- **README: Config Advisor** — new section with badge linking to the interactive Config Advisor.
- **README: benchmark tables** — added bit-sweep table ("Rerank unlocks recall at any bit depth") and dimension-scaling table (R@1 ≥ 0.87 across d=65–3072).
- **README: Recommended Setup** — updated disk estimates to reflect INT8 rerank storage (~30/116/231 MB for GloVe-200/arXiv-768/DBpedia-1536).

---

## [0.5.0] — 2026-04-10

### Added

- **`quantizer_type="dense"` is now the default** — the Haar-uniform QR + dense Gaussian quantizer (paper-faithful) replaced `"srht"` as the default. `"srht"` remains available for streaming/high-d ingest workloads. `"exact"` is accepted as a backward-compatible alias for `"dense"`.
- **`fast_mode=True` is the default** — MSE-only quantization (fastest ingest, minimum disk). Pass `fast_mode=False` to enable QJL residuals for +5–15 pp R@1 at d ≥ 1536; at d < 512 the QJL projections are too noisy and reduce recall below the MSE-only baseline, so `fast_mode=True` is recommended for low-d workloads regardless.
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
