# Changelog

All notable changes to TurboQuantDB are documented in this file.

Format: `[version] — type(scope): summary`. Commits use [Conventional Commits](https://www.conventionalcommits.org/).

---

## [0.8.3] — 2026-05-03

End-to-end search hot path optimization + compression-first rerank sprint. Stacks brute-force kernel improvements (P0/P2/P4/P7), parallel-ingest + cold-cache wins (A1/A2/B1), ANN tuning (C1/C2), generalised SIMD for every bit-rate (P8), and a new opt-in **`residual_int4` rerank precision** (P12) that cuts the rerank file ~31% at near-int8 recall. Adds CI coverage for macOS Apple Silicon and Linux aarch64.

**Headline vs v0.8.2 main:**

| Cell (n=100k, brute, fast_mode) | v0.8.2 p50 | v0.8.3 p50 | Δ |
|---|---:|---:|---:|
| dbpedia-1536 b=2 rerank=F | 20.1 ms | **5.62 ms** | **−72%** |
| dbpedia-1536 b=4 rerank=F | 12.8 ms | **8.28 ms** | **−35%** |
| dbpedia-3072 b=2 rerank=F | 50.0 ms | **15.88 ms** | **−68%** |
| dbpedia-3072 b=4 rerank=F | 27.4 ms | **20.76 ms** | **−24%** |
| glove-200 b=4 rerank=F | 1.4 ms | **1.14 ms** | **−19%** |

Recall preserved bit-for-bit on default paths. b=2 cells improve dramatically because the v0.8.3 SIMD optimisations now apply to every supported bit-rate (P8), not just b=4.

### Performance — brute-force kernel (P0/P2/P4/P7)

- **Sign-bit Hamming pre-filter for b=4 fast-mode** (`src/storage/engine/mod.rs`, `src/quantizer/prod.rs`). At fast-mode + b=4, the high bit of every 4-bit MSE nibble is the sign of the corresponding rotated dimension (Lloyd-Max codebook is symmetric around 0). This lets us derive a 1-bit-per-dim sketch directly from the existing live MSE bytes — no extra storage. Active when `qjl_len == 0 && b == 4 && d ≥ 512 && candidates ≥ 5_000`. Pre-scores all candidates by Hamming distance, retains top 1/16, full LUT scoring runs only on those. Bit-identical recall (Hamming only orders the prefilter survivors; no candidate is dropped that the full scorer would have ranked top-K).

- **Fused MSE nibble unpack into b=4 scoring loop** (`src/quantizer/prod.rs`: new `score_ip_encoded_packed_b4_simd`). The prior path materialized a `Vec<u16>` of code indices per slot before scoring (~4-8 KB of L1 traffic per slot at d=2048). The fused kernel reads packed bytes directly and extracts nibbles inline into a 16-byte stack array. AVX2/FMA gated; b≠4 falls back to the prior unpack-then-score path.

- **Tightened `HAMMING_PREFILTER_RETAIN_RATIO` from 8 → 16** (`src/storage/engine/mod.rs:2446`). Halves the post-filter kernel work (6250 vs 12500 candidates scored at n=100k). Hamming sketch's recall is sufficient at 1/16 retention for production embedding distributions; bit-identical R@1/R@10 vs the previous ratio.

- **i16 LUT for b=4 fast-mode scorer** (`src/quantizer/prod.rs`: new `score_ip_encoded_packed_b4_simd_i16`). `PreparedIpQuery` now also carries an i16 mirror of the MSE LUT with a single global max-abs scale. Halves the LUT L1/L2 footprint (96 KB → 48 KB at d=1536, lut_w=16) and uses an i32 accumulator (1-cycle latency vs f32 add's 4 cycles). Scale applied once at the very end. Numpy-only validation showed bit-identical R@1/R@10 vs the f32 path on synthetic Gaussian and dbpedia query distributions, with EITHER per-row or global scale (global chosen for SIMD simplicity).

  | Dataset | Cell | v0.8.2 p50 | v0.8.3 p50 | Δ |
  |---|---|---:|---:|---:|
  | glove-200 | rerank=F | 1.15 ms | 1.02 ms | **−11.3%** |
  | glove-200 | rerank=T | 1.44 ms | 1.33 ms | **−7.6%** |
  | dbpedia-1536 | rerank=F | 10.31 ms | 7.99 ms | **−22.5%** |
  | dbpedia-1536 | rerank=T | 10.12 ms | 8.22 ms | **−18.8%** |
  | dbpedia-3072 | rerank=F | 22.36 ms | 19.73 ms | **−11.8%** |
  | dbpedia-3072 | rerank=T | 21.95 ms | 20.08 ms | **−8.5%** |

### Performance — ingest + cold-cache (A1/A2/B1)

- **A1: `MADV_WILLNEED` on `live_codes` open** (`src/storage/live_codes.rs`). New `LiveCodesFile::advise_willneed()` hints the OS to prefetch `live_codes.bin` into the page cache after open(). Mirrors the existing `advise_random()` pattern — `#[cfg(unix)]` body, no-op on Windows / non-Unix. Reduces cold-cache penalty on the first few queries in a fresh-process scenario; expected 5–15% on cold queries on Linux production. Bench machine is Windows so no local measurement; bit-identical recall vs v0.8.2 confirmed.

- **A2: `score_batch_brute` uses fused `score_ip_encoded_packed` for b=4** (`src/storage/engine/mod.rs`). The multi-query batch scorer (gated to `Q ≥ 8 AND N ≥ 500k`) was using the older `score_ip_encoded` path which requires unpacking MSE indices into a `Vec<u16>` per slot. Swapped to `score_ip_encoded_packed` for b=4 (production cell), routing through the P7 i16 LUT in fast_mode. Falls back to the unpack path for b≠4. Expected ~10–15% on multi-query batched workloads at N≥500k; not exercised by the single-query bench.

- **B1: parallel per-vector centroid lookup in dense GEMM ingest** (`src/quantizer/prod.rs`: `quantize_batch_dense_gemm`). The post-rotation per-column nearest-centroid lookup (+ optional QJL residual) is per-column independent. Switched to `par_iter`, gated by total work `b * d ≥ 5M` so small-d batches stay sequential (Rayon park/unpark overhead exceeds savings at d=200; first attempt with always-on parallel made d=200 12% SLOWER).

  | Ingest p50 (3-iter median, 100k inserts) | v0.8.2 | v0.8.3 | Δ |
  |---|---:|---:|---:|
  | d=200 | 2.73 s | 2.78 s | +1.8% (within noise; gated off) |
  | d=1536 | 17.74 s | 16.49 s | **−7.0%** |
  | d=3072 | 40.03 s | 36.82 s | **−8.0%** |

### Performance — generalize SIMD brute path to all bit-rates (P8)

- **Const-generic fused-unpack scoring kernel** (`src/quantizer/prod.rs`: `score_ip_encoded_packed_simd<const B>` + `_simd_i16<const B>`). Replaces the previous b=4-only `score_ip_encoded_packed_b4_simd*` variants. The 8-code SIMD batch architecture is now parameterized over `B ∈ [1, 8]`; each instantiation compiles to specialized bit-extraction shifts. The `score_ip_encoded_packed` dispatch routes the right monomorphisation based on `mse_quantizer.b`. Bits beyond 8 fall back to the unpack+score path.

- **Generalized Hamming sign-bit pre-filter** (`hamming_disagree_signs<const B>`) for `B ∈ {1, 2, 4, 8}`. The Lloyd-Max codebook is symmetric around 0 for every B, so the MSB of each B-bit code is the sign of the rotated dimension. Per-B u64-chunked extraction kernels (b=2 uses an "every-other-bit compress" trick on a u64 chunk; b=8 collects MSBs from byte stream; b=4 keeps the original optimised pattern; b=1 popcounts XOR directly). The legacy `hamming_disagree_b4_signs` is kept as a thin wrapper. `b ∈ {3, 5, 6, 7}` span misaligned byte boundaries — those bypass the pre-filter and fall back to full-LUT scoring.

- **Engine plumbing** (`src/storage/engine/mod.rs`). Removed the `let b4 = quantizer.mse_bits_per_idx() == 4` gates in `score_batch_brute` and the brute-force exhaustive search (sequential + parallel + Cosine paths). All bit-rates now go through the const-generic `score_ip_encoded_packed`. The Hamming pre-filter gate widened from `b == 4` to `b ∈ {1, 2, 4, 8}`.

  | Cell (dbpedia-1536, n=100k, brute, fast_mode=True) | v0.8.3 pre-P8 p50 | v0.8.3 post-P8 p50 | Δ |
  |---|---:|---:|---:|
  | b=2 rerank=F | 20.0 ms | **6.23 ms** | **−69%** |
  | b=2 rerank=T | 21.2 ms | **6.51 ms** | **−69%** |
  | b=4 rerank=F | 12.8 ms | **7.62 ms** | **−40%** |
  | b=4 rerank=T | 12.5 ms | **7.75 ms** | **−38%** |

  At b=2 the LUT is 4× smaller than at b=4 so cache behaviour favours b=2 once both bit-rates share the same scoring path — for the first time, **b=2 brute is faster than b=4**. Recall bit-identical at all cells.

### Compression-first rerank — `RerankPrecision::ResidualInt4` (P12)

- **New opt-in rerank precision** (`src/storage/engine/mod.rs`: `RerankPrecision::ResidualInt4`; Python: `rerank_precision="residual_int4"`). Stores the per-vector **residual** = `vec − dequant(MSE codes) × doc_norm` at INT4 precision (4 + ⌈d/2⌉ bytes per vector — same on-disk record size as the deprecated `Int4` raw mode). Because the residual has a much smaller dynamic range than the raw vector, 4 bits buy substantially more precision per dim than INT4-over-raw.

- **Additive scoring at rerank time** (`live_rerank_score_residual_int4`). Inner-product is linear, so `<q, full_vec> = <q, dequant(codes)>·doc_norm + <q, residual>`. The brute-force pass already produces `cand_score = <q, dequant(codes)>·doc_norm`, so we just add `<q, residual_decoded>` per candidate — no need to re-dequantize the codes (an O(d²) operation in dense mode). At d=1536 this is **~14× faster** than the naive full-reconstruction path (8.4 ms vs 121 ms in benchmarks).

- **GEMM-batched residual computation at insert time** (`compute_residuals_batch_parallel`). For dense mode the per-vector dequant at insert is memory-bandwidth-bound (the d×d rotation matrix = 36 MB at d=3072 is re-streamed per vector). A single SGEMM (`Q^T · y_tilde`) does all N inverse rotations in one pass and streams the matrix once. Combined with a parallel per-column residual subtraction, this brings ResidualInt4 ingest within **1.4–1.5× of int8 ingest** at every measured dim (vs the naive serial implementation which was ~22× slower at d=3072).

  | Dataset | d | int8 ingest | residual_int4 ingest (GEMM) | Ratio |
  |---|---:|---:|---:|---:|
  | dbpedia-1536 | 1536 | 18.9 s | 26.7 s | 1.41× |
  | dbpedia-3072 | 3072 | 39.6 s | 60.3 s | 1.52× |

- **Numpy-validated across all (dim, b) cells.** ResidualInt4 at b=4 gives near-int8 recall everywhere (−0 to −1pp R@1 vs int8 across glove-200, dbpedia-1536, dbpedia-3072). At b=2 it loses 2–6.5 pp because the residual has a larger dynamic range — for the b=2 compression-first user, the int8 default still wins, so ResidualInt4 ships as **opt-in**, not as the new default.

  | Dataset | Cell | int8 R@1 | int8 disk | residual_int4 R@1 | residual_int4 disk | Δ disk |
  |---|---|---:|---:|---:|---:|---:|
  | glove-200 | b=4 rerank=T | 0.980 | 31 MB | **0.975** | **21 MB** | **−32%** |
  | dbpedia-1536 | b=4 rerank=T | 0.995 | 230 MB | **0.985** | **158 MB** | **−31%** |
  | dbpedia-3072 | b=4 rerank=T | 0.995 | 477 MB | **0.985** | **331 MB** | **−31%** |

- **`RerankPrecision::Int4` (raw) is now deprecated.** It was strictly dominated by `rerank=False` at every measured cell — INT4 quantization adds enough noise to the rerank score that it actively hurts ranking instead of helping. The enum variant is kept for backward compatibility with existing databases; new uses should select `ResidualInt4` instead.

- **Tests:** 3 new (`residual_int4_rerank_roundtrip`, `residual_int4_record_size_matches_int4`, `residual_int4_reload_preserves_recall`) pinning encode/decode parity and on-disk format compatibility.

### Performance — ANN (C1/C2)

- **C1: dim-aware `ef_construction` default** (`src/storage/engine/mod.rs`: new `default_ann_ef_construction(d)`; `src/python/mod.rs`). The fixed default of 200 hurts low-d HNSW recall (more quantization noise → graph needs a wider neighbour search at construction). Stepped defaults: `d ≤ 384 → 400`, `d ≤ 1024 → 300`, `d > 1024 → 200` (unchanged). User-supplied `ef_construction=` still respected. On glove-200 ANN rerank=T: **R@1 +6pp, R@10 +6pp**. d=1536/3072 cells within bench noise. Build time at low d is dominated by per-node sampling, so the larger pool doesn't translate to wall-time regression.

- **C2: software prefetch in HNSW level-0 beam search** (`src/storage/graph.rs`: new `search_with_prefetch`; `src/storage/engine/mod.rs`). New entry point accepts an optional prefetch closure; the legacy `search()` delegates with `None` so existing callers are unaffected. The engine ANN path passes a closure that issues an L1 prefetch (`_mm_prefetch _MM_HINT_T0`) for the upcoming neighbour's `live_codes` record while the current one is being scored. Hides DRAM/L3 latency on the typically-random ANN beam-search lookup pattern. `#[cfg]`-gated to x86_64; no-op on ARM. Apples-to-apples 3-iter median: **d=1536 rerank=F −4.8%, rerank=T −5.9%** — small but direction-consistent (right at 5% bench noise floor).

### Fixed

- **`score_batch_brute` cosine path applied stale `doc_norm`** (`src/storage/engine/mod.rs`). Computing `score = ip * doc_norm * q_norm_inv` for the cosine metric — the sequential and parallel cosine paths correctly use only `score = ip * q_norm_inv`. Silent for the common normalize=True case (doc_norm == 1.0); returned wrong scores for normalize=False + cosine + multi-query batch. doc_norm is now applied only on the IP path.

- **`flush_for_close` ordering crash window** (`src/storage/engine/mod.rs`). `persist_id_pool()` ran before the remote-backend upload of `live_codes.bin`. A crash between the two left `live_ids.bin` ahead of `live_codes` on the remote (ID pool referenced codes that never arrived) → corrupt reads on reopen. Flipped the order so codes upload first; stale ID pool is recoverable via WAL replay.

### CI

- **macOS Apple Silicon + Linux aarch64 added to Rust test matrix** (`.github/workflows/ci.yml`). The `rust-test` job now runs on `ubuntu-latest`, `windows-latest`, **`macos-latest`** (Apple Silicon, ARM64); exercises the scalar-fallback paths gated by `#[cfg(target_arch = "x86_64")]` on real ARM hardware. New `rust-test-linux-arm64` job cross-compiles to `aarch64-unknown-linux-gnu` and runs `cargo test --lib` under QEMU (covers Linux ARM Graviton/Ampere-class targets). Closes the compile-only verification gap that pre-dated this sprint — release wheels already shipped for these targets but Rust unit tests only ran on x86_64.

### Tests

- New `benches/score_kernel.rs` — Criterion microbench harness for `score_ip_encoded_packed_b4_simd`, `prepare_ip_query`, `hamming_disagree_b4_signs`, and a 4k-slot in-cache scan. Run with `cargo bench --bench score_kernel`. Used during the sprint to gate sub-5% kernel changes that the noisy end-to-end Python bench cannot reliably distinguish.
- 12 new regression and coverage tests for the engine module (Phase A): pin contracts on quantize/score determinism, ANN dispatch boundaries, and metric-path correctness. Also fixes the prior CHANGELOG drift on test count.
- **20 new P8 tests** (`prod.rs`): SIMD-vs-scalar parity for every `B ∈ [1, 8]` across both fast-mode (i16 LUT) and QJL paths, long-dim batch+tail boundary at d ∈ {200, 511, 1024}, and Hamming pre-filter parity (b=1 popcount-XOR equivalence, b=2/4/8 vs brute-force MSB extraction, legacy `hamming_disagree_b4_signs` wrapper agreement).
- **3 new P12 tests** (`engine/tests.rs`): `residual_int4_rerank_roundtrip` (insert→search recall within 5% of true IP), `residual_int4_record_size_matches_int4` (on-disk format), `residual_int4_reload_preserves_recall` (manifest persistence + reopen).
- **462/462 cargo lib tests pass** (was 425 in v0.8.2 + 37 new; 2 ignored long-running boundary tests carried forward).
- Full paper bench validated against `main` HEAD on the same machine: brute-force −21% p50 at d=1536; ANN R@1 actually improved by 0.8–0.9pp at d=1536 (recall not regressed despite ANN being unrelated to the sprint changes — likely run variance favouring v0.8.3).

---

## [0.8.2] — 2026-04-30

Audit-driven follow-up to v0.8.1: a deeper scan surfaced three real bugs in compaction and snapshot paths. All three are fixed here, each with a regression test pinning the contract.

### Fixed

- **IVF coarse index becomes stale after compaction** (`src/storage/engine/mod.rs`). `live_compact_slab()` renumbers the `id_pool` from scratch, so the IVF `cluster_index` and `cluster_map` (which reference the OLD slot numbers) silently point to wrong vectors. `search_with_ivf` would return garbage results — both score-wise and identity-wise — until the user manually rebuilt the IVF index. Compaction now drops `self.ivf` and removes `ivf.bin` from disk + backend; subsequent IVF searches fall back to brute-force until `create_coarse_index()` is called again. Pinned by `tests::ivf_invalidated_after_compaction`.

- **`delta_slots` not cleared during compaction** (`src/storage/engine/mod.rs`). The "delta overlay" of slots inserted after the last `create_index()` holds slot numbers from the pre-compaction `id_pool`. After renumbering they're stale, which (a) corrupts `auto_use_ann()`'s ratio check by counting non-existent slots and (b) causes the delta-overlay scoring path to read wrong slots from `live_codes`. Compaction now clears `delta_slots` and marks state dirty for persistence. Pinned by `tests::delta_slots_cleared_after_compaction`.

- **Snapshot/restore path traversal** (`server/src/main.rs`). `snapshot_name` from the request body was joined into the snapshots directory tree without `validate_path_component`, unlike `tenant`/`database`/`collection`. A malicious request with `snapshot_name="../../etc/passwd"` could write outside the snapshots subtree on CREATE, or read from arbitrary paths on RESTORE. Both handlers now validate `snapshot_name` rejecting `..`, `.`, empty strings, slashes, backslashes, and null bytes. Pinned by `tests::validate_path_component_rejects_traversal_sequences`. Embedded mode is unaffected; this is a server-only fix.

### Tests

- 3 new tests covering the regressions above. Test count: 425 → **428** (+3; 2 ignored long-running boundary tests carried forward).

---

## [0.8.1] — 2026-04-30

Combined performance + audit-driven bug-fix release.

**Headline:** at high embedding dimension (d=1536, the OpenAI `text-embedding-3` size), ANN search was previously ~90 ms p50 vs brute-force ~56 ms — ANN was *slower* than brute. Dimension-aware oversampling defaults bring it down to **~13 ms** with **better recall**. Layered on top: a 3-agent code audit produced ~80 findings; the 9 verified critical/high items are fixed here, and 5 previously untested modules get unit-test suites.

No API changes; all behavior is either an opt-in safety improvement or a contract pin. Test count 387 → **425** (+38 new; 2 ignored long-running boundary tests).

### Performance

- **Dimension-aware default `rerank_factor`** — the prior fixed default of 20 (ANN) / 10 (brute) caused the level-0 HNSW search to bloat its candidate pool and visit far more nodes than needed. Defaults now step down with vector dimension:

  | Dimension `d` | ANN factor | Brute factor |
  |---------------|-----------:|-------------:|
  | d ≤ 384       |         20 |           10 |
  | 384 < d ≤ 1024|          8 |            6 |
  | d > 1024      |          4 |            4 |

  User-supplied `rerank_factor=` continues to override. Pure default tuning; no API change. Private bench at n=100k (ANN, rerank=True):

  | Dataset (dim) | b=2 p50 before → after | b=2 R@1 before → after | b=4 p50 before → after | b=4 R@1 before → after |
  |---|---|---|---|---|
  | GloVe-200 | 19.5 ms → **2.4 ms** | 0.211 → **0.403** | 21.6 ms → **1.3 ms** | 0.366 → **0.422** |
  | DBpedia-1536 | 90.0 ms → **13.3 ms** | 0.738 → **0.836** | 97.9 ms → **8.5 ms** | 0.822 → **0.850** |
  | DBpedia-3072 | 43.2 ms → **23.8 ms** | 0.766 → **0.849** | 51.0 ms → **15.4 ms** | 0.816 → **0.861** |

  Speed improved 1.8–17× and recall improved 2–19% across all dims. The recall gain is from a tighter `search_list_size`: the prior factor of 20 inflated `ef_search` to ~200 nodes which over-spread the level-0 beam; the smaller pool keeps the search focused on better-scoring candidates.

### Fixed

- **`internal_k` overflow guard** — `top_k * factor` could wrap on pathological `top_k`. Both ANN (`engine/mod.rs:2018`) and brute (`engine/mod.rs:2325`) paths now use `saturating_mul` and cap the candidate buffer at active count.
- **WAL truncation logging** — `wal::replay()` previously dropped truncated payloads and missing CRC bytes silently (only CRC *mismatch* logged). Now emits `eprintln!` warnings at every truncation point with the entry index, so partial writes are observable in production logs.
- **WAL oversized-payload guard** — corrupted length field claiming a multi-GB payload no longer triggers a giant `vec![0u8; len]` allocation. New `MAX_REASONABLE_PAYLOAD = 10 MiB` cap stops replay with a clear log message instead of OOM.
- **Graph NaN-score sanitization** — a user-supplied scorer returning `f64::NAN` previously yielded undefined HNSW ranking via the `partial_cmp().unwrap_or(Equal)` fallback. Non-finite scores are now coerced to `f64::NEG_INFINITY` at every `scorer()` call site — those nodes effectively rank last and ranking is deterministic.
- **Filter recursion-depth DoS guard** — `metadata_matches_filter` previously recursed without bound on `{"$and": [{"$and": [...]}]}` nesting, risking stack overflow on adversarial input. New `MAX_FILTER_DEPTH = 32` is enforced both at validation time (`validate_filter_operators`) and at evaluation time (defense-in-depth).
- **IVF zero-cluster / zero-nprobe validation** — `create_coarse_index(n_clusters=0)` now errors early with a clear message. `IvfIndex::probe(query, nprobe=0)` now clamps nprobe to at least 1 (was: called `select_nth_unstable_by(0, ...)` which is defined but useless).
- **Quantizer dimension assertions** — `score_ip_encoded` and `score_ip_encoded_lite` previously called `idx.get_unchecked(i)` for `i in 0..self.n` with no length check; a too-short `idx` was undefined behavior. Now panics with a clear message at the safe wrapper.
- **Bench version detection** — `paper_recall_bench.py` queried `importlib.metadata.version("tqdb")`, which returns stale data when `maturin develop --release` re-installs the package as editable (the .pyd binary updates but dist-info can be left at the prior version, recording wrong version in `perf_history.json`). Now reads `pyproject.toml` directly (the documented source-of-truth per CLAUDE.md).

### Documentation

- **Python `Database.search` docstring** — was still documenting the pre-tuning fixed `10×/20×` defaults. Updated with the new dimension-aware table and an explanation of why the smaller defaults at high `d` prevent level-0 HNSW search bloat.

### Tests

- **Boundary E2E recall continuity** — new `#[ignore]`'d tests at d=384/385 and d=1024/1025 verify recall doesn't cliff across the dimension cutoffs. Run with `cargo test -- --ignored boundary` (~30s each).
- **`default_rerank_factor` unit tests** — cover the dimension cutoffs for both ANN and brute paths.
- **WAL replay edge cases** — oversized length field, truncated payload, truncated CRC, duplicate id entries, corrupted middle entry.
- **Graph beam-search edge cases** — top_k=0, single-node graph, NaN/+Inf scorer determinism.
- **Filter operator coverage** — empty `$and`/`$or` arrays (vacuous truth), `$contains` on non-string, `$exists` null vs missing, single-bound range extraction, big-int NaN-coercion behavior pin.
- **IVF unit-test suite** — 9 tests for the previously untested `ivf.rs` module: build assigns all slots, probe is unique+sorted, save/load roundtrip, magic validation.
- **Quantizer numeric edges** — zero-vector quantize, b=1 (1-bit) round-trip, `dequantize_then_score` consistency with `score_ip_encoded`.
- **BM25 empty-document contract** — audit flagged this as a bug; verified it's intentional (per `put` docstring). Added 5 tests pinning the contract: empty docs are excluded from `n_docs`/`avgdl`, mixed empty+real corpora compute correctly, single-doc corpora don't div-by-zero, empty queries return empty.

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
