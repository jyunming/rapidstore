# Changelog

All notable changes to TurboQuantDB are documented in this file.

Format: `[version] — type(scope): summary`. Commits use [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

### Added
- `rerank_precision` parameter to `Database.open()` — opt-in raw-vector reranking (`"f16"` / `"f32"`, default `None` = dequantization)
- f16 HNSW build scorer — HNSW construction scorer now reads f16 raw vectors when `rerank_precision="f16"`
- `half` crate dependency for f16 encoding/decoding

### Fixed
- HNSW build scorer hardcoded f32 reads — now branches on manifest `rerank_precision`

---

## [0.1.0] — 2025

### Performance

- **perf(hnsw)**: skip QJL scoring loop when `gamma=0` in fast_mode (`4a8b9ee`)
- **perf(search)**: reuse index buffer in search closures + thread_local SRHT temp (`918a137`)
- **perf(search)**: eliminate per-scoring-call allocations, fix EF_UPPER cap (`eddb1fb`)
- **perf(hnsw)**: AVX2+FMA SIMD for `score_ip_encoded_lite` construction scorer (`3f7d993`)
- **perf(hnsw)**: pre-cache encoded vecs, remove `cand_pool` floor, tunable n_refinements (`f95bcb6`)
- **perf(ingest)**: f32 hot path, AVX2 FWHT, WAL V3 packing, zero-copy batch quantize (`60f6d41`)
- **perf(quantizer, engine)**: parallel quantize/dequantize, fast_mode QJL skip, centroid-lookup HNSW build (`fb96ffb`)
- **perf(search)**: parallel brute-force scan via rayon `par_chunks` (`2329ebc`)
- **perf(quantizer)**: SGEMM acceleration + O(d log d) SRHT fast-path (`38a4804`)
- **perf(quantizer)**: faster bit-unpack for packed MSE codes (`714f8a0`)

### Fixes

- **fix(storage)**: dequantization-based reranking — no `live_vectors.bin` overhead (`4325447`)
- **fix(storage)**: change rerank default to `false`, eliminating `live_vectors.bin` overhead (`14e9e27`)
- **fix(hnsw)**: correct HNSW construction, `ef_construction` param, fair LanceDB benchmark (`f051400`)

### Features

- **feat(quantizer)**: switch to paper-conformant QR rotation and Gaussian QJL projection (`62ca95b`)
- **refactor(quantizer)**: unify rotation to SRHT-only, drop QR/Gaussian dense paths (`e7ecf46`)
- **feat(storage)**: de-duplicate codes and persist id pool (`1833fe1`)
- **feat(src)**: storage, quantizer, and linalg improvements (`1439da2`)

### Infrastructure

- **ci**: add CI and PyPI release workflows (`59e98ff`)
- **ci**: set all Python versions for Linux wheel builds (`6cb18a3`)
- **chore(version)**: reset to 0.1.0 and establish single source of truth in `pyproject.toml` (`258ff59`)
- **chore**: consolidate repo — remove stale files, fix `.gitignore`, add MCP config (`36ee808`)
- **docs**: rewrite all docs for newcomer clarity (`4aff5de`)
- **docs**: add `CLAUDE.md` with build commands and architecture overview (`6b6dd05`)

---

## Notes

- This project follows [Semantic Versioning](https://semver.org/) with `MAJOR.MINOR.PATCH`.
- It is pre-1.0 (`0.x.y`); public API may change between minor versions.
- Version is defined in `pyproject.toml` (single source of truth). `Cargo.toml` is kept in sync manually.
- Git tags (e.g., `v0.1.0`) must match `pyproject.toml` versions.
