# Repository Guidelines

## Project Structure & Module Organization
This repository implements TurboQuantDB in Rust with Python bindings.

- `src/storage/`: core database engine, WAL, segment management, graph index, metadata, and live code slab handling.
- `src/quantizer/`: quantization logic (MSE, QJL, product quantizer) and scoring paths.
- `src/python/mod.rs`: PyO3 bindings exposed to Python (`Database`, batch insert/search/index APIs).
- `python/turboquantdb/`: Python package entrypoint/wrapper.
- `benchmarks/`: comparison and performance scripts (TurboQuantDB vs LanceDB/ChromaDB).
- `target_py/wheels_fresh/`: local wheel build output.

Keep implementation changes in Rust modules and expose only necessary API updates in the Python layer.

## Build, Test, and Development Commands
- `cargo check -q`: fast compile/type validation.
- `cargo test -q --lib`: run Rust library unit tests.
- `maturin build --release --locked --out .\\target_py\\wheels_fresh`: build Python wheel.
- `python -m pip install --force-reinstall .\\target_py\\wheels_fresh\\turboquantdb-*.whl`: reinstall latest local wheel.
- `python benchmarks/tqdb_comparison_bench_local.py`: run 3-way benchmark.

Use PowerShell examples above on Windows. Run checks before committing.

## Coding Style & Naming Conventions
- Rust: follow `rustfmt` defaults (4-space indentation), idiomatic ownership/borrowing, and explicit error propagation via `Result`.
- Naming: `snake_case` for functions/variables, `CamelCase` for structs/enums, descriptive module names by subsystem.
- Prefer small, focused functions in hot paths (ingest/search/index build) and avoid unnecessary allocations.
- Python binding signatures should remain stable unless intentionally versioned.

## Testing Guidelines
- Primary framework: Rust unit tests via `cargo test`.
- Add/adjust tests when modifying quantization, indexing, WAL/segment persistence, or Windows-specific file/mmap behavior.
- For performance changes, run benchmark smoke checks at multiple scales (10k/25k/50k/100k) and compare deltas.

## Commit & Pull Request Guidelines
- Commit style: `type(scope): concise summary` (e.g., `fix(windows): release mmap before overwrite`).
- Keep commits focused (one behavior change per commit when possible).
- PRs should include:
  - What changed and why
  - Before/after benchmark highlights (ingest, ready time, disk, RAM, latency)
  - Any platform notes (especially Windows mmap/rename semantics)
  - Validation commands executed
