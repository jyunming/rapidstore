# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core Rust library (quantizer, storage engine, metadata, indexing, Python bindings bridge).
- `server/`: Axum-based service mode (`server/src/main.rs`) for multi-tenant HTTP APIs and job workers.
- `tests/`: Rust integration and benchmark-style tests for embedded engine behavior.
- `python/`: Python package surface and examples for PyO3/maturin workflows.
- `docs/`: design notes, API specs, and roadmap artifacts.
- `benchmarks/`: recall/latency scripts and CI quality-gate helpers.

## Build, Test, and Development Commands
- `cargo check -q`: fast compile validation for the Rust workspace.
- `cargo test -q`: run all Rust tests.
- `cd server && cargo test -q`: run service tests only.
- `cargo fmt --all`: format Rust code.
- `maturin develop --release`: build/install Python extension locally.
- `pytest -q`: run Python-side tests (when present/enabled).

## Coding Style & Naming Conventions
- Rust style follows `rustfmt` defaults (4-space indent, snake_case functions, PascalCase types).
- Keep modules focused: storage concerns under `src/storage/*`, API concerns in `server/src/main.rs`.
- Prefer explicit error messages with context (`tenant`, `database`, `collection`) for service paths.
- Avoid broad refactors in parity work; ship scoped, test-backed changes.

## Testing Guidelines
- Add/adjust tests with each behavior change.
- Service API changes should include endpoint-level tests in `server/src/main.rs` test module.
- Engine behavior changes should include integration coverage in `tests/integration_tests.rs` (or dedicated test files).
- Test names should describe expected behavior, e.g. `delete_endpoint_supports_filter_only`.

## Commit & Pull Request Guidelines
- Use conventional, scoped commit subjects seen in history:
  - `feat(server): ...`
  - `test(server): ...`
- Keep one logical change per commit.
- PRs should include:
  - What changed and why.
  - Affected endpoints/modules.
  - Validation commands run and results.
  - Any follow-up gaps or risks.

## Security & Configuration Tips
- Use env vars for service runtime (`TQ_SERVER_ADDR`, `TQ_LOCAL_ROOT`, `TQ_JOB_WORKERS`, auth/quota/job store paths).
- Do not commit secrets, API keys, or generated local data under `server/tenants` or temp dirs.
