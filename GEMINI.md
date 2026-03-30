# TurboQuantDB - Gemini Instructional Context

## Project Overview

TurboQuantDB is a high-performance, embedded vector database engine written in Rust, featuring native Python bindings via PyO3 and maturin. It leverages the **TurboQuant** algorithm (based on arXiv:2504.19874) to provide near-optimal, data-oblivious vector quantization for extremely fast and memory-efficient Retrieval-Augmented Generation (RAG) applications.

Key characteristics:
- **Zero Training Time:** No index training required; vectors are quantized and inserted immediately.
- **Deep Compression:** Compresses float32 embeddings down to 2 or 4 bits per coordinate using Quantized Johnson-Lindenstrauss (QJL) transforms.
- **Embedded & Service Modes:** Can run as an in-process embedded database (similar to LanceDB) in Python, or as a multi-tenant Axum-based HTTP service via the `server/` module.

## Core Technologies

- **Rust:** Core engine, math (`nalgebra`, `ndarray`), storage (`redb`, `memmap2`), and HTTP API (`server/src/main.rs` using Axum).
- **Python:** Client interface, LangChain-style RAG retrievers, built and packaged using `maturin` and `PyO3`.

## Building and Running

### Development Commands

- **Fast Compile Validation:** `cargo check -q` (validates the Rust workspace).
- **Run Rust Tests:** `cargo test -q`
- **Run Service Tests:** `cd server && cargo test -q`
- **Format Code:** `cargo fmt --all`
- **Build Python Extension:** `maturin develop --release` (requires an active virtual environment).
- **Run Python Tests:** `pytest -q` (when Python-side tests are present/enabled).

## Development Conventions

### Coding Style & Organization
- **Rust Styling:** Strictly follow `rustfmt` defaults (4-space indentation, `snake_case` for functions, `PascalCase` for types).
- **Separation of Concerns:** Keep modules focused. Storage logic belongs in `src/storage/*`, while API-specific concerns go into `server/src/main.rs`.
- **Error Handling:** Prefer explicit error messages with relevant context (e.g., specifying `tenant`, `database`, `collection`) when handling service paths.
- **Refactoring:** Avoid broad refactors during feature work; ship scoped, test-backed changes.

### Testing Guidelines
- **Mandatory Coverage:** Add or adjust tests alongside every behavior change.
- **Service APIs:** Include endpoint-level tests within the `server/src/main.rs` test module.
- **Engine Behavior:** Cover engine-level changes with integration tests in `tests/integration_tests.rs` or other dedicated test files.
- **Naming:** Test names must describe the expected behavior explicitly (e.g., `delete_endpoint_supports_filter_only`).

### Contribution & Commits
- **Conventional Commits:** Use conventional, scoped commit subjects, such as `feat(server): ...` or `test(server): ...`.
- **Granularity:** Keep one logical change per commit.
- **Pull Requests:** PR descriptions should explicitly include what changed and why, affected modules, validation commands executed, and any subsequent risks.

### Security & Configuration
- **Environment Variables:** Use env vars for service runtime configurations (e.g., `TQ_SERVER_ADDR`, `TQ_LOCAL_ROOT`, `TQ_JOB_WORKERS`, and paths for auth/quota/job stores).
- **Secrets:** Never commit secrets, API keys, or locally generated data (such as files under `server/tenants/` or temp directories).