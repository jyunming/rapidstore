# Development Guide

This file covers building TurboQuantDB from source, running tests, and contributing.
For the user-facing API and installation via pip, see the [README](README.md).

## Prerequisites

- [Rust](https://rustup.rs/) stable toolchain (`rustup update stable`)
- Python 3.10+
- C++ compiler:
  - Windows: Visual Studio Build Tools (Desktop C++ workload)
  - macOS: `xcode-select --install`
  - Linux: `build-essential`

## Build from source

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install maturin

# Build Python extension and install into active venv
maturin develop --release

# Fast type-check without codegen
cargo check -q

# Format code
cargo fmt

# Build distributable wheel
maturin build --release --locked
```

## Testing

```bash
# All Rust unit tests
cargo test -q --lib

# Integration test suites
cargo test --test integration_tests
cargo test --test bench_search
cargo test --test bench_batch_crud

# Python quality gate (requires maturin develop first)
python benchmarks/ci_quality_gate.py   # min recall 0.60, max latency 100ms
```

## Benchmarks

```bash
# Public benchmark — updates README tables + perf history
maturin develop --release
TQDB_TRACK=1 python benchmarks/paper_recall_bench.py --update-readme --track

# Private competitive benchmark (results are gitignored)
python benchmarks/run_bench_private.py

# Quantizer comparison matrix (srht vs dense × rerank × ann × b=2/4)
python benchmarks/compare_quantizers_paper.py
```

## Sprint / PR workflow

One PR per sprint. All issues for a sprint go onto a single branch (e.g. `feat/v0.5-sprint`)
and are merged via one PR. See `docs/sprint_process.md` for the full workflow.

## Git identity

```bash
git config user.email "69726231+jyunming@users.noreply.github.com"
git config user.name "jyunming"
```

Use `GITHUB_TOKEN="" git push origin <branch>` — plain `git push` returns 403 on this machine.
