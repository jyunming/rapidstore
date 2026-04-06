# Contributing to TurboQuantDB

Thank you for your interest in contributing! This document covers how to get set up, the development workflow, and what we expect from pull requests.

---

## Getting Started

### Prerequisites

- Rust stable (latest) — install via [rustup](https://rustup.rs)
- Python 3.10+ with a virtual environment
- [Maturin](https://maturin.rs) for building the Python extension

### Setup

```bash
git clone https://github.com/jyunming/TurboQuantDB.git
cd TurboQuantDB

# Create and activate a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install build tools and test dependencies
pip install maturin numpy pytest

# Build and install the extension in development mode
maturin develop --release
```

### Run tests

```bash
# Rust unit tests
cargo test -q --lib

# Python test suite
pytest tests/ -v
```

---

## Development Workflow

1. **Check the project board** — Sprint items live at https://github.com/users/jyunming/projects/3. Pick an open issue tagged with the current sprint milestone.

2. **Create a branch** — one branch per issue:
   ```bash
   git checkout -b feat/your-feature-name
   ```

3. **Make your changes** — follow the [commit style](#commit-style) below.

4. **Run the pre-commit checks** before pushing:
   ```bash
   cargo fmt --all
   cargo check -q
   cargo test -q --lib
   pytest tests/ -q
   ```
   The pre-commit hook runs these automatically on `git commit`.

5. **Open a pull request** targeting `main`. Reference the issue with `Closes #N` in the PR body.

---

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short summary

Optional longer description.
```

| Type | When to use |
|------|-------------|
| `feat` | New user-visible feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `chore` | Build, CI, tooling |

Examples:
```
feat(python): add delete_batch(where_filter=...) for filter-based bulk delete
fix(storage): prevent panic in S3Provider when called from async context
perf(search): sequential brute-force path for N ≤ 20k avoids Rayon overhead
```

---

## Versioning

TurboQuantDB follows [Semantic Versioning](https://semver.org/). The version in `pyproject.toml` is the single source of truth; `Cargo.toml` is kept in sync manually.

Bump rules on merge to `main`:

| Commits present | Version bump |
|-----------------|--------------|
| Any `feat:` | `MINOR` (reset PATCH to 0) |
| Only `fix:`, `perf:`, `refactor:` | `PATCH` |
| Any breaking API change | `MAJOR` |

---

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR.
- Include or update tests for all new behavior.
- Update `docs/PYTHON_API.md` and `.pyi` stubs if the Python API changes.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Run `maturin develop --release && pytest tests/ -q` before requesting review.

---

## Reporting Bugs

Open an issue using the **Bug report** template. Include:
- TurboQuantDB version (`pip show tqdb`)
- OS and Python version
- Minimal reproduction script
- Expected vs actual behavior

---

## Questions

Open a [GitHub Discussion](https://github.com/jyunming/TurboQuantDB/discussions) or comment on the relevant issue.
