# RapidStore

RapidStore is a Rust-first vector database project for Retrieval-Augmented Generation (RAG) workloads. It supports both:

- Embedded mode (library + Python bindings)
- Service mode (multi-tenant HTTP API with background jobs)

The quantization approach is inspired by TurboQuant, with practical storage/indexing components implemented in this repository.

## Current Capabilities

- Collection-scoped storage model (`tenant/database/collection` in service mode)
- Batch write APIs: add, upsert, update, delete
- Query APIs with include controls and pagination (`offset`, `limit`)
- Filter support (`filter` / `where_filter`) for query/delete/get paths
- Async maintenance jobs: compact, index build, snapshot
- Persisted auth/RBAC, quotas, and job store in server mode
- Quota admission controls for vectors, disk usage, and concurrent jobs

## Repository Layout

- `src/`: core Rust engine, quantization, storage, indexing, Python bindings
- `server/`: Axum HTTP service crate (`turboquantdb-server`)
- `python/`: Python package helpers
- `tests/`: Rust integration and benchmark-style tests
- `docs/`: design docs, roadmap, API notes
- `benchmarks/`: recall/latency scripts and CI gates

## Quickstart

### Build and test (Rust)

```bash
cargo check -q
cargo test -q
```

Server-only tests:

```bash
cd server
cargo test -q
```

### Python extension (maturin)

```bash
pip install maturin
maturin develop --release
```

### Basic Python usage

```python
import turboquantdb as tq

client = tq.Client(uri="data/tqdb", dimension=1536, bits=4, seed=42, metric="ip")
collection = client.get_or_create_collection("docs")
collection.upsert("id-1", embedding, {"source": "faq"}, "document text")
```

## Service Mode

See [server/README.md](server/README.md) for endpoints and env vars.

Common env vars:

- `TQ_SERVER_ADDR` (default `127.0.0.1:8080`)
- `TQ_LOCAL_ROOT` (default `./data`)
- `TQ_JOB_WORKERS` (default `2`)

## Documentation

- [Python Migration Guide](docs/PYTHON_MIGRATION.md)
- [M5 Multi-Tenant Service Design](docs/M5_MULTITENANT_SERVICE_DESIGN.md)
- [M5 API Spec](docs/M5_API_SPEC.md)
- [Compatibility Matrix](docs/COMPATIBILITY_MATRIX.md)
- [Roadmap Backlog](docs/ROADMAP_BACKLOG.md)

## Research Basis

This repository is an independent implementation that uses ideas described in the TurboQuant paper; the paper itself was authored by the original researchers.

Reference:

Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. arXiv:2504.19874.

- arXiv: https://arxiv.org/abs/2504.19874
- Local copy in this repo: [2504.19874v1.pdf](2504.19874v1.pdf)

If your academic work depends on the TurboQuant theory, please cite the original paper:

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```



