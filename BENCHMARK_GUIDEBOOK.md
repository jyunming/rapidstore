# TurboQuantDB — Benchmark Guidebook

> **Disclaimer:** All benchmarks in this guide were run on a personal computer (WSL2 on Windows 11,
> Intel Core Ultra 7 255H, 16 cores, consumer NVMe SSD). Results are indicative, not production-grade.
> HPC benchmarks are planned for future work. This is a hobby project driven purely by enthusiasm —
> all discussion, feedback, and contributions are warmly welcome.

---

## 1. Test Environment

| Item | Detail |
|---|---|
| Machine | Personal laptop — Windows 11 |
| CPU | Intel Core Ultra 7 255H (22 threads total) |
| WSL2 cores configured | 16 (updated from default 4 during testing) |
| RAM | 16 GB |
| Storage | Consumer NVMe SSD |
| OS (benchmark) | Linux native ext4 (`/dev/sdd`, WSL2) |
| Python | 3.14.3 |
| Rust toolchain | 1.94.0 (stable) |
| TQDB commit | `e0431c7` — feat: f16/f32 reranking + QJL skip for fast_mode |

> **Note on WSL2 cores:** Early runs were done with only 4 cores visible to WSL2 (default config).
> After updating `.wslconfig` and running `wsl --shutdown`, all 16 cores became available.
> Results labelled "Linux Native" use 16 cores unless noted.

---

## 2. What Is TurboQuantDB

TurboQuantDB is an embedded vector database written in Rust with Python bindings (PyO3/Maturin).
It implements the TurboQuant algorithm (arXiv:2504.19874): two-stage quantization combining
QR-rotation MSE quantization and QJL (1-bit dense Gaussian projection), achieving near-optimal
vector compression with zero training time.

**Key properties:**
- Data-oblivious quantization (seed-deterministic, no training phase)
- Always memory-mapped (mmap) — no separate "in-memory mode"
- Thread-safe via `Arc<RwLock<TurboQuantEngine>>`
- Embedded — runs in-process, no server required

---

## 3. Benchmark 1 — Standard Comparison
**50,000 vectors | dim=1536 (OpenAI ada-002) | top_k=10 | Linux Native ext4 | 16 cores**

Dataset: `KShivendu/dbpedia-entities-openai-1M` — pre-computed OpenAI embeddings, no GPU needed.

### 3.1 Performance

| Engine | Ingest | vec/s | Disk | RAM-ing | RAM-qry | CPU-ing | CPU-qry | p50 | p95 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Faiss HNSW *(ceiling)* | 15.1s | 3,316 | 306MB | 698MB | 698MB | 1331% | 100% | 0.96ms | 1.27ms | 99.75% | 1.0000 |
| ChromaDB (HNSW) | 33.5s | 1,493 | 398MB | 924MB | 924MB | 258% | 92% | 2.41ms | 3.42ms | 99.75% | 1.0000 |
| TQDB b=4 FastBuild | 22.7s | 2,199 | 70MB | 507MB | 506MB | 1262% | 104% | 4.00ms | 6.88ms | 83.10% | 0.9900 |
| TQDB b=8 FastBuild | 29.8s | 1,678 | 119MB | 552MB | 552MB | 842% | 103% | 4.89ms | 7.23ms | 95.15% | 0.9800 |
| TQDB b=4 Balanced | 26.0s | 1,922 | 70MB | 507MB | 507MB | 1313% | 196% | 7.46ms | 11.24ms | 88.70% | 0.9900 |
| TQDB b=8 HQ | 38.0s | 1,316 | 119MB | 554MB | 554MB | 1175% | 179% | 8.73ms | 11.97ms | 97.85% | 0.9950 |
| LanceDB (IVF_PQ) | 89.8s | 556 | 318MB | 1722MB | 1722MB | 1100% | 152% | 8.10ms | 9.57ms | 79.90% | 1.0000 |
| Milvus Lite (HNSW) | 251.4s | 198 | 391MB | 514MB | 493MB | 2% | 7% | 12.18ms | 14.20ms | 100.00% | 1.0000 |
| DuckDB VSS (HNSW) | 407.2s | 122 | 745MB | 1781MB | 1597MB | 132% | 193% | 177.00ms | 195.46ms | 100.00% | 1.0000 |
| Qdrant (HNSW)* | 805.0s | 62 | 685MB | 1152MB | 1152MB | 7% | 1000% | 123.94ms | 163.84ms | 100.00% | 1.0000 |

*Qdrant tested in embedded mode only — not representative of its server deployment performance.*

### 3.2 Disk Compression

Raw float32 baseline: **293 MB** for 50,000 × 1536-dim vectors.

| Engine | Disk | bytes/vec | vs float32 | vs ChromaDB |
|---|---|---|---|---|
| DuckDB VSS | 745MB | 15,619 | 0.39× | 0.53× |
| Qdrant | 685MB | 14,373 | 0.43× | 0.58× |
| ChromaDB | 398MB | 8,353 | 0.74× | baseline |
| Milvus Lite | 391MB | 8,203 | 0.75× | 1.02× |
| LanceDB | 318MB | 6,663 | 0.92× | 1.25× |
| Faiss HNSW | 306MB | 6,416 | 0.96× | 1.30× |
| **TQDB b=8** | **119MB** | **2,490** | **2.47×** | **3.35×** |
| **TQDB b=4** | **70MB** | **1,466** | **4.19×** | **5.70×** |

### 3.3 Engine Notes

- **Faiss HNSW** — ANN library only, no persistence/metadata/delete. Ceiling reference.
- **ChromaDB** — Easiest to use, good recall, moderate storage.
- **LanceDB** — IVF_PQ index, lowest recall (79.9%) but decent throughput.
- **Milvus Lite** — Single-threaded indexer (2% CPU), slow ingest but solid query quality.
- **DuckDB VSS** — HNSW index is experimental; very slow query (177ms), largest disk footprint.
- **Qdrant** — Embedded mode is not how Qdrant is designed to run; server mode is significantly faster.
- **TQDB** — Best ingest throughput among full DBs, smallest disk, lowest RAM.

---

## 4. Benchmark 2 — Windows vs Linux Comparison
**Same dataset. Windows 11 native Python vs WSL2 (/mnt/c) vs Linux native ext4.**

### Ingest time (seconds)

| Engine | Windows | WSL2 /mnt/c | Linux Native | Win→Native Δ |
|---|---|---|---|---|
| ChromaDB | 30.6s | 84.2s | 33.5s | +10% |
| LanceDB | 77.6s | 214.9s | 89.8s | +16% |
| Qdrant | 538.2s | 854.1s | 805.0s | +50% |
| TQDB b=8 HQ | 70.5s | 77.7s | **38.0s** | **−46%** |
| TQDB b=4 Balanced | 52.8s | 57.3s | **26.0s** | **−51%** |
| TQDB b=4 FastBuild | 26.7s | 56.5s | **22.7s** | **−15%** |

### Query latency p50

| Engine | Windows | WSL2 | Linux Native | Win→Native Δ |
|---|---|---|---|---|
| ChromaDB | 1.73ms | 2.88ms | 2.41ms | +39% |
| LanceDB | 9.17ms | 37.09ms | 8.10ms | −12% ✓ |
| Qdrant | 179.88ms | 105.81ms | 123.94ms | −31% ✓ |
| TQDB b=8 HQ | 10.74ms | 10.15ms | **8.73ms** | **−19%** ✓ |
| TQDB b=4 Balanced | 7.13ms | 8.06ms | **7.46ms** | −5% ✓ |
| TQDB b=4 FastBuild | 3.51ms | 4.42ms | **4.00ms** | +14% |

### Key findings

1. **WSL2 on /mnt/c is consistently slower** — Windows filesystem (DrvFS) translation overhead
   adds 30–60s of fixed I/O cost per run.
2. **Linux native ext4 beats Windows for TQDB ingest** — 46–51% faster for HQ/Balanced configs
   once the CPU core cap was fixed.
3. **Qdrant and LanceDB query faster on Linux** — native Linux binaries benefit from real mmap.
4. **Recall is deterministic across platforms** — seed=42 produces identical results on all OSes.
5. **WSL2 default core cap was 4** — all early Linux results were artificially throttled.
   After setting `processors=16` in `.wslconfig`, TQDB reached 1,922–2,199 vec/s.

---

## 5. Benchmark 3 — Patch Validation
**Commit `4a8b9ee`: skip QJL scoring loop when gamma=0 (fast_mode)**

| Config | Ingest before | Ingest after | Δ | p50 before | p50 after | Δ |
|---|---|---|---|---|---|---|
| TQDB b=8 HQ | 39.7s | 38.0s | −4% | 9.04ms | 8.73ms | −3% |
| TQDB b=4 Balanced | 33.1s | 26.0s | **−21%** | 7.58ms | 7.46ms | −2% |
| TQDB b=4 FastBuild | 32.1s | 22.7s | **−29%** | 4.91ms | 4.00ms | **−18%** |

Fix lands hardest on FastBuild (+41% ingest throughput) since `gamma=0` is its defining property.

---

## 6. TQDB Configuration Guide

| Config | bits | fast_mode | rerank | When to use |
|---|---|---|---|---|
| b=8 HQ | 8 | False | True | Best recall, moderate speed. Production default. |
| b=8 FastBuild | 8 | True | False | Good recall (95%), fast ingest. Good cold-start. |
| b=4 Balanced | 4 | False | True | Half the disk of b=8, still 88% recall. RAM-constrained. |
| b=4 FastBuild | 4 | True | False | Fastest ingest (2,199 vec/s), smallest disk. Recall trades off. |

**Recall vs storage tradeoff summary (50k vectors, 1536-dim):**

| Config | Disk | Recall@10 | Ingest | p50 |
|---|---|---|---|---|
| b=8 HQ | 119MB | 97.85% | 38.0s | 8.73ms |
| b=8 FastBuild | 119MB | 95.15% | 29.8s | 4.89ms |
| b=4 Balanced | 70MB | 88.70% | 26.0s | 7.46ms |
| b=4 FastBuild | 70MB | 83.10% | 22.7s | 4.00ms |

---

## 7. Embedding Model Guide

### Context window determines vector count — not file count

For a **1TB corpus** (~5M files, avg 200KB/file):

| Model | Dim | Context | Vectors from 1TB | Practical? |
|---|---|---|---|---|
| MiniLM-L6-v2 | 384 | 256 tok | ~975M | ❌ too many |
| bge-small/base | 384/768 | 512 tok | ~490M | ❌ too many |
| nomic-embed-v1.5 | 768 | 8,192 tok | ~30M | ✓ |
| jina-embeddings-v3 | 1024 | 8,192 tok | ~30M | ✓ |
| ada-002 / text-emb-3 | 1536 | 8,192 tok | ~30M | ✓ |
| **ada-002 + 100k chunking** | **1536** | **100k tok** | **~5M** | **✓ best** |

> Models with ≤512 token context produce 490M–975M vectors from 1TB — unmanageable for
> single-node embedded databases. Long-context models (8k+) are essential for large corpora.

### Storage for 1TB corpus by model (TQDB b=4 vs ChromaDB)

| Model | Vectors | TQDB b=4 disk | ChromaDB disk | TQDB RAM (mmap) |
|---|---|---|---|---|
| nomic-embed (768-dim) | ~30M | ~35 GB | ~158 GB | ~2 GB |
| jina-v3 (1024-dim) | ~30M | ~38 GB | ~189 GB | ~2 GB |
| ada-002 (1536-dim) | ~30M | ~44 GB | ~251 GB | ~3 GB |
| **ada-002 + 100k chunking** | **~5M** | **~7 GB** | **~42 GB** | **~512 MB** |

---

## 8. Scaling & RAM Guide

### TQDB bytes per vector (b=4, estimated)

| Dim | bytes/vec | Notes |
|---|---|---|
| 384 | ~1,034 | MiniLM, bge-small |
| 768 | ~1,178 | nomic-embed, bge-base |
| 1024 | ~1,274 | jina-v3, bge-large |
| 1536 | 1,466 | ada-002 (measured) |

Formula: `~890 bytes fixed overhead + (3 × dim / 8) bytes quantized data`

### How many vectors fit in 16GB RAM (full page cache warm)

| Config | bytes/vec | 16GB capacity |
|---|---|---|
| TQDB b=4, 768-dim | ~1,178 | ~14.5M vectors |
| TQDB b=4, 1024-dim | ~1,274 | ~13.4M vectors |
| TQDB b=4, 1536-dim | ~1,466 | ~11.6M vectors |
| TQDB b=8, 1536-dim | ~2,490 | ~6.9M vectors |

### mmap scaling — 16GB RAM can serve any corpus size

TQDB is **always in mmap mode** — there is no in-memory toggle. The OS manages what stays
in physical RAM via the page cache. Only the hot HNSW upper layers (~a few MB) are
always resident; the rest pages in on demand.

| Corpus | Vectors | Disk (b=4, 1536) | RAM needed (mmap) | Est. p50 (NVMe) |
|---|---|---|---|---|
| 1 TB | ~5M | ~7 GB | ~512 MB | ~8ms |
| 10 TB | ~50M | ~73 GB | ~1 GB | ~15–25ms |
| 20 TB | ~100M | ~147 GB | ~2 GB | ~40–80ms |
| 50 TB | ~250M | ~366 GB | ~3 GB | ~150–300ms |
| 100 TB | ~500M | ~730 GB | ~4 GB | ~500ms+ |

**Practical single-node sweet spot with 16GB RAM + NVMe: up to ~10–20TB corpus.**
Beyond that, query latency grows due to disk cache misses. Sharding across nodes
is recommended for 50TB+.

### Why retrieval stays fast at scale (HNSW properties)

- Query complexity is **O(log N)** — latency grows slowly with scale
- Top HNSW routing layers are tiny (a few MB) and stay in RAM permanently regardless of corpus size
- OS page cache automatically keeps recently-accessed nodes hot
- Latency at 50M vectors estimated at ~20ms — still usable for most applications

---

## 9. Real-World Benchmark (In Progress)

A larger benchmark using real embedded text is currently running:

| Item | Detail |
|---|---|
| Corpus | DBpedia full text (~332MB, ~1M documents) |
| Embedding model | `BAAI/bge-large-en-v1.5` (1024-dim, local CPU, fastembed) |
| Engines | ChromaDB, LanceDB, Milvus Lite, DuckDB VSS, Faiss, TQDB ×4 |
| New metrics | RAM peak + avg (ingest + query), CPU peak + avg, p50/p95/p99 |
| Status | Embedding in progress (~1M chunks × 1024-dim on CPU) |

Results will be appended to this document when complete.

---

## 10. Future Work

| Priority | Item |
|---|---|
| High | HPC benchmark (bare metal Linux, multi-core server, NVMe RAID) |
| High | Real-world benchmark results (1M vectors, bge-large-en-v1.5) |
| Medium | Benchmark at 1M, 10M, 100M vector scales |
| Medium | nomic-embed-text-v1.5 (768-dim) comparison |
| Medium | Concurrent query throughput (QPS benchmark) |
| Low | Distributed / sharded deployment exploration |
| Low | GPU-accelerated embedding pipeline |

---

## 11. How to Reproduce

```bash
# 1. Clone and enter native Linux path (avoid /mnt/c for accurate results)
git clone <repo> ~/tqdb_native && cd ~/tqdb_native

# 2. Set WSL2 cores (Windows side, then wsl --shutdown)
# %USERPROFILE%\.wslconfig → [wsl2] processors=16

# 3. Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy psutil chromadb lancedb qdrant-client \
            datasets fastembed pymilvus duckdb faiss-cpu setuptools

# 4. Build wheel
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
maturin develop --release

# 5. Run standard benchmark (50k vectors, pre-embedded)
python benchmarks/large_scale_bench.py --n_vecs 50000 --n_queries 200 --top_k 10

# 6. Run real-world benchmark (500MB corpus, local embedding)
python benchmarks/realworld_bench.py --target_mb 500 --n_queries 200 --top_k 10
```

---

## 12. Discussion

This project is a hobby effort exploring the practical tradeoffs of quantized vector storage.
Key open questions we'd love to discuss:

- **Is 83–98% recall sufficient for RAG use cases?** Most retrieval pipelines re-rank anyway.
- **Does 4–5× storage compression matter at your scale?** For edge/mobile deployments it does.
- **How does quantization interact with different embedding models?** Lower-dim models may
  be more sensitive to quantization error.
- **What workloads break TQDB's assumptions?** High-churn (many deletes/updates), streaming,
  or real-time indexing are areas not yet well benchmarked.

All feedback, results from your own hardware, and pull requests are welcome.

---

*Generated: 2026-04-03 | Platform: WSL2 Linux, Intel Core Ultra 7 255H, 16 cores*
*TQDB: arXiv:2504.19874 | Repo: TurboQuantDB*
