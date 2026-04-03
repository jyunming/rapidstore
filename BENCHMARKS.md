# TurboQuantDB Benchmarks

Benchmarks compare TurboQuantDB against ChromaDB, LanceDB, FAISS, Milvus Lite, DuckDB VSS, and Qdrant on a real-world embedding dataset.

## Setup

| Parameter | Value |
|-----------|-------|
| Dataset | DBpedia OpenAI text-embedding-ada-002 (50,000 vectors) |
| Dimension | 1,536 |
| Top-k | 10 |
| Ground truth | Brute-force cosine similarity over 200 queries |
| Recall metric | Recall@10 (fraction of true top-10 returned) |
| Latency metric | p50/p95 over 200 queries after warm-up |

## TQDB Presets

| Preset | `bits` | `fast_mode` | HNSW params | Trade-off |
|--------|--------|-------------|-------------|-----------|
| **HQ** | 8 | off | `max_degree=32, search_list=128, n_ref=10` | Best recall |
| **Balanced** | 4 | off | `max_degree=32, search_list=128, n_ref=10` | Balance of recall, speed, disk |
| **FastBuild** | 4 | on | `max_degree=16, search_list=64, n_ref=3` | Fastest ingest, lowest latency |

---

## Results — Linux Native (ext4, 16 cores) ← Recommended baseline

50,000 vectors · dim=1,536 · top_k=10

### Performance

| Engine | Ingest | vec/s | Disk | RAM | CPU-ing | CPU-qry | p50 | p95 | Recall@10 | MRR@10 |
|--------|--------|-------|------|-----|---------|---------|-----|-----|----------|-------|
| **FAISS HNSW** *(ceiling)* | 15.1s | 3,316 | 306 MB | 698 MB | 1331% | 100% | 0.96ms | 1.27ms | 99.75% | 1.0000 |
| **ChromaDB** (HNSW) | 33.5s | 1,493 | 398 MB | 924 MB | 258% | 92% | 2.41ms | 3.42ms | 99.75% | 1.0000 |
| **TQDB b=4 FastBuild** | 22.7s | 2,199 | 70 MB | 506 MB | 1262% | 104% | 4.00ms | 6.88ms | 83.10% | 0.9900 |
| **TQDB b=4 Balanced** | 26.0s | 1,922 | 70 MB | 507 MB | 1313% | 196% | 7.46ms | 11.24ms | 88.70% | 0.9900 |
| **TQDB b=8 HQ** | 38.0s | 1,316 | 119 MB | 554 MB | 1175% | 179% | 8.73ms | 11.97ms | 97.85% | 0.9950 |
| LanceDB (IVF_PQ) | 89.8s | 556 | 318 MB | 1,722 MB | 1100% | 152% | 8.10ms | 9.57ms | 79.90% | 1.0000 |
| Milvus Lite (HNSW) | 251.4s | 198 | 391 MB | 493 MB | 2% | 7% | 12.18ms | 14.20ms | 100.00% | 1.0000 |
| DuckDB VSS (HNSW) | 407.2s | 122 | 745 MB | 1,597 MB | 132% | 193% | 177.00ms | 195.46ms | 100.00% | 1.0000 |
| Qdrant (HNSW)† | 805.0s | 62 | 685 MB | 1,152 MB | 7% | 1000% | 123.94ms | 163.84ms | 100.00% | 1.0000 |

† Qdrant in embedded mode — not representative of its server deployment.

### Disk Compression

Float32 baseline for 50k × 1536: **293 MB**

| Engine | Disk | bytes/vec | vs float32 | vs ChromaDB |
|--------|------|-----------|-----------|------------|
| DuckDB VSS | 745 MB | 15,619 | 0.39× smaller | 0.53× |
| Qdrant | 685 MB | 14,373 | 0.43× smaller | 0.58× |
| ChromaDB | 398 MB | 8,353 | 0.74× smaller | baseline |
| Milvus Lite | 391 MB | 8,203 | 0.75× smaller | 1.02× |
| LanceDB | 318 MB | 6,663 | 0.92× smaller | 1.25× |
| FAISS HNSW | 306 MB | 6,416 | 0.96× smaller | 1.30× |
| **TQDB b=8 HQ** | **119 MB** | **2,490** | **2.47× smaller** | **3.35×** |
| **TQDB b=4** | **70 MB** | **1,466** | **4.19× smaller** | **5.70×** |

TQDB b=4 stores each vector in **1,466 bytes** vs float32's 6,144 bytes — a **4.2× reduction** with no training.

---

## Results — Windows 11 (12 cores)

50,000 vectors · dim=1,536 · top_k=10

| Engine | Ingest | vec/s | Disk | RAM | CPU-ing | CPU-qry | p50 | p95 | Recall@10 |
|--------|--------|-------|------|-----|---------|---------|-----|-----|----------|
| ChromaDB (HNSW) | 30.6s | 1,633 | 398 MB | 865 MB | 244% | 100% | 1.73ms | 2.44ms | 99.75% |
| **TQDB b=4 FastBuild** | **28.2s** | **1,775** | **70 MB** | **487 MB** | 801% | 95% | 5.30ms | 8.71ms | 83.35% |
| **TQDB b=4 Balanced** | 37.8s | 1,324 | 70 MB | 488 MB | 898% | 170% | 9.98ms | 15.21ms | 89.15% |
| **TQDB b=8 HQ** | 59.2s | 844 | 119 MB | 537 MB | 780% | 147% | 12.42ms | 19.34ms | 97.25% |
| LanceDB (IVF_PQ) | 77.6s | 644 | 318 MB | 526 MB | 1067% | 155% | 9.17ms | 10.51ms | 79.50% |
| Qdrant (HNSW)† | 538.2s | 92 | 685 MB | 1,151 MB | 17% | 130% | 179.88ms | 186.80ms | 100.00% |

---

## Results — WSL2 on /mnt/c (DrvFS)

> Note: WSL2 on a Windows filesystem (DrvFS) has higher I/O overhead. Run on native Linux ext4 for best results.

50,000 vectors · dim=1,536 · top_k=10

| Engine | Ingest | vec/s | Disk | RAM | p50 | p95 | Recall@10 | MRR@10 |
|--------|--------|-------|------|-----|-----|-----|----------|-------|
| ChromaDB (HNSW) | 84.2s | 594 | 398 MB | 906 MB | 2.88ms | 4.11ms | 99.75% | 1.0000 |
| **TQDB b=4 FastBuild** | 56.5s | 885 | 70 MB | 503 MB | 4.42ms | 6.59ms | 84.05% | 1.0000 |
| **TQDB b=4 Balanced** | 57.3s | 872 | 70 MB | 505 MB | 8.06ms | 10.91ms | 90.25% | 1.0000 |
| **TQDB b=8 HQ** | 77.7s | 643 | 119 MB | 551 MB | 10.15ms | 14.19ms | 98.10% | 1.0000 |
| LanceDB (IVF_PQ) | 214.9s | 232 | 318 MB | 1,220 MB | 37.09ms | 64.50ms | 80.50% | 1.0000 |
| Qdrant (HNSW)† | 854.1s | 58 | 685 MB | 1,151 MB | 105.81ms | 139.18ms | 100.00% | 1.0000 |

---

## Three-Platform Comparison (TQDB only)

| Preset | Windows | WSL2 (/mnt/c) | Linux Native | Win→Native Δ |
|--------|---------|--------------|-------------|-------------|
| **Ingest time** | | | | |
| b=8 HQ | 59.2s | 77.7s | 38.0s | −36% ✅ |
| b=4 Balanced | 37.8s | 57.3s | 26.0s | −31% ✅ |
| b=4 FastBuild | 28.2s | 56.5s | 22.7s | −19% ✅ |
| **p50 latency** | | | | |
| b=8 HQ | 12.42ms | 10.15ms | 8.73ms | −30% ✅ |
| b=4 Balanced | 9.98ms | 8.06ms | 7.46ms | −25% ✅ |
| b=4 FastBuild | 5.30ms | 4.42ms | 4.00ms | −25% ✅ |

Linux native is the best platform for TQDB due to efficient `mmap` and lower `madvise` overhead.

---

## Rerank Precision Impact

`rerank_precision` controls how raw vectors are stored for the final re-scoring step.

| Mode | Extra disk+RAM | Recall@10 | Use when |
|------|---------------|----------|---------|
| `None` (default) | 0 | ~83–88% | Default — uses dequantized vectors for reranking |
| `"f16"` | n×d×2 bytes | ~97–100% | High recall without full f32 cost |
| `"f32"` | n×d×4 bytes | ~97–100% | Maximum precision |

At n=50k, dim=1536: f16 adds **+154 MB**, f32 adds **+307 MB**.  
At n=10k, dim=384: f16 adds **+7.7 MB**, f32 adds **+15.4 MB**.

Recall measured at dim=384, n=10k, b=4, cosine:

| Mode | Recall@10 |
|------|----------|
| Default (dequant) | 83.4% |
| `rerank_precision="f16"` | **100.0%** |
| `rerank_precision="f32"` | **100.0%** |

---

## DB RAM at Rest

The **actual DB memory** after freeing input data (n=50k, dim=1536, b=4):

| Component | RAM |
|-----------|-----|
| `live_codes.bin` mmap | ~49 MB |
| ProdQuantizer matrices | ~19 MB |
| HNSW graph mmap | ~4 MB |
| IdPool + metadata | ~3 MB |
| Thread pool + process | ~37 MB |
| **Total DB overhead** | **~112 MB** |

The 500 MB figures in benchmark tables include ~307 MB of Python input data held in memory during ingest.

---

## Reproduce

```bash
# Install dependencies
pip install chromadb lancedb qdrant-client faiss-cpu milvus-lite duckdb psutil

# Run full comparison (requires maturin develop --release first)
python benchmarks/full_comparison_bench.py

# CI quality gate (min recall 0.60, max p50 100ms)
python benchmarks/ci_quality_gate.py
```
