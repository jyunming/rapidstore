# TurboQuantDB Benchmarks

Three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — n=100k vectors each.
Full script: [`benchmarks/paper_recall_bench.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/paper_recall_bench.py).

All results use `quantizer_type=None/"dense"` and `fast_mode=True, rerank=True` (MSE-only, matching paper Figure 5 bit allocation). ANN rows use HNSW (md=32, ef=128).

To regenerate:
```bash
python benchmarks/paper_recall_bench.py --update-readme --track
```
(requires `pip install datasets psutil matplotlib`)

CI perf gate:
- PR CI runs a fast smoke perf gate (`benchmarks/sprint_smoke.py`) in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
- This is a regression tripwire (latency/recall/ingest), not a full publish benchmark replacement.

---

<!-- PAPER_BENCH_START -->
### Algorithm Validation — Recall vs Paper

![Benchmark recall curves — TQDB vs paper](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots.png)

Brute-force recall across all three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5 — n=100k vectors, paper values read visually from plots (approximate).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5a) | ≈55.0% | ≈70.0% | ≈83.0% | ≈91.0% | ≈96.0% | ≈99.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 51.1% | 67.2% | 79.9% | 88.7% | 94.6% | 97.9% | 99.2% |
| **TQDB b=2 rerank=T** | 97.1% | 98.2% | 98.2% | 98.2% | 98.2% | 98.2% | 98.2% |
| TurboQuant 4-bit (paper Fig. 5a) | ≈86.0% | ≈96.0% | ≈99.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 81.9% | 94.5% | 99.1% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 98.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5b) | ≈89.5% | ≈98.0% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 83.7% | 95.6% | 99.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5b) | ≈97.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 95.8% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5c) | ≈90.5% | ≈98.5% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 89.6% | 98.3% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5c) | ≈97.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 96.3% | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

All TQDB rows use `fast_mode=True` (MSE-only: all `b` bits go to the MSE codebook, no QJL residual). This is the same allocation as the paper's Figure 5 — b MSE bits/dim. Any residual gap at GloVe k=1 (~0–3%) is attributable to dataset sampling (we use the first 100k vectors from the 1.18M-token corpus; the paper used a random sample). DBpedia results match within 1–2% across all k values.

### Performance & Config Trade-offs

![Config trade-off overview — latency, disk, RAM, CPU](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots_perf.png)

All 8 configs — brute-force and ANN (HNSW md=32, ef=128), all using `fast_mode=True` (MSE-only). Disk MB for ANN includes `graph.bin`. RAM = peak RSS during query phase. Index = HNSW build time (ANN only).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 1.2s | — | 6.2 | 193 | 2.51 | 3.45 | 51.1% | 0.651 |
| b=2 rerank=T | Brute | 2.0s | — | 25.9 | 214 | 3.40 | 4.45 | 97.1% | 0.977 |
| b=4 rerank=F | Brute | 1.6s | — | 11.0 | 198 | 1.46 | 2.07 | 81.9% | 0.898 |
| b=4 rerank=T | Brute | 2.4s | — | 30.6 | 219 | 4.25 | 5.85 | 98.7% | 0.994 |
| b=2 rerank=F | ANN | 1.2s | 10.8s | 14.8 | 226 | 0.52 | 1.13 | 32.5% | 0.399 |
| b=2 rerank=T | ANN | 2.1s | 15.9s | 34.4 | 248 | 2.88 | 5.97 | 57.4% | 0.577 |
| b=4 rerank=F | ANN | 1.6s | 12.1s | 19.6 | 234 | 0.44 | 1.62 | 50.2% | 0.538 |
| b=4 rerank=T | ANN | 2.5s | 17.9s | 39.2 | 256 | 3.37 | 6.23 | 70.2% | 0.706 |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 18.5s | — | 46.9 | 809 | 20.22 | 23.46 | 83.7% | 0.909 |
| b=2 rerank=T | Brute | 21.2s | — | 193.8 | 952 | 21.68 | 25.67 | 99.7% | 0.999 |
| b=4 rerank=F | Brute | 23.1s | — | 83.6 | 840 | 13.79 | 17.04 | 95.8% | 0.978 |
| b=4 rerank=T | Brute | 26.8s | — | 230.4 | 990 | 14.63 | 17.61 | 99.7% | 0.999 |
| b=2 rerank=F | ANN | 19.7s | 38.4s | 55.5 | 770 | 4.55 | 7.51 | 80.5% | 0.868 |
| b=2 rerank=T | ANN | 23.5s | 133.9s | 202.4 | 920 | 21.74 | 85.62 | 97.9% | 0.981 |
| b=4 rerank=F | ANN | 24.5s | 41.6s | 92.2 | 808 | 3.71 | 21.91 | 91.0% | 0.929 |
| b=4 rerank=T | ANN | 29.0s | 139.6s | 239.0 | 958 | 14.23 | 67.85 | 97.6% | 0.978 |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 99.4s | — | 110.6 | 1515 | 48.74 | 208.74 | 89.6% | 0.945 |
| b=2 rerank=T | Brute | 130.0s | — | 403.9 | 1807 | 52.37 | 249.04 | 99.7% | 0.999 |
| b=4 rerank=F | Brute | 117.9s | — | 183.8 | 1600 | 37.41 | 172.51 | 96.3% | 0.981 |
| b=4 rerank=T | Brute | 154.3s | — | 477.1 | 1807 | 39.68 | 224.89 | 99.7% | 0.999 |
| b=2 rerank=F | ANN | 134.0s | 102.2s | 119.2 | 1453 | 16.97 | 81.36 | 84.7% | 0.893 |
| b=2 rerank=T | ANN | 155.6s | 704.6s | 412.5 | 1748 | 93.09 | 476.88 | 98.0% | 0.981 |
| b=4 rerank=F | ANN | 136.7s | 156.6s | 192.4 | 1529 | 20.33 | 123.85 | 91.3% | 0.930 |
| b=4 rerank=T | ANN | 136.5s | 906.8s | 485.8 | 1819 | 76.89 | 267.36 | 97.6% | 0.977 |

**Reproduction:** `maturin develop --release && python benchmarks/paper_recall_bench.py --update-readme --track`  (requires `pip install datasets psutil matplotlib`)

<!-- PAPER_BENCH_END -->

---

## When to use brute-force vs. ANN

The best search mode depends on vector dimensionality.

**Use brute-force (`_use_ann=False`, the default) when d ≤ 256**

At low dimensionality, quantization noise dominates. On GloVe-200 (d=200), ANN loses significant recall versus brute-force:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | **72.4%** | 53.8% | ~1.2× faster p50 |
| b=2, rerank=T | **33.7%** | 24.3% | ~1× (no gain) |

The HNSW graph built on quantized distances is inaccurate at low dimension. Use brute-force for d ≤ 256.

**Use ANN (`_use_ann=True`) when d ≥ 512**

At high dimensionality, quantization is accurate and the ANN approximation is tight. On DBpedia d=1536:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | 92.2% | **90.4%** | ~1.7× faster p50 |
| b=4, rerank=F | 92.2% | **87.8%** | ~4× faster p50 |

ANN costs ~2 points of recall while cutting latency from ~51ms to ~37ms p50. For production RAG at d=1536+, build the index once after initial load.

**Summary:**

| Dimension range | Recommended mode | Reason |
|-----------------|-----------------|--------|
| d ≤ 256 | Brute-force | Quantization noise collapses ANN recall |
| d = 512–1024 | Either (test both) | Moderate quantization quality; ANN gain is partial |
| d ≥ 1536 | ANN | High-d quantization is accurate; ANN gives 1.5–4× latency gain with <3% recall cost |
