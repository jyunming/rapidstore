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
| b=2 rerank=F | Brute | 1.2s | — | 6.2 | 193 | 2.53 | 3.05 | 51.1% | 0.651 |
| b=2 rerank=T | Brute | 4.7s | — | 25.9 | 213 | 3.43 | 4.20 | 97.1% | 0.977 |
| b=4 rerank=F | Brute | 1.6s | — | 11.0 | 200 | 1.35 | 1.75 | 81.9% | 0.898 |
| b=4 rerank=T | Brute | 4.9s | — | 30.6 | 220 | 4.11 | 4.77 | 98.7% | 0.994 |
| b=2 rerank=F | ANN | 1.3s | 9.4s | 14.8 | 222 | 0.48 | 1.18 | 32.1% | 0.395 |
| b=2 rerank=T | ANN | 4.4s | 13.2s | 34.5 | 248 | 2.78 | 5.86 | 57.6% | 0.578 |
| b=4 rerank=F | ANN | 2.0s | 7.2s | 19.6 | 230 | 0.31 | 0.81 | 50.3% | 0.540 |
| b=4 rerank=T | ANN | 4.8s | 13.5s | 39.2 | 256 | 2.80 | 5.74 | 70.8% | 0.711 |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 11.4s | — | 46.9 | 801 | 18.06 | 23.84 | 83.7% | 0.909 |
| b=2 rerank=T | Brute | 14.3s | — | 193.8 | 951 | 20.77 | 23.12 | 99.7% | 0.999 |
| b=4 rerank=F | Brute | 14.5s | — | 83.6 | 842 | 9.92 | 30.26 | 95.8% | 0.978 |
| b=4 rerank=T | Brute | 27.1s | — | 230.4 | 987 | 10.17 | 11.44 | 99.7% | 0.999 |
| b=2 rerank=F | ANN | 11.7s | 34.7s | 55.5 | 769 | 4.17 | 7.79 | 79.7% | 0.861 |
| b=2 rerank=T | ANN | 14.3s | 123.0s | 202.4 | 914 | 17.48 | 29.07 | 98.3% | 0.985 |
| b=4 rerank=F | ANN | 14.7s | 34.0s | 92.2 | 802 | 2.93 | 5.88 | 91.5% | 0.934 |
| b=4 rerank=T | ANN | 26.6s | 119.2s | 239.0 | 953 | 10.99 | 17.80 | 97.4% | 0.976 |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 44.0s | — | 110.6 | 1515 | 40.88 | 78.44 | 89.6% | 0.945 |
| b=2 rerank=T | Brute | 53.9s | — | 403.9 | 1812 | 42.43 | 75.36 | 99.7% | 0.999 |
| b=4 rerank=F | Brute | 51.3s | — | 183.8 | 1592 | 29.29 | 31.14 | 96.3% | 0.981 |
| b=4 rerank=T | Brute | 60.3s | — | 477.1 | 1885 | 25.50 | 54.85 | 99.7% | 0.999 |
| b=2 rerank=F | ANN | 44.3s | 58.2s | 119.2 | 1450 | 11.00 | 16.58 | 85.7% | 0.904 |
| b=2 rerank=T | ANN | 54.0s | 461.4s | 412.5 | 1744 | 37.74 | 56.40 | 98.5% | 0.986 |
| b=4 rerank=F | ANN | 49.1s | 57.2s | 192.4 | 1523 | 8.62 | 11.97 | 91.5% | 0.931 |
| b=4 rerank=T | ANN | 63.4s | 450.5s | 485.8 | 1818 | 23.12 | 34.93 | 98.5% | 0.987 |

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

---

## Hybrid retrieval evaluation

Dense retrieval misses keyword-heavy queries; sparse retrieval misses
paraphrases. The harness at [`benchmarks/retrieval_eval.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/retrieval_eval.py)
grades all three paths (dense, BM25, hybrid RRF) on the same query sets so
new retrieval features can be judged against measured wins instead of
anecdotes. Append-only history lives at `benchmarks/retrieval_eval_history.json`.

```bash
python benchmarks/retrieval_eval.py             # 1k synthetic corpus, ~3s
python benchmarks/retrieval_eval.py --n 5000    # bigger corpus, ~15s
python benchmarks/retrieval_eval.py --no-history # don't append to history
```

Three query sets are generated from the synthetic corpus:

| Query set | What's in the query | Ideal retriever |
|-----------|---------------------|-----------------|
| `semantic` | A perturbed corpus vector, no text | Pure dense |
| `lexical`  | A rare token + a random orthogonal vector | Pure BM25 |
| `mixed`    | Half of each, interleaved | Hybrid (RRF) |

Representative output (N=1k, D=128, Q=100 per set, weight=0.5):

| query set | path | R@1 | R@10 | MRR@10 | NDCG@10 |
|-----------|------|----:|-----:|-------:|--------:|
| semantic  | dense  | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic  | bm25   | 0.000 | 0.000 | 0.000 | 0.000 |
| semantic  | hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical   | dense  | 0.010 | 0.010 | 0.010 | 0.010 |
| lexical   | bm25   | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical   | hybrid | 0.200 | 1.000 | 0.466 | 0.597 |
| mixed     | dense  | 0.500 | 0.500 | 0.500 | 0.500 |
| mixed     | bm25   | 0.500 | 0.500 | 0.500 | 0.500 |
| mixed     | hybrid | 0.550 | 1.000 | 0.689 | 0.765 |

Read the table as: **on the mixed workload, hybrid raises R@10 from 0.500
to 1.000** (+50 pp) over either path alone, while losing nothing on
semantic-only queries. On lexical queries hybrid is dominated by pure BM25
on R@1, but still recovers the gold doc in the top-10 every time, which is
what most RAG pipelines actually consume.

---

## v0.8 features — what isn't yet benchmarked

The v0.8 sprint added five Python-side features. Three of them have correctness coverage in the test suite but aren't yet tracked in the longitudinal perf history:

- **Multi-vector / MaxSim retrieval at scale** (`MultiVectorStore`) — the v0.8 wrapper is a Python layer over the existing single-vector engine. Per-config recall and latency at 10k+ docs / 80k+ tokens are pending, and will be the headline numbers when v0.9 lands the native engine integration.
- **`AsyncDatabase` concurrent-search throughput** — `tests/test_async_api.py` includes a 50-concurrent-search proof of parallelism but no longitudinal numbers; the actual ceiling depends on user thread-pool sizing and embedder latency.
- **Migration toolkit throughput** (`tqdb.migrate`) — the CLI prints rows/sec per run but those numbers aren't tracked in perf_history. For now, expect rates close to a plain `Database.insert_batch` loop on the same hardware.

LangChain / LlamaIndex / hybrid-via-integration overhead is also not benchmarked separately. The wrappers add a small Python-layer cost per call (one filter-dict translation + one numpy conversion) which has been negligible in practice on the 1k-5k corpora the test suite uses.
