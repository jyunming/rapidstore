"""bench_core.py — Shared benchmark utilities for TurboQuantDB.

Provides:
  K_VALUES          : k values [1, 2, 4, 8, 16, 32, 64]
  BITS_LIST         : bits tested [2, 4]
  PAPER_RECALL      : paper reference recall values (arXiv:2504.19874 Fig. 5)
  PERF_METRIC_ROWS  : canonical (key, label) list for perf plots — shared by
                      paper_recall_bench.py and run_bench_private.py
  CKPT_DIR          : cache directory path (auto-created on first use)
  CpuRamSampler     : context manager — background-thread CPU% / RSS sampler
  compute_recalls   : Recall@1@k for multiple k values
  compute_mrr       : Mean Reciprocal Rank
  disk_size_mb      : directory size in MB
  load_glove        : GloVe-200-angular, 100k corpus, 10k queries (cached)
  load_dbpedia(dim) : DBpedia OpenAI3, 100k corpus, 1k queries (cached)

Import example::

    from bench_core import (
        K_VALUES, PAPER_RECALL, CpuRamSampler,
        compute_recalls, compute_mrr, disk_size_mb,
        load_glove, load_dbpedia,
    )
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

try:
    import psutil as _psutil
    _PROCESS = _psutil.Process(os.getpid())
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]
    _PROCESS = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


K_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64]
BITS_LIST: list[int] = [2, 4]

# Canonical metric set used by both paper_recall_bench.py and run_bench_private.py.
# Keep these in sync — they drive both the README perf panel and the private plots.
PERF_METRIC_ROWS: list[tuple[str, str]] = [
    ("throughput_vps",      "Ingest (vps)"),
    ("p50_ms",              "p50 (ms)"),
    ("p99_ms",              "p99 (ms)"),
    ("disk_mb",             "Disk (MB)"),
    ("ram_ingest_peak_mb",  "RAM ingest (MB)"),
    ("ram_query_peak_mb",   "RAM query (MB)"),
    ("cpu_ingest_pct",      "CPU ingest (%)"),
    ("cpu_query_pct",       "CPU query (%)"),
    ("mrr",                 "MRR"),
]

CKPT_DIR: Path = Path(__file__).parent / "_paper_bench_cache"

# Approximate recall values read visually from Figure 5 of arXiv:2504.19874.
PAPER_RECALL: dict[str, dict[int, dict[int, float]]] = {
    "glove-200": {
        2: {1: 0.550, 2: 0.700, 4: 0.830, 8: 0.910, 16: 0.960, 32: 0.990, 64: 1.000},
        4: {1: 0.860, 2: 0.960, 4: 0.990, 8: 1.000, 16: 1.000, 32: 1.000, 64: 1.000},
    },
    "dbpedia-1536": {
        2: {1: 0.895, 2: 0.980, 4: 0.995, 8: 1.000, 16: 1.000, 32: 1.000, 64: 1.000},
        4: {1: 0.970, 2: 1.000, 4: 1.000, 8: 1.000, 16: 1.000, 32: 1.000, 64: 1.000},
    },
    "dbpedia-3072": {
        2: {1: 0.905, 2: 0.985, 4: 0.995, 8: 1.000, 16: 1.000, 32: 1.000, 64: 1.000},
        4: {1: 0.975, 2: 1.000, 4: 1.000, 8: 1.000, 16: 1.000, 32: 1.000, 64: 1.000},
    },
}


# ── CPU / RAM sampler ──────────────────────────────────────────────────────────

class CpuRamSampler:
    """Background-thread sampler for CPU% and process RSS.

    Samples every ``interval`` seconds while running.  Use as a context
    manager for automatic start/stop::

        with CpuRamSampler() as s:
            do_heavy_work()
        print(s.avg_cpu_pct, s.peak_ram_mb, s.delta_ram_mb)

    Or manually::

        s = CpuRamSampler()
        s.start()
        do_work()
        s.stop()

    All ``*_mb`` properties return 0 if ``psutil`` is unavailable.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: list[float] = []
        self.ram_samples_mb: list[float] = []
        self._start_rss_mb: float = 0.0

    def start(self) -> "CpuRamSampler":
        self._start_rss_mb = _PROCESS.memory_info().rss / (1 << 20) if _HAS_PSUTIL else 0.0
        self._stop.clear()
        self.cpu_samples.clear()
        self.ram_samples_mb.clear()
        if _HAS_PSUTIL:
            _psutil.cpu_percent(interval=None)  # discard first (always returns 0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if _HAS_PSUTIL:
                self.cpu_samples.append(_psutil.cpu_percent(interval=None))
                self.ram_samples_mb.append(_PROCESS.memory_info().rss / (1 << 20))

    @property
    def avg_cpu_pct(self) -> float:
        return float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0

    @property
    def peak_ram_mb(self) -> float:
        return float(max(self.ram_samples_mb)) if self.ram_samples_mb else self._start_rss_mb

    @property
    def delta_ram_mb(self) -> float:
        return self.peak_ram_mb - self._start_rss_mb

    def __enter__(self) -> "CpuRamSampler":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


# ── Recall / MRR ──────────────────────────────────────────────────────────────

def compute_recalls(
    all_results: list[list[str]],
    true_top1: np.ndarray,
    ks: list[int] = K_VALUES,
    id_fn=str,
) -> dict[int, float]:
    """Recall@1@k: fraction of queries where true nearest neighbour is in top-k.

    Args:
        all_results: ``all_results[i]`` is the ordered list of returned IDs for query i.
        true_top1:   integer array; ``true_top1[i]`` is the corpus index of the true top-1.
        ks:          k values to evaluate.
        id_fn:       converts integer index → ID string (default: ``str``).

    Returns:
        Mapping k → recall fraction in [0, 1].
    """
    n = len(true_top1)
    return {
        k: sum(
            1 for res, idx in zip(all_results, true_top1)
            if id_fn(int(idx)) in res[:k]
        ) / n
        for k in ks
    }


def compute_mrr(
    all_results: list[list[str]],
    true_top1: np.ndarray,
    id_fn=str,
) -> float:
    """Mean Reciprocal Rank of the true nearest neighbour in returned results."""
    rrs: list[float] = []
    for res, idx in zip(all_results, true_top1):
        target = id_fn(int(idx))
        try:
            rrs.append(1.0 / (res.index(target) + 1))
        except ValueError:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


def disk_size_mb(path: str | Path) -> float:
    """Total size of all files under *path* in megabytes."""
    total = 0
    for root, _, files in os.walk(str(path)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1 << 20)


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_glove() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GloVe-200-angular: 100k corpus, 10k queries, cached ground truth top-1.

    Returns:
        vecs:      ``(100_000, 200)`` float32, unit-normalised
        qvecs:     ``(10_000,  200)`` float32, unit-normalised
        true_top1: ``(10_000,)``     int64, corpus index of true nearest neighbour
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    N_DATA, N_QUERIES = 100_000, 10_000
    ckpt_vecs  = CKPT_DIR / "glove200_100000_vecs.npy"
    ckpt_qvecs = CKPT_DIR / "glove200_10000_qvecs.npy"
    ckpt_truth = CKPT_DIR / "glove200_100000_truth_top1.npy"

    if ckpt_vecs.exists() and ckpt_qvecs.exists():
        print("  Loading cached GloVe-200 vectors ...", flush=True)
        vecs  = np.load(ckpt_vecs)
        qvecs = np.load(ckpt_qvecs)
    else:
        from datasets import load_dataset
        print("  Downloading GloVe-200 corpus (100k) from open-vdb/glove-200-angular ...", flush=True)
        ds_corpus = load_dataset("open-vdb/glove-200-angular", "train", split="train", streaming=True)
        raw: list = []
        for i, row in enumerate(ds_corpus):
            if i >= N_DATA:
                break
            raw.append(row["emb"])
            if (i + 1) % 10_000 == 0:
                print(f"    corpus {i+1:>7,} / {N_DATA:,}", flush=True)
        vecs = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.where(norms > 0, norms, 1.0)

        print(f"  Downloading GloVe-200 queries ({N_QUERIES}) ...", flush=True)
        ds_queries = load_dataset("open-vdb/glove-200-angular", "test", split="test", streaming=True)
        qraw = [row["emb"] for i, row in enumerate(ds_queries) if i < N_QUERIES]
        qvecs = np.array(qraw, dtype=np.float32)
        norms_q = np.linalg.norm(qvecs, axis=1, keepdims=True)
        qvecs /= np.where(norms_q > 0, norms_q, 1.0)

        np.save(ckpt_vecs, vecs)
        np.save(ckpt_qvecs, qvecs)
        print("  Saved.", flush=True)

    if ckpt_truth.exists():
        print("  Loading cached GloVe-200 ground truth ...", flush=True)
        true_top1 = np.load(ckpt_truth)
    else:
        print(f"  Computing GloVe-200 ground truth ({N_QUERIES} queries) ...", flush=True)
        t0 = time.perf_counter()
        rows = [np.argmax(qvecs[i:i+200] @ vecs.T, axis=1) for i in range(0, N_QUERIES, 200)]
        true_top1 = np.concatenate(rows)
        print(f"  Done in {time.perf_counter() - t0:.1f}s", flush=True)
        np.save(ckpt_truth, true_top1)

    print(f"  GloVe-200: corpus={vecs.shape}, queries={qvecs.shape}", flush=True)
    return vecs, qvecs, true_top1


def load_dbpedia(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DBpedia OpenAI3: 100k corpus, 1k queries, cached ground truth top-1.

    Args:
        dim: embedding dimension — 1536 or 3072.

    Returns:
        vecs:      ``(100_000, dim)`` float32, unit-normalised
        qvecs:     ``(1_000,   dim)`` float32, unit-normalised
        true_top1: ``(1_000,)``      int64, corpus index of true nearest neighbour
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    N_DATA, N_QUERIES = 100_000, 1_000
    tag = f"dbpedia{dim}"
    ckpt_vecs  = CKPT_DIR / f"{tag}_{N_DATA}_vecs.npy"
    ckpt_qvecs = CKPT_DIR / f"{tag}_{N_QUERIES}_qvecs.npy"
    ckpt_truth = CKPT_DIR / f"{tag}_{N_DATA}_truth_top1.npy"

    if ckpt_vecs.exists() and ckpt_qvecs.exists():
        print(f"  Loading cached DBpedia-{dim} vectors ...", flush=True)
        vecs  = np.load(ckpt_vecs)
        qvecs = np.load(ckpt_qvecs)
    else:
        from datasets import load_dataset
        HF = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
        split_name = "train"  # dataset only has a single "train" split
        # Dataset only has the 3072-dim field; for 1536 we slice the first half
        # (text-embedding-3-large uses Matryoshka so prefix dims are meaningful)
        src_field = "text-embedding-3-large-3072"
        print(f"  Downloading DBpedia-{dim} from {HF} (split={split_name}) ...", flush=True)
        ds = load_dataset(HF, split=split_name, streaming=True)
        raw = []
        for i, row in enumerate(ds):
            if i >= N_DATA + N_QUERIES:
                break
            raw.append(row[src_field][:dim])
            if (i + 1) % 10_000 == 0:
                print(f"    {i+1:>7,} / {N_DATA + N_QUERIES:,}", flush=True)
        all_vecs = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        all_vecs /= np.where(norms > 0, norms, 1.0)
        vecs  = all_vecs[N_QUERIES:]
        qvecs = all_vecs[:N_QUERIES]
        np.save(ckpt_vecs, vecs)
        np.save(ckpt_qvecs, qvecs)
        print("  Saved.", flush=True)

    if ckpt_truth.exists():
        print(f"  Loading cached DBpedia-{dim} ground truth ...", flush=True)
        true_top1 = np.load(ckpt_truth)
    else:
        print(f"  Computing DBpedia-{dim} ground truth ({N_QUERIES} queries) ...", flush=True)
        t0 = time.perf_counter()
        rows = [np.argmax(qvecs[i:i+50] @ vecs.T, axis=1) for i in range(0, N_QUERIES, 50)]
        true_top1 = np.concatenate(rows)
        print(f"  Done in {time.perf_counter() - t0:.1f}s", flush=True)
        np.save(ckpt_truth, true_top1)

    print(f"  DBpedia-{dim}: corpus={vecs.shape}, queries={qvecs.shape}", flush=True)
    return vecs, qvecs, true_top1
