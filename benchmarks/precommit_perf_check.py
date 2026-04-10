"""
Pre-commit performance regression check for TurboQuantDB.

Uses the same methodology as the paper recall benchmark (arXiv:2504.19874):
  - Recall@1@k  (is the true nearest neighbour in the returned top-k?)
  - Uses insert_batch in chunks of 2000 (not individual inserts)
  - rerank=False, metric="ip"  (pure quantized search, no rerank overhead)
  - GloVe-200 cached data when available; synthetic d=200 otherwise

Tracks ALL metrics:
  recall@1@k for k=1,4,8, query latency (avg/p50/p95), speedup vs numpy,
  ingest throughput, ingest wall time, ingest CPU time,
  disk bytes, compression ratio, RAM estimate,
  live codes bytes, live vectors bytes.

The baseline (benchmarks/perf_baseline.json) stores the BEST value ever
measured locally for each metric.  The floor for the next commit is
best × 0.95 for "higher is better" metrics, or best × 1.05 for
"lower is better" metrics (5% tolerance in every direction).

When any metric improves beyond its stored best, the baseline for that
metric is automatically raised — the bar can only go up.

A failing check does NOT have to block the commit.  Two escape hatches:
  • Set env var  TQDB_PERF_SKIP=1   — records a warning, exits 0
  • Pass flag    --force             — same effect
  (Both are intentional override paths for feature-tradeoff commits.)

Typical usage:
    # Run automatically by the pre-commit hook:
    python benchmarks/precommit_perf_check.py

    # Seed / hard-reset the baseline to current measured values:
    python benchmarks/precommit_perf_check.py --update-baseline

    # Force-pass a known regression (e.g. latency tradeoff for a new feature):
    TQDB_PERF_SKIP=1 git commit -m "feat: ..."
    python benchmarks/precommit_perf_check.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Metric registry — drives all comparison logic.
# Each entry: (key, higher_is_better, display_label, unit, tol_override_pct | None)
# tol_override_pct=None  → use global tolerance_pct from baseline (default 5%)
# tol_override_pct=10.0  → use 10% (timing metrics — 1-5ms have ~150µs OS jitter
#                           on Windows, making 5% = ±70µs sub-resolution)
# ---------------------------------------------------------------------------
# Number of repetition rounds to take the best-of for noisy timing metrics.
# Higher = more reproducible floor at cost of longer pre-commit runtime.
INGEST_ROUNDS = 3    # best throughput of N full ingest passes
QUERY_ROUNDS  = 10   # per-query min across N rounds (more = less OS jitter)
CHUNK_SIZE    = 2000  # insert_batch chunk size (matches paper_recall_bench.py)

# GloVe cache (same path as paper_recall_bench.py)
CACHE_DIR = Path(__file__).parent / "_paper_bench_cache"


METRIC_DEFS: list[tuple[str, bool, str, str, float | None]] = [
    # Quality — paper Recall@1@k — deterministic, 5% gate
    ("recall_1_at_1",            True,  "recall@1@1            ", "",    None),
    ("recall_1_at_4",            True,  "recall@1@4            ", "",    None),
    ("recall_1_at_8",            True,  "recall@1@8            ", "",    None),
    # Query latency — 10% gate (OS scheduling jitter on 1-3ms queries)
    ("avg_latency_ms",           False, "query avg latency      ", "ms", 10.0),
    ("p50_latency_ms",           False, "query p50 latency      ", "ms", 10.0),
    ("p95_latency_ms",           False, "query p95 latency      ", "ms", 10.0),
    # Ingest — displayed only, not gated (sub-second measurements too noisy on Windows)
    # ("insert_throughput_vps",    True,  "ingest throughput      ", "vps",10.0),
    # ("total_ingest_wall_s",      False, "ingest wall time       ", "s",  10.0),
    # Storage — disk — deterministic, 5% gate
    ("disk_bytes",               False, "disk bytes             ", "B",  None),
    ("compression_ratio",        True,  "compression ratio      ", "x",  None),
    # Storage — RAM — deterministic, 5% gate
    ("ram_estimate_bytes",       False, "RAM estimate           ", "B",  None),
    ("live_codes_bytes",         False, "live codes bytes       ", "B",  None),
    ("live_vectors_bytes",       False, "live vectors bytes     ", "B",  None),
]

METRIC_KEYS = {key for key, *_ in METRIC_DEFS}


# ---------------------------------------------------------------------------
# Perf history tracking — run all datasets/configs, append to perf_history.json
# ---------------------------------------------------------------------------

BENCH_DIR    = Path(__file__).parent
HISTORY_PATH = BENCH_DIR / "perf_history.json"

# Datasets run for history tracking (smaller n than full benchmark for speed)
HISTORY_DS_CONFIGS: list[tuple[str, int, int, int]] = [
    # (ds_label, n, d, n_queries)  — full 100k corpus
    ("glove-200",    100_000, 200,  10_000),
    ("dbpedia-1536", 100_000, 1536,  1_000),
    # dbpedia-3072 (d=3072) omitted — ~6 hrs/commit, too expensive for CI history
]

# Full config set matching perf_tracker.py CONFIGS: (bits, rerank, ann)
HISTORY_BENCH_CONFIGS: list[tuple[int, bool, bool]] = [
    (2, False, False),  # b=2 rerank=F brute
    (2, True,  False),  # b=2 rerank=T brute
    (4, False, False),  # b=4 rerank=F brute
    (4, True,  False),  # b=4 rerank=T brute
    (2, False, True),   # b=2 rerank=F ANN
    (2, True,  True),   # b=2 rerank=T ANN
    (4, False, True),   # b=4 rerank=F ANN
    (4, True,  True),   # b=4 rerank=T ANN
]

# ANN index params (consistent with "Balanced" preset in README)
ANN_MAX_DEGREE      = 32
ANN_EF_CONSTRUCTION = 200
ANN_N_REFINEMENTS   = 5
ANN_SEARCH_LIST     = 200

HISTORY_QUERY_ROUNDS = 3  # fewer rounds than gate check; just for trending


def _git_info() -> tuple[str, str]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, cwd=BENCH_DIR.parent, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"
    return (
        _run(["git", "rev-parse", "--short", "HEAD"]),
        _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    )


def _config_key(bits: int, rerank: bool, ann: bool = False) -> str:
    """Key prefix matching perf_tracker.py CONFIGS format."""
    rr = "rerankT" if rerank else "rerankF"
    mode = "ANN" if ann else "brute"
    return f"b{bits}_{rr}_{mode}"


def _run_one_config_for_history(
    corpus: "np.ndarray",
    qs: "np.ndarray",
    true_top1s: list[str],
    bits: int,
    rerank: bool,
    ann: bool = False,
) -> dict:
    """Run one config, return metrics matching perf_history.json key suffixes."""
    import tqdb as tq  # noqa: PLC0415

    n, d = corpus.shape
    ids = [f"vec_{i}" for i in range(n)]
    max_k = 8

    # Lazy-import CpuRamSampler for realistic RSS-delta RAM measurement
    _CpuRamSampler = None
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("bench_core", Path(__file__).parent / "bench_core.py")
        if _spec is not None:
            _bc = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_bc)  # type: ignore[union-attr]
            _CpuRamSampler = _bc.CpuRamSampler
    except Exception:
        pass

    # Ingest pass — single round (trending, not gating)
    with tempfile.TemporaryDirectory() as db_dir:
        sampler = _CpuRamSampler() if _CpuRamSampler else None
        try:
            if sampler:
                sampler.start()
            db = tq.Database.open(db_dir, dimension=d, bits=bits, rerank=rerank, fast_mode=True, metric="ip")
            t0 = time.perf_counter()
            for start in range(0, n, CHUNK_SIZE):
                db.insert_batch(ids[start:start + CHUNK_SIZE], corpus[start:start + CHUNK_SIZE])
            db.flush()
            ingest_s = time.perf_counter() - t0
            # Use deterministic structural estimate — avoids RSS noise from GC/mmap faulting
            ram_mb = db.stats()["ram_estimate_bytes"] / (1 << 20)
            db.close()
            # Reopen then close to trim pre-allocated mmap capacity before measuring disk
            tq.Database.open(db_dir, dimension=d, bits=bits, rerank=rerank, fast_mode=True, metric="ip").close()
            disk_mb = sum(
                p.stat().st_size for p in Path(db_dir).iterdir() if p.is_file()
            ) / (1 << 20)
        finally:
            if sampler:
                sampler.stop()

    # Query pass (with optional ANN index)
    with tempfile.TemporaryDirectory() as db_dir:
        db = tq.Database.open(db_dir, dimension=d, bits=bits, rerank=rerank, fast_mode=True, metric="ip")
        for start in range(0, n, CHUNK_SIZE):
            db.insert_batch(ids[start:start + CHUNK_SIZE], corpus[start:start + CHUNK_SIZE])
        db.flush()
        if ann:
            db.create_index(
                max_degree=ANN_MAX_DEGREE,
                ef_construction=ANN_EF_CONSTRUCTION,
                n_refinements=ANN_N_REFINEMENTS,
            )

        min_lats = [float("inf")] * len(qs)
        last_returned: list[list[str]] = []
        for rnd in range(HISTORY_QUERY_ROUNDS):
            round_ret: list[list[str]] = []
            for j, q in enumerate(qs):
                t0 = time.perf_counter()
                results = db.search(
                    q, top_k=max_k,
                    _use_ann=ann,
                    ann_search_list_size=ANN_SEARCH_LIST if ann else None,
                )
                min_lats[j] = min(min_lats[j], (time.perf_counter() - t0) * 1000)
                round_ret.append([r["id"] for r in results])
            if rnd == HISTORY_QUERY_ROUNDS - 1:
                last_returned = round_ret
        db.close()

    lats = np.array(min_lats)
    r1at1 = float(np.mean([
        1.0 if true_top1s[j] in last_returned[j][:1] else 0.0
        for j in range(len(qs))
    ]))
    mrr_vals = []
    for j in range(len(qs)):
        for rank, rid in enumerate(last_returned[j], 1):
            if rid == true_top1s[j]:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)

    return {
        "r1at1":          round(r1at1, 4),
        "throughput":     round(n / ingest_s),
        "p50_ms":         round(float(np.percentile(lats, 50)), 3),
        "disk_mb":        round(disk_mb, 2),
        "ram_estimate_mb": round(ram_mb, 1),
        "mrr":            round(float(np.mean(mrr_vals)), 4),
    }


def _append_perf_history() -> None:
    """Run all tracked datasets/configs and append one entry to perf_history.json."""
    try:
        import tqdb  # noqa: F401, PLC0415
    except ImportError:
        print("[pre-commit] tqdb not installed — skipping history tracking", flush=True)
        return

    commit, branch = _git_info()
    try:
        import importlib.metadata
        version = importlib.metadata.version("tqdb")
    except Exception:
        version = "unknown"

    from datetime import datetime, timezone
    entry: dict = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "version":    version,
        "source":     "precommit",
        "results":    {},
    }

    # Cache dir is gitignored and survives git checkouts — safe to read at any commit
    _CACHE_DIR = Path(__file__).parent / "_paper_bench_cache"

    for ds_label, n, d, n_queries in HISTORY_DS_CONFIGS:
        print(f"  [history] {ds_label}  n={n:,}  d={d}", flush=True)
        corpus: "np.ndarray"
        qs: "np.ndarray"
        if ds_label.startswith("dbpedia-"):
            dim = int(ds_label.split("-")[1])
            tag = f"dbpedia{dim}"
            _ckpt_vecs  = _CACHE_DIR / f"{tag}_100000_vecs.npy"
            _ckpt_qvecs = _CACHE_DIR / f"{tag}_1000_qvecs.npy"
            if _ckpt_vecs.exists() and _ckpt_qvecs.exists():
                corpus = np.load(_ckpt_vecs)[:n]
                qs     = np.load(_ckpt_qvecs)[:n_queries]
                print(f"  data: DBpedia-{dim} ({corpus.shape[0]:,} corpus, {qs.shape[0]} queries)", flush=True)
            else:
                corpus, qs, _ = load_data(n, d, n_queries, seed=42)
        else:
            corpus, qs, _ = load_data(n, d, n_queries, seed=42)
        true_top1s = [f"vec_{int(np.argmax(corpus @ q))}" for q in qs]
        ds_snap: dict = {}
        for bits, rerank, ann in HISTORY_BENCH_CONFIGS:
            key = _config_key(bits, rerank, ann)
            print(f"    {key} ...", flush=True)
            metrics = _run_one_config_for_history(corpus, qs, true_top1s, bits, rerank, ann)
            for suffix, val in metrics.items():
                ds_snap[f"{key}_{suffix}"] = val
        entry["results"][ds_label] = ds_snap

    history: list = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"  [history] Appended → {HISTORY_PATH} ({len(history)} entries)", flush=True)

    tracker_path = BENCH_DIR / "perf_tracker.py"
    if tracker_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("perf_tracker", tracker_path)
            pt = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(pt)                  # type: ignore[union-attr]
            pt.generate_html_plotly(pt.load_history(HISTORY_PATH), BENCH_DIR / "_perf_history.html")
        except Exception as exc:
            print(f"  [history] Warning: HTML not regenerated: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(n: int, d: int, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load GloVe-200 corpus/query vectors from the paper bench cache when
    available (same path as paper_recall_bench.py), otherwise fall back to
    synthetic normalised Gaussian vectors with dimension d.
    Accepts any n ≤ 100 000 by slicing the 100k cache file.
    """
    vecs_exact  = CACHE_DIR / f"glove200_{n}_vecs.npy"
    vecs_100k   = CACHE_DIR / "glove200_100000_vecs.npy"
    qvecs_path  = CACHE_DIR / "glove200_10000_qvecs.npy"
    vecs_path   = vecs_exact if vecs_exact.exists() else (vecs_100k if vecs_100k.exists() else None)
    if vecs_path and qvecs_path.exists():
        corpus = np.load(str(vecs_path))[:n].astype(np.float32)
        all_q  = np.load(str(qvecs_path)).astype(np.float32)
        rng    = np.random.default_rng(seed)
        idx    = rng.choice(len(all_q), n_queries, replace=False)
        qs     = all_q[idx]
        print(f"  data: GloVe-200 ({corpus.shape[0]:,} corpus, {qs.shape[0]} queries)", flush=True)
    else:
        rng    = np.random.default_rng(seed)
        corpus = rng.standard_normal((n, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        qs     = rng.standard_normal((n_queries, d)).astype(np.float32)
        qs     /= np.linalg.norm(qs, axis=1, keepdims=True)
        print(f"  data: synthetic d={d} ({n:,} corpus, {n_queries} queries)", flush=True)
    return corpus, qs, [f"vec_{i}" for i in range(len(corpus))]


def _fmt(value: float, unit: str) -> str:
    if unit == "B":
        if value >= 1 << 20:
            return f"{value / (1 << 20):.2f} MB"
        if value >= 1 << 10:
            return f"{value / 1024:.1f} KB"
        return f"{value:.0f} B"
    if unit == "vps":
        return f"{value:,.0f} vps"
    if unit in ("ms", "s"):
        return f"{value:.3f}{unit}"
    if unit == "x":
        return f"{value:.3f}x"
    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(config: dict) -> dict:
    """
    Paper-methodology benchmark:
      - GloVe-200 cache if available, else synthetic
      - insert_batch in chunks of CHUNK_SIZE (matches paper_recall_bench.py)
      - rerank=False, metric="ip"
      - Recall@1@k for k in k_values
      - Best-of INGEST_ROUNDS / QUERY_ROUNDS for timing stability
    """
    n, d, bits  = config["n"], config["d"], config["bits"]
    k_values    = config.get("k_values", [1, 4, 8])
    n_queries   = config["queries"]
    seed        = config["seed"]

    corpus, qs, ids = load_data(n, d, n_queries, seed)
    d = corpus.shape[1]  # may differ from config if GloVe was loaded
    raw_bytes = corpus.nbytes

    # Precompute ground-truth top-1 for each query
    true_top1s = [f"vec_{int(np.argmax(corpus @ q))}" for q in qs]

    try:
        import tqdb as tq
    except ImportError:
        print("[pre-commit] tqdb not installed — run: maturin develop --release")
        sys.exit(0)

    # ── ingest: median of INGEST_ROUNDS ───────────────────────────────────
    # Median (not min) is more representative for short ingest times on
    # Windows where one lucky machine-idle round can skew the "best ever".
    ingest_walls: list[float] = []
    stats = None

    print(f"  ingest ({INGEST_ROUNDS} rounds, median) ...", flush=True)
    for _ in range(INGEST_ROUNDS):
        with tempfile.TemporaryDirectory() as db_dir:
            db = tq.Database.open(db_dir, dimension=d, bits=bits, rerank=False, fast_mode=True, metric="ip")
            wall_start = time.perf_counter()
            for start in range(0, n, CHUNK_SIZE):
                db.insert_batch(ids[start:start + CHUNK_SIZE],
                                corpus[start:start + CHUNK_SIZE])
            db.flush()
            ingest_walls.append(time.perf_counter() - wall_start)
            stats = db.stats()  # RAM metrics — capture before close
            db.close()  # trims GROW_SLOTS pre-allocation from live_codes.bin
            # measure disk post-close so pre-allocated capacity is excluded
            stats["total_disk_bytes"] = sum(
                p.stat().st_size for p in Path(db_dir).iterdir() if p.is_file()
            )

    # ── query: per-query minimum latency across QUERY_ROUNDS ───────────────
    # Taking the per-query minimum (not the best-round p50) eliminates
    # Windows OS-scheduling spikes from p50/p95 without sacrificing rounds.
    print(f"  queries ({QUERY_ROUNDS} rounds, per-query min) ...", flush=True)
    max_k = max(k_values)
    min_lats = [float("inf")] * len(qs)
    recalls_by_k: dict[int, list[float]] = {k: [] for k in k_values}

    with tempfile.TemporaryDirectory() as db_dir:
        db = tq.Database.open(db_dir, dimension=d, bits=bits, rerank=False, fast_mode=True, metric="ip")
        for start in range(0, n, CHUNK_SIZE):
            db.insert_batch(ids[start:start + CHUNK_SIZE],
                            corpus[start:start + CHUNK_SIZE])
        db.flush()

        for rnd in range(QUERY_ROUNDS):
            for j, (q, top1_id) in enumerate(zip(qs, true_top1s)):
                t0 = time.perf_counter()
                results = db.search(q, top_k=max_k)
                elapsed = (time.perf_counter() - t0) * 1000
                min_lats[j] = min(min_lats[j], elapsed)
                # Recalls are deterministic; record once from last round
                if rnd == QUERY_ROUNDS - 1:
                    returned_ids = [r["id"] for r in results]
                    for k in k_values:
                        recalls_by_k[k].append(1.0 if top1_id in returned_ids[:k] else 0.0)

        db.close()

    lats_arr   = np.array(min_lats)
    disk_bytes = float(stats["total_disk_bytes"])
    compression_ratio = raw_bytes / disk_bytes if disk_bytes > 0 else 0.0

    return {
        # quality — paper Recall@1@k
        "recall_1_at_1": float(np.mean(recalls_by_k[1])),
        "recall_1_at_4": float(np.mean(recalls_by_k[4])),
        "recall_1_at_8": float(np.mean(recalls_by_k[8])),
        # query latency (best round)
        "avg_latency_ms":           float(np.mean(lats_arr)),
        "p50_latency_ms":           float(np.percentile(lats_arr, 50)),
        "p95_latency_ms":           float(np.percentile(lats_arr, 95)),
        # ingest (median of rounds — resistant to lucky machine-idle outliers)
        "insert_throughput_vps":    float(n / float(np.median(ingest_walls))),
        "total_ingest_wall_s":      float(np.median(ingest_walls)),
        "ingest_cpu_s":             0.0,  # informational only, not gated
        # disk (deterministic)
        "disk_bytes":               disk_bytes,
        "compression_ratio":        compression_ratio,
        # RAM (deterministic)
        "ram_estimate_bytes":       float(stats["ram_estimate_bytes"]),
        "live_codes_bytes":         float(stats["live_codes_bytes"]),
        "live_vectors_bytes":       float(stats.get("live_vectors_bytes_estimate", 0)),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    measured: dict,
    metrics_doc: dict,
    tol: float,
) -> tuple[list[str], dict]:
    """
    Compare measured values against stored bests.

    Returns:
        failures       — list of human-readable regression messages
        updated_metrics — copy of metrics_doc with raised bests where applicable
    """
    failures: list[str] = []
    updated = {k: dict(v) for k, v in metrics_doc.items()}

    for key, higher_is_better, label, unit, tol_override in METRIC_DEFS:
        if key not in metrics_doc:
            continue
        m = measured[key]
        b = metrics_doc[key]["best"]
        metric_tol = tol_override if tol_override is not None else tol
        if b is None:
            updated[key]["best"] = m
            print(f"  {label}  {_fmt(m, unit)}  (no prior best — recorded)")
            continue

        delta_pct = (m - b) / b * 100.0 if b != 0 else 0.0

        if higher_is_better:
            floor = b * (1 - metric_tol / 100)
            if m > b:
                status = "↑ new best"
                updated[key]["best"] = m
            elif m >= floor:
                status = "✓"
            else:
                status = "✗"
                failures.append(
                    f"{key}: {_fmt(m, unit)} < floor {_fmt(floor, unit)}  "
                    f"(Δ{delta_pct:.1f}% — limit −{metric_tol}%)"
                )
            print(
                f"  {label}  {_fmt(m, unit)}  "
                f"(best {_fmt(b, unit)}, Δ{delta_pct:+.1f}%, floor {_fmt(floor, unit)})  {status}"
            )
        else:
            ceil = b * (1 + metric_tol / 100)
            if m < b:
                status = "↑ new best"
                updated[key]["best"] = m
            elif m <= ceil:
                status = "✓"
            else:
                status = "✗"
                failures.append(
                    f"{key}: {_fmt(m, unit)} > ceil {_fmt(ceil, unit)}  "
                    f"(Δ{delta_pct:+.1f}% — limit +{metric_tol}%)"
                )
            print(
                f"  {label}  {_fmt(m, unit)}  "
                f"(best {_fmt(b, unit)}, Δ{delta_pct:+.1f}%, ceil {_fmt(ceil, unit)})  {status}"
            )

    return failures, updated


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def _default_metrics_doc() -> dict:
    """Skeleton metrics document with all known metrics and best=null."""
    return {
        key: {"higher_is_better": hib, "best": None}
        for key, hib, *_ in METRIC_DEFS
    }


def load_doc(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
        # Back-fill any metrics added after the file was last written
        metrics = doc.setdefault("metrics", _default_metrics_doc())
        for key, hib, *_ in METRIC_DEFS:
            metrics.setdefault(key, {"higher_is_better": hib, "best": None})
        return doc
    return {
        "description": (
            "Best-ever local perf for each metric. Bar only goes up. "
            "Commits regressing >5% from any best are blocked "
            "unless --force / TQDB_PERF_SKIP=1."
        ),
        "config": {"n": 10000, "d": 200, "bits": 4, "k_values": [1, 4, 8], "queries": 100, "seed": 42},
        "tolerance_pct": 5.0,
        "metrics": _default_metrics_doc(),
    }


def save_doc(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-commit perf regression check")
    p.add_argument("--baseline", default="benchmarks/perf_baseline.json")
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help="Hard-reset ALL metric bests to current measured values",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Exit 0 even on regression (records override; use for intentional tradeoffs)",
    )
    p.add_argument(
        "--track",
        action="store_true",
        help="Run full paper benchmark (N=100k, ~20 min) and append to perf_history.json. "
             "Use before merging to main. Also enabled by TQDB_TRACK=1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    force = args.force or os.environ.get("TQDB_PERF_SKIP", "").strip() not in ("", "0")

    if not baseline_path.exists() and not args.update_baseline:
        print(f"[pre-commit] baseline not found: {baseline_path}")
        print("[pre-commit] Seed it with: python benchmarks/precommit_perf_check.py --update-baseline")
        sys.exit(1)

    doc = load_doc(baseline_path)
    config = doc["config"]
    tol = float(doc.get("tolerance_pct", 5.0))

    print(
        f"[pre-commit] perf check  "
        f"n={config['n']:,}  d={config['d']}  bits={config['bits']}  "
        f"k_values={config.get('k_values', [1,4,8])}  queries={config['queries']}  tol={tol}%"
    )

    measured = run_benchmark(config)

    # Hard reset — overwrite every best regardless of direction
    if args.update_baseline:
        for key in doc["metrics"]:
            if key in measured:
                doc["metrics"][key]["best"] = measured[key]
        save_doc(baseline_path, doc)
        print(f"[pre-commit] baseline reset → {baseline_path}")
        for key, _, label, unit, _tol in METRIC_DEFS:
            if key in measured:
                print(f"  {label}  {_fmt(measured[key], unit)}")
        # informational ingest stats
        print(f"  {'ingest throughput':22}  {measured['insert_throughput_vps']:,.0f} vps  (informational, not gated)")
        print(f"  {'ingest wall time':22}  {measured['total_ingest_wall_s']:.3f}s  (median of {INGEST_ROUNDS} rounds, not gated)")
        return

    failures, updated_metrics = compare(measured, doc["metrics"], tol)

    # Informational ingest stats (not gated — too noisy at sub-second scale)
    print(
        f"  {'ingest throughput':<22}  {measured['insert_throughput_vps']:,.0f} vps  "
        f"  {'ingest wall':<11}  {measured['total_ingest_wall_s']:.3f}s  "
        "(median, not gated)"
    )

    # Persist raised bests
    if updated_metrics != doc["metrics"]:
        doc["metrics"] = updated_metrics
        save_doc(baseline_path, doc)
        print(f"[pre-commit] ↑ baseline raised → stage: git add {baseline_path}")

    # History tracking (paper benchmark, N=100k, ~20 min) is opt-in.
    # Set TQDB_TRACK=1 or pass --track to run it.  Never runs automatically
    # in the pre-commit hook — call manually before merging to main:
    #   TQDB_TRACK=1 python benchmarks/precommit_perf_check.py
    track = getattr(args, "track", False) or os.environ.get("TQDB_TRACK", "").strip() not in ("", "0")

    if not failures:
        print("[pre-commit] ✓ all metrics within 5% of best")
        if track:
            _append_perf_history()
        return

    print("\n[pre-commit] FAIL — regression(s) detected:")
    for msg in failures:
        print(f"  ✗ {msg}")

    if force:
        print(
            "\n[pre-commit] ⚠ override active (--force / TQDB_PERF_SKIP=1) — "
            "proceeding despite regression"
        )
        if track:
            _append_perf_history()
        return

    print(
        "\n  Intentional tradeoff?  Use one of:\n"
        "    TQDB_PERF_SKIP=1 git commit ...\n"
        "    python benchmarks/precommit_perf_check.py --force\n"
        "  Reset baseline to current values:\n"
        "    python benchmarks/precommit_perf_check.py --update-baseline"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
