"""
Pre-commit performance regression check for TurboQuantDB.

Tracks ALL metrics across every dimension of performance:
  recall, query latency (avg/p50/p95), speedup vs numpy,
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
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Metric registry — drives all comparison logic.
# Each entry: (key, higher_is_better, display_label, unit)
# ---------------------------------------------------------------------------
# Number of repetition rounds to take the best-of for noisy timing metrics.
# Higher = more reproducible floor at cost of longer pre-commit runtime.
INGEST_ROUNDS = 3   # best throughput of N full ingest passes
QUERY_ROUNDS  = 5   # best-round p50 of N query passes (more = less OS noise)


METRIC_DEFS: list[tuple[str, bool, str, str]] = [
    # Quality
    ("recall_at_k",              True,  "recall@k              ", ""),
    # Query latency
    ("avg_latency_ms",           False, "query avg latency      ", "ms"),
    ("p50_latency_ms",           False, "query p50 latency      ", "ms"),
    ("p95_latency_ms",           False, "query p95 latency      ", "ms"),
    ("latency_speedup_vs_numpy", True,  "speedup vs numpy       ", "x"),
    # Ingest
    ("insert_throughput_vps",    True,  "ingest throughput      ", "vps"),
    ("total_ingest_wall_s",      False, "ingest wall time       ", "s"),
    ("ingest_cpu_s",             False, "ingest CPU time        ", "s"),
    # Storage — disk
    ("disk_bytes",               False, "disk bytes             ", "B"),
    ("compression_ratio",        True,  "compression ratio      ", "x"),
    # Storage — RAM
    ("ram_estimate_bytes",       False, "RAM estimate           ", "B"),
    ("live_codes_bytes",         False, "live codes bytes       ", "B"),
    ("live_vectors_bytes",       False, "live vectors bytes     ", "B"),
]

METRIC_KEYS = {key for key, *_ in METRIC_DEFS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exact_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> set[str]:
    scores = corpus @ query
    return {f"vec_{i}" for i in np.argsort(scores)[::-1][:k]}


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
    Runs INGEST_ROUNDS ingest passes and QUERY_ROUNDS query passes, reporting
    the best (peak) timing values.  Deterministic metrics (recall, disk, RAM,
    compression) are taken from the last ingest pass and are stable regardless.
    """
    n, d, bits = config["n"], config["d"], config["bits"]
    k, queries, seed = config["k"], config["queries"], config["seed"]

    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n, d)).astype(np.float64)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    qs = rng.standard_normal((queries, d)).astype(np.float64)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    raw_bytes = corpus.nbytes

    try:
        import tqdb as tq
    except ImportError:
        print("[pre-commit] tqdb not installed — run: maturin develop --release")
        sys.exit(0)

    # ── ingest: best of INGEST_ROUNDS ──────────────────────────────────────
    best_throughput = 0.0
    best_ingest_wall = float("inf")
    best_ingest_cpu  = float("inf")
    stats = None

    print(f"  ingest ({INGEST_ROUNDS} rounds, best-of) ...", flush=True)
    for _ in range(INGEST_ROUNDS):
        with tempfile.TemporaryDirectory() as db_dir:
            db = tq.Database.open(db_dir, dimension=d, bits=bits)
            cpu_start  = time.process_time()
            wall_start = time.perf_counter()
            for i, vec in enumerate(corpus):
                db.insert(f"vec_{i}", vec)
            db.flush()
            wall = time.perf_counter() - wall_start
            cpu  = time.process_time() - cpu_start
            tput = n / wall
            if tput > best_throughput:
                best_throughput  = tput
                best_ingest_wall = wall
                best_ingest_cpu  = cpu
                stats = db.stats()
            # keep DB open for query rounds on last pass after ingest rounds
            _db_for_query = db

    # ── query: best p50 across QUERY_ROUNDS ────────────────────────────────
    # Rebuild a fresh DB (deterministic) for query benchmarking
    print(f"  queries ({QUERY_ROUNDS} rounds, best p50) ...", flush=True)
    best_p50   = float("inf")
    best_lats: list[float] = []
    best_recalls: list[float] = []
    best_numpy_avg = 0.0

    with tempfile.TemporaryDirectory() as db_dir:
        db = tq.Database.open(db_dir, dimension=d, bits=bits)
        for i, vec in enumerate(corpus):
            db.insert(f"vec_{i}", vec)
        db.flush()

        for _ in range(QUERY_ROUNDS):
            numpy_lats: list[float] = []
            tq_lats: list[float] = []
            recalls: list[float] = []
            for q in qs:
                gt = exact_topk(corpus, q, k)
                t0 = time.perf_counter()
                exact_topk(corpus, q, k)
                numpy_lats.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                results = db.search(q, top_k=k)
                tq_lats.append((time.perf_counter() - t0) * 1000)
                recalls.append(len(gt & {r["id"] for r in results}) / k)
            p50 = float(np.percentile(tq_lats, 50))
            if p50 < best_p50:
                best_p50      = p50
                best_lats     = tq_lats
                best_recalls  = recalls
                best_numpy_avg = float(np.mean(numpy_lats))
        db.close()

    lats_arr = np.array(best_lats)
    avg_ms   = float(np.mean(lats_arr))
    disk_bytes = float(stats["total_disk_bytes"])
    compression_ratio = raw_bytes / disk_bytes if disk_bytes > 0 else 0.0

    return {
        # quality
        "recall_at_k":              float(np.mean(best_recalls)),
        # query latency  (best round)
        "avg_latency_ms":           avg_ms,
        "p50_latency_ms":           float(np.percentile(lats_arr, 50)),
        "p95_latency_ms":           float(np.percentile(lats_arr, 95)),
        "latency_speedup_vs_numpy": best_numpy_avg / avg_ms if avg_ms > 0 else 0.0,
        # ingest  (best round)
        "insert_throughput_vps":    float(best_throughput),
        "total_ingest_wall_s":      float(best_ingest_wall),
        "ingest_cpu_s":             float(best_ingest_cpu),
        # disk  (deterministic)
        "disk_bytes":               disk_bytes,
        "compression_ratio":        compression_ratio,
        # RAM  (deterministic)
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

    for key, higher_is_better, label, unit in METRIC_DEFS:
        if key not in metrics_doc:
            continue
        m = measured[key]
        b = metrics_doc[key]["best"]
        if b is None:
            updated[key]["best"] = m
            print(f"  {label}  {_fmt(m, unit)}  (no prior best — recorded)")
            continue

        delta_pct = (m - b) / b * 100.0 if b != 0 else 0.0

        if higher_is_better:
            floor = b * (1 - tol / 100)
            if m > b:
                status = "↑ new best"
                updated[key]["best"] = m
            elif m >= floor:
                status = "✓"
            else:
                status = "✗"
                failures.append(
                    f"{key}: {_fmt(m, unit)} < floor {_fmt(floor, unit)}  "
                    f"(Δ{delta_pct:.1f}% — limit −{tol}%)"
                )
            print(
                f"  {label}  {_fmt(m, unit)}  "
                f"(best {_fmt(b, unit)}, Δ{delta_pct:+.1f}%, floor {_fmt(floor, unit)})  {status}"
            )
        else:
            ceil = b * (1 + tol / 100)
            if m < b:
                status = "↑ new best"
                updated[key]["best"] = m
            elif m <= ceil:
                status = "✓"
            else:
                status = "✗"
                failures.append(
                    f"{key}: {_fmt(m, unit)} > ceil {_fmt(ceil, unit)}  "
                    f"(Δ{delta_pct:+.1f}% — limit +{tol}%)"
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
        "config": {"n": 10000, "d": 384, "bits": 4, "k": 10, "queries": 100, "seed": 42},
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
        f"k={config['k']}  queries={config['queries']}  tol={tol}%"
    )

    measured = run_benchmark(config)

    # Hard reset — overwrite every best regardless of direction
    if args.update_baseline:
        for key in doc["metrics"]:
            if key in measured:
                doc["metrics"][key]["best"] = measured[key]
        save_doc(baseline_path, doc)
        print(f"[pre-commit] baseline reset → {baseline_path}")
        for key, _, label, unit in METRIC_DEFS:
            if key in measured:
                print(f"  {label}  {_fmt(measured[key], unit)}")
        return

    failures, updated_metrics = compare(measured, doc["metrics"], tol)

    # Persist raised bests
    if updated_metrics != doc["metrics"]:
        doc["metrics"] = updated_metrics
        save_doc(baseline_path, doc)
        print(f"[pre-commit] ↑ baseline raised → stage: git add {baseline_path}")

    if not failures:
        print("[pre-commit] ✓ all metrics within 5% of best")
        return

    print("\n[pre-commit] FAIL — regression(s) detected:")
    for msg in failures:
        print(f"  ✗ {msg}")

    if force:
        print(
            "\n[pre-commit] ⚠ override active (--force / TQDB_PERF_SKIP=1) — "
            "proceeding despite regression"
        )
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
