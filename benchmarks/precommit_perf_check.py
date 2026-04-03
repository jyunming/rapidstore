"""
TurboQuantDB pre-commit performance check.

Runs a minimal benchmark (n=500, dim=128, bits=4) and compares against
the committed baseline. Fails with a non-zero exit code if any metric
regresses beyond its threshold.

Thresholds:
  ingest throughput   must stay >= 85% of baseline
  search p50 latency  must stay <= 120% of baseline
  recall@10           must not drop more than 3 percentage points

Usage:
    # Check against committed baseline
    python benchmarks/precommit_perf_check.py

    # Regenerate the baseline (run after a deliberate performance improvement)
    python benchmarks/precommit_perf_check.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BASELINE_PATH = Path(__file__).parent / "perf_baseline.json"

# Benchmark parameters — kept small so the hook completes in < 5 s
N = 500
D = 128
BITS = 4
K = 10
QUERIES = 20
SEED = 42

# Regression thresholds
INGEST_MIN_RATIO = 0.85   # throughput must stay >= 85 % of baseline
LATENCY_MAX_RATIO = 1.20  # p50 latency must stay <= 120 % of baseline
RECALL_MIN_DELTA = -0.03  # recall must not drop more than 3 pp


def exact_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> set[str]:
    scores = corpus @ query
    top = np.argsort(scores)[::-1][:k]
    return {f"v{i}" for i in top}


def run_mini_bench() -> dict:
    import turboquantdb as tq

    rng = np.random.default_rng(SEED)
    corpus = rng.standard_normal((N, D))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((QUERIES, D))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as db_dir:
        db = tq.Database.open(db_dir, dimension=D, bits=BITS)

        t0 = time.perf_counter()
        ids = [f"v{i}" for i in range(N)]
        db.insert_batch(ids, corpus)
        ingest_sec = time.perf_counter() - t0
        ingest_tps = N / ingest_sec

        latencies_ms = []
        recalls = []
        for q in queries:
            gt = exact_topk(corpus, q, K)
            t0 = time.perf_counter()
            results = db.search(q, top_k=K)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            retrieved = {r["id"] for r in results}
            recalls.append(len(gt & retrieved) / K)

    latencies_ms.sort()
    p50_ms = latencies_ms[len(latencies_ms) // 2]

    return {
        "n": N, "d": D, "bits": BITS, "k": K, "queries": QUERIES, "seed": SEED,
        "ingest_vec_per_sec": round(ingest_tps, 1),
        "search_p50_ms": round(p50_ms, 4),
        "recall_at_10": round(float(np.mean(recalls)), 4),
    }


def check_regression(current: dict, baseline: dict) -> list[str]:
    failures = []

    b_ingest = baseline["ingest_vec_per_sec"]
    c_ingest = current["ingest_vec_per_sec"]
    if c_ingest < b_ingest * INGEST_MIN_RATIO:
        pct = (c_ingest / b_ingest - 1) * 100
        failures.append(
            f"  ingest throughput regressed {pct:.1f}%: "
            f"{c_ingest:.0f} vec/s vs baseline {b_ingest:.0f} vec/s "
            f"(threshold ≥ {INGEST_MIN_RATIO*100:.0f}%)"
        )

    b_p50 = baseline["search_p50_ms"]
    c_p50 = current["search_p50_ms"]
    if c_p50 > b_p50 * LATENCY_MAX_RATIO:
        pct = (c_p50 / b_p50 - 1) * 100
        failures.append(
            f"  search p50 latency regressed +{pct:.1f}%: "
            f"{c_p50:.3f} ms vs baseline {b_p50:.3f} ms "
            f"(threshold ≤ {LATENCY_MAX_RATIO*100:.0f}%)"
        )

    b_recall = baseline["recall_at_10"]
    c_recall = current["recall_at_10"]
    if c_recall < b_recall + RECALL_MIN_DELTA:
        delta = (c_recall - b_recall) * 100
        failures.append(
            f"  recall@10 dropped {delta:.1f} pp: "
            f"{c_recall:.3f} vs baseline {b_recall:.3f} "
            f"(threshold drop ≤ {abs(RECALL_MIN_DELTA)*100:.0f} pp)"
        )

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="TurboQuantDB pre-commit perf check")
    ap.add_argument(
        "--update-baseline", action="store_true",
        help="Regenerate perf_baseline.json from current measurements and exit 0",
    )
    ap.add_argument(
        "--baseline", default=str(BASELINE_PATH),
        help="Path to baseline JSON (default: benchmarks/perf_baseline.json)",
    )
    args = ap.parse_args()

    try:
        import turboquantdb  # noqa: F401
    except ImportError:
        print("[perf-check] turboquantdb not installed — skipping (run maturin develop --release)")
        sys.exit(0)

    print(f"[perf-check] n={N} d={D} bits={BITS} k={K} queries={QUERIES}")
    current = run_mini_bench()
    print(
        f"[perf-check] ingest={current['ingest_vec_per_sec']:.0f} vec/s  "
        f"p50={current['search_p50_ms']:.3f} ms  "
        f"recall={current['recall_at_10']:.3f}"
    )

    baseline_path = Path(args.baseline)

    if args.update_baseline:
        baseline_path.write_text(json.dumps(current, indent=2) + "\n")
        print(f"[perf-check] Baseline updated → {baseline_path}")
        sys.exit(0)

    if not baseline_path.exists():
        print(f"[perf-check] No baseline found at {baseline_path}; saving current as baseline.")
        baseline_path.write_text(json.dumps(current, indent=2) + "\n")
        print("[perf-check] PASS (baseline created)")
        sys.exit(0)

    baseline = json.loads(baseline_path.read_text())
    print(
        f"[perf-check] baseline: ingest={baseline['ingest_vec_per_sec']:.0f} vec/s  "
        f"p50={baseline['search_p50_ms']:.3f} ms  "
        f"recall={baseline['recall_at_10']:.3f}"
    )

    failures = check_regression(current, baseline)
    if failures:
        print("[perf-check] FAIL — performance regression detected:")
        for msg in failures:
            print(msg)
        print()
        print("  Fix the regression, or run:")
        print("    python benchmarks/precommit_perf_check.py --update-baseline")
        print("  if the change is intentional.")
        sys.exit(1)

    print("[perf-check] PASS")


if __name__ == "__main__":
    main()
