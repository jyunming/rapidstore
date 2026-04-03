"""
CI quality gate for TurboQuantDB benchmark artifacts.

Reads:
  - Python recall benchmark artifact JSON (latest matching prefix)
  - Rust bench_search JSON artifact

Fails (non-zero exit) if configured quality thresholds are not met.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--python-artifacts-dir", default="benchmarks/artifacts")
    p.add_argument("--python-prefix", default="ci_")
    p.add_argument("--rust-artifact", default="target/benchmarks/bench_search.json")

    p.add_argument("--min-recall", type=float, default=0.60)
    p.add_argument("--max-tq-latency-ms", type=float, default=100.0)
    p.add_argument("--min-speedup-vs-numpy", type=float, default=0.20)
    p.add_argument("--max-ann-to-bruteforce-ratio", type=float, default=1.50)
    return p.parse_args()


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def load_latest_python_artifact(artifacts_dir: str, prefix: str) -> dict:
    pattern = str(Path(artifacts_dir) / f"{prefix}*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        fail(f"No Python benchmark artifacts found with pattern: {pattern}")
    latest = max(candidates, key=os.path.getmtime)
    print(f"[INFO] Using Python artifact: {latest}")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rust_artifact(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        fail(f"Rust benchmark artifact not found: {path}")
    print(f"[INFO] Using Rust artifact: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    py = load_latest_python_artifact(args.python_artifacts_dir, args.python_prefix)
    status = py.get("status")
    if status != "ok":
        fail(f"Python benchmark status must be 'ok', got: {status!r}")

    recall = py.get("recall_at_k")
    tq_ms = py.get("turboquant_avg_ms")
    speedup = py.get("latency_speedup_vs_numpy")

    if recall is None:
        fail("Python artifact missing 'recall_at_k'")
    if tq_ms is None:
        fail("Python artifact missing 'turboquant_avg_ms'")
    if speedup is None:
        fail("Python artifact missing 'latency_speedup_vs_numpy'")

    if recall < args.min_recall:
        fail(f"Recall below threshold: {recall:.4f} < {args.min_recall:.4f}")
    if tq_ms > args.max_tq_latency_ms:
        fail(f"TurboQuant latency above threshold: {tq_ms:.4f}ms > {args.max_tq_latency_ms:.4f}ms")
    if speedup < args.min_speedup_vs_numpy:
        fail(f"Speedup below threshold: {speedup:.4f} < {args.min_speedup_vs_numpy:.4f}")

    rust = load_rust_artifact(args.rust_artifact)
    ann_ms = rust.get("ann_avg_ms")
    brute_ms = rust.get("brute_ms_pre_compaction")
    if ann_ms is None or brute_ms is None:
        fail("Rust artifact missing 'ann_avg_ms' or 'brute_ms_pre_compaction'")
    if brute_ms <= 0:
        fail(f"Invalid brute-force baseline latency: {brute_ms}")

    ratio = ann_ms / brute_ms
    if ratio > args.max_ann_to_bruteforce_ratio:
        fail(
            "ANN latency ratio too high: "
            f"{ratio:.4f} > {args.max_ann_to_bruteforce_ratio:.4f} "
            f"(ann={ann_ms:.4f}ms, brute={brute_ms:.4f}ms)"
        )

    print("[PASS] Benchmark quality gates satisfied")
    print(
        "[INFO] "
        f"recall={recall:.4f}, tq_ms={tq_ms:.4f}, speedup={speedup:.4f}, "
        f"ann/brute={ratio:.4f}"
    )


if __name__ == "__main__":
    main()
