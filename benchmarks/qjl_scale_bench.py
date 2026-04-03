"""
QJL scale bug: before/after recall benchmark.

Tests multiple (d, bits) combos at n=5000 vectors, 50 queries.
Saves results to benchmarks/artifacts/qjl_scale_{label}_{ts}.json
"""

import argparse
import json
import os
import sys
import time
import tempfile
from datetime import datetime, timezone

import numpy as np

CONFIGS = [
    # (n,     d,    bits, label)
    (5_000,   128,  2,    "d128-b2"),
    (5_000,   128,  4,    "d128-b4"),
    (5_000,   768,  2,    "d768-b2"),
    (5_000,   768,  4,    "d768-b4"),
    (5_000,  1536,  2,   "d1536-b2"),
    (5_000,  1536,  4,   "d1536-b4"),
]
K = 10
QUERIES = 50
SEED = 42


def exact_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> set:
    scores = corpus @ query
    return {f"v{i}" for i in np.argsort(scores)[::-1][:k]}


def run_config(n, d, bits, label, db_mod):
    rng = np.random.default_rng(SEED)
    corpus = rng.standard_normal((n, d)).astype(np.float64)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((QUERIES, d)).astype(np.float64)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as db_dir:
        db = db_mod.Database.open(db_dir, dimension=d, bits=bits, metric="ip")
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, corpus)

        recalls = []
        latencies = []
        for q in queries:
            gt = exact_topk(corpus, q, K)
            t0 = time.perf_counter()
            res = db.search(q, top_k=K)
            latencies.append((time.perf_counter() - t0) * 1000)
            retrieved = {r["id"] for r in res}
            recalls.append(len(gt & retrieved) / K)

        db.close()

    recall = float(np.mean(recalls))
    lat_ms = float(np.mean(latencies))
    print(f"  {label:12s}  bits={bits}  d={d:5d}  recall={recall:.3f}  lat={lat_ms:.2f}ms")
    return {"label": label, "n": n, "d": d, "bits": bits, "recall": recall, "lat_ms": lat_ms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="before", help="Label for output: 'before' or 'after'")
    args = p.parse_args()

    try:
        import turboquantdb as tq
    except ImportError:
        print("ERROR: turboquantdb not installed. Run `maturin develop --release` first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  QJL Scale Benchmark  [{args.tag.upper()}]")
    print(f"  n=5000, k={K}, queries={QUERIES}")
    print(f"{'='*60}\n")

    results = []
    for n, d, bits, label in CONFIGS:
        r = run_config(n, d, bits, label, tq)
        results.append(r)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qjl_scale_{args.tag}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, "results": results, "generated_at": ts}, f, indent=2)
    print(f"\nSaved → {out_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY [{args.tag.upper()}]")
    print(f"{'='*60}")
    print(f"  {'Config':<14} {'Recall':>8}")
    for r in results:
        print(f"  {r['label']:<14} {r['recall']:>8.3f}")
    avg = np.mean([r["recall"] for r in results])
    print(f"  {'AVERAGE':<14} {avg:>8.3f}")
    print()


if __name__ == "__main__":
    main()
