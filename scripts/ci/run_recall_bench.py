"""
TurboQuantDB Recall & Latency Benchmark
========================================
Measures:
  - Recall@10 (vs exact Numpy ground truth)
  - Avg query latency (ms)
  - Insertion throughput (vectors/sec)
  - Index memory estimate

Usage:
    pip install numpy tqdb
    python benchmarks/run_recall_bench.py --n 10000 --d 1536 --bits 4
"""

import argparse
import csv
import json
from datetime import datetime, timezone
import time
import tempfile
import os
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000, help="Number of vectors")
    p.add_argument("--d", type=int, default=1536, help="Vector dimension")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4], help="TurboQuant bit width")
    p.add_argument("--k", type=int, default=10, help="Top-k for recall measurement")
    p.add_argument("--queries", type=int, default=100, help="Number of query vectors")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recall-threshold", type=float, default=0.80, help="Recall threshold gate")
    p.add_argument(
        "--enforce-recall-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce recall threshold gate (default: on; disable with --no-enforce-recall-threshold)",
    )
    p.add_argument(
        "--artifact-dir",
        type=str,
        default="benchmarks/artifacts",
        help="Directory to write benchmark artifacts",
    )
    p.add_argument(
        "--artifact-prefix",
        type=str,
        default="recall_bench",
        help="Artifact file prefix",
    )
    p.add_argument(
        "--use-ann",
        action="store_true",
        default=False,
        help="Build HNSW index before search (measures ANN recall, not brute-force recall)",
    )
    return p.parse_args()


def exact_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> set[str]:
    """Brute-force exact inner product search using Numpy."""
    scores = corpus @ query  # (N,) inner products
    top_indices = np.argsort(scores)[::-1][:k]
    return {f"vec_{i}" for i in top_indices}


def _fmt_num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_artifacts(result: dict, artifact_dir: str, artifact_prefix: str) -> dict:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{artifact_prefix}_{ts}"

    json_path = out_dir / f"{base}.json"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    csv_fields = [
        "status",
        "n",
        "d",
        "bits",
        "k",
        "queries",
        "seed",
        "numpy_avg_ms",
        "turboquant_avg_ms",
        "recall_at_k",
        "recall_threshold",
        "threshold_passed",
        "insert_throughput_vectors_per_sec",
        "compression_ratio",
        "numpy_mem_mb",
        "compressed_mem_mb",
        "latency_speedup_vs_numpy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({k: _fmt_num(result.get(k)) for k in csv_fields})

    md_lines = [
        "# TurboQuantDB Recall Benchmark Report",
        "",
        f"- Generated (UTC): `{result.get('generated_at_utc', '')}`",
        f"- Status: `{result.get('status', 'unknown')}`",
        "",
        "## Config",
        "",
        f"- n: `{result.get('n')}`",
        f"- d: `{result.get('d')}`",
        f"- bits: `{result.get('bits')}`",
        f"- k: `{result.get('k')}`",
        f"- queries: `{result.get('queries')}`",
        f"- seed: `{result.get('seed')}`",
        "",
        "## Metrics",
        "",
        f"- Numpy avg latency (ms): `{_fmt_num(result.get('numpy_avg_ms'))}`",
        f"- TurboQuant avg latency (ms): `{_fmt_num(result.get('turboquant_avg_ms'))}`",
        f"- Recall@k: `{_fmt_num(result.get('recall_at_k'))}`",
        f"- Recall threshold: `{_fmt_num(result.get('recall_threshold'))}`",
        f"- Threshold passed: `{result.get('threshold_passed')}`",
        f"- Insert throughput (vec/s): `{_fmt_num(result.get('insert_throughput_vectors_per_sec'))}`",
        f"- Compression ratio: `{_fmt_num(result.get('compression_ratio'))}`",
        "",
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def run_benchmark(args):
    print(f"\n{'='*60}")
    print(f"  TurboQuantDB Benchmark")
    print(f"  n={args.n:,}  d={args.d}  bits={args.bits}  k={args.k}  queries={args.queries}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(args.seed)

    # Generate random unit vectors (realistic embedding distribution)
    print("Generating corpus...")
    corpus = rng.standard_normal((args.n, args.d)).astype(np.float64)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus /= norms  # Unit normalize

    queries = rng.standard_normal((args.queries, args.d)).astype(np.float64)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= query_norms

    # --- Try importing TurboQuantDB ---
    try:
        import tqdb as tq
        HAS_TQ = True
    except ImportError:
        print("[WARNING] tqdb not installed. Run `maturin develop --release` first.")
        print("Showing exact Numpy baseline only.\n")
        HAS_TQ = False

    # Exact Numpy baseline
    print("Computing exact Numpy baseline...")
    numpy_latencies = []
    for query in queries:
        t0 = time.perf_counter()
        _ = exact_topk(corpus, query, args.k)
        numpy_latencies.append((time.perf_counter() - t0) * 1000)

    numpy_avg_ms = np.mean(numpy_latencies)
    numpy_mem_mb = corpus.nbytes / (1024**2)
    print(f"  Numpy exact search:  {numpy_avg_ms:.2f} ms/query  |  {numpy_mem_mb:.1f} MB")

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "numpy_only" if not HAS_TQ else "ok",
        "n": args.n,
        "d": args.d,
        "bits": args.bits,
        "k": args.k,
        "queries": args.queries,
        "seed": args.seed,
        "numpy_avg_ms": float(numpy_avg_ms),
        "numpy_mem_mb": float(numpy_mem_mb),
        "turboquant_avg_ms": None,
        "recall_at_k": None,
        "recall_threshold": float(args.recall_threshold),
        "threshold_passed": None,
        "insert_throughput_vectors_per_sec": None,
        "compression_ratio": None,
        "compressed_mem_mb": None,
        "latency_speedup_vs_numpy": None,
    }

    if not HAS_TQ:
        artifacts = write_artifacts(result, args.artifact_dir, args.artifact_prefix)
        print(f"Artifacts written: {artifacts}")
        return result

    # TurboQuantDB benchmark
    with tempfile.TemporaryDirectory() as db_dir:
        print(f"\nInserting {args.n:,} vectors into TurboQuantDB (b={args.bits})...")
        db = tq.Database.open(db_dir, dimension=args.d, bits=args.bits)

        t_insert_start = time.perf_counter()
        for i, vec in enumerate(corpus):
            db.insert(f"vec_{i}", vec, {"index": i})
        db.flush()
        insert_elapsed = time.perf_counter() - t_insert_start
        throughput = args.n / insert_elapsed

        print(f"  Insertion: {insert_elapsed:.2f}s total  |  {throughput:,.0f} vectors/sec")
        print(f"  Stats: {db.stats()}")

        # TurboQuantDB search + recall
        print(f"\nRunning {args.queries} queries, measuring Recall@{args.k}...")

        if args.use_ann:
            print("  Building HNSW index (max_degree=32, ef_construction=200)...")
            t_idx = time.perf_counter()
            db.create_index(max_degree=32, ef_construction=200, search_list_size=128)
            print(f"  Index built in {time.perf_counter() - t_idx:.2f}s")

        tq_latencies = []
        recalls = []

        for query in queries:
            # Ground truth
            gt = exact_topk(corpus, query, args.k)

            # TurboQuantDB
            t0 = time.perf_counter()
            results = db.search(query, top_k=args.k)
            tq_latencies.append((time.perf_counter() - t0) * 1000)

            retrieved = {r["id"] for r in results}
            recall = len(gt & retrieved) / args.k
            recalls.append(recall)

        db.close()

    # Results
    tq_avg_ms = np.mean(tq_latencies)
    recall_at_k = np.mean(recalls)

    # Estimate compressed memory (b bits per coord)
    compressed_bits = args.n * args.d * args.bits
    compressed_mb = compressed_bits / (8 * 1024**2)
    compression_ratio = numpy_mem_mb / compressed_mb

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Recall@{args.k}:             {recall_at_k:.1%}")
    print(f"  TurboQuantDB latency:  {tq_avg_ms:.2f} ms/query")
    print(f"  Numpy exact latency:   {numpy_avg_ms:.2f} ms/query")
    print(f"  Compressed size:       {compressed_mb:.1f} MB  (vs {numpy_mem_mb:.1f} MB)")
    print(f"  Compression ratio:     {compression_ratio:.1f}x")
    print(f"  Insertion throughput:  {throughput:,.0f} vectors/sec")
    print(f"{'='*60}\n")

    result.update(
        {
            "status": "ok",
            "turboquant_avg_ms": float(tq_avg_ms),
            "recall_at_k": float(recall_at_k),
            "threshold_passed": bool(recall_at_k >= args.recall_threshold),
            "insert_throughput_vectors_per_sec": float(throughput),
            "compression_ratio": float(compression_ratio),
            "compressed_mem_mb": float(compressed_mb),
            "latency_speedup_vs_numpy": float(numpy_avg_ms / tq_avg_ms) if tq_avg_ms > 0 else None,
        }
    )

    artifacts = write_artifacts(result, args.artifact_dir, args.artifact_prefix)
    print(f"Artifacts written: {artifacts}")

    # Validation gate
    if args.enforce_recall_threshold:
        assert recall_at_k >= args.recall_threshold, (
            f"FAIL: Recall@{args.k} = {recall_at_k:.1%} "
            f"(threshold: {args.recall_threshold:.0%})"
        )
        print(f"✓ Recall threshold passed (≥{args.recall_threshold:.0%})")

    return result


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
