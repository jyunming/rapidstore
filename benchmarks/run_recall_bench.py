"""
TurboQuantDB Recall & Latency Benchmark
========================================
Measures:
  - Recall@10 (vs exact Numpy ground truth)
  - Avg query latency (ms)
  - Insertion throughput (vectors/sec)
  - Index memory estimate

Usage:
    pip install numpy turboquantdb
    python benchmarks/run_recall_bench.py --n 10000 --d 1536 --bits 4
"""

import argparse
import time
import tempfile
import os
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000, help="Number of vectors")
    p.add_argument("--d", type=int, default=1536, help="Vector dimension")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4], help="TurboQuant bit width")
    p.add_argument("--k", type=int, default=10, help="Top-k for recall measurement")
    p.add_argument("--queries", type=int, default=100, help="Number of query vectors")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def exact_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> set[str]:
    """Brute-force exact inner product search using Numpy."""
    scores = corpus @ query  # (N,) inner products
    top_indices = np.argsort(scores)[::-1][:k]
    return {f"vec_{i}" for i in top_indices}


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
        import turboquantdb as tq
        HAS_TQ = True
    except ImportError:
        print("[WARNING] turboquantdb not installed. Run `maturin develop --release` first.")
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

    if not HAS_TQ:
        return

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

    # Validation gate
    assert recall_at_k >= 0.80, f"FAIL: Recall@{args.k} = {recall_at_k:.1%} (threshold: 80%)"
    print("✓ Recall threshold passed (≥80%)")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
