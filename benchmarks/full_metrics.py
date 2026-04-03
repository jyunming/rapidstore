import turboquantdb as tq
import numpy as np
import time
import tempfile
from pathlib import Path
import json

def run_suite():
    n = 10_000
    d = 768
    k = 10
    queries_count = 50
    seed = 42

    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

    queries = rng.standard_normal((queries_count, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print(f"Running Full Metrics Suite (n={n}, d={d}, k={k})")

    # Ground Truth
    gt = []
    print("Computing Ground Truth...")
    for q in queries:
        scores = corpus @ q
        top_idx = np.argsort(scores)[::-1][:k]
        gt.append({f"vec_{i}" for i in top_idx})

    results = []

    configs = [
        {"bits": 4, "rerank": True, "ann": True},
        {"bits": 4, "rerank": False, "ann": True},
        {"bits": 2, "rerank": True, "ann": True},
        {"bits": 2, "rerank": False, "ann": True},
        {"bits": 4, "rerank": False, "ann": False}, # Brute force
    ]

    for cfg in configs:
        bits = cfg["bits"]
        rerank = cfg["rerank"]
        use_ann = cfg["ann"]

        mode_str = f"bits={bits}, rerank={rerank}, {'ANN' if use_ann else 'BF'}"
        print(f"\nTesting: {mode_str}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            db = tq.Database.open(tmp_dir, dimension=d, bits=bits, rerank=rerank)

            # Ingest
            t0 = time.perf_counter()
            # Use insert_batch for speed
            ids = [f"vec_{i}" for i in range(n)]
            db.insert_batch(ids, corpus)
            db.flush()
            t_ingest = time.perf_counter() - t0
            throughput = n / t_ingest

            # Index
            t_idx = 0
            if use_ann:
                t0 = time.perf_counter()
                db.create_index(max_degree=16, search_list_size=128)
                t_idx = time.perf_counter() - t0

            # Search
            latencies = []
            recalls = []
            for i, q in enumerate(queries):
                t0 = time.perf_counter()
                hits = db.search(q, top_k=k, _use_ann=use_ann)
                latencies.append((time.perf_counter() - t0) * 1000)

                retrieved = {h["id"] for h in hits}
                recall = len(gt[i] & retrieved) / k
                recalls.append(recall)

            avg_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            avg_recall = np.mean(recalls)

            stats = db.stats()
            disk = stats["total_disk_bytes"] / (1024*1024)

            res = {
                "config": mode_str,
                "recall": avg_recall,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "ingest_throughput_vps": throughput,
                "index_time_s": t_idx,
                "disk_mb": disk
            }
            results.append(res)
            print(f"  Recall: {avg_recall:.1%}")
            print(f"  Latency: {avg_lat:.2f}ms (p95: {p95_lat:.2f}ms)")
            print(f"  Ingest: {throughput:.0f} vps")
            print(f"  Disk: {disk:.2f} MB")

            db.close()

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Config':<40} | {'Recall':<8} | {'Lat (ms)':<8} | {'Ingest':<8} | {'Disk (MB)':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['config']:<40} | {r['recall']:<8.1%} | {r['avg_latency_ms']:<8.2f} | {r['ingest_throughput_vps']:<8.0f} | {r['disk_mb']:<8.2f}")
    print("="*80)

if __name__ == "__main__":
    run_suite()
