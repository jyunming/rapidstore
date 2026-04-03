import subprocess
import os
import json
import time
import sys

# Configuration
SCALES = [10000, 25000, 50000, 100000]
DIM = 384
BITS = 8
SEARCH_RUNS = 30
TOP_K = 10

def run_isolated_bench(engine, N):
    print(f"  Testing {engine} at N={N}...")

    cmd = [
        sys.executable,
        "benchmarks/isolated_worker.py",
        "--engine", engine,
        "--n", str(N),
        "--dim", str(DIM),
        "--bits", str(BITS),
        "--k", str(TOP_K),
        "--runs", str(SEARCH_RUNS)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {engine} at {N}:")
            print(result.stderr)
            return None

        # Parse worker's JSON output
        worker_res = {}
        for line in result.stdout.strip().split("\n"):
            try:
                worker_res = json.loads(line)
                break
            except: continue
        return worker_res
    except Exception as e:
        print(f"Failed to run isolated bench: {e}")
        return None

def main():
    all_results = {}
    engines = ["lancedb", "chromadb", "tqdb"]

    print(f"Isolated Accuracy & Performance Benchmark")
    print(f"dim={DIM} top_k={TOP_K} bits={BITS}")
    print("-" * 135)

    header = f"{'Scale':>8} | {'Engine':>10} | {'Recall':>7} | {'Ready':>6} | {'Disk':>7} | {'RAM Ing':>9} | {'CPU Ing':>7} | {'RAM Qry':>9} | {'CPU Qry':>7} | {'p50':>6} | {'p96':>6}"
    print(header)
    print("-" * 135)

    for N in SCALES:
        all_results[N] = {}
        for engine in engines:
            res = run_isolated_bench(engine, N)
            if res:
                all_results[N][engine] = res
                line = (f"{N:>8} | {engine:>10} | {res['recall']*100:>6.1f}% | {res['ready_time']:>5.1f}s | {res['disk_mb']:>5.1f}MB | "
                        f"{res['ingest_rss_mb']:>7.1f}MB | {res['ingest_cpu_util']:>6.0f}% | "
                        f"{res['retrieve_rss_mb']:>7.1f}MB | {res['query_cpu_util']:>6.0f}% | "
                        f"{res['p50_ms']:>5.2f}ms | {res['p96_ms']:>5.2f}ms")
                print(line)
        print("-" * 135)

    # Save results
    with open("benchmarks/isolated_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to benchmarks/isolated_comparison_results.json")

if __name__ == "__main__":
    main()
