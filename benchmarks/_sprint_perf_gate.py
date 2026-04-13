"""Sprint perf gate — measures speed, disk, RAM delta, and recall for before/after comparisons.

Matches the metrics in .bench_ram_compare/final_full_after_v4/after_now_v4.json:
  build_s, p50_ms, p95_ms, p99_ms, avg_ms, qps, recall10, disk_mb,
  ram_build_delta_mb, ram_query_delta_mb

Two cases (same as the competitor comparison):
  - 100k_768  (N=100k, D=768, Q=100)
  - 1m_256    (N=1M, D=256, Q=20)  -- uses 500K for speed; p50 scales linearly

Usage:
    python benchmarks/_sprint_perf_gate.py [--case 100k_768|1m_256|all]
"""

import sys, time, tempfile, os, json, argparse
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed — RAM delta will not be measured", flush=True)

try:
    from tqdb import Database
except ImportError:
    print("ERROR: tqdb not installed. Run: maturin develop --release")
    sys.exit(1)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def disk_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def current_rss_mb():
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


N_REPEATS = 3  # Run each case 3 times; report median-of-medians to reduce noise.


def _single_run(N, D, Q, K, BITS, SEED, corpus, queries, true_top_k):
    """One timed pass: build, warmup, search. Returns (metrics_dict, lats_sorted)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rss_before_build = current_rss_mb()
        # Use rerank=False to match the competitor comparison config (no live_vectors.bin).
        db = Database.open(tmpdir, dimension=D, bits=BITS, seed=SEED,
                           metric="cosine", rerank=False, fast_mode=True)
        ids = [str(i) for i in range(N)]

        t0 = time.perf_counter()
        db.insert_batch(ids, corpus, mode="insert")
        build_s = time.perf_counter() - t0
        rss_after_build = current_rss_mb()
        ram_build_delta = rss_after_build - rss_before_build
        disk_mb = disk_size_mb(tmpdir)

        # Warmup
        for qi in range(min(10, Q)):
            db.search(queries[qi], top_k=K)

        rss_before_query = current_rss_mb()
        lats, hits = [], 0
        for qi in range(Q):
            t1 = time.perf_counter()
            results = db.search(queries[qi], top_k=K)
            lats.append((time.perf_counter() - t1) * 1000.0)
            returned = {int(r["id"]) for r in results}
            true_ids = set(true_top_k[qi].tolist())
            hits += len(returned & true_ids)
        rss_after_query = current_rss_mb()
        ram_query_delta = rss_after_query - rss_before_query
        db.close()

    lats.sort()
    n = len(lats)
    return dict(
        build_s=build_s,
        p50_ms=lats[int(n * 0.50)],
        p95_ms=lats[int(n * 0.95)],
        p99_ms=lats[min(int(n * 0.99), n - 1)],
        avg_ms=sum(lats) / n,
        recall=hits / (Q * K),
        disk_mb=disk_mb,
        ram_build_delta_mb=ram_build_delta,
        ram_query_delta_mb=ram_query_delta,
    ), lats


def run_case(name, N, D, Q, K=10, BITS=4, SEED=7):
    print(f"\n=== {name} (N={N:,}, D={D}, Q={Q}, repeats={N_REPEATS}) ===", flush=True)
    rng = np.random.default_rng(SEED)
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Brute-force ground truth (computed once)
    scores_gt = corpus @ queries.T
    true_top_k = np.argsort(-scores_gt, axis=0)[:K].T  # (Q, K)

    all_runs = []
    all_lats = []
    for rep in range(N_REPEATS):
        print(f"  Inserting {N:,} vectors (rep {rep+1}/{N_REPEATS}) ...", flush=True)
        m, lats = _single_run(N, D, Q, K, BITS, SEED, corpus, queries, true_top_k)
        all_runs.append(m)
        all_lats.extend(lats)
        print(f"    p50={m['p50_ms']:.2f}ms  p95={m['p95_ms']:.2f}ms  p99={m['p99_ms']:.2f}ms  "
              f"disk={m['disk_mb']:.1f}MB  recall={m['recall']:.3f}", flush=True)

    # Median-of-medians across 3 repeats for each metric.
    def med(key):
        return float(np.median([r[key] for r in all_runs]))

    # For latency percentiles: sort all latencies across all repeats and recompute.
    all_lats.sort()
    n_total = len(all_lats)
    p50 = all_lats[int(n_total * 0.50)]
    p95 = all_lats[int(n_total * 0.95)]
    p99 = all_lats[min(int(n_total * 0.99), n_total - 1)]
    avg_ms = sum(all_lats) / n_total
    qps = 1000.0 / avg_ms if avg_ms > 0 else 0
    recall = med("recall")

    result = dict(
        build_s=round(med("build_s"), 3),
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        avg_ms=round(avg_ms, 3),
        qps=round(qps, 2),
        recall10=round(recall, 4),
        disk_mb=round(med("disk_mb"), 3),
        ram_build_delta_mb=round(med("ram_build_delta_mb"), 2),
        ram_query_delta_mb=round(med("ram_query_delta_mb"), 2),
    )

    print(f"  [median-of-{N_REPEATS}] build={result['build_s']:.1f}s  "
          f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  "
          f"qps={qps:.1f}  recall={recall:.4f}  disk={result['disk_mb']:.1f}MB  "
          f"RAM_build={result['ram_build_delta_mb']:.0f}MB  RAM_query={result['ram_query_delta_mb']:.0f}MB")
    return result


CASES = {
    "100k_768": dict(N=100_000, D=768, Q=100),
    "1m_256":   dict(N=500_000, D=256, Q=50),   # 500K proxy for 1M (linear scaling)
}

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="all", choices=list(CASES) + ["all"])
parser.add_argument(
    "--baseline", type=str, default=None,
    help="Path to a previous summary JSON (from --save) to compare against. "
         "Gate: KEEP if both p50 and p95 improve or are flat; REVERT if either regresses."
)
parser.add_argument("--save", type=str, default=None, help="Save summary JSON to this path.")
args = parser.parse_args()

cases_to_run = list(CASES.items()) if args.case == "all" else [(args.case, CASES[args.case])]
results = {}
for name, cfg in cases_to_run:
    results[name] = run_case(name, **cfg)

print("\n\n=== SUMMARY ===")
print(json.dumps(results, indent=2))

if args.save:
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.save}")

if args.baseline:
    with open(args.baseline) as f:
        baseline = json.load(f)
    print("\n=== KEEP/DROP GATE (p50 + p95 both must improve or be flat) ===")
    decisions = []
    for name, res in results.items():
        if name not in baseline:
            print(f"  {name}: no baseline — skip gate")
            continue
        b = baseline[name]
        p50_delta = res["p50_ms"] - b["p50_ms"]
        p95_delta = res["p95_ms"] - b["p95_ms"]
        # Allow up to 2% degradation as noise floor
        NOISE_PCT = 0.02
        p50_regress = p50_delta > b["p50_ms"] * NOISE_PCT
        p95_regress = p95_delta > b["p95_ms"] * NOISE_PCT
        verdict = "REVERT" if (p50_regress or p95_regress) else "KEEP"
        decisions.append(verdict)
        print(f"  {name}: p50 {b['p50_ms']:.2f}->{res['p50_ms']:.2f}ms ({p50_delta:+.2f})  "
              f"p95 {b['p95_ms']:.2f}->{res['p95_ms']:.2f}ms ({p95_delta:+.2f})  "
              f"=> {verdict}")
    if all(d == "KEEP" for d in decisions):
        print("\nResult: KEEP all changes")
    else:
        print("\nResult: REVERT — one or more cases regressed on p50 or p95")
        sys.exit(1)
