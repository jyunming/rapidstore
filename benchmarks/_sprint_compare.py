"""Sprint 3-way comparison: main branch vs current branch vs turboquant-vectors.

Run from the repo root after `maturin develop --release`.
Saves results to _sprint_compare_results.json.

Usage:
    python benchmarks/_sprint_compare.py [--case 1m_256|100k_768|all]
"""

import sys, time, tempfile, os, json, argparse, gc
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from tqdb import Database
    HAS_TQDB = True
except ImportError:
    HAS_TQDB = False
    print("WARNING: tqdb not installed")

try:
    from turboquant_vectors import compress as tqv_compress, search as tqv_search
    HAS_TQV = True
except ImportError:
    HAS_TQV = False
    print("WARNING: turboquant-vectors not installed")

N_REPEATS = 3


def current_rss_mb():
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def disk_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def run_tqdb(name, N, D, Q, corpus, queries, true_top_k, K=10, BITS=4, SEED=7):
    if not HAS_TQDB:
        return None
    all_metrics = []
    for rep in range(N_REPEATS):
        with tempfile.TemporaryDirectory() as tmpdir:
            rss0 = current_rss_mb()
            db = Database.open(tmpdir, dimension=D, bits=BITS, seed=SEED,
                               metric="cosine", rerank=False, fast_mode=True)
            ids = [str(i) for i in range(N)]
            t0 = time.perf_counter()
            db.insert_batch(ids, corpus, mode="insert")
            build_s = time.perf_counter() - t0
            rss1 = current_rss_mb()
            disk_mb = disk_size_mb(tmpdir)

            for qi in range(min(10, Q)):
                db.search(queries[qi], top_k=K)

            rss2 = current_rss_mb()
            lats, hits = [], 0
            for qi in range(Q):
                t1 = time.perf_counter()
                results = db.search(queries[qi], top_k=K)
                lats.append((time.perf_counter() - t1) * 1000.0)
                returned = {int(r["id"]) for r in results}
                hits += len(returned & set(true_top_k[qi].tolist()))
            rss3 = current_rss_mb()
            db.close()

        lats.sort()
        n = len(lats)
        all_metrics.append(dict(
            build_s=build_s,
            p50_ms=lats[int(n * 0.50)],
            p95_ms=lats[int(n * 0.95)],
            p99_ms=lats[min(int(n * 0.99), n - 1)],
            avg_ms=sum(lats) / n,
            recall=hits / (Q * K),
            disk_mb=disk_mb,
            ram_build_delta_mb=rss1 - rss0,
            ram_query_delta_mb=rss3 - rss2,
        ))
        print(f"    tqdb rep {rep+1}/{N_REPEATS}: p50={all_metrics[-1]['p50_ms']:.2f}ms "
              f"p95={all_metrics[-1]['p95_ms']:.2f}ms disk={disk_mb:.1f}MB "
              f"recall={all_metrics[-1]['recall']:.3f}", flush=True)

    def med(k):
        return float(np.median([m[k] for m in all_metrics]))

    all_lats_flat = []
    for m in all_metrics:
        n = Q
        all_lats_flat.extend([m["p50_ms"]] * int(n * 0.5))
    # Better: just use median of per-run percentiles
    return dict(
        build_s=round(med("build_s"), 3),
        p50_ms=round(med("p50_ms"), 3),
        p95_ms=round(med("p95_ms"), 3),
        p99_ms=round(med("p99_ms"), 3),
        avg_ms=round(med("avg_ms"), 3),
        qps=round(1000.0 / med("avg_ms"), 2),
        recall10=round(med("recall"), 4),
        disk_mb=round(med("disk_mb"), 3),
        ram_build_delta_mb=round(med("ram_build_delta_mb"), 2),
        ram_query_delta_mb=round(med("ram_query_delta_mb"), 2),
    )


def run_tqv(name, N, D, Q, corpus, queries, true_top_k, K=10, BITS=4, SEED=7):
    if not HAS_TQV:
        return None
    all_metrics = []
    for rep in range(N_REPEATS):
        gc.collect()
        rss0 = current_rss_mb()
        t0 = time.perf_counter()
        compressed = tqv_compress(corpus, bits=BITS, seed=SEED)
        build_s = time.perf_counter() - t0
        rss1 = current_rss_mb()

        # Disk: save to .npz and measure file size
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name
        try:
            compressed.save(tmp_path)
            disk_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        finally:
            os.unlink(tmp_path)

        # Warmup
        for qi in range(min(5, Q)):
            tqv_search(compressed, queries[qi], top_k=K)

        rss2 = current_rss_mb()
        lats, hits = [], 0
        for qi in range(Q):
            t1 = time.perf_counter()
            idx, _ = tqv_search(compressed, queries[qi], top_k=K)
            lats.append((time.perf_counter() - t1) * 1000.0)
            hits += len(set(idx.tolist()) & set(true_top_k[qi].tolist()))
        rss3 = current_rss_mb()
        del compressed
        gc.collect()

        lats.sort()
        n = len(lats)
        all_metrics.append(dict(
            build_s=build_s,
            p50_ms=lats[int(n * 0.50)],
            p95_ms=lats[int(n * 0.95)],
            p99_ms=lats[min(int(n * 0.99), n - 1)],
            avg_ms=sum(lats) / n,
            recall=hits / (Q * K),
            disk_mb=disk_mb,
            ram_build_delta_mb=rss1 - rss0,
            ram_query_delta_mb=rss3 - rss2,
        ))
        print(f"    tqv  rep {rep+1}/{N_REPEATS}: p50={all_metrics[-1]['p50_ms']:.2f}ms "
              f"p95={all_metrics[-1]['p95_ms']:.2f}ms disk={disk_mb:.1f}MB "
              f"recall={all_metrics[-1]['recall']:.3f}", flush=True)

    def med(k):
        return float(np.median([m[k] for m in all_metrics]))

    return dict(
        build_s=round(med("build_s"), 3),
        p50_ms=round(med("p50_ms"), 3),
        p95_ms=round(med("p95_ms"), 3),
        p99_ms=round(med("p99_ms"), 3),
        avg_ms=round(med("avg_ms"), 3),
        qps=round(1000.0 / med("avg_ms"), 2),
        recall10=round(med("recall"), 4),
        disk_mb=round(med("disk_mb"), 3),
        ram_build_delta_mb=round(med("ram_build_delta_mb"), 2),
        ram_query_delta_mb=round(med("ram_query_delta_mb"), 2),
    )


CASES = {
    "100k_768": dict(N=100_000, D=768, Q=100),
    "1m_256":   dict(N=500_000, D=256, Q=50),
}

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="all", choices=list(CASES) + ["all"])
parser.add_argument("--engines", default="tqdb,tqv", help="Comma-separated engines to run")
args = parser.parse_args()

cases_to_run = list(CASES.items()) if args.case == "all" else [(args.case, CASES[args.case])]
engines = [e.strip() for e in args.engines.split(",")]
all_results = {}

for name, cfg in cases_to_run:
    N, D, Q = cfg["N"], cfg["D"], cfg["Q"]
    K = 10
    BITS = 4
    SEED = 7
    print(f"\n=== {name} (N={N:,}, D={D}, Q={Q}, bits={BITS}) ===", flush=True)

    rng = np.random.default_rng(SEED)
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scores_gt = corpus @ queries.T
    true_top_k = np.argsort(-scores_gt, axis=0)[:K].T

    case_results = {}
    if "tqdb" in engines:
        print(f"  -- TQDB --", flush=True)
        r = run_tqdb(name, N, D, Q, corpus, queries, true_top_k, K=K, BITS=BITS, SEED=SEED)
        if r:
            case_results["tqdb"] = r
    if "tqv" in engines:
        print(f"  -- turboquant-vectors --", flush=True)
        r = run_tqv(name, N, D, Q, corpus, queries, true_top_k, K=K, BITS=BITS, SEED=SEED)
        if r:
            case_results["tqv"] = r

    all_results[name] = case_results

    # Print comparison table
    print(f"\n  {'Metric':<22} {'TQDB':>12} {'TQV':>12} {'TQDB/TQV':>10}")
    print(f"  {'-'*58}")
    metrics = [
        ("build_s", "build (s)", "{:.2f}"),
        ("p50_ms", "p50 (ms)", "{:.2f}"),
        ("p95_ms", "p95 (ms)", "{:.2f}"),
        ("p99_ms", "p99 (ms)", "{:.2f}"),
        ("qps", "qps", "{:.1f}"),
        ("recall10", "recall@10", "{:.4f}"),
        ("disk_mb", "disk (MB)", "{:.1f}"),
        ("ram_build_delta_mb", "RAM build (MB)", "{:.0f}"),
        ("ram_query_delta_mb", "RAM query (MB)", "{:.0f}"),
    ]
    tqdb = case_results.get("tqdb", {})
    tqv = case_results.get("tqv", {})
    for key, label, fmt in metrics:
        a = tqdb.get(key, float("nan"))
        b = tqv.get(key, float("nan"))
        ratio = f"{a/b:.2f}x" if b and b != 0 else "n/a"
        print(f"  {label:<22} {fmt.format(a):>12} {fmt.format(b):>12} {ratio:>10}")

print("\n\n=== FULL RESULTS JSON ===")
print(json.dumps(all_results, indent=2))

out = "benchmarks/_sprint_compare_results.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out}")
