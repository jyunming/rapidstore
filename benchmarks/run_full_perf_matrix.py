"""
Full performance matrix benchmark — isolated subprocesses per combination.

Runs every (n, bits, mode) combination and prints a consolidated table.
Each combination is run in its own subprocess so memory/GC state is isolated.

Usage:
    python benchmarks/run_full_perf_matrix.py
    python benchmarks/run_full_perf_matrix.py --d 128 --queries 50 --artifact-dir benchmarks/artifacts/matrix
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

NS      = [10_000, 25_000, 50_000, 100_000]
BITS    = [2, 3, 4]
MODES   = ["brute", "ann"]   # ann = --use-ann flag
QUERIES = 50
D       = 128
SEED    = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d",           type=int, default=D,       help="Vector dimension")
    p.add_argument("--queries",     type=int, default=QUERIES,  help="Queries per run")
    p.add_argument("--seed",        type=int, default=SEED)
    p.add_argument("--artifact-dir",type=str, default="benchmarks/artifacts/matrix")
    return p.parse_args()


def run_one(n: int, bits: int, use_ann: bool, d: int, queries: int, seed: int,
            artifact_dir: str, tmpdir: str) -> dict | None:
    """Spawn a fresh Python process for one benchmark combination."""
    prefix = f"matrix_n{n}_b{bits}_{'ann' if use_ann else 'brute'}"
    cmd = [
        sys.executable,
        "benchmarks/run_recall_bench.py",
        "--n",              str(n),
        "--d",              str(d),
        "--bits",           str(bits),
        "--queries",        str(queries),
        "--seed",           str(seed),
        "--no-enforce-recall-threshold",
        "--artifact-dir",   artifact_dir,
        "--artifact-prefix", prefix,
    ]
    if use_ann:
        cmd.append("--use-ann")

    env = os.environ.copy()
    # Give each run its own temp directory so Windows temp locks don't collide
    env["TMPDIR"] = env["TEMP"] = env["TMP"] = tmpdir

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if result.returncode != 0:
            print(f"  [ERROR] n={n} bits={bits} {'ann' if use_ann else 'brute'}:\n{result.stderr[-800:]}")
            return None

        # Parse the JSON artifact written by run_recall_bench.py
        art_dir = Path(artifact_dir)
        candidates = sorted(art_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"  [WARN] No artifact found for {prefix}")
            return None
        with open(candidates[0]) as f:
            data = json.load(f)
        return data
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] n={n} bits={bits} {'ann' if use_ann else 'brute'}")
        return None
    except Exception as e:
        print(f"  [EXCEPTION] {e}")
        return None


def fmt(v, fmt_str=".2f"):
    if v is None:
        return "—"
    return format(float(v), fmt_str)


def main():
    args = parse_args()
    art_dir = args.artifact_dir
    os.makedirs(art_dir, exist_ok=True)

    # One shared temp dir per matrix run (still isolated per subprocess via env)
    tmp_base = os.path.join(os.getcwd(), "t", "bench_matrix_tmp")
    os.makedirs(tmp_base, exist_ok=True)

    total = len(NS) * len(BITS) * len(MODES)
    done  = 0
    results = []

    print(f"\n{'='*72}")
    print(f"  TurboQuantDB — Full Perf Matrix")
    print(f"  d={args.d}  queries={args.queries}  seed={args.seed}")
    print(f"  {total} combinations  (each in isolated subprocess)")
    print(f"{'='*72}\n")

    for n in NS:
        for bits in BITS:
            for mode in MODES:
                use_ann = (mode == "ann")
                done += 1
                label = f"n={n:>7,}  bits={bits}  {'ANN  ' if use_ann else 'brute'}"
                print(f"  [{done:>2}/{total}] {label} ...", end="", flush=True)

                # Per-run isolated tmpdir
                run_tmp = os.path.join(tmp_base, f"n{n}_b{bits}_{mode}")
                os.makedirs(run_tmp, exist_ok=True)

                data = run_one(n, bits, use_ann, args.d, args.queries, args.seed,
                               art_dir, run_tmp)
                if data:
                    recall   = data.get("recall_at_k")
                    tq_ms    = data.get("turboquant_avg_ms")
                    np_ms    = data.get("numpy_avg_ms")
                    speedup  = data.get("latency_speedup_vs_numpy")
                    ingest   = data.get("insert_throughput_vectors_per_sec")
                    print(f"  recall={fmt(recall,'6.1%')}  tq={fmt(tq_ms,'6.2f')}ms"
                          f"  np={fmt(np_ms,'5.2f')}ms  speedup={fmt(speedup,'.3f')}x"
                          f"  ingest={fmt(ingest,',.0f')} v/s")
                    results.append({
                        "n": n, "bits": bits, "mode": mode,
                        "recall": recall, "tq_ms": tq_ms,
                        "np_ms": np_ms, "speedup": speedup, "ingest_vps": ingest,
                    })
                else:
                    print("  FAILED")
                    results.append({"n": n, "bits": bits, "mode": mode, "recall": None})

    # Summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY TABLE  (d={args.d}, queries={args.queries})")
    print(f"{'='*72}")
    hdr = f"{'n':>8}  {'bits':>4}  {'mode':>5}  {'recall':>7}  {'tq_ms':>7}  {'np_ms':>6}  {'speedup':>7}  {'ingest v/s':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("recall") is None:
            print(f"{r['n']:>8,}  {r['bits']:>4}  {r['mode']:>5}  {'FAILED':>7}")
            continue
        print(
            f"{r['n']:>8,}  {r['bits']:>4}  {r['mode']:>5}  "
            f"{fmt(r.get('recall'), '7.1%')}  {fmt(r.get('tq_ms'), '7.2f')}  "
            f"{fmt(r.get('np_ms'), '6.2f')}  {fmt(r.get('speedup'), '7.3f')}x  "
            f"{fmt(r.get('ingest_vps'), '11,.0f')}"
        )

    # Save consolidated JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(art_dir, f"matrix_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "d": args.d, "queries": args.queries,
                   "results": results}, f, indent=2)
    print(f"\n  Consolidated results → {out_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
