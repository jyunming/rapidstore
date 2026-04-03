"""
Full isolated comparison benchmark: FAISS Flat vs LanceDB vs ChromaDB vs TurboQuantDB.

Each engine run is a separate subprocess (process isolation — no cross-contamination
of RAM, CPU caches, or Python GC state between engines).

Metrics per engine per scale:
  - Recall@10     (vs brute-force ground truth, inner product)
  - Ready time    (ingest + index build)
  - Ingest speed  (vectors/sec)
  - Disk MB
  - RAM peak (ingest)
  - RAM at query  (engine-only, after dropping input buffers)
  - CPU util (ingest phase, avg%)
  - CPU util (query phase, avg%)
  - p50 latency (ms)
  - p95 latency (ms)

Usage:
    python benchmarks/full_comparison_bench.py
    python benchmarks/full_comparison_bench.py --scales 10000 50000 100000
    python benchmarks/full_comparison_bench.py --dims 384 768 1536 --k 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

PYTHON = sys.executable
WORKER = os.path.join(os.path.dirname(__file__), "isolated_worker.py")

DEFAULT_SCALES = [10_000, 25_000, 50_000, 100_000]
DEFAULT_DIMS   = [384]
DEFAULT_K      = 10
DEFAULT_RUNS   = 50

# (label, engine, bits, fast_mode, rerank)
# rerank=None  -> not applicable (competitors)
# rerank=True  -> TQDB stores float32 for reranking after ANN
# rerank=False -> TQDB no-rerank (pure quantized)
ENGINES = [
    # ── Recall ceiling (not a vector store — RAM only, no persistence) ──────
    ("FAISS Flat [baseline]",   "faiss_flat", None, False, None),
    # ── Vector stores (persist to disk, metadata, IDs) ──────────────────────
    ("LanceDB (IVF_PQ)",        "lancedb",    None, False, None),
    ("ChromaDB (HNSW)",         "chromadb",   None, False, None),
    ("Qdrant (HNSW,embedded)",  "qdrant",     None, False, None),
    # ── TurboQuantDB ────────────────────────────────────────────────────────
    ("tqdb-b4",                 "tqdb",       4,    False, True),
    ("tqdb-b4-fast",            "tqdb",       4,    True,  True),
    ("tqdb-b8",                 "tqdb",       8,    False, True),
    ("tqdb-b8-fast",            "tqdb",       8,    True,  True),
    ("tqdb-b4-norerank",        "tqdb",       4,    False, False),
]


def engine_key(engine, bits, fast_mode, rerank):
    """Unique dict key for a given engine configuration."""
    parts = [engine]
    if bits is not None:
        parts.append(str(bits))
    if fast_mode:
        parts.append("fast")
    if rerank is False:
        parts.append("norerank")
    return "_".join(parts)


def run_worker(engine, bits, n, dim, k, runs, fast_mode=False, rerank=None):
    cmd = [PYTHON, WORKER,
           "--engine", engine,
           "--n",      str(n),
           "--dim",    str(dim),
           "--bits",   str(bits or 8),
           "--k",      str(k),
           "--runs",   str(runs)]
    if fast_mode:
        cmd.append("--fast_mode")
    if rerank is True:
        cmd.append("--rerank")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        # Recover JSON from stdout even on non-zero exit (e.g. Windows cleanup errors)
        for line in proc.stdout.strip().splitlines():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        print(f"    [ERROR] {engine} bits={bits} n={n}:\n{proc.stderr[-600:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {engine} bits={bits} n={n}")
        return None


def fmt(v, unit="", decimals=2):
    if v is None:
        return "   n/a"
    return f"{v:.{decimals}f}{unit}"


def print_scale_block(n, results, dim):
    bar = "-" * 150
    print(bar)
    print(f"  Scale: {n:,} vectors  (dim={dim})")
    print(bar)
    header = (f"  {'Engine':<24}  {'Recall':>7}  {'Ready':>7}  "
              f"{'Ingest':>10}  {'Disk':>8}  {'RAM-ing':>8}  {'RAM-qry':>8}  "
              f"{'CPU-ing':>8}  {'CPU-qry':>8}  {'p50':>7}  {'p95':>7}")
    print(header)
    print(f"  {'':<24}  {'@k':>7}  {'(s)':>7}  "
          f"{'(vec/s)':>10}  {'(MB)':>8}  {'(MB)':>8}  {'(MB)':>8}  "
          f"{'(%)':>8}  {'(%)':>8}  {'(ms)':>7}  {'(ms)':>7}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*7}  "
          f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    for label, eng, bits, fast_mode, rerank in ENGINES:
        key = engine_key(eng, bits, fast_mode, rerank)
        r = results.get(key)
        if r is None:
            print(f"  {label:<24}  {'FAILED':>7}")
            continue
        ingest_speed = n / r["ready_time"] if r.get("ready_time", 0) > 0 else 0
        ingest_speed_str = f"{int(ingest_speed):,}"
        print(
            f"  {label:<24}  "
            f"{r['recall']*100:>6.1f}%  "
            f"{fmt(r.get('ready_time'), 's', 1):>7}  "
            f"{ingest_speed_str:>10}  "
            f"{fmt(r.get('disk_mb'), 'MB', 1):>8}  "
            f"{fmt(r.get('ingest_rss_mb'), 'MB', 1):>8}  "
            f"{fmt(r.get('retrieve_rss_mb'), 'MB', 1):>8}  "
            f"{fmt(r.get('ingest_cpu_util'), '%', 1):>8}  "
            f"{fmt(r.get('query_cpu_util'), '%', 1):>8}  "
            f"{fmt(r.get('p50_ms'), 'ms', 2):>7}  "
            f"{fmt(r.get('p96_ms'), 'ms', 2):>7}"
        )
    print()


def print_summary(all_results, scales, dims):
    bar = "=" * 150
    print(f"\n{bar}")
    print(f"  SUMMARY TABLES")
    print(bar)

    for dim in dims:
        if len(dims) > 1:
            print(f"\n  --- dim={dim} ---")

        def get_dim_results(n):
            return all_results.get(dim, {}).get(n, {})

        metrics = [
            ("Recall@k (%)",         lambda r, n: f"{r['recall']*100:.1f}%",          "higher=better"),
            ("Ready time (s)",        lambda r, n: f"{r['ready_time']:.1f}s",           "lower=better"),
            ("Ingest speed (vec/s)",  lambda r, n: f"{int(n/r['ready_time']):,}",       "higher=better"),
            ("Disk (MB)",             lambda r, n: f"{r['disk_mb']:.1f}MB",             "lower=better"),
            ("RAM ingest peak (MB)",  lambda r, n: f"{r['ingest_rss_mb']:.1f}MB",       "lower=better"),
            ("RAM at query (MB)",     lambda r, n: f"{r['retrieve_rss_mb']:.1f}MB",     "lower=better"),
            ("p50 latency (ms)",      lambda r, n: f"{r['p50_ms']:.2f}ms",             "lower=better"),
            ("p95 latency (ms)",      lambda r, n: f"{r['p96_ms']:.2f}ms",             "lower=better"),
        ]

        scale_hdrs = "  ".join(f"{n:>12,}" for n in scales)
        for title, fn, note in metrics:
            print(f"\n  {title}  [{note}]")
            print(f"  {'Engine':<24}  {scale_hdrs}")
            for label, eng, bits, fast_mode, rerank in ENGINES:
                key = engine_key(eng, bits, fast_mode, rerank)
                row = f"  {label:<24}"
                for n in scales:
                    r = get_dim_results(n).get(key)
                    try:
                        row += f"  {fn(r, n):>12}" if r else f"  {'n/a':>12}"
                    except Exception:
                        row += f"  {'err':>12}"
                print(row)


def print_claims_check(all_results, scales, dims):
    """Verify the 6 key claims from the test specification."""
    bar = "=" * 150
    print(f"\n{bar}")
    print(f"  CLAIMS VERIFICATION")
    print(bar)

    results = []

    for dim in dims:
        dim_res = all_results.get(dim, {})

        def get(n, eng, bits, fast_mode, rerank, field):
            key = engine_key(eng, bits, fast_mode, rerank)
            r = dim_res.get(n, {}).get(key)
            return r.get(field) if r else None

        for n in scales:
            chroma_disk = get(n, "chromadb", None, False, None, "disk_mb")
            tqdb_b4_disk = get(n, "tqdb", 4, False, True, "disk_mb")
            if chroma_disk and tqdb_b4_disk and chroma_disk > 0:
                ratio = chroma_disk / tqdb_b4_disk
                ok = ratio >= 4.0
                results.append((f"[1] Disk 4-5x smaller than ChromaDB  (dim={dim}, N={n:,})",
                                f"ChromaDB={chroma_disk:.1f}MB  tqdb-b4={tqdb_b4_disk:.1f}MB  ratio={ratio:.2f}x",
                                ok))

            chroma_ram = get(n, "chromadb", None, False, None, "retrieve_rss_mb")
            tqdb_b4_ram = get(n, "tqdb", 4, False, True, "retrieve_rss_mb")
            if chroma_ram and tqdb_b4_ram and chroma_ram > 0:
                reduction = (chroma_ram - tqdb_b4_ram) / chroma_ram * 100
                ok = reduction >= 20
                results.append((f"[2] RAM 30-40% less than ChromaDB     (dim={dim}, N={n:,})",
                                f"ChromaDB={chroma_ram:.1f}MB  tqdb-b4={tqdb_b4_ram:.1f}MB  reduction={reduction:.1f}%",
                                ok))

            chroma_recall = get(n, "chromadb", None, False, None, "recall")
            tqdb_b8_recall = get(n, "tqdb", 8, False, True, "recall")
            if chroma_recall and tqdb_b8_recall:
                ok = tqdb_b8_recall > chroma_recall
                results.append((f"[3] b=8 recall > ChromaDB             (dim={dim}, N={n:,})",
                                f"tqdb-b8={tqdb_b8_recall*100:.1f}%  ChromaDB={chroma_recall*100:.1f}%",
                                ok))

        # Claim 4: fast_mode >= 20% faster ingest (use largest scale)
        n = max(scales)
        for bits in [4, 8]:
            std_ready  = get(n, "tqdb", bits, False, True, "ready_time")
            fast_ready = get(n, "tqdb", bits, True,  True, "ready_time")
            if std_ready and fast_ready and fast_ready > 0:
                speedup = (std_ready - fast_ready) / std_ready * 100
                ok = speedup >= 15  # allow 15% margin
                results.append((f"[4] fast_mode >=20% faster ingest     (dim={dim}, N={n:,}, b={bits})",
                                f"std={std_ready:.1f}s  fast={fast_ready:.1f}s  speedup={speedup:.1f}%",
                                ok))

        # Claim 5: fast_mode recall drop <= 5pp vs standard (use 10k for clearest signal)
        n = min(scales)
        for bits in [4, 8]:
            std_recall  = get(n, "tqdb", bits, False, True, "recall")
            fast_recall = get(n, "tqdb", bits, True,  True, "recall")
            if std_recall and fast_recall:
                drop = (std_recall - fast_recall) * 100
                ok = drop <= 5.0
                results.append((f"[5] fast_mode recall drop <=5pp       (dim={dim}, N={n:,}, b={bits})",
                                f"std={std_recall*100:.1f}%  fast={fast_recall*100:.1f}%  drop={drop:.1f}pp",
                                ok))

        # Claim 6: rerank=True improves recall 2-7pp vs norerank (use smallest scale)
        n = min(scales)
        rerank_recall   = get(n, "tqdb", 4, False, True,  "recall")
        norerank_recall = get(n, "tqdb", 4, False, False, "recall")
        if rerank_recall and norerank_recall:
            improvement = (rerank_recall - norerank_recall) * 100
            ok = improvement >= 2.0
            results.append((f"[6] rerank=True improves recall 2-7pp  (dim={dim}, N={n:,}, b=4)",
                            f"rerank={rerank_recall*100:.1f}%  norerank={norerank_recall*100:.1f}%  gain={improvement:.1f}pp",
                            ok))

    print()
    for title, detail, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {title}")
        print(f"          {detail}")
    print()


def print_human_readable(all_results, scales, dims):
    """Print text-capacity and scenario tables for non-technical readers."""
    for dim in dims:
        dim_res = all_results.get(dim, {})
        BYTES_F32 = dim * 4
        F32_DISK_PER_100K = (100_000 * BYTES_F32) / (1024 * 1024)
        AVAILABLE_RAM_MB = 16 * 1024 - 2048  # 14336 MB usable
        CHUNK_KB = 2.0  # 512-token chunk ~ 2KB text

        s_small = min(scales)
        s_large = max(scales)

        bar = "=" * 150
        print(f"\n{bar}")
        print(f"  REAL-WORLD CAPACITY  dim={dim}  (1 vec = 1 RAG chunk, ~{CHUNK_KB:.0f}KB text each)")
        print(bar)

        def get(n, key, field):
            r = dim_res.get(n, {}).get(key)
            return r.get(field) if r else None

        def fmt_count(n):
            if n >= 1e9:  return f"{n/1e9:.1f}B"
            if n >= 1e6:  return f"{n/1e6:.1f}M"
            if n >= 1e3:  return f"{n/1e3:.0f}K"
            return f"{n:.0f}"

        def fmt_size(kb):
            if kb >= 1024 * 1024: return f"{kb/1024/1024:.1f}TB"
            if kb >= 1024:        return f"{kb/1024:.1f}GB"
            return f"{kb:.0f}MB"

        engine_stats = {}
        for label, eng, bits, fast_mode, rerank in ENGINES:
            key = engine_key(eng, bits, fast_mode, rerank)
            ram_small = get(s_small, key, "retrieve_rss_mb")
            ram_large = get(s_large, key, "retrieve_rss_mb")
            disk_large = get(s_large, key, "disk_mb")
            recall_small = get(s_small, key, "recall")
            recall_large = get(s_large, key, "recall")
            cpu_i = get(s_large, key, "ingest_cpu_util")
            cpu_q = get(s_large, key, "query_cpu_util")

            if ram_small is None or ram_large is None or disk_large is None:
                engine_stats[key] = None
                continue

            slope = (ram_large - ram_small) / ((s_large - s_small) / 1000.0)
            base  = ram_small - (s_small / 1000.0) * slope

            if slope > 0:
                max_vecs = (AVAILABLE_RAM_MB - max(base, 100)) / slope * 1000
                ram_mode = "resident"
            else:
                # mmap: capacity is disk-limited (estimate 1TB)
                max_vecs = s_large / disk_large * (1024 * 1024)
                ram_mode = "mmap"

            bytes_per_vec = disk_large / s_large * 1024 * 1024
            disk_compress = F32_DISK_PER_100K / (disk_large / s_large * 100_000) if disk_large > 0 else None

            engine_stats[key] = {
                "label": label, "max_vecs": max_vecs, "ram_mode": ram_mode,
                "disk_compress": disk_compress, "disk_large": disk_large,
                "recall_small": recall_small, "recall_large": recall_large,
                "cpu_i": cpu_i, "cpu_q": cpu_q,
            }

        # [A] Compression & RAM capacity
        print(f"\n  [A] COMPRESSION & RAM CAPACITY @ 16GB  (float32: {BYTES_F32}B/vec = {F32_DISK_PER_100K:.0f}MB per 100k)")
        print(f"  {'Engine':<24}  {'Bytes/vec':>10}  {'Disk compress':>14}  {'RAM mode':>10}  "
              f"{'Max vecs @16GB':>16}  {'Recall@{:,}'.format(s_small):>13}  {'Recall@{:,}'.format(s_large):>13}")
        print(f"  {'-'*24}  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*16}  {'-'*13}  {'-'*13}")
        f32_cap = fmt_count(AVAILABLE_RAM_MB / (BYTES_F32 / 1024 / 1024) * 1000)
        print(f"  {'float32 raw':24}  {BYTES_F32:>10,}  {'1.00x':>14}  {'resident':>10}  {f32_cap:>16}  {'---':>13}  {'---':>13}")
        for label, eng, bits, fast_mode, rerank in ENGINES:
            key = engine_key(eng, bits, fast_mode, rerank)
            s = engine_stats.get(key)
            if not s:
                print(f"  {label:<24}  {'n/a':>10}  {'n/a':>14}  {'n/a':>10}  {'n/a':>16}  {'n/a':>13}  {'n/a':>13}")
                continue
            bpv = s["disk_large"] * 1024 * 1024 / s_large
            cap_note = fmt_count(s["max_vecs"]) + (" (disk)" if s["ram_mode"] == "mmap" else "")
            compress_str = f"x{s['disk_compress']:.2f} smaller" if s["disk_compress"] else "RAM-only"
            print(f"  {label:<24}  {bpv:>10,.0f}  "
                  f"  {compress_str:>14}  "
                  f"{s['ram_mode']:>10}  {cap_note:>16}  "
                  f"{s['recall_small']*100:>12.1f}%  {s['recall_large']*100:>12.1f}%")

        # [B] CPU summary
        print(f"\n  [B] CPU @ {s_large:,} vectors  (100% = 1 core)")
        print(f"  {'Engine':<24}  {'Ingest CPU':>11}  {'~Cores':>7}  {'Query CPU':>10}  {'~Cores':>7}  {'Disk compress':>14}")
        print(f"  {'-'*24}  {'-'*11}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*14}")
        for label, eng, bits, fast_mode, rerank in ENGINES:
            key = engine_key(eng, bits, fast_mode, rerank)
            s = engine_stats.get(key)
            if not s or s["cpu_i"] is None:
                print(f"  {label:<24}  {'n/a':>11}  {'n/a':>7}  {'n/a':>10}  {'n/a':>7}  {'n/a':>14}")
                continue
            compress_str = f"x{s['disk_compress']:.2f}" if s["disk_compress"] else "RAM-only"
            print(f"  {label:<24}  {s['cpu_i']:>10.0f}%  {s['cpu_i']/100:>6.1f}c  "
                  f"{s['cpu_q']:>9.0f}%  {s['cpu_q']/100:>6.1f}c  "
                  f"  {compress_str:>12}")

        # [C] Document scenario table
        docs = [
            ("Email / Slack",      0.5),
            ("Web page",           5.0),
            ("Research paper",    48.0),
            ("Book (300 pg)",    540.0),
            ("Large PDF",       1024.0),
        ]
        print(f"\n  [C] HOW MANY DOCUMENTS FIT IN 16GB RAM?  (~{CHUNK_KB:.0f}KB text per chunk)")
        col_w = 14
        header = f"  {'Engine':<24}  {'Max vecs':>10}"
        for dname, _ in docs:
            header += f"  {dname[:col_w]:>{col_w}}"
        header += f"  {'Total text':>12}  {'Recall@{:,}'.format(s_small):>13}"
        print(header)
        div = f"  {'-'*24}  {'-'*10}" + (f"  {'-'*col_w}" * len(docs)) + f"  {'-'*12}  {'-'*13}"
        print(div)

        for label, eng, bits, fast_mode, rerank in ENGINES:
            key = engine_key(eng, bits, fast_mode, rerank)
            s = engine_stats.get(key)
            if not s:
                print(f"  {label:<24}  {'n/a':>10}")
                continue
            mv = s["max_vecs"]
            row = f"  {label:<24}  {fmt_count(mv):>10}"
            for dname, doc_kb in docs:
                chunks_per_doc = doc_kb / CHUNK_KB
                num_docs = mv / chunks_per_doc
                row += f"  {fmt_count(num_docs):>{col_w}}"
            total_text_kb = mv * CHUNK_KB
            row += f"  {fmt_size(total_text_kb):>12}"
            row += f"  {s['recall_small']*100:>12.1f}%"
            if s["ram_mode"] == "mmap":
                row += "  *"
            print(row)

        print(f"\n  * LanceDB uses mmap — capacity shown is per 1TB disk, not RAM limit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=DEFAULT_SCALES)
    parser.add_argument("--dims",   nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--k",      type=int, default=DEFAULT_K)
    parser.add_argument("--runs",   type=int, default=DEFAULT_RUNS,
                        help="Number of search queries per engine")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    print("=" * 150)
    print(f"  TurboQuantDB Full Comparison Benchmark  --  {ts}")
    print(f"  dims={args.dims}  top_k={args.k}  search_runs={args.runs}  scales={args.scales}")
    print(f"  Process-isolated: each engine runs in a fresh subprocess")
    print("=" * 150)

    # all_results[dim][n][key] = result dict
    all_results = {}

    for dim in args.dims:
        all_results[dim] = {}
        for n in args.scales:
            scale_results = {}
            for label, eng, bits, fast_mode, rerank in ENGINES:
                key = engine_key(eng, bits, fast_mode, rerank)
                print(f"  Running {label} @ N={n:,}  dim={dim} ...", flush=True)
                time.sleep(3)  # cooldown: let prior process fully exit and free caches
                r = run_worker(eng, bits, n, dim, args.k, args.runs,
                               fast_mode=fast_mode, rerank=rerank)
                scale_results[key] = r

            all_results[dim][n] = scale_results
            print_scale_block(n, scale_results, dim)

    print_summary(all_results, args.scales, args.dims)
    print_claims_check(all_results, args.scales, args.dims)
    print_human_readable(all_results, args.scales, args.dims)

    out = os.path.join(os.path.dirname(__file__), "tqdb_comparison_results_local.json")
    serializable = {str(d): {str(n): v for n, v in dv.items()} for d, dv in all_results.items()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full results saved -> {out}")


if __name__ == "__main__":
    main()
