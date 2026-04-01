"""
Local 3-way benchmark with hard stop rules.

Stop rule for TurboQuantDB at each scale:
- ready/disk/ram/cpu/p50: stop if tq > 2x worst(lance, chroma)
- ingest speed (higher is better): stop if tq < 0.5x worst_speed, where worst_speed=min(lance, chroma)
"""

import gc
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

import chromadb
import lancedb
import numpy as np
import psutil
import pyarrow as pa
import turboquantdb

import sys
sys.stdout.reconfigure(encoding="utf-8")

DIM = 384
SCALES = [10_000, 25_000, 50_000, 100_000]
N_SEARCH = 30
TOP_K = 10
BATCH = 5000
TQDB_BITS = 8
RNG = np.random.default_rng(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_BASE = "/tmp/tqdb_bench"
os.makedirs(TMP_BASE, exist_ok=True)

TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models require large datasets",
    "Vector embeddings encode semantic meaning",
    "Retrieval-augmented generation improves accuracy",
    "Neural networks are inspired by biological brains",
    "Cosine similarity measures angle between vectors",
    "Apache Arrow provides a columnar in-memory format",
    "HNSW is a graph-based approximate nearest neighbour index",
    "IVF-PQ quantizes vectors using product quantization",
    "Brute force scan checks every vector in the table",
]


@dataclass
class CpuStats:
    peak: float
    avg: float


class CpuSampler:
    def __init__(self, interval: float = 0.1):
        self.proc = psutil.Process()
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.proc.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.samples.append(self.proc.cpu_percent(interval=None))
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def stats(self) -> CpuStats:
        if not self.samples:
            return CpuStats(0.0, 0.0)
        return CpuStats(max(self.samples), sum(self.samples) / len(self.samples))


def rand_f32(n):
    v = RNG.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rand_f64(n):
    v = RNG.standard_normal((n, DIM))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dir_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 / 1024


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def p50p95(fn, runs=N_SEARCH):
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    s = sorted(t)
    return s[len(s) // 2], s[int(len(s) * 0.95)]


def vps(n, ingest_s):
    return (n / ingest_s) if ingest_s > 0 else 0.0


def print_row(label, r, n, note=""):
    ready = r["ingest_s"] + r["build_s"]
    print(
        f"  {label:<24} ready={ready:>6.1f}s "
        f"ingest={r['ingest_s']:>5.1f}s ({vps(n, r['ingest_s']):>8,.0f} v/s) "
        f"idx={r['build_s']:>5.1f}s disk={r['disk_mb']:>7.2f}MB "
        f"ram={r['ram_mb']:>+7.1f}MB cpu_peak={r['cpu_peak']:>6.1f}% "
        f"p50={r['p50']:>6.1f}ms p95={r['p95']:>6.1f}ms"
        + (f" [{note}]" if note else "")
    )


def mkdtemp_local():
    return tempfile.mkdtemp(dir=TMP_BASE)


def run_lancedb(N):
    d = mkdtemp_local()
    tbl = None
    try:
        gc.collect()
        ram0 = rss_mb()
        with CpuSampler() as mon:
            db = lancedb.connect(d)
            t0 = time.perf_counter()
            for off in range(0, N, BATCH):
                bn = min(BATCH, N - off)
                batch = pa.table({
                    "id": [str(uuid.uuid4()) for _ in range(bn)],
                    "text": [TEXTS[(off + i) % len(TEXTS)] for i in range(bn)],
                    "vector": rand_f32(bn).tolist(),
                    "metadata": [json.dumps({"chunk": off + i}) for i in range(bn)],
                })
                if tbl is None:
                    tbl = db.create_table("axon", data=batch, mode="overwrite")
                else:
                    tbl.add(batch)
            ingest_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            tbl.create_index(
                index_type="IVF_PQ",
                metric="l2",
                num_partitions=min(64, max(8, N // 500)),
                num_sub_vectors=DIM // 16,
                replace=True,
            )
            build_s = time.perf_counter() - t1

            q = rand_f32(1)[0].tolist()
            p50, p95 = p50p95(lambda: tbl.search(q).limit(TOP_K).to_list())
        cpu = mon.stats()
        return {
            "ingest_s": ingest_s,
            "build_s": build_s,
            "disk_mb": dir_mb(d),
            "ram_mb": rss_mb() - ram0,
            "cpu_peak": cpu.peak,
            "cpu_avg": cpu.avg,
            "p50": p50,
            "p95": p95,
        }
    finally:
        try:
            del tbl
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)


def run_chromadb(N):
    d = mkdtemp_local()
    client = None
    col = None
    try:
        gc.collect()
        ram0 = rss_mb()
        with CpuSampler() as mon:
            client = chromadb.PersistentClient(path=d)
            col = client.get_or_create_collection("bench", metadata={"hnsw:space": "cosine"})
            t0 = time.perf_counter()
            for off in range(0, N, BATCH):
                bn = min(BATCH, N - off)
                col.add(
                    ids=[str(uuid.uuid4()) for _ in range(bn)],
                    embeddings=rand_f32(bn).tolist(),
                    documents=[TEXTS[(off + i) % len(TEXTS)] for i in range(bn)],
                    metadatas=[{"chunk": off + i} for i in range(bn)],
                )
            ingest_s = time.perf_counter() - t0
            q = rand_f32(1)[0].tolist()
            p50, p95 = p50p95(lambda: col.query(query_embeddings=[q], n_results=TOP_K))
        cpu = mon.stats()
        return {
            "ingest_s": ingest_s,
            "build_s": 0.0,
            "disk_mb": dir_mb(d),
            "ram_mb": rss_mb() - ram0,
            "cpu_peak": cpu.peak,
            "cpu_avg": cpu.avg,
            "p50": p50,
            "p95": p95,
        }
    finally:
        try:
            if client:
                client.reset()
        except Exception:
            pass
        del col, client
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)


def run_turboquantdb(N):
    d = mkdtemp_local()
    db = None
    db_path = os.path.join(d, "bench.tqdb")
    # Move large allocations before ram0 to measure DB overhead only
    vecs = rand_f64(N)
    ids = [str(uuid.uuid4()) for _ in range(N)]
    metadatas = [{"chunk": i} for i in range(N)]
    try:
        gc.collect()
        ram0 = rss_mb()
        with CpuSampler() as mon:
            db = turboquantdb.Database.open(db_path, dimension=DIM, bits=TQDB_BITS)

            t0 = time.perf_counter()
            db.insert_batch(ids=ids, vectors=vecs, metadatas=metadatas)
            db.flush()
            ingest_s = time.perf_counter() - t0

            stats_post_ingest = db.stats()
            ram_post_ingest = rss_mb() - ram0
            print(
                f"    [post-ingest] ram=+{ram_post_ingest:.1f}MB "
                f"live_codes={stats_post_ingest.get('live_codes_bytes', 'n/a')}B "
                f"live_ids={stats_post_ingest.get('live_id_count', 'n/a')} "
                f"meta_est={stats_post_ingest.get('metadata_bytes_estimate', 'n/a')}B "
                f"graph_nodes={stats_post_ingest.get('graph_nodes', 'n/a')}"
            )

            t1 = time.perf_counter()
            db.create_index()
            build_s = time.perf_counter() - t1

            stats_post_index = db.stats()
            ram_post_index = rss_mb() - ram0
            print(
                f"    [post-index]  ram=+{ram_post_index:.1f}MB "
                f"live_codes={stats_post_index.get('live_codes_bytes', 'n/a')}B "
                f"live_ids={stats_post_index.get('live_id_count', 'n/a')} "
                f"meta_est={stats_post_index.get('metadata_bytes_estimate', 'n/a')}B "
                f"graph_nodes={stats_post_index.get('graph_nodes', 'n/a')} "
                f"ann_slots={stats_post_index.get('ann_slot_count', 'n/a')}"
            )

            q = rand_f64(1)[0]
            p50, p95 = p50p95(lambda: db.search(query=q, top_k=TOP_K, use_ann=True))
        cpu = mon.stats()

        # Drop caller-side inputs before final RAM sample to report engine-only footprint.
        del vecs, ids, metadatas, q
        gc.collect()
        ram_engine_mb = rss_mb() - ram0

        stats = stats_post_index
        out = {
            "ingest_s": ingest_s,
            "build_s": build_s,
            "disk_mb": dir_mb(d),
            "ram_mb": ram_engine_mb,
            "cpu_peak": cpu.peak,
            "cpu_avg": cpu.avg,
            "p50": p50,
            "p95": p95,
            "bytes_per_vec": stats["total_disk_bytes"] / N,
            "segment_count": stats["segment_count"],
        }
        return out
    finally:
        try:
            if db:
                db.close()
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)


def tq_stop_reason(n, tq, l, c):
    l_ready = l["ingest_s"] + l["build_s"]
    c_ready = c["ingest_s"] + c["build_s"]
    t_ready = tq["ingest_s"] + tq["build_s"]

    checks = []
    checks.append(("time-to-ready", t_ready, 2.0 * max(l_ready, c_ready), "gt"))
    checks.append(("disk_mb", tq["disk_mb"], 2.0 * max(l["disk_mb"], c["disk_mb"]), "gt"))
    checks.append(("ram_mb", tq["ram_mb"], 2.0 * max(l["ram_mb"], c["ram_mb"]), "gt"))
    checks.append(("cpu_peak", tq["cpu_peak"], 2.0 * max(l["cpu_peak"], c["cpu_peak"]), "gt"))
    checks.append(("p50", tq["p50"], 2.0 * max(l["p50"], c["p50"]), "gt"))

    # speed lower-bound: if tq speed is < 0.5x worst competitor speed, stop.
    l_spd = vps(n, l["ingest_s"])
    c_spd = vps(n, c["ingest_s"])
    t_spd = vps(n, tq["ingest_s"])
    checks.append(("ingest_speed", t_spd, 0.5 * min(l_spd, c_spd), "lt"))

    reasons = []
    for name, val, lim, mode in checks:
        if mode == "gt" and val > lim:
            reasons.append(f"{name}: {val:.2f} > 2x limit {lim:.2f}")
        if mode == "lt" and val < lim:
            reasons.append(f"{name}: {val:.2f} < 0.5x floor {lim:.2f}")
    return reasons


def summary_table(all_results, title, fn):
    headers = ["LanceDB (IVF_PQ)", "ChromaDB (HNSW)", "TurboQuantDB (b8)"]
    keys = ["lance", "chroma", "tqdb"]
    print(f"\n  {title}")
    print(f"  {'':>20}  {'10k':>10}  {'25k':>10}  {'50k':>10}  {'100k':>10}")
    for label, key in zip(headers, keys):
        row = f"  {label:<20}"
        for n in SCALES:
            r = all_results.get(n, {}).get(key)
            row += f"  {fn(r, n) if r else '  stopped':>10}"
        print(row)


def main():
    print("Clean 3-Way Comparison: LanceDB (IVF_PQ) vs ChromaDB (HNSW) vs TurboQuantDB (bits=8)")
    print(f"  dim={DIM} top_k={TOP_K} search_runs={N_SEARCH} tqdb_bits={TQDB_BITS}")
    import turboquantdb
    print(f"  LanceDB {lancedb.__version__} | ChromaDB {chromadb.__version__} | TurboQuantDB {turboquantdb.__version__}")
    print("  Stop rule: if TurboQuantDB exceeds 2x worst competitor metric (or <0.5x speed), stop\n")

    all_results = {}
    tqdb_stopped = False

    for n in SCALES:
        print('-'*110)
        print(f"  N = {n:,} vectors")
        print('-'*110)

        r_lance = run_lancedb(n)
        print_row("LanceDB (IVF_PQ)", r_lance, n)

        r_chroma = run_chromadb(n)
        print_row("ChromaDB (HNSW)", r_chroma, n)

        all_results[n] = {"lance": r_lance, "chroma": r_chroma}

        if tqdb_stopped:
            print(f"  {'TurboQuantDB (bits=8)':<24} stopped at lower scale")
            all_results[n]["tqdb"] = None
            print()
            continue

        r_tqdb = run_turboquantdb(n)
        reasons = tq_stop_reason(n, r_tqdb, r_lance, r_chroma)

        note = f"{r_tqdb['bytes_per_vec']:.0f} B/vec segs={r_tqdb['segment_count']}"
        print_row("TurboQuantDB (bits=8)", r_tqdb, n, note=note)
        all_results[n]["tqdb"] = r_tqdb

        if reasons:
            print("  STOP: TurboQuantDB breached 2x guardrail:")
            for rs in reasons:
                print(f"    - {rs}")
            tqdb_stopped = True
        print()

    print(f"\n{'='*90}\n  SUMMARY\n{'='*90}")
    summary_table(all_results, "Time-to-ready (s) [lower=better]", lambda r, n: f"{r['ingest_s']+r['build_s']:.1f}s")
    summary_table(all_results, "Ingest speed (vec/s) [higher=better]", lambda r, n: f"{int(vps(n, r['ingest_s'])):,}")
    summary_table(all_results, "DB size on disk (MB) [lower=better]", lambda r, n: f"{r['disk_mb']:.1f}MB")
    summary_table(all_results, "RAM consumed (MB)", lambda r, n: f"{r['ram_mb']:+.0f}MB")
    summary_table(all_results, "CPU peak (%)", lambda r, n: f"{r['cpu_peak']:.0f}%")
    summary_table(all_results, "Retrieve p50 (ms) [lower=better]", lambda r, n: f"{r['p50']:.1f}ms")

    out = os.path.join(BASE_DIR, "tqdb_comparison_results_local.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
