"""Vector store comparison benchmark — non-server/embedded mode only.

Engines tested (all embedded, no server required):
  1. TQDB             — this project
  2. turboquant-vecs  — turboquant-vectors (competitor; correct object-reuse API)
  3. ChromaDB         — chromadb, persistent client
  4. LanceDB          — lancedb, local persistent
  5. FAISS Flat       — faiss, exact brute-force (no compression)
  6. FAISS IVF-PQ     — faiss, ANN with product quantization
  7. Qdrant           — qdrant_client, in-memory mode
  8. Annoy            — annoy, ANN tree-based

Metrics per engine:
  build_s, p50/p95/p99/avg_ms, qps, recall@10,
  disk_mb, ram_build_mb, ram_query_mb, cpu_build_pct, cpu_query_pct

Usage:
    python benchmarks/_vector_store_compare.py [--case 1m_256|100k_768|all] [--engines all|tqdb,tqv,...]
"""

from __future__ import annotations
import sys, time, tempfile, os, json, argparse, gc, threading
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed — RAM/CPU metrics will be 0", flush=True)

N_REPEATS = 3
K = 10
BITS = 4
SEED = 7

_PROC = psutil.Process(os.getpid()) if HAS_PSUTIL else None


# ── Measurement helpers ───────────────────────────────────────────────────────

def rss_mb() -> float:
    """Current RSS in MB (page-granular; call after gc.collect() for best accuracy)."""
    if not HAS_PSUTIL:
        return 0.0
    return _PROC.memory_info().rss / (1024 * 1024)


class CpuSampler:
    """Background thread that samples per-process CPU% at ~100 ms intervals."""

    def __init__(self):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if not HAS_PSUTIL:
            return
        self._samples = []
        self._stop.clear()
        # Prime the cpu_percent counter (first call returns 0.0, so discard).
        _PROC.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            v = _PROC.cpu_percent(interval=0.1)
            self._samples.append(v)

    def stop(self) -> float:
        """Stop sampling and return mean CPU% (0.0 if psutil unavailable)."""
        if not HAS_PSUTIL or self._thread is None:
            return 0.0
        self._stop.set()
        self._thread.join(timeout=1.0)
        return float(np.mean(self._samples)) if self._samples else 0.0


def disk_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def median_metrics(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {k: round(float(np.median([r[k] for r in runs])), 4) for k in keys}


def compute_recall(results_ids: list[list[int]], true_top_k: np.ndarray, Q: int) -> float:
    hits = sum(
        len(set(results_ids[qi]) & set(true_top_k[qi].tolist()))
        for qi in range(Q)
    )
    return hits / (Q * K)


def timed_search(fn, queries, Q) -> tuple[list[float], list]:
    lats, results = [], []
    for qi in range(Q):
        t0 = time.perf_counter()
        res = fn(queries[qi])
        lats.append((time.perf_counter() - t0) * 1000.0)
        results.append(res)
    return lats, results


def summarize(lats, recall, build_s, disk_mb, ram_build, ram_query,
              cpu_build=0.0, cpu_query=0.0) -> dict:
    lats.sort()
    n = len(lats)
    avg = sum(lats) / n
    return dict(
        build_s=round(build_s, 3),
        p50_ms=round(lats[int(n * 0.50)], 3),
        p95_ms=round(lats[int(n * 0.95)], 3),
        p99_ms=round(lats[min(int(n * 0.99), n - 1)], 3),
        avg_ms=round(avg, 3),
        qps=round(1000.0 / avg if avg > 0 else 0, 2),
        recall10=round(recall, 4),
        disk_mb=round(disk_mb, 3),
        ram_build_mb=round(max(ram_build, 0.0), 2),
        ram_query_mb=round(max(ram_query, 0.0), 2),
        cpu_build_pct=round(cpu_build, 1),
        cpu_query_pct=round(cpu_query, 1),
    )


# ── TQDB ─────────────────────────────────────────────────────────────────────

def bench_tqdb(corpus, queries, true_top_k, N, D, Q):
    from tqdb import Database
    runs = []
    for rep in range(N_REPEATS):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc.collect(); rss0 = rss_mb()
            db = Database.open(tmpdir, dimension=D, bits=BITS, seed=SEED,
                               metric="cosine", rerank=False, fast_mode=True)
            ids = [str(i) for i in range(N)]
            cpu_s = CpuSampler(); cpu_s.start()
            t0 = time.perf_counter()
            db.insert_batch(ids, corpus, mode="insert")
            build_s = time.perf_counter() - t0
            cpu_build = cpu_s.stop()
            gc.collect(); rss1 = rss_mb()
            for qi in range(min(10, Q)):
                db.search(queries[qi], top_k=K)
            gc.collect(); rss2 = rss_mb()
            cpu_q = CpuSampler(); cpu_q.start()
            lats, res = timed_search(
                lambda q: [int(r["id"]) for r in db.search(q, top_k=K)], queries, Q)
            cpu_query = cpu_q.stop()
            gc.collect(); rss3 = rss_mb()
            recall = compute_recall(res, true_top_k, Q)
            db.close()
            # Measure disk AFTER close: segment files are deleted on clean close,
            # so this reflects the true at-rest footprint users see on disk.
            disk = disk_size_mb(tmpdir)
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    tqdb     rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB RAM_q={rss3-rss2:.0f}MB "
              f"CPU_b={cpu_build:.0f}% CPU_q={cpu_query:.0f}%", flush=True)
    return median_metrics(runs)


# ── turboquant-vectors (correct object-reuse API) ─────────────────────────────

def bench_tqv(corpus, queries, true_top_k, N, D, Q):
    """
    Correct TQV API: instantiate TurboQuantVectors once per repetition so the
    QR decomposition (O(D^3)) is paid only at build time, not per-query.
    Module-level search() re-creates the object every call — never use that path.
    """
    try:
        from turboquant_vectors import TurboQuantVectors
    except ImportError:
        print("    turboquant-vectors not installed — skipping", flush=True)
        return None
    runs = []
    for rep in range(N_REPEATS):
        gc.collect(); rss0 = rss_mb()
        # One TurboQuantVectors instance per rep — reuse the rotation matrix.
        tq = TurboQuantVectors(dim=D, bits=BITS, seed=SEED)
        cpu_s = CpuSampler(); cpu_s.start()
        t0 = time.perf_counter()
        compressed = tq.compress(corpus)
        build_s = time.perf_counter() - t0
        cpu_build = cpu_s.stop()
        gc.collect(); rss1 = rss_mb()

        # Disk: save compressed index and measure
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name
        try:
            compressed.save(tmp_path)
            disk = disk_size_mb(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Warmup
        for qi in range(min(10, Q)):
            tq.search(compressed, queries[qi], top_k=K)

        gc.collect(); rss2 = rss_mb()
        cpu_q = CpuSampler(); cpu_q.start()
        lats, res = timed_search(
            lambda q: tq.search(compressed, q, top_k=K)[0].tolist(),
            queries, Q)
        cpu_query = cpu_q.stop()
        gc.collect(); rss3 = rss_mb()
        recall = compute_recall(res, true_top_k, Q)
        del tq, compressed
        gc.collect()
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    tqv      rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB RAM_q={rss3-rss2:.0f}MB "
              f"CPU_b={cpu_build:.0f}% CPU_q={cpu_query:.0f}%", flush=True)
    return median_metrics(runs)


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def bench_chroma(corpus, queries, true_top_k, N, D, Q):
    import chromadb
    runs = []
    for rep in range(N_REPEATS):
        tmpdir_obj = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        tmpdir = tmpdir_obj.name
        try:
            gc.collect(); rss0 = rss_mb()
            client = chromadb.PersistentClient(path=tmpdir)
            col = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
            ids = [str(i) for i in range(N)]
            cpu_s = CpuSampler(); cpu_s.start()
            t0 = time.perf_counter()
            CHUNK = 5000
            for i in range(0, N, CHUNK):
                col.add(ids=ids[i:i+CHUNK], embeddings=corpus[i:i+CHUNK].tolist())
            build_s = time.perf_counter() - t0
            cpu_build = cpu_s.stop()
            gc.collect(); rss1 = rss_mb()
            disk = disk_size_mb(tmpdir)
            col.query(query_embeddings=[queries[0].tolist()], n_results=K)
            gc.collect(); rss2 = rss_mb()
            cpu_q = CpuSampler(); cpu_q.start()
            lats, res = timed_search(
                lambda q: [int(x) for x in col.query(query_embeddings=[q.tolist()], n_results=K)["ids"][0]],
                queries, Q)
            cpu_query = cpu_q.stop()
            gc.collect(); rss3 = rss_mb()
            recall = compute_recall(res, true_top_k, Q)
            del client, col
            gc.collect()
        finally:
            tmpdir_obj.cleanup()
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    chroma   rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB CPU_b={cpu_build:.0f}%", flush=True)
    return median_metrics(runs)


# ── LanceDB ───────────────────────────────────────────────────────────────────

def bench_lancedb(corpus, queries, true_top_k, N, D, Q):
    import lancedb
    import pyarrow as pa
    runs = []
    for rep in range(N_REPEATS):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc.collect(); rss0 = rss_mb()
            db = lancedb.connect(tmpdir)
            schema = pa.schema([
                pa.field("id", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), D)),
            ])
            cpu_s = CpuSampler(); cpu_s.start()
            t0 = time.perf_counter()
            data = [{"id": i, "vector": corpus[i].tolist()} for i in range(N)]
            tbl = db.create_table("bench", data=data, schema=schema)
            build_s = time.perf_counter() - t0
            cpu_build = cpu_s.stop()
            gc.collect(); rss1 = rss_mb()
            disk = disk_size_mb(tmpdir)
            tbl.search(queries[0]).metric("cosine").limit(K).to_list()
            gc.collect(); rss2 = rss_mb()
            cpu_q = CpuSampler(); cpu_q.start()
            lats, res = timed_search(
                lambda q: [r["id"] for r in tbl.search(q).metric("cosine").limit(K).to_list()],
                queries, Q)
            cpu_query = cpu_q.stop()
            gc.collect(); rss3 = rss_mb()
            recall = compute_recall(res, true_top_k, Q)
            del db, tbl
            gc.collect()
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    lancedb  rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB CPU_b={cpu_build:.0f}%", flush=True)
    return median_metrics(runs)


# ── FAISS Flat ────────────────────────────────────────────────────────────────

def bench_faiss_flat(corpus, queries, true_top_k, N, D, Q):
    import faiss
    runs = []
    for rep in range(N_REPEATS):
        gc.collect(); rss0 = rss_mb()
        cpu_s = CpuSampler(); cpu_s.start()
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(D)
        index.add(corpus)
        build_s = time.perf_counter() - t0
        cpu_build = cpu_s.stop()
        gc.collect(); rss1 = rss_mb()
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            tmp_path = f.name
        faiss.write_index(index, tmp_path)
        disk = disk_size_mb(tmp_path)
        os.unlink(tmp_path)
        index.search(queries[:5], K)
        gc.collect(); rss2 = rss_mb()
        cpu_q = CpuSampler(); cpu_q.start()
        lats, res = timed_search(
            lambda q: index.search(q.reshape(1, -1), K)[1][0].tolist(),
            queries, Q)
        cpu_query = cpu_q.stop()
        gc.collect(); rss3 = rss_mb()
        recall = compute_recall(res, true_top_k, Q)
        del index
        gc.collect()
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    faiss-f  rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB CPU_b={cpu_build:.0f}%", flush=True)
    return median_metrics(runs)


# ── FAISS IVF-PQ ──────────────────────────────────────────────────────────────

def bench_faiss_ivfpq(corpus, queries, true_top_k, N, D, Q):
    import faiss
    M = min(D, 32)
    # Faiss requires >= 39 * nlist training points; cap to avoid degenerate recall.
    nlist = min(4096, max(64, N // 40))
    nbits = BITS
    runs = []
    for rep in range(N_REPEATS):
        gc.collect(); rss0 = rss_mb()
        cpu_s = CpuSampler(); cpu_s.start()
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)
        index.train(corpus)
        index.add(corpus)
        index.nprobe = 32
        build_s = time.perf_counter() - t0
        cpu_build = cpu_s.stop()
        gc.collect(); rss1 = rss_mb()
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            tmp_path = f.name
        faiss.write_index(index, tmp_path)
        disk = disk_size_mb(tmp_path)
        os.unlink(tmp_path)
        index.search(queries[:5], K)
        gc.collect(); rss2 = rss_mb()
        cpu_q = CpuSampler(); cpu_q.start()
        lats, res = timed_search(
            lambda q: index.search(q.reshape(1, -1), K)[1][0].tolist(),
            queries, Q)
        cpu_query = cpu_q.stop()
        gc.collect(); rss3 = rss_mb()
        recall = compute_recall(res, true_top_k, Q)
        del index, quantizer
        gc.collect()
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    faiss-pq rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB CPU_b={cpu_build:.0f}%", flush=True)
    return median_metrics(runs)


# ── Qdrant in-memory ──────────────────────────────────────────────────────────

def bench_qdrant(corpus, queries, true_top_k, N, D, Q):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    runs = []
    # Detect which query API is available (v1.x: search; v1.7+: query_points).
    def _qdrant_search(client, q):
        if hasattr(client, "query_points"):
            resp = client.query_points(collection_name="bench", query=q.tolist(), limit=K)
            return [p.id for p in resp.points]
        return [r.id for r in client.search("bench", query_vector=q.tolist(), limit=K)]

    for rep in range(N_REPEATS):
        gc.collect(); rss0 = rss_mb()
        client = QdrantClient(":memory:")
        client.create_collection("bench", vectors_config=VectorParams(size=D, distance=Distance.COSINE))
        cpu_s = CpuSampler(); cpu_s.start()
        t0 = time.perf_counter()
        CHUNK = 1000
        for i in range(0, N, CHUNK):
            pts = [PointStruct(id=j, vector=corpus[j].tolist()) for j in range(i, min(i+CHUNK, N))]
            client.upsert("bench", pts)
        build_s = time.perf_counter() - t0
        cpu_build = cpu_s.stop()
        gc.collect(); rss1 = rss_mb()
        disk = 0.0  # in-memory
        for qi in range(min(5, Q)):
            _qdrant_search(client, queries[qi])
        gc.collect(); rss2 = rss_mb()
        cpu_q = CpuSampler(); cpu_q.start()
        lats, res = timed_search(lambda q: _qdrant_search(client, q), queries, Q)
        cpu_query = cpu_q.stop()
        gc.collect(); rss3 = rss_mb()
        recall = compute_recall(res, true_top_k, Q)
        client.delete_collection("bench")
        del client
        gc.collect()
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    qdrant   rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk=in-mem "
              f"RAM_b={rss1-rss0:.0f}MB CPU_b={cpu_build:.0f}%", flush=True)
    return median_metrics(runs)


# ── Annoy ─────────────────────────────────────────────────────────────────────

def bench_annoy(corpus, queries, true_top_k, N, D, Q):
    from annoy import AnnoyIndex
    N_TREES = 50
    SEARCH_K = 500
    runs = []
    for rep in range(N_REPEATS):
        gc.collect(); rss0 = rss_mb()
        cpu_s = CpuSampler(); cpu_s.start()
        t0 = time.perf_counter()
        idx = AnnoyIndex(D, "angular")
        for i in range(N):
            idx.add_item(i, corpus[i])
        idx.build(N_TREES)
        build_s = time.perf_counter() - t0
        cpu_build = cpu_s.stop()
        gc.collect(); rss1 = rss_mb()
        # Save to temp file to measure disk; delete after freeing the index (Windows file-lock).
        with tempfile.NamedTemporaryFile(suffix=".ann", delete=False) as f:
            tmp_path = f.name
        idx.save(tmp_path)
        disk = disk_size_mb(tmp_path)
        for qi in range(min(5, Q)):
            idx.get_nns_by_vector(queries[qi], K, search_k=SEARCH_K)
        gc.collect(); rss2 = rss_mb()
        cpu_q = CpuSampler(); cpu_q.start()
        lats, res = timed_search(
            lambda q: idx.get_nns_by_vector(q, K, search_k=SEARCH_K),
            queries, Q)
        cpu_query = cpu_q.stop()
        gc.collect(); rss3 = rss_mb()
        recall = compute_recall(res, true_top_k, Q)
        del idx
        gc.collect()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        runs.append(summarize(lats, recall, build_s, disk, rss1-rss0, rss3-rss2, cpu_build, cpu_query))
        print(f"    annoy    rep {rep+1}: p50={runs[-1]['p50_ms']:.2f}ms "
              f"recall={recall:.3f} disk={disk:.1f}MB "
              f"RAM_b={rss1-rss0:.0f}MB CPU_b={cpu_build:.0f}%", flush=True)
    return median_metrics(runs)


# ── Runner ────────────────────────────────────────────────────────────────────

ENGINES = {
    "tqdb":      bench_tqdb,
    "tqv":       bench_tqv,
    "chroma":    bench_chroma,
    "lancedb":   bench_lancedb,
    "faiss_flat": bench_faiss_flat,
    "faiss_pq":  bench_faiss_ivfpq,
    "qdrant":    bench_qdrant,
    "annoy":     bench_annoy,
}

CASES = {
    "100k_768": dict(N=100_000, D=768, Q=100),
    "1m_256":   dict(N=500_000, D=256, Q=50),
}

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="all", choices=list(CASES) + ["all"])
parser.add_argument("--engines", default="all")
parser.add_argument("--repeats", type=int, default=None,
                    help="Override N_REPEATS for this run (default: use module-level N_REPEATS=3)")
args = parser.parse_args()

if args.repeats is not None:
    N_REPEATS = args.repeats  # noqa: F811 — intentional module-level override

engines_to_run = (list(ENGINES.keys()) if args.engines == "all"
                  else [e.strip() for e in args.engines.split(",")])
cases_to_run = (list(CASES.items()) if args.case == "all"
                else [(args.case, CASES[args.case])])

all_results: dict = {}
for case_name, cfg in cases_to_run:
    N, D, Q = cfg["N"], cfg["D"], cfg["Q"]
    print(f"\n{'='*78}")
    print(f"CASE: {case_name}  N={N:,}  D={D}  Q={Q}  bits={BITS}  repeats={N_REPEATS}")
    print(f"{'='*78}", flush=True)

    rng = np.random.default_rng(SEED)
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scores_gt = corpus @ queries.T
    true_top_k = np.argsort(-scores_gt, axis=0)[:K].T

    case_results: dict = {}
    for eng_name in engines_to_run:
        if eng_name not in ENGINES:
            print(f"  Unknown engine: {eng_name}", flush=True)
            continue
        print(f"\n  -- {eng_name} --", flush=True)
        try:
            r = ENGINES[eng_name](corpus, queries, true_top_k, N, D, Q)
            if r is not None:
                case_results[eng_name] = r
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    all_results[case_name] = case_results

    # Results table — all metrics
    METRIC_COLS = [
        ("build_s",        "build_s",    "{:.1f}"),
        ("p50_ms",         "p50_ms",     "{:.2f}"),
        ("p95_ms",         "p95_ms",     "{:.2f}"),
        ("p99_ms",         "p99_ms",     "{:.2f}"),
        ("qps",            "qps",        "{:.1f}"),
        ("recall10",       "recall@10",  "{:.3f}"),
        ("disk_mb",        "disk_MB",    "{:.1f}"),
        ("ram_build_mb",   "RAM_bld_MB", "{:.0f}"),
        ("ram_query_mb",   "RAM_qry_MB", "{:.0f}"),
        ("cpu_build_pct",  "CPU_bld%",   "{:.0f}"),
        ("cpu_query_pct",  "CPU_qry%",   "{:.0f}"),
    ]
    col_names = ["engine"] + [c[1] for c in METRIC_COLS]
    eng_w = 14
    num_w = 10
    hdr_fmt = f"{{:<{eng_w}}}" + f"{{:>{num_w}}}" * len(METRIC_COLS)
    print(f"\n  {'RESULTS':^{eng_w + num_w * len(METRIC_COLS)}}")
    print("  " + hdr_fmt.format(*col_names))
    print("  " + "-" * (eng_w + num_w * len(METRIC_COLS)))
    for eng, r in sorted(case_results.items(), key=lambda x: x[1].get("p50_ms", 9999)):
        vals = [eng] + [fmt.format(r.get(key, float("nan"))) for key, _, fmt in METRIC_COLS]
        print("  " + hdr_fmt.format(*vals))

print("\n\n=== FULL JSON ===")
print(json.dumps(all_results, indent=2))
out = "benchmarks/_vector_store_compare_results.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out}")
