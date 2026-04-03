"""
RAG Benchmark v3 — Two scenarios, multi-scale.

Scenario 1  (--scenario synthetic)
    Synthetic unit-gaussian vectors at scales 10K / 25K / 50K / 100K.
    Engines: ChromaDB, LanceDB, Qdrant, TQDB×4 presets, TurboDB.

Scenario 2  (--scenario realdocs)
    Real documents from --docs_dir, embedded with fastembed.
    Same engine sweep; scale = number of chunks from the corpus.

Metrics per run:
    Ingest time  /  Index time  /  Time-to-ready  /  Ingest speed (v/s)
    Disk (MB)
    RAM peak+avg (MB)  /  CPU peak+avg (%)  —  separate per phase
    Query p50 / p97 / p99 (ms)
    Recall@10  /  MRR  (vs brute-force cosine ground truth)

Usage:
    py -3.13 bench_v3.py --scenario synthetic
    py -3.13 bench_v3.py --scenario realdocs --docs_dir C:/docs
    py -3.13 bench_v3.py --scenario realdocs --docs_dir C:/docs \\
        --embed_cache C:/tmp/bench_cache
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import psutil

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_SCALES      = [10_000, 25_000, 50_000, 100_000]
DIM_SYNTHETIC       = 768          # must match embed model dim
N_QUERIES           = 100          # number of query vectors
TOP_K               = 10
CHUNK_SIZE          = 400
CHUNK_OVERLAP       = 80
COOLDOWN_S          = 3            # seconds between subprocess workers

# (label, engine_key, bits, fast_mode, rerank)
ENGINES = [
    ("ChromaDB",           "chromadb", None, False, True),
    ("LanceDB",            "lancedb",  None, False, True),
    ("Qdrant",             "qdrant",   None, False, True),
    ("TQDB b=8 HQ",        "tqdb",     8,    False, True),
    ("TQDB b=8 FastBuild", "tqdb",     8,    True,  False),
    ("TQDB b=4 Balanced",  "tqdb",     4,    False, True),
    ("TQDB b=4 FastBuild", "tqdb",     4,    True,  False),
    ("TurboDB",            "turbodb",  None, False, True),
]

PYTHON = sys.executable
WORKER = __file__

QUERIES_LITHO = [
    "What is the effect of shot noise in EUV lithography?",
    "How does acid diffusion affect critical dimension in chemically amplified resists?",
    "What are the stochastic limits to EUV scaling?",
    "What is the role of mask topography in EUV imaging?",
    "How does high-NA EUV improve resolution?",
    "What are the sources of line edge roughness in photoresist?",
    "How is optical proximity correction used in lithography simulation?",
    "What are the metrics for stochastic defects in EUV?",
    "How does crosslinking mechanism affect MOR photoresist performance?",
    "What are the fundamental limits of optical lithography?",
] * 10   # 100 queries total

QUERIES_SYNTHETIC = [
    "embedding query vector {}".format(i) for i in range(N_QUERIES)
]

# ── Resource sampler ──────────────────────────────────────────────────────────

class ResourceSampler:
    """Background thread: polls RSS + CPU% every 10 ms.

    Guarantees ≥2 samples via mandatory snapshots at start() and stop().
    """

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self._rss: list[float] = []
        self._cpu: list[float] = []
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread: threading.Thread | None = None

    def _snapshot(self) -> None:
        try:
            self._rss.append(self._proc.memory_info().rss / 1024 / 1024)
            self._cpu.append(self._proc.cpu_percent(interval=None))
        except Exception:
            pass

    def start(self) -> None:
        self._rss.clear()
        self._cpu.clear()
        self._stop.clear()
        self._proc.cpu_percent(interval=None)
        self._snapshot()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._snapshot()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._snapshot()

    def stats(self) -> dict:
        rss = self._rss or [0.0]
        cpu = self._cpu or [0.0]
        return {
            "rss_peak": float(max(rss)),
            "rss_avg":  float(sum(rss) / len(rss)),
            "cpu_peak": float(max(cpu)),
            "cpu_avg":  float(sum(cpu) / len(cpu)),
        }


# ── Text helpers ──────────────────────────────────────────────────────────────

def extract_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if os.path.getsize(path) == 0:
            return ""
        try:
            import pymupdf
            doc = pymupdf.open(path)
            text = "\n".join(
                p.get_text("text") for p in doc if p.get_text("text").strip()
            )
            doc.close()
            return text
        except Exception:
            return ""
    elif ext in (".md", ".txt", ".rst"):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Recall@K / MRR ───────────────────────────────────────────────────────────

def compute_recall_mrr(
    query_results: list[list[dict]],
    gt_top_k: np.ndarray,
    id_to_idx: dict[str, int],
    k: int,
) -> tuple[float, float]:
    recalls, mrrs = [], []
    for qi, hits in enumerate(query_results):
        gt_set = set(gt_top_k[qi].tolist())
        retrieved = [id_to_idx.get(h.get("id", ""), -1) for h in hits]
        recalls.append(
            sum(1 for idx in retrieved if idx in gt_set) / min(k, len(gt_set))
        )
        mrr = 0.0
        for rank, idx in enumerate(retrieved, 1):
            if idx in gt_set:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)
    return float(np.mean(recalls)), float(np.mean(mrrs))


# ── Worker (subprocess entry point) ──────────────────────────────────────────

def run_worker(args: argparse.Namespace) -> None:
    """Runs in a fresh subprocess. Prints one JSON line to stdout."""
    engine    = args.worker_engine
    top_k     = args.top_k
    bits      = args.bits
    fast_mode = bool(args.worker_fast_mode)
    rerank    = bool(args.worker_rerank)
    data_dir  = args.worker_data_dir

    # Boost process priority (best-effort)
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
    except Exception:
        pass

    vecs   = np.load(os.path.join(data_dir, "embeddings.npy"))
    qvecs  = np.load(os.path.join(data_dir, "query_vecs.npy"))
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ids       = meta["ids"]
    sources   = meta["sources"]
    documents = meta["documents"]
    N, DIM    = vecs.shape

    sampler   = ResourceSampler()
    tmp       = tempfile.mkdtemp(prefix=f"rag_{engine}_")
    ingest_stats = index_stats = query_stats = {}
    ingest_time = index_time = 0.0
    latencies: list[float] = []
    query_results: list[list[dict]] = []

    try:
        gc.collect()

        # ── Phase 1: Ingest ───────────────────────────────────────────────────
        sampler.start()
        t0 = time.perf_counter()

        if engine == "chromadb":
            import chromadb
            _cli = chromadb.PersistentClient(path=tmp)
            col  = _cli.create_collection("vecs", metadata={"hnsw:space": "ip"})
            for i in range(0, N, 500):
                end = min(i + 500, N)
                col.add(
                    ids=ids[i:end],
                    embeddings=vecs[i:end].tolist(),
                    documents=documents[i:end],
                    metadatas=[{"src": sources[j]} for j in range(i, end)],
                )

        elif engine == "lancedb":
            import lancedb
            db    = lancedb.connect(tmp)
            table = db.create_table("vecs", data=[
                {"vector": vecs[i].tolist(), "id": ids[i],
                 "src": sources[i], "doc": documents[i]}
                for i in range(N)
            ])

        elif engine == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, HnswConfigDiff, OptimizersConfigDiff,
                PointStruct, SearchParams, VectorParams,
            )
            _qcli = QdrantClient(path=os.path.join(tmp, "qdrant"))
            _qcli.create_collection(
                "bench",
                vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=N + 1),
            )
            for i in range(0, N, 500):
                end = min(i + 500, N)
                _qcli.upsert("bench", points=[
                    PointStruct(id=j, vector=vecs[j].tolist(),
                                payload={"src": sources[j], "doc": documents[j]})
                    for j in range(i, end)
                ])

        elif engine == "tqdb":
            import turboquantdb
            db = turboquantdb.TurboQuantDB.open(
                os.path.join(tmp, "r.tqdb"),
                dimension=DIM, bits=bits, metric="ip",
                rerank=rerank, fast_mode=fast_mode,
            )
            db.insert_batch(
                ids, vecs,
                metadatas=[{"src": sources[i]} for i in range(N)],
                documents=documents,
            )

        elif engine == "turbodb":
            import turbodb as _turbodb
            _tdb = _turbodb.TurboDB(tmp)
            _col = _tdb.create_collection("v", dim=DIM)
            for i in range(0, N, 500):
                end = min(i + 500, N)
                _col.add(
                    ids=ids[i:end],
                    vectors=vecs[i:end],
                    metadatas=[{"src": sources[j]} for j in range(i, end)],
                    documents=documents[i:end],
                )

        ingest_time  = time.perf_counter() - t0
        sampler.stop()
        ingest_stats = sampler.stats()
        gc.collect()

        # ── Phase 2: Index build ──────────────────────────────────────────────
        sampler.start()
        t1 = time.perf_counter()

        if engine == "lancedb":
            nlist = min(256, max(4, N // 40))
            table.create_index(
                num_partitions=nlist,
                num_sub_vectors=max(1, DIM // 8),
            )

        elif engine == "qdrant":
            _qcli.update_collection(
                "bench",
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
            )
            while True:
                if _qcli.get_collection("bench").status.value == "green":
                    break
                time.sleep(0.5)

        elif engine == "tqdb":
            db.create_index(max_degree=32, ef_construction=200,
                            search_list_size=128)

        # chromadb and turbodb: index built during ingest (no separate phase)

        index_time  = time.perf_counter() - t1
        sampler.stop()
        index_stats = sampler.stats()
        gc.collect()

        # Disk footprint after full index
        disk_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fnames in os.walk(tmp)
            for fn in fnames
            if not os.path.islink(os.path.join(dp, fn))
        ) / 1024 / 1024

        # ── Phase 3: Query ────────────────────────────────────────────────────
        sampler.start()

        for qvec in qvecs:
            t_q = time.perf_counter()

            if engine == "chromadb":
                res = col.query(
                    query_embeddings=[qvec.tolist()],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                hits = [
                    {"id": res["ids"][0][j],
                     "score": float(1 - res["distances"][0][j]),
                     "src": res["metadatas"][0][j].get("src", "")}
                    for j in range(len(res["ids"][0]))
                ]

            elif engine == "lancedb":
                rows = table.search(qvec).limit(top_k).nprobes(20).to_list()
                hits = [
                    {"id": r["id"], "score": float(r["_distance"]),
                     "src": r["src"]}
                    for r in rows
                ]

            elif engine == "qdrant":
                res = _qcli.query_points(
                    "bench", query=qvec.tolist(), limit=top_k,
                    search_params=SearchParams(hnsw_ef=128),
                    with_payload=True,
                )
                hits = [
                    {"id": ids[p.id], "score": float(p.score),
                     "src": p.payload.get("src", "")}
                    for p in res.points
                ]

            elif engine == "tqdb":
                res  = db.search(qvec, top_k=top_k, ann_search_list_size=128)
                hits = [
                    {"id": r.get("id", ""), "score": float(r.get("score", 0)),
                     "src": r.get("metadata", {}).get("src", "")}
                    for r in res
                ]

            elif engine == "turbodb":
                res  = _col.query(vector=qvec, k=top_k)
                hits = [
                    {"id": r.id, "score": float(r.score),
                     "src": (r.metadata or {}).get("src", "")}
                    for r in res
                ]

            latencies.append((time.perf_counter() - t_q) * 1000)
            query_results.append(hits)

        sampler.stop()
        query_stats = sampler.stats()

        lats = sorted(latencies)
        print(json.dumps({
            "n": N, "dim": DIM,
            "ingest_time":  ingest_time,
            "index_time":   index_time,
            "ready_time":   ingest_time + index_time,
            "ingest_speed": N / ingest_time if ingest_time > 0 else 0,
            "disk_mb":      disk_mb,
            "ingest_rss_peak": ingest_stats.get("rss_peak", 0),
            "ingest_rss_avg":  ingest_stats.get("rss_avg", 0),
            "ingest_cpu_peak": ingest_stats.get("cpu_peak", 0),
            "ingest_cpu_avg":  ingest_stats.get("cpu_avg", 0),
            "index_rss_peak":  index_stats.get("rss_peak", 0),
            "index_rss_avg":   index_stats.get("rss_avg", 0),
            "index_cpu_peak":  index_stats.get("cpu_peak", 0),
            "index_cpu_avg":   index_stats.get("cpu_avg", 0),
            "query_rss_peak":  query_stats.get("rss_peak", 0),
            "query_rss_avg":   query_stats.get("rss_avg", 0),
            "query_cpu_peak":  query_stats.get("cpu_peak", 0),
            "query_cpu_avg":   query_stats.get("cpu_avg", 0),
            "p50_ms":  float(np.percentile(lats, 50)),
            "p97_ms":  float(np.percentile(lats, 97)),
            "p99_ms":  float(np.percentile(lats, 99)),
            "query_results": query_results,
        }), flush=True)

    finally:
        if not sampler._stop.is_set():
            sampler.stop()
        gc.collect()
        if engine == "chromadb":
            try:
                del col, _cli
                gc.collect()
                time.sleep(0.3)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


# ── Spawn helper ──────────────────────────────────────────────────────────────

def spawn_worker(engine: str, bits: int | None, data_dir: str,
                 fast_mode: bool, rerank: bool,
                 timeout: float = 3600.0) -> dict | None:
    cmd = [
        PYTHON, WORKER,
        "--worker_engine",    engine,
        "--worker_data_dir",  data_dir,
        "--top_k",            str(TOP_K),
        "--bits",             str(bits or 8),
        "--worker_fast_mode", str(int(fast_mode)),
        "--worker_rerank",    str(int(rerank)),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8")
        for line in proc.stdout.strip().splitlines():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        if proc.stderr:
            print(f"  [ERROR] {engine}:\n{proc.stderr[-1000:]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT >{timeout:.0f}s]", file=sys.stderr)
        return None


# ── Table helpers ─────────────────────────────────────────────────────────────

def fmt(v, unit="", d=1):
    if v is None:
        return "n/a"
    return f"{v:.{d}f}{unit}"


def print_scale_table(label: str, N: int, DIM: int,
                      scale_results: dict[str, dict | None]) -> None:
    raw_mb = N * DIM * 4 / 1024 / 1024
    bar = "=" * 140
    print(f"\n{bar}")
    print(f"  {label}  |  N={N:,}  dim={DIM}  top_k={TOP_K}  "
          f"raw_float32={raw_mb:.1f}MB")
    print(bar)
    hdr1 = (f"  {'Engine':<22}  "
            f"{'Ingest':>7}  {'Index':>7}  {'Ready':>7}  {'v/s':>7}  "
            f"{'Disk':>8}  "
            f"{'i-RAM pk':>9}  {'i-CPU avg':>9}  "
            f"{'idx-RAM pk':>10}  {'idx-CPU avg':>11}  "
            f"{'q-RAM pk':>9}  {'q-CPU avg':>9}  "
            f"{'p50':>7}  {'p97':>7}  {'p99':>7}  "
            f"{'R@10':>6}  {'MRR':>6}")
    hdr2 = (f"  {'':22}  "
            f"{'(s)':>7}  {'(s)':>7}  {'(s)':>7}  {'(v/s)':>7}  "
            f"{'(MB)':>8}  "
            f"{'(MB)':>9}  {'(%)':>9}  "
            f"{'(MB)':>10}  {'(%)':>11}  "
            f"{'(MB)':>9}  {'(%)':>9}  "
            f"{'(ms)':>7}  {'(ms)':>7}  {'(ms)':>7}  "
            f"{'':>6}  {'':>6}")
    sep = "-" * len(hdr1)
    print(hdr1)
    print(hdr2)
    print(f"  {sep}")
    for label_e, *_ in ENGINES:
        r = scale_results.get(label_e)
        if not r:
            print(f"  {label_e:<22}  FAILED/SKIPPED")
            continue
        print(
            f"  {label_e:<22}  "
            f"{fmt(r['ingest_time'], 's'):>7}  "
            f"{fmt(r['index_time'],  's'):>7}  "
            f"{fmt(r['ready_time'],  's'):>7}  "
            f"{int(r['ingest_speed']):>7,}  "
            f"{fmt(r['disk_mb'],          'MB'):>8}  "
            f"{fmt(r['ingest_rss_peak'],  'MB', 0):>9}  "
            f"{fmt(r['ingest_cpu_avg'],   '%',  0):>9}  "
            f"{fmt(r['index_rss_peak'],   'MB', 0):>10}  "
            f"{fmt(r['index_cpu_avg'],    '%',  0):>11}  "
            f"{fmt(r['query_rss_peak'],   'MB', 0):>9}  "
            f"{fmt(r['query_cpu_avg'],    '%',  0):>9}  "
            f"{fmt(r['p50_ms'], 'ms', 2):>7}  "
            f"{fmt(r['p97_ms'], 'ms', 2):>7}  "
            f"{fmt(r['p99_ms'], 'ms', 2):>7}  "
            f"{r.get('recall', 0):>6.3f}  "
            f"{r.get('mrr', 0):>6.3f}"
        )


# ── Embed + cache helpers ─────────────────────────────────────────────────────

def load_or_embed(texts: list[str], queries: list[str],
                  model: str, cache_dir: str | None,
                  cache_tag: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (vecs, qvecs, embed_time). Reads cache if available."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, f"meta_{cache_tag}.json")
        vec_path  = os.path.join(cache_dir, f"vecs_{cache_tag}.npy")
        qvec_path = os.path.join(cache_dir, f"qvecs_{cache_tag}.npy")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                cmeta = json.load(f)
            if cmeta.get("model") == model and cmeta.get("n") == len(texts):
                print(f"    [cache] {cache_tag}: {len(texts):,} vecs  "
                      f"(saved {cmeta.get('embed_time', 0):.1f}s)")
                return (np.load(vec_path), np.load(qvec_path),
                        cmeta.get("embed_time", 0))

    print(f"    Embedding {len(texts):,} texts with {model} ...", flush=True)
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model)

    t0   = time.perf_counter()
    vecs = np.array(list(embedder.embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    qvecs = np.array(list(embedder.embed(queries)), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9
    t = time.perf_counter() - t0

    if cache_dir:
        np.save(vec_path,  vecs)
        np.save(qvec_path, qvecs)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "n": len(texts), "embed_time": t}, f)

    print(f"    Embedded in {t:.1f}s  dim={vecs.shape[1]}", flush=True)
    return vecs, qvecs, t


def make_data_dir(vecs: np.ndarray, qvecs: np.ndarray,
                  ids: list[str], sources: list[str],
                  documents: list[str]) -> str:
    """Serialise shared data to a temp dir for worker subprocesses."""
    data_dir = tempfile.mkdtemp(prefix="rag_data_")
    np.save(os.path.join(data_dir, "embeddings.npy"), vecs)
    np.save(os.path.join(data_dir, "query_vecs.npy"), qvecs)
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "sources": sources, "documents": documents}, f)
    return data_dir


def run_all_engines(data_dir: str, gt_top_k: np.ndarray,
                    id_to_idx: dict[str, int]) -> dict[str, dict | None]:
    """Run each engine in a subprocess with a dynamic 5× median timeout.

    After every successful run, the timeout is recalculated as
    max(120, 5 × median(completed_ready_times)).  Engines that exceed this
    are killed and recorded as None so the benchmark can continue.
    """
    results: dict[str, dict | None] = {}
    completed_times: list[float] = []

    for label, engine, bits, fast_mode, rerank in ENGINES:
        # Dynamic timeout: 5× median of successful runs so far, floor 120 s
        if completed_times:
            dyn_timeout = max(120.0, 5.0 * float(np.median(completed_times)))
        else:
            dyn_timeout = 3600.0   # no reference yet — be generous for first run

        print(f"  → {label} (limit={dyn_timeout:.0f}s)", end=" ... ", flush=True)
        time.sleep(COOLDOWN_S)
        r = spawn_worker(engine, bits, data_dir, fast_mode, rerank,
                         timeout=dyn_timeout)
        if r:
            recall, mrr = compute_recall_mrr(
                r.get("query_results", []), gt_top_k, id_to_idx, TOP_K
            )
            r["recall"] = recall
            r["mrr"]    = mrr
            completed_times.append(r["ready_time"])
            print(f"ready={r['ready_time']:.1f}s  "
                  f"p50={r['p50_ms']:.2f}ms  "
                  f"R@{TOP_K}={recall:.3f}  mrr={mrr:.3f}")
        else:
            print("TIMEOUT/FAILED")
        results[label] = r
    return results


# ── Scenario 1: Synthetic ─────────────────────────────────────────────────────

def run_synthetic(scales: list[int], dim: int, model: str,
                  cache_dir: str | None, out: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}")
    print("  SCENARIO 1 — SYNTHETIC VECTORS")
    print(f"  Scales: {scales}  dim={dim}  queries={N_QUERIES}  top_k={TOP_K}")
    print(bar)

    # Generate a single large pool and slice per scale
    max_n = max(scales)
    print(f"\n  Generating {max_n:,} random unit-gaussian vectors (dim={dim}) ...")
    rng = np.random.default_rng(42)

    all_results: dict[int, dict[str, dict | None]] = {}

    for scale in scales:
        print(f"\n{'─' * 70}")
        print(f"  Scale = {scale:,}")
        print(f"{'─' * 70}")

        cache_tag = f"synth_n{scale}_d{dim}"
        vec_path  = (os.path.join(cache_dir, f"vecs_{cache_tag}.npy")
                     if cache_dir else None)
        qvec_path = (os.path.join(cache_dir, f"qvecs_{cache_tag}.npy")
                     if cache_dir else None)

        # synthetic: generate deterministically by seed
        if cache_dir and vec_path and os.path.exists(vec_path):
            print(f"    [cache] Loading synthetic vecs for n={scale:,}")
            vecs  = np.load(vec_path)
            qvecs = np.load(qvec_path)
        else:
            vecs = rng.standard_normal((scale, dim)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            qvecs = rng.standard_normal((N_QUERIES, dim)).astype(np.float32)
            qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(vec_path,  vecs)
                np.save(qvec_path, qvecs)

        ids       = [str(i) for i in range(scale)]
        sources   = ["synth"] * scale
        documents = [""] * scale

        print(f"    Computing brute-force GT ...", end=" ", flush=True)
        scores_gt = qvecs @ vecs.T
        gt_top_k  = np.argsort(-scores_gt, axis=1)[:, :TOP_K]
        id_to_idx = {str(i): i for i in range(scale)}
        print("done")

        data_dir_tmp = make_data_dir(vecs, qvecs, ids, sources, documents)
        try:
            scale_results = run_all_engines(data_dir_tmp, gt_top_k, id_to_idx)
        finally:
            shutil.rmtree(data_dir_tmp, ignore_errors=True)

        all_results[scale] = scale_results
        print_scale_table(f"Synthetic N={scale:,}", scale, dim, scale_results)

    # Persist full results
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"scenario": "synthetic", "scales": scales,
                   "results": {str(k): v for k, v in all_results.items()}}, f)
    print(f"\n  Full results → {out}")


# ── Scenario 2: Real docs ─────────────────────────────────────────────────────

def run_realdocs(docs_dir: str, model: str,
                 cache_dir: str | None, out: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}")
    print("  SCENARIO 2 — REAL DOCUMENTS")
    print(f"  Source: {docs_dir}  model={model}  top_k={TOP_K}")
    print(bar)

    # Load + chunk
    EXTS = (".pdf", ".md", ".txt", ".rst")
    file_paths = sorted(
        os.path.join(dp, fn)
        for dp, _, fnames in os.walk(docs_dir)
        for fn in fnames
        if fn.lower().endswith(EXTS) and "__pycache__" not in dp
    )
    print(f"\n  Files: {len(file_paths)}")

    all_chunks: list[tuple[str, str, str]] = []   # (id, text, source)
    for fpath in file_paths:
        rel  = os.path.relpath(fpath, docs_dir).replace("\\", "/")
        text = extract_file(fpath)
        if not text.strip():
            continue
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append((f"{rel}::{i}", chunk, rel))

    N = len(all_chunks)
    print(f"  Chunks: {N:,}")

    queries  = QUERIES_LITHO[:N_QUERIES]
    texts    = [c[1] for c in all_chunks]
    ids      = [c[0] for c in all_chunks]
    sources  = [c[2] for c in all_chunks]
    cache_tag = f"real_n{N}_m{model.replace('/', '_')}"

    vecs, qvecs, _ = load_or_embed(texts, queries, model, cache_dir, cache_tag)
    DIM = vecs.shape[1]

    print(f"  Computing brute-force GT ...", end=" ", flush=True)
    scores_gt = qvecs @ vecs.T
    gt_top_k  = np.argsort(-scores_gt, axis=1)[:, :TOP_K]
    id_to_idx = {c[0]: idx for idx, c in enumerate(all_chunks)}
    print("done")

    data_dir_tmp = make_data_dir(vecs, qvecs, ids, sources,
                                  [c[1] for c in all_chunks])
    try:
        results = run_all_engines(data_dir_tmp, gt_top_k, id_to_idx)
    finally:
        shutil.rmtree(data_dir_tmp, ignore_errors=True)

    print_scale_table(f"Real docs  ({docs_dir})", N, DIM, results)

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"scenario": "realdocs", "n": N, "dim": DIM,
                   "results": results}, f)
    print(f"\n  Full results → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="RAG benchmark v3 — synthetic + real-doc scenarios")

    # mode
    p.add_argument("--scenario", choices=["synthetic", "realdocs"],
                   default=None,
                   help="Which scenario to run (required unless --worker_engine set)")

    # synthetic
    p.add_argument("--scales", nargs="+", type=int, default=DEFAULT_SCALES,
                   metavar="N", help="Vector counts for synthetic benchmark "
                   f"(default: {DEFAULT_SCALES})")
    p.add_argument("--dim", type=int, default=DIM_SYNTHETIC,
                   help=f"Embedding dimension (default: {DIM_SYNTHETIC})")

    # real docs
    p.add_argument("--docs_dir", default=None,
                   help="Directory of PDF/MD/TXT files for the realdocs scenario")
    p.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL,
                   help=f"fastembed model (default: {DEFAULT_EMBED_MODEL})")

    # shared
    p.add_argument("--embed_cache", default=None,
                   help="Directory to cache embeddings across runs")
    p.add_argument("--out", default="bench_v3_results.json",
                   help="Output JSON file (default: bench_v3_results.json)")
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--bits",  type=int, default=8)

    # worker (internal — do not pass manually)
    p.add_argument("--worker_engine",    default=None)
    p.add_argument("--worker_data_dir",  default=None)
    p.add_argument("--worker_fast_mode", type=int, default=0)
    p.add_argument("--worker_rerank",    type=int, default=1)

    args = p.parse_args()

    if args.worker_engine:
        run_worker(args)
        return

    if args.scenario == "synthetic":
        run_synthetic(args.scales, args.dim, args.embed_model,
                      args.embed_cache, args.out)

    elif args.scenario == "realdocs":
        if not args.docs_dir:
            p.error("--docs_dir is required for the realdocs scenario")
        run_realdocs(args.docs_dir, args.embed_model,
                     args.embed_cache, args.out)

    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
