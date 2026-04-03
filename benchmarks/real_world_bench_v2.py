"""
Real-world RAG benchmark v2 — all vector stores, same documents, same queries.

New in v2 vs v1:
  - Per-phase (ingest / index / query) CPU peak+avg and RAM peak+avg via
    background sampler thread (50 ms interval)
  - Separate ingest-data time and index-build time (→ time-to-ready = sum)
  - p50 / p97 / p99 latency (replaces p95)
  - Recall@K and MRR against brute-force ground truth (numpy dot product)
  - Raw vector disk column (N × dim × 4 bytes)
  - Process isolation: each engine runs in a fresh subprocess
  - Results saved to bench_results_v2.json

Usage:
    py -3.13 benchmarks/real_world_bench_v2.py --folder "C:/dev/studio_brain_open/Qualification/papers"
    py -3.13 benchmarks/real_world_bench_v2.py --folder "..." --top_k 10 --bits 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import gc

import numpy as np
import pymupdf
import psutil
from fastembed import TextEmbedding

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

# Default queries (lithography domain — used when --folder is the source)
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
]

# Queries for biomedical/general corpora (trec-covid / general HF datasets)
QUERIES_BIO = [
    "What are the symptoms of COVID-19 infection?",
    "How does SARS-CoV-2 spread between individuals?",
    "What treatments are effective against coronavirus?",
    "What is the role of the ACE2 receptor in viral entry?",
    "How does the immune response fight respiratory viruses?",
    "What are the risk factors for severe COVID-19 outcomes?",
    "How do mRNA vaccines generate immune protection?",
    "What is the mechanism of action of antiviral drugs?",
    "What are the long-term effects of COVID-19 infection?",
    "How does viral mutation affect vaccine efficacy?",
]

PYTHON = sys.executable
WORKER = __file__

# ── Text extraction & chunking ────────────────────────────────────────────────

def extract_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if os.path.getsize(path) == 0:
            return ""
        try:
            doc = pymupdf.open(path)
            text = "\n".join(p.get_text("text") for p in doc if p.get_text("text").strip())
            doc.close()
            return text
        except Exception:
            return ""
    elif ext in (".md", ".txt"):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Background resource sampler ───────────────────────────────────────────────

class ResourceSampler:
    """Polls RSS + CPU% every `interval` seconds in a background thread.

    Always captures at least one sample at start() and stop() so even
    sub-interval phases (e.g. a 30ms query loop) have valid readings.

    Usage:
        s = ResourceSampler(); s.start()
        ... do work ...
        s.stop(); stats = s.stats()
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
        self._proc.cpu_percent(interval=None)   # prime the counter
        self._snapshot()                        # guaranteed baseline sample
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._snapshot()                        # guaranteed final sample

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


# ── Worker ────────────────────────────────────────────────────────────────────

def run_worker_mode(args):
    """Called when --worker_engine is set. Prints one JSON line to stdout."""
    engine   = args.worker_engine
    top_k    = args.top_k
    bits     = args.bits
    data_dir = args.worker_data_dir

    # Raise process priority (best-effort)
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass

    # Load pre-computed data
    vecs      = np.load(os.path.join(data_dir, "embeddings.npy"))
    qvecs     = np.load(os.path.join(data_dir, "query_vecs.npy"))
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ids       = meta["ids"]
    sources   = meta["sources"]
    documents = meta["documents"]
    N, DIM    = vecs.shape

    sampler = ResourceSampler(interval=0.05)
    tmp = tempfile.mkdtemp(prefix=f"rag_{engine}_")

    ingest_stats = index_stats = query_stats = {}
    ingest_time = index_time = 0.0
    latencies: list[float] = []
    query_results: list[list[dict]] = []

    try:
        gc.collect()

        # ── Phase 1: INGEST ────────────────────────────────────────────────────
        sampler.start()
        t0 = time.perf_counter()

        if engine == "chromadb":
            import chromadb
            _client = chromadb.PersistentClient(path=tmp)
            col = _client.create_collection("vecs", metadata={"hnsw:space": "ip"})
            for i in range(0, N, 500):
                end = min(i + 500, N)
                col.add(
                    ids=ids[i:end],
                    embeddings=vecs[i:end].tolist(),
                    documents=documents[i:end],
                    metadatas=[{"source": sources[j]} for j in range(i, end)],
                )

        elif engine == "lancedb":
            import lancedb
            db = lancedb.connect(tmp)
            table = db.create_table("v", data=[
                {"vector": vecs[i].tolist(), "id": ids[i],
                 "source": sources[i], "doc": documents[i]}
                for i in range(N)])

        elif engine == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import (Distance, HnswConfigDiff,
                                               OptimizersConfigDiff, PointStruct,
                                               SearchParams, VectorParams)
            _qclient = QdrantClient(path=os.path.join(tmp, "qdrant"))
            _qclient.create_collection(
                "bench",
                vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=N + 1),
            )
            for i in range(0, N, 500):
                end = min(i + 500, N)
                _qclient.upsert("bench", points=[
                    PointStruct(id=j, vector=vecs[j].tolist(),
                                payload={"source": sources[j], "doc": documents[j]})
                    for j in range(i, end)])

        elif engine == "tqdb":
            import turboquantdb
            db_path = os.path.join(tmp, "rag.tqdb")
            db = turboquantdb.TurboQuantDB.open(
                db_path, dimension=DIM, bits=bits, metric="ip", rerank=True)
            metadatas = [{"source": sources[i]} for i in range(N)]
            db.insert_batch(ids, vecs, metadatas=metadatas, documents=documents)

        elif engine == "turbodb":
            import turbodb as _turbodb
            _tdb = _turbodb.TurboDB(tmp)
            _col = _tdb.create_collection("vecs", dim=DIM)
            metadatas_td = [{"source": sources[i]} for i in range(N)]
            for i in range(0, N, 500):
                end = min(i + 500, N)
                _col.add(
                    ids=ids[i:end],
                    vectors=vecs[i:end],
                    metadatas=metadatas_td[i:end],
                    documents=documents[i:end],
                )

        ingest_time = time.perf_counter() - t0
        sampler.stop()
        ingest_stats = sampler.stats()
        gc.collect()

        # ── Phase 2: INDEX BUILD ───────────────────────────────────────────────
        sampler.start()
        t1 = time.perf_counter()

        if engine == "chromadb":
            pass  # HNSW built incrementally during ingest; no separate phase

        elif engine == "lancedb":
            nlist = min(256, max(4, N // 40))
            table.create_index(
                num_partitions=nlist, num_sub_vectors=max(1, DIM // 8))

        elif engine == "qdrant":
            _qclient.update_collection(
                "bench",
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0))
            while True:
                info = _qclient.get_collection("bench")
                if info.status.value == "green":
                    break
                time.sleep(0.5)

        elif engine == "tqdb":
            db.create_index(
                max_degree=32, ef_construction=200, search_list_size=128)

        elif engine == "turbodb":
            pass  # no separate index build step

        index_time = time.perf_counter() - t1
        sampler.stop()
        index_stats = sampler.stats()
        gc.collect()

        # Disk size after full index
        disk_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fnames in os.walk(tmp)
            for fn in fnames
            if not os.path.islink(os.path.join(dp, fn))
        ) / 1024 / 1024

        # ── Phase 3: QUERY ─────────────────────────────────────────────────────
        sampler.start()
        t2 = time.perf_counter()

        for qi, qvec in enumerate(qvecs):
            t_q = time.perf_counter()

            if engine == "chromadb":
                res = col.query(
                    query_embeddings=[qvec.tolist()],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )  # ids are always returned by default
                hits = [
                    {
                        "rank": j + 1,
                        "id":     res["ids"][0][j],
                        "source": res["metadatas"][0][j].get("source", ""),
                        "score":  float(1 - res["distances"][0][j]),
                        "snippet": " ".join((res["documents"][0][j] or "").split()[:20]),
                    }
                    for j in range(len(res["ids"][0]))
                ]

            elif engine == "lancedb":
                rows = table.search(qvec).limit(top_k).nprobes(20).to_list()
                hits = [
                    {
                        "rank": j + 1,
                        "id":    r["id"],
                        "source": r["source"],
                        "score": float(r["_distance"]),
                        "snippet": " ".join((r["doc"] or "").split()[:20]),
                    }
                    for j, r in enumerate(rows)
                ]

            elif engine == "qdrant":
                res = _qclient.query_points(
                    "bench", query=qvec.tolist(), limit=top_k,
                    search_params=SearchParams(hnsw_ef=128),
                    with_payload=True)
                hits = [
                    {
                        "rank": j + 1,
                        "id":    ids[p.id],
                        "source": p.payload.get("source", ""),
                        "score": float(p.score),
                        "snippet": " ".join((p.payload.get("doc", "") or "").split()[:20]),
                    }
                    for j, p in enumerate(res.points)
                ]

            elif engine == "tqdb":
                res = db.search(qvec, top_k=top_k, ann_search_list_size=128)
                hits = [
                    {
                        "rank": j + 1,
                        "id":    r.get("id", ""),
                        "source": r.get("metadata", {}).get("source", ""),
                        "score": float(r.get("score", 0)),
                        "snippet": " ".join((r.get("document") or "").split()[:20]),
                    }
                    for j, r in enumerate(res)
                ]

            elif engine == "turbodb":
                res = _col.query(vector=qvec, k=top_k)
                hits = [
                    {
                        "rank": j + 1,
                        "id":    r.id,
                        "source": (r.metadata or {}).get("source", ""),
                        "score": float(r.score),
                        "snippet": " ".join((r.document or "").split()[:20]),
                    }
                    for j, r in enumerate(res)
                ]

            latencies.append((time.perf_counter() - t_q) * 1000)
            query_results.append(hits)

        sampler.stop()
        query_stats = sampler.stats()

        lats = sorted(latencies)

        print(json.dumps({
            # timing
            "ingest_time":  ingest_time,
            "index_time":   index_time,
            "ready_time":   ingest_time + index_time,
            "ingest_speed": N / ingest_time if ingest_time > 0 else 0,
            # disk
            "disk_mb":      disk_mb,
            # resource: ingest phase
            "ingest_rss_peak":  ingest_stats["rss_peak"],
            "ingest_rss_avg":   ingest_stats["rss_avg"],
            "ingest_cpu_peak":  ingest_stats["cpu_peak"],
            "ingest_cpu_avg":   ingest_stats["cpu_avg"],
            # resource: index phase
            "index_rss_peak":   index_stats["rss_peak"],
            "index_rss_avg":    index_stats["rss_avg"],
            "index_cpu_peak":   index_stats["cpu_peak"],
            "index_cpu_avg":    index_stats["cpu_avg"],
            # resource: query phase
            "query_rss_peak":   query_stats["rss_peak"],
            "query_rss_avg":    query_stats["rss_avg"],
            "query_cpu_peak":   query_stats["cpu_peak"],
            "query_cpu_avg":    query_stats["cpu_avg"],
            # latency percentiles
            "p50_ms":   float(np.percentile(lats, 50)),
            "p97_ms":   float(np.percentile(lats, 97)),
            "p99_ms":   float(np.percentile(lats, 99)),
            "latencies": latencies,
            # per-query results (include id for recall/mrr computation)
            "query_results": query_results,
        }))

    finally:
        if not sampler._stop.is_set():
            sampler.stop()   # guard: stop sampler if still running after exception
        gc.collect()
        if engine == "chromadb":
            try:
                del col, _client
                gc.collect()
                time.sleep(0.3)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


# ── Recall@K / MRR ────────────────────────────────────────────────────────────

def compute_recall_mrr(
    query_results: list[list[dict]],
    gt_top_k: np.ndarray,          # shape (Q, K) — indices into vecs
    id_to_idx: dict[str, int],
    k: int,
) -> tuple[float, float]:
    """Return (Recall@k, MRR) averaged over all queries."""
    recalls, mrrs = [], []
    for qi, hits in enumerate(query_results):
        gt_set = set(gt_top_k[qi].tolist())
        retrieved_indices = [id_to_idx.get(h["id"], -1) for h in hits]

        hit_count = sum(1 for idx in retrieved_indices if idx in gt_set)
        recalls.append(hit_count / min(k, len(gt_set)))

        mrr = 0.0
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in gt_set:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

    return float(np.mean(recalls)), float(np.mean(mrrs))


# ── Engines ───────────────────────────────────────────────────────────────────

ENGINES = [
    ("ChromaDB (HNSW)",  "chromadb", None),
    ("LanceDB (IVF_PQ)", "lancedb",  None),
    ("Qdrant (HNSW)",    "qdrant",   None),
    ("TQDB b=4",         "tqdb",     4),
    ("TQDB b=8",         "tqdb",     8),
    ("TurboDB 0.2.1",    "turbodb",  None),
]


def spawn_worker(engine: str, bits, data_dir: str, top_k: int):
    cmd = [PYTHON, WORKER,
           "--worker_engine",   engine,
           "--worker_data_dir", data_dir,
           "--top_k", str(top_k),
           "--bits",  str(bits or 8)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        for line in proc.stdout.strip().splitlines():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        print(f"    [ERROR] {engine}:\n{proc.stderr[-800:]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {engine}", file=sys.stderr)
        return None


def fmt(v, unit="", d=1):
    return f"{v:.{d}f}{unit}" if v is not None else "n/a"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Data source — pick one
    parser.add_argument("--folder",          default=None,
                        help="Local folder of PDF/MD/TXT files (recursive)")
    parser.add_argument("--hf_dataset",      default=None,
                        help="HuggingFace dataset id, e.g. 'mteb/trec-covid'")
    parser.add_argument("--hf_config",       default=None,
                        help="HF dataset config name (e.g. 'corpus')")
    parser.add_argument("--hf_split",        default="corpus",
                        help="HF dataset split (default: corpus)")
    parser.add_argument("--hf_text_field",   default="text",
                        help="Column name holding document text (default: text)")
    parser.add_argument("--hf_id_field",     default="_id",
                        help="Column name for doc ID (default: _id)")
    parser.add_argument("--max_docs",        type=int, default=None,
                        help="Cap number of source docs ingested (HF only)")
    parser.add_argument("--embed_model",     default=DEFAULT_EMBED_MODEL,
                        help=f"fastembed model name (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--queries",         default="auto",
                        choices=["auto", "litho", "bio"],
                        help="Query set to use: auto=litho for folder, bio for hf (default: auto)")
    # Retrieval
    parser.add_argument("--top_k",           type=int, default=10)
    parser.add_argument("--bits",            type=int, default=8)
    parser.add_argument("--show_queries",    nargs="+", type=int,
                        default=list(range(1, 11)),
                        help="Which query numbers to show results for (1-indexed)")
    parser.add_argument("--out",             default="bench_results_v2.json",
                        help="Path to save full JSON results")
    # Worker-mode args (internal use — do not pass manually)
    parser.add_argument("--worker_engine",   default=None)
    parser.add_argument("--worker_data_dir", default=None)
    args = parser.parse_args()

    if args.worker_engine:
        run_worker_mode(args)
        return

    if not args.folder and not args.hf_dataset:
        print("Usage: py -3.13 benchmarks/real_world_bench_v2.py "
              "--folder <path>  OR  --hf_dataset <id> [--max_docs N]")
        sys.exit(1)

    embed_model = args.embed_model
    bar = "=" * 96

    # Resolve query set
    if args.queries == "auto":
        QUERIES = QUERIES_BIO if args.hf_dataset else QUERIES_LITHO
    elif args.queries == "bio":
        QUERIES = QUERIES_BIO
    else:
        QUERIES = QUERIES_LITHO

    src_label = args.hf_dataset or args.folder
    print(f"\n{bar}")
    print(f"  Real-World RAG Benchmark v2  |  model={embed_model}  top_k={args.top_k}")
    print(f"  Source: {src_label}")
    print(f"{bar}\n")

    # ── Step 1: Load + chunk ──────────────────────────────────────────────────
    all_chunks = []   # list of (id, text, source)

    if args.hf_dataset:
        print(f"  Loading HuggingFace dataset '{args.hf_dataset}' "
              f"(split={args.hf_split}, max_docs={args.max_docs}) ...")
        from datasets import load_dataset
        load_kwargs: dict = {"split": args.hf_split}
        if args.hf_config:
            load_kwargs["name"] = args.hf_config
        ds = load_dataset(args.hf_dataset, **load_kwargs)
        if args.max_docs:
            ds = ds.select(range(min(args.max_docs, len(ds))))
        print(f"  Docs loaded: {len(ds)}")
        for row in ds:
            doc_id  = str(row.get(args.hf_id_field, ""))
            text    = str(row.get(args.hf_text_field, "") or "").strip()
            title   = str(row.get("title", "") or "").strip()
            if title:
                text = f"{title}\n{text}"
            if not text:
                continue
            for i, chunk in enumerate(chunk_text(text)):
                all_chunks.append((f"{doc_id}::{i}", chunk, doc_id))

    else:
        folder = args.folder
        EXTS = (".pdf", ".md", ".txt")
        file_paths = sorted(
            os.path.join(dp, fn)
            for dp, _, fnames in os.walk(folder)
            for fn in fnames
            if fn.lower().endswith(EXTS) and "__pycache__" not in dp
        )
        print(f"  Files: {len(file_paths)}")
        for fpath in file_paths:
            rel = os.path.relpath(fpath, folder).replace("\\", "/")
            text = extract_file(fpath)
            if not text.strip():
                continue
            for i, chunk in enumerate(chunk_text(text)):
                all_chunks.append((f"{rel}::{i}", chunk, rel))

    print(f"  Chunks: {len(all_chunks)}")

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    print(f"\n  Embedding {len(all_chunks)} chunks with {embed_model} ...")
    t0 = time.perf_counter()
    embedder = TextEmbedding(embed_model)
    texts = [c[1] for c in all_chunks]
    vecs  = np.array(list(embedder.embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

    qvecs = np.array(list(embedder.embed(QUERIES)), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9
    embed_time = time.perf_counter() - t0
    N, DIM = vecs.shape
    print(f"  Embedded in {embed_time:.1f}s  dim={DIM}  model={embed_model}")

    # ── Step 3: Brute-force ground truth ──────────────────────────────────────
    print(f"  Computing brute-force ground truth (Recall@{args.top_k}, MRR) ...")
    scores_gt = qvecs @ vecs.T                                       # (Q, N)
    gt_top_k  = np.argsort(-scores_gt, axis=1)[:, : args.top_k]    # (Q, top_k)
    id_to_idx = {c[0]: idx for idx, c in enumerate(all_chunks)}

    # ── Step 4: Save shared data for workers ──────────────────────────────────
    data_dir = tempfile.mkdtemp(prefix="rag_data_")
    np.save(os.path.join(data_dir, "embeddings.npy"), vecs)
    np.save(os.path.join(data_dir, "query_vecs.npy"), qvecs)
    meta = {
        "ids":       [c[0] for c in all_chunks],
        "sources":   [c[2] for c in all_chunks],
        "documents": [c[1] for c in all_chunks],
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    raw_vec_mb  = N * DIM * 4 / 1024 / 1024   # float32 baseline
    BYTES_F32   = DIM * 4

    # ── Step 5: Run each engine ───────────────────────────────────────────────
    results: dict[str, dict | None] = {}
    for label, engine, bits in ENGINES:
        print(f"\n  Running {label} ...", flush=True)
        time.sleep(3)   # cooldown between workers
        r = spawn_worker(engine, bits, data_dir, args.top_k)
        if r:
            # Attach Recall@K and MRR
            recall, mrr = compute_recall_mrr(
                r.get("query_results", []), gt_top_k, id_to_idx, args.top_k)
            r["recall"] = recall
            r["mrr"]    = mrr
            print(
                f"    ready={r['ready_time']:.1f}s  "
                f"disk={r['disk_mb']:.1f}MB  "
                f"p50={r['p50_ms']:.2f}ms  "
                f"recall@{args.top_k}={recall:.3f}  mrr={mrr:.3f}")
        results[label] = r

    shutil.rmtree(data_dir, ignore_errors=True)

    # ── Step 6: Performance table ─────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  PERFORMANCE  ({N} chunks, dim={DIM}, top_k={args.top_k})")
    print(bar)

    # Header row 1
    print(f"  {'Engine':<22}  "
          f"{'Ingest':>7}  {'Index':>7}  {'Ready':>7}  {'v/s':>6}  "
          f"{'Disk':>8}  {'RawVec':>7}  "
          f"{'i-RAM pk':>9}  {'i-RAM avg':>9}  {'i-CPU pk':>9}  {'i-CPU avg':>9}  "
          f"{'idx-RAM pk':>10}  {'idx-CPU pk':>10}  "
          f"{'q-RAM pk':>9}  {'q-RAM avg':>9}  {'q-CPU pk':>9}  {'q-CPU avg':>9}  "
          f"{'p50':>7}  {'p97':>7}  {'p99':>7}  "
          f"{'R@K':>6}  {'MRR':>6}")
    # Header row 2 (units)
    print(f"  {'':22}  "
          f"{'(s)':>7}  {'(s)':>7}  {'(s)':>7}  {'':>6}  "
          f"{'(MB)':>8}  {'(MB)':>7}  "
          f"{'(MB)':>9}  {'(MB)':>9}  {'(%)':>9}  {'(%)':>9}  "
          f"{'(MB)':>10}  {'(%)':>10}  "
          f"{'(MB)':>9}  {'(MB)':>9}  {'(%)':>9}  {'(%)':>9}  "
          f"{'(ms)':>7}  {'(ms)':>7}  {'(ms)':>7}  "
          f"{'':>6}  {'':>6}")
    sep = (f"  {'-'*22}  "
           f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  "
           f"{'-'*8}  {'-'*7}  "
           f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  "
           f"{'-'*10}  {'-'*10}  "
           f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  "
           f"{'-'*7}  {'-'*7}  {'-'*7}  "
           f"{'-'*6}  {'-'*6}")
    print(sep)

    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            print(f"  {label:<22}  FAILED")
            continue
        print(
            f"  {label:<22}  "
            f"{fmt(r['ingest_time'], 's'):>7}  "
            f"{fmt(r['index_time'],  's'):>7}  "
            f"{fmt(r['ready_time'],  's'):>7}  "
            f"{int(r['ingest_speed']):>6,}  "
            f"{fmt(r['disk_mb'],         'MB'):>8}  "
            f"{fmt(raw_vec_mb,           'MB', 0):>7}  "
            f"{fmt(r['ingest_rss_peak'], 'MB', 0):>9}  "
            f"{fmt(r['ingest_rss_avg'],  'MB', 0):>9}  "
            f"{fmt(r['ingest_cpu_peak'], '%',  0):>9}  "
            f"{fmt(r['ingest_cpu_avg'],  '%',  0):>9}  "
            f"{fmt(r['index_rss_peak'],  'MB', 0):>10}  "
            f"{fmt(r['index_cpu_peak'],  '%',  0):>10}  "
            f"{fmt(r['query_rss_peak'],  'MB', 0):>9}  "
            f"{fmt(r['query_rss_avg'],   'MB', 0):>9}  "
            f"{fmt(r['query_cpu_peak'],  '%',  0):>9}  "
            f"{fmt(r['query_cpu_avg'],   '%',  0):>9}  "
            f"{fmt(r['p50_ms'], 'ms', 2):>7}  "
            f"{fmt(r['p97_ms'], 'ms', 2):>7}  "
            f"{fmt(r['p99_ms'], 'ms', 2):>7}  "
            f"{r['recall']:>6.3f}  "
            f"{r['mrr']:>6.3f}")

    # ── Disk compression table ────────────────────────────────────────────────
    print(f"\n  Disk compression  (float32 raw = {BYTES_F32} bytes/vec = "
          f"{raw_vec_mb:.1f} MB for {N} chunks)")
    print(f"  {'Engine':<22}  {'Disk (MB)':>10}  {'bytes/vec':>10}  "
          f"{'vs float32':>13}  {'vs ChromaDB':>13}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*13}  {'-'*13}")

    chroma_disk_mb = (results.get("ChromaDB (HNSW)") or {}).get("disk_mb")
    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            continue
        bpv   = r["disk_mb"] * 1024 * 1024 / N if N > 0 else 0
        vsf   = BYTES_F32 / bpv if bpv > 0 else 0
        vscd  = (chroma_disk_mb / r["disk_mb"]
                 if chroma_disk_mb and r["disk_mb"] > 0 else None)
        vscd_str = f"x{vscd:.2f} smaller" if vscd else "---"
        print(f"  {label:<22}  {r['disk_mb']:>10.1f}  {bpv:>10.0f}  "
              f"  x{vsf:.2f} smaller  {vscd_str:>13}")

    # ── Quality table ─────────────────────────────────────────────────────────
    print(f"\n  {'Engine':<22}  {'Recall@'+str(args.top_k):>9}  {'MRR':>8}  "
          f"{'p50 (ms)':>10}  {'p97 (ms)':>10}  {'p99 (ms)':>10}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            continue
        print(f"  {label:<22}  {r['recall']:>9.3f}  {r['mrr']:>8.3f}  "
              f"{fmt(r['p50_ms'], 'ms', 2):>10}  "
              f"{fmt(r['p97_ms'], 'ms', 2):>10}  "
              f"{fmt(r['p99_ms'], 'ms', 2):>10}")

    # ── Per-query retrieval results ───────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  RETRIEVAL QUALITY  (top_{args.top_k} per engine)")
    print(bar)

    for qi, query in enumerate(QUERIES):
        qnum = qi + 1
        if qnum not in args.show_queries:
            continue
        gt_ids = {all_chunks[idx][0] for idx in gt_top_k[qi]}
        print(f"\n  Q{qnum}: {query}")
        for label, _, _ in ENGINES:
            r = results.get(label)
            if not r or not r.get("query_results"):
                continue
            hits = r["query_results"][qi] if qi < len(r["query_results"]) else []
            print(f"\n    [{label}]")
            for h in hits:
                marker = "✓" if h["id"] in gt_ids else " "
                print(f"      {marker} [{h['rank']}] {h['source']:<52} score={h['score']:.4f}")
                print(f"           \"{h['snippet']}...\"")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  SUMMARY  ({N} chunks | embed {embed_time:.1f}s | raw vec {raw_vec_mb:.1f}MB)")
    print(bar)
    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            print(f"  {label:<22}  FAILED")
            continue
        print(f"  {label:<22}  "
              f"ready={r['ready_time']:.1f}s  "
              f"disk={r['disk_mb']:.1f}MB  "
              f"RAM-qry-pk={r['query_rss_peak']:.0f}MB  "
              f"p50={r['p50_ms']:.2f}ms  p97={r['p97_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms  "
              f"recall@{args.top_k}={r['recall']:.3f}  mrr={r['mrr']:.3f}")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = args.out
    payload  = {
        "meta": {
            "source":      src_label,
            "n_chunks":    N,
            "dim":         DIM,
            "top_k":       args.top_k,
            "embed_model": embed_model,
            "embed_time_s": embed_time,
            "raw_vec_mb":  raw_vec_mb,
        },
        "results": {
            label: results[label]
            for label, _, _ in ENGINES
        },
    }
    # Strip latency arrays from JSON to keep it compact
    for v in payload["results"].values():
        if v:
            v.pop("latencies", None)
            v.pop("query_results", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
