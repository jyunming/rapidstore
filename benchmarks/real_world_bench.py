"""
Real-world RAG benchmark — all vector stores, same documents, same queries.

Embeds once, then each engine runs in an isolated subprocess so RSS/CPU
measurements are per-engine only (no cross-contamination).

Usage:
    py -3.13 benchmarks/real_world_bench.py --folder "C:/dev/studio_brain_open/Qualification/papers"
    py -3.13 benchmarks/real_world_bench.py --folder "..." --top_k 5 --bits 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
import gc

import numpy as np
import pymupdf
import psutil
from fastembed import TextEmbedding

sys.stdout.reconfigure(encoding="utf-8")

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE   = 400
CHUNK_OVERLAP = 80

QUERIES = [
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

PYTHON = sys.executable
WORKER = __file__   # this file doubles as the worker when --worker_engine is set

# ── Text extraction & chunking ─────────────────────────────────────────────────

def extract_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        doc = pymupdf.open(path)
        text = "\n".join(p.get_text("text") for p in doc if p.get_text("text").strip())
        doc.close()
        return text
    elif ext in (".md", ".txt"):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Worker — runs ONE engine in this subprocess ────────────────────────────────

def run_worker_mode(args):
    """Called when --worker_engine is set. Prints one JSON line to stdout."""
    import gc, time
    engine  = args.worker_engine
    top_k   = args.top_k
    bits    = args.bits
    data_dir = args.worker_data_dir

    # Raise process priority
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

    def rss_mb():
        gc.collect()
        return psutil.Process().memory_info().rss / 1024 / 1024

    tmp = tempfile.mkdtemp(prefix=f"rag_{engine}_")
    ingest_peak_rss = rss_mb()
    query_results   = []   # list of lists of {rank, source, score, snippet}

    try:
        # ── INGEST ────────────────────────────────────────────────────────────
        t_ingest_start = time.perf_counter()
        proc = psutil.Process()
        cpu_before_ingest = proc.cpu_times()

        if engine == "chromadb":
            import chromadb
            _client = chromadb.PersistentClient(path=tmp)
            col = _client.create_collection("vecs", metadata={"hnsw:space": "ip"})
            for i in range(0, N, 500):
                end = min(i + 500, N)
                col.add(ids=ids[i:end],
                        embeddings=vecs[i:end].tolist(),
                        documents=documents[i:end],
                        metadatas=[{"source": sources[j]} for j in range(i, end)])
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "lancedb":
            import lancedb
            db = lancedb.connect(tmp)
            table = db.create_table("v", data=[
                {"vector": vecs[i].tolist(), "id": ids[i],
                 "source": sources[i], "doc": documents[i]}
                for i in range(N)])
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            nlist = min(256, max(4, N // 40))
            table.create_index(num_partitions=nlist, num_sub_vectors=max(1, DIM // 8))
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                               SearchParams, OptimizersConfigDiff,
                                               HnswConfigDiff)
            _qclient = QdrantClient(path=os.path.join(tmp, "qdrant"))
            _qclient.create_collection(
                "bench",
                vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=N + 1))
            for i in range(0, N, 500):
                end = min(i + 500, N)
                _qclient.upsert("bench", points=[
                    PointStruct(id=j, vector=vecs[j].tolist(),
                                payload={"source": sources[j], "doc": documents[j]})
                    for j in range(i, end)])
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            _qclient.update_collection(
                "bench",
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0))
            while True:
                info = _qclient.get_collection("bench")
                if info.status.value == "green":
                    break
                time.sleep(0.5)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "tqdb":
            import turboquantdb
            db_path = os.path.join(tmp, "rag.tqdb")
            db = turboquantdb.TurboQuantDB.open(
                db_path, dimension=DIM, bits=bits, metric="ip", rerank=True)
            metadatas = [{"source": sources[i]} for i in range(N)]
            db.insert_batch(ids, vecs, metadatas=metadatas, documents=documents)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            db.create_index(max_degree=32, ef_construction=200, search_list_size=128)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        cpu_after_ingest = proc.cpu_times()
        ingest_wall = time.perf_counter() - t_ingest_start
        ingest_cpu_used = ((cpu_after_ingest.user - cpu_before_ingest.user) +
                           (cpu_after_ingest.system - cpu_before_ingest.system))
        ingest_cpu_util = ingest_cpu_used / ingest_wall * 100 if ingest_wall > 0 else 0

        # Disk size
        disk_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(tmp)
            for f in fnames
            if not os.path.islink(os.path.join(dp, f))
        ) / 1024 / 1024

        gc.collect()
        retrieve_rss = rss_mb()

        # ── QUERY ─────────────────────────────────────────────────────────────
        latencies = []
        cpu_before_query = proc.cpu_times()
        t_query_start = time.perf_counter()

        for qi, qvec in enumerate(qvecs):
            t0 = time.perf_counter()

            if engine == "chromadb":
                res = col.query(query_embeddings=[qvec.tolist()], n_results=top_k,
                                include=["documents", "metadatas", "distances"])
                hits = []
                for j in range(len(res["ids"][0])):
                    hits.append({
                        "rank": j + 1,
                        "source": res["metadatas"][0][j].get("source", ""),
                        "score": float(1 - res["distances"][0][j]),
                        "snippet": " ".join((res["documents"][0][j] or "").split()[:20]),
                    })

            elif engine == "lancedb":
                rows = table.search(qvec).limit(top_k).nprobes(20).to_list()
                hits = [{"rank": j + 1, "source": r["source"],
                         "score": float(r["_distance"]),
                         "snippet": " ".join((r["doc"] or "").split()[:20])}
                        for j, r in enumerate(rows)]

            elif engine == "qdrant":
                res = _qclient.query_points(
                    "bench", query=qvec.tolist(), limit=top_k,
                    search_params=SearchParams(hnsw_ef=128),
                    with_payload=True)
                hits = [{"rank": j + 1,
                         "source": p.payload.get("source", ""),
                         "score": float(p.score),
                         "snippet": " ".join((p.payload.get("doc", "") or "").split()[:20])}
                        for j, p in enumerate(res.points)]

            elif engine == "tqdb":
                res = db.search(qvec, top_k=top_k, ann_search_list_size=128)
                hits = [{"rank": j + 1,
                         "source": r.get("metadata", {}).get("source", ""),
                         "score": float(r.get("score", 0)),
                         "snippet": " ".join((r.get("document") or "").split()[:20])}
                        for j, r in enumerate(res)]

            latencies.append((time.perf_counter() - t0) * 1000)
            query_results.append(hits)

        query_wall = time.perf_counter() - t_query_start
        cpu_after_query = proc.cpu_times()
        query_cpu_used = ((cpu_after_query.user - cpu_before_query.user) +
                          (cpu_after_query.system - cpu_before_query.system))
        query_cpu_util = query_cpu_used / query_wall * 100 if query_wall > 0 else 0

        print(json.dumps({
            "ingest_time":      ingest_wall,
            "ingest_speed":     N / ingest_wall,
            "disk_mb":          disk_mb,
            "ingest_rss_mb":    ingest_peak_rss,
            "retrieve_rss_mb":  retrieve_rss,
            "ingest_cpu_util":  ingest_cpu_util,
            "query_cpu_util":   query_cpu_util,
            "p50_ms":           float(np.median(latencies)),
            "p95_ms":           float(np.percentile(latencies, 95)),
            "latencies":        latencies,
            "query_results":    query_results,
        }))

    finally:
        gc.collect()
        if engine == "chromadb":
            try:
                del col, _client
                gc.collect()
                time.sleep(0.3)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


# ── Harness ────────────────────────────────────────────────────────────────────

ENGINES = [
    ("ChromaDB (HNSW)",   "chromadb", None),
    ("LanceDB (IVF_PQ)",  "lancedb",  None),
    ("Qdrant (HNSW)",     "qdrant",   None),
    ("TQDB b=4",          "tqdb",     4),
    ("TQDB b=8",          "tqdb",     8),
]


def spawn_worker(engine, bits, data_dir, top_k):
    cmd = [PYTHON, WORKER,
           "--worker_engine", engine,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",          default=None)
    parser.add_argument("--top_k",           type=int, default=5)
    parser.add_argument("--bits",            type=int, default=8)
    parser.add_argument("--show_queries",    nargs="+", type=int, default=list(range(1, 11)),
                        help="Which query numbers to show results for (1-indexed)")
    # Worker-mode args (internal use)
    parser.add_argument("--worker_engine",   default=None)
    parser.add_argument("--worker_data_dir", default=None)
    args = parser.parse_args()

    if args.worker_engine:
        run_worker_mode(args)
        return

    if not args.folder:
        print("Usage: py -3.13 benchmarks/real_world_bench.py --folder <path>")
        sys.exit(1)

    folder = args.folder
    bar = "=" * 80

    print(f"\n{bar}")
    print(f"  Real-World RAG Benchmark  |  model={EMBED_MODEL}")
    print(f"  Folder: {folder}")
    print(f"{bar}\n")

    # ── Step 1: Extract + chunk ───────────────────────────────────────────────
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".pdf", ".md", ".txt")))
    print(f"  Files: {len(files)}")

    all_chunks = []
    for fname in files:
        text = extract_file(os.path.join(folder, fname))
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append((f"{fname}::{i}", chunk, fname))
    print(f"  Chunks: {len(all_chunks)}")

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    print(f"\n  Embedding {len(all_chunks)} chunks with {EMBED_MODEL}...")
    t0 = time.perf_counter()
    embedder = TextEmbedding(EMBED_MODEL)
    texts = [c[1] for c in all_chunks]
    vecs = np.array(list(embedder.embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

    qvecs = np.array(list(embedder.embed(QUERIES)), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9
    embed_time = time.perf_counter() - t0
    print(f"  Embedded in {embed_time:.1f}s  dim={vecs.shape[1]}")

    # ── Step 3: Save shared data for workers ──────────────────────────────────
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

    # ── Step 4: Run each engine ───────────────────────────────────────────────
    results = {}
    for label, engine, bits in ENGINES:
        print(f"\n  Running {label} ...", flush=True)
        time.sleep(3)  # cooldown between workers
        r = spawn_worker(engine, bits, data_dir, args.top_k)
        if r:
            print(f"    ingest={r['ingest_time']:.1f}s  "
                  f"disk={r['disk_mb']:.1f}MB  "
                  f"RAM-qry={r['retrieve_rss_mb']:.0f}MB  "
                  f"p50={r['p50_ms']:.2f}ms")
        results[label] = r

    shutil.rmtree(data_dir, ignore_errors=True)

    N = len(all_chunks)
    BYTES_F32 = vecs.shape[1] * 4

    # ── Step 5: Performance table ─────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  PERFORMANCE  ({N} chunks, dim={vecs.shape[1]}, top_k={args.top_k})")
    print(bar)
    print(f"  {'Engine':<22}  {'Ingest':>8}  {'vec/s':>7}  {'Disk':>8}  "
          f"{'RAM-ing':>8}  {'RAM-qry':>8}  {'CPU-ing':>8}  {'CPU-qry':>8}  "
          f"{'p50':>7}  {'p95':>7}")
    print(f"  {'':22}  {'(s)':>8}  {'':>7}  {'(MB)':>8}  "
          f"{'(MB)':>8}  {'(MB)':>8}  {'(%)':>8}  {'(%)':>8}  "
          f"{'(ms)':>7}  {'(ms)':>7}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*7}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")
    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            print(f"  {label:<22}  FAILED")
            continue
        print(f"  {label:<22}  "
              f"{fmt(r['ingest_time'], 's'):>8}  "
              f"{int(r['ingest_speed']):>7,}  "
              f"{fmt(r['disk_mb'], 'MB'):>8}  "
              f"{fmt(r['ingest_rss_mb'], 'MB', 0):>8}  "
              f"{fmt(r['retrieve_rss_mb'], 'MB', 0):>8}  "
              f"{fmt(r['ingest_cpu_util'], '%', 0):>8}  "
              f"{fmt(r['query_cpu_util'], '%', 0):>8}  "
              f"{fmt(r['p50_ms'], 'ms', 2):>7}  "
              f"{fmt(r['p95_ms'], 'ms', 2):>7}")

    # ── Disk compression table ────────────────────────────────────────────────
    print(f"\n  Disk compression  (float32 raw = {BYTES_F32} bytes/vec = "
          f"{N * BYTES_F32 / 1024 / 1024:.1f}MB for {N} chunks)")
    print(f"  {'Engine':<22}  {'Disk (MB)':>10}  {'bytes/vec':>10}  {'vs float32':>12}  {'vs ChromaDB':>12}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")
    chroma_disk = results.get("ChromaDB (HNSW)", {}) or {}
    chroma_disk_mb = chroma_disk.get("disk_mb")
    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            continue
        bpv  = r["disk_mb"] * 1024 * 1024 / N
        vsf  = BYTES_F32 / bpv if bpv > 0 else 0
        vscd = chroma_disk_mb / r["disk_mb"] if chroma_disk_mb and r["disk_mb"] > 0 else None
        vscd_str = f"x{vscd:.2f} smaller" if vscd else "---"
        print(f"  {label:<22}  {r['disk_mb']:>10.1f}  {bpv:>10.0f}  "
              f"  x{vsf:.2f} smaller  {vscd_str:>12}")

    # ── Step 6: Query quality ─────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  RETRIEVAL QUALITY  (top_{args.top_k} per engine)")
    print(bar)

    for qi, query in enumerate(QUERIES):
        qnum = qi + 1
        if qnum not in args.show_queries:
            continue
        print(f"\n  Q{qnum}: {query}")
        for label, _, _ in ENGINES:
            r = results.get(label)
            if not r or not r.get("query_results"):
                continue
            hits = r["query_results"][qi] if qi < len(r["query_results"]) else []
            print(f"\n    [{label}]")
            for h in hits:
                src     = h["source"]
                score   = h["score"]
                snippet = h["snippet"]
                print(f"      [{h['rank']}] {src:<52} score={score:.4f}")
                print(f"           \"{snippet}...\"")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  SUMMARY")
    print(bar)
    print(f"  Chunks: {N}  |  Embed time: {embed_time:.1f}s ({N/embed_time:.0f} chunks/s)")
    for label, _, _ in ENGINES:
        r = results.get(label)
        if not r:
            continue
        print(f"  {label:<22}  ready={r['ingest_time']:.1f}s  "
              f"disk={r['disk_mb']:.1f}MB  RAM={r['retrieve_rss_mb']:.0f}MB  "
              f"p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms")
    print()


if __name__ == "__main__":
    main()
