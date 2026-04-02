"""
Large-scale benchmark — DBpedia entities with pre-computed OpenAI embeddings.

Source: KShivendu/dbpedia-entities-openai-1M (1536-dim, text-embedding-ada-002)
No CPU embedding needed — vectors are loaded directly from HuggingFace.

Usage:
    py -3.13 benchmarks/large_scale_bench.py
    py -3.13 benchmarks/large_scale_bench.py --n_vecs 50000 --n_queries 200 --top_k 10
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil

sys.stdout.reconfigure(encoding="utf-8")

PYTHON   = sys.executable
WORKER   = __file__
DATASET  = "KShivendu/dbpedia-entities-openai-1M"
VEC_FIELD = "openai"   # 1536-dim float32 list

# (label, engine, bits, fast_mode, rerank, max_degree, ef_construction, n_refinements, ann_search_list_size)
ENGINES = [
    # High Quality: b=8, rerank, n_refinements=8, ann_sl=200
    ("TQDB b=8 HQ",        "tqdb", 8, False, True,  32, 200, 8, 200),
    # Balanced: b=4, rerank, n_refinements=5, ann_sl=128
    ("TQDB b=4 Balanced",  "tqdb", 4, False, True,  32, 200, 5, 128),
    # Fast Build: b=4, fast_mode, no rerank, n_refinements=5
    ("TQDB b=4 FastBuild", "tqdb", 4, True,  False, 32, 200, 5, 128),
]


# ── Worker ─────────────────────────────────────────────────────────────────────

def run_worker_mode(args):
    engine            = args.worker_engine
    top_k             = args.top_k
    bits              = args.bits
    fast_mode         = bool(args.fast_mode)
    rerank            = bool(args.rerank)
    max_degree        = args.max_degree
    ef_construction   = args.ef_construction
    n_refinements     = args.n_refinements
    ann_sl            = args.ann_search_list_size
    data_dir          = args.worker_data_dir

    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass

    vecs  = np.load(os.path.join(data_dir, "embeddings.npy"))
    qvecs = np.load(os.path.join(data_dir, "query_vecs.npy"))
    truth = np.load(os.path.join(data_dir, "ground_truth.npy"))   # (NQ, top_k) indices
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    ids       = meta["ids"]
    documents = meta["documents"]
    N, DIM    = vecs.shape
    NQ        = len(qvecs)

    def rss_mb():
        gc.collect()
        return psutil.Process().memory_info().rss / 1024 / 1024

    tmp = tempfile.mkdtemp(prefix=f"ls_{engine}_")
    ingest_peak_rss = rss_mb()
    found_indices   = []

    # id → index lookup for recall computation
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    try:
        proc = psutil.Process()

        # ── INGEST ────────────────────────────────────────────────────────────
        t0   = time.perf_counter()
        cpu0 = proc.cpu_times()

        if engine == "chromadb":
            import chromadb
            _client = chromadb.PersistentClient(path=tmp)
            col = _client.create_collection("vecs", metadata={"hnsw:space": "ip"})
            BS = 2000
            for i in range(0, N, BS):
                end = min(i + BS, N)
                col.add(ids=ids[i:end],
                        embeddings=vecs[i:end].tolist(),
                        documents=documents[i:end])
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "lancedb":
            import lancedb
            db = lancedb.connect(tmp)
            BS = 5000
            rows, table = [], None
            for i in range(N):
                rows.append({"vector": vecs[i].tolist(), "id": ids[i], "doc": documents[i]})
                if len(rows) == BS or i == N - 1:
                    if table is None:
                        table = db.create_table("v", data=rows)
                    else:
                        table.add(rows)
                    rows = []
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
            BS = 2000
            for i in range(0, N, BS):
                end = min(i + BS, N)
                _qclient.upsert("bench", points=[
                    PointStruct(id=j, vector=vecs[j].tolist())
                    for j in range(i, end)])
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            _qclient.update_collection(
                "bench", optimizers_config=OptimizersConfigDiff(indexing_threshold=0))
            while True:
                if _qclient.get_collection("bench").status.value == "green":
                    break
                time.sleep(1.0)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "tqdb":
            import turboquantdb
            db_path = os.path.join(tmp, "rag.tqdb")
            db = turboquantdb.Database.open(
                db_path, dimension=DIM, bits=bits, metric="ip",
                rerank=rerank, fast_mode=fast_mode)
            db.insert_batch(ids, vecs, documents=documents)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            db.create_index(max_degree=max_degree, ef_construction=ef_construction,
                            search_list_size=ann_sl, n_refinements=n_refinements)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        ingest_wall = time.perf_counter() - t0
        cpu1 = proc.cpu_times()
        ingest_cpu = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / ingest_wall * 100

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
        cpu0q = proc.cpu_times()
        tq0   = time.perf_counter()

        for qi, qvec in enumerate(qvecs):
            t0 = time.perf_counter()

            if engine == "chromadb":
                res  = col.query(query_embeddings=[qvec.tolist()], n_results=top_k)
                hits = [id_to_idx.get(x, -1) for x in res["ids"][0]]

            elif engine == "lancedb":
                rows = table.search(qvec).limit(top_k).nprobes(20).to_list()
                hits = [id_to_idx.get(str(r["id"]), -1) for r in rows]

            elif engine == "qdrant":
                res  = _qclient.query_points("bench", query=qvec.tolist(), limit=top_k,
                                              search_params=SearchParams(hnsw_ef=128))
                hits = [p.id for p in res.points]

            elif engine == "tqdb":
                res  = db.search(qvec, top_k=top_k, ann_search_list_size=ann_sl)
                hits = [id_to_idx.get(r.get("id", ""), -1) for r in res]

            latencies.append((time.perf_counter() - t0) * 1000)
            found_indices.append(hits)

        query_wall = time.perf_counter() - tq0
        cpu1q = proc.cpu_times()
        query_cpu = ((cpu1q.user - cpu0q.user) + (cpu1q.system - cpu0q.system)) / query_wall * 100

        # ── RECALL vs brute-force ground truth ────────────────────────────────
        recall_scores, mrr_scores = [], []
        for qi in range(NQ):
            gt  = set(int(x) for x in truth[qi])
            got = set(int(x) for x in found_indices[qi] if x >= 0)
            recall_scores.append(len(gt & got) / len(gt) if gt else 0.0)
            mrr = 0.0
            for rank, idx in enumerate(found_indices[qi], 1):
                if int(idx) in gt:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)

        print(json.dumps({
            "ingest_time":     ingest_wall,
            "ingest_speed":    N / ingest_wall,
            "disk_mb":         disk_mb,
            "ingest_rss_mb":   ingest_peak_rss,
            "retrieve_rss_mb": retrieve_rss,
            "ingest_cpu_util": ingest_cpu,
            "query_cpu_util":  query_cpu,
            "p50_ms":          float(np.median(latencies)),
            "p95_ms":          float(np.percentile(latencies, 95)),
            "recall_at_k":     float(np.mean(recall_scores)),
            "mrr":             float(np.mean(mrr_scores)),
        }))

    finally:
        gc.collect()
        if engine == "chromadb":
            try:
                del col, _client; gc.collect(); time.sleep(0.5)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


# ── Harness ────────────────────────────────────────────────────────────────────

def spawn_worker(label, engine, bits, fast_mode, rerank,
                 max_degree, ef_construction, n_refinements, ann_sl,
                 data_dir, top_k):
    cmd = [PYTHON, WORKER,
           "--worker_engine",       engine,
           "--worker_data_dir",     data_dir,
           "--top_k",               str(top_k),
           "--bits",                str(bits or 8),
           "--fast_mode",           str(int(bool(fast_mode))),
           "--rerank",              str(int(bool(rerank))),
           "--max_degree",          str(max_degree or 32),
           "--ef_construction",     str(ef_construction or 200),
           "--n_refinements",       str(n_refinements or 5),
           "--ann_search_list_size",str(ann_sl or 128)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        for line in proc.stdout.strip().splitlines():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        print(f"    [ERROR] {engine}:\n{proc.stderr[-1000:]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {engine}", file=sys.stderr)
        return None


def fmt(v, unit="", d=1):
    return f"{v:.{d}f}{unit}" if v is not None else "n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vecs",    type=int, default=50000,
                        help="Number of vectors to load from the dataset")
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "tmp_ls_checkpoint"),
        help="Persistent dir for downloaded vectors — resume from here if interrupted")
    parser.add_argument("--worker_engine",        default=None)
    parser.add_argument("--worker_data_dir",      default=None)
    parser.add_argument("--bits",                 type=int, default=8)
    parser.add_argument("--fast_mode",            type=int, default=0)
    parser.add_argument("--rerank",               type=int, default=1)
    parser.add_argument("--max_degree",           type=int, default=32)
    parser.add_argument("--ef_construction",      type=int, default=200)
    parser.add_argument("--n_refinements",        type=int, default=5)
    parser.add_argument("--ann_search_list_size", type=int, default=128)
    args = parser.parse_args()

    if args.worker_engine:
        run_worker_mode(args)
        return

    bar = "=" * 80
    print(f"\n{bar}")
    print(f"  Large-Scale Benchmark  |  DBpedia OpenAI embeddings (1536-dim)")
    print(f"  n_vecs={args.n_vecs:,}  n_queries={args.n_queries}  top_k={args.top_k}")
    print(f"{bar}\n")

    ckpt = os.path.abspath(args.checkpoint_dir)
    os.makedirs(ckpt, exist_ok=True)

    # checkpoint key encodes n_vecs so different sizes don't collide
    ckpt_tag  = f"dbpedia_{args.n_vecs}"
    ckpt_meta = os.path.join(ckpt, f"{ckpt_tag}_meta.json")
    ckpt_vecs = os.path.join(ckpt, f"{ckpt_tag}_vecs.npy")

    # ── Step 1: Load pre-embedded vectors (or from checkpoint) ───────────────
    if os.path.exists(ckpt_meta) and os.path.exists(ckpt_vecs):
        print("  [1/3] Loading vectors from checkpoint...", flush=True)
        with open(ckpt_meta, encoding="utf-8") as f:
            saved = json.load(f)
        ids       = saved["ids"]
        documents = saved["documents"]
        vecs      = np.load(ckpt_vecs)
        DIM       = vecs.shape[1]
        print(f"  Resumed: {len(vecs):,} vectors  dim={DIM}")
    else:
        print(f"  [1/3] Downloading {args.n_vecs:,} vectors from {DATASET}...", flush=True)
        from datasets import load_dataset
        t0 = time.perf_counter()
        ds = load_dataset(DATASET, split="train", streaming=True)
        ids, documents, raw_vecs = [], [], []
        for i, row in enumerate(ds):
            if i >= args.n_vecs:
                break
            ids.append(str(row["_id"]))
            documents.append(f"{row['title']} — {row['text'][:200]}")
            raw_vecs.append(row[VEC_FIELD])
            if (i + 1) % 5000 == 0:
                print(f"    {i+1:>6}/{args.n_vecs:,}", flush=True)
        vecs = np.array(raw_vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.where(norms > 0, norms, 1.0)
        DIM = vecs.shape[1]
        elapsed = time.perf_counter() - t0
        print(f"  Loaded {len(vecs):,} vectors  dim={DIM}  ({elapsed:.1f}s)")
        np.save(ckpt_vecs, vecs)
        with open(ckpt_meta, "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "documents": documents}, f)
        print(f"  Saved checkpoint: {ckpt_tag}")

    N = len(vecs)

    # ── Step 2: Build query set + brute-force ground truth ───────────────────
    ckpt_qvecs = os.path.join(ckpt, f"{ckpt_tag}_qvecs.npy")
    ckpt_truth = os.path.join(ckpt, f"{ckpt_tag}_truth_{args.top_k}.npy")

    if os.path.exists(ckpt_qvecs) and os.path.exists(ckpt_truth):
        print(f"\n  [2/3] Loading query/truth from checkpoint...", flush=True)
        qvecs        = np.load(ckpt_qvecs)
        ground_truth = np.load(ckpt_truth)
        print(f"  Queries: {len(qvecs):,}  truth shape: {ground_truth.shape}")
    else:
        print(f"\n  [2/3] Building {args.n_queries} query vectors + brute-force truth...", flush=True)
        rng = np.random.default_rng(42)
        q_indices = rng.choice(N, size=args.n_queries, replace=False)
        # use the actual vectors as queries (self-retrieval — ground truth idx == q_index)
        qvecs = vecs[q_indices].copy()
        print(f"  Computing brute-force ground truth (top_{args.top_k})...", flush=True)
        t0 = time.perf_counter()
        BF_BS = 50
        truth_rows = []
        for i in range(0, args.n_queries, BF_BS):
            scores = qvecs[i:i + BF_BS] @ vecs.T
            truth_rows.append(np.argsort(-scores, axis=1)[:, :args.top_k])
        ground_truth = np.vstack(truth_rows)
        print(f"  Ground truth computed in {time.perf_counter()-t0:.1f}s")
        np.save(ckpt_qvecs, qvecs)
        np.save(ckpt_truth, ground_truth)

    # ── Step 3: Save shared data for workers ─────────────────────────────────
    print(f"\n  [3/3] Preparing shared data for workers...", flush=True)
    data_dir = ckpt
    np.save(os.path.join(data_dir, "embeddings.npy"),   vecs)
    np.save(os.path.join(data_dir, "query_vecs.npy"),   qvecs)
    np.save(os.path.join(data_dir, "ground_truth.npy"), ground_truth)
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "documents": documents}, f)

    BYTES_F32 = DIM * 4

    # ── Run engines (process-isolated) ───────────────────────────────────────
    print(f"\n  Running engines (process-isolated)...\n")
    results = {}
    for label, engine, bits, fast_mode, rerank, max_degree, ef_construction, n_refinements, ann_sl in ENGINES:
        print(f"  Running {label} ...", flush=True)
        time.sleep(3)
        r = spawn_worker(label, engine, bits, fast_mode, rerank,
                         max_degree, ef_construction, n_refinements, ann_sl,
                         data_dir, args.top_k)
        if r:
            print(f"    ingest={r['ingest_time']:.1f}s  "
                  f"disk={r['disk_mb']:.1f}MB  "
                  f"RAM={r['retrieve_rss_mb']:.0f}MB  "
                  f"p50={r['p50_ms']:.2f}ms  "
                  f"recall@{args.top_k}={r['recall_at_k']*100:.1f}%  "
                  f"MRR={r['mrr']:.4f}")
        else:
            print(f"    FAILED")
        results[label] = r

    # data_dir is the checkpoint dir — keep it for future runs

    # ── Print tables ───────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  PERFORMANCE  ({N:,} vectors, dim={DIM}, top_k={args.top_k})")
    print(bar)
    print(f"  {'Engine':<22}  {'Ingest':>8}  {'vec/s':>7}  {'Disk':>8}  "
          f"{'RAM-ing':>8}  {'RAM-qry':>8}  {'CPU-ing':>8}  {'CPU-qry':>8}  "
          f"{'p50':>7}  {'p95':>7}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*7}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")
    for label, *_ in ENGINES:
        r = results.get(label)
        if not r:
            print(f"  {label:<22}  FAILED")
            continue
        print(f"  {label:<22}  "
              f"{fmt(r['ingest_time'],'s'):>8}  "
              f"{int(r['ingest_speed']):>7,}  "
              f"{fmt(r['disk_mb'],'MB'):>8}  "
              f"{fmt(r['ingest_rss_mb'],'MB',0):>8}  "
              f"{fmt(r['retrieve_rss_mb'],'MB',0):>8}  "
              f"{fmt(r['ingest_cpu_util'],'%',0):>8}  "
              f"{fmt(r['query_cpu_util'],'%',0):>8}  "
              f"{fmt(r['p50_ms'],'ms',2):>7}  "
              f"{fmt(r['p95_ms'],'ms',2):>7}")

    print(f"\n{bar}")
    print(f"  RETRIEVAL QUALITY  (ground truth = brute-force cosine, {args.n_queries} queries)")
    print(bar)
    print(f"  {'Engine':<22}  {'Recall@'+str(args.top_k):>12}  {'MRR@'+str(args.top_k):>10}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*10}")
    for label, *_ in ENGINES:
        r = results.get(label)
        if not r:
            print(f"  {label:<22}  FAILED")
            continue
        print(f"  {label:<22}  "
              f"{r['recall_at_k']*100:>11.2f}%  "
              f"{r['mrr']:>10.4f}")

    print(f"\n{bar}")
    print(f"  DISK COMPRESSION  (float32 raw = {N*BYTES_F32/1024/1024:.0f}MB for {N:,} vectors)")
    print(bar)
    chroma_disk = (results.get("ChromaDB (HNSW)") or {}).get("disk_mb")
    print(f"  {'Engine':<22}  {'Disk (MB)':>10}  {'bytes/vec':>10}  "
          f"{'vs float32':>12}  {'vs ChromaDB':>13}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*13}")
    for label, *_ in ENGINES:
        r = results.get(label)
        if not r:
            continue
        bpv  = r["disk_mb"] * 1024 * 1024 / N
        vsf  = BYTES_F32 / bpv if bpv > 0 else 0
        vscd = chroma_disk / r["disk_mb"] if chroma_disk and r["disk_mb"] > 0 else None
        print(f"  {label:<22}  {r['disk_mb']:>10.1f}  {bpv:>10.0f}  "
              f"  x{vsf:.2f} smaller  "
              f"{'x'+str(round(vscd,2))+' smaller' if vscd else '---':>13}")

    print(f"\n{bar}")
    print(f"  FULL PIPELINE SUMMARY")
    print(bar)
    print(f"  Dataset    : DBpedia OpenAI embeddings — {N:,} vectors, dim={DIM}")
    print(f"  Source     : {DATASET}  (pre-computed text-embedding-ada-002)")
    print(f"  Queries    : {args.n_queries} vectors sampled from corpus, brute-force ground truth")
    print()
    print(f"  {'Engine':<22}  {'Ready':>8}  {'Disk':>8}  {'RAM':>8}  "
          f"{'p50':>7}  {'p95':>7}  {'Recall@'+str(args.top_k):>12}  {'MRR':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*8}")
    for label, *_ in ENGINES:
        r = results.get(label)
        if not r:
            print(f"  {label:<22}  FAILED")
            continue
        print(f"  {label:<22}  "
              f"{fmt(r['ingest_time'],'s'):>8}  "
              f"{fmt(r['disk_mb'],'MB'):>8}  "
              f"{fmt(r['retrieve_rss_mb'],'MB',0):>8}  "
              f"{fmt(r['p50_ms'],'ms',2):>7}  "
              f"{fmt(r['p95_ms'],'ms',2):>7}  "
              f"{r['recall_at_k']*100:>11.2f}%  "
              f"{r['mrr']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
