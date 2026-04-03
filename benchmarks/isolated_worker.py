"""
Isolated benchmark worker — runs ONE engine in a fresh process.
Called by full_comparison_bench.py via subprocess.
Prints a single JSON line to stdout.
"""
import os
import sys
import time
import json
import argparse
import psutil
import numpy as np
import shutil
import gc
from tempfile import mkdtemp

import lancedb
import chromadb
import turboquantdb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchParams

# Raise process priority so the OS scheduler doesn't preempt us mid-benchmark,
# which would inflate wall-clock timings unpredictably.
try:
    _proc = psutil.Process()
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
    else:
        _proc.nice(-10)
except Exception:
    pass


def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def rss_mb():
    gc.collect()
    return psutil.Process().memory_info().rss / (1024 * 1024)


class CpuTimer:
    def __enter__(self):
        self.proc = psutil.Process()
        self.start_cpu = self.proc.cpu_times()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_):
        end_cpu = self.proc.cpu_times()
        dur = time.perf_counter() - self.start_time
        used = (end_cpu.user - self.start_cpu.user) + (end_cpu.system - self.start_cpu.system)
        self.util = used / dur * 100 if dur > 0 else 0


def brute_force_top_k(queries, vecs, k):
    scores = np.dot(queries, vecs.T)
    return np.argsort(-scores, axis=1)[:, :k]


def recall_at_k(found, truth):
    return float(np.mean([
        len(set(f) & set(t)) / len(t)
        for f, t in zip(found, truth)
    ]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine",    required=True)
    p.add_argument("--n",         type=int, required=True)
    p.add_argument("--dim",       type=int, required=True)
    p.add_argument("--bits",      type=int, default=8)
    p.add_argument("--k",         type=int, default=10)
    p.add_argument("--runs",      type=int, default=50)
    p.add_argument("--fast_mode", action="store_true", default=False)
    p.add_argument("--rerank",    action="store_true", default=False)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    N, DIM, K, RUNS = args.n, args.dim, args.k, args.runs

    rng = np.random.default_rng(args.seed)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

    queries = rng.standard_normal((RUNS, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9

    ids = [str(i) for i in range(N)]
    metadatas = [{"chunk": i} for i in range(N)]

    ground_truth = brute_force_top_k(queries, vecs, k=K)

    tmp = mkdtemp(prefix="bench_")
    ingest_peak_rss = 0
    ingest_time = 0
    latencies = []
    found_ids = []

    t_ready_start = time.perf_counter()

    try:
        with CpuTimer() as build_timer:

            # ── FAISS Flat (recall ceiling, not a vector store) ───────────────
            if args.engine == "faiss_flat":
                import faiss
                index = faiss.IndexFlatIP(DIM)
                t0 = time.perf_counter()
                index.add(vecs)
                ingest_time = time.perf_counter() - t0
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

            # ── LanceDB (IVF_PQ, nprobes=20) ─────────────────────────────────
            elif args.engine == "lancedb":
                db = lancedb.connect(tmp)
                t0 = time.perf_counter()
                table = db.create_table("v", data=[
                    {"vector": v, "id": i} for v, i in zip(vecs, range(N))])
                ingest_time = time.perf_counter() - t0
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                nlist = min(256, max(4, N // 40))
                table.create_index(num_partitions=nlist,
                                   num_sub_vectors=max(1, DIM // 8))
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

            # ── ChromaDB (HNSW default) ───────────────────────────────────────
            elif args.engine == "chromadb":
                _client = chromadb.PersistentClient(path=tmp)
                col = _client.create_collection(
                    "vecs", metadata={"hnsw:space": "ip"})
                t0 = time.perf_counter()
                for i in range(0, N, 5000):
                    end = min(i + 5000, N)
                    col.add(ids=ids[i:end], embeddings=vecs[i:end].tolist())
                    ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                ingest_time = time.perf_counter() - t0

            # ── Qdrant (persistent, HNSW m=32 ef=200) ───────────────────────
            elif args.engine == "qdrant":
                from qdrant_client.models import OptimizersConfigDiff, HnswConfigDiff
                _qclient = QdrantClient(path=os.path.join(tmp, "qdrant"))
                _qclient.create_collection(
                    "bench",
                    vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
                    hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
                    optimizers_config=OptimizersConfigDiff(indexing_threshold=N + 1))
                t0 = time.perf_counter()
                for i in range(0, N, 5000):
                    end = min(i + 5000, N)
                    _qclient.upsert("bench", points=[
                        PointStruct(id=j, vector=vecs[j].tolist())
                        for j in range(i, end)])
                    ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                ingest_time = time.perf_counter() - t0
                # trigger HNSW index build now that all vectors are loaded
                _qclient.update_collection(
                    "bench",
                    optimizers_config=OptimizersConfigDiff(indexing_threshold=0))
                # wait for indexing to complete
                import time as _time
                while True:
                    info = _qclient.get_collection("bench")
                    if info.status.value == "green":
                        break
                    _time.sleep(0.5)
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

            # ── TurboQuantDB ──────────────────────────────────────────────────
            elif args.engine == "tqdb":
                db_path = os.path.join(tmp, "bench.tqdb")
                db = turboquantdb.TurboQuantDB.open(
                    db_path, dimension=DIM, bits=args.bits,
                    metric="ip", rerank=args.rerank,
                    fast_mode=args.fast_mode)
                t0 = time.perf_counter()
                db.insert_batch(ids, vecs, metadatas=metadatas)
                ingest_time = time.perf_counter() - t0
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                db.create_index(max_degree=32, ef_construction=200,
                                search_list_size=128)
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        ingest_cpu = build_timer.util
        t_ready = time.perf_counter() - t_ready_start

        del vecs, ids, metadatas
        gc.collect()
        retrieve_rss = rss_mb()

        with CpuTimer() as query_timer:

            if args.engine == "faiss_flat":
                import faiss
                # warmup
                index.search(queries[0:1], K)
                for i in range(RUNS):
                    t0 = time.perf_counter()
                    _, I = index.search(queries[i:i+1], K)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    found_ids.append(I[0].tolist())

            elif args.engine == "lancedb":
                for i in range(RUNS):
                    t0 = time.perf_counter()
                    res = table.search(queries[i]).limit(K).nprobes(20).to_list()
                    latencies.append((time.perf_counter() - t0) * 1000)
                    found_ids.append([int(r["id"]) for r in res])

            elif args.engine == "chromadb":
                for i in range(RUNS):
                    t0 = time.perf_counter()
                    res = col.query(query_embeddings=queries[i:i+1].tolist(),
                                    n_results=K)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    found_ids.append([int(x) for x in res["ids"][0]])
                del col, _client
                gc.collect()
                time.sleep(0.2)

            elif args.engine == "qdrant":
                # warmup
                _qclient.query_points("bench", query=queries[0].tolist(), limit=K,
                                      search_params=SearchParams(hnsw_ef=128))
                for i in range(RUNS):
                    t0 = time.perf_counter()
                    res = _qclient.query_points("bench", query=queries[i].tolist(),
                                                limit=K,
                                                search_params=SearchParams(hnsw_ef=128))
                    latencies.append((time.perf_counter() - t0) * 1000)
                    found_ids.append([p.id for p in res.points])

            elif args.engine == "tqdb":
                db.search(queries[0], top_k=K)  # warmup
                for i in range(RUNS):
                    t0 = time.perf_counter()
                    res = db.search(queries[i], top_k=K,
                                    ann_search_list_size=128)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    found_ids.append([int(r["id"]) for r in res])

        query_cpu = query_timer.util
        recall = recall_at_k(found_ids, ground_truth)
        disk_mb = get_dir_size(tmp) / (1024 * 1024)

        print(json.dumps({
            "ready_time":       t_ready,
            "ingest_time":      ingest_time,
            "disk_mb":          disk_mb,
            "ingest_rss_mb":    ingest_peak_rss,
            "retrieve_rss_mb":  retrieve_rss,
            "ingest_cpu_util":  ingest_cpu,
            "query_cpu_util":   query_cpu,
            "p50_ms":           float(np.median(latencies)),
            "p96_ms":           float(np.percentile(latencies, 96)),
            "recall":           recall,
        }))

    finally:
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
