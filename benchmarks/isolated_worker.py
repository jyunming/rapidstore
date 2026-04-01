import os
import sys
import time
import json
import argparse
import psutil
import numpy as np
import shutil
import gc
import ctypes
from tempfile import mkdtemp

# Import engines
import lancedb
import chromadb
import turboquantdb

def get_dir_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def rss_mb():
    gc.collect()
    return psutil.Process().memory_info().rss / (1024 * 1024)

class CpuTimer:
    def __enter__(self):
        self.proc = psutil.Process()
        self.start_cpu = self.proc.cpu_times()
        self.start_time = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end_cpu = self.proc.cpu_times()
        self.end_time = time.perf_counter()
        dur = self.end_time - self.start_time
        user_diff = self.end_cpu.user - self.start_cpu.user
        sys_diff = self.end_cpu.system - self.start_cpu.system
        self.util = (user_diff + sys_diff) / dur * 100 if dur > 0 else 0

def brute_force_top_k(queries, vecs, k=10):
    scores = np.dot(queries, vecs.T)
    return np.argsort(-scores, axis=1)[:, :k]

def calculate_recall(found_ids, true_ids):
    recalls = []
    for f, t in zip(found_ids, true_ids):
        intersection = set(f).intersection(set(t))
        recalls.append(len(intersection) / len(t))
    return np.mean(recalls)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--fast_mode", action="store_true", default=False)
    args = parser.parse_args()

    N = args.n
    DIM = args.dim
    TOP_K = args.k
    RUNS = args.runs
    
    # 1. Generate Data
    vecs = np.random.randn(N, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-9)
    
    ids = [str(i) for i in range(N)]
    metadatas = [{"chunk": i} for i in range(N)]
    
    queries = np.random.randn(RUNS, DIM).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / (q_norms + 1e-9)

    # 2. Calculate Ground Truth
    ground_truth = brute_force_top_k(queries, vecs, k=TOP_K)

    tmp_dir = mkdtemp(prefix="isolated_bench_")
    ingest_peak_rss = 0
    ingest_time = 0
    latencies = []
    found_ids_list = []

    t_ready_start = time.perf_counter()
    
    try:
        with CpuTimer() as build_timer:
            if args.engine == "lancedb":
                db = lancedb.connect(tmp_dir)
                t0 = time.perf_counter()
                table = db.create_table("vectors", data=[{"vector": v, "id": i, "chunk": i} for v, i in zip(vecs, range(N))])
                ingest_time = time.perf_counter() - t0
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                table.create_index(num_partitions=256, num_sub_vectors=96)
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

            elif args.engine == "chromadb":
                _chroma_client = chromadb.PersistentClient(path=tmp_dir)
                col = _chroma_client.create_collection("vectors", metadata={"hnsw:space": "ip"})
                t0 = time.perf_counter()
                batch_size = 5000
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    col.add(ids=ids[i:end], embeddings=vecs[i:end].tolist())
                    ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                ingest_time = time.perf_counter() - t0

            elif args.engine == "tqdb":
                db_path = os.path.join(tmp_dir, "bench.tqdb")
                db = turboquantdb.TurboQuantDB.open(db_path, dimension=DIM, bits=args.bits, metric="ip", fast_mode=args.fast_mode)
                t0 = time.perf_counter()
                db.insert_batch(ids, vecs, metadatas=metadatas)
                ingest_time = time.perf_counter() - t0
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
                db.create_index(max_degree=32, ef_construction=200)
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
        
        ingest_cpu = build_timer.util
        t_ready = time.perf_counter() - t_ready_start
        
        # CLEAR DATA but keep queries and ground_truth
        del vecs, ids, metadatas
        gc.collect()
        
        retrieve_rss = rss_mb()
        
        with CpuTimer() as search_timer:
            if args.engine == "lancedb":
                _ldb_nprobes = min(20, max(4, 256 // 10))
                for i in range(RUNS):
                    t_s = time.perf_counter()
                    res = table.search(queries[i]).limit(TOP_K).nprobes(_ldb_nprobes).to_list()
                    latencies.append((time.perf_counter() - t_s) * 1000)
                    found_ids_list.append([int(r['id']) for r in res])
            elif args.engine == "chromadb":
                for i in range(RUNS):
                    t_s = time.perf_counter()
                    res = col.query(query_embeddings=queries[i:i+1].tolist(), n_results=TOP_K)
                    latencies.append((time.perf_counter() - t_s) * 1000)
                    found_ids_list.append([int(idx) for idx in res['ids'][0]])
                # Release ChromaDB handles before cleanup (Windows file-lock fix)
                del col, _chroma_client
                gc.collect()
                time.sleep(0.2)
            elif args.engine == "tqdb":
                db.search(np.random.randn(DIM).astype(np.float32), top_k=TOP_K) # Warmup
                for i in range(RUNS):
                    t_s = time.perf_counter()
                    res = db.search(queries[i], top_k=TOP_K, ann_search_list_size=128)
                    latencies.append((time.perf_counter() - t_s) * 1000)
                    found_ids_list.append([int(r['id']) for r in res])
        
        query_cpu = search_timer.util
        recall = calculate_recall(found_ids_list, ground_truth)
        disk_bytes = get_dir_size(tmp_dir)
        
        result = {
            "ready_time": t_ready,
            "disk_mb": disk_bytes / (1024 * 1024),
            "ingest_rss_mb": ingest_peak_rss,
            "ingest_cpu_util": ingest_cpu,
            "retrieve_rss_mb": retrieve_rss,
            "query_cpu_util": query_cpu,
            "p50_ms": np.median(latencies),
            "p96_ms": np.percentile(latencies, 96),
            "recall": recall
        }
        print(json.dumps(result))

    finally:
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
