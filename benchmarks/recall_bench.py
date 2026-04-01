import os
import numpy as np
import time
import json
import turboquantdb
from tempfile import mkdtemp
import shutil
import argparse

def brute_force_top_k(queries, vecs, k=1):
    # Perfect Inner Product search (exact top-k)
    scores = np.dot(queries, vecs.T)
    return np.argsort(-scores, axis=1)[:, :k]

def calculate_recall_at_1k(found_ids, true_ids):
    # found_ids: list of lists, true_ids: list of singleton ids
    recalls = []
    for f, t in zip(found_ids, true_ids):
        recalls.append(1.0 if t in f else 0.0)
    return np.mean(recalls)

def main():
    parser = argparse.ArgumentParser(description="Recall@1@k benchmark (paper-aligned).")
    parser.add_argument("--n", type=int, default=100000, help="Number of vectors (default: 100k)")
    parser.add_argument("--dim", type=int, default=1536, help="Vector dimension (default: 1536)")
    parser.add_argument("--bits", type=int, default=4, help="TurboQuantDB bits (default: 4)")
    parser.add_argument("--runs", type=int, default=200, help="Number of queries (default: 200)")
    parser.add_argument("--k", type=str, default="1,2,4,8,16,32,64", help="Comma-separated k values")
    parser.add_argument("--sls", type=int, default=128, help="ANN search_list_size (default: 128)")
    args = parser.parse_args()

    N = args.n
    DIM = args.dim
    BITS = args.bits
    RUNS = args.runs
    K_LIST = [int(x) for x in args.k.split(",") if x.strip()]

    print(f"Recall@1@k Benchmark (N={N}, DIM={DIM}, bits={BITS}, runs={RUNS})")
    print("Generating data and Ground Truth...")
    
    vecs = np.random.randn(N, DIM).astype(np.float32)
    # Normalize for easier logic, though engines handle raw
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-9)
    
    queries = np.random.randn(RUNS, DIM).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / (q_norms + 1e-9)
    
    # IDs as integers for recall calculation
    ids = [str(i) for i in range(N)]
    
    t0 = time.perf_counter()
    ground_truth = brute_force_top_k(queries, vecs, k=1).reshape(-1)
    print(f"Ground Truth calculated in {time.perf_counter()-t0:.2f}s")

    results = {}

    # --- Test TQDB ---
    tmp = mkdtemp()
    db = turboquantdb.TurboQuantDB.open(os.path.join(tmp, f"b{BITS}"), dimension=DIM, bits=BITS, metric="ip")
    db.insert_batch(ids, vecs)
    db.create_index()
    for k in K_LIST:
        found = []
        for q in queries:
            res = db.search(q, top_k=k, ann_search_list_size=args.sls)
            found.append([int(r["id"]) for r in res])
        results[f"TQDB (b{BITS})@{k}"] = calculate_recall_at_1k(found, ground_truth)
    shutil.rmtree(tmp)

    print("\nRecall@1@k Results:")
    print("-" * 30)
    for engine, recall in results.items():
        print(f"{engine:15} : {recall*100:6.2f}%")
    with open("recall_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
