"""Quick recall comparison: rerank=False vs rerank=True at multiple scales."""
import tempfile, time, numpy as np
import turboquantdb as tq

DIM, BITS, K, QUERIES, SEED = 384, 4, 10, 50, 42
rng = np.random.default_rng(SEED)

def exact_topk(corpus, query, k):
    scores = corpus @ query
    return set(np.argsort(scores)[::-1][:k].tolist())

def run(n, rerank, use_index):
    corpus = rng.standard_normal((n, DIM)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as d:
        db = tq.Database.open(d, dimension=DIM, bits=BITS, rerank=rerank)
        ids = [str(i) for i in range(n)]
        db.insert_batch(ids, corpus)
        db.flush()
        if use_index:
            db.create_index(max_degree=32, ef_construction=200)
        recalls = []
        for q in queries:
            gt = exact_topk(corpus, q, K)
            res = db.search(q, top_k=K)
            found = {int(r["id"]) for r in res}
            recalls.append(len(gt & found) / K)
        db.close()
    return np.mean(recalls)

print(f"{'Scale':>8}  {'BF rerank=F':>11}  {'BF rerank=T':>11}  {'ANN rerank=F':>12}  {'ANN rerank=T':>12}  {'BF loss':>8}  {'ANN loss':>9}")
print("-" * 82)
for n in [1_000, 5_000, 10_000, 25_000]:
    bf_f = run(n, rerank=False, use_index=False)
    bf_t = run(n, rerank=True,  use_index=False)
    ann_f = run(n, rerank=False, use_index=True)
    ann_t = run(n, rerank=True,  use_index=True)
    print(f"{n:>8,}  {bf_f:>10.1%}   {bf_t:>10.1%}   {ann_f:>11.1%}   {ann_t:>11.1%}   {bf_f-bf_t:>+7.1%}   {ann_f-ann_t:>+8.1%}")
