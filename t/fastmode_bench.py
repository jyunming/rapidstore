"""Measure fast_mode ingest and recall tradeoff."""
import tempfile, time
import numpy as np
import turboquantdb as tq

DIM, SEED, QUERIES, K = 384, 42, 50, 10
rng = np.random.default_rng(SEED)


def bench_ingest(n, bits, fast_mode, runs=5):
    corpus = rng.standard_normal((n, DIM)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    ids = [str(i) for i in range(n)]
    times = []
    for _ in range(runs):
        with tempfile.TemporaryDirectory() as d:
            db = tq.Database.open(d, dimension=DIM, bits=bits, fast_mode=fast_mode)
            t0 = time.perf_counter()
            db.insert_batch(ids, corpus)
            db.flush()
            times.append(time.perf_counter() - t0)
            db.close()
    times.sort()
    med = times[len(times) // 2]
    return med, n / med


def bench_recall(n, bits, fast_mode):
    corpus = rng.standard_normal((n, DIM)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    ids = [str(i) for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        db = tq.Database.open(d, dimension=DIM, bits=bits, fast_mode=fast_mode, rerank=True)
        db.insert_batch(ids, corpus)
        db.flush()
        db.create_index(max_degree=32, ef_construction=200)
        hits = []
        for q in queries:
            gt = set(np.argsort(corpus @ q)[::-1][:K].tolist())
            found = {int(r["id"]) for r in db.search(q, top_k=K)}
            hits.append(len(gt & found) / K)
        db.close()
    return float(np.mean(hits))


print("Ingest speed  (median of 5 cold runs, vecs/s)")
print()
print(f"  {'N':>8}  {'b4 std':>9}  {'b4 fast':>9}  {'b4 ratio':>9}  {'b8 std':>9}  {'b8 fast':>9}  {'b8 ratio':>9}")
print("  " + "-" * 70)
for n in [10_000, 25_000, 50_000]:
    _, r4s = bench_ingest(n, 4, False)
    _, r4f = bench_ingest(n, 4, True)
    _, r8s = bench_ingest(n, 8, False)
    _, r8f = bench_ingest(n, 8, True)
    r4s_s = f"{r4s:,.0f}/s"
    r4f_s = f"{r4f:,.0f}/s"
    r8s_s = f"{r8s:,.0f}/s"
    r8f_s = f"{r8f:,.0f}/s"
    print(f"  {n:>8,}  {r4s_s:>9}  {r4f_s:>9}  {r4f/r4s:>8.2f}x  {r8s_s:>9}  {r8f_s:>9}  {r8f/r8s:>8.2f}x")

print()
print("Recall@10  (ANN, max_degree=32, ef_construction=200, rerank=True)")
print()
print(f"  {'N':>8}  {'b4 std':>9}  {'b4 fast':>9}  {'b4 drop':>9}  {'b8 std':>9}  {'b8 fast':>9}  {'b8 drop':>9}")
print("  " + "-" * 70)
for n in [10_000, 25_000]:
    r4s = bench_recall(n, 4, False)
    r4f = bench_recall(n, 4, True)
    r8s = bench_recall(n, 8, False)
    r8f = bench_recall(n, 8, True)
    print(f"  {n:>8,}  {r4s:>8.1%}  {r4f:>8.1%}  {r4f-r4s:>+8.1%}  {r8s:>8.1%}  {r8f:>8.1%}  {r8f-r8s:>+8.1%}")
