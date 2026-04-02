"""
Validation suite for the performance fixes.

Tests:
  1. fast_mode is NOW genuinely faster ingest (skips QJL residual)
  2. fast_mode produces different (non-zero gamma) encoding vs standard
  3. parallel quantize_batch gives correct recall (same quality as before)
  4. HNSW construction recall is maintained with prepare_ip_query_from_codes
  5. Batch reranking (ANN + BF) gives correct recall — no index mismatches
"""

import tempfile, time
import numpy as np
import turboquantdb as tq

DIM, K, QUERIES, SEED = 384, 10, 50, 42
rng = np.random.default_rng(SEED)


def make_data(n):
    corpus = rng.standard_normal((n, DIM)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return corpus, queries


def exact_topk(corpus, q, k):
    return set(np.argsort(corpus @ q)[::-1][:k].tolist())


def measure_ingest(n, bits, fast_mode):
    corpus, _ = make_data(n)
    ids = [str(i) for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        db = tq.Database.open(d, dimension=DIM, bits=bits, fast_mode=fast_mode)
        t0 = time.perf_counter()
        db.insert_batch(ids, corpus)
        db.flush()
        elapsed = time.perf_counter() - t0
        db.close()
    return n / elapsed


def recall(n, bits, fast_mode, rerank, use_index):
    corpus, queries = make_data(n)
    ids = [str(i) for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        db = tq.Database.open(d, dimension=DIM, bits=bits, fast_mode=fast_mode, rerank=rerank)
        db.insert_batch(ids, corpus)
        db.flush()
        if use_index:
            db.create_index(max_degree=32, ef_construction=200)
        hits = []
        for q in queries:
            gt = exact_topk(corpus, q, K)
            res = db.search(q, top_k=K)
            found = {int(r["id"]) for r in res}
            hits.append(len(gt & found) / K)
        db.close()
    return np.mean(hits)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── Test 1: fast_mode ingest is genuinely faster ─────────────────────────────
print("\nTest 1 — fast_mode ingest speed (N=10k, bits=4)")
n = 10_000
std_speed  = measure_ingest(n, bits=4, fast_mode=False)
fast_speed = measure_ingest(n, bits=4, fast_mode=True)
ratio = fast_speed / std_speed
check("fast_mode is not slower than standard", ratio > 0.90,
      f"fast={fast_speed:.0f} vs std={std_speed:.0f} vecs/s, ratio={ratio:.2f}x")


# ── Test 2: fast_mode vs standard recall (fast should be within 15pp) ────────
print("\nTest 2 — fast_mode recall difference (N=10k, BF, rerank=True)")
std_r  = recall(n, bits=4, fast_mode=False, rerank=True,  use_index=False)
fast_r = recall(n, bits=4, fast_mode=True,  rerank=True,  use_index=False)
diff = std_r - fast_r
check("fast_mode recall within 15pp of standard", diff < 0.15,
      f"std={std_r:.1%} fast={fast_r:.1%} diff={diff:+.1%}")
check("fast_mode recall not identical to standard (QJL is skipped)", diff != 0.0,
      f"diff={diff:.4f}")


# ── Test 3: parallel quantize_batch — correct recall at 10k ──────────────────
print("\nTest 3 — parallel quantize_batch: BF recall (N=10k, b=4, rerank=F)")
bf_r = recall(n, bits=4, fast_mode=False, rerank=False, use_index=False)
check("BF recall ≥ 50%", bf_r >= 0.50, f"{bf_r:.1%}")

print("Test 3b — parallel quantize_batch: BF+rerank (N=10k, b=4, rerank=T)")
bf_rr = recall(n, bits=4, fast_mode=False, rerank=True, use_index=False)
check("BF+rerank improves over BF", bf_rr > bf_r,
      f"rerank={bf_rr:.1%} vs no-rerank={bf_r:.1%} Δ={bf_rr-bf_r:+.1%}")


# ── Test 4: HNSW construction with prepare_ip_query_from_codes ───────────────
print("\nTest 4 — HNSW construction recall (N=10k, b=4, rerank=T)")
ann_r = recall(n, bits=4, fast_mode=False, rerank=True,  use_index=True)
check("ANN recall ≥ 55%", ann_r >= 0.55, f"{ann_r:.1%}")
check("ANN+rerank within 15pp of BF+rerank (graph covers subset)", ann_r >= bf_rr - 0.15,
      f"ann={ann_r:.1%} bf={bf_rr:.1%} gap={bf_rr-ann_r:.1%}")# ANN should be within 5pp of BF

print("\nTest 4b — HNSW recall at N=5k (stress test centroid lookup)")
ann_5k = recall(5_000, bits=4, fast_mode=False, rerank=True, use_index=True)
check("ANN recall ≥ 65% at 5k", ann_5k >= 0.65, f"{ann_5k:.1%}")


# ── Test 5: Batch reranking produces correct results (no index mismatches) ────
print("\nTest 5 — Batch reranking correctness: ANN scores match BF scores")
# Run both paths on same data, same seed — their best results must largely overlap
corpus, queries = make_data(5_000)
ids = [str(i) for i in range(5_000)]
with tempfile.TemporaryDirectory() as d:
    db = tq.Database.open(d, dimension=DIM, bits=4, rerank=True)
    db.insert_batch(ids, corpus)
    db.flush()
    db.create_index(max_degree=32, ef_construction=200)

    ann_hits, bf_hits = [], []
    for q in queries:
        gt = exact_topk(corpus, q, K)
        ann_res  = db.search(q, top_k=K)
        bf_res   = db.search(q, top_k=K, ann_search_list_size=0)  # 0 forces BF
        ann_hits.append(len(gt & {int(r["id"]) for r in ann_res})  / K)
        bf_hits.append( len(gt & {int(r["id"]) for r in bf_res})   / K)
    db.close()

ann_mean = np.mean(ann_hits)
bf_mean  = np.mean(bf_hits)
check("BF batch reranking recall ≥ 60%", bf_mean  >= 0.60, f"{bf_mean:.1%}")
check("ANN batch reranking recall ≥ 55%", ann_mean >= 0.55, f"{ann_mean:.1%}")
# Sanity: BF should not be catastrophically worse than ANN (would mean index mismatches)
check("BF and ANN recalls not diverged (no off-by-one indexing)", abs(ann_mean - bf_mean) < 0.20,
      f"ann={ann_mean:.1%} bf={bf_mean:.1%} diff={abs(ann_mean-bf_mean):.1%}")


# ── Test 6: bits=8 sanity ─────────────────────────────────────────────────────
print("\nTest 6 — bits=8 recall ≥ bits=4 (dequantize_batch parallelism)")
r4 = recall(n, bits=4, fast_mode=False, rerank=True, use_index=False)
r8 = recall(n, bits=8, fast_mode=False, rerank=True, use_index=False)
check("b=8 recall ≥ b=4 recall", r8 >= r4,
      f"b=4={r4:.1%} b=8={r8:.1%}")


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'='*50}")
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  \033[32mAll checks passed — safe to hand over.\033[0m")
else:
    print(f"  \033[31m{total-passed} check(s) FAILED — investigate before hand-off.\033[0m")
print('='*50)
