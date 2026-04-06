"""Sprint smoke test — fast perf regression gate.

Run after each audit phase to ensure no recall/latency regression.
Uses purely synthetic data: no dataset downloads required.

Pass criteria (checked at exit):
  - Recall@10  >= 0.80
  - P95 latency <= 50 ms
  - Ingest throughput >= 1000 vec/s

Typical runtime: ~5 seconds.

Usage:
    python benchmarks/sprint_smoke.py
"""

import sys
import tempfile
import time

import numpy as np

try:
    from tqdb import Database
except ImportError:
    print("ERROR: tqdb not installed. Run: maturin develop --release")
    sys.exit(1)

# ── Parameters ──────────────────────────────────────────────────────────────
D = 200          # vector dimension
N = 10_000       # corpus size
N_Q = 500        # number of queries
K = 10           # top-k
BITS = 4
SEED = 42

RECALL_FLOOR = 0.80
P95_LAT_MS   = 50.0
INGEST_MIN   = 1_000.0   # vectors/second

# ── Generate data ────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
corpus = rng.standard_normal((N, D)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

queries = rng.standard_normal((N_Q, D)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

# Brute-force ground truth (NumPy)
scores_gt = corpus @ queries.T               # (N, N_Q)
true_top_k = np.argsort(-scores_gt, axis=0)[:K].T  # (N_Q, K) — row = query

# ── Ingest ────────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    db = Database.open(tmpdir, dimension=D, bits=BITS, seed=SEED, metric="ip", rerank=True)

    ids = [str(i) for i in range(N)]
    t0 = time.perf_counter()
    db.insert_batch(ids, corpus, mode="insert")
    ingest_s = time.perf_counter() - t0
    throughput = N / ingest_s

    # ── Search ───────────────────────────────────────────────────────────────
    latencies = []
    hits = 0
    for qi in range(N_Q):
        t1 = time.perf_counter()
        results = db.search(queries[qi], top_k=K)
        lat_ms = (time.perf_counter() - t1) * 1000.0
        latencies.append(lat_ms)

        returned_ids = {int(r["id"]) for r in results}
        true_ids     = set(true_top_k[qi].tolist())
        hits += len(returned_ids & true_ids)

    db.close()

# ── Report ────────────────────────────────────────────────────────────────────
recall = hits / (N_Q * K)
p95_ms = float(np.percentile(latencies, 95))
p50_ms = float(np.percentile(latencies, 50))

print(f"\n── Sprint Smoke ({'PASS' if recall >= RECALL_FLOOR and p95_ms <= P95_LAT_MS else 'FAIL'}) ──")
print(f"  Ingest:   {throughput:>8,.0f} vec/s  (min {INGEST_MIN:.0f})")
print(f"  Recall@{K}: {recall:>8.3f}     (min {RECALL_FLOOR})")
print(f"  P50 lat:  {p50_ms:>8.2f} ms")
print(f"  P95 lat:  {p95_ms:>8.2f} ms  (max {P95_LAT_MS} ms)")

failures = []
if recall < RECALL_FLOOR:
    failures.append(f"Recall@{K} {recall:.3f} < {RECALL_FLOOR}")
if p95_ms > P95_LAT_MS:
    failures.append(f"P95 latency {p95_ms:.1f} ms > {P95_LAT_MS} ms")
if throughput < INGEST_MIN:
    failures.append(f"Throughput {throughput:.0f} vec/s < {INGEST_MIN:.0f}")

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)

print("\n  ✓ All checks passed")
