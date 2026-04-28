"""
Retrieval evaluation harness — semantic / lexical / mixed grading.

Compares the three retrieval paths (dense, BM25 sparse, hybrid RRF) on the
same query sets, with the same metrics, so v0.7's hybrid feature can be judged
against measured wins instead of anecdotes.

The default corpus is synthetic and self-contained: every test produces 1k
documents, each with a unit-normal vector and a short keyword-bearing string.
That keeps the harness reproducible in CI and avoids HuggingFace download
dependencies. Real datasets can be plugged in by passing ``--corpus``.

Three query sets:
  - **semantic**: query = corpus_vec + small noise; ideal retriever returns
    the corresponding doc. Pure dense should win here.
  - **lexical**: query carries a rare token that uniquely identifies one doc;
    query vector is orthogonal to that doc. Pure BM25 should win here.
  - **mixed**: half-and-half (each query is one of the two flavours, randomly).

Metrics per (query_set × path): R@1, R@10, MRR@10, NDCG@10, p50/p95 latency.

Usage:
    python benchmarks/retrieval_eval.py             # synthetic, 1k corpus, prints table
    python benchmarks/retrieval_eval.py --n 5000    # bigger corpus
    python benchmarks/retrieval_eval.py --no-history  # don't append to history file

Output:
    - markdown table on stdout
    - append a row to ``benchmarks/retrieval_eval_history.json``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import string
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow `import tqdb` from the local maturin develop install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

BENCH_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BENCH_DIR / "retrieval_eval_history.json"

# ── Corpus + query generation ────────────────────────────────────────────────


COMMON_WORDS = (
    "the of and to in is for on with that this from by it as at "
    "an be are was were has have had not but or if then which when"
).split()


def _rare_token(rng: random.Random) -> str:
    """A unique-ish token unlikely to collide across docs.

    Format ``rt-{6-char base36}``. The space is large enough that 1k draws
    almost never collide; if they do, the eval is still fair because both the
    query and the expected doc share the duplicated tag.
    """
    chars = string.ascii_lowercase + string.digits
    return "rt-" + "".join(rng.choices(chars, k=6))


def synth_corpus(n: int, dim: int, seed: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(vectors, texts, rare_tokens)`` for a synthetic corpus.

    Each doc gets:
      - a unit-norm random vector,
      - a short text built from common words plus exactly one rare token,
      - the rare token recorded separately so the lexical query set can target it.
    """
    rng_np = np.random.default_rng(seed)
    rng = random.Random(seed)

    vecs = rng_np.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)

    texts: list[str] = []
    rare_tokens: list[str] = []
    for _ in range(n):
        rare = _rare_token(rng)
        rare_tokens.append(rare)
        n_common = rng.randint(15, 40)
        words = rng.choices(COMMON_WORDS, k=n_common) + [rare]
        rng.shuffle(words)
        texts.append(" ".join(words))
    return vecs, texts, rare_tokens


def make_semantic_queries(
    vecs: np.ndarray, n_q: int, noise: float, seed: int
) -> tuple[np.ndarray, list[str], list[int]]:
    """Pick `n_q` corpus rows, perturb each vector slightly, and use empty text.

    Returns ``(query_vecs, query_texts, gold_indices)``.
    """
    rng_np = np.random.default_rng(seed + 1)
    rng = random.Random(seed + 1)
    n_corpus = vecs.shape[0]
    gold = rng.sample(range(n_corpus), n_q)
    perturb = rng_np.standard_normal((n_q, vecs.shape[1])).astype(np.float32)
    perturb *= noise / max(np.linalg.norm(perturb, axis=1, keepdims=True).max(), 1e-9)
    qvecs = vecs[gold] + perturb
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True).clip(min=1e-9)
    qtexts = [""] * n_q  # semantic queries don't need text
    return qvecs, qtexts, gold


def make_lexical_queries(
    vecs: np.ndarray, rare_tokens: list[str], n_q: int, seed: int
) -> tuple[np.ndarray, list[str], list[int]]:
    """Each query: query_text = a doc's rare token; query_vec = random unit vec.

    The orthogonal-ish random vector ensures dense retrieval cannot recover the
    target — a fair bar for hybrid to clear.
    """
    rng_np = np.random.default_rng(seed + 2)
    rng = random.Random(seed + 2)
    n_corpus, dim = vecs.shape
    gold = rng.sample(range(n_corpus), n_q)
    qvecs = rng_np.standard_normal((n_q, dim)).astype(np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True).clip(min=1e-9)
    qtexts = [rare_tokens[i] for i in gold]
    return qvecs, qtexts, gold


def make_mixed_queries(
    vecs: np.ndarray, rare_tokens: list[str], n_q: int, seed: int, noise: float
) -> tuple[np.ndarray, list[str], list[int]]:
    """Half semantic, half lexical (interleaved). Mirrors realistic mixed workloads."""
    half = n_q // 2
    sv, st, sg = make_semantic_queries(vecs, half, noise, seed + 3)
    lv, lt, lg = make_lexical_queries(vecs, rare_tokens, n_q - half, seed + 4)
    qvecs = np.vstack([sv, lv])
    qtexts = list(st) + list(lt)
    gold = list(sg) + list(lg)
    return qvecs, qtexts, gold


# ── Metrics ──────────────────────────────────────────────────────────────────


def metrics(retrieved: list[list[str]], gold_ids: list[str], k: int = 10) -> dict[str, float]:
    """R@1, R@k, MRR@k, NDCG@k for binary single-target relevance.

    For a single relevant target per query, IDCG@k collapses to 1/log2(2) = 1,
    so NDCG@k = 1/log2(rank+1) when the gold doc is in the top-k, else 0.
    """
    n = len(gold_ids)
    r_at_1 = 0
    r_at_k = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    for ranking, gold in zip(retrieved, gold_ids):
        topk = ranking[:k]
        if topk and topk[0] == gold:
            r_at_1 += 1
        if gold in topk:
            r_at_k += 1
            rank = topk.index(gold) + 1  # 1-based
            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
    return {
        f"R@1": r_at_1 / n if n else 0.0,
        f"R@{k}": r_at_k / n if n else 0.0,
        f"MRR@{k}": mrr_sum / n if n else 0.0,
        f"NDCG@{k}": ndcg_sum / n if n else 0.0,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[idx]


# ── Path runners (return rankings + per-query latencies) ─────────────────────


def _ids_from(results: list[dict]) -> list[str]:
    return [str(r["id"]) for r in results]


def run_dense(db, qvecs: np.ndarray, k: int) -> tuple[list[list[str]], list[float]]:
    rankings: list[list[str]] = []
    latencies: list[float] = []
    for qv in qvecs:
        t0 = time.perf_counter()
        res = db.search(qv, top_k=k)
        latencies.append(time.perf_counter() - t0)
        rankings.append(_ids_from(res))
    return rankings, latencies


def run_hybrid(
    db, qvecs: np.ndarray, qtexts: list[str], k: int, weight: float
) -> tuple[list[list[str]], list[float]]:
    rankings: list[list[str]] = []
    latencies: list[float] = []
    for qv, qt in zip(qvecs, qtexts):
        if not qt:
            # No text → fall back to dense to avoid wasting a hybrid call on
            # an empty BM25 query (which collapses to dense anyway).
            t0 = time.perf_counter()
            res = db.search(qv, top_k=k)
            latencies.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            res = db.search(qv, top_k=k, hybrid={"text": qt, "weight": weight})
            latencies.append(time.perf_counter() - t0)
        rankings.append(_ids_from(res))
    return rankings, latencies


def run_bm25_only(
    db, qvecs: np.ndarray, qtexts: list[str], k: int
) -> tuple[list[list[str]], list[float]]:
    """weight=1.0 hybrid is "pure BM25" with the dense leg ignored by RRF."""
    rankings: list[list[str]] = []
    latencies: list[float] = []
    for qv, qt in zip(qvecs, qtexts):
        if not qt:
            # Empty text → BM25 has nothing to score; record an empty ranking.
            rankings.append([])
            latencies.append(0.0)
            continue
        t0 = time.perf_counter()
        res = db.search(qv, top_k=k, hybrid={"text": qt, "weight": 1.0})
        latencies.append(time.perf_counter() - t0)
        rankings.append(_ids_from(res))
    return rankings, latencies


# ── Orchestration ────────────────────────────────────────────────────────────


def evaluate_one(label: str, runner_out: tuple[list[list[str]], list[float]], gold_ids: list[str]) -> dict:
    rankings, latencies = runner_out
    m = metrics(rankings, gold_ids, k=10)
    m["p50_ms"] = percentile(latencies, 50) * 1000.0
    m["p95_ms"] = percentile(latencies, 95) * 1000.0
    m["path"] = label
    return m


def render_table(rows: list[dict], query_set: str) -> str:
    """Produce the markdown comparison table for one query set."""
    head = (
        f"### {query_set}\n\n"
        "| path   | R@1   | R@10  | MRR@10 | NDCG@10 | p50 (ms) | p95 (ms) |\n"
        "|--------|-------|-------|--------|---------|----------|----------|\n"
    )
    body = []
    for r in rows:
        body.append(
            f"| {r['path']:<6} | {r['R@1']:.3f} | {r['R@10']:.3f} | "
            f"{r['MRR@10']:.3f}  | {r['NDCG@10']:.3f}   | "
            f"{r['p50_ms']:>8.2f} | {r['p95_ms']:>8.2f} |"
        )
    return head + "\n".join(body) + "\n"


def append_history(report: dict) -> None:
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    history.append(report)
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="Synthetic corpus size (default 1000).")
    p.add_argument("--dim", type=int, default=128, help="Vector dimension (default 128).")
    p.add_argument("--n_queries", type=int, default=100, help="Queries per query set.")
    p.add_argument("--noise", type=float, default=0.05, help="Semantic-query perturbation.")
    p.add_argument("--weight", type=float, default=0.5, help="Hybrid BM25 weight.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-history", action="store_true", help="Don't append to history JSON.")
    args = p.parse_args()

    import tqdb  # imported here so --help works without the binding installed

    print(f"# Retrieval Evaluation — N={args.n} D={args.dim} Q={args.n_queries}", flush=True)
    print(f"# {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n", flush=True)

    vecs, texts, rare_tokens = synth_corpus(args.n, args.dim, args.seed)
    ids = [f"d{i}" for i in range(args.n)]

    query_sets = {
        "semantic": make_semantic_queries(vecs, args.n_queries, args.noise, args.seed),
        "lexical":  make_lexical_queries(vecs, rare_tokens, args.n_queries, args.seed),
        "mixed":    make_mixed_queries(vecs, rare_tokens, args.n_queries, args.seed, args.noise),
    }

    with tempfile.TemporaryDirectory(prefix="retrieval_eval_") as tmp:
        db = tqdb.Database.open(tmp, dimension=args.dim, bits=4, metric="ip")
        # 2k batches keep memory low without paying per-vector overhead.
        for start in range(0, args.n, 2000):
            end = min(start + 2000, args.n)
            db.insert_batch(ids[start:end], vecs[start:end], None, texts[start:end])
        db.flush()

        report = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n": args.n,
            "dim": args.dim,
            "n_queries": args.n_queries,
            "weight": args.weight,
            "results": {},
        }
        markdown = []
        for qs_name, (qvecs, qtexts, gold_idx) in query_sets.items():
            gold_ids = [ids[i] for i in gold_idx]
            rows = [
                evaluate_one("dense",  run_dense(db, qvecs, 10), gold_ids),
                evaluate_one("bm25",   run_bm25_only(db, qvecs, qtexts, 10), gold_ids),
                evaluate_one("hybrid", run_hybrid(db, qvecs, qtexts, 10, args.weight), gold_ids),
            ]
            report["results"][qs_name] = rows
            markdown.append(render_table(rows, qs_name))

        db.close()

    print("\n".join(markdown))

    if not args.no_history:
        append_history(report)
        print(f"\nappended to {HISTORY_PATH.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
