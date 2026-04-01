"""
recall_bench.py
===============
Paper-aligned Recall@1@k benchmark for TurboQuantDB.

Recall@1@k = fraction of queries where the exact top-1 nearest neighbour
appears anywhere in the ANN top-k results.

Usage
-----
    # Real embeddings (default — bge-base-en-v1.5, d=768, 34k corpus)
    python benchmarks/recall_bench.py

    # Real embeddings, specific bits/sls
    python benchmarks/recall_bench.py --bits 8 --sls 256

    # Synthetic random vectors (matches original paper setup at d=1536)
    python benchmarks/recall_bench.py --synthetic --n 100000 --dim 1536 --bits 8

    # Multiple bits in one run
    python benchmarks/recall_bench.py --bits 4,8,10
"""

import argparse
import json
import os
import pathlib
import shutil
import tempfile
import time

import numpy as np
import turboquantdb

_HERE         = pathlib.Path(__file__).parent.parent
_QUAL         = _HERE / "Qualification" / "ClaudeQual"
EMBED_CACHE   = _QUAL / "real_data_embeddings_bge768.npy"
QUERY_CACHE   = _QUAL / "real_data_query_embs_bge768.npy"

K_VALUES = [1, 2, 4, 8, 16, 32, 64]


# ── helpers ───────────────────────────────────────────────────────────────────

def brute_force_top1(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Return index of exact top-1 IP neighbour for each query."""
    scores = np.dot(queries, corpus.T)          # (Q, N)
    return np.argmax(scores, axis=1)            # (Q,)


def recall_at_1_at_k(true_top1: np.ndarray, found: list[list[int]]) -> float:
    """Fraction of queries where true_top1[i] appears in found[i]."""
    hits = sum(1 for t, f in zip(true_top1, found) if t in f)
    return hits / len(true_top1)


def load_real_data(embed_path: str = None, query_path: str = None):
    embed_path = embed_path or str(EMBED_CACHE)
    query_path = query_path or str(QUERY_CACHE)
    if not pathlib.Path(embed_path).exists():
        raise FileNotFoundError(
            f"Cached embeddings not found: {embed_path}\n"
            "Run bench_isolated.py once to generate them, or pass --embed-cache."
        )
    corpus  = np.load(embed_path).astype("float32")
    queries = np.load(query_path).astype("float32")
    # Normalise to unit sphere so IP == cosine
    corpus  /= np.linalg.norm(corpus,  axis=1, keepdims=True) + 1e-9
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9
    return corpus, queries


def make_synthetic(n: int, dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    corpus  = rng.standard_normal((n,    dim)).astype("float32")
    queries = rng.standard_normal((200,  dim)).astype("float32")
    corpus  /= np.linalg.norm(corpus,  axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return corpus, queries


# ── benchmark one (corpus, queries, bits, sls) combination ───────────────────

def run_one(corpus: np.ndarray, queries: np.ndarray,
            bits: int, sls: int, label: str) -> dict:
    N, DIM = corpus.shape
    Q      = len(queries)
    ids    = [str(i) for i in range(N)]

    tmp = tempfile.mkdtemp(prefix=f"recall_b{bits}_")
    try:
        db = turboquantdb.Database.open(
            os.path.join(tmp, "db"),
            dimension=DIM, bits=bits, metric="ip"
        )

        t0 = time.perf_counter()
        # insert in batches to avoid large metadatas list
        BATCH = 10_000
        for off in range(0, N, BATCH):
            sl = slice(off, min(off + BATCH, N))
            db.insert_batch(
                ids=ids[sl],
                vectors=corpus[sl],
                metadatas=[{}] * (sl.stop - sl.start),
            )
        ingest_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        db.create_index()
        build_s = time.perf_counter() - t1

        # ground truth (exact inner product on original float32)
        true_top1 = brute_force_top1(queries, corpus)

        recalls = {}
        for k in K_VALUES:
            found = []
            for q in queries:
                res = db.search(
                    query=q,
                    top_k=k,
                    _use_ann=True,
                    ann_search_list_size=max(sls, k),
                )
                found.append([int(r["id"]) for r in res])
            recalls[k] = recall_at_1_at_k(true_top1, found)

        return {
            "label":    label,
            "n":        N,
            "dim":      DIM,
            "bits":     bits,
            "sls":      sls,
            "ingest_s": round(ingest_s, 2),
            "build_s":  round(build_s, 2),
            "recalls":  recalls,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic",  action="store_true",
                    help="Use random unit vectors instead of real embeddings")
    ap.add_argument("--n",   type=int, default=100_000,
                    help="Corpus size for synthetic mode (default 100k)")
    ap.add_argument("--dim", type=int, default=1536,
                    help="Dimension for synthetic mode (default 1536)")
    ap.add_argument("--bits", default="4,8,10",
                    help="Comma-separated bits values (default: 4,8,10)")
    ap.add_argument("--sls",  type=int, default=128,
                    help="ann_search_list_size (default 128)")
    ap.add_argument("--embed-cache",  default=str(EMBED_CACHE),
                    help="Path to corpus .npy file")
    ap.add_argument("--query-cache",  default=str(QUERY_CACHE),
                    help="Path to queries .npy file")
    ap.add_argument("--out",  default="recall_results.json",
                    help="Output JSON file")
    args = ap.parse_args()

    bits_list = [int(b.strip()) for b in args.bits.split(",")]

    if args.synthetic:
        corpus, queries = make_synthetic(args.n, args.dim)
        mode = f"synthetic  n={args.n:,}  dim={args.dim}"
    else:
        corpus, queries = load_real_data(args.embed_cache, args.query_cache)
        mode = f"real (bge-base-en-v1.5)  n={len(corpus):,}  dim={corpus.shape[1]}"

    print(f"\nRecall@1@k  |  {mode}  |  sls={args.sls}")
    print("=" * 72)

    all_results = []
    for bits in bits_list:
        label = f"TQDB b{bits}"
        print(f"\n  Running {label} ...", flush=True)
        r = run_one(corpus, queries, bits=bits, sls=args.sls, label=label)
        all_results.append(r)

        # print table row for this bits value
        row = f"  {label:<10}  ingest={r['ingest_s']:.1f}s  build={r['build_s']:.1f}s  |"
        for k in K_VALUES:
            row += f"  @{k}={r['recalls'][k]*100:.1f}%"
        print(row)

    # summary table
    print()
    print("=" * 72)
    print(f"  {'':10}  " + "".join(f"  @{k:<5}" for k in K_VALUES))
    print("  " + "-" * 68)
    for r in all_results:
        cells = "".join(f"  {r['recalls'][k]*100:>5.1f}%" for k in K_VALUES)
        print(f"  {r['label']:<10}{cells}")
    print()

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
