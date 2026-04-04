"""
Reproduces Section 4.4 (Near Neighbour Search) of arXiv:2504.19874.

Three datasets, matching the paper exactly:
  (a) GloVe d=200       — 100k train sample, 10k pre-existing queries
  (b) DBpedia d=1536    — 100k train sample, 1k queries
  (c) DBpedia d=3072    — 100k train sample, 1k queries

Bits : 2 and 4, brute-force only (no HNSW)
Metric: Recall@1@k — how often true top-1 (by inner product) is in top-k
k    : 1, 2, 4, 8, 16, 32, 64

Paper values (approximate, read from Figure 5a/5b/5c):
  GloVe d=200:   2-bit k@1≈0.55, 4-bit k@1≈0.86
  DBpedia d=1536: 2-bit k@1≈0.895, 4-bit k@1≈0.970
  DBpedia d=3072: 2-bit k@1≈0.905, 4-bit k@1≈0.975

Usage:
    .venv/Scripts/python benchmarks/paper_recall_bench.py
    .venv/Scripts/python benchmarks/paper_recall_bench.py --datasets glove dbpedia1536 dbpedia3072
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

K_VALUES  = [1, 2, 4, 8, 16, 32, 64]
BITS_LIST = [2, 4]
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "_paper_bench_cache")

# Paper's approximate recall values from Figure 5 (visual read)
PAPER_RECALL = {
    "glove-200": {
        2: {1:0.55, 2:0.70, 4:0.83, 8:0.91, 16:0.96, 32:0.99, 64:1.00},
        4: {1:0.86, 2:0.96, 4:0.99, 8:1.00, 16:1.00, 32:1.00, 64:1.00},
    },
    "dbpedia-1536": {
        2: {1:0.895, 2:0.980, 4:0.995, 8:1.000, 16:1.000, 32:1.000, 64:1.000},
        4: {1:0.970, 2:1.000, 4:1.000, 8:1.000, 16:1.000, 32:1.000, 64:1.000},
    },
    "dbpedia-3072": {
        2: {1:0.905, 2:0.985, 4:0.995, 8:1.000, 16:1.000, 32:1.000, 64:1.000},
        4: {1:0.975, 2:1.000, 4:1.000, 8:1.000, 16:1.000, 32:1.000, 64:1.000},
    },
}


# ── GloVe data loading ────────────────────────────────────────────────────────

def load_glove():
    """
    Load GloVe-200-angular from open-vdb/glove-200-angular on HuggingFace.
    Corpus subset  → sample 100k train vectors (normalize to unit norm for IP)
    Neighbors subset → 10k queries + pre-computed ground truth top-1
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_vecs  = os.path.join(CKPT_DIR, "glove200_100000_vecs.npy")
    ckpt_qvecs = os.path.join(CKPT_DIR, "glove200_10000_qvecs.npy")
    ckpt_truth = os.path.join(CKPT_DIR, "glove200_100000_truth_top1.npy")

    HF_DATASET = "open-vdb/glove-200-angular"
    N_DATA, N_QUERIES = 100_000, 10_000

    if os.path.exists(ckpt_vecs) and os.path.exists(ckpt_qvecs):
        print("  Loading cached GloVe-200 vectors ...", flush=True)
        vecs  = np.load(ckpt_vecs)
        qvecs = np.load(ckpt_qvecs)
    else:
        from datasets import load_dataset
        print(f"  Downloading GloVe-200 corpus (100k sample) from {HF_DATASET} ...", flush=True)
        ds_corpus = load_dataset(HF_DATASET, "train", split="train", streaming=True)
        raw = []
        for i, row in enumerate(ds_corpus):
            if i >= N_DATA:
                break
            raw.append(row["emb"])
            if (i + 1) % 10_000 == 0:
                print(f"    corpus {i+1:>7,} / {N_DATA:,}", flush=True)
        vecs = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.where(norms > 0, norms, 1.0)

        print(f"  Downloading GloVe-200 queries ({N_QUERIES}) ...", flush=True)
        ds_queries = load_dataset(HF_DATASET, "test", split="test", streaming=True)
        qraw = []
        for i, row in enumerate(ds_queries):
            if i >= N_QUERIES:
                break
            qraw.append(row["emb"])
        qvecs = np.array(qraw, dtype=np.float32)
        norms_q = np.linalg.norm(qvecs, axis=1, keepdims=True)
        qvecs /= np.where(norms_q > 0, norms_q, 1.0)

        np.save(ckpt_vecs,  vecs)
        np.save(ckpt_qvecs, qvecs)
        print("  Saved.", flush=True)

    if os.path.exists(ckpt_truth):
        print("  Loading cached GloVe-200 ground truth ...", flush=True)
        true_top1 = np.load(ckpt_truth)
    else:
        print(f"  Computing GloVe-200 brute-force ground truth ({N_QUERIES} queries) ...", flush=True)
        t0   = time.perf_counter()
        rows = []
        for i in range(0, N_QUERIES, 200):
            scores = qvecs[i : i + 200] @ vecs.T
            rows.append(np.argmax(scores, axis=1))
        true_top1 = np.concatenate(rows)
        print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
        np.save(ckpt_truth, true_top1)

    return vecs, qvecs, true_top1, N_DATA, N_QUERIES


# ── DBpedia data loading ──────────────────────────────────────────────────────

def load_dbpedia(dim):
    """Load DBpedia OpenAI3 embeddings for the given dimension (1536 or 3072)."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    N_DATA, N_QUERIES = 100_000, 1_000
    tag = f"dbpedia{dim}"

    ckpt_vecs  = os.path.join(CKPT_DIR, f"{tag}_{N_DATA}_vecs.npy")
    ckpt_qvecs = os.path.join(CKPT_DIR, f"{tag}_{N_QUERIES}_qvecs.npy")
    ckpt_truth = os.path.join(CKPT_DIR, f"{tag}_{N_DATA}_truth_top1.npy")

    HF_DATASET = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    VEC_FIELD  = f"text-embedding-3-large-{dim}-embedding"

    if os.path.exists(ckpt_vecs) and os.path.exists(ckpt_qvecs):
        print(f"  Loading cached DBpedia-{dim} vectors ...", flush=True)
        vecs  = np.load(ckpt_vecs)
        qvecs = np.load(ckpt_qvecs)
    else:
        from datasets import load_dataset
        print(f"  Downloading DBpedia d={dim} ({N_DATA+N_QUERIES:,} rows) from HuggingFace ...", flush=True)
        ds  = load_dataset(HF_DATASET, split="train", streaming=True)
        raw = []
        for i, row in enumerate(ds):
            if i >= N_DATA + N_QUERIES:
                break
            raw.append(row[VEC_FIELD])
            if (i + 1) % 10_000 == 0:
                print(f"    {i+1:>7,} / {N_DATA+N_QUERIES:,}", flush=True)
        all_vecs = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        all_vecs /= np.where(norms > 0, norms, 1.0)
        vecs  = all_vecs[N_QUERIES:]
        qvecs = all_vecs[:N_QUERIES]
        np.save(ckpt_vecs,  vecs)
        np.save(ckpt_qvecs, qvecs)
        print("  Saved.", flush=True)

    if os.path.exists(ckpt_truth):
        print(f"  Loading cached DBpedia-{dim} ground truth ...", flush=True)
        true_top1 = np.load(ckpt_truth)
    else:
        print(f"  Computing DBpedia-{dim} brute-force ground truth ({N_QUERIES} queries) ...", flush=True)
        t0   = time.perf_counter()
        rows = []
        for i in range(0, N_QUERIES, 50):
            scores = qvecs[i : i + 50] @ vecs.T
            rows.append(np.argmax(scores, axis=1))
        true_top1 = np.concatenate(rows)
        print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
        np.save(ckpt_truth, true_top1)

    return vecs, qvecs, true_top1, N_DATA, N_QUERIES


# ── TQDB benchmark ────────────────────────────────────────────────────────────

def run_tqdb(vecs, qvecs, true_top1, bits, n_queries):
    import tqdb

    N, DIM = vecs.shape
    ids    = [str(i) for i in range(N)]
    max_k  = K_VALUES[-1]

    with tempfile.TemporaryDirectory(prefix="paper_bench_") as tmp:
        t0 = time.perf_counter()
        db = tqdb.Database.open(tmp, dimension=DIM, bits=bits, metric="ip", rerank=False)
        for start in range(0, N, 2000):
            db.insert_batch(ids[start : start + 2000], vecs[start : start + 2000])
        db.flush()
        ingest_s = time.perf_counter() - t0

        hits = {k: 0 for k in K_VALUES}
        lats = []
        for qi in range(n_queries):
            t1 = time.perf_counter()
            results = db.search(qvecs[qi], top_k=max_k, _use_ann=False)
            lats.append((time.perf_counter() - t1) * 1000)
            result_ids = [r["id"] for r in results]
            true_id    = ids[true_top1[qi]]
            for k in K_VALUES:
                if true_id in result_ids[:k]:
                    hits[k] += 1

        recall = {k: round(hits[k] / n_queries, 4) for k in K_VALUES}
        lats_s = sorted(lats)
        return {
            "bits":     bits,
            "n":        N,
            "dim":      DIM,
            "queries":  n_queries,
            "ingest_s": round(ingest_s, 3),
            "p50_ms":   round(lats_s[int(n_queries * 0.50)], 2),
            "recall":   recall,
        }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_dataset_results(ds_label, results):
    paper = PAPER_RECALL.get(ds_label, {})
    W = 36
    fig_map = {"glove-200": "a", "dbpedia-1536": "b", "dbpedia-3072": "c"}
    print(f"\n{'='*70}")
    print(f"  {ds_label.upper()}  (n={results[0]['n']:,}, d={results[0]['dim']}, q={results[0]['queries']})")
    print(f"  Paper values approximate — read from Figure 5{fig_map.get(ds_label, '')}")
    print(f"{'='*70}")
    header = f"  {'Method':<{W}}" + "".join(f" k={k:<5}" for k in K_VALUES)
    print(header)
    print("  " + "-" * (len(header)-2))
    for bits in BITS_LIST:
        p = paper.get(bits, {})
        if p:
            label = f"TurboQuant {bits}-bit (paper Fig.5)"
            row   = f"  {label:<{W}}" + "".join(f" {p[k]:.3f} " for k in K_VALUES)
            print(row)
    print()
    for r in results:
        label = f"TQDB b={r['bits']} brute  [{r['ingest_s']}s ingest, p50={r['p50_ms']}ms]"
        row   = f"  {label:<{W}}" + "".join(f" {r['recall'][k]:.3f} " for k in K_VALUES)
        print(row)
        if r['bits'] in paper:
            diffs = {k: r['recall'][k] - paper[r['bits']][k] for k in K_VALUES}
            drow  = f"  {'  diff':<{W}}" + "".join(f" {diffs[k]:+.3f}" for k in K_VALUES)
            print(drow)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        choices=["glove", "dbpedia1536", "dbpedia3072"],
                        default=["glove", "dbpedia1536", "dbpedia3072"])
    args = parser.parse_args()

    print("=" * 70)
    print("  Paper recall benchmark — arXiv:2504.19874, Section 4.4")
    print(f"  Datasets: {args.datasets}")
    print(f"  k values: {K_VALUES} | bits: {BITS_LIST} | method: brute-force")
    print("=" * 70)

    all_results = {}

    for ds in args.datasets:
        print(f"\n── Loading {ds} ──────────────────────────────────────────────", flush=True)
        if ds == "glove":
            vecs, qvecs, true_top1, n_data, n_queries = load_glove()
            ds_label = "glove-200"
        elif ds == "dbpedia1536":
            vecs, qvecs, true_top1, n_data, n_queries = load_dbpedia(1536)
            ds_label = "dbpedia-1536"
        else:
            vecs, qvecs, true_top1, n_data, n_queries = load_dbpedia(3072)
            ds_label = "dbpedia-3072"

        print(f"  Data: {vecs.shape}, Queries: {qvecs.shape}", flush=True)

        ds_results = []
        for bits in BITS_LIST:
            print(f"  Running TQDB b={bits} ...", flush=True)
            r = run_tqdb(vecs, qvecs, true_top1, bits, n_queries)
            ds_results.append(r)
            print(f"    b={bits}: ingest={r['ingest_s']}s  p50={r['p50_ms']}ms  "
                  f"Recall@1={r['recall'][1]:.3f}  @4={r['recall'][4]:.3f}  @8={r['recall'][8]:.3f}")

        print_dataset_results(ds_label, ds_results)
        all_results[ds_label] = ds_results

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "paper_recall_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()

