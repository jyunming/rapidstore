"""
TurboQuantDB parameter sweep — LOCAL ONLY, not for CI.

Tests all combinations of:
  bits   : 2, 4
  rerank : False, True
  search : brute-force (_use_ann=False) | ANN (create_index + _use_ann=True)

8 configs × 3 paper datasets = 24 runs.

Datasets (same as paper):
  (a) GloVe-200    — d=200,  100k corpus, 10k queries
  (b) DBpedia-1536 — d=1536, 100k corpus, 1k  queries
  (c) DBpedia-3072 — d=3072, 100k corpus, 1k  queries

Paper reference values (Figure 5, bits=2/4 brute-force, rerank=False):
  GloVe-200    2-bit: R@1@1≈0.55   4-bit: R@1@1≈0.86
  DBpedia-1536 2-bit: R@1@1≈0.895  4-bit: R@1@1≈0.970
  DBpedia-3072 2-bit: R@1@1≈0.905  4-bit: R@1@1≈0.975

Usage:
    python benchmarks/tqdb_param_sweep.py
    python benchmarks/tqdb_param_sweep.py --datasets glove dbpedia1536
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

sys.stdout.reconfigure(encoding="utf-8")

K_VALUES   = [1, 2, 4, 8, 16, 32, 64]
CHUNK_SIZE = 2000
CACHE_DIR  = Path(__file__).parent / "_paper_bench_cache"
PROCESS    = psutil.Process(os.getpid())

# Paper recall values (Figure 5 — rerank=False, brute-force)
PAPER_RECALL: dict[str, dict[int, dict[int, float]]] = {
    "glove-200": {
        2: {1: 0.55,  4: 0.83,  8: 0.91},
        4: {1: 0.86,  4: 0.99,  8: 1.00},
    },
    "dbpedia-1536": {
        2: {1: 0.895, 4: 0.995, 8: 1.000},
        4: {1: 0.970, 4: 1.000, 8: 1.000},
    },
    "dbpedia-3072": {
        2: {1: 0.905, 4: 0.995, 8: 1.000},
        4: {1: 0.975, 4: 1.000, 8: 1.000},
    },
}

# All 8 parameter configurations tested
CONFIGS = [
    {"bits": 2, "rerank": False, "ann": False},
    {"bits": 2, "rerank": False, "ann": True},
    {"bits": 2, "rerank": True,  "ann": False},
    {"bits": 2, "rerank": True,  "ann": True},
    {"bits": 4, "rerank": False, "ann": False},
    {"bits": 4, "rerank": False, "ann": True},
    {"bits": 4, "rerank": True,  "ann": False},
    {"bits": 4, "rerank": True,  "ann": True},
]


# ---------------------------------------------------------------------------
# Dataset loaders (identical to store_comparison_bench.py)
# ---------------------------------------------------------------------------

def load_glove() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    ckpt_vecs  = CACHE_DIR / "glove200_100000_vecs.npy"
    ckpt_qvecs = CACHE_DIR / "glove200_10000_qvecs.npy"
    ckpt_truth = CACHE_DIR / "glove200_100000_truth_top1.npy"
    N_DATA, N_QUERIES = 100_000, 10_000

    if ckpt_vecs.exists() and ckpt_qvecs.exists():
        print("  Loading cached GloVe-200 vectors ...", flush=True)
        vecs  = np.load(ckpt_vecs)
        qvecs = np.load(ckpt_qvecs)
    else:
        from datasets import load_dataset
        print("  Downloading GloVe-200 ...", flush=True)
        ds = load_dataset("open-vdb/glove-200-angular", "train", split="train", streaming=True)
        raw = [row["emb"] for i, row in enumerate(ds) if i < N_DATA]
        vecs = np.array(raw, dtype=np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        ds_q = load_dataset("open-vdb/glove-200-angular", "test", split="test", streaming=True)
        qraw = [row["emb"] for i, row in enumerate(ds_q) if i < N_QUERIES]
        qvecs = np.array(qraw, dtype=np.float32)
        qvecs /= np.maximum(np.linalg.norm(qvecs, axis=1, keepdims=True), 1e-9)
        np.save(ckpt_vecs, vecs); np.save(ckpt_qvecs, qvecs)

    if ckpt_truth.exists():
        true_top1 = np.load(ckpt_truth)
    else:
        print("  Computing ground truth ...", flush=True)
        rows = [np.argmax(qvecs[i:i+200] @ vecs.T, axis=1) for i in range(0, N_QUERIES, 200)]
        true_top1 = np.concatenate(rows)
        np.save(ckpt_truth, true_top1)

    print(f"  GloVe-200: corpus={vecs.shape}, queries={qvecs.shape}", flush=True)
    return vecs, qvecs, true_top1


def load_dbpedia(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"dbpedia{dim}"
    N_DATA, N_QUERIES = 100_000, 1_000
    ckpt_vecs  = CACHE_DIR / f"{tag}_{N_DATA}_vecs.npy"
    ckpt_qvecs = CACHE_DIR / f"{tag}_{N_QUERIES}_qvecs.npy"
    ckpt_truth = CACHE_DIR / f"{tag}_{N_DATA}_truth_top1.npy"

    if ckpt_vecs.exists() and ckpt_qvecs.exists():
        print(f"  Loading cached DBpedia-{dim} vectors ...", flush=True)
        vecs  = np.load(ckpt_vecs)
        qvecs = np.load(ckpt_qvecs)
    else:
        from datasets import load_dataset
        name = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
        split_name = "train" if dim == 3072 else "1536-train"
        print(f"  Downloading DBpedia-{dim} ...", flush=True)
        ds = load_dataset(name, split=split_name, streaming=True)
        key = "text-embedding-3-large-3072" if dim == 3072 else "text-embedding-3-large-1536"
        raw, qraw = [], []
        for i, row in enumerate(ds):
            if i < N_DATA:
                raw.append(row[key])
            elif i < N_DATA + N_QUERIES:
                qraw.append(row[key])
            else:
                break
        vecs  = np.array(raw,  dtype=np.float32)
        qvecs = np.array(qraw, dtype=np.float32)
        np.save(ckpt_vecs, vecs); np.save(ckpt_qvecs, qvecs)

    if ckpt_truth.exists():
        true_top1 = np.load(ckpt_truth)
    else:
        print("  Computing ground truth ...", flush=True)
        rows = [np.argmax(qvecs[i:i+100] @ vecs.T, axis=1) for i in range(0, N_QUERIES, 100)]
        true_top1 = np.concatenate(rows)
        np.save(ckpt_truth, true_top1)

    print(f"  DBpedia-{dim}: corpus={vecs.shape}, queries={qvecs.shape}", flush=True)
    return vecs, qvecs, true_top1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1 << 20)


def _rss_mb() -> float:
    return PROCESS.memory_info().rss / (1 << 20)


def _recalls(results_list: list[list[str]], true_top1: np.ndarray) -> dict[int, float]:
    out = {}
    for k in K_VALUES:
        hits = sum(
            1 for r, idx in zip(results_list, true_top1)
            if f"vec_{int(idx)}" in r[:k]
        )
        out[k] = hits / len(true_top1)
    return out


# ---------------------------------------------------------------------------
# TQDB runner — parameterised
# ---------------------------------------------------------------------------

def run_tqdb(
    corpus: np.ndarray,
    qs: np.ndarray,
    true_top1: np.ndarray,
    bits: int,
    rerank: bool,
    ann: bool,
) -> dict:
    label = f"b={bits} rerank={'T' if rerank else 'F'} {'ANN' if ann else 'brute'}"
    try:
        import tqdb as tq
    except ImportError:
        return {"name": label, "error": "tqdb not installed — run: maturin develop --release"}

    n, d = corpus.shape
    ids  = [f"vec_{i}" for i in range(n)]
    tmpdir = tempfile.mkdtemp()
    try:
        rss_before = _rss_mb()
        db = tq.Database.open(tmpdir, dimension=d, bits=bits, rerank=rerank, metric="ip")

        # --- ingest ---
        t0 = time.perf_counter()
        for start in range(0, n, CHUNK_SIZE):
            db.insert_batch(ids[start:start + CHUNK_SIZE],
                            corpus[start:start + CHUNK_SIZE])
        db.flush()
        ingest_wall = time.perf_counter() - t0

        # --- index build (ANN only) ---
        index_s: float | None = None
        if ann:
            t_idx = time.perf_counter()
            db.create_index(max_degree=16, search_list_size=64, alpha=1.2)
            index_s = time.perf_counter() - t_idx

        disk_mb   = _dir_mb(tmpdir)
        rss_after = _rss_mb()

        # --- search ---
        max_k = max(K_VALUES)
        lats: list[float] = []
        all_returned: list[list[str]] = []
        for q in qs:
            t1 = time.perf_counter()
            res = db.search(q, top_k=max_k, _use_ann=ann)
            lats.append((time.perf_counter() - t1) * 1000)
            all_returned.append([r["id"] for r in res])

        recalls = _recalls(all_returned, true_top1)
        db.close()
        return {
            "name":           label,
            "bits":           bits,
            "rerank":         rerank,
            "ann":            ann,
            "n":              n,
            "ingest_s":       ingest_wall,
            "throughput_vps": n / ingest_wall,
            "index_s":        index_s,
            "disk_mb":        disk_mb,
            "ram_delta_mb":   rss_after - rss_before,
            "p50_ms":         float(np.percentile(lats, 50)),
            "p95_ms":         float(np.percentile(lats, 95)),
            "recall_1_at_1":  recalls[1],
            "recall_1_at_2":  recalls[2],
            "recall_1_at_4":  recalls[4],
            "recall_1_at_8":  recalls[8],
            "recall_1_at_16": recalls[16],
            "recall_1_at_32": recalls[32],
            "recall_1_at_64": recalls[64],
        }
    except Exception as exc:
        return {"name": label, "error": str(exc)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

COLS = [
    ("Config",       "name",           "<", 22),
    ("Thruput vps",  "throughput_vps", ">", 12),
    ("Ingest s",     "ingest_s",       ">", 9),
    ("Index s",      "index_s",        ">", 8),
    ("Disk MB",      "disk_mb",        ">", 8),
    ("ΔRSS MB",      "ram_delta_mb",   ">", 8),
    ("p50 ms",       "p50_ms",         ">", 8),
    ("p95 ms",       "p95_ms",         ">", 8),
    ("R@1@1",        "recall_1_at_1",  ">", 6),
    ("R@1@2",        "recall_1_at_2",  ">", 6),
    ("R@1@4",        "recall_1_at_4",  ">", 6),
    ("R@1@8",        "recall_1_at_8",  ">", 6),
    ("R@1@16",       "recall_1_at_16", ">", 7),
    ("R@1@32",       "recall_1_at_32", ">", 7),
    ("R@1@64",       "recall_1_at_64", ">", 7),
]

FMT: dict = {
    "name":           lambda v: str(v),
    "throughput_vps": lambda v: f"{v:,.0f}" if v else "—",
    "ingest_s":       lambda v: f"{v:.1f}s"  if v else "—",
    "index_s":        lambda v: f"{v:.1f}s"  if v is not None else "—",
    "disk_mb":        lambda v: f"{v:.1f}"   if v and v > 0 else "—",
    "ram_delta_mb":   lambda v: f"{v:.0f}"   if v else "—",
    "p50_ms":         lambda v: f"{v:.2f}"   if v else "—",
    "p95_ms":         lambda v: f"{v:.2f}"   if v else "—",
    "recall_1_at_1":  lambda v: f"{v:.3f}",
    "recall_1_at_2":  lambda v: f"{v:.3f}",
    "recall_1_at_4":  lambda v: f"{v:.3f}",
    "recall_1_at_8":  lambda v: f"{v:.3f}",
    "recall_1_at_16": lambda v: f"{v:.3f}",
    "recall_1_at_32": lambda v: f"{v:.3f}",
    "recall_1_at_64": lambda v: f"{v:.3f}",
}

SEP = "  "


def _hdr()  -> str: return SEP.join(f"{lbl:{a}{w}}" for lbl, _, a, w in COLS)
def _div()  -> str: return SEP.join("─" * w      for _, _, _, w in COLS)
def _row(r: dict) -> str:
    return SEP.join(
        f"{FMT[key](r.get(key)):{a}{w}}"
        for _, key, a, w in COLS
    )


def _paper_ref_row(ds_label: str, bits: int) -> str | None:
    pr = PAPER_RECALL.get(ds_label, {}).get(bits, {})
    if not pr:
        return None
    name = f"  ↳ paper b={bits} (brute,no-rr)"
    parts = [f"{name:{COLS[0][3]}}"]
    for _, key, align, w in COLS[1:]:
        if key in ("recall_1_at_1", "recall_1_at_4", "recall_1_at_8"):
            k = int(key.split("_")[-1])
            parts.append(f"{pr[k]:.3f}".rjust(w))
        else:
            parts.append(" " * w)
    return SEP.join(parts)


def print_section_results(ds_label: str, results: list[dict]) -> None:
    ok  = [r for r in results if "error" not in r]
    n   = ok[0]["n"] if ok else 0
    hdr = _hdr()
    width = len(hdr)

    print()
    print("=" * width)
    print(f"  {ds_label.upper()}  —  n={n:,}  |  metric=ip  |  TQDB parameter sweep (8 configs)")
    print("=" * width)
    print(hdr)
    print(_div())

    prev_bits = None
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<24}  ERROR: {r['error']}")
            continue

        # print a blank separator between bits=2 and bits=4 groups
        if prev_bits is not None and r.get("bits") != prev_bits:
            print()
        prev_bits = r.get("bits")

        print(_row(r))

        # print paper reference after the last brute/no-rerank row for each bits value
        if not r.get("ann") and not r.get("rerank"):
            prow = _paper_ref_row(ds_label, r["bits"])
            if prow:
                print(prow)

    print(_div())

    for r in results:
        if "error" in r:
            print(f"  ✗ {r['name']}: {r['error']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_DATASETS = ["glove", "dbpedia1536", "dbpedia3072"]
DATASET_LOADERS = {
    "glove":       load_glove,
    "dbpedia1536": lambda: load_dbpedia(1536),
    "dbpedia3072": lambda: load_dbpedia(3072),
}
DATASET_LABELS = {
    "glove":       "glove-200",
    "dbpedia1536": "dbpedia-1536",
    "dbpedia3072": "dbpedia-3072",
}


def run_dataset(ds: str) -> None:
    ds_label = DATASET_LABELS[ds]
    sep_line = "━" * 72
    print(f"\n{sep_line}")
    print(f"  Loading {ds_label} ...")
    print(sep_line)
    corpus, qvecs, true_top1 = DATASET_LOADERS[ds]()
    n, d = corpus.shape

    results: list[dict] = []
    for cfg in CONFIGS:
        bits, rerank, ann = cfg["bits"], cfg["rerank"], cfg["ann"]
        mode = "ANN   " if ann else "brute "
        print(
            f"\n── b={bits} rerank={'T' if rerank else 'F'} {mode}"
            f" ({ds_label}, n={n:,}, d={d}, q={len(qvecs)}) ──",
            flush=True,
        )
        r = run_tqdb(corpus, qvecs, true_top1, bits=bits, rerank=rerank, ann=ann)
        if "error" in r:
            print(f"  SKIP: {r['error']}")
        else:
            idx_str = f"  index: {r['index_s']:.1f}s" if r["index_s"] is not None else ""
            print(
                f"  ingest: {r['ingest_s']:.1f}s  throughput: {r['throughput_vps']:,.0f} vps"
                f"{idx_str}  R@1@8: {r['recall_1_at_8']:.3f}"
                f"  p50: {r['p50_ms']:.2f}ms  disk: {r['disk_mb']:.1f}MB"
                f"  ΔRSS: {r['ram_delta_mb']:.0f}MB",
                flush=True,
            )
        results.append(r)

    print_section_results(ds_label, results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TQDB parameter sweep (local only)")
    p.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
        help="datasets to test (default: all three paper datasets)",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    datasets = args.datasets

    print("=" * 72)
    print("  TurboQuantDB Parameter Sweep — arXiv:2504.19874 paper methodology")
    print(f"  Datasets : {datasets}")
    print(f"  Configs  : {len(CONFIGS)} (bits∈{{2,4}} × rerank∈{{F,T}} × search∈{{brute,ANN}})")
    print(f"  k values : {K_VALUES}  |  metric: inner product")
    print("=" * 72)

    for ds in datasets:
        run_dataset(ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
