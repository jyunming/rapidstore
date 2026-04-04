"""
Vector store comparison benchmark — LOCAL ONLY, not for CI.

Compares TurboQuantDB against ChromaDB (HNSW), LanceDB (IVF-PQ) and
Qdrant (HNSW in-memory) using the same methodology as the paper recall
benchmark (arXiv:2504.19874):

  - Recall@1@k  (is the true nearest neighbour in the returned top-k?)
  - GloVe-200 cached data when available; synthetic d=200 otherwise
  - insert_batch in chunks of 2000 for TurboQuantDB
  - k values: 1, 4, 8
  - Metrics: recall@1@k, p50/p95 latency (ms), ingest throughput (vps),
             disk usage (MB), peak RSS change (MB)

This script is intentionally gitignored (benchmarks/ is excluded from git).
It requires the optional comparison dependencies:
    pip install chromadb lancedb qdrant-client

Usage:
    python benchmarks/store_comparison_bench.py
    python benchmarks/store_comparison_bench.py --n 25000 --queries 200
    python benchmarks/store_comparison_bench.py --engines tqdb chromadb qdrant
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

K_VALUES   = [1, 4, 8]
CHUNK_SIZE = 2000
CACHE_DIR  = Path(__file__).parent / "_paper_bench_cache"

PROCESS = psutil.Process(os.getpid())


# ---------------------------------------------------------------------------
# Data loading (same as paper_recall_bench.py + precommit_perf_check.py)
# ---------------------------------------------------------------------------

def load_data(n: int, d: int, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vecs_exact = CACHE_DIR / f"glove200_{n}_vecs.npy"
    vecs_100k  = CACHE_DIR / "glove200_100000_vecs.npy"
    qvecs_path = CACHE_DIR / "glove200_10000_qvecs.npy"
    vecs_path  = vecs_exact if vecs_exact.exists() else (vecs_100k if vecs_100k.exists() else None)
    if vecs_path and qvecs_path.exists():
        corpus = np.load(str(vecs_path))[:n].astype(np.float32)
        all_q  = np.load(str(qvecs_path)).astype(np.float32)
        rng    = np.random.default_rng(seed)
        idx    = rng.choice(len(all_q), n_queries, replace=False)
        qs     = all_q[idx]
        print(f"Data: GloVe-200 ({len(corpus):,} corpus, {len(qs)} queries, d={corpus.shape[1]})")
    else:
        rng    = np.random.default_rng(seed)
        corpus = rng.standard_normal((n, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        qs     = rng.standard_normal((n_queries, d)).astype(np.float32)
        qs     /= np.linalg.norm(qs, axis=1, keepdims=True)
        print(f"Data: synthetic d={d} ({n:,} corpus, {n_queries} queries)")
    ids = [f"vec_{i}" for i in range(len(corpus))]
    return corpus, qs, ids


def true_top1s(corpus: np.ndarray, qs: np.ndarray) -> list[str]:
    return [f"vec_{int(np.argmax(corpus @ q))}" for q in qs]


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


def _record_latencies(fn, qs, k):
    """Run fn(q, k) for each query, return latencies_ms list."""
    lats = []
    for q in qs:
        t0 = time.perf_counter()
        fn(q, k)
        lats.append((time.perf_counter() - t0) * 1000)
    return lats


def _recalls(results_list: list[list[str]], top1s: list[str]) -> dict[int, float]:
    out = {}
    for k in K_VALUES:
        hits = sum(1 for r, t in zip(results_list, top1s) if t in r[:k])
        out[k] = hits / len(top1s)
    return out


# ---------------------------------------------------------------------------
# Per-engine runners
# ---------------------------------------------------------------------------

def run_tqdb(corpus, qs, top1s, bits=4, label="TurboQuantDB") -> dict:
    try:
        import tqdb as tq
    except ImportError:
        return {"name": label, "error": "tqdb not installed — run: maturin develop --release"}

    n, d = corpus.shape
    ids  = [f"vec_{i}" for i in range(n)]
    tmpdir = tempfile.mkdtemp()
    try:
        rss_before = _rss_mb()
        db = tq.Database.open(tmpdir, dimension=d, bits=bits, rerank=False, metric="ip")
        t0 = time.perf_counter()
        for start in range(0, n, CHUNK_SIZE):
            db.insert_batch(ids[start:start + CHUNK_SIZE],
                            corpus[start:start + CHUNK_SIZE])
        db.flush()
        ingest_wall = time.perf_counter() - t0
        disk_mb     = _dir_mb(tmpdir)
        rss_after   = _rss_mb()

        max_k = max(K_VALUES)
        lats  = []
        all_returned: list[list[str]] = []
        for q in qs:
            t1 = time.perf_counter()
            res = db.search(q, top_k=max_k)
            lats.append((time.perf_counter() - t1) * 1000)
            all_returned.append([r["id"] for r in res])

        recalls = _recalls(all_returned, top1s)
        db.close()
        return {
            "name":            label,
            "n":               n,
            "ingest_s":        ingest_wall,
            "throughput_vps":  n / ingest_wall,
            "disk_mb":         disk_mb,
            "ram_delta_mb":    rss_after - rss_before,
            "p50_ms":          float(np.percentile(lats, 50)),
            "p95_ms":          float(np.percentile(lats, 95)),
            "recall_1_at_1":   recalls[1],
            "recall_1_at_4":   recalls[4],
            "recall_1_at_8":   recalls[8],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()


def run_chromadb(corpus, qs, top1s) -> dict:
    try:
        import chromadb
    except ImportError:
        return {"name": "ChromaDB (HNSW)", "error": "pip install chromadb"}

    n, d = corpus.shape
    rss_before = _rss_mb()
    client = chromadb.Client()
    coll = client.create_collection("bench", metadata={"hnsw:space": "ip"})

    ids = [f"vec_{i}" for i in range(n)]
    t0  = time.perf_counter()
    for start in range(0, n, CHUNK_SIZE):
        coll.add(ids=ids[start:start + CHUNK_SIZE],
                 embeddings=corpus[start:start + CHUNK_SIZE].tolist())
    ingest_wall = time.perf_counter() - t0
    rss_after   = _rss_mb()

    max_k = max(K_VALUES)
    lats: list[float] = []
    all_returned: list[list[str]] = []
    for q in qs:
        t1 = time.perf_counter()
        res = coll.query(query_embeddings=[q.tolist()], n_results=max_k)
        lats.append((time.perf_counter() - t1) * 1000)
        all_returned.append(res["ids"][0])

    recalls = _recalls(all_returned, top1s)
    client.delete_collection("bench")
    gc.collect()
    return {
        "name":           "ChromaDB (HNSW)",
        "n":              n,
        "ingest_s":       ingest_wall,
        "throughput_vps": n / ingest_wall,
        "disk_mb":        0.0,  # in-memory
        "ram_delta_mb":   rss_after - rss_before,
        "p50_ms":         float(np.percentile(lats, 50)),
        "p95_ms":         float(np.percentile(lats, 95)),
        "recall_1_at_1":  recalls[1],
        "recall_1_at_4":  recalls[4],
        "recall_1_at_8":  recalls[8],
    }


def run_lancedb(corpus, qs, top1s) -> dict:
    try:
        import lancedb
        import pyarrow as pa
    except ImportError:
        return {"name": "LanceDB (IVF-PQ)", "error": "pip install lancedb pyarrow"}

    n, d   = corpus.shape
    tmpdir = tempfile.mkdtemp()
    try:
        rss_before = _rss_mb()
        db  = lancedb.connect(tmpdir)
        ids = [f"vec_{i}" for i in range(n)]
        schema = pa.schema([
            pa.field("id",     pa.string()),
            pa.field("vector", pa.list_(pa.float32(), d)),
        ])
        tbl = None
        t0  = time.perf_counter()
        for start in range(0, n, CHUNK_SIZE):
            chunk_vecs = corpus[start:start + CHUNK_SIZE]
            pa_chunk = pa.table({
                "id":     pa.array(ids[start:start + CHUNK_SIZE]),
                "vector": pa.array(chunk_vecs.tolist(), type=pa.list_(pa.float32(), d)),
            })
            if tbl is None:
                tbl = db.create_table("bench", data=pa_chunk)
            else:
                tbl.add(pa_chunk)
        # Build IVF-PQ index — num_sub_vectors must divide d
        n_sub = next(s for s in range(min(d // 4, 16), 0, -1) if d % s == 0)
        tbl.create_index(metric="dot", index_type="IVF_PQ",
                         num_partitions=min(256, max(1, n // 39)),
                         num_sub_vectors=n_sub)
        ingest_wall = time.perf_counter() - t0
        disk_mb     = _dir_mb(tmpdir)
        rss_after   = _rss_mb()

        max_k = max(K_VALUES)
        lats: list[float] = []
        all_returned: list[list[str]] = []
        for q in qs:
            t1  = time.perf_counter()
            res = tbl.search(q.tolist()).metric("dot").limit(max_k).to_list()
            lats.append((time.perf_counter() - t1) * 1000)
            all_returned.append([r["id"] for r in res])

        recalls = _recalls(all_returned, top1s)
        return {
            "name":           "LanceDB (IVF-PQ)",
            "n":              n,
            "ingest_s":       ingest_wall,
            "throughput_vps": n / ingest_wall,
            "disk_mb":        disk_mb,
            "ram_delta_mb":   rss_after - rss_before,
            "p50_ms":         float(np.percentile(lats, 50)),
            "p95_ms":         float(np.percentile(lats, 95)),
            "recall_1_at_1":  recalls[1],
            "recall_1_at_4":  recalls[4],
            "recall_1_at_8":  recalls[8],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()


def run_qdrant(corpus, qs, top1s) -> dict:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        return {"name": "Qdrant (HNSW)", "error": "pip install qdrant-client"}

    n, d = corpus.shape
    rss_before = _rss_mb()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=d, distance=Distance.DOT),
    )
    ids = list(range(n))
    t0  = time.perf_counter()
    for start in range(0, n, CHUNK_SIZE):
        points = [
            PointStruct(id=i, vector=corpus[i].tolist())
            for i in range(start, min(start + CHUNK_SIZE, n))
        ]
        client.upsert(collection_name="bench", points=points)
    ingest_wall = time.perf_counter() - t0
    rss_after   = _rss_mb()

    max_k = max(K_VALUES)
    lats: list[float] = []
    all_returned: list[list[str]] = []
    for q in qs:
        t1  = time.perf_counter()
        res = client.query_points(
            collection_name="bench",
            query=q.tolist(),
            limit=max_k,
            with_payload=False,
        ).points
        lats.append((time.perf_counter() - t1) * 1000)
        all_returned.append([f"vec_{hit.id}" for hit in res])

    recalls = _recalls(all_returned, top1s)
    client.delete_collection("bench")
    gc.collect()
    return {
        "name":           "Qdrant (HNSW)",
        "n":              n,
        "ingest_s":       ingest_wall,
        "throughput_vps": n / ingest_wall,
        "disk_mb":        0.0,  # in-memory
        "ram_delta_mb":   rss_after - rss_before,
        "p50_ms":         float(np.percentile(lats, 50)),
        "p95_ms":         float(np.percentile(lats, 95)),
        "recall_1_at_1":  recalls[1],
        "recall_1_at_4":  recalls[4],
        "recall_1_at_8":  recalls[8],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _bar(v: float, lo: float, hi: float, width: int = 12) -> str:
    if hi == lo:
        return "█" * (width // 2)
    filled = int((v - lo) / (hi - lo) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def print_results(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("No successful results to display.")
        return

    header_cols = [
        ("Engine",         "name",           "<", 22),
        ("n",              "n",              ">", 7),
        ("Throughput/s",   "throughput_vps", ">", 13),
        ("Ingest(s)",      "ingest_s",        ">", 10),
        ("Disk MB",        "disk_mb",         ">", 8),
        ("ΔRSS MB",        "ram_delta_mb",    ">", 9),
        ("p50 ms",         "p50_ms",          ">", 8),
        ("p95 ms",         "p95_ms",          ">", 8),
        ("R@1@1",          "recall_1_at_1",   ">", 7),
        ("R@1@4",          "recall_1_at_4",   ">", 7),
        ("R@1@8",          "recall_1_at_8",   ">", 7),
    ]

    fmt_fns = {
        "name":           lambda v: v,
        "n":              lambda v: f"{int(v):,}",
        "throughput_vps": lambda v: f"{v:,.0f}",
        "ingest_s":       lambda v: f"{v:.1f}s",
        "disk_mb":        lambda v: f"{v:.1f}" if v > 0 else "—",
        "ram_delta_mb":   lambda v: f"{v:.0f}",
        "p50_ms":         lambda v: f"{v:.2f}",
        "p95_ms":         lambda v: f"{v:.2f}",
        "recall_1_at_1":  lambda v: f"{v:.3f}",
        "recall_1_at_4":  lambda v: f"{v:.3f}",
        "recall_1_at_8":  lambda v: f"{v:.3f}",
    }

    # ── header ──────────────────────────────────────────────────────────────
    sep = "  "
    header = sep.join(f"{label:{align}{w}}" for label, _, align, w in header_cols)
    divider = sep.join("─" * w for _, _, _, w in header_cols)
    print()
    print("=" * len(header))
    print("Store Comparison  (Recall@1@k — paper methodology, local only)")
    print("=" * len(header))
    print(header)
    print(divider)

    for r in results:
        if "error" in r:
            print(f"  {r['name']:<20}  ERROR: {r['error']}")
            continue
        row = sep.join(
            f"{fmt_fns[key](r.get(key, 0)):{align}{w}}"
            for _, key, align, w in header_cols
        )
        print(row)

    print(divider)

    # errors at end
    errs = [r for r in results if "error" in r]
    if errs:
        print()
        for r in errs:
            print(f"  ✗ {r['name']}: {r['error']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_ENGINES = ["tqdb", "chromadb", "lancedb", "qdrant"]

ENGINE_MAP = {
    "tqdb":     lambda corpus, qs, top1s, **kw: run_tqdb(corpus, qs, top1s, **kw),
    "chromadb": lambda corpus, qs, top1s, **_:  run_chromadb(corpus, qs, top1s),
    "lancedb":  lambda corpus, qs, top1s, **_:  run_lancedb(corpus, qs, top1s),
    "qdrant":   lambda corpus, qs, top1s, **_:  run_qdrant(corpus, qs, top1s),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vector store comparison benchmark (local only)")
    p.add_argument("--n",       type=int, default=10000,  help="corpus size (default 10000)")
    p.add_argument("--d",       type=int, default=200,    help="dimension if no GloVe cache")
    p.add_argument("--bits",    type=int, default=4,      help="TurboQuantDB bits (2 or 4)")
    p.add_argument("--queries", type=int, default=100,    help="number of query vectors")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument(
        "--engines", nargs="+", choices=ALL_ENGINES, default=ALL_ENGINES,
        help="engines to benchmark (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus, qs, ids = load_data(args.n, args.d, args.queries, args.seed)
    n, d = corpus.shape

    print(f"Computing ground-truth top-1 for {len(qs)} queries ...", flush=True)
    gt = true_top1s(corpus, qs)

    results: list[dict] = []
    for eng in args.engines:
        print(f"\n── {eng} ──", flush=True)
        kw: dict = {}
        if eng == "tqdb":
            kw = {"bits": args.bits, "label": f"TurboQuantDB b={args.bits}"}
        r = ENGINE_MAP[eng](corpus, qs, gt, **kw)
        if "error" in r:
            print(f"  SKIP: {r['error']}")
        else:
            print(
                f"  ingest: {r['ingest_s']:.1f}s  "
                f"throughput: {r['throughput_vps']:,.0f} vps  "
                f"R@1@8: {r['recall_1_at_8']:.3f}  "
                f"p50: {r['p50_ms']:.2f}ms"
            )
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
