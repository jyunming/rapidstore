"""
_bench_rotation_compare.py — Rotation-based competitor benchmark.

Tests rotation/quantization competitors that have shipped training-free or
near-training-free quantization since Apr 2026:

  - LanceDB  IVF_RQ  (RaBitQ 1-bit, rotation-based, training-free)
  - Weaviate 1.33+   hnsw+RQ  (8-bit rotational, training-free; requires Docker)

Datasets: GloVe-200 (d=200) and DBpedia-1536 (d=1536) only.

Usage:
    python benchmarks/_bench_rotation_compare.py
    python benchmarks/_bench_rotation_compare.py --datasets glove
    python benchmarks/_bench_rotation_compare.py --no-weaviate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(__file__))
from bench_core import (
    K_VALUES,
    CpuRamSampler,
    compute_recalls,
    compute_mrr,
    disk_size_mb,
    load_glove,
    load_dbpedia,
)

sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR    = Path(__file__).parent
RESULTS_PATH = BENCH_DIR / "_bench_rotation_results.json"


# ── LanceDB IVF_RQ (RaBitQ 1-bit, rotation-based) ────────────────────────────

def run_lancedb_rabitq(vecs, qvecs, true_top1) -> dict | None:
    try:
        import lancedb
        import pyarrow as pa
    except ImportError:
        print("  lancedb / pyarrow not installed — skipping", flush=True)
        return None

    N, DIM = vecs.shape
    n_q    = len(qvecs)
    max_k  = K_VALUES[-1]
    id_fn  = lambda i: f"vec_{i}"

    with tempfile.TemporaryDirectory(prefix="ldb_rq_bench_") as tmp:
        sampler_ingest = CpuRamSampler()
        sampler_ingest.start()
        t0  = time.perf_counter()
        db  = lancedb.connect(tmp)
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
        ])
        tbl = db.create_table("bench", schema=schema)
        for start in range(0, N, 2000):
            end   = min(start + 2000, N)
            chunk = [{"id": id_fn(i), "vector": vecs[i].tolist()} for i in range(start, end)]
            tbl.add(chunk)
        # IVF_RQ: LanceDB's RaBitQ — 1-bit per dimension, cosine metric
        tbl.create_index(metric="cosine", index_type="IVF_RQ", num_bits=1)
        ingest_s = time.perf_counter() - t0
        sampler_ingest.stop()
        dm = disk_size_mb(tmp)

        sampler_query = CpuRamSampler()
        sampler_query.start()
        lats: list[float] = []
        all_returned: list[list[str]] = []
        for q in qvecs:
            t1  = time.perf_counter()
            res = tbl.search(q.tolist()).metric("cosine").limit(max_k).to_list()
            lats.append((time.perf_counter() - t1) * 1000)
            all_returned.append([r["id"] for r in res])
        sampler_query.stop()

    recalls = compute_recalls(all_returned, true_top1, id_fn=id_fn)
    mrr     = compute_mrr(all_returned, true_top1, id_fn=id_fn)
    lats_s  = sorted(lats)
    return _make_result("LanceDB (IVF_RQ/RaBitQ)", N, DIM, n_q, N / ingest_s,
                        ingest_s, dm, sampler_ingest, sampler_query, lats_s, recalls, mrr)


# ── Weaviate HNSW + RQ (Docker: weaviate:1.33+) ───────────────────────────────

def run_weaviate_rq(vecs, qvecs, true_top1,
                    host: str = "localhost", port: int = 8080) -> dict | None:
    try:
        import weaviate
        import weaviate.classes as wvc
        from weaviate.classes.query import MetadataQuery
    except ImportError:
        print("  weaviate-client not installed — skipping", flush=True)
        return None

    N, DIM = vecs.shape
    n_q    = len(qvecs)
    max_k  = K_VALUES[-1]

    def idx_to_uuid(i: int) -> str:
        return str(uuid.UUID(int=i + 1))

    id_fn = lambda i: f"vec_{i}"

    try:
        client = weaviate.connect_to_local(host=host, port=port,
                                           grpc_port=50051, skip_init_checks=True)
    except Exception as e:
        print(f"  Weaviate connect failed ({e}) — skipping", flush=True)
        return None

    wv_ver = "?"
    try:
        try:
            client.collections.delete("Bench")
        except Exception:
            pass

        sampler_ingest = CpuRamSampler()
        sampler_ingest.start()
        t0 = time.perf_counter()

        try:
            meta = client.get_meta()
            wv_ver = meta.get("version", "?")
        except Exception:
            pass

        # HNSW + 8-bit Rotational Quantization (default since Weaviate 1.33)
        collection = client.collections.create(
            "Bench",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
                quantizer=wvc.config.Configure.VectorIndex.Quantizer.rq(),
            ),
        )

        with collection.batch.fixed_size(batch_size=500) as batch:
            for i in range(N):
                batch.add_object(
                    properties={},
                    vector=vecs[i].tolist(),
                    uuid=idx_to_uuid(i),
                )
        ingest_s = time.perf_counter() - t0
        sampler_ingest.stop()

        sampler_query = CpuRamSampler()
        sampler_query.start()
        lats: list[float] = []
        all_returned: list[list[str]] = []
        for q in qvecs:
            t1  = time.perf_counter()
            res = collection.query.near_vector(
                near_vector=q.tolist(),
                limit=max_k,
                return_metadata=MetadataQuery(distance=True),
            )
            lats.append((time.perf_counter() - t1) * 1000)
            returned_ids = []
            for obj in res.objects:
                orig_i = uuid.UUID(str(obj.uuid)).int - 1
                returned_ids.append(id_fn(orig_i))
            all_returned.append(returned_ids)
        sampler_query.stop()

        try:
            client.collections.delete("Bench")
        except Exception:
            pass

    finally:
        client.close()

    recalls = compute_recalls(all_returned, true_top1, id_fn=id_fn)
    mrr     = compute_mrr(all_returned, true_top1, id_fn=id_fn)
    lats_s  = sorted(lats)
    label   = f"Weaviate {wv_ver} (HNSW+RQ)"
    return _make_result(label, N, DIM, n_q, N / ingest_s, ingest_s, 0.0,
                        sampler_ingest, sampler_query, lats_s, recalls, mrr)


# ── Shared result builder ─────────────────────────────────────────────────────

def _make_result(
    engine: str, N: int, DIM: int, n_q: int,
    throughput_vps: float, ingest_s: float, disk_mb: float,
    si: CpuRamSampler, sq: CpuRamSampler,
    lats_s: list[float], recalls: dict, mrr: float,
) -> dict:
    return {
        "engine":              engine,
        "n": N, "dim": DIM, "n_queries": n_q,
        "throughput_vps":      round(throughput_vps),
        "ingest_s":            round(ingest_s, 3),
        "index_s":             None,
        "disk_mb":             round(disk_mb, 2),
        "ram_delta_mb":        round(si.delta_ram_mb, 1),
        "ram_ingest_peak_mb":  round(si.peak_ram_mb, 1),
        "ram_query_peak_mb":   round(sq.peak_ram_mb, 1),
        "cpu_ingest_pct":      round(si.avg_cpu_pct, 1),
        "cpu_query_pct":       round(sq.avg_cpu_pct, 1),
        "p50_ms":              round(lats_s[int(n_q * 0.50)], 2),
        "p95_ms":              round(lats_s[int(n_q * 0.95)], 2),
        "p99_ms":              round(lats_s[min(int(n_q * 0.99), n_q - 1)], 2),
        "mrr":                 round(mrr, 4),
        "recall":              {str(k): round(v, 4) for k, v in recalls.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        choices=["glove", "dbpedia1536"],
                        default=["glove", "dbpedia1536"])
    parser.add_argument("--no-weaviate", action="store_true",
                        help="Skip Weaviate (requires Docker with weaviate:1.33+)")
    parser.add_argument("--weaviate-host", default="localhost")
    parser.add_argument("--weaviate-port", type=int, default=8080)
    args = parser.parse_args()

    DS_MAP = {
        "glove":       ("glove-200",    load_glove),
        "dbpedia1536": ("dbpedia-1536", lambda: load_dbpedia(1536)),
    }

    print("=" * 72)
    print("  Rotation-based competitor benchmark")
    print(f"  Datasets: {args.datasets}")
    print("=" * 72, flush=True)

    all_results: dict[str, list[dict]] = {}

    for ds_key in args.datasets:
        ds_label, loader = DS_MAP[ds_key]
        print(f"\n── {ds_label} {'─'*50}", flush=True)
        vecs, qvecs, true_top1 = loader()

        ds_results: list[dict] = []

        runners = [
            ("LanceDB IVF_RQ (RaBitQ)", lambda v=vecs, q=qvecs, t=true_top1: run_lancedb_rabitq(v, q, t)),
        ]
        if not args.no_weaviate:
            h, p = args.weaviate_host, args.weaviate_port
            runners.append(
                ("Weaviate HNSW+RQ",
                 lambda v=vecs, q=qvecs, t=true_top1, hh=h, pp=p: run_weaviate_rq(v, q, t, hh, pp))
            )

        for name, fn in runners:
            print(f"  {name} ...", flush=True)
            r = fn()
            if r is not None:
                ds_results.append(r)
                print(f"    R@1={r['recall']['1']:.3f}  R@8={r['recall']['8']:.3f}  "
                      f"p50={r['p50_ms']:.1f}ms  vps={r['throughput_vps']:,}  disk={r['disk_mb']:.1f}MB", flush=True)

        all_results[ds_label] = ds_results

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)

    # Summary table
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for ds_label, rows in all_results.items():
        print(f"\n{ds_label}")
        print(f"  {'Engine':<40} {'R@1':>6} {'R@8':>6} {'vps':>8} {'p50ms':>7} {'disk_mb':>8}")
        print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
        for r in rows:
            print(f"  {r['engine']:<40} "
                  f"{r['recall']['1']:>6.3f} "
                  f"{r['recall']['8']:>6.3f} "
                  f"{r['throughput_vps']:>8,} "
                  f"{r['p50_ms']:>7.1f} "
                  f"{r['disk_mb']:>8.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
