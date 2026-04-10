"""
SRHT vs. Dense Quantizer -- Full Config Matrix
==============================================
Uses the same GloVe-200 and DBpedia-1536/3072 corpora as the paper benchmark
(arXiv:2504.19874).  Tests the full 2x2x2 matrix:

  {srht, dense} x {rerank=F, rerank=T} x {brute, ANN}   at b=2 and b=4

For each (quantizer, rerank) pair the DB is built once; brute-force and ANN
search are both measured from the same DB to avoid re-ingesting.

Metrics per config:
  Recall@k for k in [1, 2, 4, 8, 16, 32, 64]
  MRR  (Mean Reciprocal Rank of true top-1)
  Ingest throughput (vps)           -- brute configs only (shared ingest)
  Query latency  p50 / p99 (ms)
  Disk footprint (MB)
  RAM ingest peak / RAM query peak (MB)
  CPU ingest % / CPU query %

Usage:
    python benchmarks/compare_quantizers_paper.py
    python benchmarks/compare_quantizers_paper.py --datasets glove --bits 4
    python benchmarks/compare_quantizers_paper.py --save-json benchmarks/_quantizer_matrix.json
    python benchmarks/compare_quantizers_paper.py --deltas
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import os
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_core import (
    K_VALUES,
    compute_recalls,
    compute_mrr,
    disk_size_mb,
    CpuRamSampler,
    load_glove,
    load_dbpedia,
)

import tqdb

DATASETS_ALL = ["glove", "dbpedia-1536", "dbpedia-3072"]
TOP_K = 64          # retrieve this many neighbours; recall@k uses prefix
ANN_EF = 128        # ef_search for ANN queries
ANN_MAX_DEGREE = 32
ANN_EF_CONSTRUCTION = 200

# Metrics printed in order (label, key-path, format)
METRIC_ROWS: list[tuple[str, str, str]] = [
    ("recall@1",          "recalls.1",        ".4f"),
    ("recall@4",          "recalls.4",        ".4f"),
    ("recall@8",          "recalls.8",        ".4f"),
    ("mrr",               "mrr",              ".4f"),
    ("p50 latency (ms)",  "p50_ms",           ".2f"),
    ("p99 latency (ms)",  "p99_ms",           ".2f"),
    ("ingest (vps)",      "ingest_vps",       ".0f"),
    ("disk (MB)",         "disk_mb",          ".1f"),
    ("RAM ingest (MB)",   "ram_ingest_mb",    ".1f"),
    ("RAM query (MB)",    "ram_query_mb",     ".1f"),
    ("CPU ingest (%)",    "cpu_ingest_pct",   ".1f"),
    ("CPU query (%)",     "cpu_query_pct",    ".1f"),
    ("index build (s)",   "index_time_s",     ".1f"),
]


def _get(data: dict, path: str) -> float | None:
    """Dot-path accessor: 'recalls.1' -> data['recalls'][1]"""
    parts = path.split(".", 1)
    val = data.get(parts[0])
    if val is None:
        return None
    if len(parts) == 1:
        return float(val) if val is not None else None
    # second part is a dict key -- try int first
    try:
        key: Any = int(parts[1])
    except ValueError:
        key = parts[1]
    if isinstance(val, dict) and key in val:
        return float(val[key])
    return None


def run_config(
    qt: str,
    rerank: bool,
    vecs: np.ndarray,
    qvecs: np.ndarray,
    true_top1: np.ndarray,
    bits: int,
) -> dict:
    """
    Run one (quantizer, rerank) pair.  Ingests once, measures brute-force,
    builds ANN index, then measures ANN search.

    Returns:
        {
          "brute": {ingest_vps, p50_ms, p99_ms, disk_mb, ram_ingest_mb,
                    ram_query_mb, cpu_ingest_pct, cpu_query_pct, mrr, recalls},
          "ann":   {index_time_s, p50_ms, p99_ms, disk_mb, ram_query_mb,
                    cpu_query_pct, mrr, recalls,
                    # brute ingest fields forwarded for display convenience:
                    ingest_vps, ram_ingest_mb, cpu_ingest_pct},
        }
    """
    N, D = vecs.shape
    ids = [str(i) for i in range(N)]

    tmpdir = tempfile.TemporaryDirectory()
    tmp = tmpdir.name
    try:
        db = tqdb.Database.open(
            tmp, dimension=D, bits=bits, metric="ip",
            rerank=rerank, quantizer_type=qt,
        )

        # --- Ingest ---
        with CpuRamSampler() as ingest_sampler:
            t0 = time.perf_counter()
            batch = 5_000
            for s in range(0, N, batch):
                db.insert_batch(ids[s:s + batch], vecs[s:s + batch])
            db.flush()
            ingest_s = time.perf_counter() - t0

        disk_mb_codes = disk_size_mb(tmp)

        # --- Brute-force query ---
        brute_lats: list[float] = []
        brute_ids: list[list[str]] = []
        with CpuRamSampler() as brute_sampler:
            for q in qvecs:
                t0 = time.perf_counter()
                hits = db.search(q, top_k=TOP_K, _use_ann=False)
                brute_lats.append(time.perf_counter() - t0)
                brute_ids.append([r["id"] for r in hits])

        brute_data: dict[str, Any] = {
            "ingest_vps":     N / ingest_s,
            "p50_ms":         float(np.median(brute_lats)) * 1000,
            "p99_ms":         float(np.percentile(brute_lats, 99)) * 1000,
            "disk_mb":        disk_mb_codes,
            "ram_ingest_mb":  ingest_sampler.peak_ram_mb,
            "ram_query_mb":   brute_sampler.peak_ram_mb,
            "cpu_ingest_pct": ingest_sampler.avg_cpu_pct,
            "cpu_query_pct":  brute_sampler.avg_cpu_pct,
            "mrr":            compute_mrr(brute_ids, true_top1),
            "recalls":        compute_recalls(brute_ids, true_top1),
            "index_time_s":   None,
        }

        # --- Build ANN index ---
        t0 = time.perf_counter()
        db.create_index(
            max_degree=ANN_MAX_DEGREE,
            ef_construction=ANN_EF_CONSTRUCTION,
            search_list_size=ANN_EF,
        )
        index_time_s = time.perf_counter() - t0
        disk_mb_with_index = disk_size_mb(tmp)

        # --- ANN query ---
        ann_lats: list[float] = []
        ann_ids: list[list[str]] = []
        with CpuRamSampler() as ann_sampler:
            for q in qvecs:
                t0 = time.perf_counter()
                hits = db.search(q, top_k=TOP_K, _use_ann=True,
                                 ann_search_list_size=ANN_EF)
                ann_lats.append(time.perf_counter() - t0)
                ann_ids.append([r["id"] for r in hits])

        ann_data: dict[str, Any] = {
            "ingest_vps":     N / ingest_s,          # forward for display
            "p50_ms":         float(np.median(ann_lats)) * 1000,
            "p99_ms":         float(np.percentile(ann_lats, 99)) * 1000,
            "disk_mb":        disk_mb_with_index,
            "ram_ingest_mb":  ingest_sampler.peak_ram_mb,
            "ram_query_mb":   ann_sampler.peak_ram_mb,
            "cpu_ingest_pct": ingest_sampler.avg_cpu_pct,
            "cpu_query_pct":  ann_sampler.avg_cpu_pct,
            "index_time_s":   index_time_s,
            "mrr":            compute_mrr(ann_ids, true_top1),
            "recalls":        compute_recalls(ann_ids, true_top1),
        }

        return {"brute": brute_data, "ann": ann_data}
    finally:
        tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _col_header(qt: str, rerank: bool, search: str) -> str:
    rk = "rk=T" if rerank else "rk=F"
    return f"{qt}/{rk}/{search}"


def print_matrix(
    ds_name: str,
    bits: int,
    results: dict[tuple[str, bool, str], dict],
    show_deltas: bool,
) -> None:
    """
    results key: (qt, rerank, search_mode) -> metric dict
    """
    keys = [
        ("srht",  False, "brute"),
        ("srht",  False, "ann"),
        ("srht",  True,  "brute"),
        ("srht",  True,  "ann"),
        ("dense", False, "brute"),
        ("dense", False, "ann"),
        ("dense", True,  "brute"),
        ("dense", True,  "ann"),
    ]
    # filter to only configs that were actually run
    keys = [k for k in keys if k in results]

    col_w = 12
    label_w = 22
    W = label_w + col_w * len(keys) + (col_w * 2 if show_deltas else 0)

    print(f"\n{'=' * W}")
    print(f"  {ds_name}  bits={bits}")
    print(f"{'=' * W}")

    # header
    header = f"  {'Metric':<{label_w}}"
    for qt, rk, sm in keys:
        h = _col_header(qt, rk, sm)
        header += f"{h:>{col_w}}"
    print(header)
    print(f"  {'-' * (W - 2)}")

    for label, path, fmt in METRIC_ROWS:
        line = f"  {label:<{label_w}}"
        vals = [_get(results[k], path) if k in results else None for k in keys]
        for v in vals:
            if v is None:
                line += f"{'--':>{col_w}}"
            else:
                line += f"{v:>{col_w}{fmt}}"

        if show_deltas and len(keys) >= 5:
            # D dense_brute_rk=F vs srht_brute_rk=F
            k_srht  = ("srht",  False, "brute")
            k_dense = ("dense", False, "brute")
            if k_srht in results and k_dense in results:
                vs = _get(results[k_srht],  path)
                vd = _get(results[k_dense], path)
                if vs is not None and vd is not None and abs(vs) > 1e-12:
                    pct = (vd - vs) / abs(vs) * 100
                    line += f"  D/S {pct:>+6.1f}%"

        print(line)

    print(f"  {'=' * (W - 2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS_ALL,
                    default=DATASETS_ALL,
                    help="Which datasets to run (default: all three)")
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 4],
                    help="Bit widths to test (default: 2 4)")
    ap.add_argument("--save-json", default=None, metavar="PATH",
                    help="Save results as JSON to this path")
    ap.add_argument("--deltas", action="store_true",
                    help="Show D dense vs srht column in output tables")
    ap.add_argument("--skip-dense-high-d", action="store_true", default=True,
                    help="Skip dense mode for D>=3072 (default: True)")
    ap.add_argument("--no-skip-dense-high-d", dest="skip_dense_high_d",
                    action="store_false")
    args = ap.parse_args()

    print("=" * 72)
    print("SRHT vs. Dense  --  Full Config Matrix  ({srht,dense} x {rk=F,T} x {brute,ANN})")
    print("=" * 72)

    dataset_loaders: list[tuple[str, callable]] = []
    if "glove" in args.datasets:
        dataset_loaders.append(("glove-200", load_glove))
    if "dbpedia-1536" in args.datasets:
        dataset_loaders.append(("dbpedia-1536", lambda: load_dbpedia(1536)))
    if "dbpedia-3072" in args.datasets:
        dataset_loaders.append(("dbpedia-3072", lambda: load_dbpedia(3072)))

    # all_json[ds_name][f"b{bits}"][(qt, rerank, search)] = metric_dict
    all_json: dict = {}

    for ds_name, loader in dataset_loaders:
        print(f"\nLoading {ds_name} ...", flush=True)
        vecs, qvecs, truth = loader()
        N, D = vecs.shape
        Q = len(qvecs)
        print(f"  corpus {N:,} x {D},  queries {Q:,}")
        all_json[ds_name] = {}

        for bits in args.bits:
            print(f"\n  [{ds_name}  bits={bits}]", flush=True)
            all_json[ds_name][f"b{bits}"] = {}
            # results[(qt, rerank, search_mode)] = metric dict
            results: dict[tuple[str, bool, str], dict] = {}

            for qt in ["srht", "dense"]:
                if qt == "dense" and args.skip_dense_high_d and D >= 3072:
                    print(f"    dense: skipped (D={D} >= 3072, O(d^2) prohibitive)")
                    continue
                for rerank in [False, True]:
                    label = f"{qt}/rerank={'T' if rerank else 'F'}"
                    print(f"    {label}: ingesting + brute + ANN ...", flush=True)
                    pair_result = run_config(qt, rerank, vecs, qvecs, truth, bits)

                    brute = pair_result["brute"]
                    ann   = pair_result["ann"]

                    key_b = (qt, rerank, "brute")
                    key_a = (qt, rerank, "ann")
                    results[key_b] = brute
                    results[key_a] = ann

                    r1b = brute["recalls"].get(1, 0)
                    r1a = ann["recalls"].get(1, 0)
                    print(f"    {label}: brute R@1={r1b:.3f} p50={brute['p50_ms']:.1f}ms"
                          f"  |  ANN R@1={r1a:.3f} p50={ann['p50_ms']:.1f}ms"
                          f"  (idx {ann['index_time_s']:.0f}s)")

            print_matrix(ds_name, bits, results, args.deltas)

            # Store for JSON
            def _key_str(qt: str, rk: bool, sm: str) -> str:
                return f"{qt}_rk{'T' if rk else 'F'}_{sm}"

            def _jsonify(d: dict) -> dict:
                out = dict(d)
                if "recalls" in out:
                    out["recalls"] = {str(k): v for k, v in out["recalls"].items()}
                return out

            for (qt, rk, sm), d in results.items():
                all_json[ds_name][f"b{bits}"][_key_str(qt, rk, sm)] = _jsonify(d)

    print("\n" + "=" * 72)
    print("Done.")

    if args.save_json:
        import json as _json
        out = Path(args.save_json)
        out.write_text(_json.dumps(all_json, indent=2), encoding="utf-8")
        print(f"  Results saved to {out}")


if __name__ == "__main__":
    main()
