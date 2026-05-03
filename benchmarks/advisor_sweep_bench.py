"""advisor_sweep_bench.py — Generate advisor_data.json from fresh v0.8.3 measurements.

Replaces the stale advisor_data.json that was built before:
  - the v0.8.3 SIMD speedups landed (latencies were 3-5× too high)
  - residual_int4 rerank existed (the new compression-first standard)
  - dbpedia got bits=1,3 coverage
  - lastfm dim was correctly labelled (was "lastfm-64" but data is d=65)

What this measures, per dataset:
  bits ∈ {1, 2, 3, 4}        — every supported bit-rate
  rerank_precision ∈ {None, "int8", "residual_int4"}
                              — None == rerank=False (codes only)
                              — int8 == previous default (highest recall, biggest disk)
                              — residual_int4 == v0.8.3 default (half disk, ~int8 recall)
  ann ∈ {False, True}        — brute vs HNSW
  qtype = dense (Haar QR)    — paper-aligned default; srht is power-user, omitted
  fast_mode = True           — production default; fast=False is rarely used

That's 4 × 3 × 2 = 24 configs per dataset.

Datasets (all paper-real, normalized to unit L2):
  d=65   lastfm-65          (open-vdb/lastfm-64-dot)
  d=96   deep-96            (open-vdb/deep-image-96-angular)
  d=100  glove-100          (open-vdb/glove-100-angular)
  d=200  glove-200
  d=256  nytimes-256        (open-vdb/nytimes-256-angular)
  d=768  arxiv-768          (InstructorXL abstracts)
  d=960  gist-960           (open-vdb/gist-960-euclidean, normalized)
  d=1536 dbpedia-1536       (OpenAI text-embedding-3)
  d=3072 dbpedia-3072

Scale:
  N = 100,000   (matches paper bench)
  queries = 500 (sufficient for stable p50)

Output:
  benchmarks/_advisor_sweep_results.json    — raw per-config rows
  website/data/advisor_data.json            — advisor-format JSON
  website/data/advisor_data.js              — JS wrapper (window.__TQDB_ADVISOR_DATA__)

Usage:
    # Full sweep — 9 datasets × 24 configs (~30-90 min depending on hardware):
    python benchmarks/advisor_sweep_bench.py

    # Subset:
    python benchmarks/advisor_sweep_bench.py --datasets glove200 dbpedia1536

    # Smoke test (3 configs × first 2 datasets, 5k corpus):
    python benchmarks/advisor_sweep_bench.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_core import (
    K_VALUES,
    compute_recalls,
    compute_mrr,
    disk_size_mb,
    load_glove,
    load_dbpedia,
    load_arxiv768,
)
from dim_sweep_bench import (
    load_lastfm64,
    load_deepimage96,
    load_glove100,
    load_nytimes256,
    load_gist960,
)

REPO_DIR     = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "_advisor_sweep_results.json"
ADVISOR_JSON = REPO_DIR / "website" / "data" / "advisor_data.json"
ADVISOR_JS   = REPO_DIR / "website" / "data" / "advisor_data.js"

# Advisor exposes these top-k buckets (must match advisor.html chip data-rk values)
RK_KEYS = ["1", "2", "4", "8", "16", "32"]

# ── Config matrix ─────────────────────────────────────────────────────────────
# (bits, rerank_precision, ann)
# rerank_precision = None means rerank=False (codes only).
SWEEP_CONFIGS: list[tuple[int, str | None, bool]] = [
    (b, rp, ann)
    for b   in [1, 2, 3, 4]
    for rp  in [None, "int8", "residual_int4"]
    for ann in [False, True]
]

# Skip ANN for these datasets — HNSW build at d>=1024 + low bits is impractically slow
# (>20 min per config), and at N=100k brute is fast enough that ANN buys little.
ANN_SKIP_DATASETS = {"dbpedia-1536", "dbpedia-3072"}


def configs_for(ds_label: str) -> list[tuple[int, str | None, bool]]:
    """Return the config list applicable to this dataset."""
    if ds_label in ANN_SKIP_DATASETS:
        return [c for c in SWEEP_CONFIGS if not c[2]]  # drop ann=True rows
    return SWEEP_CONFIGS


def cfg_label(bits: int, rerank_prec: str | None, ann: bool) -> str:
    rp = rerank_prec or "F"
    return f"b={bits} rerank={rp} {'ann' if ann else 'brute'}"


# ── Dataset registry ──────────────────────────────────────────────────────────
# Note: lastfm dataset's actual native dim is 65 (not 64) — we use the truthful name.
ALL_DATASETS = {
    "lastfm":      ("lastfm-65",     load_lastfm64),
    "deep96":      ("deep-96",       load_deepimage96),
    "glove100":    ("glove-100",     load_glove100),
    "glove200":    ("glove-200",     load_glove),
    "nytimes":     ("nytimes-256",   load_nytimes256),
    "arxiv768":    ("arxiv-768",     load_arxiv768),
    "gist960":     ("gist-960",      load_gist960),
    "dbpedia1536": ("dbpedia-1536",  lambda: load_dbpedia(1536)),
    "dbpedia3072": ("dbpedia-3072",  lambda: load_dbpedia(3072)),
}

DEFAULT_ORDER = ["lastfm", "deep96", "glove100", "glove200", "nytimes",
                 "arxiv768", "gist960", "dbpedia1536", "dbpedia3072"]


# ── Single-config runner ──────────────────────────────────────────────────────

def run_one(vecs: np.ndarray, qvecs: np.ndarray, true_top1: np.ndarray,
            bits: int, rerank_prec: str | None, ann: bool) -> dict:
    """Run one config end-to-end and return all metrics."""
    import tqdb  # noqa: PLC0415

    N, DIM = vecs.shape
    n_q    = len(qvecs)
    max_k  = K_VALUES[-1]
    rerank = rerank_prec is not None
    ids    = [str(i) for i in range(N)]

    with tempfile.TemporaryDirectory(prefix="adv_sw_") as tmp:
        # ─ Ingest ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        kwargs = dict(
            dimension=DIM, bits=bits, metric="ip",
            rerank=rerank, fast_mode=True,
        )
        if rerank_prec:
            kwargs["rerank_precision"] = rerank_prec
        db = tqdb.Database.open(tmp, **kwargs)
        for start in range(0, N, 2000):
            db.insert_batch(ids[start:start + 2000], vecs[start:start + 2000])
        db.flush()
        ingest_s = time.perf_counter() - t0

        # Close + reopen so disk = steady-state including rerank file + segments
        db.close()
        dm = disk_size_mb(tmp)
        db = tqdb.Database.open(tmp, **kwargs)

        index_s = None
        if ann:
            t_idx = time.perf_counter()
            db.create_index(max_degree=32, search_list_size=128, alpha=1.2)
            index_s = round(time.perf_counter() - t_idx, 3)
            dm = disk_size_mb(tmp)  # graph.bin

        # ─ Warmup (page mmap into OS cache) ───────────────────────────────────
        for q in qvecs[:min(20, n_q)]:
            db.search(q, top_k=max_k, _use_ann=ann)

        # ─ Query ──────────────────────────────────────────────────────────────
        lats: list[float] = []
        all_returned: list[list[str]] = []
        for q in qvecs:
            t1  = time.perf_counter()
            res = db.search(q, top_k=max_k, _use_ann=ann)
            lats.append((time.perf_counter() - t1) * 1000)
            all_returned.append([r["id"] for r in res])

        # Close before tempdir cleanup — Windows can't delete files mmap is holding.
        db.close()

    recalls = compute_recalls(all_returned, true_top1)
    mrr     = compute_mrr(all_returned, true_top1)
    lats_s  = sorted(lats)

    return {
        "bits":              bits,
        "rerank":            rerank,
        "rerank_precision":  rerank_prec,    # None | "int8" | "residual_int4"
        "ann":               ann,
        "fast_mode":         True,
        "qtype":             "dense",        # always dense in this sweep
        "n":                 N,
        "dim":               DIM,
        "n_queries":         n_q,
        "throughput_vps":    int(N / ingest_s),
        "ingest_s":          round(ingest_s, 3),
        "index_s":           index_s,
        "disk_mb":           round(dm, 2),
        "p50_ms":            round(lats_s[int(n_q * 0.50)], 2),
        "p95_ms":            round(lats_s[int(n_q * 0.95)], 2),
        "mrr":               round(mrr, 4),
        "recall":            {str(k): round(v, 4) for k, v in recalls.items()},
    }


# ── Build advisor format ──────────────────────────────────────────────────────

def to_advisor_entry(ds_name: str, r: dict) -> dict:
    """Convert raw bench row to the format advisor.js expects."""
    rk = {k: r["recall"].get(k, 0.0) for k in RK_KEYS}
    n, d, disk = r["n"], r["dim"], r["disk_mb"]
    float32_mb = n * d * 4 / (1024 ** 2)
    compr = round(float32_mb / disk, 2) if disk > 0 else 0.0
    return {
        "ds":     ds_name,
        "d":      d,
        "bits":   r["bits"],
        "rerank": r["rerank"],
        "rerank_precision": r["rerank_precision"],
        "ann":    r["ann"],
        "fast":   r["fast_mode"],
        "qtype":  r["qtype"],
        "rk":     rk,
        "mrr":    r["mrr"],
        "p50":    r["p50_ms"],
        "disk":   r["disk_mb"],
        "compr":  compr,
        "vps":    r["throughput_vps"],
        "src":    "v0.8.3-sweep",
    }


def write_advisor_files(all_results: dict[str, list[dict]]) -> None:
    """Write website/data/advisor_data.{json,js} from raw results."""
    dims: dict[str, int] = {}
    configs: dict[str, list[dict]] = {}
    for ds_name, rows in sorted(all_results.items()):
        if not rows:
            continue
        dims[ds_name] = rows[0]["dim"]
        configs[ds_name] = [to_advisor_entry(ds_name, r) for r in rows]

    payload = {"dims": dims, "configs": configs}

    ADVISOR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ADVISOR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    js = "window.__TQDB_ADVISOR_DATA__ = " + json.dumps(payload) + ";\n"
    ADVISOR_JS.write_text(js, encoding="utf-8")
    print(f"\nWrote {ADVISOR_JSON.relative_to(REPO_DIR)} ({ADVISOR_JSON.stat().st_size//1024} KB)")
    print(f"Wrote {ADVISOR_JS.relative_to(REPO_DIR)} ({ADVISOR_JS.stat().st_size//1024} KB)")


# ── Main loop ─────────────────────────────────────────────────────────────────

N_CORPUS  = 100_000
N_QUERIES = 500


def load_slice(key: str):
    label, loader = ALL_DATASETS[key]
    print(f"\n[{label}]", flush=True)
    vecs, qvecs, _ = loader()
    corpus  = vecs[:N_CORPUS]
    queries = qvecs[:N_QUERIES]
    rows = [
        np.argmax(queries[i:i+200] @ corpus.T, axis=1)
        for i in range(0, len(queries), 200)
    ]
    true_top1 = np.concatenate(rows)
    print(f"  Slice: corpus={corpus.shape}, queries={queries.shape}", flush=True)
    return label, corpus, queries, true_top1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(ALL_DATASETS.keys()),
                   default=DEFAULT_ORDER)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    global N_CORPUS, N_QUERIES
    configs = SWEEP_CONFIGS
    datasets_to_run = args.datasets

    if args.smoke:
        configs = SWEEP_CONFIGS[:3]
        datasets_to_run = DEFAULT_ORDER[:2]
        N_CORPUS = 5_000
        N_QUERIES = 100
        print("=== SMOKE: 3 configs × 2 datasets, N=5k ===\n")

    print(f"Sweep: {len(datasets_to_run)} datasets × {len(configs)} configs"
          f" = {len(datasets_to_run) * len(configs)} runs")
    print(f"N={N_CORPUS:,}  queries={N_QUERIES}\n")

    # Resume support
    all_results: dict[str, list[dict]] = {}
    if RESULTS_PATH.exists():
        all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        prior = sum(len(v) for v in all_results.values())
        print(f"Resuming from {RESULTS_PATH.name} ({prior} prior entries)\n")

    sys.stdout.reconfigure(encoding="utf-8")

    for key in datasets_to_run:
        ds_label, corpus, queries, true_top1 = load_slice(key)
        existing = all_results.get(ds_label, [])
        # Cache key = (bits, rerank_precision, ann)
        done_keys = {(r["bits"], r["rerank_precision"], r["ann"]) for r in existing
                     if "error" not in r}
        rows = list(existing)

        # Use the per-dataset config list — skips ANN for high-dim datasets.
        ds_configs = configs if not args.smoke else configs
        if not args.smoke:
            ds_configs = configs_for(ds_label)

        n_total = len(ds_configs)
        for i, (bits, rp, ann) in enumerate(ds_configs, 1):
            key_t = (bits, rp, ann)
            label = cfg_label(bits, rp, ann)
            if key_t in done_keys:
                print(f"  [{i}/{n_total}] {label}  (cached)", flush=True)
                continue
            print(f"\n  [{i}/{n_total}] {label}", flush=True)
            try:
                r = run_one(corpus, queries, true_top1, bits, rp, ann)
                rows.append(r)
                print(f"    R@1={r['recall']['1']:.3f}  p50={r['p50_ms']:.2f}ms  "
                      f"disk={r['disk_mb']:.1f}MB  ingest={r['ingest_s']:.1f}s",
                      flush=True)
            except Exception as exc:
                print(f"    ERROR: {exc}", flush=True)
                rows.append({"bits": bits, "rerank_precision": rp, "ann": ann,
                              "error": str(exc)})

        all_results[ds_label] = rows
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n  Saved — {ds_label}: {len(rows)} entries", flush=True)

    # ─ Write advisor JSON + JS wrapper ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Writing advisor files...")
    print("=" * 72)
    # Filter out error rows when writing the advisor format
    clean = {ds: [r for r in rows if "error" not in r]
             for ds, rows in all_results.items()}
    clean = {ds: rows for ds, rows in clean.items() if rows}
    write_advisor_files(clean)

    print(f"\nDone. {sum(len(v) for v in clean.values())} valid configs across "
          f"{len(clean)} datasets.")


if __name__ == "__main__":
    main()
