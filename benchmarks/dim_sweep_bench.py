"""
Dimension sweep benchmark — full 32 configs × 9 real-dimension datasets.

Runs all 32 FULL_CONFIGS (same as full_config_bench.py) on real embedding datasets
at their native dimensions. Produces identical PNG plots and markdown tables.

Datasets (10k corpus, 500 queries each — real embeddings, no projection):
  d=65   : LastFM-64           (open-vdb/lastfm-64-dot)
  d=96   : DeepImage-96        (open-vdb/deep-image-96-angular)
  d=100  : GloVe-100           (open-vdb/glove-100-angular)
  d=200  : GloVe-200           (open-vdb/glove-200-angular — cached)
  d=256  : NYTimes-256         (open-vdb/nytimes-256-angular)
  d=768  : arXiv-768           (InstructorXL abstracts — cached)
  d=960  : GIST-960            (open-vdb/gist-960-euclidean, normalised to IP)
  d=1536 : DBpedia-1536        (OpenAI text-embedding-3 — cached)
  d=3072 : DBpedia-3072        (OpenAI text-embedding-3 — cached)

Note: no standard native embedding datasets exist between d=960–1536 or d=1536–3072.

Outputs saved to benchmarks/_dim_sweep_results/:
  dim_sweep_results.json        — same schema as _full_config_results.json
  dim_sweep_report.md           — same table layout as _full_config_report.md
  _full_config_<ds>_recall.png  — recall curves (reuses full_config_bench plots)
  _full_config_<ds>_tradeoffs_*.png — tradeoff scatter grids

Usage:
    python benchmarks/dim_sweep_bench.py
    python benchmarks/dim_sweep_bench.py --datasets lastfm glove100 arxiv768
    python benchmarks/dim_sweep_bench.py --smoke   # 3 configs × first 2 datasets
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_core import CKPT_DIR, load_glove, load_arxiv768, load_dbpedia
import full_config_bench as fcb

# ── Output directory ──────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "_dim_sweep_results"

# Redirect plot output to OUT_DIR (plot_recall/plot_tradeoffs use fcb.BENCH_DIR)
fcb.BENCH_DIR = OUT_DIR

# ── Config ────────────────────────────────────────────────────────────────────

N_CORPUS  = 10_000
N_QUERIES = 500

RESULTS_PATH = OUT_DIR / "dim_sweep_results.json"
REPORT_PATH  = OUT_DIR / "dim_sweep_report.md"


# ── New dataset loaders ───────────────────────────────────────────────────────

def _load_open_vdb(hf_id, label, n_data, n_queries):
    from datasets import load_dataset

    tag = hf_id.replace("/", "_").replace("-", "_")
    ckpt_v = CKPT_DIR / f"{tag}_{n_data}_vecs.npy"
    ckpt_q = CKPT_DIR / f"{tag}_{n_queries}_qvecs.npy"
    ckpt_t = CKPT_DIR / f"{tag}_{n_data}_truth_top1.npy"

    if ckpt_v.exists() and ckpt_q.exists():
        print(f"  Loading cached {label} ...", flush=True)
        vecs  = np.load(ckpt_v)
        qvecs = np.load(ckpt_q)
    else:
        print(f"  Downloading {label} corpus ({n_data:,}) ...", flush=True)
        ds = load_dataset(hf_id, "train", split="train", streaming=True)
        raw = []
        for i, row in enumerate(ds):
            if i >= n_data:
                break
            raw.append(row["emb"])
            if (i + 1) % 10_000 == 0:
                print(f"    {i+1:>7,}/{n_data:,}", flush=True)
        vecs = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.where(norms > 0, norms, 1.0)

        print(f"  Downloading {label} queries ({n_queries:,}) ...", flush=True)
        dsq = load_dataset(hf_id, "test", split="test", streaming=True)
        qraw = [row["emb"] for i, row in enumerate(dsq) if i < n_queries]
        qvecs = np.array(qraw, dtype=np.float32)
        norms_q = np.linalg.norm(qvecs, axis=1, keepdims=True)
        qvecs /= np.where(norms_q > 0, norms_q, 1.0)

        np.save(ckpt_v, vecs)
        np.save(ckpt_q, qvecs)
        print("  Saved.", flush=True)

    if ckpt_t.exists():
        print(f"  Loading cached {label} ground truth ...", flush=True)
        true_top1 = np.load(ckpt_t)
    else:
        print(f"  Computing {label} ground truth ...", flush=True)
        t0 = time.perf_counter()
        rows = [
            np.argmax(qvecs[i:i+200] @ vecs.T, axis=1)
            for i in range(0, len(qvecs), 200)
        ]
        true_top1 = np.concatenate(rows)
        print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
        np.save(ckpt_t, true_top1)

    print(f"  {label}: corpus={vecs.shape}, queries={qvecs.shape}", flush=True)
    return vecs, qvecs, true_top1


def load_lastfm64():
    return _load_open_vdb("open-vdb/lastfm-64-dot", "LastFM-64", 100_000, 10_000)

def load_deepimage96():
    return _load_open_vdb("open-vdb/deep-image-96-angular", "DeepImage-96", 100_000, 10_000)

def load_glove100():
    return _load_open_vdb("open-vdb/glove-100-angular", "GloVe-100", 100_000, 10_000)

def load_nytimes256():
    return _load_open_vdb("open-vdb/nytimes-256-angular", "NYTimes-256", 100_000, 10_000)

def load_gist960():
    # Euclidean source; normalised to unit norm for consistent IP metric
    return _load_open_vdb("open-vdb/gist-960-euclidean", "GIST-960", 100_000, 1_000)


# ── Dataset registry ──────────────────────────────────────────────────────────

ALL_DATASETS: dict[str, tuple[str, callable]] = {
    "lastfm":    ("lastfm-64",    load_lastfm64),
    "deep96":    ("deep-96",      load_deepimage96),
    "glove100":  ("glove-100",    load_glove100),
    "glove200":  ("glove-200",    load_glove),
    "nytimes":   ("nytimes-256",  load_nytimes256),
    "arxiv768":  ("arxiv-768",    load_arxiv768),
    "gist960":   ("gist-960",     load_gist960),
    "dbpedia1536": ("dbpedia-1536", lambda: load_dbpedia(1536)),
    "dbpedia3072": ("dbpedia-3072", lambda: load_dbpedia(3072)),
}

DEFAULT_ORDER = ["lastfm", "deep96", "glove100", "glove200", "nytimes",
                 "arxiv768", "gist960", "dbpedia1536", "dbpedia3072"]


def load_dataset_slice(key: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
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


# ── Main run loop ─────────────────────────────────────────────────────────────

def run_dataset(ds_label: str, corpus, queries, true_top1,
                configs, all_results: dict) -> None:
    existing = {r["label"]: r for r in all_results.get(ds_label, [])}
    new_results = list(existing.values())
    done_labels = set(existing.keys())

    n_total = len(configs)
    for i, (bits, rerank, ann, fast_mode, qtype) in enumerate(configs, 1):
        lbl = fcb.config_label(bits, rerank, ann, fast_mode, qtype)
        if lbl in done_labels:
            print(f"  [{i}/{n_total}] {lbl}  (cached)", flush=True)
            continue

        print(f"\n  [{i}/{n_total}] {lbl}", flush=True)
        try:
            r = fcb.run_one_config(corpus, queries, true_top1,
                                   bits, rerank, ann, fast_mode, qtype)
            new_results.append(r)
            r1   = r["recall"].get("1", 0)
            p50  = r.get("p50_ms", 0)
            disk = r.get("disk_mb", 0)
            vps  = r.get("throughput_vps", 0)
            print(f"    R@1={r1:.3f}  p50={p50:.1f}ms  disk={disk:.1f}MB  {vps:,}vps",
                  flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            new_results.append({"label": lbl, "bits": bits, "rerank": rerank,
                                 "ann": ann, "fast_mode": fast_mode,
                                 "quantizer_type": qtype, "error": str(exc)})

    # Sort to match FULL_CONFIGS order
    order = {fcb.config_label(*c): i for i, c in enumerate(fcb.FULL_CONFIGS)}
    new_results.sort(key=lambda r: order.get(r["label"], 999))
    all_results[ds_label] = new_results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(ALL_DATASETS.keys()),
                        default=DEFAULT_ORDER,
                        help="Which datasets to run (default: all 9)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick sanity check: 3 configs × first 2 datasets")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fcb.BENCH_DIR = out_dir
    results_path = out_dir / "dim_sweep_results.json"
    report_path  = out_dir / "dim_sweep_report.md"

    configs = fcb.FULL_CONFIGS
    datasets_to_run = args.datasets

    if args.smoke:
        configs = fcb.FULL_CONFIGS[:3]
        datasets_to_run = DEFAULT_ORDER[:2]
        print("=== SMOKE TEST: 3 configs × 2 datasets ===\n")

    print(f"Dim sweep: {len(datasets_to_run)} dataset(s) × {len(configs)} configs"
          f" = {len(datasets_to_run) * len(configs)} runs")
    print(f"Corpus: {N_CORPUS:,}  Queries: {N_QUERIES}  Output: {out_dir}/\n")

    # Load existing results (resume support)
    all_results: dict = {}
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"Resuming from {results_path} "
              f"({sum(len(v) for v in all_results.values())} existing entries)\n")

    for key in datasets_to_run:
        ds_label, corpus, queries, true_top1 = load_dataset_slice(key)

        n_done = sum(1 for r in all_results.get(ds_label, []) if "error" not in r)
        print(f"\n{'='*72}")
        print(f"  {ds_label}  ({n_done}/{len(configs)} already done)")
        print(f"{'='*72}")

        run_dataset(ds_label, corpus, queries, true_top1, configs, all_results)

        # Save after each dataset
        results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n  Saved — {ds_label} now has "
              f"{len(all_results[ds_label])} entries.", flush=True)

        # Per-dataset plots
        try:
            fcb.plot_recall(ds_label, all_results[ds_label])
            fcb.plot_tradeoffs(ds_label, all_results[ds_label])
        except Exception as e:
            print(f"  Warning: plot failed for {ds_label}: {e}", flush=True)

    # Final report
    try:
        report = fcb.generate_report(all_results)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport → {report_path}")
    except Exception as e:
        print(f"  Warning: report failed: {e}", flush=True)

    print(f"\nResults → {results_path}")
    total = sum(len(v) for v in all_results.values())
    expected = len(datasets_to_run) * len(configs)
    print(f"Total: {total}/{expected} entries across {len(datasets_to_run)} datasets")


if __name__ == "__main__":
    main()
