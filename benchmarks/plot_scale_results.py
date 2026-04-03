"""
Plot scale-matrix benchmark results from bench_results_scale_matrix.json.
Generates a single PNG with 6 subplots.

Usage:
    python benchmarks/plot_scale_results.py
    python benchmarks/plot_scale_results.py --input bench_results_scale_matrix.json --out bench_plots.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ENGINE_LABELS = {
    "tqdb_hq":    "TQDB b=8 HQ",
    "tqdb_bal":   "TQDB b=4 Balanced",
    "tqdb_fast":  "TQDB b=4 FastBuild",
    "faiss_hnsw": "FAISS HNSW",
    "faiss_ivfpq":"FAISS IVF-PQ",
    "chromadb":   "Engine-A (HNSW)",
    "qdrant":     "Engine-F (HNSW)",
    "lancedb":    "Engine-B (IVF-PQ)",
}

# Colors and styles
STYLES = {
    "tqdb_hq":    dict(color="#1f77b4", marker="o",  linestyle="-",  linewidth=2.5),
    "tqdb_bal":   dict(color="#2ca02c", marker="s",  linestyle="-",  linewidth=2.5),
    "tqdb_fast":  dict(color="#17becf", marker="^",  linestyle="-",  linewidth=2.5),
    "faiss_hnsw": dict(color="#ff7f0e", marker="D",  linestyle="--", linewidth=1.5),
    "faiss_ivfpq":dict(color="#e377c2", marker="x",  linestyle="--", linewidth=1.5),
    "chromadb":   dict(color="#9467bd", marker="v",  linestyle="-.", linewidth=1.5),
    "qdrant":     dict(color="#d62728", marker="P",  linestyle=":",  linewidth=1.5),
    "lancedb":    dict(color="#8c564b", marker="*",  linestyle="-.", linewidth=1.5),
}

# Engines to exclude from disk plot (no full document store)
NO_DISK = {"faiss_hnsw", "faiss_ivfpq"}


def extract(data, metric, engine, sizes):
    vals = []
    for s in sizes:
        r = data.get(str(s), {}).get(engine)
        if r and not r.get("timed_out") and not r.get("error") and metric in r:
            vals.append(r[metric])
        else:
            vals.append(None)
    return vals


def plot_metric(ax, data, sizes, engines, metric, title, ylabel, skip=None,
                pct=False, yscale="linear", ymin=None):
    skip = skip or set()
    timed_out_engines = []
    for eng in engines:
        if eng in skip:
            continue
        vals = extract(data, metric, eng, sizes)
        xs = [s for s, v in zip(sizes, vals) if v is not None]
        ys = [v * 100 if pct else v for v, s in zip(vals, sizes) if v is not None]
        # check if any size timed out
        timed = any(
            data.get(str(s), {}).get(eng, {}) and
            isinstance(data.get(str(s), {}).get(eng), dict) and
            data[str(s)][eng].get("timed_out")
            for s in sizes
        )
        if not xs:
            if timed:
                timed_out_engines.append(ENGINE_LABELS.get(eng, eng))
            continue
        st = STYLES.get(eng, {})
        ax.plot(xs, ys, label=ENGINE_LABELS.get(eng, eng), **st)
        if timed:
            timed_out_engines.append(ENGINE_LABELS.get(eng, eng))
    if timed_out_engines:
        ax.text(0.02, 0.02, "⏱ timed out: " + ", ".join(timed_out_engines),
                transform=ax.transAxes, fontsize=6.5, color="gray",
                verticalalignment="bottom")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Vectors", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.set_xticks(sizes)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", alpha=0.3)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bench_results_scale_matrix.json")
    parser.add_argument("--out",   default="bench_plots.png")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    sizes   = sorted(data.keys(), key=int)
    sizes_i = [int(s) for s in sizes]
    engines = list(next(iter(data.values())).keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "TurboQuantDB Scale Benchmark  —  DBpedia OpenAI 1536-dim, top_k=10",
        fontsize=13, fontweight="bold", y=1.01)

    # 1. Ingest time
    plot_metric(axes[0, 0], data, sizes_i, engines,
                "ingest_time", "Ingest Time (s)", "seconds", yscale="log")

    # 2. Ingest speed (vec/s)
    plot_metric(axes[0, 1], data, sizes_i, engines,
                "ingest_speed", "Ingest Speed (vec/s)", "vectors / second", yscale="log")

    # 3. Disk usage — FAISS excluded
    plot_metric(axes[0, 2], data, sizes_i, engines,
                "disk_mb", "Disk Usage (MB)\n(FAISS excluded — index only, no doc store)",
                "MB", skip=NO_DISK, yscale="log")

    # 4. RAM at query time
    plot_metric(axes[1, 0], data, sizes_i, engines,
                "retrieve_rss_mb", "RAM at Query Time (MB)", "MB", yscale="log")

    # 5. p50 latency
    plot_metric(axes[1, 1], data, sizes_i, engines,
                "p50_ms", "p50 Query Latency (ms)", "ms", yscale="log")

    # 6. Recall@10
    plot_metric(axes[1, 2], data, sizes_i, engines,
                "recall_at_k", "Recall@10", "%", pct=True, ymin=60)

    # Shared legend below all plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
