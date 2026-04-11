"""full_config_bench.py — Exhaustive config × dataset benchmark for TurboQuantDB.

Sweeps the full 32-config matrix:
    bits ∈ {2, 4}  ×  rerank ∈ {F, T}  ×  ann ∈ {F, T}
    ×  fast_mode ∈ {T, F}  ×  quantizer_type ∈ {dense, srht}

Across 4 datasets:
    GloVe-200 (d=200, 100k corpus, 10k queries)
    DBpedia-1536 (d=1536, 100k corpus, 1k queries)
    DBpedia-3072 (d=3072, 100k corpus, 1k queries)
    arXiv-768 (d=768, 100k corpus, 2k queries)  ← new

Usage:
    # Phase 1 — validate all 32 configs quickly (synthetic data, no download):
    python benchmarks/full_config_bench.py --smoke

    # Phase 2 — full paper-scale run:
    python benchmarks/full_config_bench.py --full

    # Faster subsets:
    python benchmarks/full_config_bench.py --full --no-ann
    python benchmarks/full_config_bench.py --full --datasets glove arxiv768

Outputs (both gitignored):
    benchmarks/_full_config_results.json
    benchmarks/_full_config_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_core import (
    K_VALUES,
    PERF_METRIC_ROWS,
    CpuRamSampler,
    compute_recalls,
    compute_mrr,
    disk_size_mb,
    load_glove,
    load_dbpedia,
    load_arxiv768,
)

sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR    = Path(__file__).parent
RESULTS_PATH = BENCH_DIR / "_full_config_results.json"
REPORT_PATH  = BENCH_DIR / "_full_config_report.md"

# ── Config matrix ──────────────────────────────────────────────────────────────
# 32 configs: (bits, rerank, ann, fast_mode, quantizer_type)
# quantizer_type=None  → dense / Haar QR (default)
# quantizer_type="srht"→ SRHT (faster ingest at high-d)
FULL_CONFIGS: list[tuple[int, bool, bool, bool, str | None]] = [
    (bits, rerank, ann, fast_mode, qtype)
    for bits      in [2, 4]
    for rerank    in [False, True]
    for ann       in [False, True]
    for fast_mode in [True, False]
    for qtype     in [None, "srht"]
]


def config_label(bits: int, rerank: bool, ann: bool,
                 fast_mode: bool, qtype: str | None) -> str:
    return (
        f"b={bits}"
        f" {'rerank=T' if rerank else 'rerank=F'}"
        f" {'ANN' if ann else 'brute'}"
        f" {'fast=T' if fast_mode else 'fast=F'}"
        f" {'srht' if qtype == 'srht' else 'dense'}"
    )


# ── Single-config runner ───────────────────────────────────────────────────────

def run_one_config(
    vecs: np.ndarray,
    qvecs: np.ndarray,
    true_top1: np.ndarray,
    bits: int,
    rerank: bool,
    ann: bool,
    fast_mode: bool,
    quantizer_type: str | None,
) -> dict:
    """Run one config on a dataset and return all metrics."""
    import tqdb  # noqa: PLC0415

    N, DIM = vecs.shape
    ids    = [str(i) for i in range(N)]
    n_q    = len(qvecs)
    max_k  = K_VALUES[-1]  # 64

    with tempfile.TemporaryDirectory(prefix="fcb_") as tmp:
        # ── Ingest ────────────────────────────────────────────────────────────
        sampler_ingest = CpuRamSampler()
        sampler_ingest.start()
        t0 = time.perf_counter()
        db = tqdb.Database.open(
            tmp, dimension=DIM, bits=bits, metric="ip",
            rerank=rerank, fast_mode=fast_mode,
            quantizer_type=quantizer_type,
        )
        for start in range(0, N, 2000):
            db.insert_batch(ids[start:start + 2000], vecs[start:start + 2000])
        db.flush()
        ingest_s = time.perf_counter() - t0
        sampler_ingest.stop()

        # Close to flush pre-allocated capacity; measure steady-state disk
        db.close()
        dm = disk_size_mb(tmp)
        db = tqdb.Database.open(
            tmp, dimension=DIM, bits=bits, metric="ip",
            rerank=rerank, fast_mode=fast_mode,
            quantizer_type=quantizer_type,
        )

        # ── ANN index build ───────────────────────────────────────────────────
        index_s = None
        if ann:
            t_idx = time.perf_counter()
            db.create_index(max_degree=32, search_list_size=128, alpha=1.2)
            index_s = round(time.perf_counter() - t_idx, 3)
            dm = disk_size_mb(tmp)  # re-measure with graph.bin included

        # ── Query phase ───────────────────────────────────────────────────────
        # Warmup: page mmap into OS cache before timing
        for q in qvecs[:min(20, n_q)]:
            db.search(q, top_k=max_k, _use_ann=ann)

        sampler_query = CpuRamSampler()
        sampler_query.start()
        lats: list[float] = []
        all_returned: list[list[str]] = []
        for q in qvecs:
            t1  = time.perf_counter()
            res = db.search(q, top_k=max_k, _use_ann=ann)
            lats.append((time.perf_counter() - t1) * 1000)
            all_returned.append([r["id"] for r in res])
        sampler_query.stop()

    recalls = compute_recalls(all_returned, true_top1)
    mrr     = compute_mrr(all_returned, true_top1)
    lats_s  = sorted(lats)

    return {
        "label":              config_label(bits, rerank, ann, fast_mode, quantizer_type),
        "bits":               bits,
        "rerank":             rerank,
        "ann":                ann,
        "fast_mode":          fast_mode,
        "quantizer_type":     quantizer_type,
        "n":                  N,
        "dim":                DIM,
        "n_queries":          n_q,
        "throughput_vps":     round(N / ingest_s),
        "ingest_s":           round(ingest_s, 3),
        "index_s":            index_s,
        "disk_mb":            round(dm, 2),
        "ram_delta_mb":       round(sampler_ingest.delta_ram_mb, 1),
        "ram_ingest_peak_mb": round(sampler_ingest.peak_ram_mb, 1),
        "ram_query_peak_mb":  round(sampler_query.peak_ram_mb, 1),
        "cpu_ingest_pct":     round(sampler_ingest.avg_cpu_pct, 1),
        "cpu_query_pct":      round(sampler_query.avg_cpu_pct, 1),
        "p50_ms":             round(lats_s[int(n_q * 0.50)], 2),
        "p95_ms":             round(lats_s[int(n_q * 0.95)], 2),
        "p99_ms":             round(lats_s[min(int(n_q * 0.99), n_q - 1)], 2),
        "mrr":                round(mrr, 4),
        "recall":             {str(k): round(v, 4) for k, v in recalls.items()},
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test() -> None:
    """Validate all 32 configs with synthetic data (N=2000, d=200, queries=50)."""
    import tqdb  # noqa: PLC0415

    N, DIM, N_Q = 2_000, 200, 50
    rng  = np.random.default_rng(42)
    vecs  = rng.standard_normal((N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    qvecs = rng.standard_normal((N_Q, DIM)).astype(np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
    ids   = [str(i) for i in range(N)]

    print(f"\nSmoke test — {len(FULL_CONFIGS)} configs, N={N}, d={DIM}, queries={N_Q}", flush=True)
    print("=" * 74, flush=True)

    failures: list[tuple[str, str]] = []
    for cfg in FULL_CONFIGS:
        bits, rerank, ann, fast_mode, qtype = cfg
        label = config_label(*cfg)
        try:
            with tempfile.TemporaryDirectory(prefix="smoke_") as tmp:
                db = tqdb.Database.open(
                    tmp, dimension=DIM, bits=bits, metric="ip",
                    rerank=rerank, fast_mode=fast_mode,
                    quantizer_type=qtype,
                )
                db.insert_batch(ids, vecs)
                db.flush()
                if ann:
                    db.create_index(max_degree=16, search_list_size=64, alpha=1.2)
                for q in qvecs[:5]:
                    res = db.search(q, top_k=10, _use_ann=ann)
                    assert isinstance(res, list) and len(res) > 0, "empty results"
                db.close()
            print(f"  PASS  {label}", flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            print(f"  FAIL  {label}  —  {msg}", flush=True)
            failures.append((label, msg))

    print("=" * 74, flush=True)
    if failures:
        print(f"\n{len(failures)} config(s) FAILED — fix before running --full\n", flush=True)
        sys.exit(1)
    else:
        print(f"\nAll {len(FULL_CONFIGS)} configs passed. Ready for --full.\n", flush=True)


# ── Full benchmark ─────────────────────────────────────────────────────────────

DATASET_LOADERS: dict[str, tuple[str, callable]] = {
    "glove":      ("glove-200",    load_glove),
    "arxiv768":   ("arxiv-768",    load_arxiv768),
    "dbpedia1536":("dbpedia-1536", lambda: load_dbpedia(1536)),
    "dbpedia3072":("dbpedia-3072", lambda: load_dbpedia(3072)),
}


def run_full(dataset_keys: list[str], skip_ann: bool,
             rerank_only: bool = False) -> dict:
    """Run configs for the selected datasets.

    rerank_only=True: only run rerank=True configs and merge into the existing
    results JSON (used after fixing the rerank API to patch stale results).
    """
    configs = [c for c in FULL_CONFIGS if (not c[2]) or (not skip_ann)]
    if rerank_only:
        configs = [c for c in configs if c[1]]  # c[1] = rerank flag

    # Seed from existing results so rerank_only can merge into them.
    all_results: dict[str, list[dict]] = {}
    if RESULTS_PATH.exists():
        try:
            all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    for ds_key in dataset_keys:
        ds_label, loader = DATASET_LOADERS[ds_key]
        print(f"\n{'='*74}", flush=True)
        print(f"  {ds_label.upper()}", flush=True)
        print(f"  {len(configs)} configs {'(rerank=T only)' if rerank_only else ''}", flush=True)
        print(f"{'='*74}", flush=True)

        vecs, qvecs, true_top1 = loader()

        # For rerank_only, start from existing results and replace only the
        # rerank=True entries; keep all rerank=False entries unchanged.
        if rerank_only and ds_label in all_results:
            existing = {r["label"]: r for r in all_results[ds_label]
                        if not r.get("rerank")}
        else:
            existing = {}

        new_results: list[dict] = []
        for i, cfg in enumerate(configs):
            bits, rerank, ann, fast_mode, qtype = cfg
            label = config_label(*cfg)
            print(f"\n  [{i+1:02d}/{len(configs)}] {label}", flush=True)

            t_cfg = time.perf_counter()
            try:
                r = run_one_config(vecs, qvecs, true_top1,
                                   bits, rerank, ann, fast_mode, qtype)
                elapsed = time.perf_counter() - t_cfg
                r1 = r["recall"]["1"]
                print(
                    f"    R@1={r1:.3f}  p50={r['p50_ms']:.1f}ms"
                    f"  disk={r['disk_mb']:.1f}MB"
                    f"  ingest={r['throughput_vps']:,}vps"
                    f"  ({elapsed:.0f}s)",
                    flush=True,
                )
                new_results.append(r)
            except Exception as exc:  # noqa: BLE001
                print(f"    ERROR: {exc}", flush=True)
                new_results.append({
                    "label": label, "bits": bits, "rerank": rerank,
                    "ann": ann, "fast_mode": fast_mode,
                    "quantizer_type": qtype, "error": str(exc),
                })

        if rerank_only:
            # Merge: non-rerank entries from existing + freshly run rerank entries
            merged = list(existing.values()) + new_results
            # Restore original ordering (FULL_CONFIGS order)
            order = {config_label(*c): i for i, c in enumerate(FULL_CONFIGS)}
            merged.sort(key=lambda r: order.get(r["label"], 999))
            all_results[ds_label] = merged
        else:
            all_results[ds_label] = new_results

        RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n  Saved results to {RESULTS_PATH}", flush=True)

        plot_recall(ds_label, all_results[ds_label])
        plot_tradeoffs(ds_label, all_results[ds_label])

    return all_results


# ── Plots ─────────────────────────────────────────────────────────────────────

# 4 lines per subplot: (rerank, quantizer_type) → (color, linestyle, marker, label)
_LINE_STYLE: dict[tuple, tuple] = {
    (False, None):   ("#1f77b4", "-",  "o", "no-rerank, dense"),
    (False, "srht"): ("#17becf", "--", "o", "no-rerank, srht"),
    (True,  None):   ("#d62728", "-",  "s", "rerank,    dense"),
    (True,  "srht"): ("#ff7f0e", "--", "s", "rerank,    srht"),
}


def _plot_out(ds_label: str, suffix: str) -> Path:
    slug = ds_label.replace("-", "").replace(" ", "_")
    return BENCH_DIR / f"_full_config_{slug}_{suffix}.png"


def plot_recall(ds_label: str, results: list[dict]) -> None:
    """4×2 recall-curve grid: rows=bits×fast_mode, cols=brute/ANN.
    Each subplot has 4 ranked curves (rerank × quantizer_type) with an in-plot legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bench_core import PAPER_RECALL
    except ImportError:
        print("  matplotlib not available — skipping recall plot", flush=True)
        return

    paper = PAPER_RECALL.get(ds_label, {})
    ok    = [r for r in results if "error" not in r]
    if not ok:
        return

    dim = ok[0].get("dim", "?")
    n   = ok[0].get("n", 0)
    nq  = ok[0].get("n_queries", 0)

    ROW_CFG = [(2, True), (2, False), (4, True), (4, False)]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharey=True)
    fig.suptitle(
        f"{ds_label}  (d={dim}, N={n:,}, {nq:,} queries) — Recall Curves",
        fontsize=13, fontweight="bold",
    )

    for row, (bits, fm) in enumerate(ROW_CFG):
        for col, (ann_flag, col_title) in enumerate(
            [(False, "Brute-force"), (True, "ANN (HNSW)")]
        ):
            ax = axes[row][col]
            subset = [
                r for r in ok
                if r["bits"] == bits and r["fast_mode"] == fm and r["ann"] == ann_flag
            ]

            # For brute-force: rerank is a no-op (exhaustive scan already covers all
            # candidates), so rerank=T ≡ rerank=F. Show only rerank=F to avoid
            # overlapping identical lines; annotate the subplot instead.
            rerank_filter = (lambda r: not r["rerank"]) if not ann_flag else (lambda r: True)

            # collect, sort by R@1 descending
            lines = []
            for r in subset:
                if not rerank_filter(r):
                    continue
                key = (r["rerank"], r["quantizer_type"])
                sty = _LINE_STYLE.get(key)
                if sty is None:
                    continue
                color, ls, mk, lbl = sty
                ys = [r["recall"][str(k)] for k in K_VALUES]
                lines.append((ys[0], ys, color, ls, mk, lbl))
            lines.sort(key=lambda x: x[0], reverse=True)

            handles, leg_labels = [], []
            for _, ys, color, ls, mk, lbl in lines:
                h, = ax.plot(
                    K_VALUES, ys, color=color, linestyle=ls, linewidth=2.2,
                    marker=mk, markersize=5, markeredgewidth=1.1, alpha=0.92,
                )
                handles.append(h)
                leg_labels.append(f"{lbl}  (R@1={ys[0]:.3f})")

            # paper reference
            if bits in paper:
                ys_p = [paper[bits][k] for k in K_VALUES]
                h_p, = ax.plot(
                    K_VALUES, ys_p, color="#888", linestyle="-.",
                    linewidth=1.5, marker="x", markersize=5, alpha=0.7,
                )
                handles.append(h_p)
                leg_labels.append(f"paper ref  (R@1={ys_p[0]:.3f})")

            ax.legend(handles, leg_labels, loc="lower right", fontsize=7.8,
                      framealpha=0.88, edgecolor="#ccc",
                      handlelength=2.2, labelspacing=0.35)

            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES)
            ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=8)
            ax.set_xlabel("top-k", fontsize=9)
            ax.set_ylabel("Recall@1@k", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.margins(y=0.08)
            fm_str = "fast=T (MSE-only)" if fm else "fast=F (MSE+QJL)"
            ax.set_title(f"{bits}-bit  |  {fm_str}  |  {col_title}",
                         fontsize=10, fontweight="bold")
            if not ann_flag:
                ax.text(0.98, 0.02, "rerank has no effect on brute-force",
                        transform=ax.transAxes, fontsize=6.5, color="#888",
                        ha="right", va="bottom", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = _plot_out(ds_label, "recall")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recall plot → {out}", flush=True)


def plot_tradeoffs(ds_label: str, results: list[dict]) -> None:
    """One PNG per (dataset × bits). Each PNG is a 3×4 grid of 11 metric-pair scatter panels.
    Each dot is a config; color is unique per config (tab20 palette).
    An 'ideal direction' arrow is drawn in each subplot corner."""
    try:
        import math as _math
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        print("  matplotlib not available — skipping trade-off plots", flush=True)
        return

    ok = [r for r in results if "error" not in r and "throughput_vps" in r]
    if not ok:
        return

    dim = ok[0].get("dim", "?")
    n   = ok[0].get("n", 0)

    # (x_key, y_key, subplot_title, x_label, y_label, x_better, y_better)
    # y_key "R@1" reads from recall["1"]; x/y_better: "high" or "low"
    SCATTER_PANELS = [
        ("p50_ms",             "R@1",    "Recall@1 vs p50 latency",       "p50 latency (ms)",        "Recall@1",      "low",  "high"),
        ("disk_mb",            "R@1",    "Recall@1 vs Disk",              "Disk (MB)",               "Recall@1",      "low",  "high"),
        ("throughput_vps",     "R@1",    "Recall@1 vs Ingest throughput", "Ingest throughput (vps)", "Recall@1",      "high", "high"),
        ("ingest_s",           "R@1",    "Recall@1 vs Ingest time",       "Ingest time (s)",         "Recall@1",      "low",  "high"),
        ("ram_ingest_peak_mb", "R@1",    "Recall@1 vs RAM ingest peak",   "RAM ingest peak (MB)",    "Recall@1",      "low",  "high"),
        ("ram_query_peak_mb",  "R@1",    "Recall@1 vs RAM query peak",    "RAM query peak (MB)",     "Recall@1",      "low",  "high"),
        ("cpu_ingest_pct",     "R@1",    "Recall@1 vs CPU ingest",        "CPU ingest (%)",          "Recall@1",      "low",  "high"),
        ("cpu_query_pct",      "R@1",    "Recall@1 vs CPU query",         "CPU query (%)",           "Recall@1",      "low",  "high"),
        ("mrr",                "R@1",    "Recall@1 vs MRR",               "MRR",                     "Recall@1",      "high", "high"),
        ("p50_ms",             "disk_mb","Disk vs p50 latency",           "p50 latency (ms)",        "Disk (MB)",     "low",  "low"),
        ("cpu_query_pct",      "p50_ms", "p50 latency vs CPU query",      "CPU query (%)",           "p50 latency (ms)", "low", "low"),
    ]

    # Color per (rerank, fast_mode, qtype) — 8 combos; marker encodes brute/ANN
    sub_configs = sorted(
        {(r["rerank"], r["fast_mode"], r.get("quantizer_type")) for r in ok},
        key=lambda c: (c[0], c[1], c[2] or ""),
    )
    palette   = plt.cm.tab10.colors
    cfg_color = {c: palette[i % len(palette)] for i, c in enumerate(sub_configs)}

    def _sub_key(r: dict) -> tuple:
        return (r["rerank"], r["fast_mode"], r.get("quantizer_type"))

    def _sub_label(c: tuple) -> str:
        rerank, fm, qt = c
        return (
            f"{'rerank=T' if rerank else 'rerank=F'}  "
            f"{'fast=T' if fm else 'fast=F'}  "
            f"{qt or 'dense'}"
        )

    def _add_ideal_corner(ax: "plt.Axes", x_better: str, y_better: str) -> None:
        """Shade a right-triangle in the ideal corner using axes coordinates."""
        from matplotlib.patches import Polygon as _Poly
        import numpy as _np
        cx  = 1.0  if x_better == "high" else 0.0
        cy  = 1.0  if y_better == "high" else 0.0
        off = 0.26
        dx  = -off if x_better == "high" else off
        dy  = -off if y_better == "high" else off
        tri = _np.array([[cx, cy], [cx + dx, cy], [cx, cy + dy]])
        ax.add_patch(_Poly(tri, transform=ax.transAxes, closed=True,
                           facecolor="#66bb6a", alpha=0.22, edgecolor="none", zorder=0))
        ax.text(cx + dx * 0.42, cy + dy * 0.42, "ideal",
                transform=ax.transAxes, fontsize=6, color="#2e7d32",
                ha="center", va="center", fontweight="bold", zorder=1)

    # Layout: 3×4 grid — 11 scatter panels + cell[11] holds the legend
    NCOLS = 4
    NROWS = 3   # ceil(11/4) = 3; 12th cell = legend

    for bits in [2, 4]:
        subset = [r for r in ok if r["bits"] == bits]
        if not subset:
            continue

        bits_label = f"{bits}-bit"
        fig, axes  = plt.subplots(NROWS, NCOLS, figsize=(14, 10))
        axes_flat  = axes.flatten()

        fig.suptitle(
            f"{ds_label}  —  {bits_label} configs  (d={dim}, N={n:,})  —  Trade-off Analysis\n"
            "○ = brute-force   △ = ANN (HNSW)   color = config",
            fontsize=10, fontweight="bold",
        )

        for i, (xkey, ykey, title, xlabel, ylabel, x_better, y_better) in enumerate(SCATTER_PANELS):
            ax = axes_flat[i]
            ax.set_title(title, fontsize=8.5, fontweight="bold")

            for r in subset:
                xv    = r.get(xkey)
                yv    = r["recall"].get("1") if ykey == "R@1" else r.get(ykey)
                if xv is None or yv is None:
                    continue
                color  = cfg_color.get(_sub_key(r), "#888")
                marker = "^" if r["ann"] else "o"
                ax.scatter(xv, yv, c=[color], marker=marker, s=60, alpha=0.88,
                           zorder=3, edgecolors="#333", linewidths=0.4)

            _add_ideal_corner(ax, x_better, y_better)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7.5)

        # ── Legend in cell 11 (no axes, just the legend box) ─────────────────
        legend_ax = axes_flat[len(SCATTER_PANELS)]
        legend_ax.axis("off")

        handles = [
            mlines.Line2D([], [], color=cfg_color[c], marker="o", linewidth=0,
                          markersize=7, label=_sub_label(c))
            for c in sub_configs
        ] + [
            mlines.Line2D([], [], color="#555", marker="o", linewidth=0,
                          markersize=7, label="brute-force"),
            mlines.Line2D([], [], color="#555", marker="^", linewidth=0,
                          markersize=7, label="ANN (HNSW)"),
        ]
        legend_ax.legend(
            handles, [h.get_label() for h in handles],
            loc="center", fontsize=7.5, frameon=True, framealpha=0.92,
            title=f"{bits_label}  |  color=config  shape=search",
            title_fontsize=7.5, ncol=1, borderpad=0.8,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = _plot_out(ds_label, f"tradeoffs_{bits}bit")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Trade-off plot ({bits_label}) → {out}", flush=True)


# ── Report generation ──────────────────────────────────────────────────────────

def _r1(r: dict) -> float:
    return r.get("recall", {}).get("1", -1.0)


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def generate_report(all_results: dict) -> str:
    lines: list[str] = [
        "# TurboQuantDB Full Config Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "Config matrix: bits∈{2,4} × rerank∈{F,T} × ann∈{F,T} "
        "× fast_mode∈{T,F} × quantizer_type∈{dense,srht}  = 32 configs",
        "",
        "Datasets: " + ", ".join(all_results.keys()),
        "",
    ]

    # ── 1. Best configs per dataset ──────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 1. Best Configs by Recall@1 (top 5 per dataset)",
        "",
    ]
    for ds_label, results in all_results.items():
        ok = [r for r in results if "error" not in r]
        top5 = sorted(ok, key=_r1, reverse=True)[:5]
        lines += [f"### {ds_label}", ""]
        lines += [
            f"| Rank | Config | R@1 | p50ms | Disk MB | RAM ingest MB | Ingest vps |",
            f"|------|--------|-----|-------|---------|---------------|------------|",
        ]
        for rank, r in enumerate(top5, 1):
            lines.append(
                f"| {rank} | {r['label']} "
                f"| {_fmt(_r1(r))} "
                f"| {_fmt(r.get('p50_ms'), 1)} "
                f"| {_fmt(r.get('disk_mb'), 1)} "
                f"| {_fmt(r.get('ram_ingest_peak_mb'), 0)} "
                f"| {r.get('throughput_vps', '—'):,} |"
            )
        lines.append("")

    # ── 2. Full brute-force recall vs disk matrix ─────────────────────────────
    lines += [
        "---",
        "",
        "## 2. Brute-Force Recall@1 vs Disk — All Configs",
        "",
        "_Brute-force configs only (ann=F). ANN table in Section 3._",
        "",
    ]
    for ds_label, results in all_results.items():
        brute = [r for r in results if "error" not in r and not r.get("ann")]
        if not brute:
            continue
        lines += [f"### {ds_label}", ""]
        lines += [
            "| Config | R@1 | R@4 | R@8 | p50ms | p99ms | Disk MB | RAM MB | Ingest vps |",
            "|--------|-----|-----|-----|-------|-------|---------|--------|------------|",
        ]
        for r in sorted(brute, key=lambda x: (x["bits"], x["rerank"], x["fast_mode"], x.get("quantizer_type") or "")):
            rec = r.get("recall", {})
            lines.append(
                f"| {r['label']} "
                f"| {_fmt(rec.get('1'))} "
                f"| {_fmt(rec.get('4'))} "
                f"| {_fmt(rec.get('8'))} "
                f"| {_fmt(r.get('p50_ms'), 1)} "
                f"| {_fmt(r.get('p99_ms'), 1)} "
                f"| {_fmt(r.get('disk_mb'), 1)} "
                f"| {_fmt(r.get('ram_ingest_peak_mb'), 0)} "
                f"| {r.get('throughput_vps', '—'):,} |"
            )
        lines.append("")

    # ── 3. ANN recall vs disk matrix ─────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 3. ANN Recall@1 vs Disk — All Configs",
        "",
        "_ANN configs only (ann=T, max_degree=32, ef=128)._",
        "",
    ]
    for ds_label, results in all_results.items():
        ann_res = [r for r in results if "error" not in r and r.get("ann")]
        if not ann_res:
            lines += [f"### {ds_label}", "", "_No ANN results._", ""]
            continue
        lines += [f"### {ds_label}", ""]
        lines += [
            "| Config | R@1 | R@4 | R@8 | p50ms | p99ms | Disk MB | Index s |",
            "|--------|-----|-----|-----|-------|-------|---------|---------|",
        ]
        for r in sorted(ann_res, key=lambda x: (x["bits"], x["rerank"], x["fast_mode"], x.get("quantizer_type") or "")):
            rec = r.get("recall", {})
            lines.append(
                f"| {r['label']} "
                f"| {_fmt(rec.get('1'))} "
                f"| {_fmt(rec.get('4'))} "
                f"| {_fmt(rec.get('8'))} "
                f"| {_fmt(r.get('p50_ms'), 1)} "
                f"| {_fmt(r.get('p99_ms'), 1)} "
                f"| {_fmt(r.get('disk_mb'), 1)} "
                f"| {_fmt(r.get('index_s'), 1)} |"
            )
        lines.append("")

    # ── 4. fast_mode=T vs fast_mode=F (paired diff) ───────────────────────────
    lines += [
        "---",
        "",
        "## 4. fast_mode=True vs fast_mode=False",
        "",
        "_Paired comparison at b=4, rerank=F, brute, dense._",
        "_fast_mode=False stores QJL residuals; helps at d≥1536, hurts at d<512._",
        "",
        "| Dataset | fast_mode | R@1 | p50ms | Disk MB | Ingest vps |",
        "|---------|-----------|-----|-------|---------|------------|",
    ]
    for ds_label, results in all_results.items():
        for fm in [True, False]:
            match = [
                r for r in results
                if "error" not in r
                and r.get("bits") == 4
                and r.get("rerank") is False
                and r.get("ann") is False
                and r.get("fast_mode") is fm
                and r.get("quantizer_type") is None
            ]
            if match:
                r = match[0]
                rec = r.get("recall", {})
                lines.append(
                    f"| {ds_label} "
                    f"| {'True' if fm else 'False'} "
                    f"| {_fmt(rec.get('1'))} "
                    f"| {_fmt(r.get('p50_ms'), 1)} "
                    f"| {_fmt(r.get('disk_mb'), 1)} "
                    f"| {r.get('throughput_vps', '—'):,} |"
                )
    lines.append("")

    # ── 5. dense vs srht (paired diff) ────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. quantizer_type: dense vs srht",
        "",
        "_Paired comparison at b=4, rerank=T, brute, fast_mode=T._",
        "_srht uses SRHT rotation (faster ingest at high-d); dense uses Haar QR._",
        "",
        "| Dataset | quantizer | R@1 | p50ms | Disk MB | Ingest vps |",
        "|---------|-----------|-----|-------|---------|------------|",
    ]
    for ds_label, results in all_results.items():
        for qtype in [None, "srht"]:
            match = [
                r for r in results
                if "error" not in r
                and r.get("bits") == 4
                and r.get("rerank") is True
                and r.get("ann") is False
                and r.get("fast_mode") is True
                and r.get("quantizer_type") == qtype
            ]
            if match:
                r = match[0]
                rec = r.get("recall", {})
                lines.append(
                    f"| {ds_label} "
                    f"| {'dense' if qtype is None else 'srht'} "
                    f"| {_fmt(rec.get('1'))} "
                    f"| {_fmt(r.get('p50_ms'), 1)} "
                    f"| {_fmt(r.get('disk_mb'), 1)} "
                    f"| {r.get('throughput_vps', '—'):,} |"
                )
    lines.append("")

    # ── 6. rerank=T overhead ──────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 6. rerank=False vs rerank=True",
        "",
        "_Paired at b=4, brute, fast_mode=T, dense._",
        "_rerank=True stores float32 alongside compressed codes for second-pass rescoring._",
        "",
        "| Dataset | rerank | R@1 | p50ms | Disk MB | RAM ingest MB |",
        "|---------|--------|-----|-------|---------|---------------|",
    ]
    for ds_label, results in all_results.items():
        for rr in [False, True]:
            match = [
                r for r in results
                if "error" not in r
                and r.get("bits") == 4
                and r.get("rerank") is rr
                and r.get("ann") is False
                and r.get("fast_mode") is True
                and r.get("quantizer_type") is None
            ]
            if match:
                r = match[0]
                rec = r.get("recall", {})
                lines.append(
                    f"| {ds_label} "
                    f"| {'True' if rr else 'False'} "
                    f"| {_fmt(rec.get('1'))} "
                    f"| {_fmt(r.get('p50_ms'), 1)} "
                    f"| {_fmt(r.get('disk_mb'), 1)} "
                    f"| {_fmt(r.get('ram_ingest_peak_mb'), 0)} |"
                )
    lines.append("")

    # ── 7. User guidance ──────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 7. User Guidance",
        "",
        "### Choosing bits",
        "",
        "- **bits=4** is almost always the right choice. It delivers significantly better",
        "  R@1 than bits=2 at only 2× the compressed storage (still 8–16× smaller than",
        "  float32). Use bits=2 only when storage is extremely constrained.",
        "",
        "### fast_mode",
        "",
        "- **fast_mode=True** (default) — MSE-only quantization. Best for all d<1536.",
        "  At d<512 the 1-bit QJL projections are too noisy; fast_mode=False reduces recall.",
        "- **fast_mode=False** — Adds 1-bit QJL residual codes. Beneficial only at d≥1536",
        "  where enough projection bits accumulate to add signal.",
        "  Set `fast_mode=False` only when d≥1536 AND you are not storing float32 (rerank=False).",
        "",
        "### rerank",
        "",
        "- **rerank=False** — stores compressed codes only. Best disk / RAM efficiency.",
        "- **rerank=True** — stores float32 vectors alongside codes for second-pass rescoring.",
        "  Significantly improves R@1, especially at b=2. Costs roughly 4× more disk+RAM",
        "  than rerank=False (float32 dominates storage). Recommended when recall is critical",
        "  and storage is not the bottleneck.",
        "",
        "### ANN vs brute-force",
        "",
        "- **brute-force** scales as O(N) per query. Suitable for N≤500k and/or p50<100ms.",
        "- **ANN** (HNSW) reduces query latency dramatically at N≥100k, at a small recall cost.",
        "  Use `create_index()` when sub-10ms p50 matters or N>500k.",
        "",
        "### quantizer_type",
        "",
        "- **dense** (default, quantizer_type=None) — Haar QR rotation. Slightly better recall",
        "  at all dimensions due to more structured rotation. Preferred for most use cases.",
        "- **srht** — SRHT (Walsh-Hadamard) rotation. O(d log d) vs O(d²) for QR, making",
        "  ingest faster at d≥1536. Use when insert throughput matters more than marginal recall.",
        "",
        "### Recommended presets",
        "",
        "| Use case | Config |",
        "|----------|--------|",
        "| Best recall, d≥1536 | `bits=4, rerank=True, fast_mode=False, dense, brute` |",
        "| Best recall, d<1536 | `bits=4, rerank=True, fast_mode=True, dense, brute` |",
        "| Balanced (default) | `bits=4, rerank=True, fast_mode=True, dense, brute` |",
        "| Min disk | `bits=2, rerank=False, fast_mode=True, dense, brute` |",
        "| Fast ingest, high-d | `bits=4, rerank=False, fast_mode=True, srht, brute` |",
        "| Low latency, large N | `bits=4, rerank=True, fast_mode=True, dense, ANN` |",
        "",
    ]

    # ── 8. Errors summary ─────────────────────────────────────────────────────
    errors = [
        (ds, r["label"], r["error"])
        for ds, results in all_results.items()
        for r in results
        if "error" in r
    ]
    if errors:
        lines += [
            "---",
            "",
            "## 8. Errors",
            "",
            "| Dataset | Config | Error |",
            "|---------|--------|-------|",
        ]
        for ds, label, err in errors:
            lines.append(f"| {ds} | {label} | {err[:120]} |")
        lines.append("")

    # ── 9. Metrics formula reference ─────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 9. Metrics Formula Reference",
        "",
        "| Metric | Formula | Unit | What it measures |",
        "|--------|---------|------|-----------------|",
        "| Recall@1@k | |{q : nn(q) ∈ top-k(q)}| / Q | [0, 1] | Fraction of queries whose true nearest neighbour appears in the top-k results |",
        "| MRR | (1/Q) Σ_q 1/rank_q(nn(q)) | [0, 1] | Mean reciprocal rank of the true NN across all queries |",
        "| p50 latency | median(t_query) | ms | Median wall-clock time per single query |",
        "| Ingest throughput | N / t_ingest | vec/s | Vectors inserted per second (batch ingest) |",
        "| Ingest time | t_ingest | s | Total wall-clock seconds to ingest N vectors |",
        "| Index time | t_index | s | Wall-clock seconds to build the HNSW graph after ingest |",
        "| Disk | Σ file_sizes(db_dir) | MB | Total on-disk storage footprint of the database |",
        "| RAM ingest peak | max(RSS) − baseline during ingest | MB | Peak resident-set-size delta while inserting vectors |",
        "| RAM query peak | max(RSS) − post-ingest during queries | MB | Peak RSS delta while executing queries |",
        "| CPU ingest | mean(cpu%) during ingest | % | CPU utilisation sampled during batch ingest |",
        "| CPU query | mean(cpu%) during queries | % | CPU utilisation sampled during query phase |",
        "",
    ]

    # ── 10. Scenario → config recommendation matrix ───────────────────────────
    lines += [
        "---",
        "",
        "## 10. Scenario → Config Recommendation Matrix",
        "",
        "_Best config per scenario, derived from benchmark data._",
        "_Scores are across all available datasets; 'best' = highest-ranked on primary metric._",
        "",
    ]
    lines += _scenario_matrix_md(all_results)

    return "\n".join(lines)


# ── Scenario matrix ────────────────────────────────────────────────────────────

_SCENARIOS: list[dict] = [
    {
        "name": "Max accuracy",
        "desc": "Highest R@1 regardless of cost",
        "sort_key": lambda r: -_r1(r),
        "filter": lambda r: True,
    },
    {
        "name": "Min query latency",
        "desc": "Lowest p50 ms (any recall)",
        "sort_key": lambda r: r.get("p50_ms") or 999,
        "filter": lambda r: True,
    },
    {
        "name": "High recall + low latency",
        "desc": "R@1 ≥ 0.80, then min p50 ms",
        "sort_key": lambda r: r.get("p50_ms") or 999,
        "filter": lambda r: _r1(r) >= 0.80,
    },
    {
        "name": "Min disk",
        "desc": "Smallest storage footprint",
        "sort_key": lambda r: r.get("disk_mb") or 999,
        "filter": lambda r: True,
    },
    {
        "name": "Min disk + acceptable recall",
        "desc": "R@1 ≥ 0.70, then min disk MB",
        "sort_key": lambda r: r.get("disk_mb") or 999,
        "filter": lambda r: _r1(r) >= 0.70,
    },
    {
        "name": "Max ingest throughput",
        "desc": "Fastest vector insertion (vps)",
        "sort_key": lambda r: -(r.get("throughput_vps") or 0),
        "filter": lambda r: True,
    },
    {
        "name": "Min RAM (query)",
        "desc": "Lowest peak RAM during queries",
        "sort_key": lambda r: r.get("ram_query_peak_mb") or 999,
        "filter": lambda r: True,
    },
    {
        "name": "Balanced: best recall/latency trade-off",
        "desc": "Min (norm_latency − norm_recall) on Pareto front",
        "sort_key": None,   # computed separately (Pareto)
        "filter": lambda r: True,
    },
]


def _pareto_best(ok: list[dict]) -> dict | None:
    """Pick the config with the best normalised recall-vs-latency score."""
    candidates = [r for r in ok if r.get("p50_ms") and _r1(r) >= 0]
    if not candidates:
        return None
    max_r1  = max(_r1(r) for r in candidates) or 1.0
    min_lat = min(r["p50_ms"] for r in candidates) or 1e-9
    max_lat = max(r["p50_ms"] for r in candidates) or 1.0
    def _score(r: dict) -> float:
        nr = _r1(r) / max_r1                               # higher = better
        nl = (r["p50_ms"] - min_lat) / (max_lat - min_lat + 1e-9)  # lower = better
        return nr - nl   # maximise
    return max(candidates, key=_score)


def _scenario_matrix_md(all_results: dict) -> list[str]:
    lines: list[str] = []
    datasets = list(all_results.keys())

    # Header
    header = "| Scenario | Description |" + "".join(f" {ds} |" for ds in datasets)
    sep    = "|----------|-------------|" + "".join("--------|" for _ in datasets)
    lines += [header, sep]

    for sc in _SCENARIOS:
        cells: list[str] = []
        for ds_label in datasets:
            ok = [r for r in all_results[ds_label] if "error" not in r]
            ok_f = [r for r in ok if sc["filter"](r)]
            if not ok_f:
                cells.append("—")
                continue
            if sc["sort_key"] is None:
                best = _pareto_best(ok_f)
            else:
                best = min(ok_f, key=sc["sort_key"]) if ok_f else None
            if best is None:
                cells.append("—")
            else:
                cells.append(
                    f"**{best['label']}**<br>"
                    f"R@1={_r1(best):.3f} p50={best.get('p50_ms', 0):.1f}ms"
                    f" disk={best.get('disk_mb', 0):.0f}MB"
                )
        row = f"| {sc['name']} | {sc['desc']} |" + "".join(f" {c} |" for c in cells)
        lines.append(row)

    lines.append("")
    return lines


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Full config × dataset benchmark for TurboQuantDB",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="Validate all 32 configs with synthetic data (fast)")
    mode.add_argument("--full",  action="store_true",
                      help="Run full paper-scale benchmark across all datasets")

    p.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=list(DATASET_LOADERS.keys()),
        help="Datasets to include (default: all 4)",
    )
    p.add_argument("--no-ann", action="store_true",
                   help="Skip ANN configs (halves the number of configs)")
    p.add_argument("--rerank-only", action="store_true",
                   help="Re-run only rerank=True configs and merge into existing results JSON")

    args = p.parse_args()

    if args.smoke:
        run_smoke_test()
        return

    # --full
    rerank_only = getattr(args, "rerank_only", False)
    print(f"\nFull config benchmark", flush=True)
    print(f"Datasets : {args.datasets}", flush=True)
    if rerank_only:
        n_rr = len([c for c in FULL_CONFIGS if c[1]])
        print(f"--rerank-only: re-running {n_rr} rerank=True configs per dataset", flush=True)
    else:
        print(f"Configs  : {len(FULL_CONFIGS)} total"
              f" ({len([c for c in FULL_CONFIGS if not c[2]])} brute"
              f" + {len([c for c in FULL_CONFIGS if c[2]])} ANN)", flush=True)
    if args.no_ann:
        n_skip = len([c for c in FULL_CONFIGS if c[2]])
        print(f"--no-ann : skipping {n_skip} ANN configs", flush=True)
    print(f"Results  → {RESULTS_PATH}", flush=True)
    print(f"Report   → {REPORT_PATH}", flush=True)
    print(flush=True)

    t_total = time.perf_counter()
    all_results = run_full(args.datasets, skip_ann=args.no_ann, rerank_only=rerank_only)

    # Final save
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Generate report
    print(f"\nGenerating report ...", flush=True)  # already flushed
    report = generate_report(all_results)
    REPORT_PATH.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - t_total
    n_ok = sum(
        1 for rs in all_results.values()
        for r in rs if "error" not in r
    )
    n_err = sum(
        1 for rs in all_results.values()
        for r in rs if "error" in r
    )
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)
    print(f"Configs completed: {n_ok}  errors: {n_err}", flush=True)
    print(f"Results : {RESULTS_PATH}", flush=True)
    print(f"Report  : {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
