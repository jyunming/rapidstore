#!/usr/bin/env python3
"""
perf_tracker.py — Performance history visualization for TurboQuantDB.

Reads perf_history.json (written by paper_recall_bench.py --track and
precommit_perf_check.py) and generates _perf_history.html.

Layout: one section per dataset; within each section one subplot per metric.
Each subplot shows all available configs as time-series lines.
X-axis: commit hash + timestamp.

Usage:
    python benchmarks/perf_tracker.py
    python benchmarks/perf_tracker.py --input path/to/perf_history.json
    python benchmarks/perf_tracker.py --output my_report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR    = Path(__file__).parent
HISTORY_PATH = BENCH_DIR / "perf_history.json"
OUTPUT_PATH  = BENCH_DIR / "_perf_history.html"

DATASETS = ["glove-200", "dbpedia-1536", "dbpedia-3072"]

# Metrics: (json_key_suffix, display_label, higher_is_better)
METRICS = [
    ("r1at1",        "Recall@1@1",      True),
    ("throughput",   "Throughput (vps)", True),
    ("p50_ms",       "p50 latency (ms)", False),
    ("disk_mb",      "Disk (MB)",        False),
    ("ram_delta_mb", "RAM delta (MB)",   False),
    ("mrr",          "MRR",              True),
]

# Configs: (key_prefix, display_label, hex_color, plotly_dash)
CONFIGS = [
    ("b2_rerankF_brute", "b=2 rr=F brute", "#4477AA", "solid"),
    ("b2_rerankT_brute", "b=2 rr=T brute", "#EE6677", "solid"),
    ("b4_rerankF_brute", "b=4 rr=F brute", "#228833", "solid"),
    ("b4_rerankT_brute", "b=4 rr=T brute", "#CCBB44", "solid"),
    ("b2_rerankF_ANN",   "b=2 rr=F ANN",   "#4477AA", "dot"),
    ("b2_rerankT_ANN",   "b=2 rr=T ANN",   "#EE6677", "dot"),
    ("b4_rerankF_ANN",   "b=4 rr=F ANN",   "#228833", "dot"),
    ("b4_rerankT_ANN",   "b=4 rr=T ANN",   "#CCBB44", "dot"),
]

_METRICS_PER_ROW = 3  # 3 cols → 2 rows for 6 metrics


def _norm_key(raw: str) -> str:
    return raw.replace(" ", "_").replace("=", "").replace("-", "_")


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  perf_history.json not found at {path}", flush=True)
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  Failed to read {path}: {exc}", flush=True)
        return []


def _get_value(entry: dict, ds: str, cfg_key: str, metric_suffix: str) -> float | None:
    ds_data = entry.get("results", {}).get(ds, {})
    norm_map = {_norm_key(k): v for k, v in ds_data.items()}
    return norm_map.get(_norm_key(f"{cfg_key}_{metric_suffix}"))


def generate_html_plotly(history: list[dict], out_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for HTML reports. Install it with: pip install plotly"
        ) from exc

    x_labels = [
        f"{e.get('git_commit', '?')[:7]}<br>{e.get('timestamp', '')[:10]}"
        for e in history
    ]

    n_cols = _METRICS_PER_ROW
    n_rows = (len(METRICS) + n_cols - 1) // n_cols

    html_parts = [
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>TurboQuantDB Performance History</title>"
        "<style>"
        "body{font-family:sans-serif;margin:20px;background:#fafafa}"
        "h1{color:#222}"
        "h2{color:#444;margin-top:48px;border-bottom:2px solid #ddd;padding-bottom:8px}"
        "</style>"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
        "</head><body>"
        f"<h1>TurboQuantDB — Performance History ({len(history)} entries)</h1>"
    ]

    for ds in DATASETS:
        subplot_titles = [label for _, label, _ in METRICS]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.18,
            horizontal_spacing=0.08,
        )

        for mi, (metric_suffix, _label, higher_better) in enumerate(METRICS):
            row = mi // n_cols + 1
            col = mi % n_cols + 1

            for cfg_key, cfg_label, color, dash in CONFIGS:
                xs, ys = [], []
                for xi, entry in enumerate(history):
                    val = _get_value(entry, ds, cfg_key, metric_suffix)
                    if val is not None:
                        xs.append(x_labels[xi])
                        ys.append(val)
                if not ys:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=cfg_label,
                        line=dict(color=color, width=2, dash=dash),
                        marker=dict(size=7),
                        legendgroup=cfg_label,
                        showlegend=(mi == 0),
                        hovertemplate=f"<b>{cfg_label}</b>: %{{y:.4g}}<extra></extra>",
                    ),
                    row=row, col=col,
                )

        fig.update_layout(
            height=300 * n_rows + 120,  # extra room for legend below
            width=1100,
            template="plotly_white",
            font=dict(size=11),
            hovermode="x unified",
            margin=dict(t=50, b=150, l=60, r=20),
            legend=dict(
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.18,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                tracegroupgap=4,
            ),
        )

        html_parts.append(f"<h2>{ds}</h2>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  History report saved to {out_path} ({len(history)} entries)", flush=True)


def print_summary(history: list[dict]) -> None:
    if not history:
        print("  No history entries.", flush=True)
        return
    last = history[-1]
    print(
        f"\n  Latest: {last.get('timestamp','?')[:19]}  "
        f"commit={last.get('git_commit','?')}  version={last.get('version','?')}",
        flush=True,
    )
    print(f"  Total entries: {len(history)}", flush=True)
    for ds in DATASETS:
        val = _get_value(last, ds, "b4_rerankT_brute", "r1at1")
        if val is not None:
            print(f"  {ds:20s}  b4_rerankT_brute  R@1@1={val}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise TurboQuantDB performance history"
    )
    parser.add_argument("--input",  default=str(HISTORY_PATH),
                        help=f"Path to perf_history.json (default: {HISTORY_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help=f"Output HTML path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    history = load_history(Path(args.input))
    if not history:
        print("  No history data found — run `paper_recall_bench.py --track` or `precommit_perf_check.py` first.")
        return

    print_summary(history)
    generate_html_plotly(history, Path(args.output))


if __name__ == "__main__":
    main()
