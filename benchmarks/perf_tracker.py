"""
perf_tracker.py — Performance history visualization for TurboQuantDB.

Reads perf_history.json (accumulated by `paper_recall_bench.py --track`)
and generates _perf_history.html with interactive time-series charts.

Usage:
    python benchmarks/perf_tracker.py
    python benchmarks/perf_tracker.py --input path/to/perf_history.json
    python benchmarks/perf_tracker.py --output my_report.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR    = Path(__file__).parent
HISTORY_PATH = BENCH_DIR / "perf_history.json"
OUTPUT_PATH  = BENCH_DIR / "_perf_history.html"

# Tracked metrics: (json_key_suffix, display_label, higher_is_better)
TRACKED_METRICS = [
    ("b4_rerank=T_brute_r1at1",       "b=4 rr=T brute  R@1@1",      True),
    ("b4_rerank=F_brute_r1at1",       "b=4 rr=F brute  R@1@1",      True),
    ("b4_rerank=T_brute_throughput",  "b=4 rr=T brute  Thruput vps", True),
    ("b4_rerank=T_brute_p50_ms",      "b=4 rr=T brute  p50 (ms)",    False),
    ("b4_rerank=T_brute_disk_mb",     "b=4 rr=T brute  Disk (MB)",   False),
    ("b4_rerank=T_brute_mrr",         "b=4 rr=T brute  MRR",         True),
]

DATASETS = ["glove-200", "dbpedia-1536", "dbpedia-3072"]


def _norm_key(raw: str) -> str:
    """Normalise result dict key to match TRACKED_METRICS suffixes."""
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


def _extract(entry: dict, ds: str, suffix: str) -> float | None:
    ds_data = entry.get("results", {}).get(ds, {})
    # Normalise all keys
    norm_map = {_norm_key(k): v for k, v in ds_data.items()}
    return norm_map.get(_norm_key(suffix))


def generate_html_plotly(history: list[dict], out_path: Path) -> None:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415
    except ImportError:
        print("  plotly not installed — falling back to matplotlib HTML", flush=True)
        generate_html_matplotlib(history, out_path)
        return

    n_metrics = len(TRACKED_METRICS)
    n_ds      = len(DATASETS)
    n_rows    = n_metrics
    n_cols    = n_ds

    subplot_titles = [
        f"{ds} — {lbl}"
        for _, lbl, _ in TRACKED_METRICS
        for ds in DATASETS
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    x_labels = [
        f"{e.get('git_commit', 'unknown')[:7]}<br>{e.get('timestamp', '')[:10]}"
        for e in history
    ]

    for mi, (suffix, label, higher_better) in enumerate(TRACKED_METRICS):
        for di, ds in enumerate(DATASETS):
            ys = [_extract(e, ds, suffix) for e in history]
            valid = [(x, y) for x, y in zip(x_labels, ys) if y is not None]
            if not valid:
                continue
            xs_plot, ys_plot = zip(*valid)

            row = mi + 1
            col = di + 1
            color = "rgba(0, 120, 200, 0.9)" if higher_better else "rgba(220, 60, 60, 0.9)"

            fig.add_trace(
                go.Scatter(
                    x=list(xs_plot),
                    y=list(ys_plot),
                    mode="lines+markers",
                    name=f"{ds} {label}",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    showlegend=False,
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title="TurboQuantDB — Performance History",
        height=350 * n_rows,
        width=400 * n_cols,
        template="plotly_white",
        font=dict(size=11),
    )

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    out_path.write_text(html, encoding="utf-8")
    print(f"  History report saved to {out_path} ({len(history)} entries)", flush=True)


def generate_html_matplotlib(history: list[dict], out_path: Path) -> None:
    """Fallback: embed matplotlib PNG into a minimal HTML page."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io, base64
    except ImportError:
        print("  matplotlib not available — cannot generate history report", flush=True)
        return

    n_metrics = len(TRACKED_METRICS)
    n_ds      = len(DATASETS)
    fig, axes = plt.subplots(n_metrics, n_ds, figsize=(6 * n_ds, 3.5 * n_metrics),
                              squeeze=False)
    fig.suptitle("TurboQuantDB — Performance History", fontsize=13, fontweight="bold")

    x_ticks = [e.get("git_commit", "?")[:7] for e in history]

    for mi, (suffix, label, higher_better) in enumerate(TRACKED_METRICS):
        for di, ds in enumerate(DATASETS):
            ax = axes[mi][di]
            ys = [_extract(e, ds, suffix) for e in history]
            valid_idx = [i for i, y in enumerate(ys) if y is not None]
            valid_ys  = [ys[i] for i in valid_idx]
            valid_xs  = [x_ticks[i] for i in valid_idx]

            if valid_ys:
                ax.plot(range(len(valid_ys)), valid_ys,
                        "b-o" if higher_better else "r-o", linewidth=1.5, markersize=5)
                ax.set_xticks(range(len(valid_xs)))
                ax.set_xticklabels(valid_xs, rotation=45, ha="right", fontsize=7)

            ax.set_title(f"{ds}\n{label}", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    html = (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>TurboQuantDB Performance History</title>"
        "</head><body>"
        f"<h1>TurboQuantDB — Performance History ({len(history)} entries)</h1>"
        f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%'/>"
        "</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"  History report (matplotlib) saved to {out_path}", flush=True)


def print_summary(history: list[dict]) -> None:
    if not history:
        print("  No history entries.", flush=True)
        return
    last = history[-1]
    print(f"\n  Latest entry: {last.get('timestamp', '?')[:19]}  "
          f"commit={last.get('git_commit', '?')}  "
          f"version={last.get('version', '?')}", flush=True)
    print(f"  Total entries: {len(history)}", flush=True)
    ds_data = last.get("results", {})
    for ds in DATASETS:
        if ds not in ds_data:
            continue
        norm_map = {_norm_key(k): v for k, v in ds_data[ds].items()}
        r1 = norm_map.get(_norm_key("b4_rerank=T_brute_r1at1"), "—")
        tp = norm_map.get(_norm_key("b4_rerank=T_brute_throughput"), "—")
        p5 = norm_map.get(_norm_key("b4_rerank=T_brute_p50_ms"), "—")
        print(f"  {ds:20s}  R@1@1={r1}  throughput={tp}  p50={p5}ms", flush=True)


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
        print("  No history data found — run `paper_recall_bench.py --track` first.")
        return

    print_summary(history)
    generate_html_plotly(history, Path(args.output))


if __name__ == "__main__":
    main()
