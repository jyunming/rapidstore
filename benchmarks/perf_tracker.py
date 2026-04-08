#!/usr/bin/env python3
"""
perf_tracker.py — Performance history visualization for TurboQuantDB.

Reads perf_history.json and generates _perf_history.html.

Layout:
  - Three tabs: one per dataset (GloVe-200 / DBpedia-1536 / DBpedia-3072)
  - Each tab: 4-col × 3-row metric grid
  - X-axis: integer commit index; date groups shown as colored background bands
    with date label at top of each band; commit hash on tick; full info on hover

Usage:
    python benchmarks/perf_tracker.py
    python benchmarks/perf_tracker.py --input path/to/perf_history.json
    python benchmarks/perf_tracker.py --output my_report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR    = Path(__file__).parent
HISTORY_PATH = BENCH_DIR / "perf_history.json"
OUTPUT_PATH  = BENCH_DIR / "_perf_history.html"

DATASETS = [
    ("glove-200",    "GloVe-200",    "glove"),
    ("dbpedia-1536", "DBpedia-1536", "dbpedia1536"),
    ("dbpedia-3072", "DBpedia-3072", "dbpedia3072"),
]

# (json_key_suffix, display_label, higher_is_better)
METRICS = [
    ("r1at1",              "Recall@1@1",           True),
    ("mrr",                "MRR",                  True),
    ("throughput",         "Throughput (vps)",     True),
    ("p50_ms",             "p50 latency (ms)",     False),
    ("p99_ms",             "p99 latency (ms)",     False),
    ("ingest_s",           "Ingest time (s)",      False),
    ("disk_mb",            "Disk (MB)",            False),
    ("ram_ingest_peak_mb", "RAM ingest peak (MB)", False),
    ("ram_query_peak_mb",  "RAM query peak (MB)",  False),
    ("cpu_ingest_pct",     "CPU ingest (%)",       False),
    ("cpu_query_pct",      "CPU query (%)",        False),
]

# (key_prefix, display_label, hex_color, plotly_dash)
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

# Soft alternating band colors for date groups
_BAND_COLORS = [
    "rgba(100,149,237,0.10)",
    "rgba(60,179,113,0.10)",
    "rgba(255,165,0,0.10)",
    "rgba(186,85,211,0.10)",
    "rgba(220,20,60,0.10)",
]

_N_COLS = 4
_N_ROWS = (len(METRICS) + _N_COLS - 1) // _N_COLS  # 3


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
    val = norm_map.get(_norm_key(f"{cfg_key}_{metric_suffix}"))
    return val if isinstance(val, (int, float)) else None


def _date_groups(history: list[dict]) -> list[tuple[str, int, int]]:
    """Return list of (date_str, start_idx, end_idx) for each run of same date."""
    groups = []
    for date, grp in groupby(enumerate(history), key=lambda x: x[1].get("timestamp", "")[:10]):
        indices = [i for i, _ in grp]
        groups.append((date, indices[0], indices[-1]))
    return groups


def _build_figure(history: list[dict], ds_key: str) -> str:
    """Build and return the Plotly figure HTML for one dataset tab."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("pip install plotly") from exc

    n = len(history)
    x_vals = list(range(n))

    hover_texts = [
        f"{e.get('timestamp','')[:10]}  {e.get('git_commit','?')[:8]}  {e.get('git_branch','')}"
        for e in history
    ]
    ticktext = [e.get("git_commit", "?")[:7] for e in history]
    date_groups = _date_groups(history)

    subplot_titles = [label for _, label, _ in METRICS] + [""] * (_N_COLS * _N_ROWS - len(METRICS))

    fig = make_subplots(
        rows=_N_ROWS,
        cols=_N_COLS,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
    )

    # ── Date band shading (vrects across all subplots) ────────────────────────
    for gi, (date, s, e_idx) in enumerate(date_groups):
        color = _BAND_COLORS[gi % len(_BAND_COLORS)]
        fig.add_vrect(
            x0=s - 0.45, x1=e_idx + 0.45,
            fillcolor=color,
            layer="below",
            line_width=0,
            row="all", col="all",
        )

    # ── Date labels as annotations above the top row ─────────────────────────
    # x in paper coords: map integer index to [0,1] fraction
    x_range = n - 1  # indices 0..n-1
    for gi, (date, s, e_idx) in enumerate(date_groups):
        mid_idx = (s + e_idx) / 2
        # approximate paper x: left margin ~0.05, right ~0.97
        paper_x = 0.05 + (mid_idx / max(x_range, 1)) * 0.92
        fig.add_annotation(
            x=paper_x, y=1.02,
            xref="paper", yref="paper",
            text=f"<b>{date}</b>",
            showarrow=False,
            font=dict(size=10, color="#555"),
            xanchor="center",
        )

    # ── Metric traces ─────────────────────────────────────────────────────────
    for mi, (metric_suffix, _label, _higher) in enumerate(METRICS):
        row = mi // _N_COLS + 1
        col = mi % _N_COLS + 1
        is_first = (mi == 0)

        # Anchor trace — ensures every subplot covers the full x range
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=[None] * n,
                mode="markers",
                marker=dict(opacity=0, size=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        for cfg_key, cfg_label, color, dash in CONFIGS:
            ys: list[float | None] = [
                _get_value(entry, ds_key, cfg_key, metric_suffix)
                for entry in history
            ]
            if all(v is None for v in ys):
                continue

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=ys,
                    mode="lines+markers",
                    name=cfg_label,
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=6),
                    legendgroup=cfg_label,
                    showlegend=is_first,
                    customdata=hover_texts,
                    hovertemplate=(
                        f"<b>{cfg_label}</b>: %{{y:.4g}}<br>"
                        "%{customdata}<extra></extra>"
                    ),
                    connectgaps=False,
                ),
                row=row, col=col,
            )

    # ── Apply uniform x-axis style; default to last 5 commits ───────────────
    default_min = max(-0.5, n - 5 - 0.5)
    fig.update_xaxes(
        tickvals=x_vals,
        ticktext=ticktext,
        tickangle=-45,
        tickfont=dict(size=8),
        range=[default_min, n - 0.5],
        autorange=False,
        showgrid=False,
        zeroline=False,
    )

    # ── Native Plotly range buttons (guaranteed to work) ─────────────────────
    n_axes = _N_ROWS * _N_COLS  # 12 subplots → xaxis … xaxis12
    def _range_args(last: int) -> dict:
        min_x = max(-0.5, n - last - 0.5) if last > 0 else -0.5
        max_x = n - 0.5
        d: dict = {"xaxis.range": [min_x, max_x], "xaxis.autorange": False}
        for ai in range(2, n_axes + 1):
            d[f"xaxis{ai}.range"] = [min_x, max_x]
            d[f"xaxis{ai}.autorange"] = False
        return d

    fig.update_layout(
        height=_N_ROWS * 280 + 160,
        width=1150,
        template="plotly_white",
        font=dict(size=11),
        hovermode="x unified",
        margin=dict(t=80, b=160, l=55, r=20),
        legend=dict(
            orientation="h",
            xanchor="center", x=0.5,
            y=-0.14, yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc", borderwidth=1,
            tracegroupgap=4,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=1.0, y=1.06,
            xanchor="right", yanchor="bottom",
            bgcolor="#f0f4ff",
            bordercolor="#aac",
            font=dict(size=11),
            buttons=[
                dict(label="Last 5",  method="relayout", args=[_range_args(5)]),
                dict(label="Last 10", method="relayout", args=[_range_args(10)]),
                dict(label="All",     method="relayout", args=[_range_args(0)]),
            ],
        )],
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_html_plotly(history: list[dict], out_path: Path) -> None:
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install plotly") from exc

    tab_js = """
<script>
function showTab(tabId, btn) {
  document.querySelectorAll('.tab-pane').forEach(function(d){ d.style.display='none'; });
  document.querySelectorAll('.tab-btn').forEach(function(b){ b.classList.remove('active'); });
  document.getElementById(tabId).style.display = 'block';
  btn.classList.add('active');
}
</script>
"""

    tab_css = """
<style>
body { font-family: sans-serif; margin: 24px; background: #fafafa; }
h1   { color: #222; margin-bottom: 16px; }
.tab-bar  { display: flex; gap: 8px; margin-bottom: 20px; }
.tab-btn  {
  padding: 8px 20px; border: 1px solid #ccc; border-radius: 6px 6px 0 0;
  background: #eee; cursor: pointer; font-size: 14px; font-weight: 500;
  color: #555; transition: background 0.15s;
}
.tab-btn:hover  { background: #dde; }
.tab-btn.active { background: #fff; border-bottom-color: #fff; color: #222;
                  font-weight: 700; box-shadow: 0 -2px 6px rgba(0,0,0,0.06); }
.tab-pane { border: 1px solid #ccc; border-radius: 0 6px 6px 6px;
            background: #fff; padding: 16px; }
</style>
"""

    parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>TurboQuantDB Performance History</title>",
        tab_css,
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        tab_js,
        "</head><body>",
        f"<h1>TurboQuantDB \u2014 Performance History ({len(history)} entries)</h1>",
        "<div class='tab-bar'>",
    ]

    # Tab buttons — first one active by default
    for i, (ds_key, ds_label, tab_id) in enumerate(DATASETS):
        active = " active" if i == 0 else ""
        parts.append(
            f"<button class='tab-btn{active}' "
            f"onclick=\"showTab('{tab_id}', this)\">{ds_label}</button>"
        )
    parts.append("</div>")

    # Tab panes
    for i, (ds_key, ds_label, tab_id) in enumerate(DATASETS):
        display = "block" if i == 0 else "none"
        parts.append(f"<div id='{tab_id}' class='tab-pane' style='display:{display}'>")

        if any(ds_key in e.get("results", {}) for e in history):
            fig_html = _build_figure(history, ds_key)
            parts.append(fig_html)
        else:
            parts.append(f"<p>No data for {ds_label} yet.</p>")

        parts.append("</div>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
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
    for ds_key, ds_label, _ in DATASETS:
        val = _get_value(last, ds_key, "b4_rerankT_brute", "r1at1")
        if val is not None:
            print(f"  {ds_label:20s}  b4_rerankT_brute  R@1@1={val}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise TurboQuantDB performance history")
    parser.add_argument("--input",  default=str(HISTORY_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    history = load_history(Path(args.input))
    if not history:
        print("  No history data — run `paper_recall_bench.py --track` first.")
        return

    print_summary(history)
    generate_html_plotly(history, Path(args.output))


if __name__ == "__main__":
    main()
