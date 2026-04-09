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

    fig.update_layout(
        height=_N_ROWS * 280 + 160,
        width=1150,
        template="plotly_white",
        font=dict(size=11),
        hovermode="x unified",
        margin=dict(t=60, b=160, l=55, r=20),
        legend=dict(
            orientation="h",
            xanchor="center", x=0.5,
            y=-0.14, yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc", borderwidth=1,
            tracegroupgap=4,
        ),
    )

    # Prefix with "plot_" so the div ID never collides with the tab pane ID.
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=f"plot_{ds_key.replace('-', '')}",  # e.g. "plot_glove200", "plot_dbpedia1536"
    )


def generate_html_plotly(history: list[dict], out_path: Path) -> None:
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install plotly") from exc

    n = len(history)
    n_axes = _N_ROWS * _N_COLS  # 12

    # Div IDs match the ds_key with "-" stripped: glove200, dbpedia1536, dbpedia3072
    plot_ids = {ds_key: ds_key.replace("-", "") for ds_key, _, _ in DATASETS}

    tab_js = f"""
<script>
var TQDB_N = {n};
var TQDB_N_AXES = {n_axes};

// Switch dataset tab; mark pane with CSS class so setRange can find it reliably.
function showTab(tabId, btn) {{
  document.querySelectorAll('.tab-pane').forEach(function(d){{
    d.style.display = 'none';
    d.classList.remove('active-pane');
  }});
  document.querySelectorAll('.tab-btn').forEach(function(b){{ b.classList.remove('active'); }});
  var pane = document.getElementById(tabId);
  pane.style.display = 'block';
  pane.classList.add('active-pane');
  btn.classList.add('active');
  // Resize the Plotly figure inside the newly-visible tab.
  var gd = pane.querySelector('.plotly-graph-div');
  if (gd && window.Plotly) {{ Plotly.Plots.resize(gd); }}
}}

// Set the x-axis range on all subplots of the currently active figure.
// last=0 means show all commits; last=N means show the most recent N commits.
function setRange(last) {{
  var pane = document.querySelector('.tab-pane.active-pane');
  if (!pane) pane = document.querySelector('.tab-pane');  // fallback: first pane
  if (!pane) return;
  var gd = pane.querySelector('.plotly-graph-div');
  if (!gd || !window.Plotly) return;
  var maxX = TQDB_N - 0.5;
  var minX = (last > 0) ? Math.max(-0.5, TQDB_N - last - 0.5) : -0.5;
  // Deep-copy the live layout (preserves domain, tickvals, etc.) then patch
  // only the range on every x-axis. Plotly.react re-renders efficiently with
  // the full layout object — avoids the partial-update bugs in Plotly.js 3.x.
  var layout = JSON.parse(JSON.stringify(gd.layout));
  Object.keys(layout).forEach(function(key) {{
    if (key === 'xaxis' || /^xaxis\d+$/.test(key)) {{
      layout[key].range = [minX, maxX];
      layout[key].autorange = false;
    }}
  }});
  Plotly.react(gd.id, gd.data, layout);
}}
</script>
"""

    tab_css = """
<style>
body { font-family: sans-serif; margin: 24px; background: #fafafa; }
h1   { color: #222; margin-bottom: 16px; }
.tab-bar  { display: flex; gap: 8px; margin-bottom: 0; }
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
.range-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.range-bar span { font-size: 13px; color: #555; }
.range-btn {
  padding: 4px 14px; border: 1px solid #aac; border-radius: 4px;
  background: #f0f4ff; cursor: pointer; font-size: 12px; color: #334;
}
.range-btn:hover { background: #dde8ff; }
</style>
"""

    parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>TurboQuantDB Performance History</title>",
        tab_css,
        "<script src='https://cdn.plot.ly/plotly-3.4.0.min.js'></script>",
        tab_js,
        "</head><body>",
        f"<h1>TurboQuantDB \u2014 Performance History ({n} entries)</h1>",
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

    # Tab panes (first one pre-marked as active for setRange fallback)
    for i, (ds_key, ds_label, tab_id) in enumerate(DATASETS):
        display = "block" if i == 0 else "none"
        extra_class = " active-pane" if i == 0 else ""
        parts.append(f"<div id='{tab_id}' class='tab-pane{extra_class}' style='display:{display}'>")

        # Range buttons (plain HTML, call setRange() via JS)
        parts.append(
            "<div class='range-bar'>"
            "<span>Show commits:</span>"
            "<button class='range-btn' onclick='setRange(5)'>Last 5</button>"
            "<button class='range-btn' onclick='setRange(10)'>Last 10</button>"
            "<button class='range-btn' onclick='setRange(0)'>All</button>"
            "</div>"
        )

        if any(ds_key in e.get("results", {}) for e in history):
            fig_html = _build_figure(history, ds_key)
            parts.append(fig_html)
        else:
            parts.append(f"<p>No data for {ds_label} yet.</p>")

        parts.append("</div>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  History report saved to {out_path} ({n} entries)", flush=True)


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
