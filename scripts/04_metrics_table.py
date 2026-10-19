"""
04_metrics_table.py
══════════════════════════════════════════════════════════════════════════════
MAE, RMSE, R² for all target PMUTs using the Vector (multi-output) approach
evaluated on Full Spectrum, for two training schemes:

  Progressive  — cumulative [1..N-1] → predict N  (N = 2 → 16)
  Regime-Aware — fresh model per regime boundary
                  Regime 1: PMUTs  1– 2   [1]→2
                  Regime 2: PMUTs  3–11   [3]→4 … [3..10]→11
                  Regime 3: PMUTs 12–16   [12]→13 … [12..15]→16
                  + cross-regime predictions at each boundary

Both use the same CSV schema — only the Train_PMUTs column differs.

Outputs
───────
  results/metrics/progressive_metrics.csv   ─┐  same columns
  results/metrics/regime_metrics.csv         ─┘
  results/metrics/progressive_metrics.html
  results/metrics/regime_metrics.html
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Helpers.cache_utils import load_progressive, load_regime
from Helpers.plot_utils  import MODEL_COLORS

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR         = os.path.join("results", "metrics")
PROG_CACHE      = os.path.join("results", "predictions", "progressive_cache.csv")
REGI_CACHE      = os.path.join("results", "predictions", "regime_cache.csv")
OUT_CSV_PROG    = os.path.join(OUT_DIR, "progressive_metrics.csv")
OUT_CSV_REGIME  = os.path.join(OUT_DIR, "regime_metrics.csv")
OUT_HTML_PROG   = os.path.join(OUT_DIR, "progressive_metrics.html")
OUT_HTML_REGIME = os.path.join(OUT_DIR, "regime_metrics.html")


# ── Chart builder ──────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, scheme_label: str) -> go.Figure:
    targets  = sorted(df["Target_PMUT"].unique())
    x_labels = [f"P{t}" for t in targets]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"MAE  —  {scheme_label}",
            f"RMSE  —  {scheme_label}",
            f"R²  —  {scheme_label}   (negatives clipped; red ✕ = was negative)",
            f"Mean MAE by Regime  —  {scheme_label}",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    models_in_df = [m for m in ["RF", "GB", "MLP", "PINN", "MDOF-PINN", "PINN-Informed-Loss", "RF-MDOF-PINN"] if m in df["Model"].values]
    for m in models_in_df:
        sub   = df[df["Model"] == m].set_index("Target_PMUT")
        color = MODEL_COLORS[m]

        # MAE
        fig.add_trace(go.Bar(
            x=x_labels, y=[sub.loc[t, "MAE"]  for t in targets],
            name=m, marker_color=color, legendgroup=m,
        ), row=1, col=1)

        # RMSE
        fig.add_trace(go.Bar(
            x=x_labels, y=[sub.loc[t, "RMSE"] for t in targets],
            name=m, marker_color=color, legendgroup=m, showlegend=False,
        ), row=1, col=2)

        # R² (clipped)
        r2_raw     = [sub.loc[t, "R2"]  for t in targets]
        r2_clipped = [max(0.0, v)        for v in r2_raw]
        neg_x      = [x_labels[i] for i, v in enumerate(r2_raw) if v < 0]
        neg_tip    = [f"{m} R²={r2_raw[i]:.3f}" for i, v in enumerate(r2_raw) if v < 0]

        fig.add_trace(go.Bar(
            x=x_labels, y=r2_clipped,
            name=m, marker_color=color, legendgroup=m, showlegend=False,
        ), row=2, col=1)
        if neg_x:
            fig.add_trace(go.Scatter(
                x=neg_x, y=[0.01] * len(neg_x),
                mode="markers",
                marker=dict(symbol="x", size=11, color="red"),
                showlegend=False, hovertext=neg_tip, hoverinfo="text+x",
            ), row=2, col=1)

        # Mean MAE by regime
        rm = df.groupby(["Regime", "Model"])["MAE"].mean().reset_index()
        sub_r = rm[rm["Model"] == m]
        fig.add_trace(go.Bar(
            x=sub_r["Regime"].tolist(), y=sub_r["MAE"].tolist(),
            name=m, marker_color=color, legendgroup=m, showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        barmode="group",
        title=(f"MAE / RMSE / R²  —  {scheme_label}"
               f" | Vector (multi-output) | Full Spectrum"),
        height=780, template="plotly_white",
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_yaxes(title_text="MAE",      row=1, col=1)
    fig.update_yaxes(title_text="RMSE",     row=1, col=2)
    fig.update_yaxes(title_text="R²",       row=2, col=1)
    fig.update_yaxes(title_text="Mean MAE", row=2, col=2)
    return fig


# ── HTML writer ────────────────────────────────────────────────────────────────
_SORT_JS = """<script>
function sortTable(n){
  var t=document.querySelector("table"),rows,ok=true,i,x,y,dir="asc",sw;
  while(ok){ok=false;rows=t.rows;
    for(i=1;i<rows.length-1;i++){sw=false;
      x=rows[i].cells[n];y=rows[i+1].cells[n];
      var a=isNaN(x.innerHTML)?x.innerHTML.toLowerCase():parseFloat(x.innerHTML);
      var b=isNaN(y.innerHTML)?y.innerHTML.toLowerCase():parseFloat(y.innerHTML);
      if((dir=="asc"&&a>b)||(dir=="desc"&&a<b)){
        rows[i].parentNode.insertBefore(rows[i+1],rows[i]);sw=true;ok=true;}}
    if(!sw){if(dir=="asc"){dir="desc";ok=true;}}}}
window.onload=function(){
  document.querySelectorAll("th").forEach((th,i)=>{th.onclick=()=>sortTable(i);});};
</script>"""

_STYLE = """<style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:20px}
h1,h2{color:#37474F}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:12px}
th,td{border:1px solid #ddd;padding:5px 10px;text-align:right}
th{background:#37474F;color:white;cursor:pointer;white-space:nowrap;user-select:none}
th:first-child,td:first-child,
th:nth-child(2),td:nth-child(2),
th:nth-child(3),td:nth-child(3),
th:nth-child(4),td:nth-child(4),
th:nth-child(5),td:nth-child(5){text-align:left}
tr:nth-child(even){background:#F5F5F5}
.note{font-size:12px;color:#666;margin:6px 0 16px}
.cross td{background:#FFF9C4 !important}
</style>"""


def write_html(out_path: str, scheme_label: str, note: str,
               df: pd.DataFrame, fig: go.Figure) -> None:
    chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Highlight cross-regime rows in yellow via <tr class="cross">
    tbl_html = df.to_html(
        index=False, border=0,
        float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
    )
    # Mark cross-regime rows
    for _, row in df[df["Step_Type"] == "cross"].iterrows():
        marker = f'<td>{row["Target_PMUT"]}</td>'
        tbl_html = tbl_html.replace(
            f'<tr>\n      <td>{row["Target_PMUT"]}</td>',
            f'<tr class="cross">\n      <td>{row["Target_PMUT"]}</td>',
            1,
        )

    with open(out_path, "w", encoding="utf-8") as fh:

        fh.write(
            f"<html><head><meta charset='utf-8'>"
            f"<title>{scheme_label} — Metrics</title>"
            f"{_STYLE}{_SORT_JS}</head><body>"
        )
        fh.write(f"<h1>{scheme_label}</h1>")
        fh.write(f"<p class='note'>{note}</p>")
        fh.write(chart_div)
        fh.write(
            f"<h2>Full Results "
            f"<small>(click any column header to sort)</small></h2>"
        )
        fh.write(tbl_html)
        fh.write("</body></html>")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load from pre-computed prediction caches ───────────────────────────
    print(f"\nLoading progressive cache  → {PROG_CACHE}")
    _, df_prog = load_progressive(PROG_CACHE)
    df_prog.to_csv(OUT_CSV_PROG, index=False)

    print(f"Loading regime cache       → {REGI_CACHE}")
    _, _, _, df_regime = load_regime(REGI_CACHE)
    df_regime.to_csv(OUT_CSV_REGIME, index=False)

    # ── Console summaries ──────────────────────────────────────────────────
    print("\n── Progressive — Mean by Regime × Model ─────────────────────────")
    print(df_prog.groupby(["Regime", "Model"])[["MAE", "RMSE", "R2"]]
          .mean().round(5).to_string())

    print("\n── Regime-Aware — Mean by Regime × Model ────────────────────────")
    print(df_regime.groupby(["Regime", "Model"])[["MAE", "RMSE", "R2"]]
          .mean().round(5).to_string())

    # ── Build + save HTML pages ────────────────────────────────────────────
    fig_prog = build_chart(df_prog, "Progressive Training")
    write_html(
        OUT_HTML_PROG,
        scheme_label = "Progressive Training — Vector · Full Spectrum",
        note = (
            "Cumulative training: each PMUT N is predicted by a model trained on "
            "all preceding PMUTs [1..N-1]. "
            "Approach: Vector (multi-output). Scope: Full Spectrum (1000 pts). "
            "R² negatives clipped to 0 in chart; red ✕ marks them."
        ),
        df=df_prog, fig=fig_prog,
    )

    fig_regime = build_chart(df_regime, "Regime-Aware Training")
    write_html(
        OUT_HTML_REGIME,
        scheme_label = "Regime-Aware Training — Vector · Full Spectrum",
        note = (
            "Fresh model at each regime boundary (Regimes: 1–2 | 3–11 | 12–16). "
            "Within a regime the model grows cumulatively. "
            "<span style='background:#FFF9C4;padding:1px 5px'>Yellow rows</span> = "
            "cross-regime predictions (hardest test). "
            "Approach: Vector (multi-output). Scope: Full Spectrum (1000 pts). "
            "R² negatives clipped to 0 in chart; red ✕ marks them."
        ),
        df=df_regime, fig=fig_regime,
    )

    print(f"\nSaved: {OUT_CSV_PROG}")
    print(f"Saved: {OUT_CSV_REGIME}")
    print(f"Saved: {OUT_HTML_PROG}")
    print(f"Saved: {OUT_HTML_REGIME}")


if __name__ == "__main__":
    main()
