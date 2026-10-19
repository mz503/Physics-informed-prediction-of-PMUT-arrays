"""
helpers/plot_utils.py
Plotly figure builders shared across all task scripts.
"""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODEL_COLORS = {
    "RF":                 "#2196F3",   # blue
    "GB":                 "#FF9800",   # orange
    "MLP":                "#9C27B0",   # purple
    "PINN":               "#E91E63",   # pink / magenta
    "MDOF-PINN":          "#8E24AA",   # deep purple / violet
    "PINN-Informed-Loss": "#D81B60",   # crimson / magenta red
    "RF-MDOF-PINN":       "#00796B",   # teal / cyan green (ensemble)
    "Actual":             "#4CAF50",   # green
}

DASH_STYLES = {
    "RF":                 "solid",
    "GB":                 "dash",
    "MLP":                "dot",
    "PINN":               "dashdot",
    "MDOF-PINN":          "dashdot",
    "PINN-Informed-Loss": "dash",
    "RF-MDOF-PINN":       "solid",
}




def prediction_figure(
    freqs_dict: dict,
    actual_dict: dict,
    preds_dict: dict,
    target_pmut: int,
    regime_label: str,
    train_pmuts: list,
    fwhm_lo: Optional[float] = None,
    fwhm_hi: Optional[float] = None,
    show_roi: bool = True,
) -> go.Figure:
    """
    Overlay actual vs prediction for multiple models on one figure.
    Optionally adds a second subplot showing only the FWHM ROI.

    Parameters
    ----------
    freqs_dict  : {model_name: freqs_array}
    actual_dict : {model_name: actual_array}   (all share same actual)
    preds_dict  : {model_name: preds_array}
    """
    n_cols = 2 if show_roi else 1
    titles = [f"Full Spectrum — PMUT {target_pmut}"]
    if show_roi:
        titles.append(f"FWHM ROI — PMUT {target_pmut}")

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles)

    first_model = next(iter(freqs_dict))
    actual = actual_dict[first_model]
    freqs_full = freqs_dict[first_model]

    # Actual trace (full spectrum)
    fig.add_trace(go.Scatter(
        x=freqs_full, y=actual,
        name="Actual",
        line=dict(color=MODEL_COLORS["Actual"], width=2),
    ), row=1, col=1)

    for m_name, preds in preds_dict.items():
        freqs = freqs_dict[m_name]
        fig.add_trace(go.Scatter(
            x=freqs, y=preds,
            name=m_name,
            line=dict(color=MODEL_COLORS[m_name], dash=DASH_STYLES[m_name], width=1.5),
        ), row=1, col=1)

    if show_roi and fwhm_lo is not None and fwhm_hi is not None:
        roi_mask = (freqs_full >= fwhm_lo) & (freqs_full <= fwhm_hi)

        fig.add_trace(go.Scatter(
            x=freqs_full[roi_mask], y=actual[roi_mask],
            name="Actual (ROI)", showlegend=False,
            line=dict(color=MODEL_COLORS["Actual"], width=2),
        ), row=1, col=2)

        for m_name, preds in preds_dict.items():
            freqs = freqs_dict[m_name]
            f_mask = (freqs >= fwhm_lo) & (freqs <= fwhm_hi)
            if f_mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=freqs[f_mask], y=preds[f_mask],
                name=f"{m_name} (ROI)", showlegend=False,
                line=dict(color=MODEL_COLORS[m_name], dash=DASH_STYLES[m_name], width=1.5),
            ), row=1, col=2)

        # Shade FWHM region on full-spectrum plot
        fig.add_vrect(x0=fwhm_lo, x1=fwhm_hi, fillcolor="rgba(0,0,0,0.08)",
                      line_width=0, row=1, col=1)

    train_str = ", ".join(map(str, train_pmuts))
    fig.update_layout(
        title=dict(text=(
            f"PMUT {target_pmut} | {regime_label}<br>"
            f"<sup>Train: [{train_str}] → Predict: {target_pmut}</sup>"
        ), font_size=14),
        height=420,
        legend=dict(orientation="h", y=-0.18),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Frequency (MHz)")
    fig.update_yaxes(title_text="Amplitude (R)")

    return fig


def metrics_bar_figure(df_metrics, metric: str = "MAE") -> go.Figure:
    """Bar chart comparing a metric across models for each target PMUT."""
    fig = go.Figure()
    for model in df_metrics["Model"].unique():
        sub = df_metrics[df_metrics["Model"] == model]
        fig.add_trace(go.Bar(
            x=[f"PMUT {p}" for p in sub["Target_PMUT"]],
            y=sub[metric],
            name=model,
            marker_color=MODEL_COLORS.get(model, "#999"),
        ))
    fig.update_layout(
        barmode="group",
        title=f"{metric} by Target PMUT and Model",
        xaxis_title="Target PMUT",
        yaxis_title=metric,
        template="plotly_white",
        height=450,
    )
    return fig
