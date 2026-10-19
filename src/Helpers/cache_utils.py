"""
helpers/cache_utils.py
──────────────────────────────────────────────────────────────────────────────
Load prediction CSVs written by 01_train_cache.py and reconstruct the data
structures used by downstream analysis scripts (04, 06, 07, 08).

Public API
──────────
  load_progressive(csv_path) → (all_data, df_metrics)
  load_regime(csv_path)      → (all_data, regime_map, cross_pmuts, df_metrics)

`all_data` is a dict[str(target_pmut) → entry_dict] compatible with
helpers.html_utils.build_interactive_html().

`df_metrics` covers the Vector approach only and contains:
  Target_PMUT, Regime, Step_Type, Train_PMUTs, N_train, Model, MAE, RMSE, R2
"""

import ast
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _regime_label(target: int) -> str:
    if target <= 2:   return "Regime 1 (1–2)"
    if target <= 11:  return "Regime 2 (3–11)"
    return "Regime 3 (12–16)"


def _metrics(act, pr):
    mae  = float(mean_absolute_error(act, pr))
    rmse = float(np.sqrt(np.mean((act - pr) ** 2)))
    r2   = float(r2_score(act, pr))
    return mae, rmse, r2


def _entry_from_group(tdf: pd.DataFrame):
    """
    Build one all_data entry dict from a group of rows all sharing the same
    target_pmut.  Also returns a list of flat Vector metric records.
    """
    meta        = tdf.iloc[0]
    train_pmuts = ast.literal_eval(meta["train_pmuts"])
    n_train     = int(meta["n_train"])
    fwhm_lo     = float(meta["fwhm_lo"])
    fwhm_hi     = float(meta["fwhm_hi"])

    # Canonical freq / actual series from Vector + RF (present for every target)
    vrf = (tdf[(tdf["approach"] == "Vector") & (tdf["model"] == "RF")]
           .sort_values("frequency_mhz"))
    freqs  = vrf["frequency_mhz"].values
    actual = vrf["actual"].values

    entry = {
        "train":   train_pmuts,
        "freqs":   [round(float(f), 5) for f in freqs],
        "actual":  [round(float(a), 6) for a in actual],
        "fwhm_lo": fwhm_lo,
        "fwhm_hi": fwhm_hi,
    }
    score_rows     = []
    metric_records = []

    for approach in ["Vector", "Pointwise"]:
        entry[approach] = {}
        apdf = tdf[tdf["approach"] == approach]
        models_in_data = apdf["model"].unique()
        for model in ["RF", "GB", "MLP", "PINN", "MDOF-PINN", "PINN-Informed-Loss", "RF-MDOF-PINN"]:
            if model not in models_in_data:
                continue

            mdf = apdf[apdf["model"] == model].sort_values("frequency_mhz")
            if mdf.empty:
                continue
            fq_m  = mdf["frequency_mhz"].values
            act_m = mdf["actual"].values
            pr_m  = mdf["predicted"].values

            roi   = (fq_m >= fwhm_lo) & (fq_m <= fwhm_hi)
            mae_f, rmse_f, r2_f = _metrics(act_m, pr_m)
            mae_r = (float(mean_absolute_error(act_m[roi], pr_m[roi]))
                     if roi.sum() >= 2 else None)
            r2_r  = (float(r2_score(act_m[roi], pr_m[roi]))
                     if roi.sum() >= 2 else None)

            entry[approach][model] = {
                "pred":     [round(float(p), 6) for p in pr_m],
                "mae_full": round(mae_f, 6),
                "r2_full":  round(r2_f,  4),
                "mae_roi":  round(mae_r, 6) if mae_r is not None else None,
                "r2_roi":   round(r2_r,  4) if r2_r  is not None else None,
            }
            score_rows.append([approach, model, mae_f, r2_f, mae_r, r2_r])

            if approach == "Vector":
                metric_records.append({
                    "Target_PMUT": int(meta["target_pmut"]),
                    "Train_PMUTs": meta["train_pmuts"],
                    "N_train":     n_train,
                    "Model":       model,
                    "MAE":         round(mae_f,  6),
                    "RMSE":        round(rmse_f, 6),
                    "R2":          round(r2_f,   4),
                })

    entry["score_rows"] = score_rows
    return entry, metric_records


# ── Public loaders ────────────────────────────────────────────────────────────

def load_progressive(csv_path: str):
    """
    Load results/predictions/progressive_cache.csv.

    Returns
    -------
    all_data   : dict[str → dict]   keyed by str(target_pmut)
    df_metrics : pd.DataFrame       Vector metrics, one row per (Target_PMUT, Model)
    """
    df = pd.read_csv(csv_path)
    all_data           = {}
    all_metric_records = []

    for target, tdf in df.groupby("target_pmut"):
        entry, mrecs = _entry_from_group(tdf)
        target_int   = int(target)
        for rec in mrecs:
            rec["Regime"]    = _regime_label(target_int)
            rec["Step_Type"] = "progressive"
        all_data[str(target_int)] = entry
        all_metric_records.extend(mrecs)

    df_metrics = pd.DataFrame(all_metric_records)[[
        "Target_PMUT", "Regime", "Step_Type", "Train_PMUTs",
        "N_train", "Model", "MAE", "RMSE", "R2",
    ]]
    return all_data, df_metrics


def load_regime(csv_path: str):
    """
    Load results/predictions/regime_cache.csv.

    Returns
    -------
    all_data    : dict[str → dict]   keyed by str(target_pmut);
                  entries also carry 'regime_name' and 'step_type'
    regime_map  : dict[int → int]    {target_pmut: regime_id}
    cross_pmuts : set[int]           targets that are cross-regime predictions
    df_metrics  : pd.DataFrame       Vector metrics, one row per (Target_PMUT, Model)
    """
    df = pd.read_csv(csv_path)
    all_data           = {}
    all_metric_records = []
    regime_map         = {}
    cross_pmuts        = set()

    for target, tdf in df.groupby("target_pmut"):
        target_int = int(target)
        meta       = tdf.iloc[0]
        rid        = int(meta["regime_id"])
        rname      = str(meta["regime_name"])
        step_type  = str(meta["step_type"])

        entry, mrecs = _entry_from_group(tdf)
        entry["regime_name"] = rname
        entry["step_type"]   = step_type

        for rec in mrecs:
            rec["Regime"]      = _regime_label(target_int)
            rec["Regime_Name"] = rname
            rec["Step_Type"]   = step_type

        all_data[str(target_int)] = entry
        all_metric_records.extend(mrecs)
        regime_map[target_int] = rid
        if step_type == "cross":
            cross_pmuts.add(target_int)

    df_metrics = pd.DataFrame(all_metric_records)[[
        "Target_PMUT", "Regime", "Regime_Name", "Step_Type", "Train_PMUTs",
        "N_train", "Model", "MAE", "RMSE", "R2",
    ]]
    return all_data, regime_map, cross_pmuts, df_metrics
