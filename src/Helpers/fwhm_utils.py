"""
helpers/fwhm_utils.py
Baseline-corrected FWHM utilities for all experiments.

The half-max threshold is computed relative to the curve's own baseline
so an asymmetric or elevated baseline does not inflate the window:
    h_half = A_min + (A_max - A_min) / 2
"""

import pandas as pd
import numpy as np


def compute_fwhm_window(pmut: int, df_all: pd.DataFrame):
    """
    Baseline-corrected FWHM for one PMUT.

    Returns
    -------
    f_lo_MHz, f_hi_MHz : float — left and right half-max crossing frequencies
    """
    sub = df_all[df_all["N_PMUT"] == pmut]
    a_min = sub["Amplitude_R_mean"].min()
    a_max = sub["Amplitude_R_mean"].max()
    h_half = a_min + (a_max - a_min) / 2
    above = sub[sub["Amplitude_R_mean"] >= h_half]
    return float(above["Frequency_MHz"].min()), float(above["Frequency_MHz"].max())


def compute_fwhm_all(df_all: pd.DataFrame, unique_pmuts: list) -> pd.DataFrame:
    """
    Compute baseline-corrected FWHM for every PMUT.

    Returns DataFrame with columns:
        N_PMUT, f_peak_MHz, f_lo_MHz, f_hi_MHz, FWHM_MHz, Q_factor
    """
    rows = []
    for p in unique_pmuts:
        sub = df_all[df_all["N_PMUT"] == p]
        a_min = sub["Amplitude_R_mean"].min()
        a_max = sub["Amplitude_R_mean"].max()
        h_half = a_min + (a_max - a_min) / 2
        above = sub[sub["Amplitude_R_mean"] >= h_half]
        f_peak = float(sub.loc[sub["Amplitude_R_mean"].idxmax(), "Frequency_MHz"])
        f_lo   = float(above["Frequency_MHz"].min())
        f_hi   = float(above["Frequency_MHz"].max())
        fwhm   = f_hi - f_lo
        q      = round(f_peak / fwhm, 2) if fwhm > 0 else float("nan")
        rows.append({
            "N_PMUT":     p,
            "f_peak_MHz": f_peak,
            "f_lo_MHz":   f_lo,
            "f_hi_MHz":   f_hi,
            "FWHM_MHz":   fwhm,
            "Q_factor":   q,
        })
    return pd.DataFrame(rows)


def get_window_train_df(t_pmuts: list, df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Collect only the in-band (FWHM) rows from each training PMUT's own curve.
    This is the training dataset for the FWHM-Window approach.
    """
    parts = []
    for p in t_pmuts:
        f_lo, f_hi = compute_fwhm_window(p, df_all)
        mask = (
            (df_all["N_PMUT"] == p) &
            (df_all["Frequency_MHz"] >= f_lo) &
            (df_all["Frequency_MHz"] <= f_hi)
        )
        parts.append(df_all[mask])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
