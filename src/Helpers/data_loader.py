"""
helpers/data_loader.py
Load and preprocess PMUT data from the unified CSV file.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/processed/pmuts_all.csv"


def load_data(csv_path: str = CSV_PATH):
    """
    Load the unified PMUT CSV, fit a StandardScaler on Amplitude_R_mean,
    and build per-PMUT amplitude matrices for multi-output models.

    Returns
    -------
    df_all        : DataFrame with N_PMUT, Frequency_MHz, Amplitude_R_mean, Amplitude_Scaled
    amp_scaler    : fitted StandardScaler (use for inverse_transform)
    unique_pmuts  : sorted list of PMUT IDs
    y_mo          : ndarray (n_pmuts, n_freqs) — scaled amplitude per PMUT
    X_mo          : ndarray (n_pmuts, 1)       — PMUT number as feature
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["N_PMUT", "Frequency_MHz"]).reset_index(drop=True)

    amp_scaler = StandardScaler()
    df["Amplitude_Scaled"] = amp_scaler.fit_transform(df[["Amplitude_R_mean"]])

    unique_pmuts = sorted(df["N_PMUT"].unique())

    y_mo = np.array([df[df["N_PMUT"] == p]["Amplitude_Scaled"].values
                     for p in unique_pmuts])
    X_mo = np.array(unique_pmuts).reshape(-1, 1)

    return df, amp_scaler, unique_pmuts, y_mo, X_mo


def get_regimes():
    """
    Return the three progressive training regimes used throughout all experiments.
    Each step adds one more training PMUT and predicts the next unseen device.
    """
    return {
        "Regime 1 (PMUTs 1-3)": {
            "train": [[1], [1, 2]],
            "pred":  [[2], [3]],
        },
        "Regime 2 (PMUTs 3-8)": {
            "train": [[3, 4], [3, 4, 5], [3, 4, 5, 6], [3, 4, 5, 6, 7]],
            "pred":  [[5],    [6],        [7],           [8]],
        },
        "Regime 3 (PMUTs 9-16)": {
            "train": [[9, 10], [9, 10, 11], [9, 10, 11, 12],
                      [9, 10, 11, 12, 13],  [9, 10, 11, 12, 13, 14],
                      [9, 10, 11, 12, 13, 14, 15]],
            "pred":  [[11], [12], [13], [14], [15], [16]],
        },
    }


def get_sequential_training(target: int, unique_pmuts: list):
    """
    For sequential N=2..16 plots: train on all PMUTs < target.
    Returns None if no training data is available (i.e. target == min PMUT).
    """
    train = [p for p in unique_pmuts if p < target]
    return train if train else None
