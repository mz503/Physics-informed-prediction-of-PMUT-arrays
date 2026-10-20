"""
helpers/data_loader.py
Load and preprocess PMUT data from the unified CSV file.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/processed/pmuts_all.csv"


def load_data(csv_path: str = CSV_PATH):
    """
    Load the unified PMUT CSV and return the device identifiers.

    Returns
    -------
    df_all        : DataFrame with N_PMUT, Frequency_MHz, and Amplitude_R_mean
    unique_pmuts  : sorted list of PMUT IDs
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["N_PMUT", "Frequency_MHz"]).reset_index(drop=True)
    unique_pmuts = sorted(df["N_PMUT"].unique())

    return df, unique_pmuts


def scale_target_for_training(df_all: pd.DataFrame, training_pmuts: list):
    """Fit amplitude scaling on training PMUTs and transform all amplitudes."""
    if not training_pmuts:
        raise ValueError("training_pmuts must contain at least one PMUT")

    training_mask = df_all["N_PMUT"].isin(training_pmuts)
    found_pmuts = set(df_all.loc[training_mask, "N_PMUT"].unique())
    missing_pmuts = set(training_pmuts) - found_pmuts
    if missing_pmuts:
        raise ValueError(f"Training PMUTs are missing from the data: {sorted(missing_pmuts)}")

    amp_scaler = StandardScaler()
    amp_scaler.fit(df_all.loc[training_mask, ["Amplitude_R_mean"]])

    scaled_df = df_all.copy()
    scaled_df["Amplitude_Scaled"] = amp_scaler.transform(
        df_all[["Amplitude_R_mean"]]
    ).ravel()
    return scaled_df, amp_scaler


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
