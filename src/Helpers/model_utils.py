"""
helpers/model_utils.py
Model factory and train-predict routines for Pointwise, Multi-Output,
and FWHM-Window approaches.

Each function returns:
    freqs   : ndarray of frequency values for the evaluated points
    actual  : ndarray of true amplitude values (original scale)
    preds   : ndarray of predicted amplitude values (original scale)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from Helpers.fwhm_utils import compute_fwhm_window, get_window_train_df


def _roi_mask(freqs, fwhm_lo, fwhm_hi):
    return (freqs >= fwhm_lo) & (freqs <= fwhm_hi)


def get_models(approach: str = "pointwise"):
    """
    Return fresh model instances keyed by name.
    approach : 'pointwise' | 'multioutput' | 'fwhm_window'
    """
    rf  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gb  = GradientBoostingRegressor(n_estimators=100, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000,
                       random_state=42, early_stopping=True, n_iter_no_change=20)

    if approach == "multioutput":
        gb  = MultiOutputRegressor(gb, n_jobs=-1)
        mlp = MultiOutputRegressor(mlp, n_jobs=-1)

    return {"RF": rf, "GB": gb, "MLP": mlp}


def _inv(scaler, arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# ──────────────────────────────────────────────────────────────
# Pointwise approach
# ──────────────────────────────────────────────────────────────

def train_predict_pointwise(model_name, df_all, t_pmuts, target, amp_scaler):
    """
    Features: [N_PMUT, Frequency_MHz]
    Train on all rows of training PMUTs, predict full curve of target PMUT.
    MLP inputs are feature-scaled internally.
    """
    mask_tg = df_all["N_PMUT"] == target
    tg_df = df_all.loc[mask_tg].sort_values("Frequency_MHz")
    freqs = tg_df["Frequency_MHz"].values
    actual = _inv(amp_scaler, tg_df["Amplitude_Scaled"].values)

    if model_name == "PINN":
        import torch
        from Helpers.pinn_model import fit_pinn
        pinn = fit_pinn(df_all, t_pmuts, use_two_modes=False, epochs=150, lr=0.01)
        pinn.eval()
        with torch.no_grad():
            N_tg = torch.tensor([[float(target)]], dtype=torch.float32)
            f_tg = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
            preds = pinn(N_tg, f_tg).numpy().flatten()
        return freqs, actual, preds

    if model_name in ["MDOF-PINN", "MDOF_PINN"]:
        import torch
        from Helpers.mdof_pinn_model import fit_mdof_pinn
        mdof_pinn = fit_mdof_pinn(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
        mdof_pinn.eval()
        with torch.no_grad():
            N_tg = torch.tensor([[float(target)]], dtype=torch.float32)
            f_tg = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
            preds = mdof_pinn(N_tg, f_tg).numpy().flatten()
        return freqs, actual, preds

    if model_name in ["PINN-Informed-Loss", "PINN_Informed_Loss", "PINN-Informed", "PINN-Loss"]:
        import torch
        from Helpers.pinn_informed_loss_model import fit_pinn_informed_loss
        pil_model = fit_pinn_informed_loss(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
        pil_model.eval()
        with torch.no_grad():
            N_tg = torch.tensor([[float(target)]], dtype=torch.float32)
            f_tg = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
            preds = pil_model(N_tg, f_tg).numpy().flatten()
        return freqs, actual, preds

    if model_name in ["RF-MDOF-PINN", "RF_MDOF_PINN", "RF-MDOF-Ensemble", "RF_MDOF_Ensemble"]:
        from Helpers.rf_mdof_pinn_ensemble import train_predict_rf_mdof_ensemble_pointwise
        return train_predict_rf_mdof_ensemble_pointwise(df_all, t_pmuts, target, amp_scaler)




    models = get_models("pointwise")
    model = models[model_name]

    mask_tr = df_all["N_PMUT"].isin(t_pmuts)
    X_tr = df_all.loc[mask_tr, ["N_PMUT", "Frequency_MHz"]].values
    y_tr = df_all.loc[mask_tr, "Amplitude_Scaled"].values
    X_tg = df_all.loc[mask_tg, ["N_PMUT", "Frequency_MHz"]].values

    if model_name == "MLP":
        feat_sc = StandardScaler()
        X_tr = feat_sc.fit_transform(X_tr)
        X_tg = feat_sc.transform(X_tg)

    model.fit(X_tr, y_tr)
    preds  = _inv(amp_scaler, model.predict(X_tg))

    return freqs, actual, preds


# ──────────────────────────────────────────────────────────────
# Multi-Output approach
# ──────────────────────────────────────────────────────────────

def train_predict_multioutput(model_name, X_mo, y_mo, unique_pmuts,
                               t_pmuts, target, amp_scaler, df_all):
    """
    Features: [N_PMUT]  (scalar per device)
    Target:   full 1000-point amplitude vector per device (scaled)
    Predicts the full spectrum for the target PMUT.
    """
    models = get_models("multioutput")
    model = models[model_name]

    idx_train  = [unique_pmuts.index(p) for p in t_pmuts]
    idx_target = unique_pmuts.index(target)

    X_tr = X_mo[idx_train]
    y_tr = y_mo[idx_train]

    if model_name == "MLP":
        feat_sc = StandardScaler()
        X_tr = feat_sc.fit_transform(X_tr)
        X_tg = feat_sc.transform(X_mo[[idx_target]])
    else:
        X_tg = X_mo[[idx_target]]

    model.fit(X_tr, y_tr)
    preds_scaled = model.predict(X_tg)[0]

    preds  = _inv(amp_scaler, preds_scaled)
    actual = _inv(amp_scaler, y_mo[idx_target])
    freqs  = df_all[df_all["N_PMUT"] == target]["Frequency_MHz"].values

    return freqs, actual, preds


# ──────────────────────────────────────────────────────────────
# FWHM-Window approach
# ──────────────────────────────────────────────────────────────

def train_predict_fwhm_window(model_name, df_all, t_pmuts, target, amp_scaler):
    """
    Train only on in-band (FWHM) rows of training PMUTs.
    Predict only within the target's own FWHM band.
    Returns frequencies, actuals, and predictions for the in-band region only.
    """
    models = get_models("pointwise")
    model = models[model_name]

    df_train = get_window_train_df(t_pmuts, df_all)
    if df_train.empty:
        return None, None, None

    X_tr = df_train[["N_PMUT", "Frequency_MHz"]].values
    y_tr = df_train["Amplitude_Scaled"].values

    t_lo, t_hi = compute_fwhm_window(target, df_all)
    mask_tg = (
        (df_all["N_PMUT"] == target) &
        (df_all["Frequency_MHz"] >= t_lo) &
        (df_all["Frequency_MHz"] <= t_hi)
    )
    X_tg = df_all.loc[mask_tg, ["N_PMUT", "Frequency_MHz"]].values

    if model_name == "MLP":
        feat_sc = StandardScaler()
        X_tr = feat_sc.fit_transform(X_tr)
        X_tg = feat_sc.transform(X_tg)

    model.fit(X_tr, y_tr)
    preds  = _inv(amp_scaler, model.predict(X_tg))
    actual = _inv(amp_scaler, df_all.loc[mask_tg, "Amplitude_Scaled"].values)
    freqs  = df_all.loc[mask_tg, "Frequency_MHz"].values

    return freqs, actual, preds


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Vector (Multi-Output) approach
# ──────────────────────────────────────────────────────────────

def train_predict_vector(model_name: str, df_all, t_pmuts: list,
                         target: int, amp_scaler) -> tuple:
    """
    Vector / multi-output approach.

    Features : [N_PMUT]  — one row per training device (scalar).
    Target   : full 1000-point amplitude vector for that device.

    The model sees only the device ID and must generalise the entire
    frequency response in a single forward pass.

    Returns
    -------
    freqs, actual, preds — all frequency-sorted, original-scale amplitude.
    """
    tg_df  = df_all[df_all["N_PMUT"] == target].sort_values("Frequency_MHz")
    freqs  = tg_df["Frequency_MHz"].values
    actual = _inv(amp_scaler, tg_df["Amplitude_Scaled"].values)

    if model_name == "PINN":
        import torch
        from Helpers.pinn_model import fit_pinn
        pinn = fit_pinn(df_all, t_pmuts, use_two_modes=False, epochs=150, lr=0.01)
        pinn.eval()
        with torch.no_grad():
            N_tg = torch.tensor([[float(target)]], dtype=torch.float32)
            f_tg = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
            preds = pinn(N_tg, f_tg).numpy().flatten()
        return freqs, actual, preds

    if model_name in ["MDOF-PINN", "MDOF_PINN"]:
        import torch
        from Helpers.mdof_pinn_model import fit_mdof_pinn
        mdof_pinn = fit_mdof_pinn(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
        mdof_pinn.eval()
        with torch.no_grad():
            N_tg = torch.tensor([[float(target)]], dtype=torch.float32)
            f_tg = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
            preds = mdof_pinn(N_tg, f_tg).numpy().flatten()
        return freqs, actual, preds

    if model_name in ["PINN-Informed-Loss", "PINN_Informed_Loss", "PINN-Informed", "PINN-Loss"]:
        import torch
        from Helpers.pinn_informed_loss_model import fit_pinn_informed_loss
        pil_model = fit_pinn_informed_loss(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
        pil_model.eval()
        with torch.no_grad():
            N_tg = torch.tensor([[float(target)]], dtype=torch.float32)
            f_tg = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
            preds = pil_model(N_tg, f_tg).numpy().flatten()
        return freqs, actual, preds

    if model_name in ["RF-MDOF-PINN", "RF_MDOF_PINN", "RF-MDOF-Ensemble", "RF_MDOF_Ensemble"]:
        from Helpers.rf_mdof_pinn_ensemble import train_predict_rf_mdof_ensemble_vector
        return train_predict_rf_mdof_ensemble_vector(df_all, t_pmuts, target, amp_scaler)




    # Build (n_train × 1) feature matrix and (n_train × n_freqs) label matrix
    X_tr = np.array([[float(p)] for p in t_pmuts])
    y_tr = np.stack([
        df_all[df_all["N_PMUT"] == p]
              .sort_values("Frequency_MHz")["Amplitude_Scaled"].values
        for p in t_pmuts
    ])                              # shape: (n_train, n_freqs)
    X_tg = np.array([[float(target)]])

    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_name == "GB":
        model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=30, random_state=42),
            n_jobs=-1,
        )
    else:  # MLP — handles multi-output natively
        # No early stopping for vector: training set has O(n_pmuts) rows,
        # which is too small for a validation split.
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000,
                             random_state=42, early_stopping=False)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_tg = sc.transform(X_tg)

    model.fit(X_tr, y_tr)
    preds = _inv(amp_scaler, model.predict(X_tg)[0])

    return freqs, actual, preds


def compute_metrics(actual: np.ndarray, preds: np.ndarray,
                    tolerance_frac: float = 0.10) -> dict:
    """
    Compute MAE, RMSE, R², and tolerance-based accuracy.

    tolerance_frac : fraction of amplitude range used as ±acceptance band
                     (default 10 %).  accuracy = % of points within band.
    """
    mae  = float(mean_absolute_error(actual, preds))
    rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
    r2   = float(r2_score(actual, preds))

    amp_range = float(actual.max() - actual.min())
    tol = tolerance_frac * amp_range if amp_range > 0 else 0.0
    within = np.abs(actual - preds) <= tol
    accuracy = float(100.0 * within.mean())

    return {"MAE": mae, "RMSE": rmse, "R2": r2,
            "Accuracy_%": accuracy, "n_pts": len(actual)}
