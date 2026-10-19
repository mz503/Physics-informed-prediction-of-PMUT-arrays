"""
Helpers/rf_mdof_pinn_ensemble.py
══════════════════════════════════════════════════════════════════════════════
Hybrid Ensemble Model combining Random Forest (RF) data interpolation with
Multi-Degree-Of-Freedom Physics-Informed Neural Network (MDOF-PINN) physical dynamics.

Primary Strategy: Physics-Residual Stacking (RF-MDOF-PINN)
  1. Fit MDOF-PINN to model continuous multi-mode physics across array sizes N.
  2. Compute residual error on training PMUTs: r = y_actual - y_mdof_pinn.
  3. Train Random Forest (RF) regressor to interpolate the residual error r(f, N).
  4. Final Ensemble prediction: y_ensemble = y_mdof_pinn + r_rf.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from Helpers.mdof_pinn_model import fit_mdof_pinn
from Helpers.pinn_informed_loss_model import fit_pinn_informed_loss


def _inv(scaler, arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def train_predict_rf_mdof_ensemble_vector(df_all: pd.DataFrame, t_pmuts: list,
                                          target: int, amp_scaler,
                                          ensemble_mode: str = "residual",
                                          backbone: str = "mdof_pinn") -> tuple:
    """
    Vector (multi-output) approach for RF + MDOF-PINN Ensemble.

    Features : [N_PMUT] (scalar per training device)
    Target   : 1000-point amplitude spectrum
    backbone : 'mdof_pinn' | 'pinn_informed_loss'

    Returns
    -------
    freqs, actual, preds (all in original amplitude scale)
    """
    tg_df = df_all[df_all["N_PMUT"] == target].sort_values("Frequency_MHz")
    freqs = tg_df["Frequency_MHz"].values
    actual = _inv(amp_scaler, tg_df["Amplitude_Scaled"].values)

    # 1. Fit Physical PINN Backbone
    if backbone in ["pinn_informed_loss", "pil", "informed_loss"]:
        pinn_backbone = fit_pinn_informed_loss(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
    else:
        pinn_backbone = fit_mdof_pinn(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
    
    pinn_backbone.eval()

    # Predict target using PINN Backbone
    with torch.no_grad():
        N_tg_t = torch.tensor([[float(target)]], dtype=torch.float32)
        f_tg_t = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
        preds_mdof_tg = pinn_backbone(N_tg_t, f_tg_t).numpy().flatten()

    # 2. Build training set for RF
    X_tr = np.array([[float(p)] for p in t_pmuts])
    y_tr_actual_list = []
    y_tr_mdof_list = []

    with torch.no_grad():
        for p in t_pmuts:
            pdf = df_all[df_all["N_PMUT"] == p].sort_values("Frequency_MHz")
            f_p = pdf["Frequency_MHz"].values
            a_p = _inv(amp_scaler, pdf["Amplitude_Scaled"].values)
            
            N_p_t = torch.tensor([[float(p)]], dtype=torch.float32)
            f_p_t = torch.tensor(f_p.reshape(-1, 1), dtype=torch.float32)
            pred_p = pinn_backbone(N_p_t, f_p_t).numpy().flatten()

            y_tr_actual_list.append(a_p)
            y_tr_mdof_list.append(pred_p)

    y_tr_actual = np.stack(y_tr_actual_list)  # (n_train, 1000)
    y_tr_mdof = np.stack(y_tr_mdof_list)      # (n_train, 1000)

    if ensemble_mode == "blend":
        # Direct RF fit
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr_actual)
        preds_rf_tg = rf.predict([[float(target)]])[0]
        preds_ensemble = 0.5 * preds_rf_tg + 0.5 * preds_mdof_tg
    elif ensemble_mode == "adaptive":
        # Frequency-dependent blending: favor MDOF-PINN near resonance peaks, RF elsewhere
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr_actual)
        preds_rf_tg = rf.predict([[float(target)]])[0]

        # Calculate peak proximity weight w(f)
        peak_val = np.max(preds_mdof_tg)
        min_val = np.min(preds_mdof_tg)
        norm_shape = (preds_mdof_tg - min_val) / (peak_val - min_val + 1e-8)
        w_phys = np.clip(norm_shape ** 2, 0.2, 0.8)

        preds_ensemble = w_phys * preds_mdof_tg + (1.0 - w_phys) * preds_rf_tg
    else:
        # Default: Residual Physics Stacking
        y_tr_residual = y_tr_actual - y_tr_mdof
        rf_res = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_res.fit(X_tr, y_tr_residual)
        
        preds_res_tg = rf_res.predict([[float(target)]])[0]
        preds_ensemble = preds_mdof_tg + preds_res_tg

    return freqs, actual, preds_ensemble


def train_predict_rf_mdof_ensemble_pointwise(df_all: pd.DataFrame, t_pmuts: list,
                                             target: int, amp_scaler,
                                             ensemble_mode: str = "residual",
                                             backbone: str = "mdof_pinn") -> tuple:
    """
    Pointwise approach for RF + MDOF-PINN Ensemble.

    Features : [N_PMUT, Frequency_MHz]
    backbone : 'mdof_pinn' | 'pinn_informed_loss'

    Returns
    -------
    freqs, actual, preds (all in original amplitude scale)
    """
    mask_tg = df_all["N_PMUT"] == target
    tg_df = df_all.loc[mask_tg].sort_values("Frequency_MHz")
    freqs = tg_df["Frequency_MHz"].values
    actual = _inv(amp_scaler, tg_df["Amplitude_Scaled"].values)

    # 1. Fit Physical PINN Backbone
    if backbone in ["pinn_informed_loss", "pil", "informed_loss"]:
        pinn_backbone = fit_pinn_informed_loss(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)
    else:
        pinn_backbone = fit_mdof_pinn(df_all, t_pmuts, num_modes=3, epochs=350, lr=0.005)

    pinn_backbone.eval()

    with torch.no_grad():
        N_tg_t = torch.tensor([[float(target)]], dtype=torch.float32)
        f_tg_t = torch.tensor(freqs.reshape(-1, 1), dtype=torch.float32)
        preds_mdof_tg = pinn_backbone(N_tg_t, f_tg_t).numpy().flatten()

    mask_tr = df_all["N_PMUT"].isin(t_pmuts)
    tr_df = df_all.loc[mask_tr].sort_values(["N_PMUT", "Frequency_MHz"])
    X_tr = tr_df[["N_PMUT", "Frequency_MHz"]].values
    y_tr_actual = _inv(amp_scaler, tr_df["Amplitude_Scaled"].values)

    N_tr_t = torch.tensor(X_tr[:, 0:1], dtype=torch.float32)
    f_tr_t = torch.tensor(X_tr[:, 1:2], dtype=torch.float32)

    with torch.no_grad():
        preds_mdof_tr = pinn_backbone(N_tr_t, f_tr_t).numpy().flatten()

    X_tg = tg_df[["N_PMUT", "Frequency_MHz"]].values

    if ensemble_mode == "blend":
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr_actual)
        preds_rf_tg = rf.predict(X_tg)
        preds_ensemble = 0.5 * preds_rf_tg + 0.5 * preds_mdof_tg
    else:
        # Default: Residual Physics Stacking
        y_tr_residual = y_tr_actual - preds_mdof_tr
        rf_res = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_res.fit(X_tr, y_tr_residual)

        preds_res_tg = rf_res.predict(X_tg)
        preds_ensemble = preds_mdof_tg + preds_res_tg

    return freqs, actual, preds_ensemble
