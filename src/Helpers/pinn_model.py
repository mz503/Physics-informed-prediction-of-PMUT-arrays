"""
Helpers/pinn_model.py
══════════════════════════════════════════════════════════════════════════════
Physics-Informed Neural Network (PINN) / Grey-Box Reduced-Order Model for PMUTs.

Single-mode SDOF forced damped oscillator mapping with fixed mass m and predicted resonant frequency f0 & Quality factor Q:
  Array Size (N_PMUT) ──► [alpha_N, f0_N, Q_N, G_N, A_bg]

Mathematics:
  w = 2 * pi * f_MHz
  w0_N = 2 * pi * f0_N
  k_N = m * w0_N^2
  c_N = (m * w0_N) / Q_N
  |X(w, N)| = (alpha_N * V) / sqrt( (k_N - m * w^2)^2 + (c_N * w)^2 + 1e-12 )
  A_pred(f, N) = G_N * |X(w, N)| + A_bg
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Helpers.bvd_model import fit_bvd_pmut
from Helpers.mdof_model import fit_mdof_pmut


class PhysicsInformedPMUTModel(nn.Module):
    """
    Physics-Informed PyTorch Model mapping array size N to SDOF parameters: [alpha_N, f0_N, Q_N, G_N, A_bg].
    Mass (m = 0.38) is a fixed physical constant.
    Resonant frequency f0_N (MHz) and Quality factor Q_N are predicted parameters per array size N.
    Derived parameters:
      w0_N = 2 * pi * f0_N
      c_N = (m * w0_N) / Q_N
      k_N = m * (w0_N^2)
    Positivity constraints on alpha, f0, Q, G via Softplus. A_bg is left unbounded.
    """
    def __init__(self, m: float = 0.38, V: float = 10.0,
                 use_two_modes: bool = False, hidden_dim: int = 32,
                 fix_m_k: bool = True, m_val: float = None, k_val: float = None):
        super().__init__()
        if m_val is not None:
            m = float(m_val)
        self.m = float(m)
        self.V = float(V)
        self.use_two_modes = use_two_modes

        # 5 outputs for single-mode (alpha, f0, Q, G, A_bg)
        # 9 outputs for two-mode (alpha1, f0_1, Q1, G1, alpha2, f0_2, Q2, G2, A_bg)
        n_out = 9 if use_two_modes else 5
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_out)
        )
        self.softplus = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_raw_params(self, N_tensor: torch.Tensor) -> dict:
        """Apply positivity constraints to NN outputs for alpha, f0, Q, G; leave A_bg unbounded."""
        if N_tensor.ndim == 1:
            N_tensor = N_tensor.unsqueeze(-1)
            
        N_scaled = N_tensor.float() / 16.0
        out = self.net(N_scaled)
        
        params = {}
        if not self.use_two_modes:
            params["alpha"] = self.softplus(out[:, 0:1]) + 1e-6
            params["f0"]    = self.softplus(out[:, 1:2]) + 0.1
            params["Q"]     = self.softplus(out[:, 2:3]) + 0.5
            params["G"]     = self.softplus(out[:, 3:4]) + 1e-6
            params["A_bg"]  = out[:, 4:5]  # Unbounded
            
            params["m"]     = torch.full_like(params["A_bg"], self.m)
            w0 = 2.0 * np.pi * params["f0"]
            params["k"]     = params["m"] * (w0 ** 2)
            params["c"]     = (params["m"] * w0) / (params["Q"] + 1e-8)
        else:
            params["alpha1"] = self.softplus(out[:, 0:1]) + 1e-6
            params["f0_1"]   = self.softplus(out[:, 1:2]) + 0.1
            params["Q1"]     = self.softplus(out[:, 2:3]) + 0.5
            params["G1"]     = self.softplus(out[:, 3:4]) + 1e-6
            
            params["alpha2"] = self.softplus(out[:, 4:5]) + 1e-6
            params["f0_2"]   = self.softplus(out[:, 5:6]) + 0.1
            params["Q2"]     = self.softplus(out[:, 6:7]) + 0.5
            params["G2"]     = self.softplus(out[:, 7:8]) + 1e-6
            
            params["A_bg"]   = out[:, 8:9]  # Unbounded
            params["m1"]     = torch.full_like(params["A_bg"], self.m)
            params["m2"]     = torch.full_like(params["A_bg"], self.m)
            
            w0_1 = 2.0 * np.pi * params["f0_1"]
            params["k1"]     = params["m1"] * (w0_1 ** 2)
            params["c1"]     = (params["m1"] * w0_1) / (params["Q1"] + 1e-8)
            
            w0_2 = 2.0 * np.pi * params["f0_2"]
            params["k2"]     = params["m2"] * (w0_2 ** 2)
            params["c2"]     = (params["m2"] * w0_2) / (params["Q2"] + 1e-8)
            
        return params

    def compute_X(self, N_tensor: torch.Tensor, freq_mhz_tensor: torch.Tensor) -> torch.Tensor:
        """Compute displacement magnitude spectrum |X(w, N)|."""
        p = self.get_raw_params(N_tensor)
        w = 2.0 * np.pi * freq_mhz_tensor
        w2 = w ** 2
        
        if not self.use_two_modes:
            denom_sq = (p["k"] - p["m"] * w2) ** 2 + (p["c"] * w) ** 2 + 1e-12
            X = (p["alpha"] * self.V) / torch.sqrt(denom_sq)
            return X
        else:
            denom_sq1 = (p["k1"] - p["m1"] * w2) ** 2 + (p["c1"] * w) ** 2 + 1e-12
            X1 = (p["alpha1"] * self.V) / torch.sqrt(denom_sq1)
            denom_sq2 = (p["k2"] - p["m2"] * w2) ** 2 + (p["c2"] * w) ** 2 + 1e-12
            X2 = (p["alpha2"] * self.V) / torch.sqrt(denom_sq2)
            return X1 + X2

    def forward(self, N_tensor: torch.Tensor, freq_mhz_tensor: torch.Tensor) -> torch.Tensor:
        """
        N_tensor: shape (B, 1) or (B,)
        freq_mhz_tensor: shape (B, 1) or (B, N_freqs)
        Returns predicted amplitude A_pred(f, N) = G * |X(w, N)| + A_bg.
        """
        p = self.get_raw_params(N_tensor)
        w = 2.0 * np.pi * freq_mhz_tensor
        w2 = w ** 2
        
        if not self.use_two_modes:
            denom_sq = (p["k"] - p["m"] * w2) ** 2 + (p["c"] * w) ** 2 + 1e-12
            X = (p["alpha"] * self.V) / torch.sqrt(denom_sq)
            A_pred = p["G"] * X + p["A_bg"]
        else:
            denom_sq1 = (p["k1"] - p["m1"] * w2) ** 2 + (p["c1"] * w) ** 2 + 1e-12
            X1 = (p["alpha1"] * self.V) / torch.sqrt(denom_sq1)
            denom_sq2 = (p["k2"] - p["m2"] * w2) ** 2 + (p["c2"] * w) ** 2 + 1e-12
            X2 = (p["alpha2"] * self.V) / torch.sqrt(denom_sq2)
            A_pred = p["G1"] * X1 + p["G2"] * X2 + p["A_bg"]
            
        return A_pred

    def get_physics_parameters(self, N: float) -> dict:
        """Extract SDOF parameters [m, c, k, alpha, f0_MHz, Q, G, A_bg, X_peak] for array size N."""
        self.eval()
        with torch.no_grad():
            N_t = torch.tensor([[float(N)]], dtype=torch.float32)
            p = self.get_raw_params(N_t)
            
            if not self.use_two_modes:
                c = float(p["c"].item())
                alpha = float(p["alpha"].item())
                f0 = float(p["f0"].item())
                Q = float(p["Q"].item())
                G = float(p["G"].item())
                A_bg = float(p["A_bg"].item())
                k = float(p["k"].item())
                
                w0 = 2.0 * np.pi * f0
                X_peak = float((alpha * self.V) / (c * w0 + 1e-12))
                
                return {
                    "N": N, "m": self.m, "c": c, "k": k,
                    "alpha": alpha, "f0_MHz": f0, "Q": Q, "G": G, "A_bg": A_bg,
                    "X_peak": X_peak
                }
            else:
                c1, alpha1, f0_1, Q1, G1 = float(p["c1"].item()), float(p["alpha1"].item()), float(p["f0_1"].item()), float(p["Q1"].item()), float(p["G1"].item())
                c2, alpha2, f0_2, Q2, G2 = float(p["c2"].item()), float(p["alpha2"].item()), float(p["f0_2"].item()), float(p["Q2"].item()), float(p["G2"].item())
                k1, k2 = float(p["k1"].item()), float(p["k2"].item())
                
                X_peak1 = float((alpha1 * self.V) / (c1 * (2.0 * np.pi * f0_1) + 1e-12))
                X_peak2 = float((alpha2 * self.V) / (c2 * (2.0 * np.pi * f0_2) + 1e-12))
                
                return {
                    "N": N, "A_bg": float(p["A_bg"].item()),
                    "m1": self.m, "c1": c1, "k1": k1, "alpha1": alpha1, "f0_1_MHz": f0_1, "Q1": Q1, "G1": G1, "X_peak1": X_peak1,
                    "m2": self.m, "c2": c2, "k2": k2, "alpha2": alpha2, "f0_2_MHz": f0_2, "Q2": Q2, "G2": G2, "X_peak2": X_peak2,
                }


def fit_pinn(df_all: pd.DataFrame, t_pmuts: list, m: float = 0.38, V: float = 10.0,
             use_two_modes: bool = False, epochs: int = 150, lr: float = 0.01,
             fix_m_k: bool = True, m_val: float = None, k_val: float = None) -> PhysicsInformedPMUTModel:
    """
    Fit PhysicsInformedPMUTModel on training PMUTs using Adam + MSE loss + Q-factor regularization.
    """
    if m_val is not None:
        m = float(m_val)

    model = PhysicsInformedPMUTModel(
        m=m, V=V,
        use_two_modes=use_two_modes
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare batch data from t_pmuts
    N_list, f_list, y_list = [], [], []
    for p in t_pmuts:
        pdf = df_all[df_all["N_PMUT"] == p].sort_values("Frequency_MHz")
        N_list.append(np.full((len(pdf), 1), float(p)))
        f_list.append(pdf["Frequency_MHz"].values.reshape(-1, 1))
        y_list.append(pdf["Amplitude_R_mean"].values.reshape(-1, 1))
        
    N_tr = torch.tensor(np.vstack(N_list), dtype=torch.float32)
    f_tr = torch.tensor(np.vstack(f_list), dtype=torch.float32)
    y_tr = torch.tensor(np.vstack(y_list), dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(N_tr, f_tr)
        mse_loss = nn.functional.mse_loss(pred, y_tr)
        
        # Soft regularization on Q factor
        p = model.get_raw_params(N_tr)
        if not use_two_modes:
            Q = p["Q"]
            q_reg = torch.mean(torch.relu(0.5 - Q)**2 + torch.relu(Q - 100.0)**2)
        else:
            Q1, Q2 = p["Q1"], p["Q2"]
            q_reg = torch.mean(torch.relu(0.5 - Q1)**2 + torch.relu(Q1 - 100.0)**2) + \
                    torch.mean(torch.relu(0.5 - Q2)**2 + torch.relu(Q2 - 100.0)**2)
            
        loss = mse_loss + 1e-4 * q_reg
        loss.backward()
        optimizer.step()
        
    return model


def plot_parameter_trajectories(model: PhysicsInformedPMUTModel, save_dirs: list = None):
    """
    Generate plots of [c_N, alpha_N, f0_N, k_N, Q_N, G_N, A_bg, X_peak] vs N and save to specified output paths.
    """
    if save_dirs is None:
        save_dirs = ["figures/model_parameters"]
        
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        
    pmut_range = list(range(1, 17))
    params_list = [model.get_physics_parameters(N) for N in pmut_range]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    fig.patch.set_facecolor("white")
    
    c_vals      = [p["c"] for p in params_list]
    alpha_vals  = [p["alpha"] for p in params_list]
    f0_vals     = [p["f0_MHz"] for p in params_list]
    k_vals      = [p["k"] for p in params_list]
    Q_vals      = [p["Q"] for p in params_list]
    G_vals      = [p["G"] for p in params_list]
    A_bg_vals   = [p["A_bg"] for p in params_list]
    X_peak_vals = [p["X_peak"] for p in params_list]
    
    # Panel 1: Damping c_N
    axes[0, 0].plot(pmut_range, c_vals, "s-", color="#E65100", lw=2)
    axes[0, 0].set_title("Derived Damping ($c_N$) vs N", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("$c_N$")
    
    # Panel 2: Force Coefficient alpha_N
    axes[0, 1].plot(pmut_range, alpha_vals, "o-", color="#1565C0", lw=2)
    axes[0, 1].set_title("Force Coefficient ($\\alpha_N$) vs N", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("$\\alpha_N$")
    
    # Panel 3: Resonant Frequency f0_N
    axes[0, 2].plot(pmut_range, f0_vals, "*-", color="#2E7D32", lw=2)
    axes[0, 2].set_title("Resonant Freq ($f_{0,N}$) vs N", fontsize=10, fontweight="bold")
    axes[0, 2].set_ylabel("$f_{0,N}$ (MHz)")
    
    # Panel 4: Derived Stiffness k_N
    axes[0, 3].plot(pmut_range, k_vals, "^-", color="#00897B", lw=2)
    axes[0, 3].set_title("Derived Stiffness ($k_N$) vs N", fontsize=10, fontweight="bold")
    axes[0, 3].set_ylabel("$k_N$")
    
    # Panel 5: Quality Factor Q_N
    axes[1, 0].plot(pmut_range, Q_vals, "d-", color="#6A1B9A", lw=2)
    axes[1, 0].set_title("Quality Factor ($Q_N$) vs N", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("$Q_N$")
    axes[1, 0].set_xlabel("PMUT Array Size (N)")
    
    # Panel 6: Gain Parameter G_N
    axes[1, 1].plot(pmut_range, G_vals, "p-", color="#F57F17", lw=2)
    axes[1, 1].set_title("Gain Parameter ($G_N$) vs N", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("$G_N$")
    axes[1, 1].set_xlabel("PMUT Array Size (N)")
    
    # Panel 7: Background Offset A_bg
    axes[1, 2].plot(pmut_range, A_bg_vals, "v-", color="#D81B60", lw=2)
    axes[1, 2].set_title("Background Offset ($A_{bg}$) vs N", fontsize=10, fontweight="bold")
    axes[1, 2].set_ylabel("$A_{bg}$")
    axes[1, 2].set_xlabel("PMUT Array Size (N)")
    
    # Panel 8: Peak Displacement X_peak
    axes[1, 3].plot(pmut_range, X_peak_vals, "h-", color="#00ACC1", lw=2)
    axes[1, 3].set_title("Peak Mechanical Displ ($|X|_{peak}$) vs N", fontsize=10, fontweight="bold")
    axes[1, 3].set_ylabel("$|X|_{peak}$")
    axes[1, 3].set_xlabel("PMUT Array Size (N)")
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pmut_range)
        
    plt.suptitle("PINN NN Parameters [$\\alpha_N, f_{0,N}, Q_N, G_N, A_{bg}$] & Derived Mechanical Parameters ($c_N, k_N, |X|_{peak}$) vs N",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    
    filename = "pinn_parameter_trajectories.png"
    for sdir in save_dirs:
        out_path = os.path.join(sdir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def fit_sdof_directly(
    freqs: np.ndarray, 
    actual: np.ndarray, 
    m_val: float = 0.38, 
    V: float = 10.0, 
    k_val: float = None,
    epochs: int = 1000,
    lr: float = 0.01
) -> dict:
    """
    Directly fits an SDOF model to frequency response data using PyTorch (Double Precision).
    """
    # Pre-allocate static tensors outside the optimization loop
    w = torch.tensor(2.0 * np.pi * freqs, dtype=torch.float64)
    y_true = torch.tensor(actual, dtype=torch.float64)
    var_y_true = torch.var(y_true) + 1e-12
    p_m = float(m_val)
    
    # 1. Parameter initializations
    f0_init = float(freqs[np.argmax(actual)])
    abg_init = float(np.min(actual))
    
    # Inverse softplus: inv_softplus(y) = ln(exp(y) - 1) -> y for y > 20
    raw_f0_init = np.log(np.expm1(f0_init)) if f0_init < 20.0 else f0_init
    
    # Optimize [raw_alpha, raw_f0, raw_Q, raw_Abg]
    raw_params = nn.Parameter(torch.tensor([0.5, raw_f0_init, 2.3, abg_init], dtype=torch.float64))
    optimizer = optim.Adam([raw_params], lr=lr)
    softplus = nn.Softplus()

    # Pre-allocate constant stiffness tensor if supplied
    fixed_k_tensor = torch.tensor(float(k_val), dtype=torch.float64) if k_val is not None else None

    for _ in range(epochs):
        optimizer.zero_grad()
        
        # Parameter transformations
        p_alpha = softplus(raw_params[0]) + 1e-12
        p_Q     = softplus(raw_params[2]) + 0.1
        p_Abg   = raw_params[3]
        
        if fixed_k_tensor is None:
            p_f0 = softplus(raw_params[1]) + 1e-6
            w0 = 2.0 * np.pi * p_f0
            p_k = p_m * (w0 ** 2)
        else:
            p_k = fixed_k_tensor
            w0 = torch.sqrt(p_k / p_m)
            p_f0 = w0 / (2.0 * np.pi)
            
        p_c = (p_m * w0) / (p_Q + 1e-12)
        
        # Dynamic response calculation
        w2 = w ** 2
        denom_sq = (p_k - p_m * w2) ** 2 + (p_c * w) ** 2 + 1e-18
        X = (p_alpha * V) / torch.sqrt(denom_sq)
        y_pred = X + p_Abg
        
        # Relative MSE loss + soft boundary penalty
        loss = nn.functional.mse_loss(y_pred, y_true) / var_y_true
        q_reg = torch.relu(0.1 - p_Q)**2 + torch.relu(p_Q - 500.0)**2
        
        total_loss = loss + 1e-3 * q_reg
        total_loss.backward()
        optimizer.step()
        
    # Extract final parameters
    with torch.no_grad():
        p_alpha = (softplus(raw_params[0]) + 1e-12).item()
        p_Q     = (softplus(raw_params[2]) + 0.1).item()
        p_Abg   = raw_params[3].item()
        
        if k_val is None:
            p_f0 = (softplus(raw_params[1]) + 1e-6).item()
            w0 = 2.0 * np.pi * p_f0
            p_k = p_m * (w0 ** 2)
        else:
            p_k = float(k_val)
            w0 = np.sqrt(p_k / p_m)
            p_f0 = w0 / (2.0 * np.pi)
            
        p_c = (p_m * w0) / (p_Q + 1e-12)
        X_peak = float((p_alpha * V) / (p_c * w0 + 1e-12))

    return {
        "m": p_m,
        "c": float(p_c),
        "k": float(p_k),
        "alpha": p_alpha,
        "f0": float(p_f0),
        "f0_MHz": float(p_f0),
        "Q": float(p_Q),
        "G": 1.0,
        "A_bg": float(p_Abg),
        "X_peak": X_peak
    }

def generate_physics_comparison(model: PhysicsInformedPMUTModel, df_all: pd.DataFrame, save_dirs: list = None):
    """
    Fit SDOF directly to each PMUT curve to find 'true' parameters, compare against PINN predictions.
    Saves results to CSV and HTML formats.
    """
    if save_dirs is None:
        save_dirs = ["results/direct_fits"]
        
    pmut_range = sorted(df_all["N_PMUT"].unique())
    records = []
    
    print("\n── Computing True SDOF Parameters via Direct Fit ─────────────────")
    for N in pmut_range:
        pdf = df_all[df_all["N_PMUT"] == N].sort_values("Frequency_MHz")
        freqs = pdf["Frequency_MHz"].values
        actual = pdf["Amplitude_R_mean"].values
        
        # True parameters via direct fit
        t_params = fit_sdof_directly(freqs, actual, m_val=model.m, V=model.V)
        
        # Estimated parameters via PINN model
        e_params = model.get_physics_parameters(N)
        
        records.append({
            "N_PMUT": N,
            "m_True": t_params["m"], "m_Est": e_params["m"],
            "c_True": t_params["c"], "c_Est": e_params["c"],
            "f0_True_MHz": t_params["f0_MHz"], "f0_Est_MHz": e_params["f0_MHz"],
            "k_True": t_params["k"], "k_Est": e_params["k"],
            "alpha_True": t_params["alpha"], "alpha_Est": e_params["alpha"],
            "G_True": t_params["G"], "G_Est": e_params["G"],
            "A_bg_True": t_params["A_bg"], "A_bg_Est": e_params["A_bg"],
            "Q_True": t_params["Q"], "Q_Est": e_params["Q"],
            "X_peak_True": t_params["X_peak"], "X_peak_Est": e_params["X_peak"],
        })
        print(f"  PMUT {N:>2} fitted & compared ✓")
        
    df_comp = pd.DataFrame(records)
    
    # Save CSVs
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        csv_path = os.path.join(sdir, "pinn_physics_comparison.csv")
        df_comp.to_csv(csv_path, index=False)
        
    # Generate interactive HTML table comparison
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PMUT SDOF Physics Parameter Comparison</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1500px; margin: auto; padding: 20px; background: #F8F9FA; color: #37474F; }
        h1 { color: #1565C0; margin-bottom: 5px; }
        p.subtitle { color: #666; margin-bottom: 25px; font-size: 0.95em; }
        table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
        th, td { padding: 7px 10px; text-align: center; border: 1px solid #E0E0E0; }
        th { background: #37474F; color: white; font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.3px; }
        tr:nth-child(even) { background: #F9F9F9; }
        .hdr-group { background: #CFD8DC; font-weight: bold; color: #263238; }
        .diff { font-size: 10.5px; color: #D81B60; font-weight: 500; }
    </style>
</head>
<body>
    <h1>PMUT SDOF Physics Parameter Comparison</h1>
    <p class="subtitle">
        Comparison of <b>True SDOF parameters</b> (obtained via direct gradient descent fit to the experimental response curves) 
        vs. <b>Estimated SDOF parameters</b> predicted by the Physics-Informed Neural Network (PINN).
    </p>
    <table>
        <thead>
            <tr class="hdr-group">
                <th>PMUT</th>
                <th colspan="3">Mass (m)</th>
                <th colspan="3">Resonant Freq (f0 MHz)</th>
                <th colspan="3">Quality Factor (Q)</th>
                <th colspan="3">Damping (c)</th>
                <th colspan="3">Stiffness (k)</th>
                <th colspan="3">Force Coeff (&alpha;)</th>
                <th colspan="3">Gain (G)</th>
                <th colspan="3">Background (A<sub>bg</sub>)</th>
                <th colspan="3">Peak Displ (|X|<sub>peak</sub>)</th>
            </tr>
            <tr>
                <th>N</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
                <th>True</th><th>Est</th><th>Diff %</th>
            </tr>
        </thead>
        <tbody>
"""
    for _, r in df_comp.iterrows():
        n = int(r["N_PMUT"])
        
        def cell_vals(true, est):
            diff_p = abs(true - est) / true * 100 if true > 0 else 0
            return f"<td>{true:.5f}</td><td>{est:.5f}</td><td class='diff'>{diff_p:.1f}%</td>"
            
        html_content += f"<tr><td>P{n}</td>"
        html_content += cell_vals(r["m_True"], r["m_Est"])
        html_content += cell_vals(r["f0_True_MHz"], r["f0_Est_MHz"])
        html_content += cell_vals(r["Q_True"], r["Q_Est"])
        html_content += cell_vals(r["c_True"], r["c_Est"])
        html_content += cell_vals(r["k_True"], r["k_Est"])
        html_content += cell_vals(r["alpha_True"], r["alpha_Est"])
        html_content += cell_vals(r["G_True"], r["G_Est"])
        html_content += cell_vals(r["A_bg_True"], r["A_bg_Est"])
        html_content += cell_vals(r["X_peak_True"], r["X_peak_Est"])
        html_content += "</tr>\n"
        
    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    for sdir in save_dirs:
        html_path = os.path.join(sdir, "pinn_physics_comparison.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    print("Saved comparison HTML table → results/direct_fits/")


def compute_sdof_amplitude(freqs: np.ndarray, params: dict, V: float = 10.0) -> np.ndarray:
    """
    Compute theoretical SDOF amplitude curve given frequency array and SDOF parameters dict
    containing {m, c, k, alpha, G, A_bg}.
    """
    w = 2.0 * np.pi * freqs
    denom_sq = (params["k"] - params["m"] * (w ** 2)) ** 2 + (params["c"] * w) ** 2 + 1e-12
    X = (params["alpha"] * V) / np.sqrt(denom_sq)
    return params["G"] * X + params["A_bg"]
