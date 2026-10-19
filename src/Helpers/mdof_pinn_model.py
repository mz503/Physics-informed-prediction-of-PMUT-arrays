"""
Helpers/mdof_pinn_model.py
══════════════════════════════════════════════════════════════════════════════
Multi-Mode Mechanical Physics-Informed Neural Network (MDOF-PINN) for PMUTs.

Uses normalized modal transfer function:
  H_i(w, N) = (A_i(N) * w0_i(N)^2) / ((w0_i(N)^2 - w^2) + j * (w0_i(N) * w / Q_i(N)))
  y_pred(f, N) = | sum_{i=1}^N H_i(w, N) | + A_bg(N)

At resonance (w = w0_i), |H_i(w0_i)| = A_i * Q_i.
Maps PMUT array size N ──► [A_bg(N), A_1(N), f0_1(N), Q_1(N), ..., A_N(N), f0_N(N), Q_N(N)].
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PhysicsInformedMDOFPMUTModel(nn.Module):
    """
    Physics-Informed Neural Network mapping array size N to multi-mode mechanical parameters:
      [A_bg, A_1, f0_1, Q_1, ..., A_N, f0_N, Q_N]
    """
    def __init__(self, num_modes: int = 3, hidden_dim: int = 64,
                 abg_init: float = 1.25, f0_inits: list = None):
        super().__init__()
        self.num_modes = num_modes

        if f0_inits is None or len(f0_inits) < num_modes:
            f0_inits = [0.47, 1.25, 4.3][:num_modes]

        self.abg_init = float(abg_init)
        self.f0_inits = [float(f) for f in f0_inits]

        # 1 global baseline parameter (A_bg) + 3 parameters per mode (A_i, f0_i, Q_i)
        n_out = 1 + 3 * num_modes

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_out)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_raw_params(self, N_tensor: torch.Tensor) -> dict:
        if N_tensor.ndim == 1:
            N_tensor = N_tensor.unsqueeze(-1)

        N_scaled = N_tensor.float() / 16.0
        out = self.net(N_scaled)

        # Baseline offset bounded within +/- 0.08 of abg_init
        A_bg = self.abg_init + 0.08 * torch.tanh(out[:, 0:1])

        params = {
            "A_bg": A_bg,
            "f0": [],
            "Q": [],
            "A": [],
        }

        for i in range(self.num_modes):
            idx_st = 1 + 3 * i

            # Modal amplitude A_i in range [1e-6, 0.0025]
            A_i = 1e-6 + 0.00249 * torch.sigmoid(out[:, idx_st : idx_st + 1])

            # Resonant frequency f0_i in range [f0_init - 0.20, f0_init + 0.20]
            f0_center = self.f0_inits[i]
            f0_i = f0_center + 0.20 * torch.tanh(out[:, idx_st + 1 : idx_st + 2])

            # Quality factor Q_i in range [10.0, 45.0]
            Q_i = 10.0 + 35.0 * torch.sigmoid(out[:, idx_st + 2 : idx_st + 3])

            params["A"].append(A_i)
            params["f0"].append(f0_i)
            params["Q"].append(Q_i)

        return params

    def forward(self, N_tensor: torch.Tensor, freq_mhz_tensor: torch.Tensor) -> torch.Tensor:
        if N_tensor.ndim == 1:
            N_tensor = N_tensor.unsqueeze(-1)
        if freq_mhz_tensor.ndim == 1:
            freq_mhz_tensor = freq_mhz_tensor.unsqueeze(-1)

        p = self.get_raw_params(N_tensor)
        w = 2.0 * np.pi * freq_mhz_tensor
        w2 = w ** 2

        h_total_re = torch.zeros_like(w)
        h_total_im = torch.zeros_like(w)

        for i in range(self.num_modes):
            A_i = p["A"][i]
            f0_i = p["f0"][i]
            Q_i = p["Q"][i]

            w0 = 2.0 * np.pi * f0_i
            w02 = w0 ** 2

            re_denom = w02 - w2
            im_denom = (w0 * w) / (Q_i + 1e-8)
            denom_sq = (re_denom ** 2) + (im_denom ** 2) + 1e-12

            num_real = A_i * w02
            h_re = (num_real * re_denom) / denom_sq
            h_im = (-num_real * im_denom) / denom_sq

            h_total_re = h_total_re + h_re
            h_total_im = h_total_im + h_im

        y_pred_norm = torch.sqrt(h_total_re ** 2 + h_total_im ** 2 + 1e-12)
        y_pred_actual = y_pred_norm + p["A_bg"]
        return y_pred_actual

    def get_physics_parameters(self, N: float) -> dict:
        """Extract MDOF parameters [A_bg, f0, Q, A] for array size N."""
        self.eval()
        with torch.no_grad():
            N_t = torch.tensor([[float(N)]], dtype=torch.float32)
            p = self.get_raw_params(N_t)

            res = {
                "N": float(N),
                "A_bg": float(p["A_bg"].item()),
                "f0": [float(f.item()) for f in p["f0"]],
                "Q": [float(q.item()) for q in p["Q"]],
                "A": [float(a.item()) for a in p["A"]],
            }
            return res


def fit_mdof_pinn(df_all: pd.DataFrame, t_pmuts: list, num_modes: int = 3, m_vals: list = None,
                  V: float = 10.0, epochs: int = 1000, lr: float = 0.005) -> PhysicsInformedMDOFPMUTModel:
    """
    Fit PhysicsInformedMDOFPMUTModel on training PMUTs using Adam + spectral peak initialization.
    """
    N_list, f_list, y_list = [], [], []
    abg_inits = []
    all_y_norm = []

    for p in t_pmuts:
        pdf = df_all[df_all["N_PMUT"] == p].sort_values("Frequency_MHz")
        freqs_p = pdf["Frequency_MHz"].values
        y_p = pdf["Amplitude_R_mean"].values

        abg_p = np.min(y_p)
        abg_inits.append(abg_p)

        y_norm_p = y_p - abg_p
        all_y_norm.append(y_norm_p)

        N_list.append(np.full((len(pdf), 1), float(p)))
        f_list.append(freqs_p.reshape(-1, 1))
        y_list.append(y_p.reshape(-1, 1))

    mean_abg = float(np.mean(abg_inits))

    # Spectral peak detection on training PMUT dataset
    avg_y_norm = np.mean(all_y_norm, axis=0)
    ref_freqs = df_all[df_all["N_PMUT"] == t_pmuts[0]].sort_values("Frequency_MHz")["Frequency_MHz"].values

    min_dist = max(1, len(avg_y_norm) // (num_modes * 4))
    peaks, _ = find_peaks(avg_y_norm, distance=min_dist, prominence=0.001)

    if len(peaks) >= num_modes:
        top_idx = peaks[np.argsort(avg_y_norm[peaks])[-num_modes:]]
        top_idx = np.sort(top_idx)
    else:
        top_idx = []
        chunk = len(avg_y_norm) // num_modes
        for i in range(num_modes):
            st, en = i * chunk, (i + 1) * chunk if i < num_modes - 1 else len(avg_y_norm)
            top_idx.append(st + int(np.argmax(avg_y_norm[st:en])))
        top_idx = np.array(top_idx)

    f0_inits = list(ref_freqs[top_idx])

    model = PhysicsInformedMDOFPMUTModel(
        num_modes=num_modes, abg_init=mean_abg, f0_inits=f0_inits
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    N_tr = torch.tensor(np.vstack(N_list), dtype=torch.float32)
    f_tr = torch.tensor(np.vstack(f_list), dtype=torch.float32)
    y_tr = torch.tensor(np.vstack(y_list), dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(N_tr, f_tr)
        loss = nn.functional.mse_loss(pred, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model


def plot_mdof_pinn_trajectories(model: PhysicsInformedMDOFPMUTModel, save_dirs: list = None):
    """
    Generate parameter trajectory plots for MDOF-PINN predictions across N = 1..16.
    """
    if save_dirs is None:
        save_dirs = ["figures/model_parameters"]

    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)

    pmut_range = list(range(1, 17))
    params_list = [model.get_physics_parameters(N) for N in pmut_range]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    abg_vals = [p["A_bg"] for p in params_list]

    axes[0, 0].plot(pmut_range, abg_vals, "o-", color="#8E24AA", lw=2)
    axes[0, 0].set_title("MDOF-PINN Baseline ($A_{bg}$) vs N", fontweight="bold")
    axes[0, 0].set_ylabel("$A_{bg}$")

    colors_mode = ["#1565C0", "#2E7D32", "#6A1B9A"]
    num_modes = model.num_modes

    for m in range(num_modes):
        f0_m = [p["f0"][m] for p in params_list]
        q_m = [p["Q"][m] for p in params_list]
        a_m = [p["A"][m] for p in params_list]

        axes[1, 0].plot(pmut_range, f0_m, "o-", color=colors_mode[m % 3], label=f"Mode {m+1}")
        axes[1, 1].plot(pmut_range, q_m, "s-", color=colors_mode[m % 3], label=f"Mode {m+1}")
        axes[1, 2].plot(pmut_range, a_m, "^-", color=colors_mode[m % 3], label=f"Mode {m+1}")

    axes[1, 0].set_title("Resonant Frequency ($f_0$) vs N", fontweight="bold")
    axes[1, 0].set_ylabel("f0 (MHz)")
    axes[1, 0].legend()

    axes[1, 1].set_title("Quality Factor ($Q$) vs N", fontweight="bold")
    axes[1, 1].set_ylabel("Q")
    axes[1, 1].legend()

    axes[1, 2].set_title("Modal Amplitude ($A$) vs N", fontweight="bold")
    axes[1, 2].set_ylabel("$A$")
    axes[1, 2].legend()

    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for ax in axes.flat:
        if ax not in [axes[0, 1], axes[0, 2]]:
            ax.grid(True, alpha=0.3)
            ax.set_xticks(pmut_range)
            ax.set_xlabel("PMUT Array Size (N)")

    plt.suptitle("MDOF-PINN Predicted Parameters vs PMUT Array Size (N)", fontweight="bold", fontsize=12)
    plt.tight_layout()

    filename = "mdof_pinn_parameter_trajectories.png"
    for sdir in save_dirs:
        out_path = os.path.join(sdir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved MDOF-PINN parameter trajectory plot → figures/model_parameters/")
