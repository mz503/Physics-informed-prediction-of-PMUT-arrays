"""
Helpers/pinn_informed_loss_model.py
══════════════════════════════════════════════════════════════════════════════
Physics-Informed Neural Network with Informed Loss for PMUT Arrays.

The network predicts physical parameters theta_N = [m_N, c_N, k_N, Q_N, A_N, A_bg]
for array size N, while a differentiable oscillator / equivalent-circuit layer
calculates the complete frequency response:
  H_i(w, N) = (A_i(N) * w0_i(N)^2) / ((w0_i(N)^2 - w^2) + j * (w0_i(N) * w / Q_i(N)))
  y_pred(f, N) = | sum_{i=1}^K H_i(w, N) | + A_bg(N)

where w0_i = sqrt(k_i / m_i) and Q_i = sqrt(m_i * k_i) / c_i.

Training uses a composite physics-informed loss:
  L = lambda_d * L_data + lambda_p * L_physics + lambda_s * L_smooth

where:
  L_data = (1 / M) * sum_{i=1}^M (A_pred,i - A_exp,i)^2
  L_physics enforces physical positivity (m_N > 0, c_N > 0, k_N > 0, Q_N > 0) and Q_N consistency
  L_smooth = sum_{N=1}^{15} || theta_{N+1} - theta_N ||^2  (smoothness across array sizes)
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


class PINNInformedLossPMUTModel(nn.Module):
    """
    PINN model mapping array size N -> physical parameters theta_N:
      [A_bg, (m_i, c_i, k_i, Q_i, A_i) for i=1..K]
    and computing frequency response amplitude via a Differentiable Oscillator Layer.
    """
    def __init__(self, num_modes: int = 3, hidden_dim: int = 64,
                 abg_init: float = 1.25, f0_inits: list = None):
        super().__init__()
        self.num_modes = num_modes

        if f0_inits is None or len(f0_inits) < num_modes:
            f0_inits = [0.47, 1.25, 4.3][:num_modes]

        self.abg_init = float(abg_init)
        self.f0_inits = [float(f) for f in f0_inits]

        # Output per mode: [log_m_rel, log_c_rel, log_k_rel, Q_raw, A_raw] -> 5 * num_modes + 1 (A_bg)
        n_out = 1 + 5 * num_modes

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
        """
        Map array size N -> physical parameters theta_N.
        Returns dict containing:
          - A_bg: baseline offset
          - m: modal effective masses [m_1, ..., m_K] (strictly positive)
          - c: modal damping coefficients [c_1, ..., c_K] (strictly positive)
          - k: modal stiffnesses [k_1, ..., k_K] (strictly positive)
          - Q: quality factors [Q_1, ..., Q_K] (strictly positive)
          - f0: resonant frequencies in MHz [f0_1, ..., f0_K]
          - A: modal amplitudes [A_1, ..., A_K]
          - theta_norm: concatenated normalized parameter tensor for smoothness loss
        """
        if N_tensor.ndim == 1:
            N_tensor = N_tensor.unsqueeze(-1)

        N_scaled = N_tensor.float() / 16.0
        out = self.net(N_scaled)

        # Baseline offset bounded within abg_init +/- 0.08
        A_bg = self.abg_init + 0.08 * torch.tanh(out[:, 0:1])

        params = {
            "A_bg": A_bg,
            "m": [],
            "c": [],
            "k": [],
            "Q": [],
            "f0": [],
            "A": [],
        }

        theta_norm_components = [A_bg / (self.abg_init + 1e-6)]

        for i in range(self.num_modes):
            idx_st = 1 + 5 * i

            # Modal amplitude A_i in range [1e-6, 0.0025]
            A_i = 1e-6 + 0.00249 * torch.sigmoid(out[:, idx_st : idx_st + 1])

            # Resonant frequency f0_i in range [f0_init - 0.20, f0_init + 0.20] (MHz)
            f0_center = self.f0_inits[i]
            f0_i = f0_center + 0.20 * torch.tanh(out[:, idx_st + 1 : idx_st + 2])

            # Quality factor Q_i in range [10.0, 45.0]
            Q_i = 10.0 + 35.0 * torch.sigmoid(out[:, idx_st + 2 : idx_st + 3])

            # Physical parameters derivation (m, c, k):
            # Base mass m_ref = 1.0e-9 kg
            m_ref = 1.0e-9
            # Relative mass factor in range [0.5, 2.0]
            m_rel = 0.5 + 1.5 * torch.sigmoid(out[:, idx_st + 3 : idx_st + 4])
            m_i = m_ref * m_rel

            # w0_i in rad/s (f0 in MHz -> w0 = 2*pi*f0 * 1e6 rad/s)
            w0_rad = 2.0 * np.pi * f0_i * 1.0e6
            k_i = m_i * (w0_rad ** 2)
            c_i = (m_i * w0_rad) / (Q_i + 1e-8)

            params["A"].append(A_i)
            params["f0"].append(f0_i)
            params["Q"].append(Q_i)
            params["m"].append(m_i)
            params["k"].append(k_i)
            params["c"].append(c_i)

            # Normalized parameter representations for smoothness penalty L_smooth
            theta_norm_components.extend([
                f0_i / f0_center,
                Q_i / 25.0,
                A_i / 0.001,
                m_rel,
            ])

        params["theta_norm"] = torch.cat(theta_norm_components, dim=-1)
        return params

    def forward(self, N_tensor: torch.Tensor, freq_mhz_tensor: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Oscillator Layer:
        Calculates complete frequency response amplitude A_pred(f, N) from predicted physical parameters.
        """
        if N_tensor.ndim == 1:
            N_tensor = N_tensor.unsqueeze(-1)
        if freq_mhz_tensor.ndim == 1:
            freq_mhz_tensor = freq_mhz_tensor.unsqueeze(-1)

        p = self.get_raw_params(N_tensor)
        w = 2.0 * np.pi * freq_mhz_tensor  # rad/s in MHz scale (w in MHz rad)

        h_total_re = torch.zeros_like(w)
        h_total_im = torch.zeros_like(w)

        for i in range(self.num_modes):
            A_i = p["A"][i]
            f0_i = p["f0"][i]
            Q_i = p["Q"][i]

            w0 = 2.0 * np.pi * f0_i
            w02 = w0 ** 2

            re_denom = w02 - (w ** 2)
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

    def compute_losses(self, N_tr: torch.Tensor, f_tr: torch.Tensor, y_tr: torch.Tensor,
                       lambda_d: float = 1.0, lambda_p: float = 0.1, lambda_s: float = 0.01) -> dict:
        """
        Compute total physics-informed loss:
          L = lambda_d * L_data + lambda_p * L_physics + lambda_s * L_smooth
        """
        # 1. Data Loss L_data
        y_pred = self.forward(N_tr, f_tr)
        L_data = torch.mean((y_pred - y_tr) ** 2)

        # 2. Physics Restrictions Loss L_physics (m > 0, c > 0, k > 0, Q > 0 and Q consistency)
        # Evaluate parameters across array sizes N = 1..16
        N_all = torch.arange(1, 17, dtype=torch.float32).unsqueeze(-1)
        p_all = self.get_raw_params(N_all)

        L_physics = torch.tensor(0.0, device=N_tr.device)
        for i in range(self.num_modes):
            m_i = p_all["m"][i]
            c_i = p_all["c"][i]
            k_i = p_all["k"][i]
            Q_i = p_all["Q"][i]

            # Positivity penalty (ReLU(-x))
            pos_penalty = (torch.relu(-m_i).mean() +
                           torch.relu(-c_i).mean() +
                           torch.relu(-k_i).mean() +
                           torch.relu(-Q_i).mean())

            # Physical consistency penalty Q_calc vs Q_pred
            # Q_calc = sqrt(m * k) / c
            w0_rad = 2.0 * np.pi * p_all["f0"][i] * 1.0e6
            Q_calc = (m_i * w0_rad) / (c_i + 1e-12)
            cons_penalty = torch.mean(((Q_i - Q_calc) / (Q_i + 1e-6)) ** 2)

            L_physics = L_physics + pos_penalty + cons_penalty

        # 3. Smoothness Regularization Loss L_smooth between neighboring array sizes N
        theta_all = p_all["theta_norm"]  # shape (16, num_param_dims)
        theta_diff = theta_all[1:] - theta_all[:-1]  # shape (15, num_param_dims)
        L_smooth = torch.sum(theta_diff ** 2)

        # Total combined loss
        L_total = lambda_d * L_data + lambda_p * L_physics + lambda_s * L_smooth

        return {
            "L_total": L_total,
            "L_data": L_data,
            "L_physics": L_physics,
            "L_smooth": L_smooth,
        }

    def get_physics_parameters(self, N: float) -> dict:
        """Extract physical parameters [A_bg, m, c, k, Q, f0, A] for array size N."""
        self.eval()
        with torch.no_grad():
            N_t = torch.tensor([[float(N)]], dtype=torch.float32)
            p = self.get_raw_params(N_t)

            res = {
                "N": float(N),
                "A_bg": float(p["A_bg"].item()),
                "m": [float(m.item()) for m in p["m"]],
                "c": [float(c.item()) for c in p["c"]],
                "k": [float(k.item()) for k in p["k"]],
                "Q": [float(q.item()) for q in p["Q"]],
                "f0": [float(f.item()) for f in p["f0"]],
                "A": [float(a.item()) for a in p["A"]],
            }
            return res


def fit_pinn_informed_loss(df_all: pd.DataFrame, t_pmuts: list, num_modes: int = 3,
                           epochs: int = 350, lr: float = 0.005,
                           lambda_d: float = 1.0, lambda_p: float = 0.1,
                           lambda_s: float = 0.01) -> PINNInformedLossPMUTModel:
    """
    Fit PINNInformedLossPMUTModel on training PMUTs using Adam + Informed Loss + peak initialization.
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

    model = PINNInformedLossPMUTModel(
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
        losses = model.compute_losses(N_tr, f_tr, y_tr, lambda_d=lambda_d, lambda_p=lambda_p, lambda_s=lambda_s)
        loss = losses["L_total"]
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model


def plot_pinn_informed_loss_trajectories(model: PINNInformedLossPMUTModel, save_dirs: list = None):
    """
    Generate parameter trajectory plots for PINN-Informed-Loss predictions across N = 1..16.
    """
    if save_dirs is None:
        save_dirs = ["figures/model_parameters"]

    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)

    pmut_range = list(range(1, 17))
    params_list = [model.get_physics_parameters(N) for N in pmut_range]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("white")

    abg_vals = [p["A_bg"] for p in params_list]

    axes[0, 0].plot(pmut_range, abg_vals, "o-", color="#D81B60", lw=2)
    axes[0, 0].set_title("Baseline Offset ($A_{bg}$) vs N", fontweight="bold")
    axes[0, 0].set_ylabel("$A_{bg}$")

    colors_mode = ["#1565C0", "#2E7D32", "#6A1B9A"]
    num_modes = model.num_modes

    for m in range(num_modes):
        f0_m = [p["f0"][m] for p in params_list]
        q_m = [p["Q"][m] for p in params_list]
        a_m = [p["A"][m] for p in params_list]
        m_m = [p["m"][m] * 1e9 for p in params_list]  # in nkg

        axes[0, 1].plot(pmut_range, f0_m, "o-", color=colors_mode[m % 3], label=f"Mode {m+1}")
        axes[0, 2].plot(pmut_range, q_m, "s-", color=colors_mode[m % 3], label=f"Mode {m+1}")
        axes[1, 0].plot(pmut_range, m_m, "d-", color=colors_mode[m % 3], label=f"Mode {m+1}")
        axes[1, 1].plot(pmut_range, a_m, "^-", color=colors_mode[m % 3], label=f"Mode {m+1}")

    axes[0, 1].set_title("Resonant Frequency ($f_0$) vs N", fontweight="bold")
    axes[0, 1].set_ylabel("f0 (MHz)")
    axes[0, 1].legend()

    axes[0, 2].set_title("Quality Factor ($Q$) vs N", fontweight="bold")
    axes[0, 2].set_ylabel("Q")
    axes[0, 2].legend()

    axes[1, 0].set_title("Effective Mass ($m$) vs N", fontweight="bold")
    axes[1, 0].set_ylabel("Mass $m$ (nkg)")
    axes[1, 0].legend()

    axes[1, 1].set_title("Modal Amplitude ($A$) vs N", fontweight="bold")
    axes[1, 1].set_ylabel("$A$")
    axes[1, 1].legend()

    axes[1, 2].axis("off")

    for ax in axes.flat:
        if ax != axes[1, 2]:
            ax.grid(True, alpha=0.3)
            ax.set_xticks(pmut_range)
            ax.set_xlabel("PMUT Array Size (N)")

    plt.suptitle("PINN-Informed-Loss Predicted Physical Parameters vs PMUT Array Size (N)", fontweight="bold", fontsize=12)
    plt.tight_layout()

    filename = "pinn_informed_loss_parameter_trajectories.png"
    for sdir in save_dirs:
        out_path = os.path.join(sdir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved PINN-Informed-Loss parameter trajectory plot → figures/model_parameters/")
