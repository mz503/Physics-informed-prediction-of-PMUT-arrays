"""
Helpers/mdof_model.py
══════════════════════════════════════════════════════════════════════════════
Multi-Mode Mechanical Superposition (MDOF) model fitting for PMUT displacement /
reflectance amplitude data using PyTorch double precision (torch.float64).

Mathematical Model:
  A_bg,init = min(y_true)
  y_centered = y_true - A_bg,init
  Scale Factor (S) = max(y_centered)
  y_norm = y_centered / S

  H_i(w) = (alpha_i * V) / ((k_i - m_i * w^2) + j * (c_i * w))
  y_pred,norm(w) = | sum_{i=1}^N H_i(w) |
  y_pred,actual(w) = (y_pred,norm(w) * S) + A_bg
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


def _inv_softplus(y: float) -> float:
    """Inverse softplus: softplus_inv(y) = log(exp(y) - 1)."""
    y_val = max(float(y), 1e-12)
    return float(np.log(np.expm1(y_val))) if y_val < 20.0 else y_val


def fit_mdof_pmut(
    freqs: np.ndarray,
    actual: np.ndarray,
    num_modes: int = 3,
    m_vals: list = None,
    V: float = 10.0,
    epochs: int = 3000,
    lr: float = 0.01,
) -> dict:
    """
    Fit a Multi-Mode Mechanical Superposition (MDOF) model to PMUT displacement/reflectance data
    using PyTorch double precision (torch.float64).

    Parameters
    ----------
    freqs     : np.ndarray
                Frequency points (e.g., in MHz).
    actual    : np.ndarray
                Measured displacement/reflectance amplitude R.
    num_modes : int, optional
                Number of uncoupled mechanical modes N (default 3).
    m_vals    : list of float, optional
                Mass values per mode (default [0.38] * num_modes).
    V         : float, optional
                Excitation voltage (default 10.0 V).
    epochs    : int, optional
                Number of optimization iterations (default 3000).
    lr        : float, optional
                Learning rate for Adam optimizer (default 0.01).

    Returns
    -------
    dict containing:
        - "f0"           : list of float, resonant frequencies per mode
        - "Q"            : list of float, quality factors per mode
        - "k"            : list of float, stiffness parameters per mode
        - "c"            : list of float, damping coefficients per mode
        - "alpha"        : list of float, force coefficients per mode
        - "A_bg"         : float, background baseline offset
        - "S"            : float, scale factor
        - "y_pred_actual": np.ndarray, rescaled predictions matching raw input scale
        - "y_pred_norm"  : np.ndarray, normalized predictions
        - "loss"         : float, final normalized MSE loss
    """
    if m_vals is None:
        m_vals = [0.38] * num_modes
    elif len(m_vals) < num_modes:
        m_vals = list(m_vals) + [m_vals[-1]] * (num_modes - len(m_vals))

    m_tensors = [torch.tensor(float(m_vals[i]), dtype=torch.float64) for i in range(num_modes)]

    # 1. Pre-Processing & Normalization
    abg_init = float(np.min(actual))
    y_centered = actual - abg_init
    scale_factor = float(np.max(y_centered))
    if scale_factor <= 1e-12:
        scale_factor = 1.0

    y_norm_np = y_centered / scale_factor

    # Convert to PyTorch double precision (float64)
    w = torch.tensor(2.0 * np.pi * freqs, dtype=torch.float64)
    y_norm_true = torch.tensor(y_norm_np, dtype=torch.float64)

    # 2. Pick initial resonant frequency guesses (f0) from top N peaks of y_norm
    min_dist = max(1, len(actual) // (num_modes * 4))
    amp_range = float(np.max(y_norm_np) - np.min(y_norm_np))
    prominence = amp_range * 0.02 if amp_range > 0 else None

    peaks, _ = find_peaks(y_norm_np, distance=min_dist, prominence=prominence)

    if len(peaks) >= num_modes:
        top_indices = peaks[np.argsort(y_norm_np[peaks])[-num_modes:]]
        top_indices = np.sort(top_indices)
    else:
        top_indices = []
        chunk_sz = len(y_norm_np) // num_modes
        for idx in range(num_modes):
            st = idx * chunk_sz
            en = (idx + 1) * chunk_sz if idx < num_modes - 1 else len(y_norm_np)
            top_indices.append(st + int(np.argmax(y_norm_np[st:en])))
        top_indices = np.array(top_indices)

    f0_inits = freqs[top_indices]

    # 3. Parameter initializations
    q_init = 25.0
    alpha_init = 0.1

    raw_abg = nn.Parameter(torch.tensor(abg_init, dtype=torch.float64))

    raw_f0_list = [
        nn.Parameter(torch.tensor(_inv_softplus(float(f0_inits[i])), dtype=torch.float64))
        for i in range(num_modes)
    ]
    raw_q_list = [
        nn.Parameter(torch.tensor(_inv_softplus(q_init), dtype=torch.float64))
        for _ in range(num_modes)
    ]
    raw_alpha_list = [
        nn.Parameter(torch.tensor(_inv_softplus(alpha_init), dtype=torch.float64))
        for _ in range(num_modes)
    ]

    optimizer = optim.Adam([
        {"params": [raw_abg], "lr": lr * 0.1},
        {"params": raw_f0_list, "lr": lr},
        {"params": raw_q_list + raw_alpha_list, "lr": lr},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    softplus = nn.Softplus()

    # 4. Optimization loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        h_total = torch.complex(torch.zeros_like(w), torch.zeros_like(w))
        q_penalty = torch.tensor(0.0, dtype=torch.float64)

        for i in range(num_modes):
            p_alpha = softplus(raw_alpha_list[i]) + 1e-12
            p_f0 = softplus(raw_f0_list[i]) + 1e-6
            p_q = softplus(raw_q_list[i]) + 0.1

            w0 = 2.0 * np.pi * p_f0
            m_i = m_tensors[i]

            k_i = m_i * (w0 ** 2)
            c_i = (m_i * w0) / (p_q + 1e-12)

            # Modal transfer function: H_i(w) = (alpha_i * V) / ((k_i - m_i * w^2) + j * (c_i * w))
            re_denom = k_i - m_i * (w ** 2)
            im_denom = c_i * w

            denom = torch.complex(re_denom, im_denom)
            h_i = (p_alpha * V) / denom
            h_total = h_total + h_i

            # Light penalty for non-physical Q values (Q < 0.1 or Q > 500)
            q_penalty = q_penalty + torch.relu(0.1 - p_q) ** 2 + torch.relu(p_q - 500.0) ** 2

        y_pred_norm = torch.abs(h_total)

        # MSE loss on normalized target
        mse_loss = nn.functional.mse_loss(y_pred_norm, y_norm_true)
        total_loss = mse_loss + 1e-3 * q_penalty

        total_loss.backward()
        optimizer.step()
        scheduler.step()

    # 5. Extract final physical parameters
    with torch.no_grad():
        abg_final = float(raw_abg.item())

        f0_final = []
        q_final = []
        k_final = []
        c_final = []
        alpha_final = []

        h_total_final = torch.complex(torch.zeros_like(w), torch.zeros_like(w))

        for i in range(num_modes):
            p_alpha = float((softplus(raw_alpha_list[i]) + 1e-12).item())
            p_f0 = float((softplus(raw_f0_list[i]) + 1e-6).item())
            p_q = float((softplus(raw_q_list[i]) + 0.1).item())

            w0 = 2.0 * np.pi * p_f0
            m_i = float(m_vals[i])
            k_i = m_i * (w0 ** 2)
            c_i = (m_i * w0) / (p_q + 1e-12)

            f0_final.append(p_f0)
            q_final.append(p_q)
            k_final.append(k_i)
            c_final.append(c_i)
            alpha_final.append(p_alpha)

            re_denom_t = k_i - m_i * (w ** 2)
            im_denom_t = c_i * w
            denom_t = torch.complex(re_denom_t, im_denom_t)
            h_total_final = h_total_final + ((p_alpha * V) / denom_t)

        y_pred_norm_final = torch.abs(h_total_final).numpy().tolist()
        y_pred_actual_final = ((torch.abs(h_total_final).numpy() * scale_factor) + abg_final).tolist()

        final_loss = float(nn.functional.mse_loss(
            torch.tensor(y_pred_norm_final, dtype=torch.float64),
            y_norm_true
        ).item())

    return {
        "f0": f0_final,
        "Q": q_final,
        "k": k_final,
        "c": c_final,
        "alpha": alpha_final,
        "A_bg": abg_final,
        "S": scale_factor,
        "y_pred_actual": y_pred_actual_final,
        "y_pred_norm": y_pred_norm_final,
        "loss": final_loss,
    }



def generate_mdof_comparison(
    df_all: pd.DataFrame,
    num_modes: int = 3,
    save_dirs: list = None,
    figure_dir: str = "figures/model_parameters",
):
    """
    Fit MDOF model to each PMUT curve, save extracted parameters to CSV, HTML, and plot trajectories.
    """
    if save_dirs is None:
        save_dirs = ["results/direct_fits"]

    pmut_range = sorted(df_all["N_PMUT"].unique())
    records = []

    print("\n── Computing MDOF Mechanical Parameters for all PMUTs ─────────────────")
    for N in pmut_range:
        pdf = df_all[df_all["N_PMUT"] == N].sort_values("Frequency_MHz")
        freqs = pdf["Frequency_MHz"].values
        actual = pdf["Amplitude_R_mean"].values

        mdof_res = fit_mdof_pmut(freqs, actual, num_modes=num_modes, epochs=3000, lr=0.01)

        rec = {
            "N_PMUT": N,
            "A_bg": mdof_res["A_bg"],
            "ScaleFactor_S": mdof_res["S"],
            "Loss_Norm": mdof_res["loss"],
        }
        for m_idx in range(num_modes):
            rec[f"f0_m{m_idx+1}_MHz"] = mdof_res["f0"][m_idx] if m_idx < len(mdof_res["f0"]) else np.nan
            rec[f"Q_m{m_idx+1}"] = mdof_res["Q"][m_idx] if m_idx < len(mdof_res["Q"]) else np.nan
            rec[f"k_m{m_idx+1}"] = mdof_res["k"][m_idx] if m_idx < len(mdof_res["k"]) else np.nan
            rec[f"c_m{m_idx+1}"] = mdof_res["c"][m_idx] if m_idx < len(mdof_res["c"]) else np.nan
            rec[f"alpha_m{m_idx+1}"] = mdof_res["alpha"][m_idx] if m_idx < len(mdof_res["alpha"]) else np.nan
        records.append(rec)
        print(f"  PMUT {N:>2} MDOF fitted (Loss={mdof_res['loss']:.5e}) ✓")

    df_comp = pd.DataFrame(records)

    # Save CSVs
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        csv_path = os.path.join(sdir, "mdof_parameters_comparison.csv")
        df_comp.to_csv(csv_path, index=False)

    # HTML table generation
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PMUT MDOF Mechanical Parameters</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1500px; margin: auto; padding: 20px; background: #F8F9FA; color: #37474F; }
        h1 { color: #2E7D32; margin-bottom: 5px; }
        p.subtitle { color: #666; margin-bottom: 25px; font-size: 0.95em; }
        table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
        th, td { padding: 7px 10px; text-align: center; border: 1px solid #E0E0E0; }
        th { background: #1B5E20; color: white; font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.3px; }
        tr:nth-child(even) { background: #F9F9F9; }
        .hdr-group { background: #C8E6C9; font-weight: bold; color: #1B5E20; }
    </style>
</head>
<body>
    <h1>PMUT Multi-Mode Mechanical Superposition (MDOF) Parameters</h1>
    <p class="subtitle">
        Extracted mechanical modal parameters ($A_{\text{bg}}$, Scale Factor $S$, $f_0, Q, k, c, \alpha$)
        fitted using PyTorch double precision (float64) auto-normalized optimizer.
    </p>
    <table>
        <thead>
            <tr class="hdr-group">
                <th>PMUT</th>
                <th colspan="3">Baseline & Scale</th>
                <th colspan="5">Mode 1</th>
                <th colspan="5">Mode 2</th>
                <th colspan="5">Mode 3</th>
            </tr>
            <tr>
                <th>N</th>
                <th>A<sub>bg</sub></th><th>Scale (S)</th><th>Loss</th>
                <th>f<sub>0</sub> (MHz)</th><th>Q</th><th>k</th><th>c</th><th>&alpha;</th>
                <th>f<sub>0</sub> (MHz)</th><th>Q</th><th>k</th><th>c</th><th>&alpha;</th>
                <th>f<sub>0</sub> (MHz)</th><th>Q</th><th>k</th><th>c</th><th>&alpha;</th>
            </tr>
        </thead>
        <tbody>
"""
    for _, r in df_comp.iterrows():
        n = int(r["N_PMUT"])
        html_content += f"<tr><td>P{n}</td>"
        html_content += f"<td>{r['A_bg']:.4f}</td><td>{r['ScaleFactor_S']:.4f}</td><td>{r['Loss_Norm']:.3e}</td>"
        for m_idx in range(1, 4):
            f0 = r.get(f"f0_m{m_idx}_MHz", 0.0)
            q = r.get(f"Q_m{m_idx}", 0.0)
            k_val = r.get(f"k_m{m_idx}", 0.0)
            c_val = r.get(f"c_m{m_idx}", 0.0)
            alpha_val = r.get(f"alpha_m{m_idx}", 0.0)
            html_content += f"<td>{f0:.4f}</td><td>{q:.1f}</td><td>{k_val:.3e}</td><td>{c_val:.3e}</td><td>{alpha_val:.3e}</td>"
        html_content += "</tr>\n"

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    for sdir in save_dirs:
        html_path = os.path.join(sdir, "mdof_parameters_comparison.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    print("Saved MDOF parameters HTML table → results/direct_fits/")

    # Plot MDOF trajectories
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    axes[0, 0].plot(df_comp["N_PMUT"], df_comp["A_bg"], "o-", color="#2E7D32", lw=2)
    axes[0, 0].set_title("Background Offset ($A_{bg}$) vs N", fontweight="bold")
    axes[0, 0].set_ylabel("$A_{bg}$")

    axes[0, 1].plot(df_comp["N_PMUT"], df_comp["ScaleFactor_S"], "s-", color="#1565C0", lw=2)
    axes[0, 1].set_title("Scale Factor ($S$) vs N", fontweight="bold")
    axes[0, 1].set_ylabel("$S$")

    axes[0, 2].plot(df_comp["N_PMUT"], df_comp["Loss_Norm"], "^-", color="#E65100", lw=2)
    axes[0, 2].set_title("Normalized MSE Loss vs N", fontweight="bold")
    axes[0, 2].set_ylabel("Loss")

    colors_mode = ["#1565C0", "#2E7D32", "#6A1B9A"]
    for m in range(1, 4):
        axes[1, 0].plot(df_comp["N_PMUT"], df_comp[f"f0_m{m}_MHz"], "o-", color=colors_mode[m-1], label=f"Mode {m}")
        axes[1, 1].plot(df_comp["N_PMUT"], df_comp[f"Q_m{m}"], "s-", color=colors_mode[m-1], label=f"Mode {m}")
        axes[1, 2].plot(df_comp["N_PMUT"], df_comp[f"alpha_m{m}"], "^-", color=colors_mode[m-1], label=f"Mode {m}")

    axes[1, 0].set_title("Resonant Frequency ($f_0$) vs N", fontweight="bold")
    axes[1, 0].set_ylabel("f0 (MHz)")
    axes[1, 0].legend()

    axes[1, 1].set_title("Quality Factor ($Q$) vs N", fontweight="bold")
    axes[1, 1].set_ylabel("Q")
    axes[1, 1].legend()

    axes[1, 2].set_title("Force Coefficient ($\\alpha$) vs N", fontweight="bold")
    axes[1, 2].set_ylabel("$\\alpha$")
    axes[1, 2].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PMUT Array Size (N)")

    plt.suptitle("MDOF Mechanical Superposition Parameters vs PMUT Array Size (N)", fontweight="bold", fontsize=12)
    plt.tight_layout()

    filename = "mdof_parameter_trajectories.png"
    os.makedirs(figure_dir, exist_ok=True)
    fig.savefig(os.path.join(figure_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved MDOF parameter trajectory plot → figures/model_parameters/")
