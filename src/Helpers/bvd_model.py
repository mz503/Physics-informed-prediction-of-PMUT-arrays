"""
Helpers/bvd_model.py
══════════════════════════════════════════════════════════════════════════════
Multi-branch Butterworth-Van Dyke (BVD) equivalent circuit model fitting for
PMUT electrical admittance |Y(w)| data using PyTorch double precision (float64).

Mathematical Model:
  Y_static(w) = j * w * C0
  Z_m,i(w)    = R_m,i + j * (w * L_m,i - 1 / (w * C_m,i))
  Y_m,i(w)    = 1 / Z_m,i(w)
  |Y_pred(w)| = |Y_static(w) + sum_{i=1}^N Y_m,i(w)| + A_bg
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


def fit_bvd_pmut(
    freqs: np.ndarray,
    actual: np.ndarray,
    num_modes: int = 3,
    epochs: int = 2500,
    lr: float = 0.01,
) -> dict:
    """
    Fit a multi-branch BVD equivalent circuit model to electrical admittance |Y(w)| data
    using PyTorch double precision (torch.float64).

    Parameters
    ----------
    freqs     : np.ndarray
                Frequency points (e.g., in MHz or Hz).
    actual    : np.ndarray
                Measured electrical admittance magnitude |Y(w)|.
    num_modes : int, optional
                Number of motional mechanical branches / modes N (default 3).
    epochs    : int, optional
                Number of optimization iterations (default 2500).
    lr        : float, optional
                Learning rate for Adam optimizer (default 0.01).

    Returns
    -------
    dict containing:
        - "C0"    : float, static parallel capacitance
        - "A_bg"  : float, background offset
        - "f0"    : list of float, per-mode resonant frequencies
        - "Q"     : list of float, per-mode quality factors
        - "R_m"   : list of float, per-mode motional resistances
        - "L_m"   : list of float, per-mode motional inductances
        - "C_m"   : list of float, per-mode motional capacitances
        - "y_pred": np.ndarray, predicted admittance magnitude |Y_pred(w)|
        - "loss"  : float, final variance-normalized MSE loss
    """
    # 1. Enforce double precision (float64) for all PyTorch tensors
    w = torch.tensor(2.0 * np.pi * freqs, dtype=torch.float64)
    y_true = torch.tensor(actual, dtype=torch.float64)
    var_y_true = torch.var(y_true) + 1e-12

    # 2. Pick initial resonant frequency guesses (f0) from top N magnitude peaks
    min_dist = max(1, len(actual) // (num_modes * 4))
    amp_range = np.max(actual) - np.min(actual)
    prominence = amp_range * 0.02 if amp_range > 0 else None
    
    peaks, _ = find_peaks(actual, distance=min_dist, prominence=prominence)

    if len(peaks) >= num_modes:
        # Select top num_modes peaks by amplitude
        top_indices = peaks[np.argsort(actual[peaks])[-num_modes:]]
        top_indices = np.sort(top_indices)
    else:
        # Fallback: divide spectrum into num_modes equal chunks and pick max per chunk
        top_indices = []
        chunk_sz = len(actual) // num_modes
        for idx in range(num_modes):
            st = idx * chunk_sz
            en = (idx + 1) * chunk_sz if idx < num_modes - 1 else len(actual)
            top_indices.append(st + int(np.argmax(actual[st:en])))
        top_indices = np.array(top_indices)

    f0_inits = freqs[top_indices]

    # 3. Initial parameter estimates
    w_max = float(torch.max(w).item())
    w_min = float(torch.min(w).item())
    c0_init = float(np.min(actual)) / max(w_max, 1.0)
    abg_init = float(np.min(actual)) * 0.1
    q_init = 20.0
    lm_init = 1e-3

    # 4. Parameter definitions (raw parameters transformed via Softplus)
    raw_c0 = nn.Parameter(torch.tensor(_inv_softplus(c0_init), dtype=torch.float64))
    raw_abg = nn.Parameter(torch.tensor(_inv_softplus(abg_init), dtype=torch.float64))

    raw_f0_list = [
        nn.Parameter(torch.tensor(_inv_softplus(float(f0_inits[i])), dtype=torch.float64))
        for i in range(num_modes)
    ]
    raw_q_list = [
        nn.Parameter(torch.tensor(_inv_softplus(q_init), dtype=torch.float64))
        for _ in range(num_modes)
    ]
    raw_lm_list = [
        nn.Parameter(torch.tensor(_inv_softplus(lm_init), dtype=torch.float64))
        for _ in range(num_modes)
    ]

    optimizer = optim.Adam([
        {"params": [raw_c0, raw_abg], "lr": lr * 0.1},
        {"params": raw_f0_list, "lr": lr},
        {"params": raw_q_list + raw_lm_list, "lr": lr * 0.5},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    softplus = nn.Softplus()

    # 5. Optimization loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Enforce positivity on C0, A_bg via Softplus
        p_c0 = softplus(raw_c0) + 1e-18
        p_abg = softplus(raw_abg)

        # Static branch admittance: Y_static(w) = j * w * C0
        y_static = torch.complex(torch.zeros_like(w), w * p_c0)

        # Motional branches sum
        y_total = y_static
        for i in range(num_modes):
            p_f0 = softplus(raw_f0_list[i]) + 1e-6
            p_q = softplus(raw_q_list[i]) + 0.1
            p_lm = softplus(raw_lm_list[i]) + 1e-18

            w0 = 2.0 * np.pi * p_f0
            p_cm = 1.0 / ((w0 ** 2) * p_lm + 1e-18)
            p_rm = (w0 * p_lm) / (p_q + 1e-18)

            # Motional impedance Z_m,i(w) = R_m,i + j * (w * L_m,i - 1 / (w * C_m,i))
            reactance = w * p_lm - 1.0 / (w * p_cm + 1e-18)
            z_m = torch.complex(p_rm.expand_as(w), reactance)
            y_m = 1.0 / z_m
            y_total = y_total + y_m

        # Total predicted admittance magnitude
        y_pred = torch.abs(y_total) + p_abg

        # Variance-normalized MSE loss
        loss = torch.mean((y_pred - y_true) ** 2) / var_y_true
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 6. Extract final physical parameters
    with torch.no_grad():
        c0_final = float((softplus(raw_c0) + 1e-18).item())
        abg_final = float(softplus(raw_abg).item())

        f0_final = []
        q_final = []
        rm_final = []
        lm_final = []
        cm_final = []

        y_static_final = torch.complex(torch.zeros_like(w), w * c0_final)
        y_total_final = y_static_final

        for i in range(num_modes):
            pf0 = float((softplus(raw_f0_list[i]) + 1e-6).item())
            pq = float((softplus(raw_q_list[i]) + 0.1).item())
            plm = float((softplus(raw_lm_list[i]) + 1e-18).item())

            w0_val = 2.0 * np.pi * pf0
            pcm_val = 1.0 / ((w0_val ** 2) * plm + 1e-18)
            prm_val = (w0_val * plm) / (pq + 1e-18)

            f0_final.append(pf0)
            q_final.append(pq)
            lm_final.append(plm)
            cm_final.append(pcm_val)
            rm_final.append(prm_val)

            w0_t = 2.0 * np.pi * pf0
            pcm_t = 1.0 / ((w0_t ** 2) * plm + 1e-18)
            prm_t = (w0_t * plm) / (pq + 1e-18)
            reactance_t = w * plm - 1.0 / (w * pcm_t + 1e-18)
            z_m_t = torch.complex(torch.full_like(w, prm_t), reactance_t)
            y_total_final = y_total_final + (1.0 / z_m_t)

        y_pred_final = (torch.abs(y_total_final) + abg_final).numpy()
        final_loss = float((torch.mean((torch.tensor(y_pred_final, dtype=torch.float64) - y_true) ** 2) / var_y_true).item())

    return {
        "C0": c0_final,
        "A_bg": abg_final,
        "f0": f0_final,
        "Q": q_final,
        "R_m": rm_final,
        "L_m": lm_final,
        "C_m": cm_final,
        "y_pred": y_pred_final.tolist(),
        "loss": final_loss,
    }



def generate_bvd_comparison(
    df_all: pd.DataFrame,
    num_modes: int = 3,
    save_dirs: list = None,
    figure_dir: str = "figures/model_parameters",
):
    """
    Fit BVD model to each PMUT curve, save extracted parameters to CSV, HTML, and plot trajectories.
    """
    if save_dirs is None:
        save_dirs = ["results/direct_fits"]

    pmut_range = sorted(df_all["N_PMUT"].unique())
    records = []

    print("\n── Computing BVD Circuit Parameters for all PMUTs ─────────────────")
    for N in pmut_range:
        pdf = df_all[df_all["N_PMUT"] == N].sort_values("Frequency_MHz")
        freqs = pdf["Frequency_MHz"].values
        actual = pdf["Amplitude_R_mean"].values

        bvd_res = fit_bvd_pmut(freqs, actual, num_modes=num_modes, epochs=2500, lr=0.01)

        rec = {
            "N_PMUT": N,
            "C0": bvd_res["C0"],
            "A_bg": bvd_res["A_bg"],
            "Loss": bvd_res["loss"],
        }
        for m_idx in range(num_modes):
            rec[f"f0_m{m_idx+1}_MHz"] = bvd_res["f0"][m_idx] if m_idx < len(bvd_res["f0"]) else np.nan
            rec[f"Q_m{m_idx+1}"] = bvd_res["Q"][m_idx] if m_idx < len(bvd_res["Q"]) else np.nan
            rec[f"R_m{m_idx+1}"] = bvd_res["R_m"][m_idx] if m_idx < len(bvd_res["R_m"]) else np.nan
            rec[f"L_m{m_idx+1}"] = bvd_res["L_m"][m_idx] if m_idx < len(bvd_res["L_m"]) else np.nan
            rec[f"C_m{m_idx+1}"] = bvd_res["C_m"][m_idx] if m_idx < len(bvd_res["C_m"]) else np.nan
        records.append(rec)
        print(f"  PMUT {N:>2} BVD fitted (Loss={bvd_res['loss']:.5e}) ✓")

    df_comp = pd.DataFrame(records)

    # Save CSVs
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        csv_path = os.path.join(sdir, "bvd_parameters_comparison.csv")
        df_comp.to_csv(csv_path, index=False)

    # HTML table generation
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PMUT BVD Circuit Parameters</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1500px; margin: auto; padding: 20px; background: #F8F9FA; color: #37474F; }
        h1 { color: #00838F; margin-bottom: 5px; }
        p.subtitle { color: #666; margin-bottom: 25px; font-size: 0.95em; }
        table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
        th, td { padding: 7px 10px; text-align: center; border: 1px solid #E0E0E0; }
        th { background: #00695C; color: white; font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.3px; }
        tr:nth-child(even) { background: #F9F9F9; }
        .hdr-group { background: #B2DFDB; font-weight: bold; color: #004D40; }
    </style>
</head>
<body>
    <h1>PMUT Butterworth-Van Dyke (BVD) Circuit Parameters</h1>
    <p class="subtitle">
        Extracted electrical BVD circuit parameters ($C_0$, $A_{\\text{bg}}$, and motional branch parameters $f_0, Q, R_m, L_m, C_m$)
        fitted using PyTorch double precision (float64) multi-branch optimizer.
    </p>
    <table>
        <thead>
            <tr class="hdr-group">
                <th>PMUT</th>
                <th colspan="3">Global Circuit Baseline</th>
                <th colspan="5">Motional Mode 1</th>
                <th colspan="5">Motional Mode 2</th>
                <th colspan="5">Motional Mode 3</th>
            </tr>
            <tr>
                <th>N</th>
                <th>C<sub>0</sub></th><th>A<sub>bg</sub></th><th>Fit Loss</th>
                <th>f<sub>0</sub> (MHz)</th><th>Q</th><th>R<sub>m</sub></th><th>L<sub>m</sub></th><th>C<sub>m</sub></th>
                <th>f<sub>0</sub> (MHz)</th><th>Q</th><th>R<sub>m</sub></th><th>L<sub>m</sub></th><th>C<sub>m</sub></th>
                <th>f<sub>0</sub> (MHz)</th><th>Q</th><th>R<sub>m</sub></th><th>L<sub>m</sub></th><th>C<sub>m</sub></th>
            </tr>
        </thead>
        <tbody>
"""
    for _, r in df_comp.iterrows():
        n = int(r["N_PMUT"])
        html_content += f"<tr><td>P{n}</td>"
        html_content += f"<td>{r['C0']:.4e}</td><td>{r['A_bg']:.4f}</td><td>{r['Loss']:.3e}</td>"
        for m_idx in range(1, 4):
            f0 = r.get(f"f0_m{m_idx}_MHz", 0.0)
            q = r.get(f"Q_m{m_idx}", 0.0)
            rm = r.get(f"R_m{m_idx}", 0.0)
            lm = r.get(f"L_m{m_idx}", 0.0)
            cm = r.get(f"C_m{m_idx}", 0.0)
            html_content += f"<td>{f0:.4f}</td><td>{q:.1f}</td><td>{rm:.3e}</td><td>{lm:.3e}</td><td>{cm:.3e}</td>"
        html_content += "</tr>\n"

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    for sdir in save_dirs:
        html_path = os.path.join(sdir, "bvd_parameters_comparison.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    print("Saved BVD parameters HTML table → results/direct_fits/")

    # Plot BVD trajectories
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    axes[0, 0].plot(df_comp["N_PMUT"], df_comp["C0"], "o-", color="#00838F", lw=2)
    axes[0, 0].set_title("Static Capacitance ($C_0$) vs N", fontweight="bold")
    axes[0, 0].set_ylabel("$C_0$")

    axes[0, 1].plot(df_comp["N_PMUT"], df_comp["A_bg"], "s-", color="#D81B60", lw=2)
    axes[0, 1].set_title("Background Offset ($A_{bg}$) vs N", fontweight="bold")
    axes[0, 1].set_ylabel("$A_{bg}$")

    axes[0, 2].plot(df_comp["N_PMUT"], df_comp["Loss"], "^-", color="#E65100", lw=2)
    axes[0, 2].set_title("Fit Loss vs N", fontweight="bold")
    axes[0, 2].set_ylabel("Loss")

    colors_mode = ["#1565C0", "#2E7D32", "#6A1B9A"]
    for m in range(1, 4):
        axes[1, 0].plot(df_comp["N_PMUT"], df_comp[f"f0_m{m}_MHz"], "o-", color=colors_mode[m-1], label=f"Mode {m}")
        axes[1, 1].plot(df_comp["N_PMUT"], df_comp[f"Q_m{m}"], "s-", color=colors_mode[m-1], label=f"Mode {m}")
        axes[1, 2].plot(df_comp["N_PMUT"], df_comp[f"R_m{m}"], "^-", color=colors_mode[m-1], label=f"Mode {m}")

    axes[1, 0].set_title("Resonant Frequency ($f_0$) vs N", fontweight="bold")
    axes[1, 0].set_ylabel("f0 (MHz)")
    axes[1, 0].legend()

    axes[1, 1].set_title("Quality Factor ($Q$) vs N", fontweight="bold")
    axes[1, 1].set_ylabel("Q")
    axes[1, 1].legend()

    axes[1, 2].set_title("Motional Resistance ($R_m$) vs N", fontweight="bold")
    axes[1, 2].set_ylabel("$R_m$")
    axes[1, 2].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PMUT Array Size (N)")

    plt.suptitle("BVD Equivalent Circuit Extracted Parameters vs PMUT Array Size (N)", fontweight="bold", fontsize=12)
    plt.tight_layout()

    filename = "bvd_parameter_trajectories.png"
    os.makedirs(figure_dir, exist_ok=True)
    fig.savefig(os.path.join(figure_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved BVD parameter trajectory plot → figures/model_parameters/")
