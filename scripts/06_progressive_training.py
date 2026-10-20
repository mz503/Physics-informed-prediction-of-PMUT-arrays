"""
06_progressive_training.py
══════════════════════════════════════════════════════════════════════════════
Cumulative progressive training: train on [1..N-1] → predict PMUT N, N=3..16.

Reads predictions from the pre-computed cache (results/predictions/progressive_cache.csv)
written by 01_train_cache.py — no model training happens here.

Two input/output approaches compared side-by-side:
  Vector    — Input: N_PMUT (scalar); Output: 1000-pt amplitude vector.
  Pointwise — Input: [N_PMUT, Freq_MHz]; Output: single amplitude value.

Both evaluated on Full Spectrum (1000 pts) and FWHM ROI (in-band pts).

Files written
─────────────
  supplementary/explorer/progressive/all_combined.html  — interactive page
  supplementary/explorer/progressive/progressive_metrics.csv
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


import pandas as pd

from Helpers.cache_utils import load_progressive
from Helpers.html_utils  import build_interactive_html

# ── Paths ─────────────────────────────────────────────────────────────────────
PROG_CACHE = os.path.join("results", "predictions", "progressive_cache.csv")
OUT_DIR    = os.path.join("supplementary", "explorer", "progressive")
OUT_CSV    = os.path.join(OUT_DIR, "progressive_metrics.csv")
OUT_COMB   = os.path.join(OUT_DIR, "all_combined.html")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading progressive cache → {PROG_CACHE}")
    all_data, df_vec_metrics = load_progressive(PROG_CACHE)

    # 06 starts from PMUT 3 (need at least 2 training devices)
    all_data = {k: v for k, v in all_data.items() if int(k) >= 3}

    # Attach SDOF physics parameters & BVD circuit parameters for HTML cards
    from Helpers.data_loader import load_data
    from Helpers.pinn_model import fit_sdof_directly, fit_pinn, compute_sdof_amplitude
    from Helpers.bvd_model import fit_bvd_pmut
    from Helpers.mdof_model import fit_mdof_pmut
    df_all, _ = load_data()
    full_pmuts = sorted(df_all["N_PMUT"].unique())
    pinn_global = fit_pinn(df_all, t_pmuts=full_pmuts, use_two_modes=False, epochs=150)
    
    for k, entry in all_data.items():
        N = int(k)
        pdf = df_all[df_all["N_PMUT"] == N].sort_values("Frequency_MHz")
        freqs = pdf["Frequency_MHz"].values
        actual = pdf["Amplitude_R_mean"].values
        pt = fit_sdof_directly(freqs, actual)
        bvd_res = fit_bvd_pmut(freqs, actual, num_modes=3, epochs=2500, lr=0.01)
        mdof_res = fit_mdof_pmut(freqs, actual, num_modes=3, epochs=3000, lr=0.01)
        entry["phys_true"] = pt
        entry["phys_est"] = pinn_global.get_physics_parameters(N)
        entry["theoretical"] = compute_sdof_amplitude(freqs, pt).tolist()
        entry["bvd"] = list(bvd_res["y_pred"])
        entry["bvd_params"] = {kp: vp for kp, vp in bvd_res.items() if kp != "y_pred"}
        entry["mdof"] = list(mdof_res["y_pred_actual"])
        entry["mdof_params"] = {kp: vp for kp, vp in mdof_res.items() if kp not in ["y_pred_actual", "y_pred_norm"]}




    # ── Build per-PMUT metrics CSV (all approaches + ROI) ─────────────────────
    records = []
    for target_str, entry in all_data.items():
        target = int(target_str)
        for approach in ["Vector", "Pointwise"]:
            for model in ["RF", "GB", "MLP", "PINN", "MDOF-PINN", "PINN-Informed-Loss", "RF-MDOF-PINN"]:
                if model not in entry[approach]:
                    continue
                d = entry[approach][model]

                records.append({
                    "Target_PMUT": target,
                    "N_train":     len(entry["train"]),
                    "Approach":    approach,
                    "Model":       model,
                    "Full_MAE":    d["mae_full"],
                    "Full_R2":     d["r2_full"],
                    "ROI_MAE":     d["mae_roi"],
                    "ROI_R2":      d["r2_roi"],
                })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV  → {OUT_CSV}")

    # ── Build interactive HTML ─────────────────────────────────────────────────
    html = build_interactive_html(
        title    = "Progressive Training — PMUTs 3–16",
        subtitle = (
            "Cumulative training: train on all preceding PMUTs [1..N-1], predict PMUT N. "
            "<b>Left:</b> Vector (multi-output). <b>Right:</b> Pointwise. "
            "Use controls above to filter models, approach, scope, or individual PMUT."
        ),
        all_data = all_data,
    )
    with open(OUT_COMB, "w", encoding="utf-8") as f:

        f.write(html)
    print(f"Saved HTML → {OUT_COMB}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── Mean metrics by Approach × Model ─────────────────────────────")
    print(df_metrics.groupby(["Approach", "Model"])[
        ["Full_MAE", "Full_R2", "ROI_MAE", "ROI_R2"]
    ].mean().round(5).to_string())


if __name__ == "__main__":
    main()
