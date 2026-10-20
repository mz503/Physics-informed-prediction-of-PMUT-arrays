"""
07_regime_aware_training.py
══════════════════════════════════════════════════════════════════════════════
Regime-aware progressive training with three experimental regimes:

  Regime 1  PMUTs  1– 2  within: [1]→2
  Regime 2  PMUTs  3–11  within: [3]→4, [3,4]→5 … [3..10]→11
  Regime 3  PMUTs 12–16  within: [12]→13 … [12..15]→16

  ★ Cross-regime predictions (fresh model → first PMUT of next regime):
      End of Regime 1 → [1,2]   → Predict PMUT  3
      End of Regime 2 → [3..11] → Predict PMUT 12

Reads predictions from the pre-computed cache (results/predictions/regime_cache.csv)
written by 01_train_cache.py — no model training happens here.

Two input/output approaches compared side-by-side:
  Vector    — Input: N_PMUT (scalar); Output: 1000-pt amplitude vector.
  Pointwise — Input: [N_PMUT, Freq_MHz]; Output: single amplitude value.

Both evaluated on Full Spectrum and FWHM ROI.

Files written
─────────────
  supplementary/explorer/regime/all_combined.html   — single interactive page
  supplementary/explorer/regime/regime_metrics.csv
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

from Helpers.cache_utils import load_regime
from Helpers.html_utils  import build_interactive_html

# ── Paths ─────────────────────────────────────────────────────────────────────
REGI_CACHE = os.path.join("results", "predictions", "regime_cache.csv")
OUT_DIR    = os.path.join("supplementary", "explorer", "regime")
OUT_CSV    = os.path.join(OUT_DIR, "regime_metrics.csv")
OUT_COMB   = os.path.join(OUT_DIR, "all_combined.html")

# Regime metadata for the HTML controls
REGIME_LABELS = [
    ("1", "Regime 1 (1–2)"),
    ("2", "Regime 2 (3–11)"),
    ("3", "Regime 3 (12–16)"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading regime cache → {REGI_CACHE}")
    all_data, regime_map, cross_pmuts, df_vec_metrics = load_regime(REGI_CACHE)

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
                    "Target_PMUT":  target,
                    "N_train":      len(entry["train"]),
                    "Regime":       regime_map.get(target),
                    "Regime_Name":  entry.get("regime_name", ""),
                    "Step_Type":    entry.get("step_type", "within"),
                    "Approach":     approach,
                    "Model":        model,
                    "Full_MAE":     d["mae_full"],
                    "Full_R2":      d["r2_full"],
                    "ROI_MAE":      d["mae_roi"],
                    "ROI_R2":       d["r2_roi"],
                })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV  → {OUT_CSV}")

    # ── Build interactive HTML ─────────────────────────────────────────────────
    html = build_interactive_html(
        title    = "Regime-Aware Training — PMUTs 2–16",
        subtitle = (
            "Fresh model at each regime boundary. "
            "<b>★ Cross-Regime</b>: all PMUTs of completed regime → predict first of next regime. "
            "Regimes: <b>1–2</b> | <b>3–11</b> | <b>12–16</b>. "
            "Use controls above to filter models, approach, scope, or regime."
        ),
        all_data      = all_data,
        regime_map    = regime_map,
        cross_pmuts   = cross_pmuts,
        regime_labels = REGIME_LABELS,
    )
    with open(OUT_COMB, "w", encoding="utf-8") as f:

        f.write(html)
    print(f"Saved HTML → {OUT_COMB}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── Mean metrics by Regime × Approach × Model ────────────────────")
    print(df_metrics.groupby(["Regime_Name", "Approach", "Model"])[
        ["Full_MAE", "Full_R2", "ROI_MAE", "ROI_R2"]
    ].mean().round(5).to_string())


if __name__ == "__main__":
    main()
