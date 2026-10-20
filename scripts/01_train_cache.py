"""
01_train_cache.py
══════════════════════════════════════════════════════════════════════════════
Run every model-training combination ONCE and write prediction caches to the
results/predictions/ folder. All downstream scripts (04, 06, 07, 08) read from these CSVs
instead of retraining.

Output
──────
  results/predictions/progressive_cache.csv
      Scheme : cumulative [1..N-1] → predict PMUT N,  for N = 2..16
      Rows   : one per (target_pmut × approach × model × frequency_point)

  results/predictions/regime_cache.csv
      Scheme : regime-aware (fresh model per regime boundary)
               Regime 1  PMUTs  1–2   [1]→2
               Regime 2  PMUTs  3–11  [3]→4 … [3..10]→11
               Regime 3  PMUTs 12–16  [12]→13 … [12..15]→16
               + cross-regime [1,2]→3 and [3..11]→12
      Same row schema, plus regime_id / regime_name / step_type columns.
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

import numpy as np
import pandas as pd

from Helpers.data_loader import load_data, scale_target_for_training
from Helpers.fwhm_utils  import compute_fwhm_window
from Helpers.model_utils import train_predict_vector, train_predict_pointwise

import argparse

DATA_DIR = os.path.join("results", "predictions")
PROG_CSV = os.path.join(DATA_DIR, "progressive_cache.csv")
REGI_CSV = os.path.join(DATA_DIR, "regime_cache.csv")
CACHE_TARGET_SCALING = "training_pmuts_only_v1"
TARGET_SCALED_MODELS = {"RF", "GB", "MLP"}

APPROACHES = [("Vector", train_predict_vector), ("Pointwise", train_predict_pointwise)]
ALL_MODELS = ["RF", "GB", "MLP", "PINN", "MDOF-PINN", "PINN-Informed-Loss", "RF-MDOF-PINN"]


def _stale_scaled_models(cache_df: pd.DataFrame) -> set:
    affected = cache_df[cache_df["model"].isin(TARGET_SCALED_MODELS)]
    if affected.empty:
        return set()
    if "target_scaling" not in affected:
        return set(affected["model"].unique())
    stale = affected[
        ~affected["target_scaling"].eq(CACHE_TARGET_SCALING)
    ]
    return set(stale["model"].unique())


REGIMES = [
    {
        "id": 1, "name": "Regime 1 (1–2)",
        "within":        [{"train": [1], "target": 2}],
        "cross_to_next": {"train": [1, 2], "target": 3},
    },
    {
        "id": 2, "name": "Regime 2 (3–11)",
        "within":        [{"train": list(range(3, t)), "target": t} for t in range(4, 12)],
        "cross_to_next": {"train": list(range(3, 12)), "target": 12},
    },
    {
        "id": 3, "name": "Regime 3 (12–16)",
        "within":        [{"train": list(range(12, t)), "target": t} for t in range(13, 17)],
        "cross_to_next": None,
    },
]


def _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all,
                 models_to_run: list, extra: dict = None) -> list:
    """
    Train and predict for specified approach × model combinations.
    Returns a flat list of row dicts — one per frequency point per combination.
    """
    base = {
        "target_pmut": target,
        "train_pmuts":  str(t_pmuts),
        "n_train":      len(t_pmuts),
        "fwhm_lo":      round(float(fwhm_lo), 5),
        "fwhm_hi":      round(float(fwhm_hi), 5),
        "target_scaling": CACHE_TARGET_SCALING,
        **(extra or {}),
    }
    df_scaled, amp_scaler = scale_target_for_training(df_all, t_pmuts)
    rows = []
    for approach_name, train_fn in APPROACHES:
        for model in models_to_run:
            fq, act, pr = train_fn(model, df_scaled, t_pmuts, target, amp_scaler)
            for f, a, p in zip(fq, act, pr):
                rows.append({
                    **base,
                    "approach":      approach_name,
                    "model":         model,
                    "frequency_mhz": round(float(f), 5),
                    "actual":        round(float(a), 6),
                    "predicted":     round(float(p), 6),
                })
            print(f"    PMUT {target:>2}  {model:<3}  {approach_name:<10}  ✓")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train and cache predictions for PMUT models.")
    parser.add_argument("--pinn-only", action="store_true", help="Only rerun PINN models and preserve non-PINN cache entries.")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit list of models to run (e.g. PINN RF).")
    args = parser.parse_args()

    if args.pinn_only:
        models_to_run = ["PINN"]
    elif args.models:
        models_to_run = args.models
    else:
        models_to_run = ALL_MODELS

    os.makedirs(DATA_DIR, exist_ok=True)
    df_all, unique_pmuts = load_data()

    # Load existing cache if running selectively
    existing_prog_df = pd.DataFrame()
    existing_regi_df = pd.DataFrame()
    if models_to_run != ALL_MODELS:
        if os.path.exists(PROG_CSV):
            df_ex = pd.read_csv(PROG_CSV)
            stale_models = _stale_scaled_models(df_ex)
            if not stale_models.issubset(models_to_run):
                raise RuntimeError(
                    f"{PROG_CSV} has stale target-scaled models "
                    f"{sorted(stale_models - set(models_to_run))}; include them in "
                    "--models or rebuild the full cache."
                )
            existing_prog_df = df_ex[~df_ex["model"].isin(models_to_run)]
        if os.path.exists(REGI_CSV):
            df_ex = pd.read_csv(REGI_CSV)
            stale_models = _stale_scaled_models(df_ex)
            if not stale_models.issubset(models_to_run):
                raise RuntimeError(
                    f"{REGI_CSV} has stale target-scaled models "
                    f"{sorted(stale_models - set(models_to_run))}; include them in "
                    "--models or rebuild the full cache."
                )
            existing_regi_df = df_ex[~df_ex["model"].isin(models_to_run)]

    # ── Progressive (targets 2–16) ─────────────────────────────────────────────
    print(f"\n── Progressive Training (Models: {models_to_run}) ────────────────────────────")
    prog_rows = []
    for target in [int(p) for p in unique_pmuts if p >= 2]:
        t_pmuts = [int(p) for p in unique_pmuts if p < target]
        fwhm_lo, fwhm_hi = compute_fwhm_window(target, df_all)
        print(f"  PMUT {target:>2}  train={t_pmuts}")
        prog_rows.extend(
            _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all, models_to_run))

    df_prog_new = pd.DataFrame(prog_rows)
    if not existing_prog_df.empty:
        df_prog = pd.concat([existing_prog_df, df_prog_new], ignore_index=True)
    else:
        df_prog = df_prog_new

    df_prog = df_prog.sort_values(["target_pmut", "approach", "model", "frequency_mhz"])
    df_prog.to_csv(PROG_CSV, index=False)
    print(f"\nSaved {len(df_prog):,} rows → {PROG_CSV}")

    # ── Regime-Aware ──────────────────────────────────────────────────────────
    print(f"\n── Regime-Aware Training (Models: {models_to_run}) ───────────────────────────")
    regi_rows = []
    for regime in REGIMES:
        rid, rname = regime["id"], regime["name"]
        for step in regime["within"]:
            target, t_pmuts = step["target"], step["train"]
            fwhm_lo, fwhm_hi = compute_fwhm_window(target, df_all)
            print(f"  R{rid}  PMUT {target:>2} (within)  train={t_pmuts}")
            regi_rows.extend(
                _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all,
                             models_to_run,
                             extra={"regime_id": rid, "regime_name": rname,
                                    "step_type": "within"}))
        if regime["cross_to_next"]:
            cx = regime["cross_to_next"]
            target, t_pmuts = cx["target"], cx["train"]
            fwhm_lo, fwhm_hi = compute_fwhm_window(target, df_all)
            print(f"  R{rid}  PMUT {target:>2} ★ cross  train={t_pmuts}")
            regi_rows.extend(
                _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all,
                             models_to_run,
                             extra={"regime_id": rid, "regime_name": rname,
                                    "step_type": "cross"}))

    df_regi_new = pd.DataFrame(regi_rows)
    if not existing_regi_df.empty:
        df_regi = pd.concat([existing_regi_df, df_regi_new], ignore_index=True)
    else:
        df_regi = df_regi_new

    df_regi = df_regi.sort_values(["target_pmut", "approach", "model", "frequency_mhz"])
    df_regi.to_csv(REGI_CSV, index=False)
    print(f"\nSaved {len(df_regi):,} rows → {REGI_CSV}")

    # Generate PINN parameter trajectories vs N
    print("\n── Generating PINN Parameter Trajectories ────────────────────────")
    from Helpers.pinn_model import fit_pinn, plot_parameter_trajectories, generate_physics_comparison
    full_pmuts = sorted(df_all["N_PMUT"].unique())
    pinn_model = fit_pinn(df_all, t_pmuts=full_pmuts, use_two_modes=False, epochs=150)
    plot_parameter_trajectories(pinn_model)
    print("Saved parameter trajectory plot → figures/model_parameters/")
    
    # Generate True vs Estimated physics comparison table
    generate_physics_comparison(pinn_model, df_all)

    # Generate BVD equivalent circuit parameter comparison table & trajectories
    from Helpers.bvd_model import generate_bvd_comparison
    generate_bvd_comparison(df_all)

    # Generate MDOF Mechanical Superposition parameter comparison table & trajectories
    from Helpers.mdof_model import generate_mdof_comparison
    generate_mdof_comparison(df_all)

    # Generate MDOF-PINN Parameter Trajectories vs N
    print("\n── Generating MDOF-PINN Parameter Trajectories ────────────────────")
    from Helpers.mdof_pinn_model import fit_mdof_pinn, plot_mdof_pinn_trajectories
    mdof_pinn_global = fit_mdof_pinn(df_all, t_pmuts=full_pmuts, num_modes=3, epochs=350, lr=0.005)
    plot_mdof_pinn_trajectories(mdof_pinn_global)

    # Generate PINN-Informed-Loss Parameter Trajectories vs N
    print("\n── Generating PINN-Informed-Loss Parameter Trajectories ────────────")
    from Helpers.pinn_informed_loss_model import fit_pinn_informed_loss, plot_pinn_informed_loss_trajectories
    pil_global = fit_pinn_informed_loss(df_all, t_pmuts=full_pmuts, num_modes=3, epochs=350, lr=0.005)
    plot_pinn_informed_loss_trajectories(pil_global)

    print("\nCache complete — downstream scripts will read from results/predictions/ instead of retraining.")






if __name__ == "__main__":
    main()
