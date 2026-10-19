"""
08_progressive_images.py
══════════════════════════════════════════════════════════════════════════════
Generates static matplotlib PNG images for progressive training results.

Training scheme: cumulative [1..N-1] → predict PMUT N.
Approach:        Vector / Multi-Output (one row per device, 1000-pt prediction).
Scope:           min(data frequency) – 1 MHz  (clipped for concise presentation).

Output
──────
  figures/main/
    pmut_02_vector_full.png   ← PMUT 2  (train: [1])
    pmut_03_vector_full.png   ← PMUT 3  (train: [1,2])
    …
    pmut_16_vector_full.png   ← PMUT 16 (train: [1..15])
                                15 images total — RF + GB + Actual only

    pmut_NN_vector_full_mlp.png   ← randomly selected PMUT, adds MLP trace
                                    (1 extra image)
"""

import sys, os, ast, random
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
import matplotlib
matplotlib.use("Agg")                       # non-interactive / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, r2_score

PROG_CACHE = os.path.join("results", "predictions", "progressive_cache.csv")
OUT_DIR    = os.path.join("figures", "main")
FREQ_MAX   = 1.0    # MHz — upper bound of display window
# FREQ_MIN is derived from the data minimum at runtime

COLORS  = {"Actual": "#2E7D32", "Theoretical": "#00897B", "RF": "#1565C0", "GB": "#E65100", "MLP": "#6A1B9A", "PINN": "#E91E63", "PINN-Informed-Loss": "#D81B60", "RF-MDOF-PINN": "#00796B"}
LWIDTHS = {"Actual": 2.4,       "Theoretical": 2.0,       "RF": 1.8,       "GB": 1.8,       "MLP": 1.6,       "PINN": 1.8,       "PINN-Informed-Loss": 1.8,       "RF-MDOF-PINN": 2.0}
LSTYLES = {"Actual": "-",       "Theoretical": "--",      "RF": "-",        "GB": "--",      "MLP": ":",       "PINN": "-.",      "PINN-Informed-Loss": "--",      "RF-MDOF-PINN": "-"}


random.seed(42)


def _clip(freqs, *arrays, freq_min=0.0):
    """Return freq and data arrays clipped to freq_min–FREQ_MAX."""
    mask = (freqs >= freq_min) & (freqs <= FREQ_MAX)
    return (freqs[mask],) + tuple(a[mask] for a in arrays)


def _make_axes():
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFEFF")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#CCCCCC")
    ax.grid(True, alpha=0.25, linewidth=0.6, color="#BBBBBB")
    ax.tick_params(labelsize=8.5, colors="#444")
    return fig, ax


def _draw(ax, freqs, actual, preds_dict, metrics_dict, title, models, freq_min=0.0):
    # Clip all series to the display window
    fq_c, act_c = _clip(freqs, actual, freq_min=freq_min)[0:2]

    ax.plot(fq_c, act_c,
            color=COLORS["Actual"], lw=LWIDTHS["Actual"], ls=LSTYLES["Actual"],
            label="Actual / Experimental", zorder=10)
    for m in models:
        if m not in preds_dict:
            continue
        fq_m, pred_c = _clip(freqs, preds_dict[m], freq_min=freq_min)
        mae  = metrics_dict[m]["mae"]
        r2   = metrics_dict[m]["r2"]
        sign = "+" if r2 >= 0 else ""
        name_lbl = "Theoretical (True SDOF)" if m == "Theoretical" else m
        lbl  = f"{name_lbl}   MAE = {mae:.4f}   R² = {sign}{r2:.3f}"
        ax.plot(fq_m, pred_c,
                color=COLORS[m], lw=LWIDTHS[m], ls=LSTYLES[m], label=lbl)

    ax.set_xlim(freq_min, FREQ_MAX)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=9, color="#212121")
    ax.set_xlabel("Frequency (MHz)", fontsize=9.5, labelpad=6)
    ax.set_ylabel("Amplitude (R)",   fontsize=9.5, labelpad=6)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92,
              edgecolor="#CCCCCC", fancybox=False)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from Helpers.pinn_model import fit_sdof_directly, compute_sdof_amplitude

    print(f"Loading progressive cache → {PROG_CACHE}")
    df_cache = pd.read_csv(PROG_CACHE)
    df_vec   = df_cache[df_cache["approach"] == "Vector"].copy()

    # Derive display window from data
    freq_min   = float(df_vec["frequency_mhz"].min())
    targets    = sorted(df_vec["target_pmut"].unique())
    targets    = [int(t) for t in targets if int(t) >= 2]
    mlp_target = 8  # PMUT chosen for the extra MLP showcase image

    print(f"Frequency window: {freq_min:.4f} – {FREQ_MAX:.1f} MHz")
    print(f"Generating {len(targets)} images  (Theoretical + RF + GB + PINN)")
    print(f"MLP showcase → PMUT {mlp_target}")
    print(f"Output dir   → {OUT_DIR}/\n")

    for target in targets:
        tdf       = df_vec[df_vec["target_pmut"] == target]
        train_str = ", ".join(map(str, ast.literal_eval(tdf["train_pmuts"].iloc[0])))

        models_needed = ["RF", "GB", "PINN"] + (["MLP"] if target == mlp_target else [])
        preds, metrics = {}, {}
        freqs = actual = None

        for m in models_needed:
            mdf = tdf[tdf["model"] == m].sort_values("frequency_mhz")
            if mdf.empty:
                continue
            fq  = mdf["frequency_mhz"].values
            act = mdf["actual"].values
            pr  = mdf["predicted"].values

            preds[m] = pr
            if freqs is None:
                freqs, actual = fq, act
            # Compute metrics only within the display window
            fq_c, act_c, pr_c = _clip(fq, act, pr, freq_min=freq_min)
            if len(fq_c) >= 2:
                metrics[m] = {"mae": mean_absolute_error(act_c, pr_c),
                              "r2":  r2_score(act_c, pr_c)}
            else:
                metrics[m] = {"mae": float("nan"), "r2": float("nan")}

        # Compute theoretical amplitude curve from true SDOF params
        if freqs is not None:
            pt = fit_sdof_directly(freqs, actual)
            preds["Theoretical"] = compute_sdof_amplitude(freqs, pt)
            fq_c, act_c, th_c = _clip(freqs, actual, preds["Theoretical"], freq_min=freq_min)
            metrics["Theoretical"] = {"mae": mean_absolute_error(act_c, th_c),
                                      "r2":  r2_score(act_c, th_c)}

        # ── Main image: Theoretical + RF + GB + PINN ─────────────────────────
        fig, ax = _make_axes()
        title = (f"PMUT {target}  —  Vector  |  Full Spectrum\n"
                 f"Progressive Training: [{train_str}]  →  Predict PMUT {target}")
        _draw(ax, freqs, actual, preds, metrics, title,
              models=["Theoretical", "RF", "GB", "PINN"], freq_min=freq_min)
        plt.tight_layout(pad=1.4)
        out_path = os.path.join(OUT_DIR, f"pmut_{target:02d}_vector_full.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [{'MLP+' if target==mlp_target else '     '}] → {out_path}")

        # ── Extra image: Theoretical + RF + GB + MLP + PINN ──────────────────
        if target == mlp_target:
            fig, ax = _make_axes()
            title_mlp = (f"PMUT {target}  —  Vector  |  Full Spectrum  (Theoretical · RF · GB · MLP · PINN)\n"
                         f"Progressive Training: [{train_str}]  →  Predict PMUT {target}")
            _draw(ax, freqs, actual, preds, metrics, title_mlp,
                  models=["Theoretical", "RF", "GB", "MLP", "PINN"], freq_min=freq_min)
            plt.tight_layout(pad=1.4)
            out_path_mlp = os.path.join(OUT_DIR, f"pmut_{target:02d}_vector_full_mlp.png")
            fig.savefig(out_path_mlp, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  [MLP ]   → {out_path_mlp}  ← MLP showcase")

    print(f"\nDone — {len(targets)} main images + 1 MLP image in {OUT_DIR}/")


if __name__ == "__main__":
    main()
