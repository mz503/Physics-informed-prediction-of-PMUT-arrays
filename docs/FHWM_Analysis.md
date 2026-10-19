# PMUT Frequency Response Prediction — Full Analysis
## Physics-Informed Machine Learning with FWHM

---

## 1. Project Overview

This project addresses a practical problem in MEMS device characterization: given measured frequency response data from a small number of PMUTs (Piezoelectric Micromachined Ultrasonic Transducers), can a machine learning model accurately predict the frequency response of an unseen target PMUT?

Each PMUT has a characteristic **amplitude-vs-frequency** curve with a distinct resonance peak. The goal is cross-device generalization — training on devices 1 through N and predicting device N+1.

### Dataset

| Property | Details |
|----------|---------|
| Total devices | 16 PMUTs |
| Source workbooks | `Frequency Response Data I.xlsx`, `Frequency Response Data II.xlsx` |
| Frequency points | 1000 per PMUT in the available processed CSV |
| Target variable | `Amplitude_R_mean`, normalized via `StandardScaler` |

### Experimental Regimes

Three regimes test different levels of generalization difficulty:

| Regime | Training PMUTs | Prediction PMUTs | Character |
|--------|---------------|------------------|-----------|
| **Regime 1** | [1] | Predict 2; cross-regime [1,2] predicts 3 | PMUTs 1–2 |
| **Regime 2** | Grow from [3] to [3–10] | Predict 4–11; cross-regime [3–11] predicts 12 | PMUTs 3–11 |
| **Regime 3** | Grow from [12] to [12–15] | Predict 13–16 | PMUTs 12–16 |

Each regime uses a **progressive training** strategy: at each step, one more PMUT is added to the training set and the next unseen PMUT is predicted.

---

## 2. Approaches

The scripts compare vector and pointwise prediction approaches across RF, GB,
MLP, PINN, MDOF-PINN, PINN-Informed-Loss, and RF-MDOF-PINN.

---

### Section 1 — Baseline: Pointwise Prediction

**What it does:**
Treats each `(N_PMUT, Frequency_MHz)` pair as an independent training sample. The model learns to map device identity and query frequency directly to amplitude.

**Features:** `[N_PMUT, Frequency_MHz]` → `Amplitude_Scaled`

**Training size:** `n_train_PMUTs × n_freqs` individual data points (~1000–6000 rows depending on regime step)

**Prediction:** Query the model at every frequency of the target PMUT → reconstruct full curve

**Models used:** `RandomForestRegressor`, `GradientBoostingRegressor`

**Strengths:**
- Large training set (many (freq, amp) pairs per PMUT)
- Model sees full spectral shape of training PMUTs
- Simple, interpretable setup

**Weaknesses:**
- Model must extrapolate to a new `N_PMUT` value it has never seen
- No explicit physics knowledge — treats PMUT number as an arbitrary integer
- Off-resonance baseline dominates training signal (most points are flat)

---

### Section 2 — Multi-Output Prediction

**What it does:**
Treats the entire 1000-point amplitude curve as a single structured output vector. One model is trained to map from PMUT number to the full curve at once.

**Features:** `[N_PMUT]` — one row per device

**Training size:** `n_train_PMUTs` rows (e.g., 1–6 rows depending on regime step)

**Output:** Full 1000-point amplitude vector predicted in a single inference call

**Models used:**
- `RandomForestRegressor` — natively supports multi-output
- `MultiOutputRegressor(GradientBoostingRegressor)` — one GB per output dimension

**Strengths:**
- Captures global curve shape as a unit
- Can interpolate between known whole-device profiles
- No per-frequency prediction needed

**Weaknesses:**
- Extremely small training set (only as many rows as training PMUTs)
- With 1–2 training samples, the model has almost no statistical power
- GB via `MultiOutputRegressor` fits 1000 independent regressors — computationally expensive
- Highly prone to negative R² when training set is tiny

---

### Section 3 — FWHM Background

**What is FWHM?**

Full Width at Half Maximum (FWHM) is the width of the resonance peak measured at **50% of the peak height above the baseline**:

```
baseline  = A_min                        (lowest amplitude in the sweep)
h_half    = A_min + (A_max - A_min) / 2  (half of peak height above baseline)
x1 (f_lo) = lowest  frequency where A(f) >= h_half
x2 (f_hi) = highest frequency where A(f) >= h_half
FWHM      = x2 - x1
Q-factor  = f_peak / FWHM
```

> **Why baseline correction matters:** Without subtracting the baseline, `A_max / 2` can still be above the entire off-resonance floor, causing every frequency point to qualify as "above half-max." The corrected threshold correctly identifies the true peak band edges x1 and x2.

**Physical meaning for PMUTs:**

| FWHM | Q-factor | Resonance behavior |
|------|----------|--------------------|
| Small | High | Sharp, selective peak — high sensitivity at resonance frequency |
| Large | Low | Broad peak — more bandwidth, less selective |

The Q-factor characterizes resonator quality. Higher Q means less energy loss per cycle.

---

### Section 4 — FWHM-Window Training

**What it does:**
Uses FWHM to **filter the training data** to the resonance peak region only. For each training PMUT, data is restricted to `[f_lo, f_hi]` — the frequencies where `A(f) >= h_half`. At prediction time, only the target PMUT's own FWHM band is queried.

**Features:** `[N_PMUT, Frequency_MHz]` (same as pointwise, but far fewer rows)

**Training size:** ~50–150 points per training PMUT (vs ~1000 for pointwise)

**Prediction scope:** In-band points of the target PMUT only

**Models used:** `RandomForestRegressor`, `GradientBoostingRegressor`

**Comparison to pointwise:**

| | Standard Pointwise | FWHM-Window |
|-|--------------------|-------------|
| Training data | All ~1000 pts (flat baseline + peak) | ~50–150 pts (peak band only) |
| Model sees | Full spectral shape incl. off-resonance | Only resonance peak |
| Prediction range | Full sweep | In-band points only |
| MAE evaluated on | Full curve | In-band points only |

**Hypothesis:** The resonance peak region carries the most device-specific information. Filtering to this region removes noisy off-resonance baseline data and forces the model to focus on the peak shape, potentially improving in-band accuracy while using far fewer training points.

**Plot annotation:** Each plot shows shaded bands for each training PMUT's FWHM window and the target PMUT's FWHM window, so the filtering is visually verifiable.

---

## 3. Why R² Can Be Negative

R² (coefficient of determination) is defined as:

```
R² = 1 - SS_res / SS_tot

SS_res = Σ (actual_i - predicted_i)²   ← sum of squared model errors
SS_tot = Σ (actual_i - mean(actual))²  ← total variance of the target
```

**R² < 0 means the model performs worse than simply predicting the mean value for every point.**

Formally: if `SS_res > SS_tot`, then `R² < 0`.

### When does this happen in this notebook?

**1. Multi-Output with a single training PMUT (Regime 1, Step 1: Train [1] → Predict 2)**

The model has exactly **one training sample**: `X = [[1]]`, `y = [full curve of PMUT 1]`. A random forest with one sample memorizes that one curve and returns it for any input. When predicting PMUT 2, the model outputs PMUT 1's curve verbatim. If the peak position, height, or shape differ between the two devices, the prediction error `SS_res` can exceed the variance `SS_tot` of PMUT 2's curve → R² < 0.

**2. Pointwise across large regime gaps (later progressive steps)**

In later progressive steps, the model extrapolates to a device index beyond the training PMUTs. If the predicted curve is consistently offset from the actual, the error can exceed the signal's natural variance → R² < 0.

**3. FWHM-Window in-band R²**

The in-band region is a narrow frequency slice — the resonance peak only. If the peak of the target PMUT is shifted relative to the training PMUTs (even slightly), the model predicts a peak at the wrong frequency. Within the target's narrow FWHM band, even a small frequency shift causes large squared errors relative to the variance of that region → R² < 0.

### Key interpretation

| R² | Interpretation |
|----|----------------|
| > 0.9 | Excellent — model captures the curve shape well |
| 0.5–0.9 | Acceptable — reasonable prediction with some shape error |
| 0–0.5 | Poor — some structure captured but significant error |
| < 0 | Worse than mean predictor — approach failed for this step |

**R² < 0 is not always a model failure** — it signals that the particular train→predict pair was **too far from the training distribution** for that approach. The regime/approach combinations where R² stays consistently positive are the reliable operating conditions for each method.

---

## 4. Analysis Framework Summary

```
Regime 1 (PMUTs 1–2; cross-regime prediction of 3)
Regime 2 (PMUTs 3–11; cross-regime prediction of 12)
Regime 3 (PMUTs 12–16)
```

**Expected ranking (best to worst MAE per regime):**
- Regime 1: Pointwise likely best (most data per model), Multi-Output worst (1-2 samples)
- Regime 2: Pointwise and FWHM-Window competitive; Multi-Output improves with more PMUTs
- Regime 3: FWHM-Window may be more robust as it focuses on peak shape invariants

**FWHM-Window MAE note:** This MAE is not directly comparable to Pointwise/Multi-Output MAE — it is evaluated only on the in-band frequency points (the resonance peak), not the full sweep. In-band MAE reflects how well the model captures the peak shape specifically.

---

## 5. Model Details

| Model | Type | Notes |
|-------|------|-------|
| `RandomForestRegressor` | Ensemble of decision trees | Native multi-output support; robust to outliers |
| `GradientBoostingRegressor` | Boosted trees | Sequential fitting; higher accuracy but slower |
| `MultiOutputRegressor(GB)` | Wrapper | Fits one GB per output dimension (1000 regressors) |

See `src/Helpers/model_utils.py` for the current tree/MLP implementation
settings and the individual model modules for the physics-model parameters.
These settings are source-code defaults, not a recovered record of the
hyperparameters used to generate every pre-existing result.

---

## 6. Files

| File | Description |
|------|-------------|
| `FHWM.ipynb` | Main notebook with all models, plots, and results |
| `Frequency Response Data I.xlsx` | Supplied source workbook; see `data_dictionary.md` |
| `Frequency Response Data II.xlsx` | Supplied source workbook; see `data_dictionary.md` |
| `FHWM_Analysis.md` | This analysis document |
