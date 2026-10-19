# Data dictionary and measurement provenance

## Available source files

The available workbooks are:

- `data/raw/Frequency Response Data I.xlsx`
- `data/raw/Frequency Response Data II.xlsx`

The export script reads the `Averaged_Transfer` sheet from each workbook.
`Frequency Response Data I.xlsx` supplies the frequency column
`Frequency_Hz_rounded`; `Frequency Response Data II.xlsx` supplies
`Frequency_Hz`. Both are converted from Hz to MHz by division by `1e6`.

The processed export includes the following fields:

| Field | Meaning | Unit / representation |
|---|---|---|
| `N_PMUT` | PMUT array/device identifier used by the scripts | integer |
| `Frequency_MHz` | Frequency coordinate | MHz |
| `Amplitude_R_mean` | Mean measured response amplitude | Original workbook unit; not identified in the available export script |
| `Amplitude_R_std` | Standard deviation of response amplitude | Original workbook unit |
| `Phase_deg_mean` | Mean response phase | degrees |
| `Phase_deg_std` | Standard deviation of response phase | degrees |

The exporter concatenates the two averaged sheets, drops rows with missing mean
amplitude, sorts by device and frequency, and writes
`data/processed/pmuts_all.csv`. It does not export individual repeated sweeps.
The available processed CSV contains 16,000 rows: 1,000 frequency points for
each identifier `N_PMUT=1` through `N_PMUT=16`. The coordinates align by point
index within 5e-10 MHz; tiny floating-point differences remain between the two
workbook-derived portions.

## Recorded frequency grid

The `All_Sweeps_Labeled` sheets contain 1,000 points per sweep and three
recorded sweeps for each `N_PMUT`. The recorded frequencies span approximately
100 kHz to 2.5 MHz. Their adjacent-frequency ratio is approximately constant
at 1.0032273, so the grid is near-geometric (logarithmic), not uniformly
spaced: for zero-based point index `i`, `f_i ≈ 100 kHz × 1.0032273^i`
(`i=0…999`). The interval increases from about 0.322729 kHz at the
low-frequency end to 8.042281 kHz at the high-frequency end; its median is
1.611050 kHz. The arithmetic mean interval is 2.402402 kHz, which is not a
fixed step.

The `Averaged_Transfer` mean and sample standard deviation match calculations
from the three corresponding `All_Sweeps_Labeled` records. Workbook I's
README identifies its source as a LabOne HF2LI export, but notes that PMUT and
sweep labels were assigned by assumed chunk order. The available workbooks do
not include complete instrument settings or the native acquisition export, so
the numeric grid is established by the recorded frequencies, not by a
preserved sweep-configuration record.

## Processing and model input

The export script takes the precomputed `Averaged_Transfer` sheets, converts
frequency from Hz to MHz, concatenates them, drops rows with missing mean
amplitude, sorts by device and frequency, and writes the CSV. It does not
resample or interpolate frequencies, nor does it apply a signal filter or
smoothing operation. Workbook I's `Frequency_Hz_rounded` values are rounded
to 0.001 Hz in the source sheet; the exporter uses those values. Workbook II
frequencies retain their stored floating-point values.

Feature and target scaling are separate in the model code:

- For the scikit-learn MLP models, `StandardScaler` is fitted to the training
  feature matrix only and then used to transform the held-out target inputs.
  Pointwise features are `N_PMUT` and `Frequency_MHz`; vector-model features
  contain `N_PMUT` only. RF and GB feature inputs are not standardized.
- The target is `Amplitude_R_mean`. `src/Helpers/data_loader.py` creates
  `Amplitude_Scaled` by fitting one `StandardScaler` to the complete
  amplitude column before any experiment-specific training PMUTs are
  selected. The same scaler is then used to form training targets and to
  inverse-transform predictions and reference values.
- The PINN-family models train against amplitude in its original workbook
  scale and do not use this target scaler.

Therefore, target-scaling statistics are **not training-only**: the current
pipeline uses all PMUTs, including held-out targets, to calculate the
amplitude mean and standard deviation. Feature-scaling statistics for MLP
inputs are fitted on training rows only. Metrics are returned on the original
amplitude scale, but inverse-transforming them does not remove the target
scaler's held-out-data leakage. Results should not be described as having
training-only preprocessing statistics or as leakage-free.

For each evaluated target curve, MAE and RMSE are arithmetic means over the
evaluated recorded frequency samples, and R² uses the same unweighted samples.
The FWHM ROI metrics use the unweighted samples inside the ROI. No metric
integrates over frequency or weights residuals by adjacent frequency
intervals. Because the grid is near-logarithmic, samples are much denser per
MHz at the low-frequency end; accordingly, those regions receive more weight
per unit bandwidth than the high-frequency end. Aggregated tables then take
arithmetic means of the per-target metrics.

The FWHM-Window training approach is a separate model-data selection step, not
a filter applied to the exported spectra. For each training PMUT, it computes
the threshold `A_min + (A_max - A_min) / 2`, takes the minimum and maximum
recorded frequencies whose mean amplitude meets that threshold, and retains
training rows within that inclusive frequency interval. Half-height crossings
are not interpolated; the band edges are recorded sample frequencies.

The prediction cache CSVs are rounded by the training script when rows are
written; they should not be treated as full-precision raw predictions.
