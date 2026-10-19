# Reproducibility and known limitations

## Available

- Source Excel workbooks and a preprocessed averaged-spectrum CSV.
- Progressive and regime prediction caches.
- Progressive/regime metric exports and direct physics-fit tables.
- Source scripts and seven model identifiers in `scripts/01_train_cache.py`.
- Static figures, interactive HTML explorers, and explanatory material.

## Missing from the supplied project

- Native instrument exports and complete instrument acquisition settings.
- Validation that the workbook's assumed PMUT/sweep chunk labels correspond to
  the intended physical devices and acquisition runs.
- Standalone train/validation/test lists, regime/ROI configuration files, and
  recorded seeds for each reported run.
- Full run-specific hyperparameters, loss weights, modal constraints,
  parameter bounds, stopping records, and training/validation logs.
- Saved model checkpoints, fitted scalers, software-version snapshots, and
  runtimes.
- Source-data panels and figure-generation scripts for every supplied figure.
- Code/data licence and citation metadata.

Do not fill these gaps with values copied from narrative examples unless those
values are verified against the exact run that generated the reported results.
The target amplitude scaler currently fits against all PMUT spectra before a
training split, which leaks held-out distribution information. Review this
behavior before using the exported metrics as a final evaluation.

Scaling and evaluation details: MLP input-feature scalers are fitted on the
training feature matrices, but the shared target `StandardScaler` for
`Amplitude_R_mean` is fitted in `load_data()` on all PMUTs before experiment
splits. Thus preprocessing statistics are not wholly training-only, and
held-out target amplitudes influence the target scaling statistics. Reported
MAE, RMSE, and R² are computed on the original amplitude scale with equal
weight per recorded frequency sample; they are not weighted or integrated by
frequency interval. Since the grid is near-logarithmic, that gives more weight
per unit bandwidth to the denser low-frequency region. Summary tables average
per-target metrics arithmetically.

The supplied workbooks do contain three `All_Sweeps_Labeled` records per
`N_PMUT`; their arithmetic means and sample standard deviations reproduce the
corresponding `Averaged_Transfer` values. The recorded 1,000-point grid spans
100 kHz to 2.5 MHz and is near-geometric, with adjacent-frequency ratio about
1.0032273. The processing script does not interpolate, resample, filter, or
smooth these frequency-response measurements. See
[`data_dictionary.md`](data_dictionary.md) for the spacing and FWHM-window
selection details.
