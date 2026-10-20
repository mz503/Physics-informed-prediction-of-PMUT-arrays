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

Scaling and evaluation details: for scikit-learn MLPs, feature scaling is
separate from target scaling. The feature `StandardScaler` is fitted on each
training feature matrix only; RF and GB feature inputs are not standardized.
For RF, GB, and MLP, the `Amplitude_R_mean` target `StandardScaler` is fitted
separately for each progressive or regime-aware experiment, using only that
experiment's training PMUTs. PINN-family models use amplitudes in their
original scale; the RF-MDOF-PINN hybrid also fits its residual model on
original-scale amplitudes. The resulting predictions and reference values are
evaluated on the original amplitude scale.

Reported MAE, RMSE, and R² weight recorded frequency samples equally. ROI
metrics do the same over included ROI samples; none of these metrics weights
samples by frequency spacing or integrates over frequency. As the grid is
near-logarithmic, that gives more weight per unit bandwidth to the denser
low-frequency region. Summary tables average per-target metrics
arithmetically.

The prediction caches and metric exports already present in the repository
were generated before the training-only target-scaling correction. Rebuild
the full prediction caches before treating those supplied results as outputs
of the current pipeline. Selectively rerunning RF, GB, and MLP together can
replace the stale scaled-model entries while retaining previously cached
physics-model entries, which do not train against the scaled target. Other
selective reruns are blocked when they would preserve stale RF, GB, or MLP
entries.
These statements apply to the maintained cache-generation scripts; the
standalone [`FHWM.ipynb`](../supplementary/FHWM.ipynb) is a legacy workflow and
should not be used to substantiate training-only scaling without a separate
update.

The supplied workbooks do contain three `All_Sweeps_Labeled` records per
`N_PMUT`; their arithmetic means and sample standard deviations reproduce the
corresponding `Averaged_Transfer` values. The recorded 1,000-point grid spans
100 kHz to 2.5 MHz and is near-geometric, with adjacent-frequency ratio about
1.0032273. The processing script does not interpolate, resample, filter, or
smooth these frequency-response measurements. See
[`data_dictionary.md`](data_dictionary.md) for the spacing and FWHM-window
selection details.
