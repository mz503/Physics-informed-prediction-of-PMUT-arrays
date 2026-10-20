# Physics-Informed Predictive Modelling of PMUT Arrays

This project processes frequency-response measurements for PMUT arrays and evaluates seven implemented models using progressive and regime-aware training:
Random Forest (RF), Gradient Boosting (GB), MLP, PINN, MDOF-PINN,
PINN-Informed-Loss, and RF-MDOF-PINN.

## Current project contents

- `data/raw/` contains the two supplied Excel workbooks and the available `Pmuts.zip` archive.
- `data/processed/pmuts_all.csv` is the unified spectrum file exported by the preparation script.
- `results/predictions/` contains the cached progressive and regime predictions.
- `results/metrics/` contains the exported metric tables.
- `results/direct_fits/` contains direct physics/model fit tables and HTML exports.
- `src/Helpers/` contains the model, data, physics, plotting, and evaluation modules.
- `scripts/` contains runnable preparation, training, metrics, and figure scripts.
- `figures/` contains the available prediction and parameter-trajectory plots.
- `docs/` contains the project notes and data dictionary.
- `supplementary/` contains interactive explorers, model explanations, and the notebook.

See [docs/data_dictionary.md](docs/data_dictionary.md) and
[docs/reproducibility.md](docs/reproducibility.md) for known data and provenance
limitations.

## Installation

Use Python 3.12. The exact original run environment was not included, so
`requirements.txt` pins a consistent current dependency set rather than
claiming to recover the historical versions. The Windows run script creates a
separate `.venv-runall` environment and installs those requirements, leaving
the supplied `.venv` untouched.

## Prepare, train, and reproduce

From PowerShell, run the complete pipeline from the project root:

```powershell
.\scripts\run_all.ps1
```

If Python 3.12 is not discoverable through the Windows `py` launcher, provide
its executable path:

```powershell
.\scripts\run_all.ps1 -PythonPath "C:\Path\To\Python312\python.exe"
```

The Bash entry point remains available after the dependencies are installed:

```bash
bash scripts/run_all.sh
```

Training all seven models is computationally expensive. `01_train_cache.py`
also supports `--models` and `--pinn-only`; cached predictions are consumed by
the metrics, explorer, and image-export scripts.

## Reproducibility status

The original project does not include standalone training/validation/test
configuration files, run logs, checkpoints, fitted scalers, or recorded
software versions. Several model settings and regime definitions remain
embedded in the scripts. The available processed file is based on the
`Averaged_Transfer` Excel sheets; raw repeat-level sweeps, acquisition settings,
and exact repeat counts/measurement identifiers were not supplied in the
available files. Do not interpret the current outputs as a complete
reproduction package until these items and provenance are added.

The original `scratch/` directory was inadvertently removed during this
reorganization before its experimental images, `params_table.json`, and
`extract_param_table.py` could be moved. Those scratch artifacts are not
present in this reorganized copy.

## Licensing and citation

If you use or build upon this work, please cite the associated publication(s) and this repository, including the release or commit used. Third-party materials remain subject to their original licences and attribution requirements.
