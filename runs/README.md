# Reported runs

## Latest full pipeline

- **Status:** Completed successfully (all stages exited with code 0).
- **Run date:** 2026-10-19.
- **Command:** `.\scripts\run_all.ps1` on Windows.
- **Environment:** `.venv-runall`, Python 3.12.14. The pinned direct
  dependencies are listed in [`requirements.txt`](../requirements.txt).
- **Pipeline:** Data export, progressive and regime-aware training/cache
  generation for the seven implemented models, metrics, interactive
  explorers, and progressive figure generation.
- **Data export:** 16,000 rows total, 1,000 rows for each PMUT ID 1–16.
- **Generated figures:** 15 main PMUT figures and one MLP showcase figure
  under [`figures/main/`](../figures/main/).

The output files from this run are available in:

- [`results/predictions/`](../results/predictions/) — progressive and regime
  prediction caches.
- [`results/metrics/`](../results/metrics/) — progressive and regime metric
  tables and HTML views.
- [`supplementary/explorer/`](../supplementary/explorer/) — interactive
  progressive and regime explorers.
- [`figures/main/`](../figures/main/) — generated main figures.

The shell reported successful completion of all data preparation, training,
metrics, explorers, and figures. The run duration, per-model training logs,
and machine/CPU details were not captured in durable run metadata. No
checkpoints, fitted scalers, or serialized per-run configurations were saved
under `runs/`; the generated results therefore document the outputs, but do
not provide a complete archival record of the training state. The transient
terminal output showed very negative R² values for some model/regime
combinations, so successful pipeline execution should not be read as a
validation of model quality.
