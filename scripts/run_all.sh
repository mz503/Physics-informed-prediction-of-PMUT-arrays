#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all.sh — Build all PMUT analysis outputs from scratch
#
# Usage:
#   chmod +x scripts/run_all.sh   (first time only)
#   ./scripts/run_all.sh
#
# Scripts run (in order):
#   00  00_export_data.py       Excel -> data/processed/pmuts_all.csv
#
#   01  01_train_cache.py       Train all models ONCE and cache predictions
#                                 results/predictions/*.csv
#
#   04  04_metrics_table.py     MAE/RMSE/R² tables from cache (no retraining)
#                                 results/metrics/*.{csv,html}
#
#   06  06_progressive_training.py  Interactive HTML from cache
#                                 supplementary/explorer/progressive/
#
#   07  07_regime_aware_training.py  Interactive HTML from cache
#                                 supplementary/explorer/regime/
#
#   08  08_progressive_images.py  Static PNG images from cache
#                                 figures/main/
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

REQUIRED_IMPORTS="import pandas, numpy, sklearn, plotly, matplotlib, scipy, torch, openpyxl"
if [[ -f ".venv-runall/Scripts/python.exe" ]] && .venv-runall/Scripts/python.exe -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    PYTHON=".venv-runall/Scripts/python.exe"
elif [[ -x ".venv-runall/bin/python" ]] && .venv-runall/bin/python -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    PYTHON=".venv-runall/bin/python"
elif [[ -f ".venv/Scripts/python.exe" ]] && .venv/Scripts/python.exe -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    PYTHON=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python3" ]] && .venv/bin/python3 -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    PYTHON=".venv/bin/python3"
elif command -v python &>/dev/null && python -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    PYTHON="python"
elif command -v python3 &>/dev/null && python3 -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "No Python interpreter with the project dependencies is available." >&2
    echo "On Windows, run: .\\scripts\\run_all.ps1" >&2
    echo "On Bash, create a Python 3.12 environment and install: pip install -r requirements.txt" >&2
    exit 1
fi

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

step() { echo -e "\n${BOLD}${CYAN}▶  $*${RESET}"; }
ok()   { echo -e "${GREEN}✓  $* complete${RESET}"; }
fail() { echo -e "${RED}✗  $* failed — see error above${RESET}"; exit 1; }

run() {
    local script="$1"
    step "$script"
    "$PYTHON" "scripts/$script" && ok "$script" || fail "$script"
}

echo -e "${BOLD}PMUT Frequency Response Analysis — full rebuild${RESET}"
echo "Python : $("$PYTHON" --version)"
echo "Dir    : $PROJECT_DIR"
echo "Started: $(date)"

# ── 00  Export raw Excel data → CSV ───────────────────────────────────────────
run 00_export_data.py

# ── 01  Train all models, write prediction caches ─────────────────────────────
run 01_train_cache.py

# ── 04  Metrics tables (Vector · Full Spectrum · both training schemes) ────────
run 04_metrics_table.py

# ── 06  Progressive training — interactive HTML (reads cache) ─────────────────
run 06_progressive_training.py

# ── 07  Regime-aware training — interactive HTML (reads cache) ────────────────
run 07_regime_aware_training.py

# ── 08  Static matplotlib images (reads cache) ────────────────────────────────
run 08_progressive_images.py

# ── Summary ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}══════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  All outputs built successfully.${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════${RESET}"
echo ""
echo "Data / cache:"
echo "  data/processed/pmuts_all.csv"
echo "  results/predictions/progressive_cache.csv"
echo "  results/predictions/regime_cache.csv"
echo ""
echo "  results/metrics/"
echo "    progressive_metrics.csv  progressive_metrics.html"
echo "    regime_metrics.csv       regime_metrics.html"
echo ""
echo "  supplementary/explorer/progressive/"
echo "    all_combined.html        progressive_metrics.csv"
echo ""
echo "  supplementary/explorer/regime/"
echo "    all_combined.html        regime_metrics.csv"
echo ""
echo "  figures/main/"
echo "    pmut_NN_vector_full.png      (15 images — RF + GB)"
echo "    pmut_NN_vector_full_mlp.png  ( 1 image  — RF + GB + MLP)"
echo ""
echo "Finished: $(date)"
