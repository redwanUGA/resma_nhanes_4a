#!/usr/bin/env bash
set -euo pipefail

# Windows-friendly Bash runner for the NHANES workflow.
# Chooses an available Python command (python3, python, or py -3).

if command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
elif command -v py >/dev/null 2>&1; then
  PY_CMD="py -3"
else
  echo "Python not found on PATH. Please install Python 3 and retry." >&2
  exit 1
fi

# Create/refresh a local virtual environment and use it for all steps
VENV_DIR=".venv"
WIN_PY="$VENV_DIR/Scripts/python.exe"
NIX_PY="$VENV_DIR/bin/python"
if [ -x "$WIN_PY" ]; then
  VENV_PY="$WIN_PY"
elif [ -x "$NIX_PY" ]; then
  VENV_PY="$NIX_PY"
else
  echo "Creating virtual environment at $VENV_DIR" | tee -a "$LOG"
  eval "$PY_CMD -m venv $VENV_DIR" >> "$LOG" 2>&1
  if [ -x "$WIN_PY" ]; then VENV_PY="$WIN_PY"; else VENV_PY="$NIX_PY"; fi
fi
echo "Ensuring required packages are installed…" | tee -a "$LOG"
eval "$VENV_PY -m pip install --disable-pip-version-check -U pip setuptools wheel" >> "$LOG" 2>&1
eval "$VENV_PY -m pip install --disable-pip-version-check -r requirements.txt" >> "$LOG" 2>&1
PY_CMD="$VENV_PY"

LOG="run_log_$(date +'%Y%m%d_%H%M%S').txt"
echo "NHANES Multimarker Workflow (Windows Bash)" | tee -a "$LOG"

# If data folder missing or empty, download first
if [ ! -d nhanes_data ] || [ -z "$(ls -A nhanes_data 2>/dev/null)" ]; then
  echo "Downloading data..." | tee -a "$LOG"
  eval "$PY_CMD 1_download_data.py" >> "$LOG" 2>&1
else
  echo "Data directory present; skipping download." | tee -a "$LOG"
fi

echo "Merging & preprocessing..." | tee -a "$LOG"
eval "$PY_CMD 2_merge_preprocess_multimarker.py" >> "$LOG" 2>&1

echo "Running regressions..." | tee -a "$LOG"
eval "$PY_CMD 3_regression_multimarker.py" >> "$LOG" 2>&1

echo "Creating figures..." | tee -a "$LOG"
eval "$PY_CMD 4_visualize_multimarker.py" >> "$LOG" 2>&1

echo "Done. See output_data/ and output_figures/"

