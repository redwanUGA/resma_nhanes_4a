#!/bin/sh
set -eu

LOG="run_log_$(date +'%Y%m%d_%H%M%S').txt"

log() {
  msg=$1
  printf '%s\n' "$msg" | tee -a "$LOG"
}

log "[INFO] NHANES Multimarker Workflow"

# Pick an available Python command (python3, python, py -3)
PY_CMD=""
PY_FLAG=""
if command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
elif command -v py >/dev/null 2>&1; then
  PY_CMD="py"
  PY_FLAG="-3"
else
  log "[ERROR] Python 3 interpreter not found on PATH (python3, python, py -3)."
  exit 1
fi
DISPLAY_CMD=$PY_CMD
if [ -n "$PY_FLAG" ]; then
  DISPLAY_CMD="$DISPLAY_CMD $PY_FLAG"
fi
log "[INFO] Using Python command: $DISPLAY_CMD"

run_py_cmd() {
  if [ -n "$PY_FLAG" ]; then
    "$PY_CMD" "$PY_FLAG" "$@"
  else
    "$PY_CMD" "$@"
  fi
}

VENV_DIR=".venv"
WIN_PY="$VENV_DIR/Scripts/python.exe"
NIX_PY="$VENV_DIR/bin/python"
VENV_PY=""

setup_venv() {
  if [ -x "$WIN_PY" ]; then
    VENV_PY="$WIN_PY"
  elif [ -x "$NIX_PY" ]; then
    VENV_PY="$NIX_PY"
  fi

  if [ -n "$VENV_PY" ]; then
    log "[INFO] Ensuring Python dependencies from requirements.txt"
    "$VENV_PY" -m pip install --disable-pip-version-check -q -r requirements.txt >> "$LOG" 2>&1
    return
  fi

  log "[INFO] Creating virtual environment at $VENV_DIR"
  run_py_cmd -m venv "$VENV_DIR" >> "$LOG" 2>&1

  if [ -x "$WIN_PY" ]; then
    VENV_PY="$WIN_PY"
  else
    VENV_PY="$NIX_PY"
  fi

  log "[INFO] Upgrading pip, setuptools, and wheel"
  "$VENV_PY" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel >> "$LOG" 2>&1

  log "[INFO] Installing required packages"
  "$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt >> "$LOG" 2>&1
}

setup_venv
PY_BIN="$VENV_PY"
log "[INFO] Activated venv Python: $PY_BIN"

run_py() {
  step_msg=$1
  script=$2
  log "[STEP] $step_msg"
  if ! "$PY_BIN" "$script" >> "$LOG" 2>&1; then
    log "[ERROR] $script failed. Check $LOG for details."
    exit 1
  fi
}

NEED_DOWNLOAD=0
if [ ! -d nhanes_data ] || [ -z "$(ls -A nhanes_data 2>/dev/null)" ]; then
  NEED_DOWNLOAD=1
fi

if [ "$NEED_DOWNLOAD" -eq 1 ]; then
  run_py "Downloading NHANES source files..." "1_download_data.py"
else
  log "[INFO] Data directory detected; skipping download."
fi

run_py "Merging and preprocessing data..." "2_merge_preprocess_multimarker.py"
run_py "Running regression models..." "3_regression_multimarker.py"
run_py "Generating figures..." "4_visualize_multimarker.py"

log "[OK] Done. See output_data/, output_figures/, and $LOG for details."
