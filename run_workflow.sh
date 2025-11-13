#!/bin/bash
set -e
LOG="run_log_$(date +'%Y%m%d_%H%M%S').txt"

echo "[INFO] NHANES Multimarker Workflow" | tee -a "$LOG"

if [ ! -d nhanes_data ] || [ -z "$(ls -A nhanes_data 2>/dev/null)" ]; then
  echo "[INFO] Downloading data..." | tee -a "$LOG"
  python3 1_download_data.py >> "$LOG" 2>&1
else
  echo "[SKIP]  Data already present." | tee -a "$LOG"
fi

echo "[INFO] Merging & preprocessing..." | tee -a "$LOG"
python3 2_merge_preprocess_multimarker.py >> "$LOG" 2>&1

echo "[INFO] Running regressions..." | tee -a "$LOG"
python3 3_regression_multimarker.py >> "$LOG" 2>&1

echo "[INFO] Creating figures..." | tee -a "$LOG"
python3 4_visualize_multimarker.py >> "$LOG" 2>&1

echo "[OK] Done. See output_data/ and output_figures/"
