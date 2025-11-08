#!/bin/bash
set -e
LOG="run_log_$(date +'%Y%m%d_%H%M%S').txt"

echo "🚀 NHANES Multimarker Workflow" | tee -a "$LOG"

if [ ! -d nhanes_data ] || [ -z "$(ls -A nhanes_data 2>/dev/null)" ]; then
  echo "📥 Downloading data..." | tee -a "$LOG"
  python3 1_download_data.py >> "$LOG" 2>&1
else
  echo "⏭️  Data already present." | tee -a "$LOG"
fi

echo "🧬 Merging & preprocessing..." | tee -a "$LOG"
python3 2_merge_preprocess_multimarker.py >> "$LOG" 2>&1

echo "📐 Running regressions..." | tee -a "$LOG"
python3 3_regression_multimarker.py >> "$LOG" 2>&1

echo "🎨 Creating figures..." | tee -a "$LOG"
python3 4_visualize_multimarker.py >> "$LOG" 2>&1

echo "✅ Done. See output_data/ and output_figures/"
