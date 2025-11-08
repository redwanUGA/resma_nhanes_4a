# NHANES Amalgam – Multimarker Inflammation Analysis (1999–2018)

End-to-end, reproducible pipeline:
- **Download** NHANES files across cycles
- **Merge & preprocess** dental amalgam, CRP, CBC-derived indices (NLR, MLR, PLR, SII), smoking, alcohol, BMI, diabetes, mercury
- **Run WLS regressions** (per cycle, adjusted) for **CRP, NLR, MLR, PLR, SII**
- **Generate figures** (forest, trend, heatmaps) for each marker + behavior distributions

## Quickstart
```bash
python3 1_download_data.py
python3 2_merge_preprocess_multimarker.py
python3 3_regression_multimarker.py
python3 4_visualize_multimarker.py
```

Or run the full workflow:
```bash
chmod +x run_workflow.sh
./run_workflow.sh
```

### Outputs
- `output_data/nhanes_merged_multimarker.csv` – unified dataset
- `output_data/regression_results_by_cycle_<MARKER>.csv` – β ± 95% CI per cycle
- `output_data/stratified_results_<MARKER>.csv` – β by Sex × Race
- `output_figures/` – Forest, Trend, Heatmap per marker + CRP/behavior figs

> Note: Regressions use **weighted least squares (WTMEC2YR)** as a practical approximation to complex survey design.
