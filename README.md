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
This script auto-detects `python3`/`python`/`py -3`, provisions `.venv/`, installs `requirements.txt`, and logs to `run_log_YYYYMMDD_HHMMSS.txt` across Linux, macOS, and Windows (Git Bash).

### Outputs
- `output_data/nhanes_merged_multimarker.csv` – unified analytic dataset
- `output_data/table1_demographics_by_amalgam.csv` – weighted characteristics for Table 1
- `output_data/sample_overview_metrics.csv` – high-level Ns/weighted percentages
- `output_data/table2_regression_amalgam_mercury_adjusted.csv` – Model 1 vs Model 2 contrasts for all inflammatory markers (Table 2)
- `output_data/marker_weighted_geomeans.csv` – geometric means across amalgam and mercury strata
- `output_data/mercury_on_amalgam_regression.csv` – β (log-mercury ~ amalgam surfaces) supporting Fig. 2 / Section 4.3
- `output_data/regression_cycle_crp.csv` & `output_data/stratified_crp_heatmap.csv` – inputs for Figures 4–5
- `output_data/mediation_results_crp.csv` & `output_data/behavior_interaction_results.csv` – mediation + interaction diagnostics for Sections 4.8–4.9
- Legacy exports: `regression_results_by_cycle_<MARKER>.csv`, `stratified_results_<MARKER>.csv`
- `output_figures/` now houses the publication-ready panels: `Fig1_BloodMercury_Distribution.png` through `Fig7_Smoking_Drinking.png` aligned with Sections 4.1–4.10

> Note: Regressions use **weighted least squares (WTMEC2YR)** as a practical approximation to complex survey design.

## Required NHANES `.xpt` files
Place the original SAS transport files you download from the [CDC NHANES portal](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=1999) into `nhanes_data/` with the exact names listed below (one row per 2-year cycle):

| Cycle | Demographics | Oral Health | CRP / hsCRP | CBC | Smoking | Alcohol | Body Measures | Diabetes | Mercury |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1999-2000 | DEMO.xpt | OHXDENT.xpt | LAB11.xpt | LAB25.xpt | SMQ.xpt | ALQ.xpt | BMX.xpt | DIQ.xpt | LAB06HM.xpt |
| 2001-2002 | DEMO_B.xpt | OHXDEN_B.xpt | L11_B.xpt | LAB25_B.xpt | SMQ_B.xpt | ALQ_B.xpt | BMX_B.xpt | DIQ_B.xpt | L06_2_B.xpt |
| 2003-2004 | DEMO_C.xpt | OHXDEN_C.xpt | L11_C.xpt | LAB25_C.xpt | SMQ_C.xpt | ALQ_C.xpt | BMX_C.xpt | DIQ_C.xpt | L06BMT_C.xpt |
| 2005-2006 | DEMO_D.xpt | OHXDEN_D.xpt | CRP_D.xpt | CBC_D.xpt | SMQ_D.xpt | ALQ_D.xpt | BMX_D.xpt | DIQ_D.xpt | PBCD_D.xpt |
| 2007-2008 | DEMO_E.xpt | OHXDEN_E.xpt | CRP_E.xpt | CBC_E.xpt | SMQ_E.xpt | ALQ_E.xpt | BMX_E.xpt | DIQ_E.xpt | PBCD_E.xpt |
| 2009-2010 | DEMO_F.xpt | OHXDEN_F.xpt | CRP_F.xpt | CBC_F.xpt | SMQ_F.xpt | ALQ_F.xpt | BMX_F.xpt | DIQ_F.xpt | PBCD_F.xpt |
| 2011-2012 | DEMO_G.xpt | OHXDEN_G.xpt | CRP_G.xpt | CBC_G.xpt | SMQ_G.xpt | ALQ_G.xpt | BMX_G.xpt | DIQ_G.xpt | PBCD_G.xpt |
| 2013-2014 | DEMO_H.xpt | OHXDEN_H.xpt | CRP_H.xpt | CBC_H.xpt | SMQ_H.xpt | ALQ_H.xpt | BMX_H.xpt | DIQ_H.xpt | PBCD_H.xpt |
| 2015-2016 | DEMO_I.xpt | OHXDEN_I.xpt | CRP_I.xpt | CBC_I.xpt | SMQ_I.xpt | ALQ_I.xpt | BMX_I.xpt | DIQ_I.xpt | PBCD_I.xpt |
| 2017-2018 | DEMO_J.xpt | OHXDEN_J.xpt | HSCRP_J.xpt | CBC_J.xpt | SMQ_J.xpt | ALQ_J.xpt | BMX_J.xpt | DIQ_J.xpt | PBCD_J.xpt |

### When downloads are unavailable
- `1_download_data.py` keeps retry/open internet logic, but in restricted environments Python's `requests` may fail to resolve `wwwn.cdc.gov` even though simple `curl` calls succeed. In such cases, manually download via a browser and move the files into `nhanes_data/`.
- `2_merge_preprocess_multimarker.py` now requires the real `.xpt` files. If nothing is present it will emit a warning, write an empty `output_data/nhanes_merged_multimarker.csv`, and exit cleanly—no synthetic data are generated.
- `3_regression_multimarker.py` and `4_visualize_multimarker.py` detect empty inputs and skip computation/figure generation while still completing without errors so that automated workflows can run end-to-end even when the raw data are missing.
