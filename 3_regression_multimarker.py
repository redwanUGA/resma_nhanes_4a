# 3_regression_multimarker.py
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTPUT_DIR = "output_data"
MARKERS = ["CRP", "NLR", "MLR", "PLR", "SII"]
WEIGHT_COL = "WTMEC2YR"


def safe_log(series: pd.Series) -> pd.Series:
    """Natural log transform that gracefully handles non-positive values."""
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values > 0)
    return np.log(values)


def weighted_total(weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce")
    return float(weights.dropna().sum()) if weights is not None else np.nan


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    w = weights[mask]
    if w.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=w))


def weighted_percent(condition: pd.Series, weights: pd.Series) -> float:
    indicator = condition.astype(float)
    return weighted_mean(indicator, weights) * 100.0


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    values, weights = values[mask], weights[mask]
    sorter = np.argsort(values)
    values, weights = values.iloc[sorter], weights.iloc[sorter]
    cumulative = weights.cumsum()
    total = cumulative.iloc[-1]
    if total == 0:
        return np.nan
    cumulative /= total
    return float(np.interp(quantile, cumulative, values))


def weighted_geometric_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values > 0
    mask &= weights.notna()
    if mask.sum() == 0:
        return np.nan
    log_vals = np.log(values[mask])
    w = weights[mask]
    if w.sum() == 0:
        return np.nan
    return float(np.exp(np.average(log_vals, weights=w)))


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that modeling columns exist even if empty."""
    needed = [
        "BMXBMI",
        "PIR",
        "DIQ010",
        "SMQ020",
        "SMQ040",
        "ALQ101",
        "Gender",
        "Race",
        "log_mercury",
        "amalgam_group",
        "mercury_quartile",
        "Cycle",
        "CycleMidpoint",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    if "INDFMPIR" in df.columns:
        df["PIR"] = df["INDFMPIR"].where(df["INDFMPIR"].notna(), df["PIR"])
    df["DiabetesFlag"] = (df["DIQ010"] == 1).astype(float)
    df["EverSmoker"] = (df["SMQ020"] == 1).astype(float)
    df["CurrentSmoker"] = df["SMQ040"].isin([1, 2]).astype(float)
    df["AlcoholUse"] = (df["ALQ101"] == 1).astype(float)
    return df


def build_design_matrix(
    data: pd.DataFrame,
    include_mercury: bool = True,
    interactions: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    X = pd.DataFrame(
        {
            "amalgam_surfaces": data["amalgam_surfaces"],
            "Age": data["Age"],
            "BMI": data["BMXBMI"],
            "PIR": data["PIR"],
            "Diabetes": data["DiabetesFlag"],
            "EverSmoker": data["EverSmoker"],
            "CurrentSmoker": data["CurrentSmoker"],
            "AlcoholUse": data["AlcoholUse"],
        }
    )
    if include_mercury:
        X["log_mercury"] = data["log_mercury"]
    dummies = []
    if "Gender" in data.columns:
        dummies.append(pd.get_dummies(data["Gender"], prefix="Sex", drop_first=True))
    if "Race" in data.columns:
        dummies.append(pd.get_dummies(data["Race"], prefix="Race", drop_first=True))
    if dummies:
        X = pd.concat([X] + dummies, axis=1)
    if interactions:
        for term, (main_effect, modifier) in interactions.items():
            if main_effect in X.columns and modifier in data.columns:
                X[term] = X[main_effect] * data[modifier]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = sm.add_constant(X, has_constant="add")
    return X


def prepare_subset(
    df: pd.DataFrame,
    outcome: str,
    include_mercury: bool,
    min_n: int,
    log_outcome: bool,
) -> Optional[pd.DataFrame]:
    required = ["amalgam_surfaces", "Age", "BMXBMI", "PIR", WEIGHT_COL, outcome]
    if include_mercury:
        required.append("log_mercury")
    subset = df.dropna(subset=required).copy()
    subset[WEIGHT_COL] = pd.to_numeric(subset[WEIGHT_COL], errors="coerce")
    subset = subset[subset[WEIGHT_COL] > 0]
    if log_outcome:
        subset["model_y"] = safe_log(subset[outcome])
    else:
        subset["model_y"] = pd.to_numeric(subset[outcome], errors="coerce")
    subset = subset.dropna(subset=["model_y"])
    if include_mercury and not subset["log_mercury"].notna().any():
        return None
    if len(subset) < min_n:
        return None
    return subset


def run_wls(
    df: pd.DataFrame,
    outcome: str,
    include_mercury: bool = True,
    interactions: Optional[Dict[str, Tuple[str, str]]] = None,
    min_n: int = 200,
    log_outcome: bool = True,
) -> Optional[Tuple[sm.regression.linear_model.RegressionResultsWrapper, int, float]]:
    subset = prepare_subset(df, outcome, include_mercury, min_n, log_outcome)
    if subset is None:
        return None
    X = build_design_matrix(subset, include_mercury=include_mercury, interactions=interactions)
    y = subset["model_y"]
    w = subset[WEIGHT_COL]
    try:
        model = sm.WLS(y, X, weights=w).fit()
    except Exception:
        return None
    return model, len(subset), float(w.sum())


def extract_param(
    model: sm.regression.linear_model.RegressionResultsWrapper, param: str
) -> Tuple[float, float, float, float]:
    if param not in model.params:
        return (np.nan, np.nan, np.nan, np.nan)
    beta = model.params[param]
    se = model.bse[param]
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    p = model.pvalues[param]
    return (float(beta), float(ci_low), float(ci_high), float(p))


def create_table1(df: pd.DataFrame) -> None:
    weights = df[WEIGHT_COL]
    total_weight = weighted_total(weights)
    rows = []
    grouped = df.groupby("amalgam_group", dropna=False)
    for name, grp in grouped:
        group_weights = grp[WEIGHT_COL]
        weighted_n = weighted_total(group_weights)
        rows.append(
            {
                "Amalgam_Group": "Missing" if pd.isna(name) else name,
                "Weighted_N": weighted_n,
                "Weighted_%": (weighted_n / total_weight * 100.0) if total_weight else np.nan,
                "Median_Age": weighted_quantile(grp["Age"], group_weights, 0.5),
                "Mean_BMI": weighted_mean(grp["BMXBMI"], group_weights),
                "Median_PIR": weighted_quantile(grp["PIR"], group_weights, 0.5),
                "Median_LBXTHG": weighted_quantile(grp["LBXTHG"], group_weights, 0.5),
                "Percent_Female": weighted_percent(grp["Gender"] == "Female", group_weights),
                "Percent_CurrentSmoker": weighted_percent(
                    grp["CurrentSmoker"] == 1, group_weights
                ),
                "Percent_Drinker": weighted_percent(grp["AlcoholUse"] == 1, group_weights),
            }
        )
    table1 = pd.DataFrame(rows)
    table1.to_csv(os.path.join(OUTPUT_DIR, "table1_demographics_by_amalgam.csv"), index=False)


def create_overview_metrics(df: pd.DataFrame) -> None:
    weights = df[WEIGHT_COL]
    overview = {
        "Participants_unweighted": int(len(df)),
        "Weighted_population_total": weighted_total(weights),
        "Percent_any_amalgam": weighted_percent(df["amalgam_surfaces"] > 0, weights),
        "Percent_six_plus_amalgam": weighted_percent(df["amalgam_surfaces"] >= 6, weights),
        "Median_mercury": weighted_quantile(df["LBXTHG"], weights, 0.5),
        "Median_CRP": weighted_quantile(df["CRP"], weights, 0.5),
    }
    complete_mask = df["amalgam_surfaces"].notna() & df["LBXTHG"].notna()
    for marker in MARKERS:
        complete_mask &= df[marker].notna()
    overview["Complete_case_unweighted"] = int(complete_mask.sum())
    overview["Complete_case_weighted"] = weighted_total(weights[complete_mask])
    pd.DataFrame([overview]).to_csv(
        os.path.join(OUTPUT_DIR, "sample_overview_metrics.csv"), index=False
    )


def create_marker_geomeans(df: pd.DataFrame) -> None:
    rows = []
    weights = df[WEIGHT_COL]
    groupings = [
        ("amalgam_group", "Amalgam Group"),
        ("mercury_quartile", "Mercury Quartile"),
    ]
    for group_col, label in groupings:
        if group_col not in df.columns:
            continue
        for group_name, grp in df.groupby(group_col, dropna=False):
            group_weights = grp[WEIGHT_COL]
            for marker in MARKERS:
                val = weighted_geometric_mean(grp[marker], group_weights)
                rows.append(
                    {
                        "Grouping": label,
                        "Group": "Missing" if pd.isna(group_name) else group_name,
                        "Marker": marker,
                        "Weighted_geomean": val,
                    }
                )
    geodf = pd.DataFrame(rows)
    geodf.to_csv(os.path.join(OUTPUT_DIR, "marker_weighted_geomeans.csv"), index=False)


def mercury_vs_amalgam_regression(df: pd.DataFrame) -> None:
    """Model log-mercury as the dependent variable to quantify the dose response."""
    result = run_wls(df, "log_mercury", include_mercury=False, min_n=300, log_outcome=False)
    if not result:
        pd.DataFrame(columns=["Beta", "CI_Low", "CI_High", "p_value", "N", "Weighted_N"]).to_csv(
            os.path.join(OUTPUT_DIR, "mercury_on_amalgam_regression.csv"), index=False
        )
        return
    res, n, w = result
    beta, lo, hi, p = extract_param(res, "amalgam_surfaces")
    out = pd.DataFrame(
        [
            {
                "Beta": beta,
                "CI_Low": lo,
                "CI_High": hi,
                "p_value": p,
                "N": n,
                "Weighted_N": w,
                "Adj_R2": res.rsquared_adj,
            }
        ]
    )
    out.to_csv(os.path.join(OUTPUT_DIR, "mercury_on_amalgam_regression.csv"), index=False)


def build_model_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for marker in MARKERS:
        model1 = run_wls(df, marker, include_mercury=False, min_n=300)
        if model1:
            res, n, w = model1
            beta, lo, hi, p = extract_param(res, "amalgam_surfaces")
            rows.append(
                {
                    "Marker": marker,
                    "Model": "Model 1 (No mercury)",
                    "Beta_amalgam": beta,
                    "CI_low_amalgam": lo,
                    "CI_high_amalgam": hi,
                    "p_amalgam": p,
                    "Beta_mercury": np.nan,
                    "CI_low_mercury": np.nan,
                    "CI_high_mercury": np.nan,
                    "p_mercury": np.nan,
                    "N": n,
                    "Weighted_N": w,
                    "Adj_R2": res.rsquared_adj,
                }
            )
        model2 = run_wls(df, marker, include_mercury=True, min_n=300)
        if model2:
            res, n, w = model2
            beta_a, lo_a, hi_a, p_a = extract_param(res, "amalgam_surfaces")
            beta_m, lo_m, hi_m, p_m = extract_param(res, "log_mercury")
            rows.append(
                {
                    "Marker": marker,
                    "Model": "Model 2 (+ log mercury)",
                    "Beta_amalgam": beta_a,
                    "CI_low_amalgam": lo_a,
                    "CI_high_amalgam": hi_a,
                    "p_amalgam": p_a,
                    "Beta_mercury": beta_m,
                    "CI_low_mercury": lo_m,
                    "CI_high_mercury": hi_m,
                    "p_mercury": p_m,
                    "N": n,
                    "Weighted_N": w,
                    "Adj_R2": res.rsquared_adj,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        os.path.join(OUTPUT_DIR, "table2_regression_amalgam_mercury_adjusted.csv"), index=False
    )
    return comparison


def cycle_specific_crp(df: pd.DataFrame) -> None:
    rows = []
    for cycle, grp in df.groupby("Cycle"):
        result = run_wls(grp, "CRP", include_mercury=True, min_n=120)
        if not result:
            continue
        res, n, w = result
        beta_a, lo_a, hi_a, _ = extract_param(res, "amalgam_surfaces")
        beta_m, lo_m, hi_m, _ = extract_param(res, "log_mercury")
        rows.append(
            {
                "Cycle": cycle,
                "CycleMidpoint": grp["CycleMidpoint"].dropna().iloc[0]
                if grp["CycleMidpoint"].notna().any()
                else np.nan,
                "Beta_amalgam": beta_a,
                "CI_low_amalgam": lo_a,
                "CI_high_amalgam": hi_a,
                "Beta_mercury": beta_m,
                "CI_low_mercury": lo_m,
                "CI_high_mercury": hi_m,
                "N": n,
                "Weighted_N": w,
            }
        )
    if not rows:
        print("[WARN]  Skipping cycle-specific CRP (insufficient cycle data).")
        cols = [
            "Cycle",
            "CycleMidpoint",
            "Beta_amalgam",
            "CI_low_amalgam",
            "CI_high_amalgam",
            "Beta_mercury",
            "CI_low_mercury",
            "CI_high_mercury",
            "N",
            "Weighted_N",
        ]
        pd.DataFrame(columns=cols).to_csv(
            os.path.join(OUTPUT_DIR, "regression_cycle_crp.csv"), index=False
        )
        return
    cycle_df = pd.DataFrame(rows).sort_values("Cycle")
    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "regression_cycle_crp.csv"), index=False)


def stratified_crp_heatmaps(df: pd.DataFrame) -> None:
    rows = []
    grouped = df.groupby(["Gender", "Race"])
    for (sex, race), grp in grouped:
        result = run_wls(grp, "CRP", include_mercury=True, min_n=80)
        if not result:
            continue
        res, n, w = result
        beta_a, lo_a, hi_a, _ = extract_param(res, "amalgam_surfaces")
        beta_m, lo_m, hi_m, _ = extract_param(res, "log_mercury")
        rows.append(
            {
                "Sex": sex,
                "Race": race,
                "Beta_amalgam": beta_a,
                "CI_low_amalgam": lo_a,
                "CI_high_amalgam": hi_a,
                "Beta_mercury": beta_m,
                "CI_low_mercury": lo_m,
                "CI_high_mercury": hi_m,
                "N": n,
                "Weighted_N": w,
            }
        )
    heat = pd.DataFrame(rows)
    heat.to_csv(os.path.join(OUTPUT_DIR, "stratified_crp_heatmap.csv"), index=False)


def mediation_analysis(df: pd.DataFrame) -> None:
    cols = [
        "amalgam_surfaces",
        "log_mercury",
        "CRP",
        "Age",
        "BMXBMI",
        "PIR",
        "DiabetesFlag",
        "EverSmoker",
        "CurrentSmoker",
        "AlcoholUse",
        "Gender",
        "Race",
        WEIGHT_COL,
    ]
    subset = df.dropna(subset=cols).copy()
    if len(subset) < 300:
        pd.DataFrame(columns=["Indirect", "Direct", "Total", "Proportion_mediated"]).to_csv(
            os.path.join(OUTPUT_DIR, "mediation_results_crp.csv"), index=False
        )
        return
    subset["log_CRP"] = safe_log(subset["CRP"])
    subset = subset.dropna(subset=["log_CRP"])
    mediator_model = run_wls(
        subset, "log_mercury", include_mercury=False, min_n=300, log_outcome=False
    )
    outcome_model = run_wls(subset, "CRP", include_mercury=True, min_n=300)
    if not mediator_model or not outcome_model:
        pd.DataFrame(columns=["Indirect", "Direct", "Total", "Proportion_mediated"]).to_csv(
            os.path.join(OUTPUT_DIR, "mediation_results_crp.csv"), index=False
        )
        return
    med_res = mediator_model[0]
    out_res = outcome_model[0]
    a = med_res.params.get("amalgam_surfaces", np.nan)
    b = out_res.params.get("log_mercury", np.nan)
    c_prime = out_res.params.get("amalgam_surfaces", np.nan)
    indirect = a * b if np.all(np.isfinite([a, b])) else np.nan
    total = indirect + c_prime if np.all(np.isfinite([indirect, c_prime])) else np.nan
    if np.isfinite(total) and total != 0:
        proportion = indirect / total
    else:
        proportion = np.nan
    med_df = pd.DataFrame(
        [
            {
                "a_path_amalgam_to_mercury": a,
                "b_path_mercury_to_CRP": b,
                "Direct_effect": c_prime,
                "Indirect_effect": indirect,
                "Total_effect": total,
                "Proportion_mediated": proportion,
                "N": len(subset),
                "Weighted_N": weighted_total(subset[WEIGHT_COL]),
            }
        ]
    )
    med_df.to_csv(os.path.join(OUTPUT_DIR, "mediation_results_crp.csv"), index=False)


def interaction_analysis(df: pd.DataFrame) -> None:
    interactions = {
        "Hg_x_CurrentSmoker": ("log_mercury", "CurrentSmoker"),
        "Hg_x_AlcoholUse": ("log_mercury", "AlcoholUse"),
    }
    rows = []
    for name, term in interactions.items():
        result = run_wls(df, "CRP", include_mercury=True, interactions={name: term}, min_n=300)
        if not result:
            continue
        res, n, w = result
        base_mercury, base_lo, base_hi, _ = extract_param(res, "log_mercury")
        interaction_beta, int_lo, int_hi, p_int = extract_param(res, name)
        cov = res.cov_params()
        exposed_beta = np.nan
        exposed_lo = np.nan
        exposed_hi = np.nan
        if np.isfinite(base_mercury) and np.isfinite(interaction_beta):
            exposed_beta = base_mercury + interaction_beta
            if (
                isinstance(cov, pd.DataFrame)
                and "log_mercury" in cov.index
                and name in cov.index
                and "log_mercury" in cov.columns
                and name in cov.columns
            ):
                var = (
                    cov.loc["log_mercury", "log_mercury"]
                    + cov.loc[name, name]
                    + 2 * cov.loc["log_mercury", name]
                )
                if var >= 0:
                    se = np.sqrt(var)
                    exposed_lo = exposed_beta - 1.96 * se
                    exposed_hi = exposed_beta + 1.96 * se
        rows.append(
            {
                "Interaction": name,
                "Base_mercury_beta": base_mercury,
                "Base_ci_low": base_lo,
                "Base_ci_high": base_hi,
                "Interaction_beta": interaction_beta,
                "Interaction_ci_low": int_lo,
                "Interaction_ci_high": int_hi,
                "Interaction_p": p_int,
                "Beta_in_exposed_group": exposed_beta,
                "Exposed_ci_low": exposed_lo,
                "Exposed_ci_high": exposed_hi,
                "N": n,
                "Weighted_N": w,
            }
        )
    columns = [
        "Interaction",
        "Base_mercury_beta",
        "Interaction_beta",
        "Interaction_p",
        "Beta_in_exposed_group",
        "N",
        "Weighted_N",
    ]
    if not rows:
        print("[WARN]  Skipping interaction analysis (insufficient data).")
        pd.DataFrame(columns=columns).to_csv(
            os.path.join(OUTPUT_DIR, "behavior_interaction_results.csv"), index=False
        )
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        os.path.join(OUTPUT_DIR, "behavior_interaction_results.csv"), index=False
    )


def legacy_marker_exports(df: pd.DataFrame) -> None:
    """Preserve historical outputs used by older figure scripts."""
    for marker in MARKERS:
        cycle_rows = []
        for cycle, grp in df.groupby("Cycle"):
            result = run_wls(grp, marker, include_mercury=False, min_n=80)
            if not result:
                continue
            res, _, _ = result
            beta, lo, hi, _ = extract_param(res, "amalgam_surfaces")
            cycle_rows.append(
                {"Marker": marker, "Cycle": cycle, "Beta": beta, "CI_Low": lo, "CI_High": hi}
            )
        pd.DataFrame(cycle_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"regression_results_by_cycle_{marker}.csv"), index=False
        )
        strat_rows = []
        for (sex, race), grp in df.groupby(["Gender", "Race"]):
            result = run_wls(grp, marker, include_mercury=False, min_n=80)
            if not result:
                continue
            res, _, _ = result
            beta, lo, hi, _ = extract_param(res, "amalgam_surfaces")
            strat_rows.append(
                {"Marker": marker, "Sex": sex, "Race": race, "Beta": beta, "CI_Low": lo, "CI_High": hi}
            )
        pd.DataFrame(strat_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"stratified_results_{marker}.csv"), index=False
        )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_path = os.path.join(OUTPUT_DIR, "nhanes_merged_multimarker.csv")
    if not os.path.exists(merged_path):
        print(
            "[WARN]  output_data/nhanes_merged_multimarker.csv not found. Run step 2 first.",
            file=sys.stderr,
        )
        return
    df = pd.read_csv(merged_path)
    if df.empty:
        print("[WARN]  nhanes_merged_multimarker.csv is empty; skipping regressions.")
        return
    df = ensure_required_columns(df)
    df = df.dropna(subset=[WEIGHT_COL, "amalgam_surfaces", "Age"], how="any")

    create_table1(df)
    create_overview_metrics(df)
    create_marker_geomeans(df)
    build_model_comparisons(df)
    mercury_vs_amalgam_regression(df)
    cycle_specific_crp(df)
    stratified_crp_heatmaps(df)
    mediation_analysis(df)
    interaction_analysis(df)
    legacy_marker_exports(df)
    print("[OK] Saved regression tables, mediation outputs, and interaction summaries.")


if __name__ == "__main__":
    main()
