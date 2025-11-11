# 3_regression_multimarker.py
import pandas as pd, numpy as np, statsmodels.api as sm, os

os.makedirs("output_data", exist_ok=True)
df = pd.read_csv("output_data/nhanes_merged_multimarker.csv")

# Outcomes
MARKERS = ["CRP","NLR","MLR","PLR","SII"]

# Prepare covariates
def prep_df(d):
    d = d.copy()
    d = d.dropna(subset=["WTMEC2YR","amalgam_surfaces","Age","RIAGENDR","RIDRETH1"], how="any")
    # Build model matrix
    X = pd.DataFrame({
        "amalgam_surfaces": d["amalgam_surfaces"],
        "Age": d["Age"],
        "BMI": d.get("BMXBMI", np.nan),
        "PIR": d.get("INDFMPIR", d.get("PIR", np.nan)),
        "Diabetes": (d.get("DIQ010", np.nan)==1).astype(float),
        "EverSmoker": (d.get("SMQ020", np.nan)==1).astype(float),
        "Mercury": d.get("LBXTHG", np.nan)
    })
    # Categorical controls via dummies
    sex = pd.get_dummies(d["Gender"], prefix="Sex", drop_first=True)
    race = pd.get_dummies(d["Race"], prefix="Race", drop_first=True)
    X = pd.concat([X, sex, race], axis=1)
    # Ensure all predictors are numeric (booleans become 0/1) before fitting statsmodels
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = sm.add_constant(X, has_constant="add")
    w = d["WTMEC2YR"]
    return X, w

def wls_fit(y, X, w):
    # Drop rows with missing y
    mask = ~y.isna()
    y2, X2, w2 = y[mask], X.loc[mask], w[mask]
    if len(y2) < 50:  # avoid tiny groups
        return np.nan, np.nan, np.nan
    model = sm.WLS(y2, X2, weights=w2)
    res = model.fit()
    if "amalgam_surfaces" not in res.params:
        return np.nan, np.nan, np.nan
    beta = res.params["amalgam_surfaces"]
    se = res.bse["amalgam_surfaces"]
    lo, hi = beta - 1.96*se, beta + 1.96*se
    return beta, lo, hi

def log_transform(s):
    return np.log1p(s)

# Per-cycle regression for each marker
for marker in MARKERS:
    rows = []
    for cycle, sub in df.groupby("Cycle"):
        X, w = prep_df(sub)
        y = log_transform(sub[marker]) if marker in sub.columns else pd.Series(np.nan, index=sub.index)
        beta, lo, hi = wls_fit(y, X, w)
        rows.append({"Marker": marker, "Cycle": cycle, "Beta": beta, "CI_Low": lo, "CI_High": hi})
    if rows:
        out = pd.DataFrame(rows)
        if "Cycle" in out.columns:
            out = out.sort_values("Cycle")
    else:
        out = pd.DataFrame(columns=["Marker","Cycle","Beta","CI_Low","CI_High"])
    out.to_csv(f"output_data/regression_results_by_cycle_{marker}.csv", index=False)
    print(f"✅ Saved output_data/regression_results_by_cycle_{marker}.csv")

# Stratified (Sex x Race) within pooled data for each marker
for marker in MARKERS:
    subrows = []
    for (sex, race), grp in df.groupby(["Gender","Race"]):
        X, w = prep_df(grp)
        y = log_transform(grp[marker]) if marker in grp.columns else pd.Series(np.nan, index=grp.index)
        beta, lo, hi = wls_fit(y, X, w)
        subrows.append({"Marker": marker, "Sex": sex, "Race": race, "Beta": beta})
    strat_df = pd.DataFrame(subrows) if subrows else pd.DataFrame(columns=["Marker","Sex","Race","Beta"])
    strat_df.to_csv(f"output_data/stratified_results_{marker}.csv", index=False)
    print(f"✅ Saved output_data/stratified_results_{marker}.csv")
