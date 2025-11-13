# 2_merge_preprocess_multimarker.py
import os, sys, numpy as np, pandas as pd, pyreadstat

os.makedirs("output_data", exist_ok=True)
DATA_DIR = "nhanes_data"

CYCLES = {
    "1999-2000": {"DEMO":"DEMO.xpt","OHX":"OHXDENT.xpt","CRP":"LAB11.xpt","CBC":"LAB25.xpt",
                  "SMQ":"SMQ.xpt","ALQ":"ALQ.xpt","BMX":"BMX.xpt","DIQ":"DIQ.xpt","HGM":"LAB06HM.xpt"},
    "2001-2002": {"DEMO":"DEMO_B.xpt","OHX":"OHXDEN_B.xpt","CRP":"L11_B.xpt","CBC":"LAB25_B.xpt",
                  "SMQ":"SMQ_B.xpt","ALQ":"ALQ_B.xpt","BMX":"BMX_B.xpt","DIQ":"DIQ_B.xpt","HGM":"L06_2_B.xpt"},
    "2003-2004": {"DEMO":"DEMO_C.xpt","OHX":"OHXDEN_C.xpt","CRP":"L11_C.xpt","CBC":"LAB25_C.xpt",
                  "SMQ":"SMQ_C.xpt","ALQ":"ALQ_C.xpt","BMX":"BMX_C.xpt","DIQ":"DIQ_C.xpt","HGM":"L06BMT_C.xpt"},
    "2005-2006": {"DEMO":"DEMO_D.xpt","OHX":"OHXDEN_D.xpt","CRP":"CRP_D.xpt","CBC":"CBC_D.xpt",
                  "SMQ":"SMQ_D.xpt","ALQ":"ALQ_D.xpt","BMX":"BMX_D.xpt","DIQ":"DIQ_D.xpt","HGM":"PBCD_D.xpt"},
    "2007-2008": {"DEMO":"DEMO_E.xpt","OHX":"OHXDEN_E.xpt","CRP":"CRP_E.xpt","CBC":"CBC_E.xpt",
                  "SMQ":"SMQ_E.xpt","ALQ":"ALQ_E.xpt","BMX":"BMX_E.xpt","DIQ":"DIQ_E.xpt","HGM":"PBCD_E.xpt"},
    "2009-2010": {"DEMO":"DEMO_F.xpt","OHX":"OHXDEN_F.xpt","CRP":"CRP_F.xpt","CBC":"CBC_F.xpt",
                  "SMQ":"SMQ_F.xpt","ALQ":"ALQ_F.xpt","BMX":"BMX_F.xpt","DIQ":"DIQ_F.xpt","HGM":"PBCD_F.xpt"},
    "2011-2012": {"DEMO":"DEMO_G.xpt","OHX":"OHXDEN_G.xpt","CRP":"CRP_G.xpt","CBC":"CBC_G.xpt",
                  "SMQ":"SMQ_G.xpt","ALQ":"ALQ_G.xpt","BMX":"BMX_G.xpt","DIQ":"DIQ_G.xpt","HGM":"PBCD_G.xpt"},
    "2013-2014": {"DEMO":"DEMO_H.xpt","OHX":"OHXDEN_H.xpt","CRP":"CRP_H.xpt","CBC":"CBC_H.xpt",
                  "SMQ":"SMQ_H.xpt","ALQ":"ALQ_H.xpt","BMX":"BMX_H.xpt","DIQ":"DIQ_H.xpt","HGM":"PBCD_H.xpt"},
    "2015-2016": {"DEMO":"DEMO_I.xpt","OHX":"OHXDEN_I.xpt","CRP":"CRP_I.xpt","CBC":"CBC_I.xpt",
                  "SMQ":"SMQ_I.xpt","ALQ":"ALQ_I.xpt","BMX":"BMX_I.xpt","DIQ":"DIQ_I.xpt","HGM":"PBCD_I.xpt"},
    "2017-2018": {"DEMO":"DEMO_J.xpt","OHX":"OHXDEN_J.xpt","CRP":"HSCRP_J.xpt","CBC":"CBC_J.xpt",
                  "SMQ":"SMQ_J.xpt","ALQ":"ALQ_J.xpt","BMX":"BMX_J.xpt","DIQ":"DIQ_J.xpt","HGM":"PBCD_J.xpt"},
}

race_map = {1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",4:"Non-Hispanic Black",5:"Other/Multi-Racial"}

def compute_cycle_midpoint(label):
    """Return the midpoint year for plot-ready time series (e.g., '1999-2000' -> 1999.5)."""
    try:
        start, end = label.split("-")
        start, end = int(start), int(end)
        return start + (end - start) / 2.0
    except Exception:
        return np.nan

def weighted_quantile(values, quantiles, sample_weight):
    """Compute weighted quantiles for positive weights."""
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    sample_weight = np.asarray(sample_weight)
    mask = (~np.isnan(values)) & (~np.isnan(sample_weight))
    values, sample_weight = values[mask], sample_weight[mask]
    if values.size == 0:
        return np.array([np.nan] * len(quantiles))
    sorter = np.argsort(values)
    values, sample_weight = values[sorter], sample_weight[sorter]
    cumulative = np.cumsum(sample_weight)
    if cumulative[-1] == 0:
        return np.array([np.nan] * len(quantiles))
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)

def assign_mercury_quartiles(df):
    """Add weighted blood-mercury quartile labels for downstream figures/tables."""
    if "LBXTHG" not in df.columns:
        df["mercury_quartile"] = np.nan
        return df
    mask = df["LBXTHG"].notna() & (df["LBXTHG"] > 0) & df["WTMEC2YR"].notna()
    if mask.sum() < 50:
        df["mercury_quartile"] = np.nan
        return df
    qs = weighted_quantile(df.loc[mask, "LBXTHG"], [0.25, 0.5, 0.75], df.loc[mask, "WTMEC2YR"])
    bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    df["mercury_quartile"] = pd.cut(df["LBXTHG"], bins=bins, labels=labels)
    return df

def required_files_missing(cycle, files):
    must_have = ["DEMO","OHX","CRP","CBC","SMQ","ALQ","BMX","DIQ"]
    missing = []
    for key in must_have:
        fname = files[key]
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(fname)
    return missing

def load_xpt(path, label):
    try:
        df, _ = pyreadstat.read_xport(path)
        return df
    except Exception as err:
        raise RuntimeError(f"Failed to read {label} ({path}): {err}") from err

def count_amalgam(df_ohx):
    cols = [c for c in df_ohx.columns if c.startswith("OHX") and c.endswith(("TC","FS","FT"))]
    if not cols: 
        df_ohx["amalgam_surfaces"]=0
        return df_ohx[["SEQN","amalgam_surfaces"]]
    df_ohx["amalgam_surfaces"] = (df_ohx[cols] == 2).sum(axis=1)
    return df_ohx[["SEQN","amalgam_surfaces"]]

def compute_inflammation(df):
    try:
        df["Neutrophils"] = df["LBXWBCSI"] * df["LBXNEPCT"]/100.0
        df["Lymphocytes"] = df["LBXWBCSI"] * df["LBXLYPCT"]/100.0
        df["Monocytes"]  = df["LBXWBCSI"] * df["LBXMOPCT"]/100.0
        df["Platelets"]  = df["LBXPLTSI"]
        df["NLR"] = df["Neutrophils"]/df["Lymphocytes"]
        df["MLR"] = df["Monocytes"]/df["Lymphocytes"]
        df["PLR"] = df["Platelets"]/df["Lymphocytes"]
        df["SII"] = (df["Neutrophils"]*df["Platelets"])/df["Lymphocytes"]
    except Exception:
        pass
    return df

def detect_crp_col(df):
    for c in df.columns:
        u=c.upper()
        if u.startswith("LBXCRP") or u.startswith("LBXHSCRP"):
            return c
    return None

def detect_mercury_col(df):
    # Prefer total blood mercury (LBXTHG), fallback any *THG*
    candidates=[c for c in df.columns if "THG" in c.upper()]
    return candidates[0] if candidates else None

def select_alcohol_flag(alq_df):
    """Return a yes/no alcohol flag column available for the current cycle."""
    preferred = ["ALQ101","ALQ100","ALQ111","ALQ110"]
    for col in preferred:
        if col in alq_df.columns:
            return col
    return None

frames = []

for cycle, files in CYCLES.items():
    missing = required_files_missing(cycle, files)
    if missing:
        print(f"[WARN] Skipping {cycle}: missing files {', '.join(missing)}")
        continue
    try:
        demo = load_xpt(os.path.join(DATA_DIR, files["DEMO"]), "DEMO")
        ohx  = load_xpt(os.path.join(DATA_DIR, files["OHX"]), "OHX")
        crp  = load_xpt(os.path.join(DATA_DIR, files["CRP"]), "CRP")
        cbc  = load_xpt(os.path.join(DATA_DIR, files["CBC"]), "CBC")
        smq  = load_xpt(os.path.join(DATA_DIR, files["SMQ"]), "SMQ")
        alq  = load_xpt(os.path.join(DATA_DIR, files["ALQ"]), "ALQ")
        bmx  = load_xpt(os.path.join(DATA_DIR, files["BMX"]), "BMX")
        diq  = load_xpt(os.path.join(DATA_DIR, files["DIQ"]), "DIQ")
        try:
            hgm  = load_xpt(os.path.join(DATA_DIR, files["HGM"]), "HGM")
        except Exception as err:
            print(f"[WARN] {cycle}: unable to read mercury file ({err}); continuing without it.")
            hgm = pd.DataFrame()

        ohx2 = count_amalgam(ohx)
        crp_col = detect_crp_col(crp)
        if not crp_col:
            print(f"[WARN] Skipping {cycle}: CRP column not found in {files['CRP']}")
            continue
        df = demo.merge(ohx2, on="SEQN").merge(crp[["SEQN",crp_col]], on="SEQN", how="left")
        df["CRP"] = df[crp_col]
        df = df.merge(cbc, on="SEQN", how="left")
        df = compute_inflammation(df)
        df = df.merge(smq[["SEQN","SMQ020","SMQ040"]], on="SEQN", how="left")
        drink_flag_col = "ALQ_DRINK_FLAG"
        alq_flag = select_alcohol_flag(alq)
        if alq_flag:
            df = df.merge(alq[["SEQN",alq_flag]], on="SEQN", how="left")
            df.rename(columns={alq_flag: drink_flag_col}, inplace=True)
        else:
            df[drink_flag_col] = np.nan
            print(f"[WARN] {cycle}: no ALQ yes/no drinking variable found; Drinker will be NaN.")
        df = df.merge(bmx[["SEQN","BMXBMI"]], on="SEQN", how="left")  # BMI
        if "DIQ010" in diq.columns:
            df = df.merge(diq[["SEQN","DIQ010"]], on="SEQN", how="left")  # Doctor told diabetes
        else:
            df["DIQ010"] = np.nan
        if not hgm.empty:
            merc_col = detect_mercury_col(hgm)
            if merc_col:
                df = df.merge(hgm[["SEQN", merc_col]], on="SEQN", how="left")
                df.rename(columns={merc_col:"LBXTHG"}, inplace=True)

        # Derived labels
        df["Gender"] = df["RIAGENDR"].replace({1:"Male",2:"Female"})
        df["Race"] = df["RIDRETH1"].replace(race_map)
        df["Age"] = df["RIDAGEYR"]
        df["PIR"] = df.get("INDFMPIR", np.nan)
        df["Smoker"] = df["SMQ020"].replace({1:"Ever",2:"Never"})
        df["CurrentSmoker"] = df["SMQ040"].replace({1:"Everyday",2:"Somedays",3:"Not at all"})
        if drink_flag_col in df.columns:
            df["Drinker"] = np.where(
                df[drink_flag_col] == 1,
                "Yes",
                np.where(df[drink_flag_col] == 2, "No", None),
            )
        else:
            df["Drinker"] = np.nan
        df["amalgam_group"] = pd.cut(df["amalgam_surfaces"], [-1,0,5,10,np.inf],
                                     labels=["None","Low (1-5)","Medium (6-10)","High (>10)"])
        df["Cycle"] = cycle
        frames.append(df)
        print(f"[OK] Merged {cycle}, n={len(df)}")
    except Exception as e:
        print(f"[ERROR] Skipped {cycle}: {e}")

if not frames:
    print("[WARN] No NHANES cycles were processed. Ensure nhanes_data/ contains the downloaded .xpt files from CDC.")
    empty_cols = ["SEQN","Cycle","amalgam_surfaces","CRP","NLR","MLR","PLR","SII",
                  "Gender","Race","Age","PIR","WTMEC2YR"]
    pd.DataFrame(columns=empty_cols).to_csv("output_data/nhanes_merged_multimarker.csv", index=False)
    sys.exit(0)

full = pd.concat(frames, ignore_index=True)
if "LBXTHG" not in full.columns:
    full["LBXTHG"] = np.nan
full["log_mercury"] = np.where(full["LBXTHG"] > 0, np.log(full["LBXTHG"]), np.nan)
full["any_amalgam"] = (full["amalgam_surfaces"] > 0).astype(float)
full["amalgam_high_burden"] = (full["amalgam_surfaces"] >= 6).astype(float)
full["CycleMidpoint"] = full["Cycle"].apply(compute_cycle_midpoint)
full = assign_mercury_quartiles(full)
full.to_csv("output_data/nhanes_merged_multimarker.csv", index=False)
print("[OK] Saved output_data/nhanes_merged_multimarker.csv")
