# 4_visualize_multimarker.py
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_config"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".xdg_cache"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

sns.set(style="whitegrid", font_scale=1.1)

OUTPUT_FIG_DIR = "output_figures"
DATA_DIR = "output_data"
MERGED_PATH = os.path.join(DATA_DIR, "nhanes_merged_multimarker.csv")
WEIGHT_COL = "WTMEC2YR"
MARKERS = ["CRP", "NLR", "MLR", "PLR", "SII"]
AMALGAM_ORDER = ["None", "Low (1-5)", "Medium (6-10)", "High (>10)"]
MERCURY_ORDER = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]


def weighted_quantile(series, weights, quantile):
    series = pd.to_numeric(series, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = series.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    series, weights = series[mask], weights[mask]
    sorter = np.argsort(series)
    series, weights = series.iloc[sorter], weights.iloc[sorter]
    cumulative = weights.cumsum()
    total = cumulative.iloc[-1]
    if total == 0:
        return np.nan
    cumulative /= total
    return float(np.interp(quantile, cumulative, series))


def weighted_percent(df, group_col, indicator_col):
    rows = []
    for group, grp in df.groupby(group_col):
        total = grp[WEIGHT_COL].sum()
        if total == 0:
            continue
        for status_value, label in [(1, "Yes"), (0, "No")]:
            weight = grp.loc[grp[indicator_col] == status_value, WEIGHT_COL].sum()
            rows.append(
                {
                    "Exposure": group,
                    "Status": label,
                    "Weighted_pct": 100 * weight / total,
                }
            )
    return pd.DataFrame(rows)


def ensure_category_order(df, col, order):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


def plot_mercury_distribution(df):
    subset = df.dropna(subset=["amalgam_group", "LBXTHG", WEIGHT_COL])
    if subset.empty:
        print("[WARN]  Skipping Figure 1 (no amalgam/mercury data).")
        return
    subset = ensure_category_order(subset, "amalgam_group", AMALGAM_ORDER)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=subset,
        x="amalgam_group",
        y="LBXTHG",
        cut=0,
        inner="quartile",
        order=AMALGAM_ORDER,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Amalgam Exposure Category")
    ax.set_ylabel("Blood Mercury (µg/L, log scale)")
    ax.set_title("Figure 1. Weighted blood-mercury distribution by amalgam burden")
    medians = []
    for idx, group in enumerate(AMALGAM_ORDER):
        grp = subset[subset["amalgam_group"] == group]
        if grp.empty:
            medians.append(np.nan)
            continue
        medians.append(weighted_quantile(grp["LBXTHG"], grp[WEIGHT_COL], 0.5))
    ax.plot(
        range(len(AMALGAM_ORDER)),
        medians,
        marker="D",
        linestyle="None",
        color="#d62728",
        label="Weighted median",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig1_BloodMercury_Distribution.png"), dpi=300)
    plt.close(fig)


def plot_mercury_regression(df):
    subset = df.dropna(subset=["amalgam_surfaces", "LBXTHG", WEIGHT_COL])
    subset = subset[subset["LBXTHG"] > 0]
    if subset.empty:
        print("[WARN]  Skipping Figure 2 (insufficient mercury values).")
        return
    subset["log_mercury"] = np.log(subset["LBXTHG"])
    sample_n = 8000
    if len(subset) > sample_n:
        subset = subset.sample(
            n=sample_n,
            weights=subset[WEIGHT_COL],
            random_state=42,
            replace=False,
        )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=subset,
        x="amalgam_surfaces",
        y="log_mercury",
        scatter_kws={"alpha": 0.25, "s": 12, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=ax,
    )
    ax.set_xlabel("Number of amalgam surfaces")
    ax.set_ylabel("log(LBXTHG)")
    ax.set_title("Figure 2. log(LBXTHG) vs amalgam surfaces (survey-weighted sample)")
    reg_path = os.path.join(DATA_DIR, "mercury_on_amalgam_regression.csv")
    if os.path.exists(reg_path):
        reg_df = pd.read_csv(reg_path)
        if not reg_df.empty and "Beta" in reg_df.columns:
            beta = reg_df["Beta"].iloc[0]
            ci_low = reg_df["CI_Low"].iloc[0]
            ci_high = reg_df["CI_High"].iloc[0]
            ax.text(
                0.02,
                0.95,
                f"β = {beta:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f})",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig2_Amalgam_vs_Mercury.png"), dpi=300)
    plt.close(fig)


def plot_marker_distributions(df):
    subset = df.dropna(subset=["mercury_quartile"])
    subset = ensure_category_order(subset, "mercury_quartile", MERCURY_ORDER)
    if subset.empty:
        print("[WARN]  Skipping Figure 3 (no mercury quartiles).")
        return
    geopath = os.path.join(DATA_DIR, "marker_weighted_geomeans.csv")
    geodf = pd.read_csv(geopath) if os.path.exists(geopath) else pd.DataFrame()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for idx, marker in enumerate(MARKERS):
        ax = axes[idx]
        sns.boxplot(
            data=subset,
            x="mercury_quartile",
            y=marker,
            order=MERCURY_ORDER,
            ax=ax,
            showfliers=False,
        )
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(f"{marker} (log scale)")
        ax.set_title(marker)
        if not geodf.empty:
            gsub = geodf[
                (geodf["Grouping"] == "Mercury Quartile") & (geodf["Marker"] == marker)
            ]
            if not gsub.empty:
                ordered = []
                for label in MERCURY_ORDER:
                    val = gsub[gsub["Group"] == label]["Weighted_geomean"].values
                    ordered.append(val[0] if len(val) else np.nan)
                ax.plot(range(len(MERCURY_ORDER)), ordered, marker="o", color="#d62728")
    axes[-1].axis("off")
    fig.suptitle(
        "Figure 3. Inflammatory marker distributions across blood-mercury quartiles",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig3_Inflammation_by_MercuryQuartile.png"), dpi=300)
    plt.close(fig)


def plot_heatmaps():
    path = os.path.join(DATA_DIR, "stratified_crp_heatmap.csv")
    if not os.path.exists(path):
        print("[WARN]  Skipping Figure 4 (stratified heatmap data missing).")
        return
    heat = pd.read_csv(path)
    if heat.empty:
        print("[WARN]  Skipping Figure 4 (empty stratified results).")
        return
    piv_amalgam = heat.pivot(index="Sex", columns="Race", values="Beta_amalgam")
    piv_mercury = heat.pivot(index="Sex", columns="Race", values="Beta_mercury")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        piv_amalgam,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        ax=axes[0],
    )
    axes[0].set_title("Amalgam → log(CRP)")
    sns.heatmap(
        piv_mercury,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        ax=axes[1],
    )
    axes[1].set_title("log(Mercury) → log(CRP)")
    fig.suptitle("Figure 4. Adjusted β-coefficients by sex and race", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig4_Heatmap_MercuryAdjusted.png"), dpi=300)
    plt.close(fig)


def plot_temporal_trend():
    path = os.path.join(DATA_DIR, "regression_cycle_crp.csv")
    if not os.path.exists(path):
        print("[WARN]  Skipping Figure 5 (cycle-level results missing).")
        return
    trend = pd.read_csv(path)
    if trend.empty:
        print("[WARN]  Skipping Figure 5 (empty cycle results).")
        return
    if "CycleMidpoint" not in trend.columns or trend["CycleMidpoint"].isna().all():
        trend["CycleMidpoint"] = trend["Cycle"].str.slice(0, 4).astype(float) + 0.5
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        trend["CycleMidpoint"],
        trend["Beta_amalgam"],
        marker="o",
        label="Amalgam → log(CRP)",
        color="#1f77b4",
    )
    ax.fill_between(
        trend["CycleMidpoint"],
        trend["CI_low_amalgam"],
        trend["CI_high_amalgam"],
        color="#1f77b4",
        alpha=0.15,
    )
    ax.plot(
        trend["CycleMidpoint"],
        trend["Beta_mercury"],
        marker="s",
        label="log(Mercury) → log(CRP)",
        color="#d62728",
    )
    ax.fill_between(
        trend["CycleMidpoint"],
        trend["CI_low_mercury"],
        trend["CI_high_mercury"],
        color="#d62728",
        alpha=0.15,
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Survey cycle midpoint year")
    ax.set_ylabel("β coefficient")
    ax.set_title("Figure 5. Temporal trends of β-coefficients (CRP models)")
    valid_ticks = trend["CycleMidpoint"].notna()
    if valid_ticks.any():
        tick_positions = trend.loc[valid_ticks, "CycleMidpoint"].tolist()
        if "Cycle" in trend.columns and trend["Cycle"].notna().any():
            fallback_labels = pd.Series(tick_positions).round().astype(int).astype(str)
            tick_labels = (
                trend.loc[valid_ticks, "Cycle"]
                .reset_index(drop=True)
                .fillna(fallback_labels)
                .astype(str)
                .tolist()
            )
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right")
            ax.set_xlabel("Cycle")
        else:
            ax.set_xticks(tick_positions)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig5_Trend_MultiMarker.png"), dpi=300)
    plt.close(fig)


def plot_mediation_diagram():
    med_path = os.path.join(DATA_DIR, "mediation_results_crp.csv")
    table_path = os.path.join(DATA_DIR, "table2_regression_amalgam_mercury_adjusted.csv")
    mercury_reg_path = os.path.join(DATA_DIR, "mercury_on_amalgam_regression.csv")
    if not os.path.exists(med_path):
        print("[WARN]  Skipping Figure 6 (mediation results missing).")
        return
    med = pd.read_csv(med_path)
    if med.empty:
        print("[WARN]  Skipping Figure 6 (empty mediation results).")
        return
    prop = med["Proportion_mediated"].iloc[0] if "Proportion_mediated" in med else np.nan
    beta_amalgam = beta_mercury = beta_direct = np.nan
    if os.path.exists(table_path):
        table = pd.read_csv(table_path)
        crp_row = table[(table["Marker"] == "CRP") & (table["Model"].str.contains("Model 2"))]
        if not crp_row.empty:
            beta_direct = crp_row["Beta_amalgam"].iloc[0]
            beta_mercury = crp_row["Beta_mercury"].iloc[0]
    if os.path.exists(mercury_reg_path):
        reg = pd.read_csv(mercury_reg_path)
        if not reg.empty and "Beta" in reg.columns:
            beta_amalgam = reg["Beta"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    boxes = {
        "Amalgam\nSurfaces": (0.05, 0.4),
        "Blood\nMercury": (0.4, 0.7),
        "Systemic\nInflammation": (0.75, 0.4),
    }
    box_width, box_height = 0.2, 0.2
    for label, (x, y) in boxes.items():
        rect = patches.FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="#f7f7f7",
        )
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + box_height / 2, label, ha="center", va="center", fontsize=12)
    ax.annotate(
        "",
        xy=(boxes["Blood\nMercury"][0], boxes["Blood\nMercury"][1] + box_height / 2),
        xytext=(boxes["Amalgam\nSurfaces"][0] + box_width, boxes["Amalgam\nSurfaces"][1] + box_height / 2),
        arrowprops=dict(arrowstyle="->", linewidth=2, color="#1f77b4"),
    )
    ax.annotate(
        "",
        xy=(boxes["Systemic\nInflammation"][0], boxes["Systemic\nInflammation"][1] + box_height / 2),
        xytext=(boxes["Blood\nMercury"][0] + box_width, boxes["Blood\nMercury"][1] + box_height / 2),
        arrowprops=dict(arrowstyle="->", linewidth=2, color="#d62728"),
    )
    ax.annotate(
        "",
        xy=(boxes["Systemic\nInflammation"][0], boxes["Systemic\nInflammation"][1] + box_height / 2),
        xytext=(boxes["Amalgam\nSurfaces"][0] + box_width, boxes["Amalgam\nSurfaces"][1] + box_height / 2),
        arrowprops=dict(arrowstyle="->", linewidth=2, linestyle="--", color="#555555"),
    )
    ax.text(
        0.22,
        0.65,
        f"β = {beta_amalgam:.3f}\n(amalgam → mercury)" if np.isfinite(beta_amalgam) else "amalgam → mercury",
        fontsize=10,
    )
    ax.text(
        0.58,
        0.65,
        f"β = {beta_mercury:.3f}\n(mercury → CRP)" if np.isfinite(beta_mercury) else "mercury → CRP",
        fontsize=10,
    )
    ax.text(
        0.48,
        0.35,
        f"Direct β = {beta_direct:.3f}" if np.isfinite(beta_direct) else "Direct amalgam → CRP",
        fontsize=10,
    )
    if np.isfinite(prop):
        ax.text(
            0.35,
            0.15,
            f"≈ {prop * 100:.1f}% of total effect mediated via mercury",
            fontsize=11,
            ha="center",
        )
    ax.set_title("Figure 6. Conceptual mediation pathway", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig6_Mediation_Path.png"), dpi=300)
    plt.close(fig)


def plot_behavior_figures(df):
    df = df.copy()
    df["current_smoker_flag"] = df["SMQ040"].isin([1, 2]).astype(int)
    drink_col = "ALQ_DRINK_FLAG" if "ALQ_DRINK_FLAG" in df.columns else (
        "ALQ101" if "ALQ101" in df.columns else None
    )
    if drink_col:
        df["alcohol_flag"] = (df[drink_col] == 1).astype(int)
    else:
        df["alcohol_flag"] = 0
        print("[WARN]  ALQ drinking variable missing; alcohol behavior panel will be blank.")
    df = ensure_category_order(df, "amalgam_group", AMALGAM_ORDER)
    smoke = weighted_percent(df.dropna(subset=["amalgam_group"]), "amalgam_group", "current_smoker_flag")
    drink = weighted_percent(df.dropna(subset=["amalgam_group"]), "amalgam_group", "alcohol_flag")
    interact_path = os.path.join(DATA_DIR, "behavior_interaction_results.csv")
    interaction = pd.DataFrame()
    if os.path.exists(interact_path):
        try:
            interaction = pd.read_csv(interact_path)
        except pd.errors.EmptyDataError:
            interaction = pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if not smoke.empty:
        sns.barplot(
            data=smoke,
            x="Exposure",
            y="Weighted_pct",
            hue="Status",
            order=AMALGAM_ORDER,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Smoking status by amalgam exposure")
        axes[0, 0].set_ylabel("Weighted percent")
    else:
        axes[0, 0].text(0.5, 0.5, "No smoking data", ha="center")
    if not drink.empty:
        sns.barplot(
            data=drink,
            x="Exposure",
            y="Weighted_pct",
            hue="Status",
            order=AMALGAM_ORDER,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Alcohol use by amalgam exposure")
        axes[0, 1].set_ylabel("")
    else:
        axes[0, 1].text(0.5, 0.5, "No alcohol data", ha="center")

    def interaction_plot(ax, name, labels):
        if interaction.empty:
            ax.text(0.5, 0.5, "Interaction data missing", ha="center")
            return
        row = interaction[interaction["Interaction"] == name]
        if row.empty:
            ax.text(0.5, 0.5, "Interaction data missing", ha="center")
            return
        base = row["Base_mercury_beta"].iloc[0]
        exposed = row["Beta_in_exposed_group"].iloc[0]
        sns.lineplot(
            x=["No " + labels[0], labels[0]],
            y=[base, exposed],
            marker="o",
            ax=ax,
        )
        ax.set_ylabel("β for log(Mercury) → log(CRP)")
        if np.isfinite(base) and np.isfinite(exposed):
            span = max(base, exposed) - min(base, exposed)
            buffer = max(span * 0.2, 0.01)
            ax.set_ylim(min(base, exposed) - buffer, max(base, exposed) + buffer)
        ax.set_title(f"Interaction: Mercury × {labels[0]}")

    interaction_plot(axes[1, 0], "Hg_x_CurrentSmoker", ("Smoking",))
    interaction_plot(axes[1, 1], "Hg_x_AlcoholUse", ("Alcohol Use",))
    fig.suptitle(
        "Figure 7. Behavioral distributions and mercury-interaction effects", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig7_Smoking_Drinking.png"), dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    if not os.path.exists(MERGED_PATH):
        print("[WARN]  nhanes_merged_multimarker.csv not found; run preprocessing first.")
        sys.exit(0)
    merged = pd.read_csv(MERGED_PATH)
    if merged.empty:
        print("[WARN]  nhanes_merged_multimarker.csv is empty; skipping figures.")
        sys.exit(0)
    plot_mercury_distribution(merged)
    plot_mercury_regression(merged)
    plot_marker_distributions(merged)
    plot_heatmaps()
    plot_temporal_trend()
    plot_mediation_diagram()
    plot_behavior_figures(merged)
    print("[OK] Saved updated figures to output_figures/")


if __name__ == "__main__":
    main()
