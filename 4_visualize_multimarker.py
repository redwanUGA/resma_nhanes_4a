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


def safe_log_series(series):
    values = pd.to_numeric(series, errors="coerce")
    positive = values.where(values > 0)
    return np.log(positive)


def ensure_category_order(df, col, order):
    if col in df.columns:
        df = df.copy()
        df.loc[:, col] = pd.Categorical(df[col], categories=order, ordered=True)
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


def plot_marker_ridgelines(df):
    subset = df.dropna(subset=["amalgam_group"])
    subset = ensure_category_order(subset, "amalgam_group", AMALGAM_ORDER)
    subset = subset.copy()
    subset[WEIGHT_COL] = pd.to_numeric(subset[WEIGHT_COL], errors="coerce")
    subset = subset.dropna(subset=[WEIGHT_COL])
    if subset.empty:
        print("[WARN]  Skipping Figure 3B (no amalgam group data).")
        return
    fig, axes = plt.subplots(len(MARKERS), 1, figsize=(10, 14), sharex=False)
    palette = sns.color_palette("viridis", len(AMALGAM_ORDER))
    legend_handles = None
    for idx, marker in enumerate(MARKERS):
        ax = axes[idx]
        data = subset[[marker, "amalgam_group", WEIGHT_COL]].copy()
        data["log_marker"] = safe_log_series(data[marker])
        data = data.dropna(subset=["log_marker"])
        if data.empty:
            ax.text(0.5, 0.5, f"No positive values for {marker}", ha="center", va="center")
            ax.set_axis_off()
            continue
        plot = sns.kdeplot(
            data=data,
            x="log_marker",
            hue="amalgam_group",
            hue_order=AMALGAM_ORDER,
            weights=data[WEIGHT_COL],
            multiple="stack",
            alpha=0.7,
            linewidth=1.0,
            fill=True,
            ax=ax,
            palette=palette,
            common_norm=False,
        )
        legend = plot.get_legend()
        if legend and legend_handles is None:
            legend_handles = legend.legend_handles
        if legend:
            legend.remove()
        medians = []
        for order_idx, group in enumerate(AMALGAM_ORDER):
            grp = data[data["amalgam_group"] == group]
            if grp.empty:
                medians.append(np.nan)
                continue
            weighted = weighted_quantile(np.exp(grp["log_marker"]), grp[WEIGHT_COL], 0.5)
            medians.append(np.log(weighted) if np.isfinite(weighted) and weighted > 0 else np.nan)
        for median in medians:
            if np.isfinite(median):
                ax.axvline(median, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_ylabel(marker)
        if idx == len(MARKERS) - 1:
            ax.set_xlabel("log(marker value)")
        else:
            ax.set_xlabel("")
        ax.set_title(f"{marker} density by amalgam burden", loc="left", fontsize=11)
    if legend_handles:
        labels = [label for label in AMALGAM_ORDER]
        fig.legend(
            handles=legend_handles,
            labels=labels,
            loc="upper right",
            title="Amalgam group",
        )
    fig.suptitle(
        "Figure 3B. Weighted ridgeline densities of inflammatory markers by amalgam exposure",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig3B_MarkerRidgeline.png"), dpi=300)
    plt.close(fig)


def plot_heatmaps():
    path = os.path.join(DATA_DIR, "stratified_crp_heatmap.csv")
    if not os.path.exists(path):
        print("[WARN]  Skipping Figure 4 (stratified heatmap data missing).")
        return
    try:
        heat = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print("[WARN]  Skipping Figure 4 (stratified heatmap file empty).")
        return
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


def plot_geomean_dumbbell():
    path = os.path.join(DATA_DIR, "marker_weighted_geomeans.csv")
    if not os.path.exists(path):
        print("[WARN]  Skipping Figure 4B (geometric means file missing).")
        return
    geodf = pd.read_csv(path)
    if geodf.empty:
        print("[WARN]  Skipping Figure 4B (geometric means empty).")
        return
    exposures = [
        (
            "Amalgam Group",
            "Amalgam group",
            ("Low (1-5)", "High (>10)"),
            "Fig4B_GeomeanDumbbell.png",
            "Figure 4B. Weighted geometric means: low vs. high amalgam burden",
        ),
        (
            "Mercury Quartile",
            "Mercury quartile",
            ("Q1 (lowest)", "Q4 (highest)"),
            "Fig4C_GeomeanDumbbell_Mercury.png",
            "Figure 4C. Weighted geometric means: lowest vs. highest mercury quartile",
        ),
    ]
    for grouping, label, extremes, filename, title in exposures:
        subset = geodf[geodf["Grouping"] == grouping]
        if subset.empty:
            print(f"[WARN]  Skipping {title} (no rows for {grouping}).")
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        y_positions = np.arange(len(MARKERS))
        plotted = False
        for idx, marker in enumerate(MARKERS):
            rows = subset[subset["Marker"] == marker]
            low_val = (
                rows.loc[rows["Group"] == extremes[0], "Weighted_geomean"].astype(float).dropna()
            )
            high_val = (
                rows.loc[rows["Group"] == extremes[1], "Weighted_geomean"].astype(float).dropna()
            )
            if low_val.empty or high_val.empty:
                continue
            low_value = low_val.iloc[0]
            high_value = high_val.iloc[0]
            if not np.isfinite(low_value) or not np.isfinite(high_value):
                continue
            plotted = True
            xs = [low_value, high_value]
            ax.plot(xs, [y_positions[idx]] * 2, color="#555555", linewidth=2)
            ax.scatter(
                [low_value],
                [y_positions[idx]],
                color="#1f77b4",
                s=60,
                label="Lower exposure" if idx == 0 else "",
            )
            ax.scatter(
                [high_value],
                [y_positions[idx]],
                color="#d62728",
                s=60,
                label="Higher exposure" if idx == 0 else "",
            )
            if low_value > 0:
                pct_change = (high_value / low_value - 1) * 100
                ax.text(
                    max(xs) * 1.05,
                    y_positions[idx],
                    f"{pct_change:+.0f}%",
                    va="center",
                    fontsize=9,
                    color="#333333",
                )
        if not plotted:
            plt.close(fig)
            print(f"[WARN]  Skipping {title} (insufficient data for extremes).")
            continue
        ax.set_xlabel("Weighted geometric mean (original units, log scale)")
        ax.set_xscale("log")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(MARKERS)
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right", title=label.title())
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=300)
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
        if "Cycle" in trend.columns and trend["Cycle"].notna().any():
            trend["CycleMidpoint"] = trend["Cycle"].str.slice(0, 4).astype(float) + 0.5
        else:
            print("[WARN]  Skipping Figure 5 (no usable cycle midpoint).")
            return
    trend = trend.sort_values("CycleMidpoint")
    betas_amalgam = trend["Beta_amalgam"].astype(float)
    betas_mercury = trend["Beta_mercury"].astype(float)
    if betas_amalgam.notna().sum() == 0 and betas_mercury.notna().sum() == 0:
        print("[WARN]  Skipping Figure 5 (no β estimates to visualize).")
        return
    cycle_mid = trend["CycleMidpoint"].astype(float).to_numpy()
    amalgam_levels = np.linspace(0, 12, 60)
    mercury_levels = np.linspace(-2, 2, 60)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    if betas_amalgam.notna().sum() > 0:
        beta_vals = betas_amalgam.fillna(method="ffill").fillna(method="bfill").fillna(0).to_numpy()
        X_a, Y_a = np.meshgrid(cycle_mid, amalgam_levels)
        Z_a = np.outer(amalgam_levels, beta_vals)
        contour_a = axes[0].contourf(
            X_a,
            Y_a,
            Z_a,
            levels=20,
            cmap="coolwarm",
            extend="both",
        )
        fig.colorbar(contour_a, ax=axes[0], label="Predicted Δ log(CRP)")
        axes[0].set_ylabel("Amalgam surfaces (count)")
        axes[0].set_title("Amalgam burden × cycle")
        axes[0].axhline(6, color="white", linestyle="--", linewidth=1, alpha=0.7)
    else:
        axes[0].text(0.5, 0.5, "No amalgam estimates", ha="center", va="center")
    if betas_mercury.notna().sum() > 0:
        beta_vals_m = betas_mercury.fillna(method="ffill").fillna(method="bfill").fillna(0).to_numpy()
        X_m, Y_m = np.meshgrid(cycle_mid, mercury_levels)
        Z_m = np.outer(mercury_levels, beta_vals_m)
        contour_m = axes[1].contourf(
            X_m,
            Y_m,
            Z_m,
            levels=20,
            cmap="coolwarm",
            extend="both",
        )
        fig.colorbar(contour_m, ax=axes[1], label="Predicted Δ log(CRP)")
        axes[1].set_ylabel("Δ log(Mercury) (arbitrary units)")
        axes[1].set_title("Mercury × cycle")
        axes[1].axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No mercury estimates", ha="center", va="center")
    for ax in axes:
        ax.set_xlabel("Cycle midpoint year")
        if "Cycle" in trend.columns and trend["Cycle"].notna().any():
            labels = trend["Cycle"].astype(str).tolist()
            ax.set_xticks(cycle_mid)
            ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("Figure 5. Time-stratified exposure–response surfaces", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig5_Surface_Response.png"), dpi=300)
    plt.close(fig)


def plot_mediation_diagram():
    med_path = os.path.join(DATA_DIR, "mediation_results_crp.csv")
    table_path = os.path.join(DATA_DIR, "table2_regression_amalgam_mercury_adjusted.csv")
    mercury_reg_path = os.path.join(DATA_DIR, "mercury_on_amalgam_regression.csv")
    if not os.path.exists(med_path):
        print("[WARN]  Skipping Figure 6 (mediation results missing).")
        return
    try:
        med = pd.read_csv(med_path)
    except pd.errors.EmptyDataError:
        print("[WARN]  Skipping Figure 6 (mediation results empty).")
        return
    if med.empty:
        print("[WARN]  Skipping Figure 6 (empty mediation results).")
        return
    prop = med["Proportion_mediated"].iloc[0] if "Proportion_mediated" in med else np.nan
    beta_amalgam = beta_mercury = beta_direct = np.nan
    if os.path.exists(table_path):
        try:
            table = pd.read_csv(table_path)
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if not table.empty:
            crp_row = table[(table["Marker"] == "CRP") & (table["Model"].str.contains("Model 2"))]
            if not crp_row.empty:
                beta_direct = crp_row["Beta_amalgam"].iloc[0]
                beta_mercury = crp_row["Beta_mercury"].iloc[0]
    if os.path.exists(mercury_reg_path):
        try:
            reg = pd.read_csv(mercury_reg_path)
        except pd.errors.EmptyDataError:
            reg = pd.DataFrame()
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


def plot_stratified_forest():
    path = os.path.join(DATA_DIR, "stratified_crp_heatmap.csv")
    if not os.path.exists(path):
        print("[WARN]  Skipping Figure 6B (stratified file missing).")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("[WARN]  Skipping Figure 6B (stratified file empty).")
        return
    df["Label"] = df["Sex"].fillna("Unknown") + " · " + df["Race"].fillna("Unknown")

    def make_panel(ax, beta_col, lo_col, hi_col, title, color):
        if beta_col not in df.columns:
            ax.text(0.5, 0.5, "Data missing", ha="center")
            ax.set_axis_off()
            return
        subset = df.dropna(subset=[beta_col])
        if subset.empty:
            ax.text(0.5, 0.5, "Data missing", ha="center")
            ax.set_axis_off()
            return
        subset = subset.sort_values(beta_col)
        y_positions = np.arange(len(subset))
        betas = subset[beta_col].astype(float).to_numpy()
        lo = subset[lo_col].astype(float).to_numpy() if lo_col in subset else np.full_like(betas, np.nan)
        hi = subset[hi_col].astype(float).to_numpy() if hi_col in subset else np.full_like(betas, np.nan)
        lower_err = np.where(np.isfinite(lo), betas - lo, np.nan)
        upper_err = np.where(np.isfinite(hi), hi - betas, np.nan)
        errors = np.vstack([lower_err, upper_err])
        ax.errorbar(
            betas,
            y_positions,
            xerr=errors,
            fmt="o",
            color=color,
            ecolor=color,
            linewidth=1.5,
            capsize=4,
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(subset["Label"])
        ax.set_title(title)
        ax.set_xlabel("β coefficient")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        xmax = ax.get_xlim()[1]
        for y, n, w in zip(y_positions, subset["N"], subset["Weighted_N"]):
            label = f"N={int(n)}"
            if pd.notna(w):
                label += f" / W={w/1e6:.1f}M"
            ax.text(
                xmax * 0.98,
                y,
                label,
                ha="right",
                va="center",
                fontsize=8,
                color="#333333",
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    make_panel(
        axes[0],
        "Beta_amalgam",
        "CI_low_amalgam",
        "CI_high_amalgam",
        "Amalgam → log(CRP)",
        "#1f77b4",
    )
    make_panel(
        axes[1],
        "Beta_mercury",
        "CI_low_mercury",
        "CI_high_mercury",
        "log(Mercury) → log(CRP)",
        "#d62728",
    )
    fig.suptitle("Figure 6B. Stratified effect forest plots", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Fig6B_Stratified_Forest.png"), dpi=300)
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
        base_lo = row["Base_ci_low"].iloc[0] if "Base_ci_low" in row else np.nan
        base_hi = row["Base_ci_high"].iloc[0] if "Base_ci_high" in row else np.nan
        exp_lo = row["Exposed_ci_low"].iloc[0] if "Exposed_ci_low" in row else np.nan
        exp_hi = row["Exposed_ci_high"].iloc[0] if "Exposed_ci_high" in row else np.nan
        xs = np.array([0, 1])
        ys = np.array([base, exposed], dtype=float)
        ax.plot(xs, ys, color="#9467bd", linewidth=2, marker="o")
        lower_err = [
            ys[0] - base_lo if np.isfinite(base_lo) else np.nan,
            ys[1] - exp_lo if np.isfinite(exp_lo) else np.nan,
        ]
        upper_err = [
            base_hi - ys[0] if np.isfinite(base_hi) else np.nan,
            exp_hi - ys[1] if np.isfinite(exp_hi) else np.nan,
        ]
        errors = np.vstack([lower_err, upper_err])
        ax.errorbar(
            xs,
            ys,
            yerr=errors,
            fmt="none",
            ecolor="#9467bd",
            elinewidth=1.5,
            capsize=4,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"No {labels[0]}", labels[0]])
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("β for log(Mercury) → log(CRP)")
        ymin = np.nanmin(ys[np.isfinite(ys)]) if np.isfinite(ys).any() else -0.1
        ymax = np.nanmax(ys[np.isfinite(ys)]) if np.isfinite(ys).any() else 0.1
        span = ymax - ymin
        buffer = max(span * 0.2, 0.01)
        ax.set_ylim(ymin - buffer, ymax + buffer)
        p_val = row["Interaction_p"].iloc[0] if "Interaction_p" in row else np.nan
        label = f"p_int = {p_val:.3f}" if np.isfinite(p_val) else "p_int unavailable"
        ax.text(0.5, 0.9, label, ha="center", transform=ax.transAxes, fontsize=10)
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
    merged = pd.read_csv(MERGED_PATH, low_memory=False)
    if merged.empty:
        print("[WARN]  nhanes_merged_multimarker.csv is empty; skipping figures.")
        sys.exit(0)
    plot_mercury_distribution(merged)
    plot_mercury_regression(merged)
    plot_marker_distributions(merged)
    plot_marker_ridgelines(merged)
    plot_heatmaps()
    plot_geomean_dumbbell()
    plot_temporal_trend()
    plot_mediation_diagram()
    plot_stratified_forest()
    plot_behavior_figures(merged)
    print("[OK] Saved updated figures to output_figures/")


if __name__ == "__main__":
    main()
