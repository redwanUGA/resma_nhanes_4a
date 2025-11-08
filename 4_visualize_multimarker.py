# 4_visualize_multimarker.py
import os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
sns.set(style="whitegrid", font_scale=1.1)
os.makedirs("output_figures", exist_ok=True)

merged = pd.read_csv("output_data/nhanes_merged_multimarker.csv")
MARKERS = ["CRP","NLR","MLR","PLR","SII"]

# Figure A: CRP distribution by amalgam burden
plt.figure(figsize=(8,6))
sns.violinplot(data=merged, x="amalgam_group", y="CRP", cut=0, inner="quartile")
plt.yscale("log"); plt.xlabel("Amalgam Exposure Group"); plt.ylabel("CRP (mg/L, log scale)")
plt.title("CRP by Amalgam Burden")
plt.tight_layout(); plt.savefig("output_figures/figA_crp_by_burden.png"); plt.close()

# Behavior distributions
smoke = (merged.dropna(subset=["amalgam_group","CurrentSmoker","WTMEC2YR"])
               .groupby(["amalgam_group","CurrentSmoker"])
               .apply(lambda x: np.sum(x["WTMEC2YR"])/np.sum(merged["WTMEC2YR"]))
               .reset_index(name="Weighted_Proportion"))
plt.figure(figsize=(8,5))
sns.barplot(data=smoke, x="amalgam_group", y="Weighted_Proportion", hue="CurrentSmoker")
plt.title("Smokers by Amalgam Burden"); plt.tight_layout()
plt.savefig("output_figures/figB_smoking_distribution.png"); plt.close()

drink = (merged.dropna(subset=["amalgam_group","Drinker","WTMEC2YR"])
               .groupby(["amalgam_group","Drinker"])
               .apply(lambda x: np.sum(x["WTMEC2YR"])/np.sum(merged["WTMEC2YR"]))
               .reset_index(name="Weighted_Proportion"))
plt.figure(figsize=(8,5))
sns.barplot(data=drink, x="amalgam_group", y="Weighted_Proportion", hue="Drinker")
plt.title("Drinkers by Amalgam Burden"); plt.tight_layout()
plt.savefig("output_figures/figC_drinking_distribution.png"); plt.close()

# For each marker: forest, trend, heatmap
for marker in MARKERS:
    reg = pd.read_csv(f"output_data/regression_results_by_cycle_{marker}.csv")
    reg = reg.dropna(subset=["Beta"])
    # Forest
    plt.figure(figsize=(7,5))
    plt.errorbar(reg["Beta"], reg["Cycle"],
                 xerr=[reg["Beta"]-reg["CI_Low"], reg["CI_High"]-reg["Beta"]],
                 fmt="o", ecolor="gray", capsize=4)
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel(f"Change in log({marker}) per amalgam surface (β ±95% CI)")
    plt.ylabel("NHANES Cycle"); plt.title(f"{marker}: Coefficients by Cycle")
    plt.tight_layout(); plt.savefig(f"output_figures/{marker}_forest.png"); plt.close()

    # Trend
    order = reg.reset_index(drop=True)
    plt.figure(figsize=(8,5))
    sns.lineplot(data=order, x="Cycle", y="Beta", marker="o")
    if {"CI_Low","CI_High"}.issubset(order.columns):
        plt.fill_between(np.arange(len(order)), order["CI_Low"], order["CI_High"], alpha=0.2)
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("β"); plt.title(f"{marker}: Temporal Trend")
    plt.tight_layout(); plt.savefig(f"output_figures/{marker}_trend.png"); plt.close()

    # Heatmap
    strat = pd.read_csv(f"output_data/stratified_results_{marker}.csv")
    pvt = strat.pivot(index="Sex", columns="Race", values="Beta")
    plt.figure(figsize=(7,4))
    sns.heatmap(pvt, annot=True, cmap="coolwarm", center=0, fmt=".3f")
    plt.title(f"{marker}: Adjusted β by Sex and Race")
    plt.tight_layout(); plt.savefig(f"output_figures/{marker}_heatmap.png"); plt.close()

print("✅ All multimarker figures saved to output_figures/")
