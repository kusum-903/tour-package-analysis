import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Output folder ─────────────────────────────────────────
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Style ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13,
                     "axes.labelsize": 11, "xtick.labelsize": 9,
                     "ytick.labelsize": 9})

# ── Load ───────────────────────────────────────────────────
df = pd.read_csv("tour_package_cleaned.csv")

CAT_COLS = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
            "MaritalStatus", "Designation"]
NUM_COLS = ["Age", "DurationOfPitch", "NumberOfPersonVisiting",
            "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
            "PitchSatisfactionScore", "NumberOfChildrenVisiting", "MonthlyIncome"]
TARGET   = "ProdTaken"

print("=" * 60)
print("EDA — tour_package_cleaned.csv")
print("=" * 60)
print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns\n")

# ══════════════════════════════════════════════════════════
# 1. TARGET DISTRIBUTION
# ══════════════════════════════════════════════════════════
print("── 1. Target Distribution ──")
vc = df[TARGET].value_counts()
labels = ["Not Taken (0)", "Taken (1)"]
colors  = ["#7c9cbf", "#e07b54"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(labels, vc.values, color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Package Taken Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontsize=10, fontweight="bold")

axes[1].pie(vc.values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Package Taken Split")
plt.suptitle("Target Variable — ProdTaken", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("01_target_distribution.png")
print(f"  Not Taken: {vc[0]}  |  Taken: {vc[1]}  |  Conversion rate: {vc[1]/len(df)*100:.1f}%")

# ══════════════════════════════════════════════════════════
# 2. NUMERIC — DISTRIBUTIONS (histograms + KDE)
# ══════════════════════════════════════════════════════════
print("\n-- 2. Numeric Distributions --")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#5b8db8", bins=30, edgecolor="white")
    axes[i].set_title(col)
    axes[i].set_xlabel("")
plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("02_numeric_distributions.png")

# ══════════════════════════════════════════════════════════
# 3. NUMERIC — BOXPLOTS (spot outliers)
# ══════════════════════════════════════════════════════════
print("── 3. Numeric Boxplots ──")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    sns.boxplot(y=df[col], ax=axes[i], color="#88b4d4", width=0.4,
                flierprops=dict(marker="o", markerfacecolor="#e07b54", markersize=3))
    axes[i].set_title(col)
    axes[i].set_xlabel("")
plt.suptitle("Numeric Feature Boxplots", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("03_numeric_boxplots.png")

# ══════════════════════════════════════════════════════════
# 4. CATEGORICAL — BAR CHARTS
# ══════════════════════════════════════════════════════════
print("── 4. Categorical Distributions ──")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(CAT_COLS):
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, hue=col, ax=axes[i],
                  palette="muted", edgecolor="white", legend=False)
    axes[i].set_title(col)
    axes[i].set_xlabel("Count")
    axes[i].set_ylabel("")
    for bar in axes[i].patches:
        axes[i].text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                     f"{int(bar.get_width())}", va="center", fontsize=8)
plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("04_categorical_distributions.png")

# ══════════════════════════════════════════════════════════
# 5. BIVARIATE — NUMERIC vs TARGET (violin + box)
# ══════════════════════════════════════════════════════════
print("── 5. Numeric vs Target (Violin Plots) ──")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
df["_target_str"] = df[TARGET].map({0: "Not Taken", 1: "Taken"})
for i, col in enumerate(NUM_COLS):
    sns.violinplot(data=df, x="_target_str", y=col, hue="_target_str", ax=axes[i],
                   palette={"Not Taken": "#7c9cbf", "Taken": "#e07b54"},
                   order=["Not Taken", "Taken"], inner="quartile",
                   cut=0, legend=False)
    axes[i].set_title(col)
    axes[i].set_xlabel("")
plt.suptitle("Numeric Features vs ProdTaken (Violin)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("05_numeric_vs_target_violin.png")

# ══════════════════════════════════════════════════════════
# 6. BIVARIATE — CATEGORICAL vs TARGET (conversion rate bars)
# ══════════════════════════════════════════════════════════
print("── 6. Categorical vs Target (Conversion Rate) ──")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(CAT_COLS):
    conv = df.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
    conv.plot(kind="bar", ax=axes[i], color="#5b8db8", edgecolor="white", width=0.6)
    axes[i].set_title(f"Conversion Rate by {col}")
    axes[i].set_ylabel("Conversion Rate (%)")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30)
    axes[i].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    for bar in axes[i].patches:
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=8)
plt.suptitle("Conversion Rate (ProdTaken=1) by Category", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("06_categorical_vs_target_conversion.png")

# ══════════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════
print("── 7. Correlation Heatmap ──")
corr_cols = NUM_COLS + [TARGET]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Correlation Matrix (Numeric Features + Target)", fontsize=14, fontweight="bold")
plt.tight_layout()
save("07_correlation_heatmap.png")

# Print top correlations with target
target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("\n  Top correlations with ProdTaken:")
for feat, val in target_corr.items():
    print(f"    {feat:<30} {val:.3f}")

# ══════════════════════════════════════════════════════════
# 8. AGE DISTRIBUTION by TARGET
# ══════════════════════════════════════════════════════════
print("\n── 8. Age Distribution by Target ──")
fig, ax = plt.subplots(figsize=(10, 5))
for val, label, color in [(0, "Not Taken", "#7c9cbf"), (1, "Taken", "#e07b54")]:
    sns.kdeplot(df[df[TARGET] == val]["Age"], ax=ax, label=label,
                color=color, fill=True, alpha=0.35, linewidth=2)
ax.set_title("Age Distribution by ProdTaken", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
save("08_age_by_target.png")

# ══════════════════════════════════════════════════════════
# 9. MONTHLY INCOME by DESIGNATION (box)
# ══════════════════════════════════════════════════════════
print("── 9. Monthly Income by Designation ──")
order = df.groupby("Designation")["MonthlyIncome"].median().sort_values().index
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=df, x="Designation", y="MonthlyIncome", order=order,
            hue="Designation", palette="Blues", ax=ax, legend=False,
            flierprops=dict(marker="o", markerfacecolor="#e07b54", markersize=3))
ax.set_title("Monthly Income by Designation", fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Monthly Income")
ax.tick_params(axis="x", rotation=20)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
save("09_income_by_designation.png")

# ══════════════════════════════════════════════════════════
# 10. PRODUCT PITCHED vs TARGET (stacked 100% bar)
# ══════════════════════════════════════════════════════════
print("── 10. ProductPitched vs Target ──")
prod_order = ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
ct = df.groupby("ProductPitched")[TARGET].value_counts(normalize=True).unstack().reindex(prod_order)

fig, ax = plt.subplots(figsize=(9, 5))
ct.plot(kind="bar", stacked=True, ax=ax,
        color=["#7c9cbf", "#e07b54"], edgecolor="white", width=0.55)
ax.set_title("Product Pitched — Package Conversion (Stacked %)", fontsize=13, fontweight="bold")
ax.set_xlabel("Product Pitched")
ax.set_ylabel("Proportion")
ax.set_xticklabels(prod_order, rotation=0)
ax.legend(["Not Taken", "Taken"], loc="upper right")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
save("10_product_vs_target_stacked.png")

# ══════════════════════════════════════════════════════════
# 11. CITYTIER vs TARGET
# ══════════════════════════════════════════════════════════
print("── 11. CityTier vs Target ──")
ct2 = df.groupby("CityTier")[TARGET].value_counts(normalize=True).unstack()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
conv_city = df.groupby("CityTier")[TARGET].mean() * 100
axes[0].bar(conv_city.index.astype(str), conv_city.values,
            color=["#5b8db8", "#7aab8a", "#d4875a"], edgecolor="white", width=0.5)
axes[0].set_title("Conversion Rate by City Tier")
axes[0].set_xlabel("City Tier")
axes[0].set_ylabel("Conversion Rate (%)")
for i, v in enumerate(conv_city.values):
    axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

ct2.plot(kind="bar", stacked=True, ax=axes[1],
         color=["#7c9cbf", "#e07b54"], edgecolor="white", width=0.5)
axes[1].set_title("Stacked Proportion by City Tier")
axes[1].set_xlabel("City Tier")
axes[1].set_xticklabels(ct2.index.astype(str), rotation=0)
axes[1].legend(["Not Taken", "Taken"])
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.suptitle("CityTier vs ProdTaken", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save("11_citytier_vs_target.png")

# ══════════════════════════════════════════════════════════
# 12. PASSPORT & OWN CAR vs TARGET
# ══════════════════════════════════════════════════════════
print("── 12. Passport & OwnCar vs Target ──")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, lbl in zip(axes, ["Passport", "OwnCar"], ["Passport", "Own Car"]):
    conv = df.groupby(col)[TARGET].mean() * 100
    ax.bar(["No", "Yes"], conv.values,
           color=["#7c9cbf", "#e07b54"], edgecolor="white", width=0.4)
    ax.set_title(f"Conversion Rate by {lbl}")
    ax.set_ylabel("Conversion Rate (%)")
    for i, v in enumerate(conv.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
plt.suptitle("Passport & OwnCar vs ProdTaken", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save("12_passport_owncar_vs_target.png")

# ══════════════════════════════════════════════════════════
# 13. PAIRPLOT (key numeric features, colour = target)
# ══════════════════════════════════════════════════════════
print("── 13. Pairplot (key features) ──")
pair_cols = ["Age", "MonthlyIncome", "DurationOfPitch", "NumberOfTrips", TARGET]
pair_df   = df[pair_cols].copy()
pair_df[TARGET] = pair_df[TARGET].map({0: "Not Taken", 1: "Taken"})
g = sns.pairplot(pair_df, hue=TARGET, palette={"Not Taken": "#7c9cbf", "Taken": "#e07b54"},
                 plot_kws={"alpha": 0.35, "s": 12}, diag_kind="kde")
g.figure.suptitle("Pairplot — Key Features by ProdTaken", y=1.01,
                  fontsize=13, fontweight="bold")
save("13_pairplot_key_features.png")

# ══════════════════════════════════════════════════════════
# SUMMARY STATS PRINT
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(df[NUM_COLS + [TARGET]].describe().round(2).to_string())

print("\n" + "=" * 60)
print("Categorical Value Counts")
print("=" * 60)
for col in CAT_COLS:
    print(f"\n{col}:\n{df[col].value_counts().to_string()}")

print("\n" + "=" * 60)
print(f"All plots saved to './{OUTPUT_DIR}/'")
print("=" * 60)
