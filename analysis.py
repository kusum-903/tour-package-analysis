"""
analysis.py  —  Tour Package Sales: Complete Statistical Analysis
=================================================================
Reads  : tour_package_cleaned.csv
Outputs: analysis_report/
         ├── 01_dataset_summary.csv
         ├── 02_descriptive_stats.csv
         ├── 03_target_distribution.csv
         ├── 04_conversion_by_category.csv
         ├── 05_numeric_by_target.csv
         ├── 06_correlation_with_target.csv
         ├── 07_outlier_report.csv
         ├── 08_crosstab_product_marital.csv
         ├── 09_crosstab_designation_passport.csv
         ├── 10_income_by_designation.csv
         ├── 11_followup_pivot.csv
         ├── 12_age_band_analysis.csv
         ├── 13_chi_square_results.csv
         ├── 14_ttest_results.csv
         └── 15_full_analysis_report.txt
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import pandas as pd
import numpy as np
from scipy import stats

os.makedirs("analysis_report", exist_ok=True)

SEP  = "=" * 65
SEP2 = "-" * 65
OUT  = []   # collects all text for the master report

def log(text=""):
    print(text)
    OUT.append(str(text))

# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("tour_package_cleaned.csv")

CAT_COLS = ["TypeofContact", "Occupation", "Gender",
            "ProductPitched", "MaritalStatus", "Designation"]
NUM_COLS = ["Age", "DurationOfPitch", "NumberOfPersonVisiting",
            "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
            "PitchSatisfactionScore", "NumberOfChildrenVisiting", "MonthlyIncome"]
TARGET   = "ProdTaken"

# ─────────────────────────────────────────────────────────────
# 1. Dataset Summary
# ─────────────────────────────────────────────────────────────
log(SEP)
log("SECTION 1 — DATASET SUMMARY")
log(SEP)

summary = pd.DataFrame({
    "Metric": [
        "Total Rows", "Total Columns",
        "Numeric Columns", "Categorical Columns",
        "Binary Columns", "Target Column",
        "Missing Values", "Duplicate Rows",
    ],
    "Value": [
        df.shape[0], df.shape[1],
        len(NUM_COLS), len(CAT_COLS),
        2,  TARGET,
        df.isnull().sum().sum(), df.duplicated().sum(),
    ],
})
summary.to_csv("analysis_report/01_dataset_summary.csv", index=False)
log(summary.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 2. Descriptive Statistics
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 2 — DESCRIPTIVE STATISTICS (Numeric Features)")
log(SEP)

desc = df[NUM_COLS].describe().T.round(2)
desc["skew"]     = df[NUM_COLS].skew().round(3)
desc["kurtosis"] = df[NUM_COLS].kurt().round(3)
desc.to_csv("analysis_report/02_descriptive_stats.csv")
log(desc.to_string())

# ─────────────────────────────────────────────────────────────
# 3. Target Distribution
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 3 — TARGET DISTRIBUTION")
log(SEP)

vc = df[TARGET].value_counts().reset_index()
vc.columns = ["ProdTaken", "Count"]
vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
vc["Label"] = vc["ProdTaken"].map({0: "Not Taken", 1: "Taken"})
vc.to_csv("analysis_report/03_target_distribution.csv", index=False)
log(vc.to_string(index=False))
log(f"\n  Conversion Rate : {df[TARGET].mean()*100:.2f}%")
log(f"  Class Imbalance Ratio (0:1) : {(vc.set_index('ProdTaken')['Count'][0] / vc.set_index('ProdTaken')['Count'][1]):.2f}:1")

# ─────────────────────────────────────────────────────────────
# 4. Conversion Rate by Each Categorical Feature
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 4 — CONVERSION RATE BY CATEGORICAL FEATURES")
log(SEP)

conv_all_rows = []
for col in CAT_COLS:
    grp = (
        df.groupby(col)[TARGET]
        .agg(Total="count", Sold="sum")
        .assign(
            ConversionRate=lambda x: (x["Sold"] / x["Total"] * 100).round(2),
            NotSold=lambda x: x["Total"] - x["Sold"],
        )
        .sort_values("ConversionRate", ascending=False)
        .reset_index()
    )
    grp.insert(0, "Feature", col)
    grp.rename(columns={col: "Category"}, inplace=True)
    conv_all_rows.append(grp)
    log(f"\n  {col}:")
    log("  " + grp.to_string(index=False))

conv_df = pd.concat(conv_all_rows, ignore_index=True)
conv_df.to_csv("analysis_report/04_conversion_by_category.csv", index=False)

# ─────────────────────────────────────────────────────────────
# 5. Numeric Feature Means — Taken vs Not Taken
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 5 — NUMERIC FEATURE MEANS: TAKEN vs NOT TAKEN")
log(SEP)

grp_means = df.groupby(TARGET)[NUM_COLS].mean().round(2)
grp_means.index = ["Not Taken (0)", "Taken (1)"]
diff = grp_means.loc["Taken (1)"] - grp_means.loc["Not Taken (0)"]
grp_means.loc["Difference (Taken - Not)"] = diff.round(2)
grp_means.to_csv("analysis_report/05_numeric_by_target.csv")
log(grp_means.T.to_string())

# ─────────────────────────────────────────────────────────────
# 6. Correlation with Target
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 6 — PEARSON CORRELATION WITH TARGET (ProdTaken)")
log(SEP)

corr_series = df[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET)
corr_df = pd.DataFrame({
    "Feature":        corr_series.index,
    "Correlation":    corr_series.round(4).values,
    "Abs_Correlation":corr_series.abs().round(4).values,
    "Direction":      ["Positive" if v > 0 else "Negative" for v in corr_series.values],
    "Strength":       [
        "Strong"   if abs(v) >= 0.3 else
        "Moderate" if abs(v) >= 0.15 else
        "Weak"     if abs(v) >= 0.05 else "Negligible"
        for v in corr_series.values
    ],
}).sort_values("Abs_Correlation", ascending=False)
corr_df.to_csv("analysis_report/06_correlation_with_target.csv", index=False)
log(corr_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. Outlier Detection (IQR method)
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 7 — OUTLIER DETECTION (IQR Method)")
log(SEP)

outlier_rows = []
for col in NUM_COLS:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr
    n_out  = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_rows.append({
        "Feature":       col,
        "Q1":            round(q1, 2),
        "Q3":            round(q3, 2),
        "IQR":           round(iqr, 2),
        "Lower_Fence":   round(lower, 2),
        "Upper_Fence":   round(upper, 2),
        "Outlier_Count": int(n_out),
        "Outlier_%":     round(n_out / len(df) * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_rows).sort_values("Outlier_Count", ascending=False)
outlier_df.to_csv("analysis_report/07_outlier_report.csv", index=False)
log(outlier_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. Cross-tab: Product Pitched × Marital Status
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 8 — CROSS-TAB: ProductPitched × MaritalStatus (Conversion Rate %)")
log(SEP)

prod_order = ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
ct8 = df.groupby(["ProductPitched", "MaritalStatus"])[TARGET].mean().unstack() * 100
ct8 = ct8.round(1)
ct8.to_csv("analysis_report/08_crosstab_product_marital.csv")
log(ct8.to_string())

# ─────────────────────────────────────────────────────────────
# 9. Cross-tab: Designation × Passport
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 9 — CROSS-TAB: Designation × Passport (Conversion Rate %)")
log(SEP)

ct9 = df.groupby(["Designation", "Passport"])[TARGET].mean().unstack() * 100
ct9 = ct9.round(1).rename(columns={0: "No Passport", 1: "Has Passport"})
ct9.to_csv("analysis_report/09_crosstab_designation_passport.csv")
log(ct9.to_string())

# ─────────────────────────────────────────────────────────────
# 10. Income Analysis by Designation
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 10 — MONTHLY INCOME ANALYSIS BY DESIGNATION")
log(SEP)

inc = (
    df.groupby("Designation")["MonthlyIncome"]
    .agg(Count="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
    .round(0)
    .sort_values("Median")
    .reset_index()
)
inc.to_csv("analysis_report/10_income_by_designation.csv", index=False)
log(inc.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 11. Follow-up Analysis
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 11 — FOLLOW-UP COUNT: CONVERSION & INCOME PIVOT")
log(SEP)

fu = (
    df.groupby("NumberOfFollowups")
    .agg(
        Customers     =("ProdTaken", "count"),
        Sold          =("ProdTaken", "sum"),
        ConversionRate=("ProdTaken", lambda s: round(s.mean() * 100, 2)),
        MeanIncome    =("MonthlyIncome", lambda s: round(s.mean(), 0)),
        MedianAge     =("Age", "median"),
    )
    .reset_index()
)
fu.to_csv("analysis_report/11_followup_pivot.csv", index=False)
log(fu.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 12. Age Band Analysis
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 12 — AGE BAND CONVERSION ANALYSIS")
log(SEP)

df_age = df.copy()
df_age["AgeBand"] = pd.cut(
    df_age["Age"],
    bins=[17, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)
age_df = (
    df_age.groupby("AgeBand", observed=True)
    .agg(
        Count         =("ProdTaken", "count"),
        Sold          =("ProdTaken", "sum"),
        ConversionRate=("ProdTaken", lambda s: round(s.mean() * 100, 2)),
        MeanIncome    =("MonthlyIncome", lambda s: round(s.mean(), 0)),
        MeanAge       =("Age", lambda s: round(s.mean(), 1)),
        PassportRate  =("Passport", lambda s: round(s.mean() * 100, 1)),
    )
    .reset_index()
)
age_df.to_csv("analysis_report/12_age_band_analysis.csv", index=False)
log(age_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 13. Chi-Square Tests (Categorical vs Target)
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 13 — CHI-SQUARE TEST: Categorical Features vs ProdTaken")
log(SEP)
log("  (Tests whether each categorical feature is statistically")
log("   independent of the purchase decision.)")
log()

chi_rows = []
for col in CAT_COLS:
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    chi_rows.append({
        "Feature":    col,
        "Chi2":       round(chi2, 3),
        "p_value":    round(p, 6),
        "df":         dof,
        "Significant": "YES ***" if p < 0.001 else
                       "YES **"  if p < 0.01  else
                       "YES *"   if p < 0.05  else "NO",
        "Conclusion": "Dependent (linked to purchase)" if p < 0.05
                      else "Independent (no significant link)",
    })

chi_df = pd.DataFrame(chi_rows).sort_values("p_value")
chi_df.to_csv("analysis_report/13_chi_square_results.csv", index=False)
log(chi_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 14. Independent T-Tests (Numeric vs Target)
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 14 — INDEPENDENT T-TEST: Numeric Features vs ProdTaken")
log(SEP)
log("  (Tests whether mean of each numeric feature differs")
log("   significantly between Taken and Not Taken groups.)")
log()

taken     = df[df[TARGET] == 1]
not_taken = df[df[TARGET] == 0]

ttest_rows = []
for col in NUM_COLS:
    t_stat, p_val = stats.ttest_ind(taken[col], not_taken[col], equal_var=False)
    mean_taken     = round(taken[col].mean(), 3)
    mean_not_taken = round(not_taken[col].mean(), 3)
    ttest_rows.append({
        "Feature":       col,
        "Mean_Taken":    mean_taken,
        "Mean_NotTaken": mean_not_taken,
        "Difference":    round(mean_taken - mean_not_taken, 3),
        "T_Statistic":   round(t_stat, 3),
        "p_value":       round(p_val, 6),
        "Significant":   "YES ***" if p_val < 0.001 else
                         "YES **"  if p_val < 0.01  else
                         "YES *"   if p_val < 0.05  else "NO",
    })

ttest_df = pd.DataFrame(ttest_rows).sort_values("p_value")
ttest_df.to_csv("analysis_report/14_ttest_results.csv", index=False)
log(ttest_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 15. Key Findings Summary
# ─────────────────────────────────────────────────────────────
log()
log(SEP)
log("SECTION 15 — KEY FINDINGS & BUSINESS INSIGHTS")
log(SEP)

best_cat_conv = {}
for col in CAT_COLS:
    g = df.groupby(col)[TARGET].mean()
    best_cat_conv[col] = (g.idxmax(), round(g.max() * 100, 1))

findings = f"""
DATASET
  Total customers     : {len(df):,}
  Packages sold       : {df[TARGET].sum():,}
  Conversion rate     : {df[TARGET].mean()*100:.2f}%
  Class imbalance     : {(df[TARGET]==0).sum()}:1 (Not Taken : Taken)

DEMOGRAPHIC INSIGHTS
  Best age group      : Customers aged 18-35 convert more than 36+ (younger = higher rate)
  Younger customers (mean age Taken={taken['Age'].mean():.1f}) convert vs older (Not Taken={not_taken['Age'].mean():.1f})
  Gender gap          : Male {df[df['Gender']=='Male'][TARGET].mean()*100:.1f}%  |  Female {df[df['Gender']=='Female'][TARGET].mean()*100:.1f}%

TOP CONVERTING SEGMENTS
  Occupation          : {best_cat_conv['Occupation'][0]} ({best_cat_conv['Occupation'][1]}%)
  Marital Status      : {best_cat_conv['MaritalStatus'][0]} ({best_cat_conv['MaritalStatus'][1]}%)
  Designation         : {best_cat_conv['Designation'][0]} ({best_cat_conv['Designation'][1]}%)
  Product Pitched     : {best_cat_conv['ProductPitched'][0]} ({best_cat_conv['ProductPitched'][1]}%)
  Contact Type        : {best_cat_conv['TypeofContact'][0]} ({best_cat_conv['TypeofContact'][1]}%)

PASSPORT & CAR EFFECT
  Passport holders    : {df[df['Passport']==1][TARGET].mean()*100:.1f}% conversion vs {df[df['Passport']==0][TARGET].mean()*100:.1f}% (no passport)
  Own car             : {df[df['OwnCar']==1][TARGET].mean()*100:.1f}% conversion vs {df[df['OwnCar']==0][TARGET].mean()*100:.1f}% (no car)

PITCH QUALITY
  More follow-ups lead to higher conversion (Taken mean={taken['NumberOfFollowups'].mean():.2f} vs Not Taken={not_taken['NumberOfFollowups'].mean():.2f})
  Higher pitch satisfaction = higher conversion (Taken={taken['PitchSatisfactionScore'].mean():.2f} vs {not_taken['PitchSatisfactionScore'].mean():.2f})
  Longer pitch duration = higher conversion (Taken={taken['DurationOfPitch'].mean():.1f} min vs {not_taken['DurationOfPitch'].mean():.1f} min)

INCOME
  Taken group earns LESS on average (INR {taken['MonthlyIncome'].mean():,.0f}) vs Not Taken (INR {not_taken['MonthlyIncome'].mean():,.0f})
  Suggests lower-income customers are more motivated to buy discounted packages

STATISTICAL SIGNIFICANCE (Chi-Square, p < 0.05)
  Significant features: {', '.join(chi_df[chi_df['p_value'] < 0.05]['Feature'].tolist())}

STATISTICAL SIGNIFICANCE (T-Test, p < 0.05)
  Significant features: {', '.join(ttest_df[ttest_df['p_value'] < 0.05]['Feature'].tolist())}

STRONGEST PREDICTORS (by |Correlation| with target)
{corr_df[['Feature','Correlation','Strength']].head(5).to_string(index=False)}
"""

log(findings)

# ─────────────────────────────────────────────────────────────
# Save master text report
# ─────────────────────────────────────────────────────────────
full_report = "\n".join(OUT)
with open("analysis_report/15_full_analysis_report.txt", "w", encoding="utf-8") as f:
    f.write(full_report)

print()
print(SEP)
print("All outputs saved to  analysis_report/")
files = sorted(os.listdir("analysis_report"))
for fn in files:
    size = os.path.getsize(f"analysis_report/{fn}")
    print(f"  {fn:<45}  {size:>6,} bytes")
print(SEP)
