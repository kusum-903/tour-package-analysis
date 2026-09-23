import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Load the dataset
# ─────────────────────────────────────────────
df = pd.read_csv("tour_package.csv")

print("=" * 60)
print("STEP 1 — Dataset Overview")
print("=" * 60)
print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}\n")
print(df.dtypes)
print()

# ─────────────────────────────────────────────
# 2. Missing Values
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Missing Values")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_report = missing_report[missing_report["Missing Count"] > 0]
if missing_report.empty:
    print("No missing values found.\n")
else:
    print(missing_report.to_string())
    print()

# ─────────────────────────────────────────────
# 3. Duplicate Rows
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Duplicate Rows")
print("=" * 60)
dup_count = df.duplicated().sum()
print(f"Duplicate rows : {dup_count}\n")

# ─────────────────────────────────────────────
# 4. Incorrect / Inconsistent Values
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4 — Incorrect / Inconsistent Values")
print("=" * 60)

# 4a. Gender — check unexpected categories
gender_values = df["Gender"].value_counts(dropna=False)
print("Gender value counts:")
print(gender_values.to_string())
fe_male_rows = df[df["Gender"] == "Fe Male"].shape[0]
print(f"\n  >> 'Fe Male' (typo for 'Female'): {fe_male_rows} rows\n")

# 4b. ProductPitched — check all categories
print("ProductPitched value counts:")
print(df["ProductPitched"].value_counts(dropna=False).to_string())
print()

# 4c. MaritalStatus — check categories
print("MaritalStatus value counts:")
print(df["MaritalStatus"].value_counts(dropna=False).to_string())
print()

# 4d. Numeric columns — basic stats to spot outliers
print("Numeric column statistics:")
print(df.describe().T.to_string())
print()

# 4e. MonthlyIncome — flag suspiciously high values (> 3× IQR fence)
q1 = df["MonthlyIncome"].quantile(0.25)
q3 = df["MonthlyIncome"].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 3 * iqr
outliers_income = df[df["MonthlyIncome"] > upper_fence]
print(f"MonthlyIncome outliers (> {upper_fence:.0f}, 3×IQR fence): {len(outliers_income)} rows")
if not outliers_income.empty:
    print(outliers_income[["CustomerID", "MonthlyIncome"]].to_string(index=False))
print()

# ─────────────────────────────────────────────
# 5. Data Cleaning
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5 — Applying Fixes")
print("=" * 60)

df_clean = df.copy()

# Fix 1 — Correct 'Fe Male' → 'Female'
df_clean["Gender"] = df_clean["Gender"].replace("Fe Male", "Female")
print(f"  [Fixed] 'Fe Male' -> 'Female' in Gender  ({fe_male_rows} rows)")

# Fix 2 — Fill missing Age with median (robust to skew)
age_median = df_clean["Age"].median()
age_missing = df_clean["Age"].isnull().sum()
df_clean["Age"] = df_clean["Age"].fillna(age_median)
print(f"  [Fixed] Missing Age filled with median ({age_median})  ({age_missing} rows)")

# Fix 3 — Fill missing MonthlyIncome with median per Designation
income_missing = df_clean["MonthlyIncome"].isnull().sum()
df_clean["MonthlyIncome"] = df_clean.groupby("Designation")["MonthlyIncome"].transform(
    lambda x: x.fillna(x.median())
)
print(f"  [Fixed] Missing MonthlyIncome filled with Designation-wise median  ({income_missing} rows)")

# Fix 4 — Fill missing DurationOfPitch with median
dur_missing = df_clean["DurationOfPitch"].isnull().sum()
dur_median = df_clean["DurationOfPitch"].median()
df_clean["DurationOfPitch"] = df_clean["DurationOfPitch"].fillna(dur_median)
print(f"  [Fixed] Missing DurationOfPitch filled with median ({dur_median})  ({dur_missing} rows)")

# Fix 5 — Fill missing NumberOfFollowups with median
if df_clean["NumberOfFollowups"].isnull().sum() > 0:
    nf_missing = df_clean["NumberOfFollowups"].isnull().sum()
    nf_median = df_clean["NumberOfFollowups"].median()
    df_clean["NumberOfFollowups"] = df_clean["NumberOfFollowups"].fillna(nf_median)
    print(f"  [Fixed] Missing NumberOfFollowups filled with median ({nf_median})  ({nf_missing} rows)")

# Fix 6 — Fill missing PreferredPropertyStar with mode
if df_clean["PreferredPropertyStar"].isnull().sum() > 0:
    pps_missing = df_clean["PreferredPropertyStar"].isnull().sum()
    pps_mode = df_clean["PreferredPropertyStar"].mode()[0]
    df_clean["PreferredPropertyStar"] = df_clean["PreferredPropertyStar"].fillna(pps_mode)
    print(f"  [Fixed] Missing PreferredPropertyStar filled with mode ({pps_mode})  ({pps_missing} rows)")

# Fix 7 — Cap MonthlyIncome outliers at upper fence
outlier_count = len(df_clean[df_clean["MonthlyIncome"] > upper_fence])
df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].clip(upper=upper_fence)
print(f"  [Fixed] MonthlyIncome values above {upper_fence:.0f} capped  ({outlier_count} rows)")

# Fix 8 — Drop remaining rows with any missing value (safety net)
rows_before = len(df_clean)
df_clean.dropna(inplace=True)
rows_dropped = rows_before - len(df_clean)
print(f"  [Fixed] Dropped {rows_dropped} rows still containing NaN after imputation")

# ─────────────────────────────────────────────
# 6. Post-Cleaning Validation
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6 — Post-Cleaning Validation")
print("=" * 60)
print(f"Shape after cleaning : {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"Remaining missing    : {df_clean.isnull().sum().sum()}")
print(f"Gender unique values : {sorted(df_clean['Gender'].unique())}")
print(f"ProductPitched values: {sorted(df_clean['ProductPitched'].unique())}")
print(f"MaritalStatus values : {sorted(df_clean['MaritalStatus'].unique())}")
print()

# ─────────────────────────────────────────────
# 7. Save cleaned dataset
# ─────────────────────────────────────────────
df_clean.to_csv("tour_package_cleaned.csv", index=False)
print("Cleaned dataset saved to: tour_package_cleaned.csv")
