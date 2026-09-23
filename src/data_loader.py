"""
src/data_loader.py
------------------
Loads the raw tour-package CSV, cleans it, and returns a tidy DataFrame.
All cleaning steps are documented so they can be audited easily.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np

# ── Column groups ──────────────────────────────────────────────────────────
CAT_COLS = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]
NUM_COLS = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "PitchSatisfactionScore",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
BINARY_COLS = ["Passport", "OwnCar"]
TARGET = "ProdTaken"

# ── Column descriptions (used in the UI) ──────────────────────────────────
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "CustomerID":                "Unique customer identifier",
    "ProdTaken":                 "Target – 1 if customer purchased the tour package, else 0",
    "Age":                       "Customer age (years)",
    "TypeofContact":             "How the customer was contacted (Self Enquiry / Company Invited)",
    "CityTier":                  "City tier (1 = metro, 3 = smaller city)",
    "DurationOfPitch":           "Duration of the sales pitch (minutes)",
    "Occupation":                "Customer occupation",
    "Gender":                    "Customer gender",
    "NumberOfPersonVisiting":    "Number of people travelling with the customer",
    "NumberOfFollowups":         "Number of follow-up calls made",
    "ProductPitched":            "Tour product shown to the customer",
    "PreferredPropertyStar":     "Preferred hotel star rating",
    "MaritalStatus":             "Marital status of the customer",
    "NumberOfTrips":             "Average number of trips per year",
    "Passport":                  "1 if customer holds a passport, else 0",
    "PitchSatisfactionScore":    "Satisfaction score for the sales pitch (1–5)",
    "OwnCar":                    "1 if customer owns a car, else 0",
    "NumberOfChildrenVisiting":  "Number of children travelling",
    "Designation":               "Job designation / seniority",
    "MonthlyIncome":             "Customer's monthly income (INR)",
}


def load_raw(path: str | pathlib.Path) -> pd.DataFrame:
    """Read the raw CSV.  Strips the BOM if present."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a fresh DataFrame.

    Steps
    -----
    1. Fix 'Fe Male' typo → 'Female'
    2. Fill missing Age with column median
    3. Fill missing MonthlyIncome with per-Designation median
    4. Fill missing DurationOfPitch with column median
    5. Fill missing NumberOfFollowups with column median
    6. Fill missing PreferredPropertyStar with column mode
    7. Cap MonthlyIncome outliers at the 3×IQR upper fence
    8. Drop any rows that still contain NaN
    """
    out = df.copy()

    # 1. Gender typo
    out["Gender"] = out["Gender"].replace("Fe Male", "Female")

    # 2. Age
    out["Age"] = out["Age"].fillna(out["Age"].median())

    # 3. MonthlyIncome (group-wise)
    out["MonthlyIncome"] = out.groupby("Designation")["MonthlyIncome"].transform(
        lambda s: s.fillna(s.median())
    )

    # 4. DurationOfPitch
    out["DurationOfPitch"] = out["DurationOfPitch"].fillna(out["DurationOfPitch"].median())

    # 5. NumberOfFollowups
    out["NumberOfFollowups"] = out["NumberOfFollowups"].fillna(out["NumberOfFollowups"].median())

    # 6. PreferredPropertyStar
    mode_val = out["PreferredPropertyStar"].mode()
    if len(mode_val):
        out["PreferredPropertyStar"] = out["PreferredPropertyStar"].fillna(mode_val[0])

    # 7. Cap income outliers (3×IQR fence)
    q1, q3 = out["MonthlyIncome"].quantile([0.25, 0.75])
    upper_fence = q3 + 3 * (q3 - q1)
    out["MonthlyIncome"] = out["MonthlyIncome"].clip(upper=upper_fence)

    # 8. Drop any remaining NaNs
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)

    return out


def load_clean(path: str | pathlib.Path) -> pd.DataFrame:
    """Convenience: load raw → clean → return."""
    return clean(load_raw(path))
