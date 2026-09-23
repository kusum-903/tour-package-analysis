"""
src/analysis.py
---------------
Pure-pandas grouping and summary helpers.
Every function receives a DataFrame and returns a DataFrame (or scalar).
No Streamlit / Plotly imports here — keeps it independently testable.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ── Overview ───────────────────────────────────────────────────────────────

def conversion_summary(df: pd.DataFrame) -> dict:
    """High-level KPIs."""
    total      = len(df)
    taken      = int(df["ProdTaken"].sum())
    not_taken  = total - taken
    rate       = round(taken / total * 100, 2)
    return {
        "total_customers": total,
        "packages_sold":   taken,
        "not_sold":        not_taken,
        "conversion_rate": rate,
    }


# ── Grouping helpers ────────────────────────────────────────────────────────

def conversion_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Conversion rate and counts grouped by a categorical column."""
    grp = (
        df.groupby(col, observed=True)["ProdTaken"]
        .agg(Total="count", Sold="sum")
        .assign(ConversionRate=lambda x: (x["Sold"] / x["Total"] * 100).round(2))
        .sort_values("ConversionRate", ascending=False)
        .reset_index()
    )
    return grp


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Descriptive stats for a list of numeric columns."""
    return df[cols].describe().T.round(2)


def mean_by(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Mean of *value_col* grouped by *group_col*, sorted descending."""
    return (
        df.groupby(group_col)[value_col]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={value_col: f"Mean_{value_col}"})
    )


def distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Value counts + percentage for a column."""
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "Count"]
    vc["Pct"] = (vc["Count"] / len(df) * 100).round(2)
    return vc


# ── Specific business summaries ────────────────────────────────────────────

def product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion stats per product pitched."""
    order = ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    grp   = conversion_by(df, "ProductPitched")
    grp["ProductPitched"] = pd.Categorical(grp["ProductPitched"], categories=order, ordered=True)
    return grp.sort_values("ProductPitched").reset_index(drop=True)


def income_by_designation(df: pd.DataFrame) -> pd.DataFrame:
    """Median monthly income per designation."""
    return (
        df.groupby("Designation")["MonthlyIncome"]
        .median()
        .sort_values()
        .round(0)
        .reset_index()
        .rename(columns={"MonthlyIncome": "MedianIncome"})
    )


def age_band_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion rate bucketed into 10-year age bands."""
    out = df.copy()
    out["AgeBand"] = pd.cut(
        out["Age"],
        bins=[18, 28, 38, 48, 58, 100],
        labels=["18-27", "28-37", "38-47", "48-57", "58+"],
        right=False,
    )
    return conversion_by(out, "AgeBand")


def passport_owncar_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion for Passport and OwnCar (binary) side by side."""
    rows = []
    for col in ["Passport", "OwnCar"]:
        for val, label in [(0, "No"), (1, "Yes")]:
            sub   = df[df[col] == val]
            rate  = sub["ProdTaken"].mean() * 100
            rows.append({"Feature": col, "Value": label,
                         "Total": len(sub), "ConversionRate": round(rate, 2)})
    return pd.DataFrame(rows)


def correlation_with_target(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Pearson correlation of numeric cols with ProdTaken, sorted by abs value."""
    corr = df[num_cols + ["ProdTaken"]].corr()["ProdTaken"].drop("ProdTaken")
    return (
        corr.abs()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", "ProdTaken": "AbsCorrelation"})
    )


def followup_income_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Mean income and conversion rate by number of follow-ups."""
    return (
        df.groupby("NumberOfFollowups")
        .agg(
            Customers    =("ProdTaken", "count"),
            ConversionRate=("ProdTaken", lambda s: round(s.mean() * 100, 2)),
            MeanIncome   =("MonthlyIncome", lambda s: round(s.mean(), 0)),
        )
        .reset_index()
    )
