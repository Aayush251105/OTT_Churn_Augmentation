"""Reusable exploratory-analysis helpers.

Used by notebook 01 and by the app's Data Explorer page, so they return plain DataFrames
instead of drawing anything.
"""

import numpy as np
import pandas as pd


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, kind (numeric/categorical), unique values, missing values, an example."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "kind": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
            "unique": s.nunique(),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(rows)


def quality_report(df: pd.DataFrame, id_col: str | None = None) -> pd.Series:
    """Headline data-quality counts: rows, columns, missing cells, duplicate rows, duplicate IDs."""
    features = df.drop(columns=[id_col]) if id_col else df
    out = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(features.duplicated().sum()),
    }
    if id_col:
        out["duplicate_ids"] = int(df[id_col].duplicated().sum())
    return pd.Series(out)


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def outlier_summary(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """IQR outlier count, share and bounds for each numeric column."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = iqr_outliers(df[c], k)
        rows.append({
            "column": c, "lower": q1 - k * iqr, "upper": q3 + k * iqr,
            "n_outliers": int(mask.sum()), "pct": round(100 * mask.mean(), 2),
            "min": df[c].min(), "max": df[c].max(), "skew": round(df[c].skew(), 2),
        })
    return pd.DataFrame(rows)


def churn_rate_by(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    """Churn rate and customer count for each category of `col`, sorted by churn rate (highest first)."""
    g = df.groupby(col, observed=True)[target].agg(churn_rate="mean", customers="size")
    return g.sort_values("churn_rate", ascending=False).reset_index()


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series (0 = no association, 1 = perfect)."""
    from scipy.stats import chi2_contingency

    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def target_association(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                       target: str) -> pd.DataFrame:
    """Association of every feature with a binary target.

    Numeric features use Spearman's rho (signed); categorical features use Cramér's V (0..1).
    Sorted by absolute strength.
    """
    rows = [{"feature": c, "measure": "spearman", "value": df[c].corr(df[target], method="spearman")}
            for c in numeric]
    rows += [{"feature": c, "measure": "cramers_v", "value": cramers_v(df[c], df[target])}
             for c in categorical]
    out = pd.DataFrame(rows)
    out["strength"] = out["value"].abs()
    return out.sort_values("strength", ascending=False).reset_index(drop=True)
