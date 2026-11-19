"""Phase 2: cleaning for datasets A and S, plus the model-ready preprocessing transformer.

Cleaning decisions are based on notebook 01 and explained in notebook 02:

Dataset A
  * drop `avg_watch_time_per_day`: it is exactly watch_hours / (last_login_days + 1), and 10 rows exceed 24 h/day
  * drop `monthly_fee`: a one-to-one recoding of `subscription_type`
  * keep the long `watch_hours` tail (plausible heavy viewers); it is log-transformed for linear models
Dataset S
  * drop `total_charges`: exactly account_age x monthly_charges
  * convert the four Yes/No columns to 0/1

Every step is logged (rows before/after, how many values were affected), so the notebook and the app can show
a cleaning log.

Run `python -m churn.preprocessing` to regenerate data/interim/ and reports/results/cleaning_log.csv.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config, data

ID, TARGET = config.ID_COL, config.TARGET

# ---------------------------------------------------------------- model-ready feature lists (after cleaning)
A_FEATURES_NUMERIC = ["age", "watch_hours", "last_login_days", "number_of_profiles"]
A_FEATURES_CATEGORICAL = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
A_LOG_COLS = ["watch_hours"]  # right-skewed (skew ~2.3); log1p before scaling

S_FEATURES_NUMERIC = [
    "account_age", "monthly_charges", "viewing_hours_per_week", "avg_viewing_duration",
    "content_downloads_per_month", "user_rating", "support_tickets_per_month", "watchlist_size",
]
S_FEATURES_BINARY = ["paperless_billing", "multi_device_access", "parental_control", "subtitles_enabled"]
S_FEATURES_CATEGORICAL = ["subscription_type", "payment_method", "content_type", "device", "favorite_genre", "gender"]

# Valid ranges used for checks (inclusive). None = unbounded on that side.
A_RANGES = {"age": (0, 120), "watch_hours": (0, None), "last_login_days": (0, None), "number_of_profiles": (1, None)}
S_RANGES = {
    "account_age": (0, None), "monthly_charges": (0, None), "viewing_hours_per_week": (0, 168),
    "avg_viewing_duration": (0, None), "content_downloads_per_month": (0, None), "user_rating": (1, 5),
    "support_tickets_per_month": (0, None), "watchlist_size": (0, None),
}


class _Log:
    """Collects one row per cleaning step."""

    def __init__(self, dataset: str):
        self.dataset, self.rows = dataset, []

    def add(self, step: str, before: pd.DataFrame, after: pd.DataFrame, n_affected: int, note: str) -> None:
        self.rows.append({
            "dataset": self.dataset, "step": step,
            "rows_before": len(before), "rows_after": len(after),
            "cols_before": before.shape[1], "cols_after": after.shape[1],
            "n_affected": int(n_affected), "note": note,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _common_steps(df: pd.DataFrame, log: _Log) -> pd.DataFrame:
    """Steps applied to every dataset: tidy strings, drop duplicates and rows with missing values."""
    before = df
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df = df.copy()
    changed = 0
    for c in str_cols:
        stripped = df[c].str.strip()
        changed += int((stripped != df[c]).sum())
        df[c] = stripped
    log.add("Trim whitespace in text columns", before, df, changed, "Normalise category labels")

    before = df
    df = df.drop_duplicates(subset=ID)
    log.add("Remove duplicate customer IDs", before, df, len(before) - len(df), "Keep the first record per ID")

    before = df
    df = df[~df.drop(columns=ID).duplicated()]
    log.add("Remove duplicate rows (ignoring ID)", before, df, len(before) - len(df), "Identical records")

    before = df
    df = df.dropna()
    log.add("Drop rows with missing values", before, df, len(before) - len(df),
            "No imputation needed: both datasets are complete")
    return df


def _check_ranges(df: pd.DataFrame, ranges: dict, log: _Log) -> pd.DataFrame:
    """Drop rows whose numeric values fall outside documented valid ranges."""
    before = df
    bad = pd.Series(False, index=df.index)
    for col, (lo, hi) in ranges.items():
        lo = -np.inf if lo is None else lo
        hi = np.inf if hi is None else hi
        bad |= ~df[col].between(lo, hi)
    df = df[~bad]
    log.add("Drop rows with out-of-range values", before, df, int(bad.sum()),
            "Checked: " + ", ".join(ranges))
    return df


def clean_a(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw Dataset A. Returns (clean_df, log_df)."""
    log = _Log("A")
    df = _common_steps(df, log)
    df = _check_ranges(df, A_RANGES, log)

    before = df
    n_invalid = int((df["avg_watch_time_per_day"] > 24).sum())
    df = df.drop(columns="avg_watch_time_per_day")
    log.add("Drop avg_watch_time_per_day", before, df, n_invalid,
            f"Derived: watch_hours / (last_login_days + 1); {n_invalid} rows were > 24 h/day")

    before = df
    df = df.drop(columns="monthly_fee")
    log.add("Drop monthly_fee", before, df, 0, "Redundant: one fixed fee per subscription_type")

    q1, q3 = df["watch_hours"].quantile([0.25, 0.75])
    n_out = int((df["watch_hours"] > q3 + 1.5 * (q3 - q1)).sum())
    log.add("Keep watch_hours outliers", df, df, n_out,
            "Plausible heavy viewers; log1p-transformed for Logistic Regression")

    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True), log.frame()


def clean_s(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw Dataset S (from `data.load_s`). Returns (clean_df, log_df)."""
    log = _Log("S")
    df = _common_steps(df, log)
    df = _check_ranges(df, S_RANGES, log)

    before = df
    df = df.drop(columns="total_charges")
    log.add("Drop total_charges", before, df, 0, "Derived: exactly account_age x monthly_charges")

    before = df
    df = df.copy()
    n_yes = 0
    for c in S_FEATURES_BINARY:
        unexpected = set(df[c].unique()) - {"Yes", "No"}
        if unexpected:
            raise ValueError(f"{c} has unexpected values {unexpected}")
        n_yes += int((df[c] == "Yes").sum())
        df[c] = (df[c] == "Yes").astype(int)
    log.add("Convert Yes/No columns to 1/0", before, df, n_yes, ", ".join(S_FEATURES_BINARY))

    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True), log.frame()


def make_preprocessor(numeric, categorical, binary=(), log_cols=()) -> ColumnTransformer:
    """Model-ready transformer: log1p (optional) + standard scaling for numeric columns, one-hot encoding for
    categorical columns, and pass-through for 0/1 columns.

    Put it inside an sklearn Pipeline with the model so that it is **fitted on the training split only**.
    Output is a pandas DataFrame with readable column names (useful for SHAP and feature importance).
    """
    log_cols = [c for c in numeric if c in set(log_cols)]
    plain = [c for c in numeric if c not in set(log_cols)]
    transformers = []
    if log_cols:
        transformers.append(("log_num", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), log_cols))
    if plain:
        transformers.append(("num", StandardScaler(), plain))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)))
    if binary:
        transformers.append(("bin", "passthrough", list(binary)))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def run() -> dict[str, pd.DataFrame]:
    """Clean A and S, write data/interim/*.csv and reports/results/cleaning_log.csv."""
    config.ensure_dirs()
    a, log_a = clean_a(data.load_a())
    s, log_s = clean_s(data.load_s())
    a.to_csv(config.CLEAN_A, index=False)
    s.to_csv(config.CLEAN_S, index=False)
    log = pd.concat([log_a, log_s], ignore_index=True)
    log.to_csv(config.RESULTS_DIR / "cleaning_log.csv", index=False)
    return {"a": a, "s": s, "log": log}


if __name__ == "__main__":
    out = run()
    print(f"A: {out['a'].shape} -> {config.CLEAN_A}")
    print(f"S: {out['s'].shape} -> {config.CLEAN_S}")
    print(f"log: {len(out['log'])} steps -> {config.RESULTS_DIR / 'cleaning_log.csv'}")
