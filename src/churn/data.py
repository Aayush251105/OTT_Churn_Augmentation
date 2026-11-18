"""Loaders for the raw datasets.

Dataset A (primary, 5K):   data/raw/netflix_customer_churn.csv
Dataset B (supplementary): data/raw/netflix_large_user_data.xlsx

`load_b` renames B's columns to snake_case. Wherever a column has a counterpart
in A, it gets A's name, so the shared features line up for matching.
"""

import pandas as pd

from churn import config

# Raw Dataset B header -> snake_case name (shared features use Dataset A's names).
B_COLUMN_MAP = {
    "Customer ID": "customer_id",
    "Subscription Length (Months)": "subscription_length_months",
    "Customer Satisfaction Score (1-10)": "satisfaction_score",
    "Daily Watch Time (Hours)": "daily_watch_time",
    "Engagement Rate (1-10)": "engagement_rate",
    "Device Used Most Often": "device",
    "Genre Preference": "favorite_genre",
    "Region": "region",
    "Payment History (On-Time/Delayed)": "payment_history",
    "Subscription Plan": "subscription_type",
    "Churn Status (Yes/No)": "churn_b",
    "Support Queries Logged": "support_queries",
    "Age": "age",
    "Monthly Income ($)": "monthly_income",
    "Promotional Offers Used": "promo_offers_used",
    "Number of Profiles Created": "number_of_profiles",
}

A_NUMERIC = [
    "age", "watch_hours", "last_login_days", "monthly_fee",
    "number_of_profiles", "avg_watch_time_per_day",
]
A_CATEGORICAL = [
    "gender", "subscription_type", "region", "device",
    "payment_method", "favorite_genre",
]

B_NUMERIC = [
    "subscription_length_months", "satisfaction_score", "daily_watch_time",
    "engagement_rate", "support_queries", "age", "monthly_income",
    "promo_offers_used", "number_of_profiles",
]
B_CATEGORICAL = [
    "device", "favorite_genre", "region", "payment_history", "subscription_type",
]


def load_a() -> pd.DataFrame:
    """Load raw Dataset A (5,000 rows x 14 columns). Target column: `churned` (0/1)."""
    return pd.read_csv(config.RAW_A)


def load_b() -> pd.DataFrame:
    """Load raw Dataset B (1,000 rows x 16 columns) with snake_case column names.

    `churn_b` is B's own churn label ("Yes"/"No"). It may be used in EDA,
    but it must never be transferred to Dataset A as a feature.
    """
    df = pd.read_excel(config.RAW_B, engine="openpyxl")
    missing = set(B_COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset B is missing expected columns: {sorted(missing)}")
    return df.rename(columns=B_COLUMN_MAP)[list(B_COLUMN_MAP.values())]


def load_clean_a() -> pd.DataFrame:
    """Load cleaned Dataset A produced by Phase 2 (data/interim/a_clean.csv)."""
    return pd.read_csv(config.CLEAN_A)


def load_clean_b() -> pd.DataFrame:
    """Load cleaned Dataset B produced by Phase 2 (data/interim/b_clean.csv)."""
    return pd.read_csv(config.CLEAN_B)


def load_experiment(name: str) -> pd.DataFrame:
    """Load one experiment dataset (A_full, A_matched, A_augmented, A_random)."""
    return pd.read_csv(config.experiment_path(name))
