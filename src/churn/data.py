"""Loaders for the raw and cleaned datasets.

Dataset S (main, 243K):     data/raw/train.csv
Dataset A (comparison, 5K): data/raw/netflix_customer_churn.csv
Old dataset B (audit only): data/raw/netflix_large_user_data.xlsx

Every loader returns snake_case columns with the ID in `customer_id` and the target in `churned` (0/1).
Columns that exist in more than one dataset get the same name (e.g. `subscription_type`, `device`).
"""

import pandas as pd

from churn import config

# ---------------------------------------------------------------- Dataset A
A_NUMERIC = [
    "age", "watch_hours", "last_login_days", "monthly_fee",
    "number_of_profiles", "avg_watch_time_per_day",
]
A_CATEGORICAL = [
    "gender", "subscription_type", "region", "device",
    "payment_method", "favorite_genre",
]

# ---------------------------------------------------------------- Dataset S
S_COLUMN_MAP = {
    "CustomerID": "customer_id",
    "AccountAge": "account_age",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "SubscriptionType": "subscription_type",
    "PaymentMethod": "payment_method",
    "PaperlessBilling": "paperless_billing",
    "ContentType": "content_type",
    "MultiDeviceAccess": "multi_device_access",
    "DeviceRegistered": "device",
    "ViewingHoursPerWeek": "viewing_hours_per_week",
    "AverageViewingDuration": "avg_viewing_duration",
    "ContentDownloadsPerMonth": "content_downloads_per_month",
    "GenrePreference": "favorite_genre",
    "UserRating": "user_rating",
    "SupportTicketsPerMonth": "support_tickets_per_month",
    "Gender": "gender",
    "WatchlistSize": "watchlist_size",
    "ParentalControl": "parental_control",
    "SubtitlesEnabled": "subtitles_enabled",
    "Churn": "churned",
}
S_NUMERIC = [
    "account_age", "monthly_charges", "total_charges", "viewing_hours_per_week",
    "avg_viewing_duration", "content_downloads_per_month", "user_rating",
    "support_tickets_per_month", "watchlist_size",
]
S_CATEGORICAL = [
    "subscription_type", "payment_method", "content_type", "device", "favorite_genre", "gender",
]
S_BINARY = [  # Yes/No in the raw file
    "paperless_billing", "multi_device_access", "parental_control", "subtitles_enabled",
]

# ---------------------------------------------------------------- old Dataset B (audit only)
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
    "Churn Status (Yes/No)": "churned",
    "Support Queries Logged": "support_queries",
    "Age": "age",
    "Monthly Income ($)": "monthly_income",
    "Promotional Offers Used": "promo_offers_used",
    "Number of Profiles Created": "number_of_profiles",
}
B_NUMERIC = [
    "subscription_length_months", "satisfaction_score", "daily_watch_time",
    "engagement_rate", "support_queries", "age", "monthly_income",
    "promo_offers_used", "number_of_profiles",
]
B_CATEGORICAL = [
    "device", "favorite_genre", "region", "payment_history", "subscription_type",
]


def _check_columns(df: pd.DataFrame, expected, name: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing expected columns: {sorted(missing)}")


def load_a() -> pd.DataFrame:
    """Raw Dataset A: 5,000 rows x 14 columns."""
    return pd.read_csv(config.RAW_A)


def load_s() -> pd.DataFrame:
    """Raw Dataset S: 243,787 rows x 21 columns, renamed to snake_case.

    Yes/No columns are left as strings here; `preprocessing.clean_s` converts them to 0/1.
    """
    df = pd.read_csv(config.RAW_S)
    _check_columns(df, S_COLUMN_MAP, "Dataset S")
    return df.rename(columns=S_COLUMN_MAP)[list(S_COLUMN_MAP.values())]


def load_s_descriptions() -> pd.DataFrame:
    """Column descriptions shipped with Dataset S, with snake_case names added."""
    d = pd.read_csv(config.RAW_S_DESCRIPTIONS)
    d.insert(1, "column", d["Column_name"].map(S_COLUMN_MAP))
    return d


def load_b() -> pd.DataFrame:
    """Raw old Dataset B (1,000 rows), audit only. `churned` is converted from Yes/No to 1/0."""
    df = pd.read_excel(config.RAW_B, engine="openpyxl")
    _check_columns(df, B_COLUMN_MAP, "Dataset B")
    df = df.rename(columns=B_COLUMN_MAP)[list(B_COLUMN_MAP.values())]
    df["churned"] = (df["churned"] == "Yes").astype(int)
    return df


def load_clean_a() -> pd.DataFrame:
    """Cleaned Dataset A from Phase 2 (data/interim/a_clean.csv)."""
    return pd.read_csv(config.CLEAN_A)


def load_clean_s() -> pd.DataFrame:
    """Cleaned Dataset S from Phase 2 (data/interim/s_clean.csv)."""
    return pd.read_csv(config.CLEAN_S)


def load_experiment(name: str) -> pd.DataFrame:
    """One experiment dataset from data/processed/ (see config.EXPERIMENTS)."""
    return pd.read_csv(config.experiment_path(name))
