"""Phase 3: feature mapping between Dataset A (augmentation target) and Dataset S (donor).

A and S share five concepts (plan, device, genre, payment method, gender), but they code them differently.
`harmonize` adds one harmonized column per shared concept to both datasets, using the same labels on both
sides, so the two can be compared and matched (Phase 4). The original columns are kept unchanged, because the
A models still use A's own, more detailed categories.

Mapping decisions (see notebook 03):
  * subscription_type: identical labels (Basic / Standard / Premium)
  * device: A's Laptop and Desktop -> Computer (S's label)
  * favorite_genre: the 4 shared genres are kept; A's Horror/Romance/Documentary and S's Fantasy -> Other
  * payment_method: only "credit card" exists in both, so methods are grouped by how the customer pays:
      card    = credit / debit card                      (A: Credit Card, Debit Card; S: Credit card)
      digital = online, non-card transfer               (A: PayPal, Crypto;          S: Electronic check, Bank transfer)
      manual  = prepaid or paper                        (A: Gift Card;               S: Mailed check)
  * gender: Male / Female kept; A's "Other" has no S counterpart and stays "Other"
  * monthly_fee vs monthly_charges are NOT matched: A's fee is fixed by plan, S's charge is independent of plan

Transfer features are S's behavioural columns that A_hard lacks and that carry churn signal in S
(|Spearman rho| >= 0.05 with churn, measured in notebook 03).
"""

import pandas as pd

from churn import config

ID, TARGET = config.ID_COL, config.TARGET

# ---------------------------------------------------------------- mappings (raw label -> harmonized label)
DEVICE_MAP = {
    "TV": "TV", "Mobile": "Mobile", "Tablet": "Tablet",
    "Laptop": "Computer", "Desktop": "Computer", "Computer": "Computer",
}
SHARED_GENRES = ["Action", "Comedy", "Drama", "Sci-Fi"]
PAYMENT_MAP = {
    # Dataset A
    "Credit Card": "card", "Debit Card": "card",
    "PayPal": "digital", "Crypto": "digital",
    "Gift Card": "manual",
    # Dataset S
    "Credit card": "card",
    "Electronic check": "digital", "Bank transfer": "digital",
    "Mailed check": "manual",
}
GENDER_MAP = {"Male": "Male", "Female": "Female", "Other": "Other"}

# ---------------------------------------------------------------- feature lists
# Harmonized shared columns (present in both a_h and s_h). Used for matching, never the target.
COMMON_FEATURES = ["plan_h", "device_h", "genre_h", "payment_h", "gender_h"]

# S behavioural columns that A does not have: the candidates checked in the transferability pre-check.
# monthly_charges is left out on purpose: it is pricing, not behaviour, and A's fee means something else.
TRANSFER_CANDIDATES = [
    "account_age", "viewing_hours_per_week", "avg_viewing_duration", "content_downloads_per_month",
    "user_rating", "support_tickets_per_month", "watchlist_size",
]
# Candidates with churn signal in S (|rho| >= 0.05). These are the columns copied into A_hard_augmented.
TRANSFER_FEATURES = [
    "account_age", "avg_viewing_duration", "content_downloads_per_month",
    "viewing_hours_per_week", "support_tickets_per_month",
]
MIN_CHURN_ASSOCIATION = 0.05


def _map(series: pd.Series, mapping: dict, name: str) -> pd.Series:
    """Apply a label mapping and fail loudly if a label has no mapping."""
    out = series.map(mapping)
    unmapped = sorted(series[out.isna()].unique())
    if unmapped:
        raise ValueError(f"No harmonized label for {name} values {unmapped}")
    return out


def harmonize_one(df: pd.DataFrame) -> pd.DataFrame:
    """Add the harmonized shared columns (COMMON_FEATURES) to one cleaned dataset (A or S)."""
    out = df.copy()
    out["plan_h"] = out["subscription_type"]
    out["device_h"] = _map(out["device"], DEVICE_MAP, "device")
    out["genre_h"] = out["favorite_genre"].where(out["favorite_genre"].isin(SHARED_GENRES), "Other")
    out["payment_h"] = _map(out["payment_method"], PAYMENT_MAP, "payment_method")
    out["gender_h"] = _map(out["gender"], GENDER_MAP, "gender")
    return out


def harmonize(a: pd.DataFrame, s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonize cleaned A and cleaned S. Returns (a_h, s_h): the inputs plus the COMMON_FEATURES columns."""
    return harmonize_one(a), harmonize_one(s)


def mapping_table() -> pd.DataFrame:
    """Every raw label of the shared columns and the harmonized label it maps to (for the notebook and app)."""
    rows = []
    for feature, h_col, mapping in [
        ("device", "device_h", DEVICE_MAP),
        ("payment_method", "payment_h", PAYMENT_MAP),
        ("gender", "gender_h", GENDER_MAP),
    ]:
        rows += [{"feature": feature, "harmonized": h_col, "raw_label": k, "harmonized_label": v}
                 for k, v in mapping.items()]
    for g in ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Documentary", "Fantasy"]:
        rows.append({"feature": "favorite_genre", "harmonized": "genre_h", "raw_label": g,
                     "harmonized_label": g if g in SHARED_GENRES else "Other"})
    for p in ["Basic", "Standard", "Premium"]:
        rows.append({"feature": "subscription_type", "harmonized": "plan_h", "raw_label": p, "harmonized_label": p})
    return pd.DataFrame(rows)


def category_shares(a_h: pd.DataFrame, s_h: pd.DataFrame, cols=COMMON_FEATURES) -> pd.DataFrame:
    """Share of customers in each harmonized category, side by side for A and S."""
    rows = []
    for c in cols:
        for name, df in [("A", a_h), ("S", s_h)]:
            share = df[c].value_counts(normalize=True)
            rows += [{"feature": c, "category": k, "dataset": name, "share": v} for k, v in share.items()]
    out = pd.DataFrame(rows).pivot_table(index=["feature", "category"], columns="dataset", values="share")
    return out.fillna(0.0).reset_index()


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio eta (0..1): how much of a numeric column's variance is explained by a categorical one.
    eta^2 is the R^2 of predicting the numeric column from the category means."""
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    g = values.groupby(categories)
    between = (g.size() * (g.mean() - values.mean()) ** 2).sum()
    return float((between / total) ** 0.5)


def shared_vs_behaviour(s_h: pd.DataFrame, common=COMMON_FEATURES, targets=TRANSFER_CANDIDATES) -> pd.DataFrame:
    """Correlation ratio between every shared column (rows) and every behavioural column of S (columns).
    Values near 0 mean matching on the shared columns cannot tell us anything about behaviour."""
    return pd.DataFrame({t: [correlation_ratio(s_h[c], s_h[t]) for c in common] for t in targets},
                        index=list(common))


def churn_by_harmonized(a_h: pd.DataFrame, s_h: pd.DataFrame, cols=COMMON_FEATURES) -> pd.DataFrame:
    """Churn rate per harmonized category in A and in S. If a shared column means the same thing in both
    datasets, its categories should rank churn in the same order."""
    rows = []
    for c in cols:
        for name, df in [("A", a_h), ("S", s_h)]:
            g = df.groupby(c)[TARGET].agg(["mean", "size"])
            rows += [{"feature": c, "category": k, "dataset": name, "churn_rate": r["mean"], "customers": int(r["size"])}
                     for k, r in g.iterrows()]
    return pd.DataFrame(rows)
