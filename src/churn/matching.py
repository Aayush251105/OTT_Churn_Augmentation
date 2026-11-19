"""Phase 4: augmentation study. Pre-check, cosine-KNN matching of A customers to S customers, feature transfer
and the random control.

Method (CLAUDE.md section 3):
  1. Pre-check (`transferability_check`): can the shared columns predict the features we want to transfer?
     If the cross-validated R^2 is about 0, similarity on the shared columns says nothing about behaviour and the
     transferred values are expected to be noise.
  2. Matching (`find_neighbours`): the harmonized shared columns are one-hot encoded and standardized (scaler
     fitted on S, the donor), then each A customer gets its K most cosine-similar S customers. Neighbours with
     similarity below SIM_THRESHOLD are discarded; A customers with no valid neighbour are dropped.
  3. Transfer (`augment`, mode="knn"): similarity-weighted mean of each numeric transfer feature, and
     similarity-weighted majority for categorical ones.
  4. Control (`augment`, mode="random"): the same A rows, each given the same number of S donors, but drawn at
     random and weighted equally. If knn does not beat random, the matching adds nothing.

The churn label is never part of the matching vector, and S's churn label is never transferred.
S is shuffled once (seeded) before matching, so ties between identical S profiles are broken at random rather
than by file order.

Run `python -m churn.matching` to write data/processed/A_hard*.csv, reports/results/transferability.csv and
reports/results/match_report.csv.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config, features, preprocessing as pp

ID, TARGET = config.ID_COL, config.TARGET

# A_hard = cleaned A without the engagement columns
A_HARD_NUMERIC = [c for c in pp.A_FEATURES_NUMERIC if c not in config.A_ENGAGEMENT]
A_HARD_CATEGORICAL = [c for c in pp.A_FEATURES_CATEGORICAL if c not in config.A_ENGAGEMENT]
A_HARD_COLUMNS = [ID, *A_HARD_NUMERIC, *A_HARD_CATEGORICAL, TARGET]


# ---------------------------------------------------------------- 1. pre-check
def transferability_check(s_h: pd.DataFrame, common=features.COMMON_FEATURES,
                          targets=features.TRANSFER_CANDIDATES, n_splits: int = 5,
                          seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Cross-validated R^2 of predicting each target column of S from the shared columns only.

    Uses a gradient-boosting regressor on one-hot shared columns, so non-linear and interaction effects count.
    Returns one row per target: feature, r2_mean, r2_std (R^2 ~ 0 means the shared columns carry no
    information about that feature, so similarity-based transfer cannot recover it).
    """
    X = pd.get_dummies(s_h[list(common)], dtype=float)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for t in targets:
        model = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
        r2 = cross_val_score(model, X, s_h[t], cv=cv, scoring="r2")
        rows.append({"feature": t, "r2_mean": float(r2.mean()), "r2_std": float(r2.std())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. matching
def _shuffle_s(s_h: pd.DataFrame, seed: int) -> pd.DataFrame:
    """S in a seeded random order (random tie-breaking between identical profiles)."""
    return s_h.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def encode_common(a_h: pd.DataFrame, s_h: pd.DataFrame, common=features.COMMON_FEATURES):
    """One-hot encode + standardize the shared columns. Encoder categories cover both datasets (so A's gender
    "Other" gets its own column); the scaler is fitted on S, the donor population.
    Returns (X_a, X_s, column_names)."""
    common = list(common)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=float)
    enc.fit(pd.concat([a_h[common], s_h[common]], ignore_index=True))
    oa, os_ = enc.transform(a_h[common]), enc.transform(s_h[common])
    scaler = StandardScaler().fit(os_)
    scaler.scale_[scaler.scale_ == 0] = 1.0  # a category absent from S keeps its raw 0/1 value
    return scaler.transform(oa), scaler.transform(os_), list(enc.get_feature_names_out(common))


def find_neighbours(a_h: pd.DataFrame, s_h: pd.DataFrame, k: int = config.K,
                    common=features.COMMON_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """The k most cosine-similar S rows for every A row.

    Returns (indices into s_h, similarities), both shaped (len(a_h), k), most similar first.
    Pass an already shuffled s_h (see `augment`) for random tie-breaking.
    """
    X_a, X_s, _ = encode_common(a_h, s_h, common)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X_s)
    dist, idx = nn.kneighbors(X_a)
    return idx, 1.0 - dist


def match_report(a_h: pd.DataFrame, sim: np.ndarray, threshold: float = config.SIM_THRESHOLD) -> pd.DataFrame:
    """One row per A customer: how many neighbours pass the threshold, best and mean similarity, kept or not."""
    valid = sim >= threshold
    n_valid = valid.sum(axis=1)
    mean_sim = np.where(n_valid > 0, (sim * valid).sum(axis=1) / np.maximum(n_valid, 1), np.nan)
    return pd.DataFrame({
        ID: a_h[ID].to_numpy(),
        "n_valid_neighbours": n_valid,
        "max_sim": sim.max(axis=1),
        "mean_sim": mean_sim,
        "kept": n_valid > 0,
    })


# ---------------------------------------------------------------- 3. transfer
def _weighted_transfer(donors: pd.DataFrame, idx: np.ndarray, weights: np.ndarray, cols) -> pd.DataFrame:
    """Weighted mean (numeric) or weighted majority (other dtypes) of donor rows, one output row per row of idx.
    Weights of 0 switch a donor off."""
    out = {}
    w_sum = weights.sum(axis=1)
    for c in cols:
        values = donors[c].to_numpy()[idx]  # (n, k)
        if pd.api.types.is_numeric_dtype(donors[c]):
            out[c] = (values * weights).sum(axis=1) / w_sum
        else:
            votes = []
            for v_row, w_row in zip(values, weights):
                tally = pd.Series(w_row).groupby(v_row).sum()
                votes.append(tally.idxmax())
            out[c] = votes
    return pd.DataFrame(out)


def augment(a_h: pd.DataFrame, s_h: pd.DataFrame, k: int = config.K, threshold: float = config.SIM_THRESHOLD,
            mode: str = "knn", seed: int = config.RANDOM_STATE, transfer=features.TRANSFER_FEATURES,
            common=features.COMMON_FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an augmented A_hard dataset.

    mode="knn":    transferred values are the similarity-weighted aggregate of the valid neighbours.
    mode="random": the same kept A rows (decided by the KNN match), each given as many random S donors as it
                   had valid neighbours, weighted equally. This is the control.

    Returns (augmented_a, match_report). augmented_a has only the kept rows and the columns
    customer_id, A_hard features, transferred features, churned. match_report covers every A row.
    """
    if mode not in ("knn", "random"):
        raise ValueError("mode must be 'knn' or 'random'")
    transfer = list(transfer)
    if TARGET in transfer or TARGET in common:
        raise ValueError("The churn label must never be matched on or transferred")

    s_shuf = _shuffle_s(s_h, seed)
    idx, sim = find_neighbours(a_h, s_shuf, k, common)
    report = match_report(a_h, sim, threshold)
    kept = report["kept"].to_numpy()
    valid = (sim >= threshold)[kept]

    if mode == "knn":
        donor_idx, weights = idx[kept], np.where(valid, sim[kept], 0.0)
    else:
        rng = np.random.default_rng(seed)
        donor_idx = rng.integers(0, len(s_shuf), size=valid.shape)
        weights = valid.astype(float)  # same number of donors per row as the KNN match, equal weights

    base = a_h.loc[kept, A_HARD_COLUMNS].reset_index(drop=True)
    moved = _weighted_transfer(s_shuf, donor_idx, weights, transfer)
    out = pd.concat([base.drop(columns=TARGET), moved, base[[TARGET]]], axis=1)
    return out, report


def neighbours_for(customer_id, a_h: pd.DataFrame, s_h: pd.DataFrame, k: int = config.K,
                   threshold: float = config.SIM_THRESHOLD, seed: int = config.RANDOM_STATE,
                   common=features.COMMON_FEATURES, transfer=features.TRANSFER_FEATURES) -> pd.DataFrame:
    """The k S neighbours of one A customer (for the app and the notebook): rank, S customer_id, similarity,
    whether it passes the threshold, its shared columns and the values it would donate."""
    row = a_h[a_h[ID] == customer_id]
    if row.empty:
        raise KeyError(f"customer_id {customer_id!r} not found in A")
    s_shuf = _shuffle_s(s_h, seed)
    X_a, X_s, _ = encode_common(row, s_shuf, common)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X_s)
    dist, idx = nn.kneighbors(X_a)
    nb = s_shuf.iloc[idx[0]][[ID, *common, *transfer]].reset_index(drop=True)
    nb.insert(0, "rank", np.arange(1, k + 1))
    nb.insert(2, "similarity", 1.0 - dist[0])
    nb.insert(3, "valid", nb["similarity"] >= threshold)
    return nb


# ---------------------------------------------------------------- outputs
def build_experiments(a_h: pd.DataFrame, s_h: pd.DataFrame, seed: int = config.RANDOM_STATE) -> dict:
    """The three A_hard experiment datasets (same rows) plus the match report."""
    augmented, report = augment(a_h, s_h, mode="knn", seed=seed)
    random_ctrl, _ = augment(a_h, s_h, mode="random", seed=seed)
    kept_ids = report.loc[report["kept"], ID]
    a_hard = a_h.loc[a_h[ID].isin(kept_ids), A_HARD_COLUMNS].reset_index(drop=True)
    return {"A_hard": a_hard, "A_hard_augmented": augmented, "A_hard_random": random_ctrl, "match_report": report}


def run() -> dict:
    """Harmonize, run the pre-check, match, transfer, and write every Phase 4 output."""
    from churn import data

    config.ensure_dirs()
    a_h, s_h = features.harmonize(data.load_clean_a(), data.load_clean_s())
    check = transferability_check(s_h)
    check.to_csv(config.RESULTS_DIR / "transferability.csv", index=False)
    out = build_experiments(a_h, s_h)
    out["match_report"].to_csv(config.RESULTS_DIR / "match_report.csv", index=False)
    for name in ("A_hard", "A_hard_augmented", "A_hard_random"):
        out[name].to_csv(config.experiment_path(name), index=False)
    out["transferability"] = check
    return out


if __name__ == "__main__":
    res = run()
    print(res["transferability"].round(4).to_string(index=False))
    rep = res["match_report"]
    print(f"kept {rep['kept'].sum()} / {len(rep)} A customers")
    for name in ("A_hard", "A_hard_augmented", "A_hard_random"):
        print(f"{name}: {res[name].shape} -> {config.experiment_path(name)}")
