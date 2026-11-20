"""Phase 6: interpretation, risk tiers and retention suggestions.

  * Feature importance: Random Forest impurity importance and mean |SHAP| of the XGBoost model (TreeSHAP is
    exact and fast for XGBoost; for the 300-tree, depth-16 Random Forest it would take hours on S). One-hot
    columns are summed back to their original feature, and each method is normalised to sum to 1, so S and A
    can be compared on the same scale.
  * Risk tiers: churn probability -> low / medium / high using config.RISK_BINS. Tiers use the XGBoost model's
    probabilities (RISK_MODEL). On S it is the final model trained without class weighting, so its
    probabilities are on the real scale (mean 0.18 = the churn rate); the balanced models' probabilities are
    shifted upwards and would put most customers in the medium tier.
  * Retention rules: segments with clearly higher churn, each with a suggested action. The rationale gives the
    observed churn rates. These are associations, not proven effects: an offer should be A/B tested first.

Run `python -m churn.risk` to write reports/results/feature_importance.csv, risk_profiles.csv and
retention_rules.json (it trains the final models first if models/ is empty).
"""

import json

import numpy as np
import pandas as pd

from churn import config

ID, TARGET = config.ID_COL, config.TARGET
RISK_MODEL = "xgboost"
SHAP_SAMPLE = 2000  # rows explained per experiment

# ---------------------------------------------------------------- risk tiers
def assign_tier(proba: float) -> str:
    """Risk tier for one churn probability: low < 0.3 <= medium < 0.6 <= high (config.RISK_BINS)."""
    lo, hi = config.RISK_BINS
    low, medium, high = config.RISK_LABELS
    return low if proba < lo else medium if proba < hi else high


def assign_tiers(proba) -> pd.Series:
    """Vectorised `assign_tier` for an array of probabilities (ordered categorical)."""
    lo, hi = config.RISK_BINS
    return pd.cut(pd.Series(np.asarray(proba, dtype=float)), [-np.inf, lo, hi, np.inf], right=False,
                  labels=list(config.RISK_LABELS))


def risk_profiles(df: pd.DataFrame, experiment: str, numeric) -> pd.DataFrame:
    """One row per tier: customer count and share, mean predicted and observed churn, and the mean of each
    numeric feature. `df` needs a `proba` column, the target and the numeric columns."""
    d = df.assign(tier=pd.Categorical(assign_tiers(df["proba"]).astype(str).to_numpy(),
                                      categories=list(config.RISK_LABELS), ordered=True))
    g = d.groupby("tier", observed=False)
    out = pd.DataFrame({
        "experiment": experiment,
        "n_customers": g.size(),
        "share": g.size() / len(d),
        "mean_proba": g["proba"].mean(),
        "churn_rate": g[TARGET].mean(),
    })
    out = out.join(g[list(numeric)].mean().add_prefix("mean_"))
    return out.reset_index().rename(columns={"index": "tier"})[["experiment", "tier", *out.columns.drop("experiment")]]


# ---------------------------------------------------------------- retention rules
# (dataset, condition evaluated with DataFrame.eval, suggested action)
RULE_DEFINITIONS = [
    ("S", "account_age < 12",
     "First-year onboarding: a welcome series and a discounted month at the 6- and 11-month marks."),
    ("S", "avg_viewing_duration < 30",
     "Short sessions: surface personalised picks and shorter formats on the home screen."),
    ("S", "viewing_hours_per_week < 5",
     "Low viewing: send a re-engagement digest with new releases in the customer's favourite genre."),
    ("S", "content_downloads_per_month < 5",
     "Few downloads: promote offline viewing (download reminders before commutes and travel)."),
    ("S", "support_tickets_per_month >= 7",
     "Many support tickets: proactive follow-up from the support team and a goodwill credit."),
    ("S", "monthly_charges >= 17",
     "High bill: offer a loyalty discount or a cheaper plan instead of losing the customer."),
    ("A", "last_login_days > 30",
     "Inactive for over a month: win-back email with a 'continue watching' list and a limited-time offer."),
    ("A", "watch_hours < 5",
     "Very low watch time: guided content discovery (short quiz, curated starter list)."),
    ("A", "subscription_type == 'Basic'",
     "Basic plan: free trial of Standard features to show the value of a higher plan."),
    ("A", "payment_method in ['Crypto', 'Gift Card']",
     "Prepaid / non-recurring payment: nudge to switch to an auto-renewing card before the balance runs out."),
]


def _rule_mask(df: pd.DataFrame, condition: str) -> pd.Series:
    return df.eval(condition, engine="python").astype(bool)


def build_retention_rules(s: pd.DataFrame, a: pd.DataFrame) -> list[dict]:
    """Retention rules with a data-based rationale (segment churn vs the rest) computed on cleaned S and A."""
    sources = {"S": s, "A": a}
    rules = []
    for dataset, condition, suggestion in RULE_DEFINITIONS:
        df = sources[dataset]
        mask = _rule_mask(df, condition)
        seg, rest = df.loc[mask, TARGET].mean(), df.loc[~mask, TARGET].mean()
        rules.append({
            "dataset": dataset,
            "condition": condition,
            "suggestion": suggestion,
            "rationale": (f"In dataset {dataset}, {seg:.0%} of these customers churned vs {rest:.0%} of the rest "
                          f"({int(mask.sum()):,} customers, {mask.mean():.0%} of the dataset). "
                          "Association only: test the offer before rolling it out."),
            "segment_churn_rate": round(float(seg), 4),
            "rest_churn_rate": round(float(rest), 4),
            "segment_share": round(float(mask.mean()), 4),
            "lift": round(float(seg / rest), 2) if rest > 0 else None,
        })
    return rules


def load_retention_rules() -> list[dict]:
    """The saved reports/results/retention_rules.json (falls back to the rule definitions without rationale)."""
    path = config.RESULTS_DIR / "retention_rules.json"
    if path.exists():
        return json.loads(path.read_text())
    return [{"dataset": d, "condition": c, "suggestion": s, "rationale": ""} for d, c, s in RULE_DEFINITIONS]


def infer_dataset(row) -> str | None:
    """'S' or 'A' from the columns a customer record has (None if it cannot tell)."""
    keys = set(dict(row))
    if "account_age" in keys:
        return "S"
    if keys & {"age", "watch_hours", "last_login_days", "number_of_profiles"}:
        return "A"
    return None


def retention_suggestions(row, rules: list[dict] | None = None, dataset: str | None = None) -> list[str]:
    """Suggested actions for one customer (dict or Series).

    Only the rules of the customer's own dataset are used (`dataset` "S" or "A", inferred from the columns if not
    given), because a segment's churn pattern in one dataset does not carry over to the other. Rules whose
    columns are missing (e.g. engagement rules for an A_hard customer) are skipped. If the row has a `proba`,
    a high-risk customer gets a prioritisation note first."""
    rules = rules if rules is not None else load_retention_rules()
    dataset = dataset or infer_dataset(row)
    one = pd.DataFrame([dict(row)])
    out = []
    if "proba" in one and assign_tier(float(one["proba"].iloc[0])) == config.RISK_LABELS[-1]:
        out.append("High churn risk: prioritise this customer for a retention offer.")
    for r in rules:
        if dataset and r.get("dataset", dataset) != dataset:
            continue
        try:
            if bool(_rule_mask(one, r["condition"]).iloc[0]):
                out.append(r["suggestion"])
        except (KeyError, NameError, pd.errors.UndefinedVariableError):
            continue  # the rule uses a column this customer's dataset does not have
    return out


# ---------------------------------------------------------------- feature importance
def original_feature(column: str, categorical) -> str:
    """Map a preprocessed column name (e.g. 'device_TV') back to its source feature ('device')."""
    for c in sorted(categorical, key=len, reverse=True):
        if column.startswith(f"{c}_"):
            return c
    return column


def _aggregate(values: pd.Series, categorical) -> pd.Series:
    """Sum per-column importance back to original features and normalise to sum 1."""
    agg = values.groupby([original_feature(c, categorical) for c in values.index]).sum()
    return (agg / agg.sum()).sort_values(ascending=False)


def rf_importance(pipe, categorical) -> pd.Series:
    """Impurity importance of a fitted Random Forest pipeline, per original feature (sums to 1)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    return _aggregate(pd.Series(pipe.named_steps["model"].feature_importances_, index=names), categorical)


def shap_values(pipe, X: pd.DataFrame):
    """TreeSHAP values (log-odds) of a fitted XGBoost pipeline for rows X. Returns (values, transformed X)."""
    import shap

    Xt = pipe.named_steps["prep"].transform(X)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return explainer.shap_values(Xt), Xt


def shap_importance(pipe, X: pd.DataFrame, categorical) -> pd.Series:
    """Mean |SHAP| per original feature (sums to 1)."""
    values, Xt = shap_values(pipe, X)
    return _aggregate(pd.Series(np.abs(values).mean(axis=0), index=Xt.columns), categorical)


# ---------------------------------------------------------------- runner
def experiment_data(experiment: str) -> pd.DataFrame:
    """The dataset behind an experiment (cleaned S / A, or a processed A_hard file)."""
    from churn import data

    if experiment == "S_full":
        return data.load_clean_s()
    if experiment == "A_full":
        return data.load_clean_a()
    return data.load_experiment(experiment)


def load_or_fit(experiment: str, df: pd.DataFrame) -> dict:
    """The saved final models of an experiment, trained (and saved) first if any file is missing."""
    from churn import models

    names = models.load_final_config()[experiment]["models"]
    if all(config.model_path(experiment, m).exists() for m in names):
        return {m: models.load_model(experiment, m) for m in names}
    return models.fit_final_models(experiment, df)


def interpret_experiment(experiment: str, seed: int = config.RANDOM_STATE) -> dict:
    """Feature importance (rf + shap) and risk profiles for one experiment that has final models."""
    from churn import models

    cfg = models.load_final_config()[experiment]
    feats = cfg["features"]
    df = experiment_data(experiment)
    fitted = load_or_fit(experiment, df)
    _, _, test = models.stratified_split(df, 0)  # the same seed-0 split the final models were trained on
    cols = [*feats["numeric"], *feats["categorical"], *feats["binary"]]

    imp = []
    if "random_forest" in fitted:
        imp.append(rf_importance(fitted["random_forest"], feats["categorical"]).rename("importance")
                   .rename_axis("feature").reset_index().assign(method="rf"))
    if "xgboost" in fitted:
        sample = test.sample(min(SHAP_SAMPLE, len(test)), random_state=seed)
        imp.append(shap_importance(fitted["xgboost"], sample[cols], feats["categorical"]).rename("importance")
                   .rename_axis("feature").reset_index().assign(method="shap"))
    importance = pd.concat(imp, ignore_index=True).assign(experiment=experiment)
    importance = importance[["experiment", "method", "feature", "importance"]]

    scored = test.assign(proba=fitted[RISK_MODEL].predict_proba(test[cols])[:, 1])
    profiles = risk_profiles(scored, experiment, feats["numeric"])
    return {"importance": importance, "profiles": profiles, "scored": scored, "models": fitted}


def run(experiments=None) -> dict:
    """Interpret every experiment in final_models.json (or the given ones) and write the Phase 6 outputs."""
    from churn import data, models

    config.ensure_dirs()
    experiments = experiments or list(models.load_final_config())
    results = {e: interpret_experiment(e) for e in experiments}
    importance = pd.concat([r["importance"] for r in results.values()], ignore_index=True)
    profiles = pd.concat([r["profiles"] for r in results.values()], ignore_index=True)
    rules = build_retention_rules(data.load_clean_s(), data.load_clean_a())
    importance.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
    profiles.to_csv(config.RESULTS_DIR / "risk_profiles.csv", index=False)
    (config.RESULTS_DIR / "retention_rules.json").write_text(json.dumps(rules, indent=2))
    return {"results": results, "importance": importance, "profiles": profiles, "rules": rules}


if __name__ == "__main__":
    out = run()
    print(out["importance"].round(3).to_string(index=False))
    print(out["profiles"].round(3).to_string(index=False))
    print(f"{len(out['rules'])} retention rules written")
