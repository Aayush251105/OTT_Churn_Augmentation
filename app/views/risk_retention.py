import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn import config, models, risk
from components import charts, store, ui

# Concept groups used only for the "drivers common to both datasets" comparison (presentation, not modelling).
CONCEPTS = {
    "Engagement": {"S": ["viewing_hours_per_week", "avg_viewing_duration", "content_downloads_per_month", "watchlist_size"],
                   "A": ["watch_hours", "last_login_days"]},
    "Tenure": {"S": ["account_age"], "A": []},
    "Plan and price": {"S": ["subscription_type", "monthly_charges"], "A": ["subscription_type"]},
    "Payment": {"S": ["payment_method", "paperless_billing"], "A": ["payment_method"]},
    "Support and satisfaction": {"S": ["support_tickets_per_month", "user_rating"], "A": []},
    "Profile and devices": {"S": ["gender", "device", "favorite_genre", "content_type", "multi_device_access",
                                  "parental_control", "subtitles_enabled"],
                            "A": ["age", "gender", "region", "device", "favorite_genre", "number_of_profiles"]},
}


@st.cache_data(show_spinner="Scoring high-risk customers…")
def high_risk_table(exp: str) -> pd.DataFrame | None:
    """Test customers in the high tier (XGBoost probability), with their strongest SHAP driver."""
    pipe = store.model(exp, "xgboost")
    finals = store.final_models()
    df = store.clean_s() if exp == "S_full" else store.clean_a()
    if pipe is None or finals is None or df is None:
        return None
    feats = finals[exp]["features"]
    cols = [*feats["numeric"], *feats["categorical"], *feats["binary"]]
    _, _, test = models.stratified_split(df, 0)
    proba = pipe.predict_proba(test[cols])[:, 1]
    hi = test.assign(proba=proba)[proba >= config.RISK_BINS[1]].sort_values("proba", ascending=False)
    if hi.empty:
        return hi
    values, Xt = risk.shap_values(pipe, hi[cols])
    contrib = pd.DataFrame(values, columns=Xt.columns)
    contrib = contrib.T.groupby([risk.original_feature(c, feats["categorical"]) for c in contrib.columns]).sum().T
    top = contrib.idxmax(axis=1).to_numpy()
    return pd.DataFrame({"customer_id": hi["customer_id"].to_numpy(), "churn_probability": hi["proba"].to_numpy(),
                         "tier": "high", "top_driver": top, "churned": hi["churned"].map({0: "No", 1: "Yes"}).to_numpy()})


ui.page_header("Risk & retention", "Who is likely to leave, why, and what to offer them")
ds = st.segmented_control("Dataset", ["S", "A"], default="S", format_func=lambda v: f"Dataset {v}", key="rr_ds") or "S"
exp = "S_full" if ds == "S" else "A_full"

profiles, imp, rules = (store.results_csv("risk_profiles.csv"), store.results_csv("feature_importance.csv"),
                        store.results_json("retention_rules.json"))
if profiles is None or imp is None or rules is None:
    ui.empty_state(hint=store.missing_hint("risk_profiles.csv", "feature_importance.csv", "retention_rules.json"))
    st.stop()

# ---------------------------------------------------------------- tiers
prof = profiles[profiles["experiment"] == exp].set_index("tier").reindex(list(config.RISK_LABELS))
overall = (prof["churn_rate"] * prof["n_customers"]).sum() / prof["n_customers"].sum()
hi_lift = prof.loc["high", "churn_rate"] / overall
ui.section("Risk tiers", f"XGBoost probabilities on the test split, cut at {config.RISK_BINS[0]:.0%} and "
           f"{config.RISK_BINS[1]:.0%}. High-risk customers churn at {hi_lift:.1f}× the average.")
labels = {"low": f"below {config.RISK_BINS[0]:.0%} predicted", "medium": f"{config.RISK_BINS[0]:.0%}–{config.RISK_BINS[1]:.0%} predicted",
          "high": f"above {config.RISK_BINS[1]:.0%} predicted"}
for col, tier in zip(st.columns(3, gap="medium"), config.RISK_LABELS):
    r = prof.loc[tier]
    col.html(f'<div class="ss-kpi" style="border-top:2px solid {charts.TIER_COLORS[tier]}">'
             f'<div style="display:flex;justify-content:space-between">{ui.badge(tier)}'
             f'<span class="sub">{labels[tier]}</span></div>'
             f'<div class="lbl" style="margin-top:12px">Actual churn rate</div><div class="val">{r["churn_rate"]:.1%}</div>'
             f'<div class="sub">{ui.num(r["n_customers"])} customers · {r["share"]:.1%} of the test set</div></div>')

# ---------------------------------------------------------------- tier profiles
ui.section("How the tiers differ", "Average per tier. The highest value in each row is highlighted.")
means = prof[[c for c in prof.columns if c.startswith("mean_") and c != "mean_proba"]].T.dropna(how="all")
means.index = [i.removeprefix("mean_").replace("_", " ") for i in means.index]
st.dataframe(means.style.format("{:.2f}").highlight_max(axis=1, props="background-color:#2A2A33;font-weight:600"),
             column_config={t: t.capitalize() for t in config.RISK_LABELS})

# ---------------------------------------------------------------- importance
ui.section("What drives churn", "Two methods. Both are normalised to sum to 1 so they compare on one scale.")
c1, c2 = st.columns(2, gap="medium")
for col, method, title, caption in [
    (c1, "rf", "Random Forest importance", "Mean decrease in impurity, one-hot columns summed per feature"),
    (c2, "shap", "SHAP importance", "Mean |SHAP value| of the XGBoost model"),
]:
    t = imp[(imp["experiment"] == exp) & (imp["method"] == method)].nlargest(10, "importance").iloc[::-1]
    with col.container(border=True):
        if t.empty:
            ui.empty_state(hint="Not computed for this dataset.")
            continue
        ui.chart_title(f"{title}: {t.iloc[-1]['feature'].replace('_', ' ')} matters most", caption)
        fig = go.Figure(go.Bar(x=t["importance"], y=t["feature"], orientation="h",
                               marker_color=[charts.TEAL if i >= len(t) - 3 else charts.NEUTRAL_BAR for i in range(len(t))],
                               text=[f"{v:.3f}" for v in t["importance"]], textposition="outside"))
        fig.update_xaxes(range=[0, t["importance"].max() * 1.25])
        charts.show(fig, 340, key=f"imp_{method}")

# ---------------------------------------------------------------- common drivers
ui.section("Drivers common to both datasets", "Same idea, different columns. Ranked by summed SHAP importance.")
rows = []
for concept, cols in CONCEPTS.items():
    row = {"Concept": concept}
    for d, e in [("S", "S_full"), ("A", "A_full")]:
        t = imp[(imp["experiment"] == e) & (imp["method"] == "shap")].set_index("feature")["importance"]
        row[f"cols_{d}"] = ", ".join(cols[d]) or "–"
        row[f"imp_{d}"] = t.reindex(cols[d]).fillna(0).sum() if cols[d] else np.nan
    rows.append(row)
cd = pd.DataFrame(rows)
for d in ["S", "A"]:
    cd[f"Rank in {d}"] = cd[f"imp_{d}"].rank(ascending=False, method="min").map(lambda v: "–" if pd.isna(v) else f"{v:.0f}")
cd = cd.sort_values("imp_S", ascending=False)
st.dataframe(cd[["Concept", "cols_S", "Rank in S", "cols_A", "Rank in A"]], hide_index=True,
             column_config={"cols_S": "Columns in S", "cols_A": "Columns in A"})

# ---------------------------------------------------------------- rules
ui.section("Retention rules", "Simple triggers built from the drivers above. Associations only: test an offer before rolling it out.")
rt = pd.DataFrame([r for r in rules if r["dataset"] == ds])
st.dataframe(rt[["condition", "suggestion", "segment_churn_rate", "rest_churn_rate", "segment_share", "lift"]]
             .sort_values("lift", ascending=False), hide_index=True,
             column_config={"condition": "Condition", "suggestion": st.column_config.TextColumn("Suggested action", width="large"),
                            "segment_churn_rate": st.column_config.NumberColumn("Churn in segment", format="percent"),
                            "rest_churn_rate": st.column_config.NumberColumn("Churn elsewhere", format="percent"),
                            "segment_share": st.column_config.NumberColumn("Share of customers", format="percent"),
                            "lift": st.column_config.NumberColumn("Lift", format="%.2f×")})

# ---------------------------------------------------------------- high-risk list
ui.section("High-risk customers", f"Test customers with a predicted probability of {config.RISK_BINS[1]:.0%} or more, "
           "and the feature pushing each one hardest towards churn.")
hr = high_risk_table(exp)
if hr is None:
    ui.empty_state("Models not found", "Run `python -m churn.models` to rebuild the saved models.")
elif hr.empty:
    st.caption("No customer reaches the high tier.")
else:
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    q = c1.text_input("Search by customer ID", key="rr_q", placeholder="Type part of an ID")
    view = hr[hr["customer_id"].str.contains(q, case=False)] if q else hr
    c2.download_button("Download CSV", hr.to_csv(index=False), file_name=f"stayscope_high_risk_{exp}.csv",
                       mime="text/csv", width="stretch")
    st.dataframe(view, hide_index=True, height=320, column_config={
        "customer_id": "Customer ID",
        "churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent"),
        "tier": "Tier", "top_driver": "Top driver", "churned": "Actually churned"})
