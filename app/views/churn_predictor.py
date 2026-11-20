import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn import config, models, preprocessing as pp, risk
from components import charts, store, ui

ui.page_header("Churn predictor", "Score one customer, see what drives the score and what to do about it")

xgb = store.model("S_full", "xgboost")
finals, preds = store.final_models(), store.predictions()
if xgb is None or finals is None or preds is None:
    ui.empty_state("Models not found", "Run `python -m churn.models` to rebuild the saved models (about 15 seconds).")
    st.stop()

SPEC = models.SPEC_S
LABELS = {
    "account_age": ("Account age", "months"), "monthly_charges": ("Monthly charges", "$"),
    "viewing_hours_per_week": ("Viewing hours per week", "h"), "avg_viewing_duration": ("Average viewing duration", "min"),
    "content_downloads_per_month": ("Downloads per month", ""), "user_rating": ("User rating", "of 5"),
    "support_tickets_per_month": ("Support tickets per month", ""), "watchlist_size": ("Watchlist size", "titles"),
}
RANGES = {"account_age": (1, 119, 1), "monthly_charges": (4.99, 19.99, 0.5), "viewing_hours_per_week": (1.0, 40.0, 0.5),
          "avg_viewing_duration": (5.0, 180.0, 1.0), "content_downloads_per_month": (0, 49, 1),
          "user_rating": (1.0, 5.0, 0.1), "support_tickets_per_month": (0, 9, 1), "watchlist_size": (0, 24, 1)}
DEFAULT = {"account_age": 4, "monthly_charges": 9.99, "subscription_type": "Basic", "payment_method": "Electronic check",
           "gender": "Female", "viewing_hours_per_week": 3.0, "avg_viewing_duration": 38.0,
           "content_downloads_per_month": 22, "support_tickets_per_month": 6, "user_rating": 2.0, "watchlist_size": 8,
           "device": "Mobile", "favorite_genre": "Comedy", "content_type": "TV Shows", "paperless_billing": 1,
           "multi_device_access": 0, "parental_control": 0, "subtitles_enabled": 1}
CHOICES = {"subscription_type": ["Basic", "Standard", "Premium"],
           "payment_method": ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
           "gender": ["Female", "Male"], "device": ["Computer", "Mobile", "Tablet", "TV"],
           "favorite_genre": ["Action", "Comedy", "Drama", "Fantasy", "Sci-Fi"],
           "content_type": ["Movies", "TV Shows", "Both"]}

if "cp_profile" not in st.session_state:
    st.session_state.cp_profile = dict(DEFAULT)


def score(profile: dict) -> float:
    return float(xgb.predict_proba(pd.DataFrame([profile])[SPEC.columns])[:, 1][0])


left, right = st.columns([5, 7], gap="large")

# ---------------------------------------------------------------- input form
with left.container(border=True):
    st.html('<p class="ss-chart-title">Customer profile</p>')
    p = st.session_state.cp_profile
    with st.form("cp_form", border=False):
        st.caption("Account")
        new = {}
        new["account_age"] = st.slider("Account age (months)", 1, 119, int(p["account_age"]), 1)
        new["monthly_charges"] = st.slider("Monthly charges ($)", 4.99, 19.99, float(p["monthly_charges"]), 0.5)
        new["subscription_type"] = st.segmented_control("Subscription plan", CHOICES["subscription_type"],
                                                        default=p["subscription_type"]) or p["subscription_type"]
        c1, c2 = st.columns(2)
        new["payment_method"] = c1.selectbox("Payment method", CHOICES["payment_method"],
                                             index=CHOICES["payment_method"].index(p["payment_method"]))
        new["gender"] = c2.selectbox("Gender", CHOICES["gender"], index=CHOICES["gender"].index(p["gender"]))
        st.caption("Viewing")
        for f in ["viewing_hours_per_week", "avg_viewing_duration", "content_downloads_per_month", "user_rating", "watchlist_size"]:
            lo, hi, step = RANGES[f]
            name, unit = LABELS[f]
            cast = int if isinstance(step, int) else float
            new[f] = st.slider(f"{name} ({unit})" if unit else name, lo, hi, cast(p[f]), step)
        new["support_tickets_per_month"] = st.number_input("Support tickets per month", 0, 9, int(p["support_tickets_per_month"]))
        c1, c2, c3 = st.columns(3)
        for col, f in zip([c1, c2, c3], ["device", "favorite_genre", "content_type"]):
            new[f] = col.selectbox(f.replace("_", " ").capitalize(), CHOICES[f], index=CHOICES[f].index(p[f]))
        st.caption("Settings")
        c1, c2 = st.columns(2)
        for i, f in enumerate(pp.S_FEATURES_BINARY):
            new[f] = int((c1 if i % 2 == 0 else c2).toggle(f.replace("_", " ").capitalize(), value=bool(p[f])))
        b1, b2 = st.columns([3, 1])
        submitted = b1.form_submit_button("Predict churn", type="primary", width="stretch")
        reset = b2.form_submit_button("Reset", width="stretch")
    if submitted:
        st.session_state.cp_profile = new
        st.rerun()
    if reset:
        st.session_state.cp_profile = dict(DEFAULT)
        st.rerun()

profile = st.session_state.cp_profile
proba = score(profile)
tier = risk.assign_tier(proba)
base = preds.loc[(preds["experiment"] == "S_full") & (preds["model"] == "xgboost"), "y_true"].mean()

# ---------------------------------------------------------------- result
with right:
    with st.container(border=True):
        st.html(f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <div><div class="ss-chart-caption">Churn probability · XGBoost</div>
               <div class="ss-big">{proba:.0%}</div>
               <div class="ss-chart-caption">likely to cancel · {proba / base:.1f}× the base rate of {base:.1%}</div></div>
          <div>{ui.badge(tier)}</div>
        </div>
        <div class="ss-riskbar"><div class="m" style="left:calc({proba * 100:.1f}% - 1px)"></div></div>
        <div class="ss-riskscale"><span>0%</span><span>Low &lt; 30%</span><span>Medium 30–60%</span><span>High &gt; 60%</span><span>100%</span></div>""")

    with st.container(border=True):
        values, Xt = risk.shap_values(xgb, pd.DataFrame([profile])[SPEC.columns])
        contrib = pd.Series(values[0], index=Xt.columns)
        contrib = contrib.groupby([risk.original_feature(c, SPEC.categorical) for c in contrib.index]).sum()
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:6])[::-1]

        def label(f):
            v = profile[f]
            unit = LABELS.get(f, (None, ""))[1]
            return f"{LABELS.get(f, (f.replace('_', ' ').capitalize(),))[0]} = {v:g}{(' ' + unit) if unit else ''}" \
                if isinstance(v, (int, float)) and f not in pp.S_FEATURES_BINARY else f"{f.replace('_', ' ').capitalize()} = {v}"

        lead = [LABELS.get(f, (f,))[0].lower() for f in top.index[::-1][:2] if top[f] > 0]
        ui.chart_title(f"{' and '.join(lead).capitalize()} push this customer towards churn" if lead
                       else "Nothing pushes this customer strongly towards churn",
                       "Contribution to the score (SHAP, log-odds) · red pushes towards churn, teal away")
        fig = go.Figure(go.Bar(x=top.values, y=[label(f) for f in top.index], orientation="h",
                               marker_color=[charts.RED if v > 0 else charts.TEAL for v in top.values],
                               text=[f"{'▲' if v > 0 else '▼'} {v:+.2f}" for v in top.values], textposition="outside",
                               hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        span = top.abs().max() * 1.5
        fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor=charts.BORDER)
        charts.show(fig, 260)

    with st.container(border=True):
        st.html('<p class="ss-chart-title">Suggested actions</p>')
        actions = risk.retention_suggestions({**profile, "proba": proba}, dataset="S")
        if actions:
            ui.numbered_list([(a, "", "") for a in actions[:4]])
        else:
            st.caption("No retention rule fires for this profile. Keep in the standard engagement programme.")

# ---------------------------------------------------------------- other models
ui.section("Do the other models agree?", "Their outputs are class-weighted scores, not probabilities, so they are "
           "compared by rank: how this customer ranks among the 36,569 test customers.")
rows = []
for mdl in config.MODELS:
    pipe = store.model("S_full", mdl)
    if pipe is None:
        continue
    s_ = float(pipe.predict_proba(pd.DataFrame([profile])[SPEC.columns])[:, 1][0])
    dist = preds.loc[(preds["experiment"] == "S_full") & (preds["model"] == mdl), "proba"].to_numpy()
    rows.append({"Model": charts.MODEL_NAMES[mdl], "Score": round(s_, 3),
                 "Class weighting": finals["S_full"]["models"][mdl]["class_weight"],
                 "Riskier than": f"{(dist < s_).mean():.0%} of customers",
                 "Flagged at tuned threshold": "yes" if s_ >= finals["S_full"]["models"][mdl]["threshold"] else "no"})
st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---------------------------------------------------------------- what if
ui.section("What if", "Change one feature and keep the rest of the profile fixed.")
with st.container(border=True):
    c1, c2, c3 = st.columns([1.2, 2, 1.4], vertical_alignment="bottom")
    feat = c1.selectbox("Feature", list(RANGES), format_func=lambda f: LABELS[f][0], key="cp_wi_feat")
    lo, hi, step = RANGES[feat]
    cast = int if isinstance(step, int) else float
    default_new = cast(min(hi, max(lo, profile[feat] * 4 if feat == "account_age" else profile[feat])))
    val = c2.slider(f"New value ({LABELS[feat][1]})" if LABELS[feat][1] else "New value", lo, hi, default_new, step,
                    key=f"cp_wi_val_{feat}")
    new_p = score({**profile, feat: val})
    delta = new_p - proba
    new_tier = risk.assign_tier(new_p)
    c3.html(ui.kpi(f"{LABELS[feat][0]}: {profile[feat]:g} → {val:g}", f"{proba:.0%} → {new_p:.0%}", small=True,
                   delta=f"{'▼' if delta < 0 else '▲'} {abs(delta) * 100:.0f} points"
                         + (f" · moves to {new_tier.capitalize()}" if new_tier != tier else ""),
                   tone="down" if delta < 0 else "up" if delta > 0 else "flat"))
