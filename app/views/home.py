import plotly.graph_objects as go
import streamlit as st

from components import charts, store, ui

ui.page_header("StayScope", "Churn prediction, drivers and retention for streaming services")

s = store.clean_s()
m = store.test_metrics()
if s is None:
    ui.empty_state(hint="Run `python -m churn.preprocessing` to create the cleaned datasets.")
    st.stop()

# ---------------------------------------------------------------- KPI row
n, churned = len(s), int(s["churned"].sum())
best, best_auc = store.best_model("S_full")
aug_delta = None
if m is not None and {"A_hard", "A_hard_augmented"} <= set(m["experiment"]):
    g = m.groupby(["experiment", "model"])["roc_auc"].mean().unstack(0)
    aug_delta = float((g["A_hard_augmented"] - g["A_hard"]).mean())

cards = [
    ui.kpi("Customers analysed", ui.num(n), sub="Dataset S, streaming service"),
    ui.kpi("Churn rate", ui.pct(churned / n), sub=f"{ui.num(churned)} cancelled, roughly 1 in 5"),
    ui.kpi("Best model", charts.MODEL_NAMES.get(best, "–"), small=True,
           sub=f"ROC-AUC {best_auc:.3f} on the S test set" if best else "run notebook 05"),
    ui.kpi("Augmentation effect", "No gain" if aug_delta is not None and abs(aug_delta) < 0.01 else
           (ui.signed(aug_delta) if aug_delta is not None else "–"), small=True,
           sub=f"ROC-AUC vs A_hard: {ui.signed(aug_delta)}" if aug_delta is not None else "run notebooks 04–05"),
]
ui.kpi_row(cards)

# ---------------------------------------------------------------- models + augmentation
left, right = st.columns(2, gap="medium")
with left.container(border=True):
    ui.chart_title("All four models beat chance by a similar margin",
                   "ROC-AUC on the Dataset S test set · mean of 5 seeds · axis starts at 0.50 (chance)")
    if m is None:
        ui.empty_state(hint=store.missing_hint("metrics.csv"))
    else:
        g = m[m["experiment"] == "S_full"].groupby("model")["roc_auc"].mean().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=[charts.MODEL_NAMES[k] for k in g.index], orientation="h",
                               marker_color=[charts.MODEL_COLORS[k] for k in g.index],
                               text=[f"{v:.3f}" for v in g.values], textposition="outside",
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(range=[0.5, 0.8])
        charts.show(fig, 220)
    st.page_link("views/model_comparison.py", label="Compare models →")

with right.container(border=True):
    ui.chart_title("Borrowed features don't move the score",
                   "XGBoost on Dataset A · engagement columns removed · ± std over 5 seeds · axis 0.50–1.00")
    exps = ["A_hard", "A_hard_augmented", "A_hard_random", "A_full"]
    if m is None or not set(exps) <= set(m["experiment"]):
        ui.empty_state(hint="Runs once notebooks 04 and 05 have produced the A_hard experiments.")
    else:
        x = m[(m["model"] == "xgboost") & m["experiment"].isin(exps)].groupby("experiment")["roc_auc"].agg(["mean", "std"])
        x = x.reindex(exps)
        fig = go.Figure(go.Bar(x=x["mean"], y=exps, orientation="h", error_x=dict(array=x["std"], color="#A1A1AA"),
                               marker_color=[charts.EXPERIMENT_COLORS[e] for e in exps],
                               text=[f"{v:.3f}" for v in x["mean"]], textposition="inside", insidetextanchor="start",
                               textfont=dict(color="#0F0F12"),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(range=[0.5, 1.08])
        fig.update_yaxes(autorange="reversed")
        charts.show(fig, 220)
    st.page_link("views/augmentation_study.py", label="See the study →")

# ---------------------------------------------------------------- key findings
ui.section("Key findings", "What separates customers who leave from those who stay, in Dataset S.")
first_year = s.loc[s["account_age"] < 12, "churned"].mean()
five_plus = s.loc[s["account_age"] >= 60, "churned"].mean()
light = s.loc[s["viewing_hours_per_week"] < 5, "churned"].mean()
heavy = s.loc[s["viewing_hours_per_week"] >= 30, "churned"].mean()
short = s.loc[s["avg_viewing_duration"] < 30, "churned"].mean()
long_ = s.loc[s["avg_viewing_duration"] >= 150, "churned"].mean()
findings = [
    (f"{first_year / five_plus:.1f}× higher churn", "Newer accounts churn far more than long-standing ones.",
     f"{ui.pct(first_year, 0)} leave in their first year, against {ui.pct(five_plus, 0)} after five years.",
     "views/risk_retention.py", "Risk & retention →"),
    (f"{ui.pct(light, 0)} churn among light viewers", "People who barely watch are the most likely to leave.",
     f"Under 5 viewing hours a week. Heavy viewers (30 h+) churn at {ui.pct(heavy, 0)}.",
     "views/data_explorer.py", "Data explorer →"),
    (f"{short / long_:.1f}× higher churn", "Short sessions are an early warning.",
     f"Average sessions under 30 minutes: {ui.pct(short, 0)} churn, against {ui.pct(long_, 0)} for 2.5 h+.",
     "views/churn_predictor.py", "Churn predictor →"),
]
for col, (stat, head, body, page, link) in zip(st.columns(3, gap="medium"), findings):
    with col.container(border=True):
        st.html(f'<div class="ss-kpi" style="border:0;padding:0;background:none"><div class="val sm up">{stat}</div>'
                f'<div style="font-weight:600;margin-top:6px">{head}</div><div class="sub">{body}</div></div>')
        st.page_link(page, label=link)

# ---------------------------------------------------------------- pipeline
ui.section("Pipeline", "From raw data to retention decisions. Each stop opens its page.")
stops = [("Data", "views/data_explorer.py"), ("Cleaning", "views/preprocessing.py"),
         ("Feature mapping", "views/preprocessing.py"), ("Augmentation study", "views/augmentation_study.py"),
         ("Models", "views/model_comparison.py"), ("Insights", "views/risk_retention.py"),
         ("Retention", "views/budget_simulator.py")]
with st.container(horizontal=True, gap="small"):
    for i, (label, page) in enumerate(stops, 1):
        st.page_link(page, label=f"{i} · {label}")

# ---------------------------------------------------------------- datasets
ui.section("Datasets", "S drives the models. A is the comparison. Old B was audited and left out.")
a = store.clean_a()
b = store.raw_b()
cols = st.columns(3, gap="medium")
cols[0].html(ui.kpi("Dataset S · main", f"{ui.num(len(s))} customers", small=True,
                    sub=f"{ui.pct(s['churned'].mean())} churn · viewing hours, session length, downloads, "
                        "support tickets, ratings, watchlist"))
cols[1].html(ui.kpi("Dataset A · comparison", f"{ui.num(len(a))} customers" if a is not None else "–", small=True,
                    sub=(f"{ui.pct(a['churned'].mean())} churn · " if a is not None else "")
                        + "Netflix-style profiles, almost perfectly predictable (ROC-AUC 0.999)"))
cols[2].html(ui.kpi("Old Dataset B · excluded", f"{ui.num(len(b))} customers", small=True,
                    sub=f"{ui.pct(b['churned'].mean())} churn · audit only: no feature predicts churn "
                        "(ROC-AUC 0.54, same as chance)"))
