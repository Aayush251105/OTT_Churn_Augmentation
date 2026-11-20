import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn import config, data, features
from components import charts, store, ui

@st.cache_data(show_spinner="Finding neighbours…")
def neighbours(cid: str) -> pd.DataFrame:
    """The K nearest S customers of one A customer (matching.neighbours_for on the harmonized data)."""
    from churn import matching

    a_h, s_h = store.harmonized()
    return matching.neighbours_for(cid, a_h, s_h)


ui.page_header("Augmentation study", "Borrowing behaviour from similar customers to fill missing engagement data")
ui.callout("Can features borrowed from similar customers in Dataset S replace the engagement data that Dataset A is "
           "missing? A_hard is Dataset A with its watch-time and login columns removed.", tone="grey")

check = store.results_csv("transferability.csv")
report = store.results_csv("match_report.csv")
metrics = store.metrics()

# ---------------------------------------------------------------- 1. pre-check
ui.section("Pre-check: can shared columns predict the borrowed features?",
           "If plan, device, genre, payment and gender can't predict behaviour, neighbours found through them can't either.",
           step=1)
if check is None:
    ui.empty_state(hint=store.missing_hint("transferability.csv"))
else:
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1.container(border=True):
        ui.chart_title("Shared columns explain none of the behaviour",
                       "Cross-validated R² of predicting each S behavioural column from the shared columns · axis −0.05 to 0.25")
        t = check.sort_values("r2_mean")
        fig = go.Figure(go.Bar(x=t["r2_mean"], y=t["feature"], orientation="h", marker_color=charts.NEUTRAL_BAR,
                               error_x=dict(array=t["r2_std"], color=charts.TEXT_2),
                               text=[f"{v:+.4f}" for v in t["r2_mean"]], textposition="outside",
                               hovertemplate="%{y}: R² %{x:.5f}<extra></extra>"))
        fig.add_vline(x=0, line_color=charts.MUTED, line_dash="dash", annotation_text="R² = 0",
                      annotation_font_color=charts.MUTED)
        fig.update_xaxes(range=[-0.05, 0.25])
        charts.show(fig, 280)
    with c2:
        best = check["r2_mean"].max()
        ui.callout(f"The best R² is {best:.4f}. Shared columns carry no information about behaviour, so the transfer "
                   "is expected to add noise rather than signal.", title="Verdict", tone="amber")

# ---------------------------------------------------------------- 2. matching
ui.section("Matching", f"Each A customer is matched to the {config.K} most similar S customers on the harmonized shared "
           "columns (cosine similarity).", step=2)
if report is None:
    ui.empty_state(hint=store.missing_hint("match_report.csv"))
else:
    kept = report["kept"].sum()
    dropped = len(report) - kept
    ui.kpi_row([
        ui.kpi("Matched rows", ui.num(kept), sub=f"{kept / len(report):.1%} of A"),
        ui.kpi("Dropped rows", ui.num(dropped), sub=f"below {config.SIM_THRESHOLD} similarity"),
        ui.kpi("Mean similarity", f"{report.loc[report['kept'], 'mean_sim'].mean():.2f}", sub="of kept matches"),
        ui.kpi("Neighbours", f"K = {config.K}", sub="per A customer"),
        ui.kpi("Threshold", f"{config.SIM_THRESHOLD}", sub="cosine similarity"),
    ])
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1.container(border=True):
        share_above = (report["max_sim"] >= config.SIM_THRESHOLD).mean()
        ui.chart_title(f"{share_above:.0%} of customers clear the {config.SIM_THRESHOLD} similarity threshold",
                       f"Similarity of each A customer to their nearest S customers · dashed line = threshold")
        fig = go.Figure(go.Histogram(x=report["mean_sim"].fillna(report["max_sim"]), nbinsx=40,
                                     marker_color=charts.NEUTRAL_BAR, hovertemplate="%{x:.2f}: %{y} customers<extra></extra>"))
        fig.add_vline(x=config.SIM_THRESHOLD, line_dash="dash", line_color=charts.RED,
                      annotation_text=f"threshold {config.SIM_THRESHOLD}", annotation_font_color=charts.RED)
        fig.update_xaxes(title="cosine similarity", range=[0, 1.02])
        fig.update_yaxes(title="A customers")
        charts.show(fig, 260)
    with c2.container(border=True):
        ui.chart_title("Every customer finds a usable match" if dropped == 0 else f"{kept / len(report):.0%} of A customers get a usable match",
                       f"Of {ui.num(len(report))} Dataset A customers")
        fig = go.Figure()
        fig.add_bar(y=[""], x=[kept], orientation="h", name=f"Kept · {ui.num(kept)}", marker_color=charts.TEAL)
        fig.add_bar(y=[""], x=[dropped], orientation="h", name=f"Dropped · {ui.num(dropped)}", marker_color=charts.MUTED)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(range=[0, len(report)])
        charts.show(fig, 140)
        st.caption("Shared columns are few and coarse (5 categoricals), so many A customers have exact profile twins in S. "
                   "High similarity here does not mean similar behaviour (see step 1).")

# ---------------------------------------------------------------- 3. result
ui.section("Result", "The same four models and protocol as notebook 05 · test split · ± std over 5 seeds.", step=3)
exps = ["A_hard", "A_hard_augmented", "A_hard_random"]
m = store.test_metrics(exps + ["A_full"])
if m is None or not set(exps) <= set(m["experiment"]):
    ui.empty_state(hint="Run notebook 05 after notebook 04 to train the A_hard experiments.")
else:
    mdl = st.segmented_control("Model", config.MODELS, default="xgboost", format_func=charts.MODEL_NAMES.get,
                               key="aug_model") or "xgboost"
    g = m[m["model"] == mdl].groupby("experiment")[["roc_auc", "f1"]].agg(["mean", "std"])
    eff = g.loc["A_hard_augmented", ("roc_auc", "mean")] - g.loc["A_hard", ("roc_auc", "mean")]
    ctrl = g.loc["A_hard_random", ("roc_auc", "mean")] - g.loc["A_hard", ("roc_auc", "mean")]
    noise = g.loc[exps, ("roc_auc", "std")].max()
    gap = g.loc["A_full", ("roc_auc", "mean")] - g.loc["A_hard", ("roc_auc", "mean")]
    ui.kpi_row([
        ui.kpi("Effect vs A_hard", ui.signed(eff), sub=f"seed noise ± {noise:.3f}", tone="flat"),
        ui.kpi("Random control vs A_hard", ui.signed(ctrl), sub="features from random S customers"),
        ui.kpi("Gap to ceiling", f"{gap:.3f}", sub="what real engagement data adds"),
    ])
    with st.container(border=True):
        tab_auc, tab_f1 = st.tabs(["ROC-AUC", "F1"])
        for tab, metric in [(tab_auc, "roc_auc"), (tab_f1, "f1")]:
            with tab:
                verdict = ("Borrowed features don't move the score" if abs(eff) <= max(noise, 0.005)
                           else "Borrowed features change the score")
                ui.chart_title(verdict, f"{charts.MODEL_NAMES[mdl]} on Dataset A · engagement columns removed · "
                                        "dashed line = A_full ceiling")
                fig = go.Figure(go.Bar(x=exps, y=g.loc[exps, (metric, "mean")],
                                       error_y=dict(array=g.loc[exps, (metric, "std")], color=charts.TEXT_2),
                                       marker_color=[charts.EXPERIMENT_COLORS[e] for e in exps],
                                       text=[f"{v:.3f}" for v in g.loc[exps, (metric, 'mean')]], textposition="inside",
                                       insidetextanchor="start", textfont=dict(color="#0F0F12")))
                ceil = g.loc["A_full", (metric, "mean")]
                fig.add_hline(y=ceil, line_dash="dash", line_color=charts.MUTED,
                              annotation_text=f"A_full ceiling {ceil:.3f}", annotation_font_color=charts.MUTED)
                fig.update_yaxes(range=[0.4 if metric == "roc_auc" else 0, 1.05])
                charts.show(fig, 300, key=f"aug_{metric}")
        same = abs(eff - ctrl) <= max(noise, 0.005)
        st.caption("Augmentation performs the same as random neighbours. The difference is noise, as the pre-check predicted."
                   if same else "Augmentation and the random control differ by more than the seed noise.")

# ---------------------------------------------------------------- inspector
with st.expander("Customer inspector: see which S customers one A customer borrowed from"):
    h = store.harmonized()
    if h is None:
        ui.empty_state(hint="Run `python -m churn.preprocessing` first.")
    else:
        a_h, s_h = h
        cid = st.selectbox("Dataset A customer", a_h["customer_id"].head(500), key="aug_cid",
                           help="First 500 customers of Dataset A")
        row = a_h[a_h["customer_id"] == cid].iloc[0]
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            prof = {"Age": row["age"], "Plan": row["plan_h"], "Device": row["device_h"], "Genre": row["genre_h"],
                    "Payment": row["payment_h"], "Gender": row["gender_h"], "Churned": "Yes" if row["churned"] else "No"}
            st.dataframe(pd.DataFrame(prof.items(), columns=["Field", "Value"]).astype(str), hide_index=True)
        with c2:
            nb = neighbours(cid)
            st.dataframe(nb[["rank", "customer_id", "similarity", "plan_h", "device_h", "genre_h", "payment_h", "gender_h"]],
                         hide_index=True, column_config={
                             "similarity": st.column_config.ProgressColumn("Similarity", min_value=0.6, max_value=1.0, format="%.2f")})
        path = config.experiment_path("A_hard_augmented")
        if path.exists():
            aug = data.load_experiment("A_hard_augmented")
            r = aug[aug["customer_id"] == cid]
            if not r.empty:
                moved = pd.DataFrame({"Feature": features.TRANSFER_FEATURES,
                                      "Transferred value": [round(float(r.iloc[0][f]), 2) for f in features.TRANSFER_FEATURES],
                                      "Method": "similarity-weighted mean of valid neighbours"})
                st.dataframe(moved, hide_index=True)
