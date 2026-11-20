import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn import business, config
from components import charts, store, ui

ui.page_header("Budget simulator", "Turn churn scores into a retention budget")
ui.callout("Who is worth targeting with a retention offer? Every customer scored at or above the threshold gets the "
           "offer. Only churners who accept it create value; offers to customers who would have stayed are pure cost.",
           tone="grey")

preds, s = store.predictions(), store.clean_s()
if preds is None or s is None:
    ui.empty_state(hint=store.missing_hint("predictions.csv", "data/interim/s_clean.csv"))
    st.stop()

left, right = st.columns([4, 8], gap="large")
with left.container(border=True):
    st.html('<p class="ss-chart-title">Assumptions</p>')
    cost = st.slider("Offer cost per customer ($)", 0.0, 20.0, 5.0, 0.5)
    success = st.slider("Offer success rate (%)", 0, 100, 30, 1,
                        help="Share of reached churners who stay because of the offer") / 100
    fee = st.slider("Average monthly fee ($)", 5.0, 20.0, round(float(s["monthly_charges"].mean()), 2), 0.5,
                    help="Defaults to the average monthly charge in Dataset S")
    months = st.slider("Months retained if saved", 1, 12, 6, 1)
    mdl = st.selectbox("Model", config.MODELS, index=config.MODELS.index("xgboost"), format_func=charts.MODEL_NAMES.get)
    st.caption(f"Value of one saved customer: ${fee:.2f} × {months} = ${fee * months:,.2f}")

p = preds[(preds["experiment"] == "S_full") & (preds["model"] == mdl)]
scale = len(s) / len(p)  # express test-split results for the whole customer base
sim = business.simulate_budget(p["y_true"], p["proba"], cost, success, fee, months, scale=scale)
best = business.best_row(sim)
y, sc = p["y_true"].to_numpy(), p["proba"].to_numpy()
top10 = sc >= np.quantile(sc, 0.9)
strategies = pd.DataFrame([
    {"Strategy": "Target everyone", **business.strategy(y, sc, np.ones_like(y), cost, success, fee, months, scale)},
    {"Strategy": "Target the riskiest 10%", **business.strategy(y, sc, top10, cost, success, fee, months, scale)},
    {"Strategy": f"Optimal threshold ({best['threshold']:.2f})",
     **business.strategy(y, sc, sc >= best["threshold"], cost, success, fee, months, scale)},
])

with right:
    profitable = best["net_value"] > 0
    ui.kpi_row([
        ui.kpi("Best threshold", f"{best['threshold']:.2f}" if profitable else "none", sub="highest net value"),
        ui.kpi("Customers targeted", ui.num(best["targeted"]) if profitable else "0",
               sub=f"{best['share_targeted']:.1%} of the base" if profitable else "no campaign pays off"),
        ui.kpi("Expected saved", ui.num(best["saved"]) if profitable else "0",
               sub=f"of {ui.num(best['churners_reached'])} churners reached"),
        ui.kpi("Net value", ui.money(best["net_value"]) if profitable else "$0",
               delta=f"vs {ui.money(strategies.loc[1, 'net_value'])} for the riskiest 10%",
               tone="down" if best["net_value"] >= strategies.loc[1, "net_value"] else "flat"),
    ])
    with st.container(border=True):
        ui.chart_title(f"Net value peaks at a threshold of {best['threshold']:.2f}" if profitable
                       else "No threshold makes this campaign profitable",
                       "Expected net value of the offer at each cut-off, scaled to all 243,787 customers · "
                       "below the dashed line the campaign loses money")
        fig = go.Figure()
        fig.add_scatter(x=sim["threshold"], y=sim["net_value"], mode="lines", line=dict(color=charts.TEAL, width=2.5),
                        customdata=np.stack([sim["targeted"], sim["saved"]], axis=1),
                        hovertemplate="threshold %{x:.2f}<br>net %{y:$,.0f}<br>targeted %{customdata[0]:,.0f}"
                                      "<br>saved %{customdata[1]:,.0f}<extra></extra>", name="net value")
        fig.add_hline(y=0, line_dash="dash", line_color=charts.MUTED)
        if profitable:
            fig.add_scatter(x=[best["threshold"]], y=[best["net_value"]], mode="markers+text",
                            marker=dict(size=10, color=charts.TEAL, line=dict(color=charts.TEXT, width=1.5)),
                            text=[f"  {ui.money(best['net_value'])} at {best['threshold']:.2f}"], textposition="middle right",
                            textfont=dict(color=charts.TEXT), hoverinfo="skip", showlegend=False)
        fig.update_xaxes(title="decision threshold", range=[0, 1])
        fig.update_yaxes(title="net value ($)", tickprefix="$", tickformat=",.0f")
        fig.update_layout(showlegend=False)
        charts.show(fig, 320)

ui.section("Strategies", "Three ways to choose who gets the offer, under the same assumptions.")
table = pd.DataFrame({
    "Strategy": strategies["Strategy"],
    "Customers targeted": [ui.num(v) for v in strategies["targeted"]],
    "Cost": [ui.money(v) for v in strategies["cost"]],
    "Value saved": [ui.money(v) for v in strategies["value_saved"]],
    "Net": [("+" if v > 0 else "") + ui.money(v) for v in strategies["net_value"]],
})
st.dataframe(table, hide_index=True)

with st.expander("How net value is worked out"):
    st.markdown(
        "For each threshold: **churners reached × success rate × monthly fee × months retained**, minus "
        "**customers targeted × offer cost**. Results come from the test split (36,569 customers) and are scaled "
        "to the full base of 243,787. The simulator only ranks customers, so it works with any model's scores, "
        "including the class-weighted ones. The success rate is an assumption: in practice it should come from an "
        "A/B test of the offer.")
