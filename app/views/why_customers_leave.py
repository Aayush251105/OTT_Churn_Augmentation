import plotly.graph_objects as go
import streamlit as st

from components import charts, store, ui

ui.page_header("Why customers leave", "Reasons for cancelling, from app reviews")

topics = store.results_csv("reviews_topics.csv")
sentiment = store.results_csv("reviews_sentiment.csv")

if topics is None or sentiment is None:
    ui.callout("What subscribers say when they cancel, from Netflix app reviews on Google Play. The models on the "
               "other pages predict who leaves; this page will show why.", tone="grey")
    ui.empty_state("Coming soon: Phase 8, reviews NLP",
                   "Appears automatically once notebook 07 writes reports/results/reviews_topics.csv and "
                   "reviews_sentiment.csv.")
    ui.section("What this page will show")
    ui.numbered_list([
        ("Sentiment over time", "Monthly share of 1–2 star reviews, to spot spikes after price or policy changes.", "line chart"),
        ("Top reasons for cancelling", "Topics in reviews that mention cancelling (price, ads, password sharing, "
                                       "bugs, content library).", "topic shares"),
        ("Topic detail", "Rating distribution and example reviews for one topic.", "drill-down"),
        ("Link to the models", "Compare review complaints with the churn drivers found in Dataset S.", "callout"),
    ])
    st.stop()

# ---------------------------------------------------------------- live version (once Phase 8 results exist)
# Expected columns: reviews_topics.csv: topic, share, description (optional) · reviews_sentiment.csv: month, negative_share
ui.section("Top reasons for cancelling", "Share of cancelling reviews per topic.")
t = topics.sort_values("share")
with st.container(border=True):
    ui.chart_title(f"{t.iloc[-1]['topic']} is the most common reason to leave")
    fig = go.Figure(go.Bar(x=t["share"], y=t["topic"], orientation="h", marker_color=charts.NEUTRAL_BAR,
                           text=[f"{v:.0%}" for v in t["share"]], textposition="outside"))
    charts.pct_axis(fig, "x")
    charts.show(fig, 60 + 34 * len(t))

ui.section("Sentiment over time", "Monthly share of 1–2 star reviews.")
with st.container(border=True):
    fig = go.Figure(go.Scatter(x=sentiment["month"], y=sentiment["negative_share"], mode="lines",
                               line=dict(color=charts.RED, width=2)))
    charts.pct_axis(fig)
    charts.show(fig, 280)
