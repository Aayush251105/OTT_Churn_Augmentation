from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts, store, ui

ui.page_header("Why customers leave", "Reasons for cancelling, from Netflix app reviews on Google Play")

topics = store.results_csv("reviews_topics.csv")
sentiment = store.results_csv("reviews_sentiment.csv")
trends = store.results_csv("reviews_theme_trends.csv")
examples = store.results_csv("reviews_examples.csv")
clean_log = store.results_csv("reviews_cleaning_log.csv")

if topics is None or sentiment is None:
    ui.empty_state("Review results not found",
                   "Run notebook 07 (or `python -m churn.reviews`) to write reports/results/reviews_*.csv.")
    st.stop()

from churn import reviews  # noqa: E402  (imported after the guard: only needed when results exist)

ui.callout("The models on the other pages predict who leaves; these reviews say why, in customers' own words. "
           "Reviewers cannot be linked to customers in Dataset S, so the link to churn drivers is supporting evidence, "
           "not proof.", tone="grey")

# ---------------------------------------------------------------- KPIs
sentiment["month"] = pd.to_datetime(sentiment["month"])
n_total = int(clean_log["rows_after"].iloc[-1]) if clean_log is not None else int(sentiment["n_reviews"].sum())
w = sentiment["n_reviews"]
neg_share = float((sentiment["negative_share"] * w).sum() / w.sum())
cancel_share = float((sentiment["cancel_share"] * w).sum() / w.sum())
first, last = sentiment["month"].min(), sentiment["month"].max()
peak = sentiment.loc[sentiment["negative_share"].idxmax()]
ui.kpi_row([
    ui.kpi("Reviews analysed", ui.num(n_total), sub=f"Google Play, {first:%b %Y} – {last:%b %Y}"),
    ui.kpi("Negative reviews", ui.pct(neg_share), sub="1–2 stars"),
    ui.kpi("Mention cancelling", ui.pct(cancel_share), sub="“cancel”, “unsubscribe”, “not worth”, “switching to”…"),
    ui.kpi("Most negative month", f"{peak['month']:%b %Y}", small=True,
           sub=f"{peak['negative_share']:.0%} of reviews were 1–2 stars"),
])

# ---------------------------------------------------------------- sentiment over time
EVENTS = [(pd.Timestamp(d + "-01"), label) for d, label in reviews.EVENTS]


EVENT_KEY = " · ".join(f"{i} {label} ({date:%b %Y})" for i, (date, label) in enumerate(EVENTS, 1))


def add_events(fig: go.Figure, y_top: float) -> None:
    """Dotted lines for real-world events, each tagged with a number; the key (EVENT_KEY) sits under the chart.
    Several events fall within a year, so full labels on the chart would overlap."""
    for i, (date, label) in enumerate(EVENTS, 1):
        fig.add_vline(x=date, line_dash="dot", line_color=charts.MUTED, line_width=1)
        fig.add_annotation(x=date, y=y_top, yref="y", text=str(i), showarrow=False, yshift=-10 * ((i + 1) % 2),
                           font=dict(size=11, color=charts.TEXT), bgcolor=charts.RAISED, bordercolor=charts.BORDER,
                           borderpad=2, hovertext=f"{label} ({date:%b %Y})")


ui.section("Sentiment over time", "Monthly share of reviews · dotted lines mark price, plan and sharing changes "
           "(approximate global rollout dates).")
with st.container(border=True):
    tab_neg, tab_cancel = st.tabs(["Negative reviews", "Mentions cancelling"])
    for tab, col, colour, label in [(tab_neg, "negative_share", charts.RED, "1–2 star reviews"),
                                    (tab_cancel, "cancel_share", charts.AMBER, "reviews mentioning cancelling")]:
        with tab:
            top_row = sentiment.loc[sentiment[col].idxmax()]
            near = min(EVENTS, key=lambda e: abs((e[0] - top_row["month"]).days))
            gap = abs((near[0] - top_row["month"]).days)
            context = f", around the {near[1]}" if gap <= 120 else ""
            ui.chart_title(f"Peak: {top_row[col]:.0%} of {label} in {top_row['month']:%b %Y}{context}",
                           f"Monthly share of {label} · months with fewer than 100 reviews left out")
            fig = go.Figure(go.Scatter(x=sentiment["month"], y=sentiment[col], mode="lines",
                                       line=dict(color=colour, width=2), customdata=sentiment["n_reviews"],
                                       hovertemplate="%{x|%b %Y}: %{y:.1%} of %{customdata:,} reviews<extra></extra>"))
            y_top = float(sentiment[col].max()) * 1.18
            add_events(fig, y_top)
            fig.update_yaxes(range=[0, y_top * 1.04], tickformat=".0%")
            fig.update_layout(showlegend=False)
            charts.show(fig, 340, key=f"wcl_trend_{col}")
            st.caption(EVENT_KEY)

# ---------------------------------------------------------------- top reasons
ui.section("Top reasons", "Each negative review is assigned its main theme (NMF topics on the review text, "
           "grouped into named themes).")
t = topics[topics["share"] > 0].copy()
with st.container(border=True):
    view = st.segmented_control("Among", ["negative", "cancel"], default="cancel", key="wcl_view",
                                format_func={"negative": "All negative reviews",
                                             "cancel": "Reviews that mention cancelling"}.get) or "cancel"
    col = "share" if view == "negative" else "cancel_share"
    t = t.sort_values(col)
    lead = t.iloc[-1]
    if view == "cancel":
        common = topics.sort_values("share").iloc[-1]
        title = (f"{lead['topic']} drives cancellations ({lead[col]:.0%}), though "
                 f"{common['topic'].lower()} is the most common complaint overall")
    else:
        title = f"{lead['topic']} is the most common complaint ({lead[col]:.0%} of negative reviews)"
    ui.chart_title(title, "Share of reviews per theme · switch the view to compare everyday complaints with "
                          "reasons people give when they cancel")
    fig = go.Figure(go.Bar(
        x=t[col], y=t["topic"], orientation="h",
        marker_color=[charts.RED if i == len(t) - 1 else charts.NEUTRAL_BAR for i in range(len(t))],
        text=[f"{v:.0%}" for v in t[col]], textposition="outside", customdata=t["description"],
        hovertemplate="%{y}: %{x:.1%}<br>%{customdata}<extra></extra>"))
    fig.update_xaxes(range=[0, t[col].max() * 1.2], tickformat=".0%")
    charts.show(fig, 60 + 40 * len(t), key=f"wcl_reasons_{view}")

# ---------------------------------------------------------------- theme trends
ui.section("Theme trends", "Share of each quarter's negative reviews that mention a theme (keyword flags, so one "
           "review can mention several; ads and password sharing are visible here even though topic modelling "
           "merges them).")
if trends is not None:
    labels = dict(zip(topics["theme"], topics["topic"])) | {"cancel_intent": "Mentions cancelling"}
    options = [th for th in labels if th in set(trends["theme"])]
    with st.container(border=True):
        picked = st.pills("Themes", options, default=["price_billing", "account_sharing", "ads", "cancel_intent"],
                          selection_mode="multi", format_func=labels.get, key="wcl_themes") or ["price_billing"]
        tr = trends[trends["theme"].isin(picked)].copy()
        tr["date"] = pd.PeriodIndex(tr["quarter"], freq="Q").to_timestamp()
        rise = (tr.sort_values("date").groupby("theme")["share"]
                .agg(lambda s: s.iloc[-4:].mean() - s.iloc[:4].mean()).sort_values())
        mover = rise.index[-1]
        ui.chart_title(f"{labels[mover]} grew the most: {rise.iloc[-1]:+.0%} points from the first to the last year",
                       "Quarterly share of negative reviews mentioning each theme")
        palette = [charts.RED, "#7C9CF5", charts.TEAL, charts.AMBER, "#A78BFA", charts.TEXT_2, "#E879A6", "#9C9CA8",
                   "#5BC0EB", "#C4A484"]
        fig = go.Figure()
        for i, th in enumerate(picked):
            d = tr[tr["theme"] == th].sort_values("date")
            fig.add_scatter(x=d["date"], y=d["share"], mode="lines", name=labels[th],
                            line=dict(color=palette[i % len(palette)], width=2,
                                      dash="dot" if th == "cancel_intent" else None),
                            customdata=d["n_negative"],
                            hovertemplate=f"{labels[th]}<br>%{{x|%Y Q%q}}: %{{y:.1%}} of %{{customdata:,}} negative reviews<extra></extra>")
        y_top = float(tr["share"].max()) * 1.25
        add_events(fig, y_top)
        fig.update_yaxes(range=[0, y_top * 1.04], tickformat=".0%")
        charts.show(fig, 380, key="wcl_theme_trends")
        st.caption(EVENT_KEY)

# ---------------------------------------------------------------- theme detail
ui.section("Theme detail", "What each theme covers, how big it is, and what reviewers actually wrote.")
detail_options = topics.loc[topics["n_reviews"] > 0, "theme"].tolist()
labels = dict(zip(topics["theme"], topics["topic"]))
th = st.selectbox("Theme", detail_options, format_func=labels.get, key="wcl_detail")
row = topics.set_index("theme").loc[th]
ui.callout(row["description"], title=row["topic"], tone="red" if th == "price_billing" else "grey")
ui.kpi_row([
    ui.kpi("Share of negative reviews", ui.pct(row["share"]), sub=f"{ui.num(row['n_reviews'])} reviews"),
    ui.kpi("Share of cancellation reviews", ui.pct(row["cancel_share"]), sub="reviews that mention cancelling"),
    ui.kpi("Mentioned in", ui.pct(row["keyword_share"]), sub="of negative reviews (keyword check)"),
    ui.kpi("Matching driver in Dataset S", (row["s_driver"] if isinstance(row["s_driver"], str) else "–")
           .replace("_", " "), small=True, sub="see the table below"),
])
if examples is not None:
    ex = examples[examples["theme"] == th].sort_values("thumbs_up", ascending=False).head(3)
    if not ex.empty:
        st.caption("Example reviews (most upvoted)")
        for col, r in zip(st.columns(len(ex), gap="medium"), ex.itertuples()):
            stars = "★" * int(r.score) + "☆" * (5 - int(r.score))
            date = pd.Timestamp(r.date)
            col.html(f'<div class="ss-quote"><div class="q">“{escape(r.content)}”</div>'
                     f'<div class="m"><span class="stars">{stars}</span> · {date:%b %Y} · '
                     f'{int(r.thumbs_up):,} found this helpful</div></div>')

# ---------------------------------------------------------------- link to the models
ui.section("Link to the models", "Each review theme next to the Dataset S column that measures the same thing, and "
           "what S shows for it.")
s = store.clean_s()
rules = store.results_json("retention_rules.json") or []
s_rules = {r["condition"].split()[0]: r for r in rules if r["dataset"] == "S"}


def s_evidence(driver: str) -> str:
    """One line of what Dataset S shows for a driver: a retention rule's segment churn, or churn by a 0/1 setting."""
    if driver in s_rules:
        r = s_rules[driver]
        return (f"{r['condition'].replace('_', ' ')}: {r['segment_churn_rate']:.0%} churn vs "
                f"{r['rest_churn_rate']:.0%} for everyone else")
    if s is not None and driver in s.columns and set(s[driver].unique()) <= {0, 1}:
        g = s.groupby(driver)["churned"].mean()
        return f"with it: {g.get(1, float('nan')):.1%} churn · without: {g.get(0, float('nan')):.1%} (no real difference)"
    return "–"


link = topics[topics["s_driver"].notna() & (topics["s_driver"] != "")].copy()
link = link.sort_values("cancel_share", ascending=False)
st.dataframe(pd.DataFrame({
    "Review theme": link["topic"],
    "Share of cancellation reviews": link["cancel_share"],
    "Dataset S column": link["s_driver"].str.replace("_", " "),
    "What Dataset S shows": [s_evidence(d) for d in link["s_driver"]],
}), hide_index=True, column_config={
    "Share of cancellation reviews": st.column_config.NumberColumn(format="percent"),
    "What Dataset S shows": st.column_config.TextColumn(width="large")})
price = topics.set_index("theme").loc["price_billing"]
if "monthly_charges" in s_rules:
    r = s_rules["monthly_charges"]
    ui.callout(f"Price is the reason most often given by people who cancel ({price['cancel_share']:.0%}), and in "
               f"Dataset S customers paying $17 or more churn at {r['segment_churn_rate']:.0%} against "
               f"{r['rest_churn_rate']:.0%} for everyone else. Two independent sources point the same way.",
               title="Price shows up in both", tone="amber")

# ---------------------------------------------------------------- method
with st.expander("How the reviews were processed"):
    if clean_log is not None:
        rows = [(r.step.replace("_", " ").capitalize(), r.note,
                 f"{ui.num(r.rows_before)} → {ui.num(r.rows_after)}") for r in clean_log.itertuples()]
        ui.numbered_list(rows)
    st.markdown(
        "- **Sentiment** = the star rating: 1–2 negative, 3 neutral, 4–5 positive.\n"
        "- **Themes:** NMF finds topics in the negative reviews; each topic is mapped to a named theme. Themes the "
        "topic model merges away (ads, password sharing) are tracked with a keyword check instead.\n"
        "- **Cancel intent:** the review mentions cancelling, unsubscribing, a refund, “not worth it” or switching "
        "to another service.\n"
        "- **Caveat:** people who write reviews are not a random sample of subscribers, and they cannot be matched "
        "to customers in the churn datasets. Details in notebook 07.")
