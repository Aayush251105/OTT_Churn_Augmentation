import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn import data, eda, features
from components import charts, store, ui

ui.page_header("Preprocessing", "Cleaning, fixes and how the datasets were aligned")


@st.cache_data(show_spinner="Computing associations…")
def cramers_matrix(name: str) -> pd.DataFrame:
    df = store.raw_s() if name == "S" else store.raw_a()
    cols = data.S_CATEGORICAL + data.S_BINARY if name == "S" else data.A_CATEGORICAL
    m = pd.DataFrame(1.0, index=cols, columns=cols)
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            m.loc[x, y] = m.loc[y, x] = eda.cramers_v(df[x], df[y])
    return m


name = st.segmented_control("Dataset", ["S", "A"], default="S", format_func=lambda v: f"Dataset {v}", key="pp_ds") or "S"
log = store.results_csv("cleaning_log.csv")

# ---------------------------------------------------------------- cleaning steps
ui.section("Cleaning steps", "Applied in this order. No customer was removed from either dataset.")
if log is None:
    ui.empty_state(hint=store.missing_hint("cleaning_log.csv"))
else:
    steps = log[log["dataset"] == name]
    rows = []
    for r in steps.itertuples():
        if r.cols_after != r.cols_before:
            right = f"{r.cols_before} → {r.cols_after} columns"
        elif r.rows_after != r.rows_before:
            right = f"{ui.num(r.rows_before)} → {ui.num(r.rows_after)} rows"
        else:
            right = f"{ui.num(r.n_affected)} values" if r.n_affected else "no change"
        rows.append((r.step, r.note, right))
    rows.append(("Encode and scale (inside each model)",
                 "One-hot categoricals, standard-scale numerics; fitted on the training split only to avoid leakage.",
                 "33 model features"))
    ui.numbered_list(rows)

# ---------------------------------------------------------------- before / after
ui.section("Before and after", "The column that was removed, and why.")
c1, c2 = st.columns(2, gap="medium")
if name == "A":
    a = store.raw_a()
    bad = a["avg_watch_time_per_day"] > 24
    with c1.container(border=True):
        ui.chart_title(f"Before: {bad.sum()} customers watch more than a day lasts",
                       "avg_watch_time_per_day · impossible values (> 24 h) in red")
        fig = go.Figure()
        fig.add_histogram(x=a.loc[~bad, "avg_watch_time_per_day"].clip(upper=24), nbinsx=48,
                          marker_color=charts.NEUTRAL_BAR, name="valid")
        fig.add_histogram(x=a.loc[bad, "avg_watch_time_per_day"].clip(upper=30), nbinsx=8, marker_color=charts.RED,
                          name="> 24 h/day")
        fig.update_yaxes(type="log", title="customers (log)")
        fig.add_vline(x=24, line_dash="dash", line_color=charts.RED)
        charts.show(fig, 260)
    with c2.container(border=True):
        ui.chart_title("After: the column is dropped",
                       "It is exactly watch_hours / (last_login_days + 1), so the models keep its two inputs instead")
        fig = go.Figure(go.Scattergl(x=a["watch_hours"] / (a["last_login_days"] + 1), y=a["avg_watch_time_per_day"],
                                     mode="markers", marker=dict(size=4, color=np.where(bad, charts.RED, charts.MUTED))))
        fig.update_xaxes(title="watch_hours / (last_login_days + 1)")
        fig.update_yaxes(title="avg_watch_time_per_day")
        charts.show(fig, 260)
else:
    s = store.raw_s().sample(4000, random_state=0)
    with c1.container(border=True):
        ui.chart_title("Before: total_charges repeats two other columns",
                       "total_charges vs account_age × monthly_charges · 4,000-customer sample")
        fig = go.Figure(go.Scattergl(x=s["account_age"] * s["monthly_charges"], y=s["total_charges"], mode="markers",
                                     marker=dict(size=4, color=charts.MUTED)))
        fig.update_xaxes(title="account_age × monthly_charges")
        fig.update_yaxes(title="total_charges")
        charts.show(fig, 260)
    with c2.container(border=True):
        ui.chart_title("After: total_charges dropped, Yes/No columns converted to 1/0",
                       "Share of customers with each setting switched on")
        raw = store.raw_s()
        yes = [(raw[c] == "Yes").mean() for c in data.S_BINARY]
        fig = go.Figure(go.Bar(x=data.S_BINARY, y=yes, marker_color=charts.NEUTRAL_BAR,
                               text=[f"{v:.1%}" for v in yes], textposition="outside"))
        fig.update_yaxes(range=[0, 0.7], tickformat=".0%")
        charts.show(fig, 260)

# ---------------------------------------------------------------- harmonization
ui.section("Harmonization A ↔ S", "How the shared columns were aligned before matching customers across datasets.")
a_raw, s_raw = store.raw_a(), store.raw_s()
mt = features.mapping_table()
rows = []
for feat in ["subscription_type", "device", "favorite_genre", "payment_method", "gender"]:
    sub = mt[mt["feature"] == feat]
    va, vs = sorted(a_raw[feat].unique()), sorted(s_raw[feat].unique())
    mapped = sub[sub["raw_label"] != sub["harmonized_label"]]
    note = ("same labels" if mapped.empty else
            ", ".join(f"{r} → {h}" for r, h in zip(mapped["raw_label"], mapped["harmonized_label"])))
    rows.append({"Feature": feat, "Dataset A values": ", ".join(va), "Dataset S values": ", ".join(vs), "Mapped to": note})
rows.append({"Feature": "monthly_fee ↔ monthly_charges", "Dataset A values": "3 fixed fees, tied to plan",
             "Dataset S values": "continuous $4.99–19.99, unrelated to plan", "Mapped to": "not matched (different meaning)"})
st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---------------------------------------------------------------- association
ui.section("Feature association", "How strongly columns relate to each other within the selected dataset.")
with st.container(border=True):
    tab_cat, tab_num = st.tabs(["Categorical (Cramér's V)", "Numeric (Spearman)"])
    with tab_cat:
        cm = cramers_matrix(name)
        off = cm.where(~np.eye(len(cm), dtype=bool)).max().max()
        ui.chart_title("Categorical columns are close to independent",
                       f"Cramér's V · 0 = unrelated, 1 = identical · strongest pair {off:.3f}")
        charts.show(charts.heatmap(cm, diverging=False, zmin=0, zmax=1), 120 + 40 * len(cm))
    with tab_num:
        df = store.raw_s() if name == "S" else store.raw_a()
        nums = data.S_NUMERIC if name == "S" else data.A_NUMERIC
        ui.chart_title("Only the derived columns correlate", "Spearman correlation · red negative, teal positive")
        charts.show(charts.heatmap(df[nums].corr(method="spearman")), 120 + 40 * len(nums))

ui.section("Features borrowed from Dataset S", "Behavioural columns with churn signal in S, transferred into A_hard.")
ui.tags(features.TRANSFER_FEATURES)
