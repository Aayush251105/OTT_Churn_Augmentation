import plotly.graph_objects as go
import streamlit as st

from churn import data, eda
from components import charts, store, ui

ui.page_header("Data explorer", "Schema, distributions and churn patterns for each dataset")

choice = st.segmented_control("Dataset", ["Dataset S", "Dataset A", "Old B"], default="Dataset S", key="de_ds")
choice = choice or "Dataset S"

if choice == "Dataset S":
    df, numeric, categorical = store.raw_s(), data.S_NUMERIC, data.S_CATEGORICAL + data.S_BINARY
    role, note = "Dataset S · main", "behavioural columns"
elif choice == "Dataset A":
    df, numeric, categorical = store.raw_a(), data.A_NUMERIC, data.A_CATEGORICAL
    role, note = "Dataset A · comparison", "profile and engagement columns"
else:
    df, numeric, categorical = store.raw_b(), data.B_NUMERIC, data.B_CATEGORICAL
    role, note = "Old B · audit only", "excluded from modelling"

# ---------------------------------------------------------------- old B audit callout
if choice == "Old B":
    c1, c2 = st.columns([2, 3], gap="medium")
    with c1:
        ui.callout("No feature in this dataset predicts churn (ROC-AUC 0.54, about the same as shuffled labels). "
                   "It is excluded from modelling and kept here as a data-audit finding.", title="Data audit", tone="amber")
    with c2.container(border=True):
        ui.chart_title("Old B is no better than chance", "5-fold ROC-AUC, gradient boosting (notebook 01, section 9)")
        sc = store.SIGNAL_CHECK.iloc[::-1]
        fig = go.Figure(go.Bar(x=sc["roc_auc"], y=sc["run"], orientation="h",
                               marker_color=[charts.DATASET_COLORS[d] for d in sc["dataset"]],
                               text=[f"{v:.3f}" for v in sc["roc_auc"]], textposition="outside"))
        fig.add_vline(x=0.5, line_dash="dash", line_color=charts.MUTED, annotation_text="chance",
                      annotation_font_color=charts.MUTED)
        fig.update_xaxes(range=[0, 1.12])
        charts.show(fig, 230)

# ---------------------------------------------------------------- KPIs
q = eda.quality_report(df, "customer_id")
ui.kpi_row([
    ui.kpi("Rows", ui.num(len(df)), sub=role),
    ui.kpi("Columns", str(df.shape[1]), sub=f"{len(numeric)} numeric · {len(categorical)} categorical · {note}"),
    ui.kpi("Churn rate", ui.pct(df["churned"].mean()), sub=f"{ui.num(df['churned'].sum())} customers"),
    ui.kpi("Missing values", ui.num(q["missing_cells"]), sub=f"{ui.num(q['duplicate_rows'])} duplicate rows"),
])

# ---------------------------------------------------------------- schema
ui.section("Schema", "What each column holds. Click a header to sort.")
schema = eda.schema_summary(df)
schema["kind"] = ["target" if c == "churned" else ("id" if c == "customer_id" else k)
                  for c, k in zip(schema["column"], schema["kind"])]
schema["example"] = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in schema["example"]]
st.dataframe(schema[["column", "kind", "unique", "missing", "example"]], hide_index=True, height=300,
             column_config={"column": "Column", "kind": "Type", "unique": st.column_config.NumberColumn("Unique", format="localized"),
                            "missing": "Missing", "example": "Example"})

# ---------------------------------------------------------------- distributions
ui.section("Distributions", "Share of customers in each bin, so churned and retained compare at the same scale.")
with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    feature = c1.selectbox("Feature", numeric, key=f"de_num_{choice}")
    split = c2.toggle("Split by churn", value=True, key="de_split")
    skewed = df[feature].skew() > 2
    if split:
        fig = charts.histogram_by_status(df, feature, clip_q=0.99 if skewed else None)
    else:
        x = df[feature].clip(upper=df[feature].quantile(0.99)) if skewed else df[feature]
        fig = go.Figure(go.Histogram(x=x, nbinsx=40, histnorm="probability", marker_color=charts.NEUTRAL_BAR))
        charts.pct_axis(fig)
    med = df.groupby("churned")[feature].median()
    ui.chart_title(f"{feature}: median {med.get(1, float('nan')):.3g} for churners vs {med.get(0, float('nan')):.3g} for retained",
                   "Values above the 99th percentile are grouped into the last bin" if skewed else None)
    charts.show(fig, 300)

# ---------------------------------------------------------------- churn patterns
ui.section("Churn patterns")
c1, c2 = st.columns(2, gap="medium")
with c1.container(border=True):
    cat = st.selectbox("Category", categorical, key=f"de_cat_{choice}")
    t = eda.churn_rate_by(df, cat, "churned")
    top = t.iloc[0]
    ui.chart_title(f"{top[cat]} has the highest churn in {cat} ({top['churn_rate']:.1%})",
                   "Churn rate by category · red/teal = more than 2 points above/below the overall rate")
    charts.show(charts.churn_rate_bars(t, cat, df["churned"].mean()), 60 + 34 * len(t))
with c2.container(border=True):
    corr = df[numeric + ["churned"]].corr(method="spearman")
    strongest = corr["churned"].drop("churned").abs().idxmax()
    ui.chart_title(f"{strongest} relates most to churn",
                   "Spearman correlation between numeric features · red negative, teal positive")
    charts.show(charts.heatmap(corr), 120 + 40 * len(corr))
