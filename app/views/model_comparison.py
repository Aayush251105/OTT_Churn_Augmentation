import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from churn import config, evaluate as ev
from components import charts, store, ui

ui.page_header("Model comparison", "Four models on the same split, with the trade-offs behind each score")

metrics, preds, finals = store.metrics(), store.predictions(), store.final_models()
if metrics is None or preds is None or finals is None:
    ui.empty_state(hint=store.missing_hint("metrics.csv", "predictions.csv", "final_models.json"))
    st.stop()

# ---------------------------------------------------------------- filters
c1, c3, c4 = st.columns([1, 1.2, 1], gap="medium")
ds = c1.segmented_control("Dataset", ["S", "A"], default="S", format_func=lambda v: f"Dataset {v}", key="mc_ds") or "S"
exp = "S_full" if ds == "S" else "A_full"
chosen = st.pills("Models", config.MODELS, default=config.MODELS, selection_mode="multi",
                  format_func=charts.MODEL_NAMES.get, key="mc_models") or config.MODELS
weighting = c3.segmented_control("Class weighting", ["final", "none", "balanced"], default="final",
                                 format_func={"final": "Chosen", "none": "None", "balanced": "Balanced"}.get,
                                 key="mc_cw", help="'Chosen' = the setting with the best validation F1 for each model") or "final"
t_type = c4.segmented_control("Threshold", ["tuned", "default"], default="tuned",
                              format_func={"tuned": "Tuned", "default": "0.50"}.get, key="mc_tt") or "tuned"

if ds == "S":
    ui.callout("Accuracy is misleading here: predicting “no churn” for everyone scores 81.9%. "
               "Compare models on recall, F1 and the two AUCs.", title="Read this first", tone="amber")

m = metrics[(metrics["experiment"] == exp) & (metrics["split"] == "test") & (metrics["threshold_type"] == t_type)
            & metrics["model"].isin(chosen)]
if weighting == "final":
    cw = {k: v["class_weight"] for k, v in finals[exp]["models"].items()}
    m = m[m["class_weight"] == m["model"].map(cw)]
else:
    m = m[m["class_weight"] == weighting]

# ---------------------------------------------------------------- metrics table
ui.section("Metrics", f"Mean ± std over 5 seeds, Dataset {ds} test set. Best value in each column in bold.")
summ = ev.summarize(m, by=("model",))
cols = ev.METRICS
best = {c: summ[f"{c}_mean"].max() for c in cols}
head = "".join(f"<th>{c.replace('_', '-').upper() if 'auc' in c else c.capitalize()}</th>" for c in cols)
body = ""
for r in summ.sort_values("roc_auc_mean", ascending=False).itertuples():
    cells = "".join(
        f"<td style='font-weight:{600 if getattr(r, f'{c}_mean') == best[c] else 400}'>"
        f"{getattr(r, f'{c}_mean'):.3f} <span style='color:#71717A'>± {getattr(r, f'{c}_std'):.3f}</span></td>"
        for c in cols)
    dot = f"<span style='display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:8px;background:{charts.MODEL_COLORS[r.model]}'></span>"
    body += f"<tr><td>{dot}{charts.MODEL_NAMES[r.model]}</td>{cells}</tr>"
st.html(f"""<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr style="color:#A1A1AA;text-align:left"><th>Model</th>{head}</tr></thead><tbody>{body}</tbody></table>
<style>table th,table td{{padding:10px 12px;border-bottom:1px solid #2A2A33;text-align:right}}
table th:first-child,table td:first-child{{text-align:left}}</style>""")

# ---------------------------------------------------------------- grouped bars
with st.container(border=True):
    lead = summ.set_index("model")[[f"{c}_mean" for c in ["f1", "roc_auc", "pr_auc"]]].idxmax()
    winner = lead.mode().iloc[0]
    ui.chart_title(f"{charts.MODEL_NAMES[winner]} leads on most metrics, but the margins are small",
                   "Scores per metric, grouped by model · accuracy left out (see note above)")
    fig = go.Figure()
    shown = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    for r in summ.itertuples():
        fig.add_bar(name=charts.MODEL_NAMES[r.model], x=[c.replace("_", "-").upper() if "auc" in c else c.capitalize() for c in shown],
                    y=[getattr(r, f"{c}_mean") for c in shown], marker_color=charts.MODEL_COLORS[r.model],
                    error_y=dict(array=[getattr(r, f"{c}_std") for c in shown], color="#A1A1AA", thickness=1),
                    hovertemplate="%{x}: %{y:.3f}<extra>" + charts.MODEL_NAMES[r.model] + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05])
    charts.show(fig, 320)

# ---------------------------------------------------------------- curves (final models, seed 0 test predictions)
p = preds[(preds["experiment"] == exp) & preds["model"].isin(chosen)]
ui.section("Curves", "Final model of each type · test split, seed 0.")
with st.container(border=True):
    tab_roc, tab_pr = st.tabs(["ROC curves", "Precision–recall curves"])
    base = p["y_true"].mean() if len(p) else 0
    with tab_roc:
        ui.chart_title("All four models beat chance; the gaps between them are small", "One line per model · AUC in the legend")
        fig = go.Figure()
        for mdl, g in p.groupby("model"):
            fpr, tpr, _ = roc_curve(g["y_true"], g["proba"])
            auc = ev.metrics_from_proba(g["y_true"], g["proba"])["roc_auc"]
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{charts.MODEL_NAMES[mdl]} · {auc:.3f}",
                            line=dict(color=charts.MODEL_COLORS[mdl], width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance · 0.500", line=dict(color=charts.MUTED, dash="dot"))
        fig.update_xaxes(title="false positive rate")
        fig.update_yaxes(title="true positive rate (recall)")
        charts.show(fig, 360)
    with tab_pr:
        ui.chart_title("Precision falls as recall rises", f"Dotted line = churn rate ({base:.1%}), the chance level")
        fig = go.Figure()
        for mdl, g in p.groupby("model"):
            prec, rec, _ = precision_recall_curve(g["y_true"], g["proba"])
            ap = ev.metrics_from_proba(g["y_true"], g["proba"])["pr_auc"]
            fig.add_scatter(x=rec, y=prec, mode="lines", name=f"{charts.MODEL_NAMES[mdl]} · {ap:.3f}",
                            line=dict(color=charts.MODEL_COLORS[mdl], width=2))
        fig.add_hline(y=base, line_dash="dot", line_color=charts.MUTED)
        fig.update_xaxes(title="recall")
        fig.update_yaxes(title="precision", range=[0, 1])
        charts.show(fig, 360)

# ---------------------------------------------------------------- threshold explorer
ui.section("Decision threshold", "Move the slider to see how one cut-off changes who gets flagged.")
with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    mdl = c1.selectbox("Model", chosen, format_func=charts.MODEL_NAMES.get, key="mc_thr_model")
    tuned = finals[exp]["models"][mdl]["threshold"]
    t = c2.slider("Decision threshold", 0.05, 0.95, float(tuned), 0.01, key=f"mc_thr_{exp}_{mdl}",
                  help=f"Starts at the threshold tuned on validation ({tuned:.2f})")
    g = p[p["model"] == mdl]
    met = ev.metrics_from_proba(g["y_true"], g["proba"], t)
    conf = ev.confusion(g["y_true"], g["proba"], t)
    flagged = conf["tp"] + conf["fp"]
    ui.kpi_row([ui.kpi("Precision", f"{met['precision']:.3f}", small=True), ui.kpi("Recall", f"{met['recall']:.3f}", small=True),
                ui.kpi("F1", f"{met['f1']:.3f}", small=True),
                ui.kpi("Flagged", ui.pct(flagged / len(g), 0), small=True, sub=f"{ui.num(flagged)} of {ui.num(len(g))}")])
    one_in = (flagged / conf["tp"]) if conf["tp"] else float("inf")
    st.caption(f"At {t:.2f} the model catches {met['recall']:.0%} of churners; about 1 in {one_in:.1f} flagged "
               f"customers actually leaves. Class weighting: {finals[exp]['models'][mdl]['class_weight']}.")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        curve = ev.threshold_curve(g["y_true"], g["proba"])
        ui.chart_title("Raising the threshold trades recall for precision", f"{charts.MODEL_NAMES[mdl]}, Dataset {ds} test set")
        fig = go.Figure()
        for col, color in [("precision", "#7C9CF5"), ("recall", charts.RED), ("f1", charts.TEXT_2)]:
            fig.add_scatter(x=curve["threshold"], y=curve[col], name=col.capitalize() if col != "f1" else "F1",
                            mode="lines", line=dict(color=color, width=2, dash="dash" if col == "f1" else None))
        fig.add_vline(x=t, line_color=charts.TEAL, line_width=1.5)
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        fig.update_xaxes(title="threshold")
        charts.show(fig, 280)
    with right:
        n = len(g)
        ui.chart_title(f"Confusion matrix at {t:.2f}", f"{ui.num(n)} test customers")
        z = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
        labels = [["TN", "FP"], ["FN", "TP"]]
        text = [[f"{labels[i][j]}<br>{z[i, j]:,}<br>{z[i, j] / n:.1%}" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted stay", "Predicted churn"], y=["Actual stay", "Actual churn"],
                                   text=text, texttemplate="%{text}", colorscale=charts.SEQUENTIAL, showscale=False,
                                   xgap=3, ygap=3, textfont=dict(color=charts.TEXT, size=13), hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, side="top")
        charts.show(fig, 280)
