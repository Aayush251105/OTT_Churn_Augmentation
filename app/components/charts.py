"""One Plotly template and a few chart helpers so every chart in StayScope looks the same."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FONT = "Inter, Arial, sans-serif"  # keep simple: unquoted multi-word names break Plotly text measuring
TEXT, TEXT_2, MUTED, BORDER, RAISED = "#F2F2F5", "#A1A1AA", "#71717A", "#2A2A33", "#1E1E25"
RED, AMBER, TEAL = "#E5484D", "#F5A524", "#2BB3A3"
NEUTRAL_BAR = "#4A4A55"

MODEL_COLORS = {"logistic_regression": "#7C9CF5", "decision_tree": "#A78BFA",
                "random_forest": "#2BB3A3", "xgboost": "#F5A524"}
MODEL_NAMES = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
               "random_forest": "Random Forest", "xgboost": "XGBoost"}
EXPERIMENT_COLORS = {"A_hard": "#7C9CF5", "A_hard_augmented": "#2BB3A3", "A_hard_random": "#A1A1AA",
                     "A_full": "#71717A", "S_full": "#2BB3A3"}
DATASET_COLORS = {"S": "#2BB3A3", "A": "#7C9CF5", "Old B": "#71717A"}
TIER_COLORS = {"low": TEAL, "medium": AMBER, "high": RED}
STATUS_COLORS = {"Churned": RED, "Retained": TEAL}

_axis = dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, ticks="", showline=False, automargin=True,
             tickfont=dict(color=TEXT_2, size=11), title=dict(font=dict(color=TEXT_2, size=12)))
pio.templates["stayscope"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=12, color=TEXT_2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=list(MODEL_COLORS.values()) + [RED, MUTED],
    xaxis=_axis, yaxis=_axis,
    margin=dict(l=8, r=8, t=8, b=8),
    legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=TEXT_2, size=12),
                bgcolor="rgba(0,0,0,0)", title=None),
    hoverlabel=dict(bgcolor=RAISED, bordercolor=BORDER, font=dict(color=TEXT, family=FONT, size=12)),
    bargap=0.3, bargroupgap=0.08,
    coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=TEXT_2))),
))

DIVERGING = [[0, RED], [0.5, "#26262E"], [1, TEAL]]
SEQUENTIAL = [[0, "#1E1E25"], [1, "#9C9CA8"]]


MAX_INLINE_HEIGHT = 520         # px: fits a ~768px-tall laptop viewport with the page header above it
TITLE_KEY = "_ss_chart_title"   # set by ui.chart_title, read by the next show()
COUNTER_KEY = "_ss_chart_n"     # reset by the router on every run, gives each chart a stable widget key
EXPANDED_CONFIG = {"displayModeBar": True, "displaylogo": False, "responsive": True,
                   "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
                   "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "stayscope_chart"}}


def _expanded(fig: go.Figure, title: str | None, caption: str | None, height: int, key: str) -> None:
    """Open the chart in a near full-screen dialog: same title and caption, taller, with zoom/pan/download."""
    @st.dialog(title or "Chart", width="large")
    def _dialog():
        st.html('<div class="ss-full"></div>')  # marker the CSS uses to size this dialog to the viewport
        hint = "Drag to zoom · double-click to reset · camera icon saves a PNG"
        st.caption(f"{caption} · {hint}" if caption else hint)
        big = go.Figure(fig)
        big.update_layout(height=None, autosize=True, font=dict(size=14))
        big.update_traces(textfont=dict(size=13), selector=dict(type="heatmap"))  # cell numbers readable full size
        big.update_xaxes(tickfont=dict(size=13))
        big.update_yaxes(tickfont=dict(size=13))
        # The chart stretches to fill a container that style.css sizes from the viewport (100vh minus the dialog's
        # header and caption), so the expanded chart always fits on one screen without scrolling.
        with st.container(key=f"ssfull-{key}"):
            st.plotly_chart(big, theme=None, height="stretch", key=f"{key}_full", config=EXPANDED_CONFIG)
        # Plotly measures its box only when it first draws; once the dialog has its full size, ask it to re-measure
        # (Streamlit's charts listen for window resize). Only this one event is dispatched; nothing else runs.
        st.html("<script>[150, 600].forEach(t => setTimeout(() => window.dispatchEvent(new Event('resize')), t));</script>",
                unsafe_allow_javascript=True)

    _dialog()


def show(fig: go.Figure, height: int = 320, key: str | None = None, expandable: bool = True) -> None:
    """Apply the template and render. The Streamlit theme is switched off so the template wins.

    Every chart gets an Expand button (top right of its card) that opens it near full screen, with the title and
    caption set by the `ui.chart_title` call just before it.
    """
    height = min(height, MAX_INLINE_HEIGHT)  # every inline chart fits one laptop screen; Expand gives full size
    fig.update_layout(template="stayscope", height=height, margin=dict(l=4, r=12, t=8, b=8, autoexpand=True))
    # automargin must be set on the figure itself: from the template it is not applied, which clips tick labels
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    title, caption, slot = st.session_state.pop(TITLE_KEY, (None, None, None))
    n = st.session_state.get(COUNTER_KEY, 0) + 1
    st.session_state[COUNTER_KEY] = n
    key = key or f"chart_{n}"
    if expandable:
        # In the right-hand slot of the chart title's row (ui.chart_title); a chart without a title gets its own row
        with (slot or st.container()):
            with st.container(key=f"ss-expand-{key}"):
                if st.button("Expand", icon=":material/open_in_full:", type="tertiary", key=f"{key}_expand",
                             help="Open this chart full screen"):
                    _expanded(fig, title, caption, height, key)
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "responsive": True})


def pct_axis(fig: go.Figure, axis: str = "y", decimals: int = 0) -> go.Figure:
    getattr(fig, f"update_{axis}axes")(tickformat=f".{decimals}%")
    return fig


def histogram_by_status(df: pd.DataFrame, col: str, target: str = "churned", bins: int = 30,
                        clip_q: float | None = None) -> go.Figure:
    """Share of customers per bin for churned vs retained (pre-binned with numpy: fast for 243K rows)."""
    x = df[col].astype(float)
    hi = x.quantile(clip_q) if clip_q else x.max()
    lo = x.min()
    discrete = df[col].nunique() <= 30 and np.allclose(x.dropna() % 1, 0)
    edges = (np.arange(lo - 0.5, hi + 1.5, 1.0) if discrete else np.linspace(lo, hi, bins + 1))
    fig = go.Figure()
    for status, flag in [("Retained", 0), ("Churned", 1)]:
        v = x[df[target] == flag]
        counts, _ = np.histogram(v.clip(upper=hi), bins=edges)
        share = counts / max(counts.sum(), 1)
        mids = (edges[:-1] + edges[1:]) / 2
        fig.add_bar(x=mids, y=share, name=status, marker_color=STATUS_COLORS[status],
                    hovertemplate=f"{status}<br>%{{x:.3g}}: %{{y:.1%}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0)
    return pct_axis(fig)


def churn_rate_bars(table: pd.DataFrame, col: str, overall: float, margin: float = 0.02) -> go.Figure:
    """Horizontal churn-rate bars, coloured only when a category is clearly above/below the overall rate."""
    t = table.sort_values("churn_rate")
    colors = [RED if r > overall + margin else TEAL if r < overall - margin else NEUTRAL_BAR for r in t["churn_rate"]]
    fig = go.Figure(go.Bar(
        x=t["churn_rate"], y=t[col].astype(str), orientation="h", marker_color=colors,
        text=[f"{r:.1%}" for r in t["churn_rate"]], textposition="outside", textfont=dict(color=TEXT_2),
        customdata=t["customers"], hovertemplate="%{y}: %{x:.1%} churn · %{customdata:,} customers<extra></extra>"))
    fig.add_vline(x=overall, line_dash="dash", line_color=MUTED, line_width=1,
                  annotation_text=f"overall {overall:.1%}", annotation_font_color=MUTED, annotation_position="top")
    fig.update_xaxes(range=[0, max(t["churn_rate"].max() * 1.25, overall * 1.3)])
    return pct_axis(fig, "x")


def heatmap(matrix: pd.DataFrame, diverging: bool = True, zmin: float = -1, zmax: float = 1) -> go.Figure:
    """Annotated correlation-style heatmap (lower triangle only)."""
    m = matrix.copy()
    mask = np.triu(np.ones(m.shape, dtype=bool), k=1)
    z = m.to_numpy(dtype=float, copy=True)  # pandas 3 can return a read-only view
    z[mask] = np.nan
    text = np.where(mask, "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(
        z=z, x=list(m.columns), y=list(m.index), zmin=zmin, zmax=zmax,
        colorscale=DIVERGING if diverging else SEQUENTIAL, text=text, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT), hoverongaps=False, xgap=2, ygap=2,
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=8, len=0.8)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return fig
