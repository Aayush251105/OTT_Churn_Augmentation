"""Shared matplotlib/seaborn style for the notebooks.

Colours match the StayScope design tokens, so the figures in the report and in the app look consistent.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from churn import config

CHURN = "#E5484D"      # churned / high risk
RETAINED = "#2BB3A3"   # retained / low risk
MEDIUM = "#F5A524"     # medium risk
NEUTRAL = "#71717A"
CHURN_PALETTE = {0: RETAINED, 1: CHURN, "No": RETAINED, "Yes": CHURN}

MODEL_COLORS = {
    "logistic_regression": "#7C9CF5",
    "decision_tree": "#A78BFA",
    "random_forest": "#2BB3A3",
    "xgboost": "#F5A524",
}
EXPERIMENT_COLORS = {
    "A_full": "#71717A",
    "A_matched": "#7C9CF5",
    "A_augmented": "#2BB3A3",
    "A_random": "#A1A1AA",
}

DIVERGING = sns.diverging_palette(10, 175, s=75, l=55, as_cmap=True)  # red <-> teal


def set_style() -> None:
    """Apply the project's plot style. Call once at the top of each notebook."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": config.FIG_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",  # Arial has no semibold face
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "grid.alpha": 0.4,
        "legend.frameon": False,
    })


def savefig(fig: plt.Figure, name: str) -> None:
    """Save a figure as reports/figures/<name>.png (name like '01_target_balance')."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")
