"""Cached access to everything the app reads. The app never retrains: it only loads saved artifacts.

Every loader returns None when its file is missing, so pages can show an empty state instead of crashing.
"""

import json

import pandas as pd
import streamlit as st

from churn import config, data, models

RESULTS = config.RESULTS_DIR

# ROC-AUC from the signal check in notebook 01, section 9 (5-fold HistGradientBoosting). It is not saved to a
# results file because it is a one-off diagnostic, so it is mirrored here. Update it if notebook 01 changes.
SIGNAL_CHECK = pd.DataFrame([
    ("S · all features", "S", 0.746), ("A · all features", "A", 0.999),
    ("A · without engagement (A_hard)", "A", 0.625), ("Old B · all features", "Old B", 0.543),
    ("Old B · shuffled labels", "Old B", 0.524),
], columns=["run", "dataset", "roc_auc"])


@st.cache_data(show_spinner=False)
def results_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def results_json(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def metrics() -> pd.DataFrame | None:
    return results_csv("metrics.csv")


def predictions() -> pd.DataFrame | None:
    return results_csv("predictions.csv")


def final_models() -> dict | None:
    return results_json("final_models.json")


@st.cache_data(show_spinner="Loading Dataset S…")
def raw_s() -> pd.DataFrame:
    return data.load_s()


@st.cache_data(show_spinner=False)
def raw_a() -> pd.DataFrame:
    return data.load_a()


@st.cache_data(show_spinner=False)
def raw_b() -> pd.DataFrame:
    return data.load_b()


@st.cache_data(show_spinner="Loading cleaned data…")
def clean_s() -> pd.DataFrame | None:
    return data.load_clean_s() if config.CLEAN_S.exists() else None


@st.cache_data(show_spinner=False)
def clean_a() -> pd.DataFrame | None:
    return data.load_clean_a() if config.CLEAN_A.exists() else None


@st.cache_data(show_spinner="Harmonizing A and S…")
def harmonized():
    """(a_h, s_h) with the shared columns from features.harmonize, or None if the clean files are missing."""
    from churn import features

    a, s = clean_a(), clean_s()
    if a is None or s is None:
        return None
    return features.harmonize(a, s)


@st.cache_resource(show_spinner="Loading model…")
def model(experiment: str, name: str):
    """A saved final model, or None if `python -m churn.models` has not been run."""
    if not config.model_path(experiment, name).exists():
        return None
    return models.load_model(experiment, name)


def test_metrics(experiments=None, threshold_type: str = "tuned", final_only: bool = True) -> pd.DataFrame | None:
    """Test-split metrics, optionally restricted to each model's final class-weight setting."""
    m = metrics()
    if m is None:
        return None
    m = m[(m["split"] == "test") & (m["threshold_type"] == threshold_type)]
    if experiments is not None:
        m = m[m["experiment"].isin(experiments)]
    fm = final_models() or {}
    if final_only and fm:
        chosen = {(e, k): v["class_weight"] for e, cfg in fm.items() for k, v in cfg["models"].items()}
        m = m[[chosen.get((e, k)) == cw for e, k, cw in zip(m["experiment"], m["model"], m["class_weight"])]]
    return m


def best_model(experiment: str = "S_full", metric: str = "roc_auc"):
    """(model name, mean test metric) of the best final model of an experiment."""
    m = test_metrics([experiment])
    if m is None or m.empty:
        return None, None
    g = m.groupby("model")[metric].mean()
    return g.idxmax(), float(g.max())


def missing_hint(*files: str) -> str:
    return "Missing: " + ", ".join(files) + ". Run the notebooks (or `python -m churn.models`) to create them."
