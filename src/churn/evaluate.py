"""Metrics, threshold tuning and summaries for the churn classifiers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def metrics_from_proba(y_true, proba, threshold: float = 0.5) -> dict:
    """All report metrics for one set of predicted churn probabilities at a given decision threshold.

    ROC-AUC and PR-AUC don't depend on the threshold; the other metrics do.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Metrics for a fitted classifier (anything with `predict_proba`)."""
    return metrics_from_proba(y, model.predict_proba(X)[:, 1], threshold)


def confusion(y_true, proba, threshold: float = 0.5) -> dict:
    """Confusion-matrix counts as a dict: tn, fp, fn, tp."""
    tn, fp, fn, tp = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_curve(y_true, proba, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Precision, recall, F1 and the share of customers flagged as churners, for each decision threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for t in grid:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "share_flagged": y_pred.mean(),
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, metric: str = "f1", grid=THRESHOLD_GRID) -> float:
    """Decision threshold that maximises `metric` (use it on the VALIDATION split, never on test)."""
    curve = threshold_curve(y_true, proba, grid)
    return float(curve.loc[curve[metric].idxmax(), "threshold"])


def summarize(metrics: pd.DataFrame, by=("experiment", "model", "class_weight", "threshold_type"),
              split: str = "test", cols=METRICS) -> pd.DataFrame:
    """Mean and std over seeds for one split, e.g. the test-set table in notebook 05."""
    m = metrics[metrics["split"] == split]
    g = m.groupby(list(by), dropna=False)[list(cols)]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out[[f"{c}_{s}" for c in cols for s in ("mean", "std")]].reset_index()


def format_mean_std(summary: pd.DataFrame, cols=METRICS, digits: int = 3) -> pd.DataFrame:
    """Turn *_mean / *_std column pairs into readable '0.752 ± 0.003' strings."""
    out = summary.drop(columns=[c for c in summary.columns if c.endswith(("_mean", "_std"))]).copy()
    for c in cols:
        out[c] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"].fillna(0))]
    return out
