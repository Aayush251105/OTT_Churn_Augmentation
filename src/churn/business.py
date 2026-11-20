"""Budget simulator: turn churn scores into a retention-campaign decision.

For a decision threshold t, every customer with score >= t gets a retention offer.

    targeted          = customers with score >= t
    churners_reached  = targeted customers who would actually have churned
    saved             = churners_reached x success_rate
    value_saved       = saved x monthly_fee x months_retained
    cost              = targeted x offer_cost
    net_value         = value_saved - cost

Only churners who are reached and saved create value. Offers sent to customers who would have stayed anyway
are pure cost. The simulator works with any score (calibrated probabilities or class-weighted scores),
because it only ranks customers by the threshold.

`scale` lets results computed on the test split be expressed for the whole customer base
(scale = base size / test size).
"""

import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def simulate_budget(y_true, proba, offer_cost: float, success_rate: float, monthly_fee: float, months: int,
                    thresholds=THRESHOLDS, scale: float = 1.0) -> pd.DataFrame:
    """Net value of the campaign at every threshold. One row per threshold."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(proba)
    order = np.argsort(-p)
    p_sorted, y_sorted = p[order], y[order]
    cum_churners = np.cumsum(y_sorted)
    value_per_save = monthly_fee * months

    rows = []
    for t in thresholds:
        n = int(np.searchsorted(-p_sorted, -t, side="right"))  # number of customers with score >= t
        reached = int(cum_churners[n - 1]) if n > 0 else 0
        saved = reached * success_rate
        value = saved * value_per_save
        cost = n * offer_cost
        rows.append({"threshold": float(t), "targeted": n * scale, "churners_reached": reached * scale,
                     "saved": saved * scale, "cost": cost * scale, "value_saved": value * scale,
                     "net_value": (value - cost) * scale})
    out = pd.DataFrame(rows)
    out["share_targeted"] = out["targeted"] / (len(p) * scale)
    out["share_of_churners_reached"] = out["churners_reached"] / max(y.sum() * scale, 1)
    return out


def best_row(sim: pd.DataFrame) -> pd.Series:
    """The threshold with the highest net value."""
    return sim.loc[sim["net_value"].idxmax()]


def strategy(y_true, proba, mask, offer_cost: float, success_rate: float, monthly_fee: float, months: int,
             scale: float = 1.0) -> dict:
    """Cost and value of targeting an arbitrary set of customers (`mask` = True means the customer gets the offer)."""
    y = np.asarray(y_true).astype(bool)
    m = np.asarray(mask).astype(bool)
    targeted = m.sum()
    saved = (y & m).sum() * success_rate
    value = saved * monthly_fee * months
    cost = targeted * offer_cost
    return {"targeted": targeted * scale, "cost": cost * scale, "value_saved": value * scale,
            "net_value": (value - cost) * scale}
