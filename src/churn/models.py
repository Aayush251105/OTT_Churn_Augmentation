"""Phase 5: the four churn classifiers, the 70/15/15 split, tuning and the experiment runner.

Workflow (the same for every experiment, so the results are comparable):
  1. `tune`: small hyper-parameter grid per model, fitted on train (seed 0) and scored on VALIDATION ROC-AUC
  2. `run_experiment`: for each seed x class-weight setting, fit on train, pick the F1-optimal threshold on
     validation, then score validation and test at both the default 0.5 and the tuned threshold
  3. `fit_final_models`: refit the chosen configuration (seed 0) and save models/<experiment>__<model>.joblib

Run `python -m churn.models` to retrain and save the final models from reports/results/final_models.json
(the .joblib files are gitignored because the Random Forest is about 30 MB).
"""

import json
from dataclasses import dataclass, field
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from churn import config, evaluate as ev, preprocessing as pp

TARGET, ID = config.TARGET, config.ID_COL
FINAL_MODELS_FILE = config.RESULTS_DIR / "final_models.json"


# ---------------------------------------------------------------- feature specs
@dataclass(frozen=True)
class FeatureSpec:
    """Which columns a model uses and how each group is preprocessed."""
    numeric: list
    categorical: list
    binary: list = field(default_factory=list)
    log_cols: list = field(default_factory=list)

    @property
    def columns(self) -> list:
        return [*self.numeric, *self.categorical, *self.binary]

    def drop(self, cols) -> "FeatureSpec":
        """A copy without `cols` (e.g. to build A_hard)."""
        cols = set(cols)
        keep = lambda xs: [c for c in xs if c not in cols]
        return FeatureSpec(keep(self.numeric), keep(self.categorical), keep(self.binary), keep(self.log_cols))

    def add(self, numeric=(), categorical=(), binary=()) -> "FeatureSpec":
        """A copy with extra columns (e.g. transferred features for A_hard_augmented)."""
        return FeatureSpec([*self.numeric, *numeric], [*self.categorical, *categorical],
                           [*self.binary, *binary], list(self.log_cols))


SPEC_S = FeatureSpec(pp.S_FEATURES_NUMERIC, pp.S_FEATURES_CATEGORICAL, pp.S_FEATURES_BINARY)
SPEC_A = FeatureSpec(pp.A_FEATURES_NUMERIC, pp.A_FEATURES_CATEGORICAL, log_cols=pp.A_LOG_COLS)
SPEC_A_HARD = SPEC_A.drop(config.A_ENGAGEMENT)

# ---------------------------------------------------------------- models and grids
DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0},
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 50},
    "random_forest": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20},
    "xgboost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4},
}
PARAM_GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [20, 100]},
    "random_forest": {"n_estimators": [300], "max_depth": [8, 12, 16], "min_samples_leaf": [5, 20]},
    "xgboost": {"n_estimators": [400], "learning_rate": [0.03, 0.1], "max_depth": [3, 4, 6]},
}
CLASS_WEIGHTS = [None, "balanced"]


def make_model(name: str, params: dict | None = None, class_weight=None, pos_weight: float = 1.0,
               seed: int = config.RANDOM_STATE):
    """One untrained classifier. `class_weight="balanced"` up-weights churners
    (XGBoost has no class_weight, so it uses scale_pos_weight = negatives / positives instead)."""
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed, **p)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight=class_weight, random_state=seed, **p)
    if name == "random_forest":
        return RandomForestClassifier(class_weight=class_weight, n_jobs=-1, random_state=seed, **p)
    if name == "xgboost":
        return XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                             scale_pos_weight=pos_weight if class_weight == "balanced" else 1.0,
                             random_state=seed, **p)
    raise ValueError(f"Unknown model {name!r}")


def make_pipeline(name: str, spec: FeatureSpec, y_train=None, params=None, class_weight=None,
                  seed: int = config.RANDOM_STATE) -> Pipeline:
    """Preprocessor (fitted on train only, inside the pipeline) + classifier."""
    pos_weight = 1.0
    if y_train is not None:
        y = np.asarray(y_train)
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    return Pipeline([
        ("prep", pp.make_preprocessor(spec.numeric, spec.categorical, spec.binary, spec.log_cols)),
        ("model", make_model(name, params, class_weight, pos_weight, seed)),
    ])


def build_models(numeric, categorical, binary=(), log_cols=(), class_weight=None, params=None,
                 seed: int = config.RANDOM_STATE) -> dict[str, Pipeline]:
    """All four untrained pipelines for one feature set (the interface agreed in CLAUDE.md)."""
    spec = FeatureSpec(list(numeric), list(categorical), list(binary), list(log_cols))
    params = params or {}
    return {m: make_pipeline(m, spec, params=params.get(m), class_weight=class_weight, seed=seed)
            for m in config.MODELS}


def stratified_split(df: pd.DataFrame, seed: int = config.RANDOM_STATE):
    """70 / 15 / 15 train / validation / test split, stratified on the target."""
    train_frac, val_frac, test_frac = config.SPLIT
    train, rest = train_test_split(df, train_size=train_frac, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                                 stratify=rest[TARGET], random_state=seed)
    return train, val, test


# ---------------------------------------------------------------- tuning
def tune(df: pd.DataFrame, spec: FeatureSpec, seed: int = 0, grids=PARAM_GRIDS):
    """Grid-search each model on the validation split (ROC-AUC, class_weight=None).

    Returns (results DataFrame, best params dict). The test split is never touched here.
    """
    train, val, _ = stratified_split(df, seed)
    rows = []
    for name, grid in grids.items():
        keys = list(grid)
        for values in product(*grid.values()):
            params = dict(zip(keys, values))
            pipe = make_pipeline(name, spec, train[TARGET], params, seed=seed).fit(train[spec.columns], train[TARGET])
            proba = pipe.predict_proba(val[spec.columns])[:, 1]
            rows.append({"model": name, "params": json.dumps(params),
                         "val_roc_auc": ev.metrics_from_proba(val[TARGET], proba)["roc_auc"]})
    res = pd.DataFrame(rows)
    best = {m: json.loads(res.loc[res[res["model"] == m]["val_roc_auc"].idxmax(), "params"]) for m in grids}
    return res, best


# ---------------------------------------------------------------- experiments
def run_experiment(df: pd.DataFrame, experiment: str, spec: FeatureSpec, params: dict,
                   seeds=config.SEEDS, class_weights=CLASS_WEIGHTS, keep_test_proba_seed: int = 0):
    """Fit every model for every seed x class-weight setting.

    Returns (metrics DataFrame, test-probabilities DataFrame for `keep_test_proba_seed`).
    Each metrics row is one (model, class_weight, seed, split, threshold_type) combination.
    """
    metric_rows, proba_rows = [], []
    for seed in seeds:
        train, val, test = stratified_split(df, seed)
        for name, cw in product(config.MODELS, class_weights):
            pipe = make_pipeline(name, spec, train[TARGET], params.get(name), cw, seed)
            pipe.fit(train[spec.columns], train[TARGET])
            p_val = pipe.predict_proba(val[spec.columns])[:, 1]
            p_test = pipe.predict_proba(test[spec.columns])[:, 1]
            tuned = ev.best_threshold(val[TARGET], p_val, "f1")
            cw_label = cw or "none"
            for split, y, p in [("val", val[TARGET], p_val), ("test", test[TARGET], p_test)]:
                for t_type, t in [("default", 0.5), ("tuned", tuned)]:
                    metric_rows.append({"experiment": experiment, "model": name, "class_weight": cw_label,
                                        "seed": seed, "split": split, "threshold_type": t_type, "threshold": t,
                                        **ev.metrics_from_proba(y, p, t)})
            if seed == keep_test_proba_seed:
                proba_rows.append(pd.DataFrame({"experiment": experiment, "model": name, "class_weight": cw_label,
                                                ID: test[ID].to_numpy(), "y_true": test[TARGET].to_numpy(),
                                                "proba": p_test}))
    return pd.DataFrame(metric_rows), pd.concat(proba_rows, ignore_index=True)


def choose_final(metrics: pd.DataFrame) -> pd.DataFrame:
    """For each experiment and model, choose the class-weight setting with the best mean VALIDATION F1 at the
    tuned threshold, and report that setting's mean tuned threshold. The test split is not used to choose."""
    m = metrics[(metrics["split"] == "val") & (metrics["threshold_type"] == "tuned")]
    g = m.groupby(["experiment", "model", "class_weight"])[["f1", "threshold"]].mean().reset_index()
    best = g.loc[g.groupby(["experiment", "model"])["f1"].idxmax()]
    return best.rename(columns={"f1": "val_f1", "threshold": "threshold"}).reset_index(drop=True)


# ---------------------------------------------------------------- final models
def save_final_config(final: pd.DataFrame, params_by_experiment: dict, specs: dict) -> None:
    """Write reports/results/final_models.json: everything needed to rebuild and use the final models."""
    out = {}
    for exp, g in final.groupby("experiment"):
        spec = specs[exp]
        out[exp] = {
            "features": {"numeric": spec.numeric, "categorical": spec.categorical,
                         "binary": spec.binary, "log_cols": spec.log_cols},
            "models": {r.model: {"class_weight": r.class_weight, "threshold": round(float(r.threshold), 2),
                                 "val_f1": round(float(r.val_f1), 4), "params": params_by_experiment[exp][r.model]}
                       for r in g.itertuples()},
        }
    existing = json.loads(FINAL_MODELS_FILE.read_text()) if FINAL_MODELS_FILE.exists() else {}
    existing.update(out)
    FINAL_MODELS_FILE.write_text(json.dumps(existing, indent=2))


def load_final_config() -> dict:
    """The saved final_models.json as a dict."""
    return json.loads(FINAL_MODELS_FILE.read_text())


def fit_final_models(experiment: str, df: pd.DataFrame, seed: int = 0) -> dict[str, Pipeline]:
    """Refit the chosen configuration of every model on the seed-0 training split and save it to models/."""
    cfg = load_final_config()[experiment]
    spec = FeatureSpec(**cfg["features"])
    train, _, _ = stratified_split(df, seed)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for name, m in cfg["models"].items():
        cw = None if m["class_weight"] == "none" else m["class_weight"]
        pipe = make_pipeline(name, spec, train[TARGET], m["params"], cw, seed)
        pipe.fit(train[spec.columns], train[TARGET])
        joblib.dump(pipe, config.model_path(experiment, name), compress=3)
        fitted[name] = pipe
    return fitted


def load_model(experiment: str, model: str) -> Pipeline:
    """Load a saved final model (run `python -m churn.models` first if the file is missing)."""
    return joblib.load(config.model_path(experiment, model))


if __name__ == "__main__":
    from churn import data

    sources = {"S_full": data.load_clean_s, "A_full": data.load_clean_a}
    for exp in load_final_config():
        if exp in sources:
            fit_final_models(exp, sources[exp]())
            print(f"saved final models for {exp}")
        else:
            print(f"skipped {exp}: build it with notebook 05 (needs data/processed/{exp}.csv)")
