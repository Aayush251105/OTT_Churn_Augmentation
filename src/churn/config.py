"""Project-wide settings: paths, seeds and method parameters.

Shared file - coordinate with your teammate before changing values here.
Never hard-code any of these values elsewhere; import them from this module.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_A = RAW_DIR / "netflix_customer_churn.csv"
RAW_B = RAW_DIR / "netflix_large_user_data.xlsx"

CLEAN_A = INTERIM_DIR / "a_clean.csv"
CLEAN_B = INTERIM_DIR / "b_clean.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42
SEEDS = [0, 1, 2, 3, 4]  # repeated splits for mean +/- std comparison

# ---------------------------------------------------------------- target / ids
TARGET = "churned"
ID_COL = "customer_id"

# ---------------------------------------------------------------- matching (report section 3.1.3)
K = 5
SIM_THRESHOLD = 0.6

# ---------------------------------------------------------------- split (report section 3.1.7)
SPLIT = (0.70, 0.15, 0.15)  # train, validation, test - stratified on TARGET

# ---------------------------------------------------------------- experiments
EXPERIMENTS = ["A_full", "A_matched", "A_augmented", "A_random"]
MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

# ---------------------------------------------------------------- risk tiers (report section 3.1.8)
RISK_BINS = (0.3, 0.6)  # low < 0.3 <= medium < 0.6 <= high
RISK_LABELS = ("low", "medium", "high")

# ---------------------------------------------------------------- figures
FIG_DPI = 150


def experiment_path(name: str) -> Path:
    """Path of the processed CSV for one experiment, e.g. data/processed/A_augmented.csv."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}; expected one of {EXPERIMENTS}")
    return PROCESSED_DIR / f"{name}.csv"


def model_path(experiment: str, model: str) -> Path:
    """Path of a saved model, e.g. models/A_full__random_forest.joblib."""
    return MODELS_DIR / f"{experiment}__{model}.joblib"


def ensure_dirs() -> None:
    """Create every output folder if it is missing."""
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
