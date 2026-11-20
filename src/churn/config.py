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

RAW_A = RAW_DIR / "netflix_customer_churn.csv"     # comparison + augmentation target (5K)
RAW_S = RAW_DIR / "train.csv"                      # main modelling dataset (243K, streaming)
RAW_S_DESCRIPTIONS = RAW_DIR / "data_descriptions.csv"
RAW_B = RAW_DIR / "netflix_large_user_data.xlsx"   # old dataset B: audit only, never modelled
RAW_REVIEWS = RAW_DIR / "netflix_reviews.csv"      # Netflix Google Play reviews (Kaggle, Phase 8 NLP)

CLEAN_A = INTERIM_DIR / "a_clean.csv"
CLEAN_S = INTERIM_DIR / "s_clean.csv"
CLEAN_REVIEWS = INTERIM_DIR / "reviews_clean.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42
SEEDS = [0, 1, 2, 3, 4]  # repeated splits for mean +/- std comparison

# ---------------------------------------------------------------- target / ids (same names in every dataset)
TARGET = "churned"
ID_COL = "customer_id"

# Dataset A's engagement columns, removed to build the harder A_hard setting
A_ENGAGEMENT = ["watch_hours", "last_login_days", "avg_watch_time_per_day"]

# ---------------------------------------------------------------- matching (report section 3.1.3)
K = 5
SIM_THRESHOLD = 0.6

# ---------------------------------------------------------------- split (report section 3.1.7)
SPLIT = (0.70, 0.15, 0.15)  # train, validation, test - stratified on TARGET

# ---------------------------------------------------------------- experiments
EXPERIMENTS = ["S_full", "A_full", "A_hard", "A_hard_augmented", "A_hard_random"]
MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

# ---------------------------------------------------------------- risk tiers (report section 3.1.8)
RISK_BINS = (0.3, 0.6)  # low < 0.3 <= medium < 0.6 <= high
RISK_LABELS = ("low", "medium", "high")

# ---------------------------------------------------------------- reviews NLP (Phase 8)
REVIEW_MIN_WORDS = 3         # shorter reviews ("good app") carry no topic
REVIEW_N_TOPICS = 12         # NMF topics on negative reviews (notebook 07: finest k whose theme shares are seed-stable)
REVIEW_MIN_MONTH_N = 100     # months with fewer reviews are left out of the monthly sentiment series

# ---------------------------------------------------------------- figures
FIG_DPI = 150


def experiment_path(name: str) -> Path:
    """Path of the processed CSV for one experiment, e.g. data/processed/A_hard_augmented.csv."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}; expected one of {EXPERIMENTS}")
    return PROCESSED_DIR / f"{name}.csv"


def model_path(experiment: str, model: str) -> Path:
    """Path of a saved model, e.g. models/S_full__random_forest.joblib."""
    return MODELS_DIR / f"{experiment}__{model}.joblib"


def ensure_dirs() -> None:
    """Create every output folder if it is missing."""
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
