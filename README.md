# StayScope: OTT churn prediction

Predict which streaming subscribers will cancel, explain why, and estimate what retention is worth.
This is a Foundations of Data Science course project by Aayush Gupta and Sukhraj Singh Nandhra.

## Research questions
1. Which classifier (Logistic Regression, Decision Tree, Random Forest, XGBoost) best predicts churn on realistic,
   imbalanced streaming data, and what drives churn?
2. **Augmentation study:** when a platform lacks engagement data, can behavioural features borrowed from similar
   customers in another dataset (cosine KNN, K = 5, similarity ≥ 0.6) recover the missing signal?

## Datasets
| | File | Rows | Churn | Role | Source |
|---|---|---|---|---|---|
| **S** | `data/raw/train.csv` | 243,787 | 18.1% | main modelling dataset, and the donor for augmentation | [Kaggle: Predictive Analytics for Customer Churn](https://www.kaggle.com/) (`train.csv` + `data_descriptions.csv`) |
| **A** | `data/raw/netflix_customer_churn.csv` | 5,000 | 50.3% | comparison dataset, and the augmentation target | [Kaggle: Netflix Customer Churn](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) |
| Old B | `data/raw/netflix_large_user_data.xlsx` | 1,000 | 53.9% | audit only (no churn signal) | [Kaggle: Netflix Customer Churn and Engagement](https://www.kaggle.com/datasets/dddtra/netflix-customer-churn-and-engagement-dataset) |

Download `train.csv` and `data_descriptions.csv` from the S dataset into `data/raw/`. `test.csv` has no labels and is not used.

## Pipeline
Data understanding → Cleaning → Feature mapping (A ↔ S) → Augmentation study → Modelling →
Interpretation, risk tiers & retention → StayScope dashboard → App-review NLP → Report

Experiments: `S_full` · `A_full` (ceiling) · `A_hard` (A without engagement columns) · `A_hard_augmented` ·
`A_hard_random` (control).

## Setup
Requires Python 3.10 or later (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
python -m ipykernel install --user --name ott-churn --display-name "Python (ott-churn)"
python -m churn.preprocessing     # writes the cleaned datasets to data/interim/
```
Pick the **Python (ott-churn)** kernel when you open the notebooks.

## Layout
```
data/raw/          source data (read-only)
data/interim/      cleaned datasets (generated)
data/processed/    experiment datasets (generated)
notebooks/         01-07, the analysis narrative with plots
src/churn/         all pipeline logic (imported by the notebooks and the app)
models/            trained models
reports/figures/   saved plots
reports/results/   metrics and other result files
app/               StayScope Streamlit dashboard
```

## Notebooks
| # | Notebook | Owner |
|---|---|---|
| 01 | Data understanding | Aayush |
| 02 | Cleaning & preprocessing | Aayush |
| 03 | Feature mapping (A ↔ S) | Sukhraj |
| 04 | Augmentation study | Sukhraj |
| 05 | Modelling & comparison | Aayush |
| 06 | Interpretation, risk tiers & retention | Sukhraj |
| 07 | Why customers leave (app-review NLP) | Sukhraj |

## Dashboard
```bash
streamlit run app/Home.py
```
