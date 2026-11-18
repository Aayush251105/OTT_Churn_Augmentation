# OTT Churn Augmentation

Netflix customer churn prediction using **similarity-based dataset augmentation**.
This is a Foundations of Data Science course project by Aayush Gupta and Sukhraj Singh Nandhra.

**Research question:** if we add supplementary engagement features to a churn dataset, transferred from
similar customer profiles in a second dataset (cosine KNN, K = 5, similarity ≥ 0.6), does churn
classification improve compared with using the original dataset alone?

## Datasets
| | File | Rows | Source |
|---|---|---|---|
| A (primary) | `data/raw/netflix_customer_churn.csv` | 5,000 | [Kaggle: Netflix Customer Churn](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) |
| B (supplementary) | `data/raw/netflix_large_user_data.xlsx` | 1,000 | [Kaggle: Netflix Customer Churn and Engagement](https://www.kaggle.com/datasets/dddtra/netflix-customer-churn-and-engagement-dataset) |

## Pipeline
Clean → harmonize common features → select B-only features → cosine KNN matching (K = 5, threshold 0.6)
→ similarity-weighted transfer → train Logistic Regression, Decision Tree, Random Forest and XGBoost
→ compare original vs augmented → feature importance, risk tiers, retention suggestions.

Experiments: `A_full`, `A_matched` (fair baseline), `A_augmented`, `A_random` (random-match control).

## Setup
Requires Python 3.10 or later (developed on 3.12).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name ott-churn --display-name "Python (ott-churn)"
```
Pick the **Python (ott-churn)** kernel when you open the notebooks.

## Layout
```
data/raw/          source data (read-only)
data/interim/      cleaned A and B
data/processed/    experiment datasets
notebooks/         01-07, the analysis narrative with plots
src/churn/         all pipeline logic (imported by the notebooks and the app)
models/            trained models
reports/figures/   saved plots
reports/results/   metrics and other result CSVs
app/               Streamlit dashboard
```

## Notebooks
| # | Notebook | Owner |
|---|---|---|
| 01 | Data understanding | Aayush |
| 02 | Cleaning & preprocessing | Aayush |
| 03 | Feature mapping & association | Sukhraj |
| 04 | Similarity matching & augmentation | Sukhraj |
| 05 | EDA on augmented data | Sukhraj |
| 06 | Modelling & comparison | Aayush |
| 07 | Interpretation, risk tiers & retention | Sukhraj |

## Dashboard
```bash
streamlit run app/Home.py
```
