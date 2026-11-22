# StayScope: churn prediction, cross-dataset augmentation and review mining for video streaming

**Course:** Foundations of Data Science, Shiv Nadar University
**Authors:** Aayush Gupta (2310110645) and Sukhraj Singh Nandhra (2310110310)

StayScope predicts which streaming subscribers will cancel, explains why, and estimates what a retention campaign is
worth. It answers three questions:

1. **Prediction.** Which of Logistic Regression, Decision Tree, Random Forest and XGBoost best predicts churn on
   realistic, imbalanced streaming data, and what drives churn?
2. **Augmentation.** When a dataset has no engagement data, can behavioural features borrowed from similar customers
   in another dataset (cosine KNN matching) recover the missing signal?
3. **Reasons.** What reasons do customers give for leaving in Netflix app reviews, and do they match what the models
   find?

All results are presented in **StayScope**, a nine-page Streamlit dashboard.

---

## Contents
1. [Quick start: run the dashboard](#1-quick-start-run-the-dashboard)
2. [Reproduce the analysis](#2-reproduce-the-analysis)
3. [Repository structure](#3-repository-structure)
4. [Datasets](#4-datasets)
5. [What we did and what we found](#5-what-we-did-and-what-we-found)

---

## 1. Quick start: run the dashboard

All datasets and result files are included in the repository, so nothing needs to be downloaded.
Python 3.10 or later is required (the project was developed and tested on Python 3.12).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # macOS / Linux

# 2. Install the dependencies (this also installs the project package `churn` from src/)
pip install -r requirements.txt

# 3. Build the files that are not stored in git (1–2 minutes in total)
python -m churn.preprocessing      # cleaned datasets       -> data/interim/
python -m churn.matching           # augmentation datasets  -> data/processed/
python -m churn.models             # trained final models   -> models/

# 4. Start the dashboard
streamlit run app/Home.py
```

The dashboard opens at http://localhost:8501. It only reads saved results and never retrains a model.

**Dashboard pages**

| Group | Page | What it shows |
|---|---|---|
| Overview | Home | Headline results and the project pipeline |
| | Data explorer | Distributions and churn rates for datasets S, A and old B, with the data-audit finding |
| | Preprocessing | Every cleaning step with row counts, the dropped derived columns, and the A ↔ S column mapping |
| Modelling | Model comparison | The four models on S and A, ROC and precision-recall curves, and a threshold slider |
| | Churn predictor | Enter a customer's details and get a churn probability, risk tier and the reasons behind it |
| | Augmentation study | The pre-check, the KNN matching, and the result against the A_full ceiling |
| Insights | Risk & retention | Risk tiers, churn drivers (Random Forest importance and SHAP) and retention rules |
| | Budget simulator | Net value of a retention campaign as a function of the decision threshold |
| | Why customers leave | Sentiment over time, complaint themes, reasons given for cancelling, and example reviews |

---

## 2. Reproduce the analysis

The analysis is written up in seven Jupyter notebooks, one per phase. **They are saved with their outputs**, so they
can be read directly on GitHub or in Jupyter without running anything.

To re-run them, register the environment as a Jupyter kernel and select **Python (ott-churn)** in each notebook:

```bash
python -m ipykernel install --user --name ott-churn --display-name "Python (ott-churn)"
jupyter notebook notebooks/
```

Run the step-3 commands from the quick start first, then run the notebooks in order:

| # | Notebook | Contents |
|---|---|---|
| 01 | `01_data_understanding.ipynb` | Schema, data quality, distributions, churn rates and a signal check of every dataset |
| 02 | `02_cleaning_preprocessing.ipynb` | Cleaning steps, dropped derived columns and the leakage-free preprocessing pipeline |
| 03 | `03_feature_mapping.ipynb` | Mapping the columns shared by A and S, and choosing the features to transfer |
| 04 | `04_augmentation_study.ipynb` | Transferability pre-check, KNN matching, feature transfer and the random control |
| 05 | `05_modelling.ipynb` | Tuning, five-seed comparison of four models, class weighting and threshold tuning, all experiments |
| 06 | `06_interpretation.ipynb` | Feature importance and SHAP, risk tiers and retention rules |
| 07 | `07_reviews_nlp.ipynb` | Review cleaning, sentiment classifier, NMF topics, complaint themes and the event timeline |

Notebook 05 takes the longest, because it trains every model on five random splits for every experiment.

The review results can also be rebuilt from the command line with `python -m churn.reviews`.

---

## 3. Repository structure

```
├── app/                    StayScope dashboard (Streamlit)
│   ├── Home.py             entry point and page navigation
│   ├── views/              one script per dashboard page
│   ├── components/         shared layout, chart styling and cached data loaders
│   └── assets/             stylesheet and logo
├── data/
│   ├── raw/                original datasets, never modified
│   ├── interim/            cleaned datasets (generated by churn.preprocessing)
│   └── processed/          augmentation experiment datasets (generated by churn.matching)
├── models/                 trained final models (generated by churn.models)
├── notebooks/              the seven analysis notebooks, with outputs
├── reports/
│   ├── figures/            every figure from the notebooks, named by notebook number
│   └── results/            metrics, predictions and other result files read by the dashboard
├── src/churn/              all pipeline code, imported by the notebooks and the dashboard
├── .streamlit/             dashboard theme
├── requirements.txt        Python dependencies
└── pyproject.toml          makes src/churn installable as a package
```

**Modules in `src/churn/`**

| Module | Purpose |
|---|---|
| `config.py` | Every path, random seed and method parameter in one place |
| `data.py` | Loaders for each dataset with consistent column names (`customer_id`, `churned`) |
| `eda.py`, `plotting.py` | Data-quality summaries, association measures and the shared figure style |
| `preprocessing.py` | Cleaning with a step-by-step log, and the scikit-learn preprocessing pipeline |
| `features.py` | Harmonising the columns shared by datasets A and S |
| `matching.py` | Transferability pre-check, cosine KNN matching, feature transfer and the random control |
| `models.py`, `evaluate.py` | Model definitions, tuning, the five-seed protocol, threshold tuning and metrics |
| `risk.py` | SHAP explanations, risk tiers and retention suggestions |
| `business.py` | Retention budget simulator |
| `reviews.py` | Review cleaning, sentiment model, topic model, themes and the event timeline |

Code lives in `src/churn/`. The notebooks and the dashboard import from it, so the dashboard shows the outputs of the
same code that the notebooks run.

---

## 4. Datasets

| | File | Rows | Target | Role |
|---|---|---|---|---|
| **S** | `data/raw/train.csv` | 243,787 | 18.1% churn | Main modelling dataset, and the donor for augmentation |
| **A** | `data/raw/netflix_customer_churn.csv` | 5,000 | 50.3% churn | Comparison dataset, and the augmentation target |
| **Old B** | `data/raw/netflix_large_user_data.xlsx` | 1,000 | 53.9% churn | Data audit only. It has no usable churn signal, so it was excluded |
| **Reviews** | `data/raw/netflix_reviews.csv` | 165,694 | 1–5 stars | Google Play reviews of the Netflix app, Sep 2018 – Nov 2026 |

Sources (Kaggle):
[Predictive Analytics for Customer Churn](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset) (S; column descriptions in `data/raw/data_descriptions.csv`),
[Netflix Customer Churn](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) (A),
[Netflix Customer Churn and Engagement](https://www.kaggle.com/datasets/dddtra/netflix-customer-churn-and-engagement-dataset) (old B),
[Netflix Reviews (Daily Updated)](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated) (reviews).
Dataset S's `test.csv` has no labels and is not used.

---

## 5. What we did and what we found

**1. Data understanding and audit.** We profiled all four datasets and ran a quick signal check on each (five-fold
ROC-AUC). Old dataset B, the donor in our original proposal, cannot predict its own churn label (0.543, against 0.524
for shuffled labels). Dataset A is almost perfectly predictable (0.999). We therefore changed the plan: S became the
main dataset, and augmentation was tested on *A_hard*, dataset A with its three engagement columns removed (0.625).

**2. Cleaning.** No missing values or duplicates were found. Derived columns were dropped (`total_charges` =
account age × monthly charges in S, `avg_watch_time_per_day` in A), and all encoding and scaling is fitted on the
training split only to prevent leakage.

**3. Feature mapping.** Five columns appear in both A and S (plan, device, genre, payment method, gender). We mapped
them to common categories and chose the five S behavioural columns most associated with churn as transfer features.

**4. Augmentation study.** Before matching, a pre-check measures how well the shared columns predict each transfer
feature. R² was about 0 for all of them, which means matching cannot transfer useful information. We still ran the
full study: every A customer was matched to its five most similar S customers (cosine similarity ≥ 0.6) and given
their similarity-weighted feature values, and a control copied features from random S customers instead.

**5. Modelling.** Four models were run on five experiments (`S_full`, `A_full`, `A_hard`, `A_hard_augmented`,
`A_hard_random`), using a stratified 70/15/15 split, five random seeds, class weighting on and off, and a decision
threshold tuned on validation.

| Finding | Result |
|---|---|
| Best models on S | Logistic Regression, XGBoost and Random Forest are tied (test ROC-AUC 0.745–0.750, PR-AUC about 0.40 against an 18.1% base rate). Decision Tree is weaker (0.718). |
| Class imbalance | At the default 0.5 threshold, recall is only 3–12%. Class weighting or a tuned threshold each lift it to about 0.6; using both adds nothing. |
| Augmentation | No gain. KNN-augmented, baseline and random-control results all lie within ±0.015 ROC-AUC, inside the seed noise, exactly as the pre-check predicted. Real engagement data (A_full) is worth about +0.33. |

**6. Interpretation.** Random Forest importance and SHAP agree. Account age is the strongest driver in S: customers
in their first year churn at 32%, against 12% after five years. Viewing behaviour comes next (session length,
downloads, weekly hours), then price and support tickets. In A, days since last login and watch hours dominate. We
built three risk tiers (the high tier churns at 65.5%, 3.6 times the average) and six data-backed retention rules.

**7. Dashboard and budget simulator.** StayScope presents every phase interactively. The budget simulator computes the
net value of a retention offer at every threshold. With default assumptions, the best threshold (0.23) targets 28% of
customers and nets about $221,000 across the customer base, while targeting everyone loses money.

**8. Why customers leave.** We cleaned 165,694 Netflix app reviews (145,174 kept), trained a TF-IDF + Logistic
Regression sentiment model (test ROC-AUC 0.956), and found complaint themes with NMF topic modelling. App crashes are
the most common complaint (26% of negative reviews), but among reviewers who say they are cancelling, price and
billing (46%) and the content library (25%) dominate. Complaint themes rise after real events such as price increases
and the paid password-sharing rollout.

**Limitations.** Datasets S and A appear to be synthetic, so absolute scores may not carry over to real platforms.
Reviews cannot be linked to individual customers, so the comparison with the models is qualitative. Retention rules are
associations, and the budget results depend on an assumed offer success rate.
