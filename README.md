# Credit Card Default Prediction -- CSE374 Classification Project

End-to-end binary classification on the **UCI "Default of Credit Card Clients" (Taiwan, 2005)** dataset.
The project predicts whether a client will default on the next month's payment and compares two
models (Logistic Regression + XGBoost) through a single leakage-safe pipeline.

## Team

- Abdlrhman Hisham Ismail -- 2300343
- Mariam Maged Mohammad -- 2300670
- Asmaa Salaheldin Abdelhamid -- 2300181

Course: **CSE374s (UG2023) -- Machine Learning and Pattern Recognition**,
Faculty of Engineering, Ain Shams University.

## Repository Layout

```
.
├── notebooks/
│   └── credit_default_merged.ipynb   # main deliverable (run top-to-bottom)
├── dataset/
│   └── default of credit card clients.xls
├── imgs/                              # EDA figures + confusion matrices (auto-generated)
├── docs/
│   └── CSE374_Project.pdf             # course project brief
└── README.md
```

## Pipeline Summary

1. **Data loading** -- 30,000 raw rows -> 29,965 unique clients after dropping 35 exact duplicates.
2. **EDA (5 figures)** -- class distribution, `PAY_0` vs default, `LIMIT_BAL` distribution,
   demographic profile, correlation heatmap.
3. **Feature engineering** -- five hand-crafted features (`NUM_MONTHS_DELAYED`, `MAX_DELAY`,
   `UTILIZATION`, `PAYMENT_RATIO`, `BILL_TREND`).
4. **Split & preprocessing** -- stratified 70/15/15 split, `ColumnTransformer`
   (median-impute + `StandardScaler` for numerics, mode-impute + `OneHotEncoder` for
   categorical-like features).
5. **Imbalance handling** -- **SMOTE** applied only inside the training fold (wrapped in an
   `imblearn.Pipeline` for leakage-safe cross-validation).
6. **Modelling** -- two tuned models compared head-to-head:
   - **Logistic Regression** with SMOTE inside every CV fold.
   - **XGBoost** with native `scale_pos_weight` imbalance handling.
   - Both tuned with `GridSearchCV` on **average precision** (AUC-PR).
7. **Evaluation** -- held-out test set at the default `predict_proba > 0.5` cutoff,
   reporting Accuracy, Weighted F1, F1 (Default), Recall (Default), AUC-ROC, and AUC-PR.

## Running the Notebook

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost xlrd
jupyter lab notebooks/credit_default_merged.ipynb
```

Run all cells top-to-bottom. All figures are saved to `imgs/` and all results tables are
produced inline. No external configuration is required.

## Dataset

- **Source**: UCI Machine Learning Repository
  -- <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- **Target**: `default payment next month` (1 = default, 0 = no default).
- **Class balance**: ~22.12% default / ~77.88% no default (3.5 : 1 imbalance).
