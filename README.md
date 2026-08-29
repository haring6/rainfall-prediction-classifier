# Rainfall Prediction Classifier — Melbourne, Australia

A supervised machine learning project that predicts same-day rainfall in the Melbourne metropolitan area using historical weather observations from the Australian Bureau of Meteorology. Built as part of an applied machine learning specialization, then extended with independent analysis of feature importance, class imbalance, and model comparison.

## Overview

Rainfall prediction is a classic imbalanced binary classification problem: rain is the minority class, and a naive "always predict no rain" baseline can look deceptively accurate. This project builds a full preprocessing → modeling → evaluation pipeline that goes beyond raw accuracy to properly assess predictive performance, using the [Kaggle "Rain in Australia" dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) (sourced from the Bureau of Meteorology).

**Key design decision:** the original dataset frames the task as predicting *tomorrow's* rainfall using *today's* full-day observations — but several of those features (e.g. `MaxTemp`, `Evaporation`, `WindGustDir`) aren't actually knowable until the day is over, which makes them unusable for a genuine next-day forecast. This project reframes the problem to predict **today's rainfall using data available up to and including yesterday**, which is both leakage-free and practically useful (e.g., "should I bike to work today?").

## Dataset

- **Source:** Bureau of Meteorology daily weather observations, 2008–2017, via Kaggle
- **Scope:** Filtered to Melbourne, Melbourne Airport, and Watsonia (geographically adjacent stations, <20km apart) to build a spatially consistent, localized model rather than a one-size-fits-all national model
- **Size after cleaning:** 7,557 daily observations across 23 original features
- **Target:** Binary — did it rain today (Yes/No)
- **Class balance:** ~76% No / ~24% Yes — a meaningfully imbalanced target that shapes the entire evaluation strategy

## Methodology

1. **Data cleaning** — dropped rows with missing values (retaining 56,420 of 145,460 national records before location filtering); columns with excessive missingness (`Sunshine`, `Cloud`) were considered but ultimately dropped rows rather than imputed, given sufficient remaining sample size
2. **Leakage correction** — renamed and reframed the target to eliminate features that wouldn't be available at prediction time
3. **Feature engineering** — extracted a categorical `Season` feature from the date column, hypothesizing seasonal weather patterns would carry predictive signal
4. **Preprocessing pipeline** — `ColumnTransformer` combining `StandardScaler` for numeric features and `OneHotEncoder` for categorical features, wrapped in a single `sklearn.Pipeline` to prevent train/test leakage during cross-validation
5. **Model selection** — `GridSearchCV` with `StratifiedKFold` (5-fold, stratified to preserve class balance in every fold) over two classifiers:
   - **Random Forest Classifier** — tuned over `n_estimators`, `max_depth`, `min_samples_split`
   - **Logistic Regression** — tuned over penalty type (`l1`/`l2`), solver, and class weighting
6. **Evaluation** — accuracy alone is insufficient given class imbalance, so precision, recall (true positive rate), F1-score, and confusion matrices were used as the primary comparison metrics
7. **Feature importance analysis** — extracted from the fitted Random Forest to identify which meteorological variables carry the most predictive signal

## Results

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Test Accuracy | 84% | 83% |
| True Positive Rate (Recall, "Yes") | 51% | 51% |
| Precision ("Yes") | 75% | — |

The two models perform almost identically — both correctly flag only about half of actual rain days despite ~84% overall accuracy, which is the expected signature of an imbalanced classification problem: accuracy is inflated by strong performance on the majority ("No rain") class, while the true positive rate on the minority class remains the harder, more informative benchmark.

**Most predictive feature:** `Humidity3pm` — afternoon humidity was consistently the strongest single predictor of rain, ahead of pressure and sunshine duration, which aligns with meteorological intuition (rising humidity precedes precipitation events).

## Key Takeaways

- On imbalanced classification tasks, accuracy alone can mask poor minority-class performance — recall/TPR is the metric that actually matters here
- A simple, interpretable Logistic Regression model performed within 1 percentage point of a tuned Random Forest, suggesting the underlying signal is close to linearly separable given good features
- Afternoon humidity is a stronger rain predictor than morning readings or wind-related features, at least in this geographic region
- Future improvement paths: SMOTE or class-weighting to directly address imbalance, richer feature engineering (e.g. lagged rainfall, pressure trends), and testing gradient-boosted models (XGBoost/LightGBM)

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

## Repository Structure

```
rainfall-prediction-classifier/
├── README.md
├── FinalProject_AUSWeather.ipynb   # Full analysis notebook, executed end-to-end
└── requirements.txt
```

## Running Locally

```bash
git clone https://github.com/haring6/rainfall-prediction-classifier.git
cd rainfall-prediction-classifier
pip install -r requirements.txt
jupyter notebook FinalProject_AUSWeather.ipynb
```

## Acknowledgements

Dataset originally compiled by the Australian Bureau of Meteorology, distributed via Kaggle. Base notebook structure adapted from an IBM Skills Network course project; all analysis, interpretation, and written findings above are original.
