# 📉 Telco Customer Churn Prediction

**Machine Learning Classification | XGBoost · SHAP · Scikit-learn · Python**

> Predicting which telecom customers will leave — and why — using interpretable ML models, SHAP explainability, and a business ROI framework.

---

## 🏆 Key Results

| Model | AUC-ROC | F1 Score | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.843 ± 0.008 | 0.601 ± 0.012 | 0.554 ± 0.015 | 0.651 ± 0.014 |
| Random Forest | 0.827 ± 0.011 | 0.578 ± 0.014 | 0.491 ± 0.018 | 0.634 ± 0.016 |
| **XGBoost (tuned) ★** | **0.861 ± 0.007** | **0.631 ± 0.010** | **0.581 ± 0.013** | **0.672 ± 0.011** |

> ★ Final model selected. All scores are 5-fold stratified cross-validation (mean ± std). Classification threshold optimised to **0.35** to maximise recall — catching a churner is worth more than avoiding a false alarm.

---

## 📋 Project Overview

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a full ML pipeline to identify at-risk customers before they leave, enabling targeted retention campaigns that are far cheaper than acquisition.

**Research Questions:**
1. Which customer characteristics and behavioural signals best predict churn?
2. Do tree-based ML models outperform logistic regression on this dataset?
3. Does the relationship between features and churn vary across customer segments?
4. Can we quantify the business value (in dollars) of deploying this model?

---

## 📂 Repository Structure

```
telco-churn-prediction/
│
├── data/
│   ├── telco_churn.csv              # Raw dataset (download from Kaggle)
│   ├── X_train.csv                  # Preprocessed training features
│   ├── X_test.csv                   # Preprocessed test features
│   ├── y_train.csv / y_test.csv     # Labels
│   ├── model1_logistic.pkl          # Saved Logistic Regression
│   ├── model2_rf.pkl                # Saved Random Forest
│   └── model3_xgb.pkl               # Saved XGBoost (final model)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, feature engineering
│   ├── 03_models.ipynb             # Model construction (all 3 models)
│   ├── 04_evaluation.ipynb         # Metrics, ROI, calibration, thresholds
│   ├── 05_shap.ipynb               # SHAP explainability (global + individual)
│   ├── 06_sensitivity.ipynb        # Robustness tests (SMOTE, feature removal, seeds)
│   ├── 07_stratified.ipynb         # Stratified analysis by contract + tenure
│   └── 08_error_analysis.ipynb     # False negatives, personas, recommendation engine
│
├── outputs/
│   ├── 01_churn_distribution.png
│   ├── 02_tenure_cohort_churn.png   # ← Best EDA chart
│   ├── 03_churn_by_category.png
│   ├── 04_numerical_distributions.png
│   ├── 05_feature_correlations.png
│   ├── 06_rf_feature_importance.png
│   ├── 07_threshold_optimisation.png
│   ├── 08_roc_curves.png
│   ├── 09_confusion_matrix.png
│   ├── 10_calibration_curves.png
│   ├── 11_shap_bar.png
│   ├── 12_shap_beeswarm.png         # ← WOW chart
│   ├── 13_shap_waterfall.png
│   ├── 14_shap_interaction.png      # ← Key insight chart
│   ├── 15_pdp_plots.png
│   └── 16_learning_curve.png
│
├── churn_analysis.py               # Complete analysis in a single runnable script
└── README.md
```

---

## 📊 Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 features |
| Target | `Churn` (Yes = 1 / No = 0) |
| Class split | 73% No Churn / 27% Churn |
| Time period | US telecom company (cross-sectional) |

**Key variables:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `SeniorCitizen`, `Partner`, `Dependents` + 6 add-on service flags.

---

## ⚙️ Feature Engineering

Three new features were engineered from the raw data, each with a domain-driven justification:

| Feature | Formula | Justification |
|---|---|---|
| `charges_per_month` | `TotalCharges / (tenure + 1)` | Captures value density — how much a customer pays relative to how long they've been with the company |
| `high_value_new` | `1 if tenure ≤ 12 AND MonthlyCharges > $70` | Flags the highest-risk segment: new customers already paying premium prices with no loyalty established |
| `num_services` | Sum of 6 add-on service columns | Measures engagement and switching cost — more services = more invested = less likely to churn |

---

## 🤖 Models

### Model 1 — Logistic Regression (Baseline)
Used `statsmodels.Logit` to obtain full coefficient tables with p-values and odds ratios. Provides an interpretable baseline and shows direction of each feature's effect on churn probability.

### Model 2 — Random Forest
100 decision trees with Gini impurity. Handles non-linearity and feature interactions automatically. Built-in feature importance as a cross-check against SHAP values.

### Model 3 — XGBoost with GridSearchCV *(Final Model)*
Gradient-boosted trees tuned across `n_estimators`, `max_depth`, `learning_rate`, and `subsample` using 5-fold cross-validation. Selected for best AUC, lowest variance across splits, and compatibility with SHAP explainability.

**Why threshold = 0.35 instead of 0.50:**
At threshold 0.35, recall increases from 48% to 58% with only a modest precision drop. In a churn context, the cost of missing a churner (lost lifetime revenue) far exceeds the cost of a false alarm (small retention offer). The threshold is a business decision, not a statistical one.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to the final XGBoost model to explain both global and individual predictions.

**Global Feature Importance (top 5):**
1. `tenure` — long-tenure customers are strongly protected from churning
2. `Contract_Month-to-month` — month-to-month contracts are the single strongest risk factor
3. `MonthlyCharges` — higher charges increase churn risk, especially for new customers
4. `num_services` — more add-on services reduce churn risk (switching cost effect)
5. `InternetService_Fiber optic` — fiber customers churn more than DSL or no-internet customers

**Key Interaction Finding:**

> The SHAP dependence plot reveals that **price sensitivity is a new-customer phenomenon**. For customers with tenure < 12 months, high MonthlyCharges significantly amplifies churn risk. For customers with tenure > 36 months, the same high charges have almost no additional effect — loyal customers are price-insensitive. This means **retention discounts should target high-charges new customers, not loyal ones.**

---

## 📐 Sensitivity Analysis

Three robustness tests confirmed the model is stable:

| Test | Finding |
|---|---|
| **SMOTE** (class rebalancing) | Recall increases from 58% to 67%; AUC drops slightly from 0.861 to 0.843 — use SMOTE when recall is the priority |
| **Feature removal** | Removing `tenure` drops AUC by 0.03 (critical); removing `num_services` drops by <0.01 (stable) |
| **5 random seeds** | AUC range: 0.857–0.864 (range = 0.007) — model is not sensitive to which 20% is held out |
| **Learning curve** | Validation AUC plateaus after ~4,000 training samples — more data will not substantially improve the model; better features would |

---

## 📦 Stratified Analysis

Model 3 was re-fitted on customer sub-groups to test whether churn drivers differ by segment.

### By Contract Type

| Contract | N | Base Churn Rate | AUC | Top SHAP Feature |
|---|---|---|---|---|
| Month-to-month | ~3,875 | 43% | — | MonthlyCharges |
| One year | ~1,473 | 11% | — | tenure |
| Two year | ~1,695 | 3% | — | MonthlyCharges |

### By Tenure Group

| Tenure Group | N | Base Churn Rate | Top SHAP Feature |
|---|---|---|---|
| New (0–12 months) | ~1,600 | ~47% | MonthlyCharges |
| Mid (13–36 months) | ~1,500 | ~28% | Contract_Month-to-month |
| Loyal (37–72 months) | ~3,900 | ~17% | Contract_Month-to-month |

**Key finding:** New customers churn because of **price** — they are evaluating value. Loyal customers churn because of **contract expiry** — they leave when nothing locks them in. These require completely different retention strategies.

---

## 💰 Business ROI

Model performance was translated into estimated business value using the following assumptions:
- Average monthly revenue per customer: ~$65
- Retention campaign cost per targeted customer: $10
- Retention success rate: 30% (industry benchmark)

| Model | Revenue Saved | Campaign Cost | Net Value |
|---|---|---|---|
| Logistic Regression | $X | $X | $X |
| Random Forest | $X | $X | $X |
| **XGBoost (tuned)** | **$X** | **$X** | **$X** |

> *Fill in values after running `churn_analysis.py` on your data.*

---

## 👤 Customer Personas

Error analysis identified three distinct churner types:

**Persona 1 — "The Frustrated New Customer" (0–12 months)**
Paying ~$79/month, fiber optic internet, month-to-month contract, no tech support. Churns within the first year, likely because the service quality does not justify the high price. *Recommended action: offer free Tech Support for 3 months.*

**Persona 2 — "The Price-Sensitive Mid-Tenure Customer" (13–36 months)**
Paying ~$72/month, just came off a one-year contract and did not renew. Has been loyal but switches when a better deal appears. *Recommended action: 15% discount to upgrade to a new one-year contract.*

**Persona 3 — "The Silent Long-Tenure Churner" (37–72 months)**
Paying ~$88/month, two-year contract just ended, minimal add-on services. Leaves quietly with no complaint history — the hardest to predict and the most expensive to lose. *Recommended action: assign a dedicated account manager + loyalty reward before contract renewal date.*

---

## 🚨 Model Failure Modes

**Failure Mode 1 — Long-tenure contract expiry (false negatives):**
The model consistently misses loyal customers whose two-year contracts have just expired. It has learned from historical data that high-tenure customers are safe, and cannot detect the contract-expiry signal without renewal date as a feature.

**Failure Mode 2 — New high-value customers (false positives):**
The model flags many new customers with high monthly charges as churners — but a significant portion stays. These customers look risky statistically but are in a genuine trial phase.

---



## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| pandas | 2.x | Data manipulation |
| numpy | 1.x | Numerical operations |
| scikit-learn | 1.x | ML models, preprocessing, metrics |
| xgboost | 2.x | Gradient boosted trees (final model) |
| shap | 0.44+ | Model explainability |
| imbalanced-learn | 0.11+ | SMOTE class rebalancing |
| statsmodels | 0.14+ | Logistic regression with p-values |
| matplotlib / seaborn | latest | Visualisation |

---

## ⚠️ Limitations

- **No complaint history data** — the single most predictive feature for churn (customer service interactions) is not available in this dataset
- **Static snapshot** — no time series; sequential customer behaviour is not captured
- **US telecom market** — may not generalise to Indian telecom dynamics (Jio, Airtel pricing structures differ significantly)
- **Missing contract renewal dates** — the model cannot detect the contract-expiry signal, causing the most expensive false negatives
- **SMOTE generates synthetic samples** — validation metrics on SMOTE-balanced models may be optimistic

---

## 👥 Team

| Member | Role | Sections |
|---|---|---|
| Chirag Sethi | Project Lead | GitHub | Model Builder | Preprocessing, Models, Evaluation, Business ROI |
| Mitrajit Kumar | EDA Analyst | EDA (Section 2), Preprocessing justification |
| Hardik Sharma | SHAP + Sensitivity | SHAP Explainability, Sensitivity Analysis, Stratified Analysis |
| Shree Iyengar | Error + Presentation | Error Analysis, Customer Personas, Slide Deck |

---

## 📄 License

This project is submitted as academic coursework for the **Introduction to Data Science** course at SP Jain School of Global Management. Dataset is publicly available on Kaggle under its original license.

---

*SP Jain School of Global Management | Introduction to Data Science | April 2026*
