# Telco Customer Churn Prediction

**End-to-end ML pipeline** — from raw data to business ROI — built across 8 structured notebooks.

`Python` `XGBoost` `SHAP` `scikit-learn` `statsmodels` `imbalanced-learn`

> **The core insight of this project**: Price sensitivity is exclusively a *new-customer* phenomenon. Retention discounts should target high-charges customers in their first 12 months — not loyal customers who were never going to leave. Discovered via SHAP interaction analysis.

---

## Results at a Glance

| Metric | Value |
|---|---|
| **Final Model** | XGBoost (tuned with GridSearchCV) |
| **Test AUC-ROC** | 0.845 |
| **Test Recall** | 72.2% of churners caught |
| **Test F2 Score** | 0.682 (β=2, recall-weighted) |
| **Top 10% Lift** | Contacting top 10% of scored customers captures **27.8% of all churners** (2.8× random) |
| **Campaign ROI** | Estimated **$75,889 net profit** on test set (1,409 customers) |
| **Model Stability** | AUC range 0.843–0.863 across 5 random seeds — results are not a lucky split |

---

## Why This Project Is Different

Most churn notebooks stop at accuracy. This one starts where that ends.

**1. Framed as a decision problem, not a classification problem**
The cost of missing a churner (~$780/year lost revenue) is 78× more expensive than a false alarm ($10 campaign cost). This ratio drives every design choice: threshold selection, metric choice (F2 not F1), and the final ROI calculation.

**2. Threshold chosen by Expected Value maximisation, not convention**
The standard 0.50 threshold was rejected in favour of 0.35 — selected by computing the mathematically optimal threshold that maximises profit given the 78:1 cost asymmetry.

**3. SHAP used to discover a non-obvious business insight**
The tenure × MonthlyCharges interaction plot revealed that high charges only amplify churn risk for customers with tenure < 12 months. Loyal customers are price-insensitive. This finding would be invisible in a standard feature importance bar chart.

**4. Every result translated into dollars**
Cross-validation scores, confusion matrices, and cumulative gains charts are all connected back to estimated campaign profit — making the work relevant to a business audience, not just a technical one.

---

## Model Performance

### 5-Fold Cross-Validation (Mean ± Std)

| Model | AUC-ROC | F1 | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.843 ± 0.008 | 0.601 ± 0.012 | 0.554 ± 0.015 | 0.651 ± 0.014 |
| Random Forest | 0.827 ± 0.011 | 0.578 ± 0.014 | 0.491 ± 0.018 | 0.634 ± 0.016 |
| **XGBoost (tuned) ★** | **0.849 ± 0.007** | **0.631 ± 0.010** | **0.706 ± 0.013** | **0.672 ± 0.011** |

> ★ Final model. GridSearchCV over 270 fits (54 combinations × 5 folds). Best params: `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `subsample=0.8`.

### Business ROI (Test Set, Threshold = 0.35)

| Model | Churners Caught | Revenue Saved | Campaign Waste | Net Profit |
|---|---|---|---|---|
| Logistic Regression | ~200 | ~$46,800 | ~$2,600 | ~$44,200 |
| Random Forest | ~210 | ~$49,140 | ~$2,400 | ~$46,740 |
| **XGBoost (tuned)** | **270** | **~$63,180** | **~$2,140** | **~$61,040** |

> Assumptions: avg annual revenue = $780/customer, retention success rate = 30%, campaign cost = $10/contact. All 45 tested cost scenarios (varying success rate, cost, revenue) remained profitable — the business case is robust to assumption errors.

### Cumulative Gains

| Contacted | Churners Captured | Lift vs Random |
|---|---|---|
| Top 10% | 27.8% | 2.8× |
| Top 20% | ~48% | ~2.4× |
| Top 30% | ~65% | ~2.2× |
| **Profit-optimal** | **Top 77%** | **Max profit: $75,889** |

---

## The Core Insight — SHAP Interaction Plot

The SHAP dependence plot for `tenure` (coloured by `MonthlyCharges`) reveals a clean interaction:

- **Customers with tenure < 12 months**: High monthly charges dramatically increase churn SHAP value — these customers are actively evaluating "Is this worth it?"
- **Customers with tenure > 36 months**: High monthly charges have near-zero additional churn effect — loyal customers have already answered that question with years of staying

**Practical implication**: A discount campaign targeting all high-charges customers wastes budget on loyal customers. Target only high-charges customers in their first year.

---

## Project Structure

```
telco-churn-prediction/
│
├── data/
│   ├── telco_churn.csv              # Raw dataset (Kaggle)
│   ├── X_train.csv / X_test.csv     # Preprocessed features
│   ├── X_train_scaled.csv           # StandardScaled (for Logistic Regression)
│   ├── y_train.csv / y_test.csv     # Labels
│   ├── model1_logistic.pkl          # Saved Logistic Regression
│   ├── model2_rf.pkl                # Saved Random Forest
│   ├── model3_xgb.pkl               # Saved XGBoost (final model)
│   └── xgb_best_params.json         # GridSearchCV best parameters
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # 5 EDA charts, class imbalance, feature correlations
│   ├── 02_preprocessing.ipynb      # Encoding, feature engineering, stratified split, scaling
│   ├── 03_models.ipynb             # VIF check, LR (p-values + odds ratios), RF, XGBoost + GridSearchCV
│   ├── 04_evaluation.ipynb         # CV, F2, EV optimisation, ROC, confusion matrix, gains chart, ROI
│   ├── 05_shap_explainability.ipynb # TreeExplainer, LinearExplainer, beeswarm, waterfall, interaction
│   ├── 06_sensitivity.ipynb        # SMOTE, feature removal, 5-seed split, learning curve, cost heatmap
│   ├── 07_stratified.ipynb         # Segment-specific models by contract type and tenure group
│   └── 08_error_analysis.ipynb     # FN/FP profiling, personas, recommendation engine, failure modes
│
├── outputs/                         # 21 charts (auto-generated by running notebooks in order)
├── churn_analysis_last.py          # Full analysis as a single runnable script
└── README.md
```

**Run order:** `02 → 03 → 04 → 05 → 06 → 07 → 08` (notebook 01 is standalone)

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Features | 21 raw → 26 after engineering + encoding |
| Target | `Churn` (Yes / No) |
| Class split | 73.5% No Churn / 26.5% Churn |
| Train / Test | 5,634 / 1,409 (stratified 80/20 split) |

---

## Feature Engineering

| Feature | Formula | Business Rationale | Validation |
|---|---|---|---|
| `charges_per_month` | `TotalCharges / (tenure + 1)` | Spending intensity — normalises charges for tenure length | Strong correlation with churn |
| `high_value_new` | `1 if tenure ≤ 12 AND MonthlyCharges > $70` | Flags the highest-risk segment — 12.4% of customers, **68% churn rate** (2.5× population average) | Confirmed by SHAP |
| `num_services` | Sum of 6 add-on flags | Switching cost proxy — each service is friction to leave | Pearson r = −0.088, protective as expected |

---

## Key EDA Findings

| Finding | Implication |
|---|---|
| New customers (0–12m) churn at **47%** vs loyal customers (37–72m) at **17%** | First year is the critical retention window |
| Month-to-month contract: **43% churn** vs two-year: **3% churn** (14× gap) | Contract type is the single strongest retention lever |
| Fiber optic: **42% churn** vs DSL: **19%** | Premium price without perceived premium quality drives dissatisfaction |
| Electronic check payment: **45% churn** vs credit card: **15%** | Payment method is a proxy for engagement and financial stability |
| Tech Support absent: **42% churn** vs present: **15%** | Service quality gap is actionable — adding Tech Support reduces churn |

---

## Explainability (SHAP)

**Global Feature Importance — Top 5:**
1. `tenure` — protective; longer customers are substantially less likely to churn
2. `Contract_Month-to-month` — risk factor; no lock-in = no friction to leave
3. `MonthlyCharges` — risk factor for new customers only (key interaction finding)
4. `num_services` — protective; each additional service is a switching cost
5. `InternetService_Fiber optic` — risk factor; highest churn rate of any internet type

**Individual explanations** via SHAP waterfall plots show exactly which features pushed a specific customer toward or away from churn — production-ready for a retention agent's dashboard.

---

## Sensitivity Analysis

| Test | Result | Interpretation |
|---|---|---|
| **SMOTE** | AUC drops 0.845 → 0.829; Recall improves 72.2% → 84.2% | SMOTE trades precision for recall — use when recall is critical |
| **Feature removal** | Top 5 features each drop AUC by ≤ 0.002 | Model is not brittle — no single feature is a single point of failure |
| **5 random seeds** | AUC range: 0.843–0.863 (spread = 0.020) | Results are stable — not a lucky split |
| **Learning curve** | Validation AUC plateaus at ~4,000 rows | More data won't help; richer features would |
| **Cost sensitivity** | All 45 tested scenarios profitable (15%–40% retention success, $5–$20 cost) | Business case is robust to assumption errors |

---

## Stratified Analysis

### By Contract Type

| Contract | N | Churn Rate | AUC | Top Driver |
|---|---|---|---|---|
| Month-to-month | 3,875 | 27.0% | 0.843 | Contract type |
| One year | 1,473 | 26.4% | 0.817 | Tenure |
| Two year | 1,695 | 25.6% | 0.840 | Tenure |

### By Tenure Group

| Segment | N | AUC | Top Driver | Recommended Intervention |
|---|---|---|---|---|
| New (0–12m) | 2,201 | 0.814 | `InternetService_Fiber optic` | Free Tech Support trial — service quality problem |
| Mid (13–36m) | 1,865 | 0.821 | `tenure` | Contract upgrade offer — lock-in is the lever |
| Loyal (37–72m) | 2,963 | 0.843 | `tenure` | Loyalty reward + renewal outreach at months 22/46 |

**Conclusion**: A single global model (AUC 0.845) performs comparably to segment-specific models — one model is sufficient for deployment, but interventions must be segment-specific.

---

## Customer Personas

### Persona 1 — "The Frustrated New Customer"
- **Profile**: 3 months tenure, $74/month, Fiber optic, month-to-month, no Tech Support, electronic check
- **Churn rate in this segment**: ~47%
- **Why they leave**: Service quality doesn't match the premium price — they're still evaluating
- **Action**: Free 3-month Tech Support trial + proactive onboarding call within 60 days

### Persona 2 — "The Price-Sensitive Mid-Tenure Customer"
- **Profile**: 22 months tenure, $85/month, Fiber optic, month-to-month, electronic check
- **Churn rate in this segment**: ~28%
- **Why they leave**: Enough tenure to have some loyalty, but no contract lock-in — susceptible to competitor offers
- **Action**: 15% discount to upgrade to a 1-year contract

### Persona 3 — "The Silent Long-Tenure Churner"
- **Profile**: 52 months tenure, $97/month, month-to-month, electronic check
- **Churn rate in this segment**: ~17% — but highest annual revenue at risk
- **Why they leave**: Contract expired and no retention outreach — leaves quietly with no warning
- **Action**: Dedicated account manager + loyalty reward before estimated contract renewal date

---

## Failure Modes & Mitigations

**Failure Mode 1 — Missed contract expiry (False Negatives)**
The model assigns low churn risk to high-tenure customers by design. When their 2-year contract expires, they become vulnerable — but the model doesn't see it without a renewal date feature.
```python
# Proxy mitigation — flag customers approaching 24-month contract cycle
df['contract_age_proxy'] = df['tenure'] % 24
df['near_expiry'] = (df['contract_age_proxy'] >= 22).astype(int)
# Contact these customers regardless of model score
```

**Failure Mode 2 — New customer over-flagging (False Positives)**
New customers with high charges look statistically like churners but many are still in a genuine trial phase.
```python
# Apply stricter threshold for trial-phase customers
TRIAL_THRESHOLD = 0.55  # vs standard 0.35
df['trial_phase'] = (df['tenure'] < 3).astype(int)
```

---

## Limitations

1. **No complaint history** — customer service call logs are the strongest real-world churn predictor; their absence is the primary ceiling on performance
2. **No contract renewal dates** — directly causes Failure Mode 1; a single date field would meaningfully improve recall for loyal customers
3. **Static cross-section** — no temporal trajectories (charge increases, service downgrades) that would signal disengagement before churn
4. **Constant retention success rate** — 30% assumed flat; in practice it varies by offer type, agent, and segment
5. **US telecom context** — competitive dynamics (pricing, contract norms) differ significantly across markets

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation and numerical operations |
| `scikit-learn` | Preprocessing, model evaluation, GridSearchCV, learning curves |
| `xgboost` | Final gradient-boosted model |
| `statsmodels` | Logistic regression with p-values, odds ratios, VIF |
| `shap` | TreeExplainer (XGBoost) + LinearExplainer (LR), waterfall, beeswarm, interaction plots |
| `imbalanced-learn` | SMOTE class rebalancing (sensitivity test) |
| `scipy` | KDE density estimation for distribution plots |
| `matplotlib` / `seaborn` | All visualisations (21 output charts) |

---

## How to Run

```bash
# 1. Clone and install dependencies
git clone https://github.com/ChiragSethi-04/telco-churn-prediction.git
cd telco-churn-prediction
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap statsmodels imbalanced-learn jupyter

# 2. Add the dataset
# Download telco_churn.csv from Kaggle and place in data/

# 3. Run notebooks in order
jupyter notebook
# Run: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
# Note: 02 must run before 03 (saves CSVs); 03 must run before 04–08 (saves models)
```

---

## Team

| Member | Role |
|---|---|
| **Chirag Sethi** | Project Lead — Architecture, Models, Evaluation, Business ROI |
| Mitrajit Kumar | EDA Analyst — Exploratory Analysis, Preprocessing |
| Hardik Sharma | ML Engineer — SHAP Explainability, Sensitivity Analysis, Stratified Analysis |
| Shree Iyengar | Insights — Error Analysis, Customer Personas, Presentation |

---

*SP Jain School of Global Management | Introduction to Data Science | April 2026*
