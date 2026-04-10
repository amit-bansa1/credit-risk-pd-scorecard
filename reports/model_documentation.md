# Credit Risk PD Scorecard Model
## Model Documentation

**Author:** Amit Bansal  
**Date:** April 2026  
**Version:** 1.0  
**Status:** Development Complete — Validation Passed  

---

## 1. Executive Summary

This document describes the development and validation of a Probability of Default (PD) scorecard model for retail credit risk.
The model was built following industry-standard Basel II/III aligned methodology using Weight of Evidence (WoE) feature engineering and Logistic Regression.

The model achieves a KS statistic of 49.2% and Gini coefficient of 63.2% on the held-out test set — within the typical performance range of retail credit scorecards deployed at major banks.

| Metric | Test Result | Benchmark | Status |
|---|---|---|---|
| KS Statistic | 49.24% | 40–60% = Good | ✅ Good |
| Gini Coefficient | 63.19% | 50–70% = Good | ✅ Good |
| AUC-ROC | 0.8159 | 0.75–0.85 = Good | ✅ Good |
| PSI | 0.0001 | <0.10 = Stable | ✅ Stable |

---

## 2. Business Objective

### 2.1 Problem Statement
Retail lenders need to assess the creditworthiness of loan applicants before extending credit. The core question is: given a borrower's 
financial profile, what is the probability they will default within 
the next 24 months?

### 2.2 Model Use Case
This model produces a points-based credit score (range 378–560) for each borrower. Lower scores indicate higher default risk. The score 
can be used to:
- Accept or reject loan applications
- Set interest rates proportional to risk
- Determine credit limits
- Monitor portfolio risk over time

### 2.3 Target Variable
`SeriousDlqin2yrs` — binary indicator of whether the borrower experienced 90+ days delinquency within 2 years of observation.
- 0 = No default (93.3% of population)
- 1 = Default (6.7% of population)

---

## 3. Data

### 3.1 Dataset
- **Source:** Give Me Some Credit — Kaggle Competition Dataset
- **Population:** US retail borrowers, 2007–2009 (subprime crisis period)
- **Size:** 150,000 borrowers, 11 variables
- **Final modelling sample:** 149,999 rows after cleaning

### 3.2 Variable Dictionary

| Variable | Description | Type |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | Credit card balance ÷ credit limit | Continuous |
| age | Age of borrower in years | Continuous |
| NumberOfTime30-59DaysPastDueNotWorse | Times 30–59 days late on payment | Count |
| DebtRatio | Monthly debt payments ÷ monthly income | Continuous |
| MonthlyIncome | Self-reported monthly income (USD) | Continuous |
| NumberOfOpenCreditLinesAndLoans | Total open credit lines and loans | Count |
| NumberOfTimes90DaysLate | Times 90+ days late on payment | Count |
| NumberRealEstateLoansOrLines | Mortgage and real estate loans | Count |
| NumberOfTime60-89DaysPastDueNotWorse | Times 60–89 days late on payment | Count |
| NumberOfDependents | Number of family dependents | Count |

### 3.3 Data Quality Issues & Treatment

| Issue | Variables Affected | Treatment |
|---|---|---|
| Extreme outliers | RevolvingUtilization, DebtRatio, MonthlyIncome | Winsorization at 99th percentile |
| Hard boundary violation | RevolvingUtilization (max 50,708) | Hard cap at 1.0 (ratio definition) |
| Sentinel values (96/97/98) | All three delinquency count variables | Replaced with NaN, imputed with 0 |
| Missing values (~20%) | MonthlyIncome | Median imputation ($5,400) |
| Missing values (~2.6%) | NumberOfDependents | Median imputation (0) |
| Invalid values | age = 0 (1 row) | Row dropped |

**Note on median imputation:** Median chosen over mean for MonthlyIncome and NumberOfDependents because both variables are right-skewed. The mean would be pulled upward by remaining high values, producing an unrepresentative imputation value.

**Note on sentinel values:** Values of 96, 97, 98 in delinquency count fields are system-generated codes in banking data meaning "data unavailable" — not actual delinquency counts. This is a common real-world data quality issue in bank model development.

---

## 4. Methodology

### 4.1 Development Approach
The model follows the standard retail credit scorecard development process used by banks under Basel II/III:

1. Exploratory Data Analysis → understand distributions and data quality
2. Data cleaning → treat outliers, missing values, and invalid data
3. WoE binning → transform variables into risk-encoded features
4. IV-based feature selection → objectively select predictive variables
5. Logistic Regression → train model on WoE features
6. Scorecard scaling → convert model output to points-based score
7. Validation → assess discriminatory power and stability

### 4.2 Train/Test Split
- 70% training (104,999 rows), 30% test (45,000 rows)
- Stratified split to preserve 6.7% default rate in both sets
- Split performed **before** WoE binning to prevent data leakage

### 4.3 Weight of Evidence (WoE) Transformation

WoE encodes each variable bin as:

```
WoE = ln(Distribution of Events / Distribution of Non-Events)
```

Positive WoE indicates a riskier-than-average bin.
Negative WoE indicates a safer-than-average bin.

WoE transformation was fitted on training data only and applied to test data — preventing information leakage from test set.

### 4.4 Information Value (IV) — Feature Selection

IV measures each variable's predictive power:

```
IV = Σ (Distribution of Events − Distribution of Non-Events) × WoE
```

| IV Range | Interpretation |
|---|---|
| < 0.02 | Useless — drop |
| 0.02–0.10 | Weak predictor |
| 0.10–0.30 | Medium predictor |
| 0.30–0.50 | Strong predictor |
| > 0.50 | Suspicious — check for data leakage |

### 4.5 Feature Selection Results

| Variable | IV | Decision |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 1.1101 | Kept — investigated, no leakage found |
| NumberOfTime30-59DaysPastDueNotWorse | 0.4270 | Kept — Strong |
| age | 0.2418 | Kept — Medium |
| DebtRatio | 0.0764 | Kept — Weak |
| MonthlyIncome | 0.0676 | Kept — Weak |
| NumberOfOpenCreditLinesAndLoans | 0.0667 | Kept — Weak |
| NumberOfDependents | 0.0236 | Kept — Borderline |
| NumberRealEstateLoansOrLines | 0.0128 | **Dropped** — below threshold |
| NumberOfTimes90DaysLate | 0.0000 | **Dropped** — zero-inflation |
| NumberOfTime60-89DaysPastDueNotWorse | 0.0000 | **Dropped** — zero-inflation |

### 4.6 Logistic Regression Model

```
log(P(default) / P(non-default)) = β₀ + β₁×WoE₁ + ... + β₇×WoE₇
```

**Key parameters:**
- `class_weight='balanced'` — compensates for 93/7 class imbalance by assigning ~13.9x more weight to defaulter observations
- `C=0.5` — L2 regularisation to prevent overfitting, particularly for weak variables like NumberOfDependents
- `random_state=42` — reproducibility

**Model coefficients:**

| Variable | Coefficient |
|---|---|
| RevolvingUtilizationOfUnsecuredLines | 0.8382 |
| NumberOfTime30-59DaysPastDueNotWorse | 0.8264 |
| DebtRatio | 0.7179 |
| age | 0.4886 |
| MonthlyIncome | 0.3729 |
| NumberOfOpenCreditLinesAndLoans | 0.3332 |
| NumberOfDependents | 0.2067 |
| Intercept | −0.0068 |

All coefficients are positive — expected, since higher WoE values indicate higher risk and should increase the log-odds of default.

### 4.7 Scorecard Scaling

The scorecard converts log-odds into a human-readable points score using the industry-standard PDO (Points to Double the Odds) method:

```
Score = Offset + Factor × (−log odds)
Factor = PDO / ln(2) = 28.85
Offset = Base score − Factor × ln(Base odds) = 487.12
```

**Scaling parameters:**
- Base score: 600 (industry convention)
- Base odds: 50:1 (non-default:default ratio at base score)
- PDO: 20 (score increase needed to halve default risk — standard)

Each variable contributes a partial score. Final score = sum of all partial scores. Higher score = lower default risk.

---

## 5. Validation Results

### 5.1 Discriminatory Power

| Metric | Train | Test | Benchmark | Status |
|---|---|---|---|---|
| KS Statistic | 49.56% | 49.24% | 40–60% = Good | ✅ Good |
| Gini Coefficient | 62.70% | 63.19% | 50–70% = Good | ✅ Good |
| AUC-ROC | 0.8135 | 0.8159 | 0.75–0.85 = Good | ✅ Good |

**Note:** Test metrics marginally exceed train metrics — confirms L2 regularisation prevented overfitting. Model generalises well.

### 5.2 Score Distribution

| Metric | Value |
|---|---|
| Score range | 378–560 |
| Median score | 514 |
| Non-defaulters average | 508 |
| Defaulters average | 464 |
| Score gap | 43.9 points |

### 5.3 Population Stability

| Metric | Value | Benchmark | Status |
|---|---|---|---|
| PSI | 0.0001 | <0.10 = Stable | ✅ Exceptionally Stable |

PSI of 0.0001 indicates near-perfect score distribution consistency between training and test sets. Model would behave consistently on new data.

---

## 6. Model Limitations

The following limitations should be considered before any production deployment:

**1. Dataset vintage**
Data is from 2007–2009 US subprime lending market during the financial crisis — an atypical economic period. Model would require recalibration before deployment on current portfolio data.

**2. Geographic specificity**
Population is US borrowers. Variable distributions and risk relationships may differ materially in other markets (India, UK, etc.).

**3. Variable set**
Only 7 variables used. Production scorecards typically use 15–25 variables including bureau-sourced variables (credit history length, number of inquiries, payment history across all lenders) not available in this dataset.

**4. Delinquency variable signal loss**
NumberOfTimes90DaysLate and NumberOfTime60-89DaysPastDueNotWorse showed zero IV due to zero-inflation caused by sentinel value treatment. In production, these variables with proper bureau data would likely be strong predictors.

**5. MonthlyIncome imputation artefact**
29,731 missing income values (20%) were imputed with median ($5,400), creating an artificial concentration at that value. A production model would source income from payroll data, ITR filings, or bank statement analysis — reducing missing data significantly.

**6. Score range**
Score range (378–560) is narrower than commercial bureau scores (300–900) due to limited variable set. Direct comparison with CIBIL or FICO scores is not appropriate.

**7. Binning methodology**
WoE binning used automated optimal binning (OptimalBinning library) rather than the manual fine-to-coarse progression used in production bank models. Production scorecards involve manual review of fine bins (20-50 bins per variable), monotonicity enforcement, minimum bin size validation (≥5% population per bin), and documented business rationale for each bin boundary decision.

**8. Hyperparameter decisions**
This is a logistic regression model so hyperparameter tuning is limited to regularisation strength (C=0.5). In a production environment, C would 
be selected via cross-validated grid search rather than set to a reasonable default.

---

## 7. Real-World Data Context

In a production banking environment, the variables in this model would be derived from the following raw data sources:

| Variable | Raw Data Source |
|---|---|
| RevolvingUtilization | Credit bureau — outstanding balance ÷ limit across all revolving accounts |
| NumberOfTime30-59DaysPastDue | Payment history table — delinquency events in rolling 24-month window |
| age | Customer master table — date of birth |
| DebtRatio | Payroll data + loan system — total EMI obligations ÷ verified income |
| MonthlyIncome | ITR filings, payroll data, or bank statement analysis |
| NumberOfOpenCreditLines | Credit bureau — count of active tradelines |
| NumberOfDependents | KYC/onboarding form — self-declared |

This upstream data engineering — from raw tables to model-ready features — typically involves SQL/SAS/BigQuery extraction, data lineage mapping, and variable logic validation across systems.

---

## 8. Conclusion

This model demonstrates a complete, Basel II/III aligned PD scorecard development process from raw data to validated scorecard. 

The validation results — KS 49.2%, Gini 63.2%, AUC 0.816, PSI 0.0001 — confirm the model is performing at Good level across all metrics, within the range of retail credit scorecards deployed at major banks.

The scorecard methodology (WoE/IV + Logistic Regression + PDO scaling) is directly transferable to production credit risk environments including retail banking, NBFC lending, and fintech credit products.

---

*Document prepared by Amit Bansal | github.com/amit-bansa1/credit-risk-pd-scorecard*