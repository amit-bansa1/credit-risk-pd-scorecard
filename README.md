# Credit Risk PD Scorecard Model

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Methodology](https://img.shields.io/badge/Methodology-Basel%20II%2FIII%20Aligned-orange)

## Overview
An end-to-end Probability of Default (PD) scorecard model built on 
150,000 real loan applicants. Follows industry-standard credit risk 
methodology — the same approach used in Basel II/III retail models 
at major banks globally.

## Validation Results

| Metric | Result | Benchmark | Status |
|---|---|---|---|
| KS Statistic | 49.24% | 40–60% = Good | ✅ Good |
| Gini Coefficient | 63.19% | 50–70% = Good | ✅ Good |
| AUC-ROC | 0.8159 | 0.75–0.85 = Good | ✅ Good |
| PSI | 0.0001 | <0.10 = Stable | ✅ Stable |

## Methodology
```
Raw Data → EDA & Cleaning → WoE Binning → IV Feature Selection 
→ Logistic Regression → Scorecard Scaling → Validation
```

1. **EDA & Data Cleaning** — outlier treatment (Winsorization), 
   sentinel value handling, missing value imputation
2. **WoE Binning** — transforms each variable into a risk-encoded 
   feature. Fitted on training data only to prevent data leakage.
3. **IV Feature Selection** — 7 of 10 variables selected using 
   Information Value threshold (IV ≥ 0.02)
4. **Logistic Regression** — trained on WoE features with L2 
   regularisation and balanced class weights
5. **Scorecard Scaling** — PDO method converts log-odds to 
   points-based credit score (Base score 600, PDO 20)
6. **Validation** — KS, Gini, AUC-ROC, PSI on held-out test set

## Key Findings
- **Dominant predictor:** RevolvingUtilizationOfUnsecuredLines (IV 1.11)
- **Strongest delinquency signal:** NumberOfTime30-59DaysPastDue (IV 0.43)
- **Score gap:** Non-defaulters average 508 vs Defaulters average 464 
  (44-point separation)
- **No overfitting:** Test AUC (0.816) ≥ Train AUC (0.814)

## Tech Stack
Python | pandas | numpy | scikit-learn | optbinning | 
matplotlib | seaborn

## Project Structure
```
credit-risk-pd-scorecard/
├── data/
│   ├── raw.csv                  # Original dataset
│   ├── cleaned.csv              # After outlier treatment & imputation
│   ├── train_woe.csv            # WoE transformed training set
│   ├── test_woe.csv             # WoE transformed test set
│   ├── train_scores.csv         # Final credit scores — train
│   ├── test_scores.csv          # Final credit scores — test
│   ├── iv_summary.csv           # IV scores for all variables
│   ├── scorecard_table.csv      # Points per bin per variable
│   └── validation_summary.csv  # Final validation metrics
├── notebooks/
│   ├── 01_eda.ipynb             # EDA & data cleaning
│   ├── 02_feature_engineering.ipynb  # WoE & IV
│   ├── 03_model.ipynb           # Model training & scorecard
│   └── 04_validation.ipynb     # KS, Gini, AUC, PSI
├── reports/
│   ├── model_documentation.md  # Full MRM-style model document
│   └── *.png                   # Validation charts
└── src/                        # Helper functions
```

## Dataset
Give Me Some Credit — Kaggle (150,000 US borrowers, 2007–2009)

## Full Documentation
See [model_documentation.md](reports/model_documentation.md) 
for complete methodology, results, and limitations.

## Author
**Amit Bansal** — Manager, Decision Science at HSBC  
[GitHub](https://github.com/amit-bansa1) | 
[LinkedIn](https://www.linkedin.com/in/theamitbansal)