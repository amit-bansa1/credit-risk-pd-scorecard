# Credit Risk PD Scorecard Model

## Overview
An end-to-end Probability of Default (PD) scorecard model built on the 
"Give Me Some Credit" dataset (150,000 loan applicants). Follows 
industry-standard credit risk methodology used in Basel II/III retail models.

## Methodology
- Exploratory Data Analysis & data cleaning
- Weight of Evidence (WoE) binning & Information Value (IV) feature selection
- Logistic Regression with points-based scorecard scaling
- Model validation: KS statistic, Gini, AUC-ROC, PSI

## Tech Stack
Python | pandas | scikit-learn | scorecardpy | optbinning

## Project Structure
- `data/` — raw and cleaned datasets
- `notebooks/` — step-by-step analysis notebooks
- `src/` — reusable helper functions
- `reports/` — model documentation and validation charts

## Dataset
Give Me Some Credit — Kaggle (150,000 loan applicants, 11 variables)

## Status
🔄 In Progress — Week 1 complete (environment setup, data verified)
