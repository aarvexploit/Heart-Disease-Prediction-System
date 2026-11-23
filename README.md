# Heart Disease Prediction System

An end-to-end machine learning pipeline for predicting heart disease presence from clinical patient records using the multi-center **UCI Heart Disease dataset** (`Cleveland`, `Hungary`, `VA Long Beach`, and `Switzerland`).

## Overview
Clinical datasets often contain disguised missing values, cross-hospital distribution shifts, and asymmetric error costs where false negatives are far riskier than false positives. This project builds a reproducible, leak-free ML workflow focused on:
- **Clinical Data Hygiene:** Distinguishing valid zero measurements (`oldpeak`, `ca`, `fbs == False`, `exang == False`, `num == 0`) from clinically impossible zeros (`chol == 0`, `trestbps == 0`) treated as missing values (`NaN`).
- **Leak-Free Preprocessing:** Fitting imputers, scalers, and encoders strictly inside `StratifiedKFold` cross-validation pipelines.
- **Screening-Focused Evaluation:** Optimizing for **ROC-AUC**, **Recall**, and **F2-score** with custom decision threshold tuning rather than relying solely on raw accuracy.
- **Model Explainability:** Using **SHAP** to surface which clinical features drive individual risk predictions.

## Dataset Summary
- **Source:** UCI Machine Learning Repository (`heart_disease_uci.csv`).
- **Records:** 920 patient observations across 4 clinical cohorts (`Cleveland`, `Hungary`, `VA Long Beach`, `Switzerland`).
- **Features:** Age (`age`), sex (`sex`), chest pain type (`cp`), resting blood pressure (`trestbps`), serum cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG (`restecg`), max heart rate (`thalch`), exercise-induced angina (`exang`), ST depression (`oldpeak`), ST slope (`slope`), fluoroscopy vessel count (`ca`), and thalassemia (`thal`).
- **Target (`num`):** Original angiographic disease status (`0` = absence, `1–4` = severity levels), modeled as binary classification (`0` vs. `> 0`) or multi-class severity.

## Project Structure
```text
heart-disease-prediction/
├── data/
│   └── raw/                 # Original UCI dataset (read-only)
├── notebooks/               # Exploratory data analysis & experiments
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loading & clinical cleaning rules
│   ├── features.py          # Preprocessing & feature engineering pipelines
│   ├── train.py             # Stratified CV & model training
│   └── evaluate.py          # Metrics, threshold analysis & SHAP plots
├── artifacts/               # Saved model pipelines
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md