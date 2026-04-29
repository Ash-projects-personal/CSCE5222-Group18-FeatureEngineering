# CSCE 5222 — Feature Engineering for Time-Series
## Group 18 | Ashish Rathnakar Shetty & Kushal Sai Venigalla
### University of North Texas

---

## About This Notebook

This notebook is the live demo for our Project 2 submission. It runs the complete feature engineering pipeline we built, end-to-end, across three different time-series tasks:

| Task | Dataset | Goal |
|---|---|---|
| **Classification** | UCR Archive (5 datasets) | Classify time-series patterns |
| **Forecasting** | ETTh1 (electricity transformer) | Predict oil temperature one step ahead |
| **Anomaly Detection** | WESAD-like PPG signal | Find bad sensor segments without labels |

We test four feature engineering strategies across all three tasks:

1. **Raw features** — baseline, no engineering
2. **Catch22** — 22 compact statistical features we wrote from scratch in pure NumPy
3. **Self-supervised MLP embeddings** — a small neural net learns features on its own
4. **SHAP-guided pruning** — use explainability to cut redundant features

---

## How to Run

1. Click **Runtime → Run all** at the top of Colab
2. Or run each cell one by one to explain what each part does

**First run takes about 2–3 minutes** because it downloads:
- Five UCR classification datasets (~5 MB)
- ETTh1 electricity transformer dataset (~2 MB)
- The `shap` Python package (the only pip install needed)

Everything else (NumPy, pandas, scikit-learn, matplotlib) is already available in Google Colab.

---

## What Each Section Does

| Section | What Happens |
|---|---|
| **Cell 1** | Installs `shap` and imports all libraries |
| **Cell 2** | Defines our custom Catch22 implementation (22 features in pure NumPy) |
| **Cell 3** | Defines the SHAP-guided feature pruning helper |
| **Cell 4** | Downloads all three datasets and plots the PPG signal |
| **Cells 5–8** | **Task 1: Classification** — runs all 5 UCR datasets, shows accuracy chart, SHAP importance, and PCA projection |
| **Cells 9–11** | **Task 2: Forecasting** — builds lag/rolling features, trains all 4 models, shows MSE comparison |
| **Cells 12–13** | **Task 3: Anomaly Detection** — Isolation Forest on raw vs Catch22 vs SHAP-selected |
| **Cell 14** | Prints the final summary table of all results |

---

## Key Results (What You Should See)

**Classification (UCR):**
- SHAP-pruned Catch22 reaches 95.0% on SyntheticControl (+9.3% over raw)
- Catch22 improves TwoLeadECG by 8.8% even with only 23 training samples
- Raw features win on ItalyPowerDemand (local timing event Catch22 misses)

**Forecasting (ETTh1):**
- Lag + Rolling statistics reduce MSE by 62% over the raw lag-1 baseline
- SHAP pruning removes 12 of 28 features and achieves the best MSE of 0.4734

**Anomaly Detection (PPG):**
- Raw windows: F1 = 0.871
- Catch22 drops to 0.613 because of amplitude normalization
- SHAP selection recovers to 0.645 by keeping `diff_variance`

---

## Repository and Report

- **GitHub:** https://github.com/Ash-projects-personal/CSCE5222-Group18-FeatureEngineering
- **Report:** 10-page IEEE-format PDF available in the same repository
- **Course:** CSCE 5222 Feature Engineering, Spring 2026

---

*Run all cells below. Each one is self-contained and explained inline.*
