# CSCE 5222 Feature Engineering – Project 2
**Group 18 | University of North Texas**

**Authors:** Ashish Rathnakar Shetty · Kushal Sai Venigalla

---

## Project Overview

This repository contains the complete implementation of our feature engineering framework for time-series data, submitted for CSCE 5222 (Feature Engineering) at UNT. Building on our Phase 1 proposal, we implemented and evaluated four feature engineering strategies across three tasks:

| Task | Dataset | Metric |
|---|---|---|
| Classification | UCR Archive (5 datasets) | Accuracy, F1-Score |
| Forecasting | ETTh1 (Electricity Transformer) | MSE, MAE |
| Anomaly Detection | WESAD-like PPG | F1-Score, ROC-AUC |

## Repository Structure

```
project2/
├── code/
│   ├── feature_engineering_pipeline.py   # Main experiment script
│   ├── generate_figures.py               # Visualization generator
│   └── README.md                         # Code documentation
├── figures/                              # All generated plots (12 figures)
├── results/                              # Experimental results (JSON)
├── report/
│   └── report.tex                        # IEEE LaTeX report (Overleaf-ready)
└── slides/
    └── Group18_Presentation.pptx         # Demo presentation
```

## How to Reproduce

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm shap

# 2. Run the full pipeline (downloads datasets automatically)
cd code
python feature_engineering_pipeline.py

# 3. Generate all figures
python generate_figures.py
```

## Key Results

- **Classification:** Catch22 + LightGBM improved accuracy by up to 6.3% over raw features (SyntheticControl). SHAP pruning removed 40% of features with zero accuracy loss.
- **Forecasting:** Lag + Rolling Statistics reduced MSE from 1.285 to 0.228 (82% improvement over raw lag-1 baseline).
- **Anomaly Detection:** SHAP-guided feature selection improved F1 from 0.581 to 0.710 compared to unguided Catch22.
