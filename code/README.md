# CSCE 5222 Feature Engineering - Project 2
**Group 18**
**Authors:** Ashish Rathnakar Shetty, Kushal Sai Venigalla

## Overview
This repository contains the full implementation of our proposed feature engineering pipeline for time-series data. We evaluate four feature engineering strategies across three distinct tasks:
1. **Classification (UCR Datasets):** Catch22 features vs. Raw features
2. **Forecasting (ETTh1):** Lag + Rolling Statistics + Self-Supervised Embeddings vs. Raw Lag-1
3. **Anomaly Detection (WESAD-like PPG):** Catch22 features vs. Raw windows
4. **Automated Pruning (All Tasks):** SHAP-guided feature selection

## Project Structure
- `feature_engineering_pipeline.py`: The main script that runs all experiments. It implements the Catch22 features from scratch in pure NumPy, trains self-supervised MLP embeddings, and performs SHAP-guided pruning using LightGBM.
- `generate_figures.py`: Generates all visualizations, charts, and the methodology flowchart used in the final report based on the experimental results.
- `../data/`: Directory where datasets (UCR, ETTh1) are automatically downloaded and stored.
- `../results/`: Directory where the raw experimental results are saved as JSON.
- `../figures/`: Directory where all generated plots and charts are saved.

## Requirements
The pipeline is designed to run in a standard Python 3 environment.
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm shap
```
*Note: We implemented the Catch22 feature set manually in Python to avoid C-compiler dependencies associated with the `pycatch22` package.*

## How to Run
1. **Run the Pipeline:**
   ```bash
   python feature_engineering_pipeline.py
   ```
   This will download the necessary datasets, run all three tasks, and save the metrics to `../results/all_results.json`.

2. **Generate Figures:**
   ```bash
   python generate_figures.py
   ```
   This will read the results JSON and generate all the high-quality plots found in our report, saving them to the `../figures/` directory.

## Results Highlights
- **Classification:** Catch22 features significantly outperformed raw features on datasets like SyntheticControl (0.950 vs 0.886). SHAP pruning successfully reduced the feature space by 40% with zero loss in accuracy.
- **Forecasting:** Lag and rolling statistics drastically reduced the MSE on ETTh1 from 1.285 (raw lag-1) to 0.228.
- **Anomaly Detection:** SHAP-guided feature selection improved the F1-score of the Catch22-based Isolation Forest from 0.581 to 0.710 by removing noisy features.
