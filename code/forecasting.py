# forecasting.py — ETTh1 multivariate forecasting pipeline
import os
import pandas as pd
import numpy as np

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def load_ett():
    """Download and load the ETTh1 dataset."""
    local = os.path.join(DATA_DIR, 'ETTh1.csv')
    if not os.path.exists(local):
        import urllib.request
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(ETT_URL, local)
    df = pd.read_csv(local, parse_dates=['date'])
    return df


def build_lag_features_v1(series, lags=(1, 2, 3)):
    """Build lag features — v1: has off-by-one alignment bug."""
    df = pd.DataFrame({'value': series})
    for lag in lags:
        # BUG: shift(lag) is correct but we're not dropping NaN rows
        # so the first `lag` rows will have NaN features but valid targets
        df[f'lag_{lag}'] = df['value'].shift(lag)
    return df  # NaN rows not dropped — will cause issues in training
