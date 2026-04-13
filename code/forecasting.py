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


def build_lag_features(series, lags=(1, 2, 3, 6, 12, 24), windows=(3, 6, 12, 24)):
    """Build lag and rolling statistics features with proper NaN handling."""
    df = pd.DataFrame({'value': series})
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    for w in windows:
        roll = df['value'].rolling(window=w)
        df[f'roll_mean_{w}'] = roll.mean()
        df[f'roll_std_{w}']  = roll.std()
        df[f'roll_min_{w}']  = roll.min()
        df[f'roll_max_{w}']  = roll.max()
    df.dropna(inplace=True)  # drop rows with NaN from lag/rolling
    return df
