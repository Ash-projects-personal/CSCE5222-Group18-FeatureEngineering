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
