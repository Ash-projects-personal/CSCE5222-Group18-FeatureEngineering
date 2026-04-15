# embeddings.py — self-supervised MLP embeddings for time-series
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def build_windows(series, window=24):
    """Create sliding windows from a time series."""
    arr = np.asarray(series, dtype=float)
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i])
    return np.array(X), np.array(y)

def train_mlp_embeddings(series, window=24, hidden=32, epochs=80):
    """Train MLP to predict next value; use hidden layer as embeddings."""
    X, y = build_windows(series, window)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mlp = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=epochs,
                       random_state=42, early_stopping=True,
                       n_iter_no_change=10, validation_fraction=0.1)
    mlp.fit(X_scaled, y)
    W0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    embeddings = np.tanh(X_scaled @ W0 + b0)
    return embeddings, mlp, scaler
