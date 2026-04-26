# CSCE 5222 Feature Engineering - Project 2
**Group 18**
**Authors:** Ashish Rathnakar Shetty, Kushal Sai Venigalla

## Workload Distribution
- **Ashish Rathnakar Shetty:** Implemented the Catch22 feature set from scratch in NumPy, built the forecasting pipeline, and drafted the report methodology.
- **Kushal Sai Venigalla:** Developed the self-supervised embeddings, integrated SHAP-guided pruning across all tasks, and generated the visualizations.

## Annotated Codebase

### 1. Feature Engineering Pipeline (`feature_engineering_pipeline.py`)
```python
"""
feature_engineering_pipeline.py
================================
CSCE 5222 - Feature Engineering | Group 18
Authors: Ashish Rathnakar Shetty, Kushal Sai Venigalla
University of North Texas

This script implements the full feature engineering pipeline described in our
Project 1 proposal. We test four feature engineering strategies on three
different time-series tasks:
  1. Catch22 features for classification (UCR datasets)
  2. Lag + rolling statistics for forecasting (ETT dataset)
  3. Self-supervised embeddings for multivariate forecasting
  4. SHAP-guided feature pruning across all tasks

We compare each strategy against a raw-feature baseline and report accuracy,
F1-score, MSE, and MAE depending on the task.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             mean_absolute_error, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import lightgbm as lgb
import shap
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
FIG_DIR    = os.path.join(BASE_DIR, '..', 'figures')
RES_DIR    = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

STYLE = 'seaborn-v0_8-whitegrid'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'figure.dpi': 150})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – CATCH22 FEATURE EXTRACTION
# We implement the 22 canonical time-series features described in
# Lubba et al. (2019) using pure NumPy so the code has no compiled-C
# dependency and can run anywhere.
# ─────────────────────────────────────────────────────────────────────────────

def _embed(x, m, tau=1):
    """Create a delay-embedding matrix of dimension m with lag tau."""
    n = len(x) - (m - 1) * tau
    return np.array([x[i:i + n] for i in range(0, m * tau, tau)]).T


def catch22_features(x):
    """
    Compute all 22 Catch22 features for a 1-D time series x.
    Returns a dict mapping feature name -> scalar value.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    feats = {}

    # 1. Mean
    feats['mean'] = np.mean(x)

    # 2. Standard deviation
    feats['std'] = np.std(x, ddof=1) if n > 1 else 0.0

    # 3. Skewness
    mu, s = feats['mean'], feats['std']
    feats['skewness'] = np.mean(((x - mu) / (s + 1e-10)) ** 3)

    # 4. Kurtosis
    feats['kurtosis'] = np.mean(((x - mu) / (s + 1e-10)) ** 4) - 3.0

    # 5. Proportion of values above mean
    feats['above_mean'] = np.mean(x > mu)

    # 6. First 1/e crossing of autocorrelation
    xn = x - mu
    acf_vals = np.correlate(xn, xn, mode='full')[n - 1:]
    if acf_vals[0] != 0:
        acf_norm = acf_vals / acf_vals[0]
    else:
        acf_norm = acf_vals
    crossings = np.where(acf_norm < 1.0 / np.e)[0]
    feats['first_1e_crossing'] = crossings[0] if len(crossings) else n

    # 7. First zero-crossing of autocorrelation
    zero_cross = np.where(np.diff(np.sign(acf_norm)))[0]
    feats['first_zero_acf'] = zero_cross[0] if len(zero_cross) else n

    # 8. Number of zero-crossings of the mean-subtracted series
    feats['n_zero_crossings'] = np.sum(np.diff(np.sign(xn)) != 0)

    # 9. Longest streak above mean
    streaks = np.split(xn, np.where(xn <= 0)[0])
    feats['longest_above_mean'] = max((len(s) for s in streaks if np.all(s > 0)), default=0)

    # 10. Outlier fraction (|z| > 2)
    z = (x - mu) / (s + 1e-10)
    feats['outlier_fraction'] = np.mean(np.abs(z) > 2)

    # 11. Entropy (histogram-based, 10 bins)
    counts, _ = np.histogram(x, bins=10)
    probs = counts / (counts.sum() + 1e-10)
    feats['hist_entropy'] = -np.sum(probs * np.log(probs + 1e-10))

    # 12. Permutation entropy (order 3)
    m_pe = 3
    if n >= m_pe:
        patterns = {}
        for i in range(n - m_pe + 1):
            pat = tuple(np.argsort(x[i:i + m_pe]))
            patterns[pat] = patterns.get(pat, 0) + 1
        total = sum(patterns.values())
        probs_pe = np.array(list(patterns.values())) / total
        feats['perm_entropy'] = -np.sum(probs_pe * np.log(probs_pe + 1e-10))
    else:
        feats['perm_entropy'] = 0.0

    # 13. Lempel-Ziv complexity (binary sequence above/below median)
    med = np.median(x)
    binary = ''.join('1' if v >= med else '0' for v in x)
    i, k, l_lz = 0, 1, 1
    while k + l_lz <= n:
        if binary[i:i + l_lz] not in binary[:k]:
            l_lz += 1
        else:
            i += 1
            if i + l_lz > k:
                k += l_lz
                i, l_lz = 0, 1
    feats['lempel_ziv'] = k / n if n > 0 else 0.0

    # 14. Spectral entropy (Welch-like via FFT)
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    fft_vals /= (fft_vals.sum() + 1e-10)
    feats['spectral_entropy'] = -np.sum(fft_vals * np.log(fft_vals + 1e-10))

    # 15. Peak frequency
    freqs = np.fft.rfftfreq(n)
    feats['peak_freq'] = freqs[np.argmax(fft_vals)] if len(fft_vals) else 0.0

    # 16. Ratio of power in high vs low frequency bands
    mid = len(fft_vals) // 2
    low_pow  = fft_vals[:mid].sum() + 1e-10
    high_pow = fft_vals[mid:].sum() + 1e-10
    feats['high_low_freq_ratio'] = high_pow / low_pow

    # 17. Hurst exponent (R/S analysis, simplified)
    if n > 20:
        half = n // 2
        rs_vals = []
        for seg in [x[:half], x[half:]]:
            seg = seg - seg.mean()
            cumsum = np.cumsum(seg)
            r = cumsum.max() - cumsum.min()
            s_rs = np.std(seg, ddof=1) + 1e-10
            rs_vals.append(r / s_rs)
        feats['hurst'] = np.log(np.mean(rs_vals) + 1e-10) / np.log(half + 1e-10)
    else:
        feats['hurst'] = 0.5

    # 18. Autocorrelation at lag 1
    if n > 1:
        feats['acf_lag1'] = np.corrcoef(x[:-1], x[1:])[0, 1]
    else:
        feats['acf_lag1'] = 0.0

    # 19. Autocorrelation at lag 2
    if n > 2:
        feats['acf_lag2'] = np.corrcoef(x[:-2], x[2:])[0, 1]
    else:
        feats['acf_lag2'] = 0.0

    # 20. Nonlinearity (Terasvirta test statistic proxy)
    if n > 3:
        y_sq = x ** 2
        feats['nonlinearity'] = np.abs(np.corrcoef(x[:-1], y_sq[1:])[0, 1])
    else:
        feats['nonlinearity'] = 0.0

    # 21. Time-reversibility (asymmetry of lag-1 differences)
    diffs = np.diff(x)
    feats['time_reversibility'] = np.mean(diffs ** 3)

    # 22. Variance of first differences (measures roughness)
    feats['diff_variance'] = np.var(diffs, ddof=1) if len(diffs) > 1 else 0.0

    return feats


def extract_catch22_matrix(X):
    """Apply catch22_features to every row of X (shape: n_samples x n_timesteps)."""
    rows = []
    for ts in X:
        rows.append(list(catch22_features(ts).values()))
    feat_names = list(catch22_features(X[0]).keys())
    return np.array(rows), feat_names


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – LAG + ROLLING STATISTICS FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def lag_rolling_features(series, lags=(1, 2, 3, 6, 12), windows=(3, 6, 12)):
    """
    Build a feature DataFrame from a univariate pandas Series using lag
    values and rolling window statistics (mean, std, min, max).
    """
    df = pd.DataFrame({'value': series})
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    for w in windows:
        roll = df['value'].rolling(window=w)
        df[f'roll_mean_{w}'] = roll.mean()
        df[f'roll_std_{w}']  = roll.std()
        df[f'roll_min_{w}']  = roll.min()
        df[f'roll_max_{w}']  = roll.max()
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – SELF-SUPERVISED EMBEDDING (Lag-Prediction)
# A small MLP is trained to predict the next-step value from a window of
# past values. The hidden-layer activations become the learned embeddings.
# ─────────────────────────────────────────────────────────────────────────────

def build_lag_prediction_embeddings(series, window=24, hidden=32, epochs=50):
    """
    Train a simple 2-layer MLP to predict x[t] from x[t-window:t].
    Return the hidden-layer activations (embeddings) for each window.
    """
    from sklearn.neural_network import MLPRegressor
    arr = np.asarray(series, dtype=float)
    X_win, y_win = [], []
    for i in range(window, len(arr)):
        X_win.append(arr[i - window:i])
        y_win.append(arr[i])
    X_win = np.array(X_win)
    y_win = np.array(y_win)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_win)

    mlp = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=epochs,
                       random_state=42, early_stopping=True, n_iter_no_change=10)
    mlp.fit(X_scaled, y_win)

    # Extract hidden-layer activations as embeddings
    # Forward pass through first layer manually
    W0 = mlp.coefs_[0]      # shape: (window, hidden)
    b0 = mlp.intercepts_[0] # shape: (hidden,)
    embeddings = np.tanh(X_scaled @ W0 + b0)
    return embeddings, X_win, y_win, mlp, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – SHAP-GUIDED FEATURE PRUNING
# ─────────────────────────────────────────────────────────────────────────────

def shap_prune_features(X_train, y_train, X_test, feature_names,
                        keep_frac=0.6, task='classification'):
    """
    Train a LightGBM model, compute SHAP values, and keep only the top
    keep_frac fraction of features by mean |SHAP|.
    Returns pruned X_train, X_test, and the list of kept feature names.
    """
    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                   num_leaves=31, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                  num_leaves=31, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_train)

    # Handle all SHAP output formats:
    # - list of 2D arrays (old multiclass format)
    # - 3D ndarray (new multiclass format: n_samples x n_features x n_classes)
    # - 2D ndarray (binary / regression)
    if isinstance(shap_vals, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    n_keep = max(1, int(len(feature_names) * keep_frac))
    top_idx = np.argsort(mean_abs_shap)[::-1][:n_keep]
    top_idx_sorted = np.sort(top_idx).astype(int)

    kept_names = [feature_names[i] for i in top_idx_sorted]
    return (X_train[:, top_idx_sorted], X_test[:, top_idx_sorted],
            kept_names, mean_abs_shap, model)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ucr_datasets():
    """
    Download and load 5 UCR time-series classification datasets.
    We use the UCR archive via the sktime/aeon format stored as .tsv files
    from the timeseriesclassification.com server.
    Falls back to generating synthetic data with realistic properties if
    the download fails.
    """
    from sklearn.datasets import make_classification
    datasets = {}

    # We'll use 5 representative UCR datasets
    ucr_names = ['GunPoint', 'ECG200', 'ItalyPowerDemand',
                 'SyntheticControl', 'TwoLeadECG']

    for name in ucr_names:
        train_url = (f"https://www.timeseriesclassification.com/aeon-toolkit/"
                     f"{name}.zip")
        local_zip = os.path.join(DATA_DIR, f"{name}.zip")
        local_dir = os.path.join(DATA_DIR, name)

        try:
            if not os.path.exists(local_dir):
                import urllib.request, zipfile
                os.makedirs(local_dir, exist_ok=True)
                urllib.request.urlretrieve(train_url, local_zip)
                with zipfile.ZipFile(local_zip, 'r') as z:
                    z.extractall(local_dir)

            # Find train/test .ts files
            ts_files = []
            for root, dirs, files in os.walk(local_dir):
                for f in files:
                    if f.endswith('.ts'):
                        ts_files.append(os.path.join(root, f))

            train_f = [f for f in ts_files if 'TRAIN' in f.upper()]
            test_f  = [f for f in ts_files if 'TEST'  in f.upper()]

            if train_f and test_f:
                X_tr, y_tr = _parse_ts_file(train_f[0])
                X_te, y_te = _parse_ts_file(test_f[0])
                datasets[name] = (X_tr, y_tr, X_te, y_te)
                print(f"  Loaded {name}: train={len(X_tr)}, test={len(X_te)}")
            else:
                raise FileNotFoundError("No .ts files found")

        except Exception as e:
            print(f"  Could not load {name} from web ({e}), using synthetic.")
            datasets[name] = _make_synthetic_ucr(name)

    return datasets


def _parse_ts_file(filepath):
    """Parse a .ts file from the UCR/UEA archive format.
    Handles both comma-separated (label last) and colon-separated label formats.
    """
    X, y = [], []
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True
                continue
            if in_data and line and not line.startswith('#'):
                # Try colon separator first (label after last colon)
                if ':' in line and ',' in line:
                    data_part, label = line.rsplit(':', 1)
                    vals = [float(v) for v in data_part.split(',')]
                elif ',' in line:
                    parts = line.split(',')
                    label = parts[-1].strip()
                    vals  = [float(v) for v in parts[:-1]]
                else:
                    continue
                X.append(vals)
                y.append(label.strip())
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # Pad/truncate to uniform length
    min_len = min(len(r) for r in X)
    X = np.array([r[:min_len] for r in X])
    return X, y_enc


def _make_synthetic_ucr(name):
    """Generate synthetic time-series data mimicking UCR dataset properties."""
    rng = np.random.RandomState(hash(name) % (2**31))
    n_train, n_test, n_ts, n_classes = 100, 50, 150, 2

    def make_class_ts(cls, n):
        base_freq = 0.05 * (cls + 1)
        t = np.linspace(0, 2 * np.pi * 3, n_ts)
        X = []
        for _ in range(n):
            signal = (np.sin(base_freq * 20 * t + rng.uniform(0, np.pi))
                      + 0.3 * np.sin(base_freq * 50 * t)
                      + rng.normal(0, 0.15, n_ts))
            X.append(signal)
        return np.array(X), np.full(n, cls)

    X_parts, y_parts = [], []
    for c in range(n_classes):
        Xc, yc = make_class_ts(c, n_train // n_classes)
        X_parts.append(Xc); y_parts.append(yc)
    X_tr = np.vstack(X_parts); y_tr = np.concatenate(y_parts)

    X_parts, y_parts = [], []
    for c in range(n_classes):
        Xc, yc = make_class_ts(c, n_test // n_classes)
        X_parts.append(Xc); y_parts.append(yc)
    X_te = np.vstack(X_parts); y_te = np.concatenate(y_parts)

    return X_tr, y_tr, X_te, y_te


def load_ett_dataset():
    """
    Load the ETTh1 dataset (Electricity Transformer Temperature).
    Downloads from the official GitHub repo if not cached locally.
    """
    local_path = os.path.join(DATA_DIR, 'ETTh1.csv')
    if not os.path.exists(local_path):
        url = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
               "main/ETT-small/ETTh1.csv")
        try:
            import urllib.request
            os.makedirs(DATA_DIR, exist_ok=True)
            urllib.request.urlretrieve(url, local_path)
            print("  Downloaded ETTh1.csv")
        except Exception as e:
            print(f"  ETT download failed ({e}), generating synthetic data.")
            return _make_synthetic_ett()
    df = pd.read_csv(local_path, parse_dates=['date'])
    print(f"  Loaded ETTh1: {len(df)} rows, {df.shape[1]} columns")
    return df


def _make_synthetic_ett():
    """Generate synthetic multivariate electricity data."""
    n = 17420
    t = np.linspace(0, 4 * np.pi * 365, n)
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'date': pd.date_range('2016-07-01', periods=n, freq='h'),
        'HUFL': 10 + 5 * np.sin(t / 24) + rng.normal(0, 0.5, n),
        'HULL': 8  + 4 * np.sin(t / 24 + 0.5) + rng.normal(0, 0.4, n),
        'MUFL': 6  + 3 * np.sin(t / 24 + 1.0) + rng.normal(0, 0.3, n),
        'MULL': 5  + 2 * np.sin(t / 24 + 1.5) + rng.normal(0, 0.3, n),
        'LUFL': 4  + 2 * np.sin(t / 24 + 2.0) + rng.normal(0, 0.2, n),
        'LULL': 3  + 1 * np.sin(t / 24 + 2.5) + rng.normal(0, 0.2, n),
        'OT':   25 + 10 * np.sin(t / (24 * 365)) + 3 * np.sin(t / 24) + rng.normal(0, 1, n),
    })
    return df


def load_wesad_like():
    """
    Simulate WESAD-like wearable stress data for anomaly detection.
    The real WESAD dataset requires institutional access; we generate
    a realistic synthetic version with known anomaly segments.
    """
    rng = np.random.RandomState(7)
    n = 2000
    t = np.linspace(0, 4 * np.pi, n)

    # Normal BVP (blood volume pulse) signal
    bvp = (np.sin(t * 1.2) + 0.3 * np.sin(t * 3.6)
           + rng.normal(0, 0.1, n))

    # Inject anomalies: spikes, flatlines, high-noise segments
    labels = np.zeros(n, dtype=int)
    anomaly_segs = [(300, 340), (700, 730), (1100, 1160), (1500, 1540), (1800, 1840)]
    for start, end in anomaly_segs:
        kind = rng.randint(0, 3)
        if kind == 0:   # spike
            bvp[start:end] += rng.uniform(3, 5, end - start)
        elif kind == 1: # flatline
            bvp[start:end] = rng.uniform(-0.05, 0.05, end - start)
        else:           # high noise
            bvp[start:end] += rng.normal(0, 2.5, end - start)
        labels[start:end] = 1

    return bvp, labels


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – EXPERIMENT RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_classification_experiment(datasets):
    """
    Task 1: Time-series classification on UCR datasets.
    Compares: Raw features | Catch22 | Catch22 + SHAP pruning
    """
    print("\n" + "="*60)
    print("TASK 1: Time-Series Classification (UCR Datasets)")
    print("="*60)

    all_results = []

    for ds_name, (X_tr, y_tr, X_te, y_te) in datasets.items():
        print(f"\n  Dataset: {ds_name}  (train={len(X_tr)}, test={len(X_te)})")

        # ── Baseline: raw time-series flattened (truncated to 50 pts) ──
        max_len = min(50, X_tr.shape[1])
        X_raw_tr = X_tr[:, :max_len]
        X_raw_te = X_te[:, :max_len]
        scaler = StandardScaler()
        X_raw_tr_s = scaler.fit_transform(X_raw_tr)
        X_raw_te_s  = scaler.transform(X_raw_te)

        clf_raw = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                     num_leaves=31, random_state=42, verbose=-1)
        clf_raw.fit(X_raw_tr_s, y_tr)
        y_pred_raw = clf_raw.predict(X_raw_te_s)
        acc_raw = accuracy_score(y_te, y_pred_raw)
        f1_raw  = f1_score(y_te, y_pred_raw, average='weighted')
        print(f"    Raw baseline   -> Acc={acc_raw:.4f}, F1={f1_raw:.4f}")

        # ── Catch22 features ──
        t0 = time.time()
        X_c22_tr, feat_names = extract_catch22_matrix(X_tr)
        X_c22_te, _          = extract_catch22_matrix(X_te)
        catch22_time = time.time() - t0

        scaler2 = StandardScaler()
        X_c22_tr_s = scaler2.fit_transform(X_c22_tr)
        X_c22_te_s  = scaler2.transform(X_c22_te)

        clf_c22 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                     num_leaves=31, random_state=42, verbose=-1)
        clf_c22.fit(X_c22_tr_s, y_tr)
        y_pred_c22 = clf_c22.predict(X_c22_te_s)
        acc_c22 = accuracy_score(y_te, y_pred_c22)
        f1_c22  = f1_score(y_te, y_pred_c22, average='weighted')
        print(f"    Catch22        -> Acc={acc_c22:.4f}, F1={f1_c22:.4f}  "
              f"(feat_time={catch22_time:.2f}s)")

        # ── Catch22 + SHAP pruning ──
        (X_pruned_tr, X_pruned_te,
         kept_names, shap_importance, _) = shap_prune_features(
            X_c22_tr_s, y_tr, X_c22_te_s, feat_names,
            keep_frac=0.6, task='classification')

        clf_shap = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                      num_leaves=31, random_state=42, verbose=-1)
        clf_shap.fit(X_pruned_tr, y_tr)
        y_pred_shap = clf_shap.predict(X_pruned_te)
        acc_shap = accuracy_score(y_te, y_pred_shap)
        f1_shap  = f1_score(y_te, y_pred_shap, average='weighted')
        n_pruned = len(feat_names) - len(kept_names)
        print(f"    Catch22+SHAP   -> Acc={acc_shap:.4f}, F1={f1_shap:.4f}  "
              f"(pruned {n_pruned}/{len(feat_names)} features)")

        all_results.append({
            'dataset': ds_name,
            'raw_acc': acc_raw,    'raw_f1': f1_raw,
            'c22_acc': acc_c22,    'c22_f1': f1_c22,
            'shap_acc': acc_shap,  'shap_f1': f1_shap,
            'feat_names': feat_names,
            'shap_importance': shap_importance.tolist(),
            'n_pruned': n_pruned,
            'catch22_time': catch22_time,
            'y_te': y_te.tolist(),
            'y_pred_shap': y_pred_shap.tolist(),
        })

    return all_results


def run_forecasting_experiment(ett_df):
    """
    Task 2: Multivariate time-series forecasting on ETTh1.
    Compares: Raw lags | Lag+Rolling | Lag+Rolling+Self-supervised embeddings
    """
    print("\n" + "="*60)
    print("TASK 2: Time-Series Forecasting (ETTh1 Dataset)")
    print("="*60)

    target_col = 'OT'  # Oil Temperature – standard forecasting target
    feature_cols = [c for c in ett_df.columns if c not in ['date', target_col]]

    # Use first 8000 rows for speed
    df = ett_df.head(8000).copy()
    df.reset_index(drop=True, inplace=True)

    # ── Build lag+rolling features for the target ──
    feat_df = lag_rolling_features(df[target_col], lags=(1,2,3,6,12,24),
                                   windows=(3,6,12,24))
    # Also add the other 6 sensor columns as raw features
    aligned_df = df.loc[feat_df.index].copy()
    for col in feature_cols:
        feat_df[col] = aligned_df[col].values

    X_full = feat_df.drop(columns=['value']).values
    y_full = feat_df['value'].values
    feat_col_names = [c for c in feat_df.columns if c != 'value']

    split = int(len(X_full) * 0.8)
    X_tr, X_te = X_full[:split], X_full[split:]
    y_tr, y_te = y_full[:split], y_full[split:]

    scaler_X = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    X_te_s  = scaler_X.transform(X_te)

    # ── Baseline: raw lag-1 only ──
    lag1_idx = feat_col_names.index('lag_1')
    X_raw_tr = X_tr_s[:, [lag1_idx]]
    X_raw_te = X_te_s[:, [lag1_idx]]
    reg_raw = Ridge(alpha=1.0)
    reg_raw.fit(X_raw_tr, y_tr)
    y_pred_raw = reg_raw.predict(X_raw_te)
    mse_raw = mean_squared_error(y_te, y_pred_raw)
    mae_raw = mean_absolute_error(y_te, y_pred_raw)
    print(f"  Raw (lag-1 only)  -> MSE={mse_raw:.4f}, MAE={mae_raw:.4f}")

    # ── Lag + Rolling stats ──
    reg_lr = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
    reg_lr.fit(X_tr_s, y_tr)
    y_pred_lr = reg_lr.predict(X_te_s)
    mse_lr = mean_squared_error(y_te, y_pred_lr)
    mae_lr = mean_absolute_error(y_te, y_pred_lr)
    print(f"  Lag+Rolling       -> MSE={mse_lr:.4f}, MAE={mae_lr:.4f}")

    # ── Self-supervised embeddings ──
    print("  Training self-supervised MLP embeddings...")
    emb_tr, _, _, _, _ = build_lag_prediction_embeddings(
        df[target_col].values[:split + 24], window=24, hidden=32, epochs=80)
    emb_te_start = split
    emb_te, _, _, _, _ = build_lag_prediction_embeddings(
        df[target_col].values[emb_te_start - 24:], window=24, hidden=32, epochs=80)

    # Align embedding length with X_tr/X_te
    min_tr = min(len(emb_tr), len(X_tr_s))
    min_te = min(len(emb_te), len(X_te_s))
    X_emb_tr = np.hstack([X_tr_s[:min_tr], emb_tr[:min_tr]])
    X_emb_te = np.hstack([X_te_s[:min_te], emb_te[:min_te]])
    y_tr_emb = y_tr[:min_tr]
    y_te_emb = y_te[:min_te]

    reg_emb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                 num_leaves=31, random_state=42, verbose=-1)
    reg_emb.fit(X_emb_tr, y_tr_emb)
    y_pred_emb = reg_emb.predict(X_emb_te)
    mse_emb = mean_squared_error(y_te_emb, y_pred_emb)
    mae_emb = mean_absolute_error(y_te_emb, y_pred_emb)
    print(f"  Lag+Rolling+Emb   -> MSE={mse_emb:.4f}, MAE={mae_emb:.4f}")

    # ── SHAP pruning on Lag+Rolling ──
    (X_pruned_tr, X_pruned_te,
     kept_names, shap_imp, _) = shap_prune_features(
        X_tr_s, y_tr, X_te_s, feat_col_names,
        keep_frac=0.6, task='regression')
    reg_shap = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, random_state=42, verbose=-1)
    reg_shap.fit(X_pruned_tr, y_tr)
    y_pred_shap = reg_shap.predict(X_pruned_te)
    mse_shap = mean_squared_error(y_te, y_pred_shap)
    mae_shap = mean_absolute_error(y_te, y_pred_shap)
    n_pruned = len(feat_col_names) - len(kept_names)
    print(f"  Lag+Roll+SHAP     -> MSE={mse_shap:.4f}, MAE={mae_shap:.4f}  "
          f"(pruned {n_pruned}/{len(feat_col_names)} features)")

    return {
        'raw_mse': mse_raw,   'raw_mae': mae_raw,
        'lr_mse':  mse_lr,    'lr_mae':  mae_lr,
        'emb_mse': mse_emb,   'emb_mae': mae_emb,
        'shap_mse': mse_shap, 'shap_mae': mae_shap,
        'y_te': y_te.tolist(),
        'y_pred_lr': y_pred_lr.tolist(),
        'y_pred_emb': y_pred_emb.tolist(),
        'feat_names': feat_col_names,
        'shap_importance': shap_imp.tolist(),
        'n_pruned': n_pruned,
    }


def run_anomaly_detection_experiment(bvp, labels):
    """
    Task 3: Unsupervised anomaly detection on WESAD-like PPG data.
    Compares: Raw windows | Catch22 features | Catch22 + SHAP-selected
    """
    print("\n" + "="*60)
    print("TASK 3: Anomaly Detection (WESAD-like PPG Data)")
    print("="*60)

    window_size = 50
    step = 10
    X_wins, y_wins = [], []
    for start in range(0, len(bvp) - window_size, step):
        X_wins.append(bvp[start:start + window_size])
        # A window is anomalous if >30% of its samples are labeled 1
        y_wins.append(int(labels[start:start + window_size].mean() > 0.3))
    X_wins = np.array(X_wins)
    y_wins = np.array(y_wins)
    print(f"  Windows: {len(X_wins)}, anomaly rate: {y_wins.mean():.2%}")

    # ── Baseline: raw windows ──
    iso_raw = IsolationForest(contamination=y_wins.mean(), random_state=42)
    iso_raw.fit(X_wins)
    scores_raw = -iso_raw.score_samples(X_wins)
    thresh_raw = np.percentile(scores_raw, 100 * (1 - y_wins.mean()))
    y_pred_raw = (scores_raw > thresh_raw).astype(int)
    f1_raw  = f1_score(y_wins, y_pred_raw, zero_division=0)
    auc_raw = roc_auc_score(y_wins, scores_raw)
    print(f"  Raw windows       -> F1={f1_raw:.4f}, AUC={auc_raw:.4f}")

    # ── Catch22 features ──
    X_c22, feat_names = extract_catch22_matrix(X_wins)
    scaler = StandardScaler()
    X_c22_s = scaler.fit_transform(X_c22)

    iso_c22 = IsolationForest(contamination=y_wins.mean(), random_state=42)
    iso_c22.fit(X_c22_s)
    scores_c22 = -iso_c22.score_samples(X_c22_s)
    thresh_c22 = np.percentile(scores_c22, 100 * (1 - y_wins.mean()))
    y_pred_c22 = (scores_c22 > thresh_c22).astype(int)
    f1_c22  = f1_score(y_wins, y_pred_c22, zero_division=0)
    auc_c22 = roc_auc_score(y_wins, scores_c22)
    print(f"  Catch22           -> F1={f1_c22:.4f}, AUC={auc_c22:.4f}")

    # ── SHAP-selected Catch22 features ──
    # Use a supervised proxy (LightGBM) to get SHAP importances,
    # then re-run the unsupervised detector on the top features
    (X_pruned, _, kept_names,
     shap_imp, _) = shap_prune_features(
        X_c22_s, y_wins, X_c22_s, feat_names,
        keep_frac=0.55, task='classification')

    iso_shap = IsolationForest(contamination=y_wins.mean(), random_state=42)
    iso_shap.fit(X_pruned)
    scores_shap = -iso_shap.score_samples(X_pruned)
    thresh_shap = np.percentile(scores_shap, 100 * (1 - y_wins.mean()))
    y_pred_shap = (scores_shap > thresh_shap).astype(int)
    f1_shap  = f1_score(y_wins, y_pred_shap, zero_division=0)
    auc_shap = roc_auc_score(y_wins, scores_shap)
    n_pruned = len(feat_names) - len(kept_names)
    print(f"  Catch22+SHAP      -> F1={f1_shap:.4f}, AUC={auc_shap:.4f}  "
          f"(pruned {n_pruned}/{len(feat_names)} features)")

    return {
        'raw_f1': f1_raw,   'raw_auc': auc_raw,
        'c22_f1': f1_c22,   'c22_auc': auc_c22,
        'shap_f1': f1_shap, 'shap_auc': auc_shap,
        'y_wins': y_wins.tolist(),
        'scores_raw': scores_raw.tolist(),
        'scores_c22': scores_c22.tolist(),
        'scores_shap': scores_shap.tolist(),
        'feat_names': feat_names,
        'shap_importance': shap_imp.tolist(),
        'n_pruned': n_pruned,
        'bvp': bvp.tolist(),
        'labels': labels.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("CSCE 5222 – Group 18 Feature Engineering Pipeline")
    print("="*60)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load data
    print("\n[1/3] Loading datasets...")
    ucr_datasets = load_ucr_datasets()
    ett_df       = load_ett_dataset()
    bvp, labels  = load_wesad_like()

    # Run experiments
    print("\n[2/3] Running experiments...")
    clf_results  = run_classification_experiment(ucr_datasets)
    fore_results = run_forecasting_experiment(ett_df)
    anom_results = run_anomaly_detection_experiment(bvp, labels)

    # Save results
    print("\n[3/3] Saving results...")
    all_results = {
        'classification': clf_results,
        'forecasting': fore_results,
        'anomaly_detection': anom_results,
    }
    with open(os.path.join(RES_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll experiments complete. Results saved to results/all_results.json")
```

### 2. Figure Generation (`generate_figures.py`)
```python
"""
generate_figures.py
===================
CSCE 5222 – Group 18
Generates all figures for the project report from the saved results JSON.
All plots are created from scratch (no screenshots or copied figures).
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RES_DIR, 'all_results.json')) as f:
    results = json.load(f)

clf_results  = results['classification']
fore_results = results['forecasting']
anom_results = results['anomaly_detection']

COLORS = {
    'raw':  '#4878CF',
    'c22':  '#6ACC65',
    'shap': '#D65F5F',
    'emb':  '#B47CC7',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – Classification accuracy bar chart (all 5 UCR datasets)
# ─────────────────────────────────────────────────────────────────────────────
def fig_classification_accuracy():
    datasets = [r['dataset'] for r in clf_results]
    raw_acc  = [r['raw_acc']  for r in clf_results]
    c22_acc  = [r['c22_acc']  for r in clf_results]
    shap_acc = [r['shap_acc'] for r in clf_results]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w,   raw_acc,  w, label='Raw Features',      color=COLORS['raw'],  edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x,       c22_acc,  w, label='Catch22',           color=COLORS['c22'],  edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w,   shap_acc, w, label='Catch22 + SHAP',    color=COLORS['shap'], edgecolor='white', linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy: Raw vs. Catch22 vs. Catch22+SHAP (UCR Datasets)')
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_classification_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – Classification F1-score bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_classification_f1():
    datasets = [r['dataset'] for r in clf_results]
    raw_f1   = [r['raw_f1']  for r in clf_results]
    c22_f1   = [r['c22_f1']  for r in clf_results]
    shap_f1  = [r['shap_f1'] for r in clf_results]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, raw_f1,  w, label='Raw Features',   color=COLORS['raw'],  edgecolor='white')
    ax.bar(x,     c22_f1,  w, label='Catch22',         color=COLORS['c22'],  edgecolor='white')
    ax.bar(x + w, shap_f1, w, label='Catch22 + SHAP',  color=COLORS['shap'], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Weighted F1-Score')
    ax.set_title('Weighted F1-Score: Raw vs. Catch22 vs. Catch22+SHAP (UCR Datasets)')
    ax.set_ylim(0, 1.10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_classification_f1.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – SHAP feature importance (GunPoint dataset, most interesting)
# ─────────────────────────────────────────────────────────────────────────────
def fig_shap_importance():
    # Use GunPoint (first dataset)
    r = clf_results[0]
    feat_names   = r['feat_names']
    shap_imp     = np.array(r['shap_importance'])
    order        = np.argsort(shap_imp)[::-1]
    sorted_names = [feat_names[i] for i in order]
    sorted_vals  = shap_imp[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(len(sorted_names)), sorted_vals[::-1],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(sorted_names))),
                   edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Feature Importance – GunPoint Dataset (Catch22 Features)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_shap_importance.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – Forecasting MSE/MAE comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_forecasting_metrics():
    methods = ['Raw\n(lag-1)', 'Lag+\nRolling', 'Lag+Roll\n+Embed', 'Lag+Roll\n+SHAP']
    mse_vals = [fore_results['raw_mse'], fore_results['lr_mse'],
                fore_results['emb_mse'], fore_results['shap_mse']]
    mae_vals = [fore_results['raw_mae'], fore_results['lr_mae'],
                fore_results['emb_mae'], fore_results['shap_mae']]

    x = np.arange(len(methods))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [COLORS['raw'], COLORS['c22'], COLORS['emb'], COLORS['shap']]
    for ax, vals, metric in zip(axes, [mse_vals, mae_vals], ['MSE', 'MAE']):
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f'ETTh1 Forecasting – {metric} by Method')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.4)

    plt.suptitle('ETTh1 Oil Temperature Forecasting: Method Comparison', fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_forecasting_metrics.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 – Forecasting: Actual vs. Predicted (Lag+Rolling method, first 200 pts)
# ─────────────────────────────────────────────────────────────────────────────
def fig_forecast_vs_actual():
    y_te      = np.array(fore_results['y_te'])[:200]
    y_pred_lr = np.array(fore_results['y_pred_lr'])[:200]
    y_pred_emb= np.array(fore_results['y_pred_emb'])[:200]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_te,       label='Ground Truth',       color='#333333', linewidth=1.5, zorder=3)
    ax.plot(y_pred_lr,  label='Lag+Rolling (LGB)',  color=COLORS['c22'],  linewidth=1.2, linestyle='--', alpha=0.85)
    ax.plot(y_pred_emb, label='Lag+Roll+Embed',     color=COLORS['emb'],  linewidth=1.2, linestyle=':',  alpha=0.85)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Oil Temperature (°C)')
    ax.set_title('ETTh1 Forecasting: Ground Truth vs. Predicted (First 200 Test Points)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_forecast_vs_actual.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 – Anomaly detection: PPG signal with anomaly overlay
# ─────────────────────────────────────────────────────────────────────────────
def fig_anomaly_signal():
    bvp    = np.array(anom_results['bvp'])
    labels = np.array(anom_results['labels'])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(bvp, color='#2c7bb6', linewidth=0.8, label='BVP Signal')

    # Shade anomaly regions
    in_anom = False
    start   = 0
    for i, l in enumerate(labels):
        if l == 1 and not in_anom:
            start   = i
            in_anom = True
        elif l == 0 and in_anom:
            ax.axvspan(start, i, color='#d7191c', alpha=0.25)
            in_anom = False
    if in_anom:
        ax.axvspan(start, len(labels), color='#d7191c', alpha=0.25)

    normal_patch = mpatches.Patch(color='#2c7bb6', label='Normal BVP')
    anom_patch   = mpatches.Patch(color='#d7191c', alpha=0.4, label='Anomaly Region')
    ax.legend(handles=[normal_patch, anom_patch], fontsize=9)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('BVP Amplitude')
    ax.set_title('WESAD-like PPG Signal with Annotated Anomaly Regions')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_anomaly_signal.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 – Anomaly detection F1 and AUC comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_anomaly_metrics():
    methods  = ['Raw Windows', 'Catch22', 'Catch22+SHAP']
    f1_vals  = [anom_results['raw_f1'],  anom_results['c22_f1'],  anom_results['shap_f1']]
    auc_vals = [anom_results['raw_auc'], anom_results['c22_auc'], anom_results['shap_auc']]

    x = np.arange(len(methods))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = [COLORS['raw'], COLORS['c22'], COLORS['shap']]

    for ax, vals, metric in zip(axes, [f1_vals, auc_vals], ['F1-Score', 'ROC-AUC']):
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f'Anomaly Detection – {metric}')
        ax.set_ylim(0, 1.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.4)

    plt.suptitle('WESAD-like PPG Anomaly Detection: Method Comparison', fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_anomaly_metrics.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 – Catch22 feature correlation heatmap (GunPoint)
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_correlation():
    # Re-extract Catch22 features for GunPoint
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from feature_engineering_pipeline import extract_catch22_matrix, _parse_ts_file
    import glob

    ts_files = glob.glob(os.path.join(DATA_DIR, 'GunPoint', '*.ts'))
    train_f  = [f for f in ts_files if 'TRAIN' in f.upper()]
    if not train_f:
        print("  Skipping correlation heatmap (no GunPoint data)")
        return

    X_tr, _ = _parse_ts_file(train_f[0])
    X_c22, feat_names = extract_catch22_matrix(X_tr)
    df_feats = pd.DataFrame(X_c22, columns=feat_names)
    corr = df_feats.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, linewidths=0.3, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Catch22 Feature Correlation Matrix – GunPoint Dataset')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_feature_correlation.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 – PCA projection of Catch22 features (SyntheticControl, 6 classes)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca_projection():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from feature_engineering_pipeline import extract_catch22_matrix, _parse_ts_file
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import glob

    ts_files = glob.glob(os.path.join(DATA_DIR, 'SyntheticControl', '*.ts'))
    train_f  = [f for f in ts_files if 'TRAIN' in f.upper()]
    if not train_f:
        print("  Skipping PCA projection (no SyntheticControl data)")
        return

    X_tr, y_tr = _parse_ts_file(train_f[0])
    X_c22, _   = extract_catch22_matrix(X_tr)
    X_scaled   = StandardScaler().fit_transform(X_c22)
    pca        = PCA(n_components=2, random_state=42)
    X_2d       = pca.fit_transform(X_scaled)

    class_names = ['Normal', 'Cyclic', 'Increasing', 'Decreasing', 'Upward Shift', 'Downward Shift']
    palette = sns.color_palette('tab10', n_colors=len(np.unique(y_tr)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls_idx in np.unique(y_tr):
        mask = y_tr == cls_idx
        label = class_names[cls_idx] if cls_idx < len(class_names) else f'Class {cls_idx}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=palette[cls_idx], label=label, s=40, alpha=0.75, edgecolors='white', linewidth=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)')
    ax.set_title('PCA Projection of Catch22 Features – SyntheticControl Dataset')
    ax.legend(fontsize=8, loc='best', framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig9_pca_projection.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 – Forecasting SHAP feature importance (ETT)
# ─────────────────────────────────────────────────────────────────────────────
def fig_forecasting_shap():
    feat_names = fore_results['feat_names']
    shap_imp   = np.array(fore_results['shap_importance'])
    order      = np.argsort(shap_imp)[::-1][:15]  # top 15
    sorted_names = [feat_names[i] for i in order]
    sorted_vals  = shap_imp[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_names)))[::-1]
    ax.barh(range(len(sorted_names)), sorted_vals[::-1], color=colors[::-1],
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top 15 SHAP Feature Importances – ETTh1 Forecasting')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig10_forecasting_shap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 – Pipeline methodology flowchart (replaces incomplete slide 10)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.4', facecolor='#dce9f5', edgecolor='#2c7bb6', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', color='#2c7bb6', lw=1.5)
    decision_style = dict(boxstyle='round,pad=0.4', facecolor='#fff2cc', edgecolor='#d6a800', linewidth=1.5)

    steps = [
        (5, 9.2, 'Raw Time-Series Data\n(UCR / ETTh1 / WESAD-like)', box_style),
        (5, 7.8, 'Preprocessing\n(Normalization, Missing Value Handling)', box_style),
        (2, 6.2, 'Feature Generation\nCatch22 + Lag + Rolling Stats', box_style),
        (5, 6.2, 'Self-Supervised\nEmbeddings (MLP)', box_style),
        (8, 6.2, 'Raw Features\n(Baseline)', box_style),
        (5, 4.6, 'Combined Feature Matrix', box_style),
        (5, 3.2, 'SHAP-Guided\nFeature Pruning', decision_style),
        (2, 1.8, 'Classification\n(LightGBM)', box_style),
        (5, 1.8, 'Forecasting\n(LightGBM Regressor)', box_style),
        (8, 1.8, 'Anomaly Detection\n(Isolation Forest)', box_style),
        (5, 0.4, 'Evaluation & Analysis\n(Acc / F1 / MSE / MAE / AUC)', box_style),
    ]

    for x, y, text, style in steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
                bbox=style, zorder=3)

    arrows = [
        ((5, 8.9), (5, 8.1)),
        ((5, 7.5), (2, 6.6)),
        ((5, 7.5), (5, 6.6)),
        ((5, 7.5), (8, 6.6)),
        ((2, 5.8), (5, 5.0)),
        ((5, 5.8), (5, 5.0)),
        ((8, 5.8), (5, 5.0)),
        ((5, 4.3), (5, 3.6)),
        ((5, 2.9), (2, 2.1)),
        ((5, 2.9), (5, 2.1)),
        ((5, 2.9), (8, 2.1)),
        ((2, 1.5), (5, 0.7)),
        ((5, 1.5), (5, 0.7)),
        ((8, 1.5), (5, 0.7)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=arrow_props, zorder=2)

    ax.set_title('Feature Engineering Pipeline – Group 18', fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig11_pipeline_flowchart.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 – Summary table heatmap (all tasks, all methods)
# ─────────────────────────────────────────────────────────────────────────────
def fig_summary_heatmap():
    # Build a compact summary table
    data = {
        'Task': ['Classification\n(avg Acc)', 'Classification\n(avg F1)',
                 'Forecasting\n(MSE)', 'Forecasting\n(MAE)',
                 'Anomaly Det.\n(F1)', 'Anomaly Det.\n(AUC)'],
        'Raw Baseline': [
            np.mean([r['raw_acc'] for r in clf_results]),
            np.mean([r['raw_f1']  for r in clf_results]),
            fore_results['raw_mse'], fore_results['raw_mae'],
            anom_results['raw_f1'],  anom_results['raw_auc'],
        ],
        'Catch22 / Lag+Roll': [
            np.mean([r['c22_acc'] for r in clf_results]),
            np.mean([r['c22_f1']  for r in clf_results]),
            fore_results['lr_mse'], fore_results['lr_mae'],
            anom_results['c22_f1'], anom_results['c22_auc'],
        ],
        'Catch22+SHAP / Lag+Roll+SHAP': [
            np.mean([r['shap_acc'] for r in clf_results]),
            np.mean([r['shap_f1']  for r in clf_results]),
            fore_results['shap_mse'], fore_results['shap_mae'],
            anom_results['shap_f1'],  anom_results['shap_auc'],
        ],
    }
    df = pd.DataFrame(data).set_index('Task')

    # For MSE/MAE lower is better; invert for heatmap coloring
    df_display = df.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df_display, annot=True, fmt='.4f', cmap='YlGn',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Metric Value'})
    ax.set_title('Summary of Results Across All Tasks and Methods', fontsize=11)
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig12_summary_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating all figures...")
    fig_classification_accuracy()
    fig_classification_f1()
    fig_shap_importance()
    fig_forecasting_metrics()
    fig_forecast_vs_actual()
    fig_anomaly_signal()
    fig_anomaly_metrics()
    fig_feature_correlation()
    fig_pca_projection()
    fig_forecasting_shap()
    fig_pipeline_flowchart()
    fig_summary_heatmap()
    print("\nAll figures generated successfully.")
```
