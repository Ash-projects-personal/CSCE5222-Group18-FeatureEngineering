# anomaly_detection.py — unsupervised anomaly detection on PPG-like data
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score

def generate_ppg_signal(n=2000, seed=7):
    """Generate synthetic BVP signal with injected anomalies."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n)
    bvp = np.sin(t * 1.2) + 0.3 * np.sin(t * 3.6) + rng.normal(0, 0.1, n)
    labels = np.zeros(n, dtype=int)
    anomaly_segs = [(300, 340), (700, 730), (1100, 1160), (1500, 1540), (1800, 1840)]
    for start, end in anomaly_segs:
        kind = rng.randint(0, 3)
        if kind == 0:
            bvp[start:end] += rng.uniform(3, 5, end - start)   # spike
        elif kind == 1:
            bvp[start:end] = rng.uniform(-0.05, 0.05, end - start)  # flatline
        else:
            bvp[start:end] += rng.normal(0, 2.5, end - start)  # high noise
        labels[start:end] = 1
    return bvp, labels

def sliding_windows(signal, window=50, step=10):
    """Extract sliding windows from a signal."""
    X, y = [], []
    for start in range(0, len(signal) - window, step):
        X.append(signal[start:start + window])
        y.append(int(np.mean(signal[start:start + window]) > 0.3))
    return np.array(X), np.array(y)


def isolation_forest_detect(X, contamination=0.15):
    """Run Isolation Forest and return anomaly scores."""
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    scores = -iso.score_samples(X)
    thresh = np.percentile(scores, 100 * (1 - contamination))
    preds = (scores > thresh).astype(int)
    return scores, preds


# Analysis note (April 16):
# Catch22 features normalize the signal, which removes the amplitude
# information that makes spike anomalies detectable.
# However, features like diff_variance, time_reversibility, and
# lempel_ziv_complexity should still capture flatline and noise anomalies.
# Plan: use SHAP to identify which catch22 features are most useful
# for anomaly detection, then re-run with only those features.
