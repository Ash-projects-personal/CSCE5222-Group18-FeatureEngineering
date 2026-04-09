# catch22_features.py  — fixed skewness formula
import numpy as np

def mean_val(x):
    return np.mean(x)

def std_val(x):
    return np.std(x, ddof=1)

def skewness(x):
    mu = np.mean(x)
    s = np.std(x, ddof=1) + 1e-10
    # Fixed: standardize by s only, not s**2
    return np.mean(((x - mu) / s) ** 3)

def kurtosis(x):
    mu = np.mean(x)
    s = np.std(x, ddof=1) + 1e-10
    return np.mean(((x - mu) / s) ** 4) - 3.0

def above_mean_fraction(x):
    return np.mean(x > np.mean(x))

def n_zero_crossings(x):
    xn = x - np.mean(x)
    return int(np.sum(np.diff(np.sign(xn)) != 0))
