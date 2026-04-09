# catch22_features.py  — first attempt at implementing Catch22 features
# NOTE: skewness formula uses wrong denominator (s**2 instead of s**3)
import numpy as np

def mean_val(x):
    return np.mean(x)

def std_val(x):
    return np.std(x, ddof=1)

def skewness(x):
    mu, s = np.mean(x), np.std(x)
    # BUG: dividing by s**2 instead of s**3 — wrong formula
    return np.mean(((x - mu) / (s + 1e-10)) ** 3) / (s ** 2 + 1e-10)

def kurtosis(x):
    mu, s = np.mean(x), np.std(x, ddof=1)
    return np.mean(((x - mu) / (s + 1e-10)) ** 4) - 3.0

def above_mean_fraction(x):
    return np.mean(x > np.mean(x))
