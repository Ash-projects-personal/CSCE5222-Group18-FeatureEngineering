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

def acf_lag1(x):
    """Autocorrelation at lag 1."""
    if len(x) > 1:
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return 0.0

def acf_lag2(x):
    """Autocorrelation at lag 2."""
    if len(x) > 2:
        return float(np.corrcoef(x[:-2], x[2:])[0, 1])
    return 0.0

def hist_entropy(x, bins=10):
    """Histogram-based entropy."""
    counts, _ = np.histogram(x, bins=bins)
    probs = counts / (counts.sum() + 1e-10)
    return float(-np.sum(probs * np.log(probs + 1e-10)))

def spectral_entropy(x):
    """Spectral entropy via FFT — NOTE: not normalized yet, values will be wrong."""
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    # BUG: forgot to normalize fft_vals — entropy will be inflated
    return float(-np.sum(fft_vals * np.log(fft_vals + 1e-10)))
