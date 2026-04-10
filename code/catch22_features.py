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
    """Spectral entropy via FFT — fixed: normalize before computing entropy."""
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    fft_vals = fft_vals / (fft_vals.sum() + 1e-10)  # normalize to probability dist
    return float(-np.sum(fft_vals * np.log(fft_vals + 1e-10)))

def permutation_entropy(x, m=3):
    """Permutation entropy with embedding dimension m."""
    n = len(x)
    if n < m:
        return 0.0
    patterns = {}
    for i in range(n - m + 1):
        pat = tuple(np.argsort(x[i:i + m]))
        patterns[pat] = patterns.get(pat, 0) + 1
    total = sum(patterns.values())
    probs = np.array(list(patterns.values())) / total
    return float(-np.sum(probs * np.log(probs + 1e-10)))

def hurst_exponent(x):
    """Simplified Hurst exponent via R/S analysis."""
    n = len(x)
    if n <= 20:
        return 0.5
    half = n // 2
    rs_vals = []
    for seg in [x[:half], x[half:]]:
        seg = seg - seg.mean()
        cumsum = np.cumsum(seg)
        r = cumsum.max() - cumsum.min()
        s = np.std(seg, ddof=1) + 1e-10
        rs_vals.append(r / s)
    return float(np.log(np.mean(rs_vals) + 1e-10) / np.log(half + 1e-10))

def lempel_ziv_complexity(x):
    """Lempel-Ziv complexity on binary sequence (above/below median)."""
    n = len(x)
    med = np.median(x)
    binary = ''.join('1' if v >= med else '0' for v in x)
    i, k, lz = 0, 1, 1
    while k + lz <= n:
        if binary[i:i + lz] not in binary[:k]:
            lz += 1
        else:
            i += 1
            if i + lz > k:
                k += lz
                i, lz = 0, 1
    return k / n if n > 0 else 0.0
