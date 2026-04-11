# shap_pruner.py — SHAP-guided feature selection
import numpy as np
import shap
import lightgbm as lgb

def shap_prune(X_train, y_train, X_test, feature_names, keep_frac=0.6, task='classification'):
    """Train LightGBM, compute SHAP values, keep top features by mean |SHAP|."""
    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)

    # Handle all SHAP output formats
    if isinstance(shap_vals, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        mean_abs = np.abs(shap_vals).mean(axis=(0, 2))  # fix for multiclass 3D
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)
    n_keep = max(1, int(len(feature_names) * keep_frac))
    top_idx = np.argsort(mean_abs)[::-1][:n_keep]
    return X_train[:, top_idx], X_test[:, top_idx], [feature_names[i] for i in top_idx]

# KNOWN BUG: When dataset has >2 classes (e.g. SyntheticControl with 6 classes),
# shap_vals comes back as a 3D ndarray (n_samples x n_features x n_classes)
# and mean(axis=0) gives a 2D matrix, not a 1D vector.
# This causes TypeError when indexing feature_names.
# Fix needed: detect ndim and reduce accordingly.
