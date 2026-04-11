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

    # TODO: handle multiclass case where shap_vals is 3D
    mean_abs = np.abs(shap_vals).mean(axis=0)
    n_keep = max(1, int(len(feature_names) * keep_frac))
    top_idx = np.argsort(mean_abs)[::-1][:n_keep]
    return X_train[:, top_idx], X_test[:, top_idx], [feature_names[i] for i in top_idx]
