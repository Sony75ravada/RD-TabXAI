"""Explainability: SHAP (global + local) and permutation importance.

Designed for the tabular evidence models (T-B ranker and T-C classifier). SHAP is
attempted with TreeExplainer for tree models and falls back to a model-agnostic
explainer; if SHAP is unavailable, permutation importance still runs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..io_utils import LOG


def permutation_importance_df(model, X, y, feature_names, n_repeats=10, seed=0,
                              scoring="accuracy"):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed,
                               scoring=scoring)
    df = pd.DataFrame({"feature": feature_names,
                       "importance_mean": r.importances_mean,
                       "importance_std": r.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_global(model, X, feature_names, max_samples=200, seed=0):
    """Return mean |SHAP| per feature. Robust to model type and SHAP version."""
    try:
        import shap
    except Exception:
        LOG.warning("shap not installed -> skipping SHAP, use permutation importance")
        return None
    rng = np.random.RandomState(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
    except Exception:
        try:
            explainer = shap.Explainer(model.predict, X)
            sv = explainer(X).values
        except Exception as e:
            LOG.warning("SHAP failed: %s", e)
            return None
    sv = np.array(sv)
    if sv.ndim == 3:        # (classes, samples, features) or (samples, features, classes)
        sv = np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != sv.ndim - 1)) \
            if sv.shape[-1] == len(feature_names) else np.abs(sv).mean(axis=(0, 1))
    else:
        sv = np.abs(sv).mean(axis=0)
    sv = np.ravel(sv)[:len(feature_names)]
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": sv})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(model, x_row, feature_names, top_n=8):
    """Per-case contribution for a single instance (top_n features)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = np.array(explainer.shap_values(x_row.reshape(1, -1)))
        if sv.ndim == 3:
            sv = sv[0]
        contrib = np.ravel(sv)[:len(feature_names)]
    except Exception:
        contrib = np.zeros(len(feature_names))
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [{"feature": feature_names[i], "contribution": float(contrib[i]),
             "value": float(x_row[i])} for i in order]
