"""Model factory. Optional deps (catboost, torch/pytorch-tabular) are skipped
gracefully if not installed, with a logged note — the pipeline still runs."""
from __future__ import annotations
from ..io_utils import LOG

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


def _has(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def available_models(cfg) -> dict:
    """Return {name: (estimator_factory, param_grid)} for models present in the env."""
    seed = cfg["seed"]
    reg = {}

    # classical
    reg["logreg"] = (lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed),
                     {"C": [0.1, 1.0, 10.0]})
    reg["gaussian_nb"] = (lambda: GaussianNB(), {"var_smoothing": [1e-9, 1e-7]})
    reg["svm"] = (lambda: SVC(probability=True, class_weight="balanced", random_state=seed),
                  {"C": [1.0, 10.0], "kernel": ["rbf", "linear"]})
    reg["random_forest"] = (lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                           class_weight="balanced_subsample",
                                                           random_state=seed),
                            {"max_depth": [None, 20]})
    reg["extra_trees"] = (lambda: ExtraTreesClassifier(n_estimators=300, n_jobs=-1,
                                                       class_weight="balanced",
                                                       random_state=seed),
                          {"max_depth": [None, 20]})
    reg["mlp"] = (lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                        early_stopping=True, random_state=seed),
                  {"alpha": [1e-4, 1e-3]})

    # boosting
    if _has("xgboost"):
        import xgboost as xgb
        reg["xgboost"] = (lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1,
            tree_method="hist", random_state=seed),
            {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]})
    else:
        LOG.info("xgboost not available -> skipped")
    if _has("lightgbm"):
        import lightgbm as lgb
        reg["lightgbm"] = (lambda: lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.9,
            class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1),
            {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1]})
    else:
        LOG.info("lightgbm not available -> skipped")
    if _has("catboost"):
        from catboost import CatBoostClassifier
        reg["catboost"] = (lambda: CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, verbose=0,
            random_seed=seed, auto_class_weights="Balanced"),
            {"depth": [4, 6]})
    else:
        LOG.info("catboost not available -> skipped (optional)")

    # deep tabular (TabNet / FT-Transformer) require torch + pytorch-tabular
    if not (_has("torch") and _has("pytorch_tabular")):
        LOG.info("torch/pytorch-tabular not available -> TabNet/FT-Transformer skipped (optional)")

    return reg
