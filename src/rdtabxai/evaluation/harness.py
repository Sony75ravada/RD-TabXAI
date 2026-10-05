"""Training/evaluation harness.

Critical leakage controls enforced here:
  - PhenotypeEncoder vocabulary fit on TRAIN profiles only
  - Information content computed from TRAIN diseases only
  - Hyperparameters tuned by inner CV on TRAIN; test touched once
  - Splits are group-disjoint (see splits.py)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.calibration import CalibratedClassifierCV

from ..io_utils import LOG
from ..data.hpo import information_content
from ..features.encoding import PhenotypeEncoder
from .splits import make_split
from .metrics import classification_metrics, calibration_metrics, ranking_metrics
from ..models.registry import available_models


# ============================================================= T-C / T-A
def run_classification(profiles, disease_terms, ancestors, cfg, target_col="Disease_Group",
                       strategy="disease_holdout", encoder_kwargs=None, tune=True):
    seed = cfg["seed"]
    encoder_kwargs = encoder_kwargs or {}
    df = profiles.reset_index(drop=True)
    tr_mask, te_mask = make_split(df, strategy, cfg["splits"]["test_size"], seed)

    # IC from TRAIN diseases only
    train_orpha = set(df.loc[tr_mask, "ORPHA_ID"])
    dh_long = pd.DataFrame([(o, t) for o, ts in disease_terms.items() if o in train_orpha
                            for t in ts], columns=["ORPHA_ID", "HPO_ID"])
    ic = information_content(dh_long, ancestors)

    term_sets = df["terms"].apply(lambda s: set(s.split("|"))).tolist()
    enc = PhenotypeEncoder(ancestors=ancestors,
                           ontology_aware=encoder_kwargs.get("ontology_aware", cfg["features"]["ontology_aware"]),
                           ic_weighting=encoder_kwargs.get("ic_weighting", cfg["features"]["ic_weighting"]),
                           frequency_weighting=False,
                           min_doc_freq=cfg["features"]["min_term_document_freq"])
    enc.fit([term_sets[i] for i in np.where(tr_mask)[0]], ic=ic)
    X = enc.transform(term_sets).toarray().astype(np.float64)
    y_raw = df[target_col].astype(str).values
    Xtr, ytr_raw = X[tr_mask], y_raw[tr_mask]
    Xte, yte_raw = X[te_mask], y_raw[te_mask]
    labels = sorted(set(ytr_raw))
    # restrict test to classes seen in train (group_holdout may introduce unseen)
    seen = np.isin(yte_raw, labels)
    Xte, yte_raw = Xte[seen], yte_raw[seen]
    if len(labels) < 2 or len(yte_raw) == 0:
        return {"_skipped": f"insufficient classes for {strategy}/{target_col}"}
    # integer-encode labels for estimator compatibility (e.g. XGBoost)
    code = {l: i for i, l in enumerate(labels)}
    ytr = np.array([code[v] for v in ytr_raw])
    yte = np.array([code[v] for v in yte_raw])
    # cluster id for each TEST item = its source disease (independent statistical unit)
    clusters_te = df.loc[te_mask, "ORPHA_ID"].values[seen]

    reg = available_models(cfg)
    use = [m for m in (cfg["models"]["classical"] + cfg["models"]["boosting"] + cfg["models"]["deep"])
           if m in reg]
    results, preds_store = {}, {}
    n_inner = cfg["splits"]["n_inner_folds"]
    for name in use:
        factory, grid = reg[name]
        est = factory()
        try:
            if tune and grid:
                inner = StratifiedKFold(n_splits=min(n_inner, _min_class(ytr)), shuffle=True, random_state=seed)
                gs = GridSearchCV(est, {k: v for k, v in grid.items()},
                                  scoring="f1_macro", cv=inner, n_jobs=-1, error_score="raise")
                gs.fit(Xtr, ytr); base = gs.best_estimator_
            else:
                base = est.fit(Xtr, ytr)
            # calibration on train via internal CV
            try:
                cal = CalibratedClassifierCV(base, method=cfg["calibration"]["method"],
                                             cv=min(3, _min_class(ytr)))
                cal.fit(Xtr, ytr); model = cal
            except Exception:
                model = base
            proba = model.predict_proba(Xte)
            cls_order = list(model.classes_)
            ypred = np.array(cls_order)[proba.argmax(1)]
            m = classification_metrics(yte, ypred, proba, cls_order)
            m.update(calibration_metrics(yte, ypred, proba, cls_order))
            m["abstain_rate"] = float((proba.max(1) < cfg["calibration"]["abstain_threshold"]).mean())
            results[name] = m
            preds_store[name] = {"y_true": yte.tolist(), "y_pred": ypred.tolist(),
                                 "clusters": list(map(str, clusters_te))}
        except Exception as e:
            LOG.warning("model %s failed on %s: %s", name, strategy, e)
            results[name] = {"_error": str(e)}
    return {"metrics": results, "preds": preds_store, "n_features": X.shape[1],
            "n_train": int(tr_mask.sum()), "n_test": int(len(yte)), "n_classes": len(labels),
            "n_test_clusters": int(len(set(map(str, clusters_te))))}


def _min_class(y):
    import collections
    c = collections.Counter(y)
    return max(2, min(c.values()))


# ============================================================= T-B ranking
RANK_FEATURES_FULL = ["pheno_gene_overlap", "assoc_present", "clingen_validity_ord",
                      "clinvar_path_fraction", "clinvar_star_path", "HI", "TS"]

ABLATION_FEATURE_SETS = {
    "phenotype_only": ["pheno_gene_overlap"],
    "pheno+gene": ["pheno_gene_overlap", "assoc_present"],
    "pheno+clinvar": ["pheno_gene_overlap", "clinvar_path_fraction", "clinvar_star_path"],
    "pheno+clingen": ["pheno_gene_overlap", "clingen_validity_ord", "HI", "TS"],
    "all_evidence": RANK_FEATURES_FULL,
}


def run_ranking(pairs, cfg, strategy="disease_holdout", feature_cols=None, model_name="lightgbm"):
    seed = cfg["seed"]
    feature_cols = feature_cols or RANK_FEATURES_FULL
    df = pairs.reset_index(drop=True).copy()
    # add grouping cols needed by splitter
    df["causal_gene"] = df["candidate_gene"].where(df["causal"] == 1)
    df["causal_gene"] = df.groupby("profile_id")["causal_gene"].transform("first")
    df["Disease_Group"] = "NA"
    tr_mask, te_mask = make_split(df, strategy, cfg["splits"]["test_size"], seed)
    Xtr, ytr = df.loc[tr_mask, feature_cols].values, df.loc[tr_mask, "causal"].values
    reg = available_models(cfg)
    if model_name not in reg:
        model_name = "random_forest"
    factory, _ = reg[model_name]
    model = factory().fit(Xtr, ytr)
    df_te = df.loc[te_mask].copy()
    if hasattr(model, "predict_proba"):
        df_te["score"] = model.predict_proba(df.loc[te_mask, feature_cols].values)[:, list(model.classes_).index(1)] \
            if 1 in model.classes_ else 0.0
    else:
        df_te["score"] = model.decision_function(df.loc[te_mask, feature_cols].values)
    return _rank_eval(df_te, cfg)


def run_ranking_heuristic(pairs, cfg, strategy, score_col):
    seed = cfg["seed"]
    df = pairs.reset_index(drop=True).copy()
    df["causal_gene"] = df["candidate_gene"].where(df["causal"] == 1)
    df["causal_gene"] = df.groupby("profile_id")["causal_gene"].transform("first")
    df["Disease_Group"] = "NA"
    _, te_mask = make_split(df, strategy, cfg["splits"]["test_size"], seed)
    df_te = df.loc[te_mask].copy()
    df_te["score"] = df_te[score_col]
    return _rank_eval(df_te, cfg)


def run_ranking_precomputed(df_with_score, cfg, strategy):
    """Evaluate ranking when a 'score' column already exists (e.g. external tool output)."""
    seed = cfg["seed"]
    df = df_with_score.reset_index(drop=True).copy()
    if "score" not in df.columns:
        return {"_skipped": "no score column"}
    df["causal_gene"] = df["candidate_gene"].where(df["causal"] == 1)
    df["causal_gene"] = df.groupby("profile_id")["causal_gene"].transform("first")
    df["Disease_Group"] = "NA"
    _, te_mask = make_split(df, strategy, cfg["splits"]["test_size"], seed)
    return _rank_eval(df.loc[te_mask].copy(), cfg)


def _rank_eval(df_te, cfg):
    ks = tuple(cfg["ranking"]["k_values"])
    queries, per_query = [], []
    for pid, g in df_te.groupby("profile_id"):
        if g["causal"].sum() == 0:
            continue  # true gene not in candidate pool for this profile
        scores = g["score"].values.astype(float)
        rel = g["causal"].values.astype(int)
        queries.append((scores, rel))
        order = np.argsort(-scores, kind="stable")
        rsorted = rel[order]
        pos = np.where(rsorted == 1)[0]
        per_query.append({"profile_id": pid,
                          "ORPHA_ID": g["ORPHA_ID"].iloc[0] if "ORPHA_ID" in g.columns else pid,
                          "rr": 1.0 / (pos[0] + 1) if len(pos) else 0.0,
                          "hit@1": float(rsorted[:1].sum() > 0),
                          "hit@10": float(rsorted[:10].sum() > 0)})
    if not queries:
        return {"_skipped": "no evaluable queries"}
    m = ranking_metrics(queries, ks=ks)
    m["n_queries"] = len(queries)
    m["per_query"] = per_query
    return m
