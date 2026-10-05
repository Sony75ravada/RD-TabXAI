"""Learning-to-rank for phenotype -> causal-gene, done fairly and leakage-safely.

Why the previous learned ranker lost (diagnosed, not assumed):
  - 5/7 of its features were constant in this data regime (assoc_present, ClinGen,
    ClinVar, dosage), so it effectively trained on a single feature (Jaccard overlap);
  - the strongest phenotype signal (IC-weighted semantic similarity / Phrank) was used
    by the baselines but NOT given to the model.

Fix (legitimate feature engineering + proper LTR objective + fair protocol):
  - engineer a rich, ontology-aware feature set computed with TRAIN-ONLY information
    content (no leakage): Jaccard, binary cosine, Phrank-style shared-IC, Resnik-BMA,
    Lin-BMA, plus set-size / shared-IC summary stats;
  - train LightGBM LambdaRank with per-profile query groups;
  - compare against EACH single-similarity baseline computed on the SAME test split with
    the SAME train-only IC (apples-to-apples);
  - ablate feature groups to show each component's contribution.

All similarity baselines and model features share one IC dictionary fit on the training
diseases, so the comparison is fair and free of train/test leakage.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from functools import lru_cache
from ..io_utils import LOG
from ..data.hpo import information_content
from .splits import make_split
from .metrics import ranking_metrics

FEATURE_GROUPS = {
    "overlap":   ["jaccard", "cosine_bin", "n_shared", "n_profile", "n_gene"],
    "ic_shared": ["phrank", "sum_ic_shared", "max_ic_shared", "mean_ic_profile"],
    "bma":       ["resnik_bma", "lin_bma"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _build_ic_and_cache(disease_terms, train_orpha, ancestors):
    dh = pd.DataFrame([(o, t) for o in train_orpha for t in disease_terms.get(o, [])],
                      columns=["ORPHA_ID", "HPO_ID"])
    ic = information_content(dh, ancestors)

    cache = {}

    def mica(t1, t2):
        key = (t1, t2) if t1 <= t2 else (t2, t1)
        v = cache.get(key)
        if v is None:
            common = ancestors.get(t1, {t1}) & ancestors.get(t2, {t2})
            v = max((ic.get(c, 0.0) for c in common), default=0.0)
            cache[key] = v
        return v
    return ic, mica


def _fmean(xs):
    """Order-independent mean (exactly rounded sum)."""
    return math.fsum(xs) / len(xs) if xs else 0.0


def _pair_features(P, G, ic, mica):
    P = set(P); G = set(G)
    shared = P & G
    union = P | G
    # math.fsum is exactly rounded, so these sums do not depend on set iteration order
    sum_ic_shared = math.fsum(ic.get(t, 0.0) for t in shared)
    max_ic_shared = max((ic.get(t, 0.0) for t in shared), default=0.0)
    mean_ic_profile = (math.fsum(ic.get(t, 0.0) for t in P) / len(P)) if P else 0.0
    jac = len(shared) / len(union) if union else 0.0
    cos = len(shared) / np.sqrt(len(P) * len(G)) if P and G else 0.0
    # Phrank-style = sum IC over shared ancestral closure
    pc = set().union(*[ic_ancestors(t) for t in P]) if P else set()
    gc = set().union(*[ic_ancestors(t) for t in G]) if G else set()
    phrank = math.fsum(ic.get(t, 0.0) for t in (pc & gc))
    # BMA scores (bounded loops; gene term sets are capped upstream)
    if P and G:
        resnik = (_fmean([max((mica(x, y) for y in G), default=0.0) for x in P]) +
                  _fmean([max((mica(y, x) for x in P), default=0.0) for y in G])) / 2
        def lin(x, y):
            m = mica(x, y); d = ic.get(x, 0.0) + ic.get(y, 0.0)
            return (2 * m / d) if d > 0 else 0.0
        linbma = (_fmean([max((lin(x, y) for y in G), default=0.0) for x in P]) +
                  _fmean([max((lin(y, x) for x in P), default=0.0) for y in G])) / 2
    else:
        resnik = linbma = 0.0
    return {"jaccard": jac, "cosine_bin": cos, "n_shared": len(shared),
            "n_profile": len(P), "n_gene": len(G), "phrank": phrank,
            "sum_ic_shared": sum_ic_shared, "max_ic_shared": max_ic_shared,
            "mean_ic_profile": mean_ic_profile, "resnik_bma": resnik, "lin_bma": linbma}


# ancestor closure cache is set per-run (module global, reset each call)
_ANC = {}
def ic_ancestors(t):
    return _ANC.get(t, {t})


def compute_features(pairs, profiles, gene_to_terms, disease_terms, ancestors, cfg,
                     strategy="disease_holdout", gene_term_cap=60,
                     gene_disease_terms=None, leave_target_out=False):
    """Compute leakage-safe rich features ONCE for a split (train-only IC). Returns
    (df_with_features, tr_mask, te_mask) so model fitting / ablation can reuse them.

    If leave_target_out=True and gene_disease_terms is given, each candidate gene's
    phenotype profile is built from its OTHER associated diseases (excluding the profile's
    source disease), which removes annotation-recovery circularity for the true gene."""
    global _ANC
    seed = cfg["seed"]
    df = pairs.reset_index(drop=True).copy()
    df["causal_gene"] = df["candidate_gene"].where(df["causal"] == 1)
    df["causal_gene"] = df.groupby("profile_id")["causal_gene"].transform("first")
    df["Disease_Group"] = "NA"
    tr_mask, te_mask = make_split(df, strategy, cfg["splits"]["test_size"], seed)

    train_orpha = set(df.loc[tr_mask, "ORPHA_ID"])
    ic, mica = _build_ic_and_cache(disease_terms, train_orpha, ancestors)
    _ANC = {t: ancestors.get(t, {t}) for o in disease_terms for t in disease_terms[o]}

    pterms = profiles.set_index("profile_id")["terms"].apply(lambda s: set(s.split("|"))).to_dict()

    def cap(ts):
        ts = list(ts)
        if len(ts) > gene_term_cap:
            # explicit tie-break on the term ID: the kept terms no longer depend on
            # Python's per-process set iteration order
            ts = sorted(ts, key=lambda t: (-ic.get(t, 0.0), t))[:gene_term_cap]
        return set(ts)

    gterms_full = {g: cap(ts) for g, ts in gene_to_terms.items()}

    def gene_terms_for(g, src_disease):
        if leave_target_out and gene_disease_terms is not None:
            dd = gene_disease_terms.get(g, {})
            acc = set()
            for d, ts in dd.items():
                if str(d) == str(src_disease):
                    continue  # exclude the target disease's own annotations
                acc |= ts
            return cap(acc)
        return gterms_full.get(g, set())

    rows = [_pair_features(pterms.get(r.profile_id, set()),
                           gene_terms_for(r.candidate_gene, r.ORPHA_ID), ic, mica)
            for r in df.itertuples()]
    F = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, F], axis=1)
    return df, tr_mask, te_mask


def run_ltr(pairs, profiles, gene_to_terms, disease_terms, ancestors, cfg,
            strategy="disease_holdout", feature_set=None, gene_term_cap=60,
            precomputed=None):
    """Train LambdaRank with leakage-safe rich features; return proposed + baseline metrics.
    Pass `precomputed=(df, tr_mask, te_mask)` to reuse features across ablation runs."""
    seed = cfg["seed"]
    feats = feature_set or ALL_FEATURES
    if precomputed is not None:
        df, tr_mask, te_mask = precomputed
    else:
        df, tr_mask, te_mask = compute_features(pairs, profiles, gene_to_terms,
                                                disease_terms, ancestors, cfg, strategy, gene_term_cap)
    proposed = _fit_lambdarank(df, tr_mask, te_mask, feats, seed)
    baselines = {}
    for b in ["phrank", "jaccard", "resnik_bma", "lin_bma"]:
        baselines[b] = _rank_by_column(df.loc[te_mask], b, cfg)
    return {"proposed_ltr": proposed, "baselines": baselines,
            "n_train_pairs": int(tr_mask.sum()), "n_test_pairs": int(te_mask.sum())}


def _fit_lambdarank(df, tr_mask, te_mask, feats, seed):
    import lightgbm as lgb
    tr = df.loc[tr_mask].sort_values("profile_id")
    te = df.loc[te_mask].sort_values("profile_id")
    grp_tr = tr.groupby("profile_id").size().values
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, num_leaves=31,
                           learning_rate=0.05, min_child_samples=20, subsample=0.9,
                           colsample_bytree=0.9, random_state=seed, verbose=-1,
                           deterministic=True, force_row_wise=True, num_threads=1)
    model.fit(tr[feats].values, tr["causal"].values, group=grp_tr)
    te = te.copy()
    te["score"] = model.predict(te[feats].values)
    imp = dict(zip(feats, model.feature_importances_))
    out = _rank_eval_df(te)
    out["feature_importance"] = imp
    return out


def _rank_by_column(te, col, cfg):
    te = te.copy(); te["score"] = te[col]
    return _rank_eval_df(te)


def _rank_eval_df(te):
    ks = (1, 3, 5, 10)
    queries, per_q = [], []
    for pid, g in te.groupby("profile_id"):
        if g["causal"].sum() == 0:
            continue
        s = g["score"].values.astype(float); rel = g["causal"].values.astype(int)
        queries.append((s, rel))
        order = np.argsort(-s, kind="stable"); rs = rel[order]; pos = np.where(rs == 1)[0]
        per_q.append({"profile_id": pid, "ORPHA_ID": g["ORPHA_ID"].iloc[0],
                      "rr": 1.0 / (pos[0] + 1) if len(pos) else 0.0,
                      "hit1": float(rs[:1].sum() > 0), "hit10": float(rs[:10].sum() > 0)})
    if not queries:
        return {"_skipped": "no evaluable queries"}
    m = ranking_metrics(queries, ks=ks)
    m["n_queries"] = len(queries)
    m["per_query"] = per_q
    return m
