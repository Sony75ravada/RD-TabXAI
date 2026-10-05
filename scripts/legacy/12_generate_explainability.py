#!/usr/bin/env python3
"""Stage 12 — explainability for the T-B ranker: permutation importance + SHAP global,
and per-case local explanations for a few test profiles."""
import json, numpy as np, pandas as pd
from _common import cfg_and_seed, load_processed, P, LOG
from rdtabxai.evaluation import harness
from rdtabxai.evaluation.splits import make_split
from rdtabxai.explainability import explain
from rdtabxai.models.registry import available_models

def main():
    cfg = cfg_and_seed(); A = load_processed(cfg); pairs = A["pairs"].reset_index(drop=True)
    feats = harness.RANK_FEATURES_FULL
    df = pairs.copy()
    df["causal_gene"] = df["candidate_gene"].where(df["causal"] == 1)
    df["causal_gene"] = df.groupby("profile_id")["causal_gene"].transform("first")
    df["Disease_Group"] = "NA"
    tr, te = make_split(df, "disease_holdout", cfg["splits"]["test_size"], cfg["seed"])
    Xtr, ytr = df.loc[tr, feats].values, df.loc[tr, "causal"].values
    model = available_models(cfg)["lightgbm"][0]().fit(Xtr, ytr)
    Xte, yte = df.loc[te, feats].values, df.loc[te, "causal"].values
    out = P(cfg, "results_tables")
    perm = explain.permutation_importance_df(model, Xte, yte, feats,
                                             n_repeats=10, seed=cfg["seed"], scoring="roc_auc")
    perm.to_csv(out / "explain_permutation_importance.csv", index=False)
    sh = explain.shap_global(model, Xte, feats, seed=cfg["seed"])
    if sh is not None:
        sh.to_csv(out / "explain_shap_global.csv", index=False)
    cases = [explain.local_explanation(model, Xte[i], feats) for i in range(min(5, len(Xte)))]
    (out / "explain_local_cases.json").write_text(json.dumps(cases, indent=2))
    LOG.info("Stage12 done: permutation + SHAP + %d local cases", len(cases))

if __name__ == "__main__":
    main()
