#!/usr/bin/env python3
"""Stage 14 — generate figures from the result frames produced by earlier stages."""
import numpy as np, pandas as pd
from _common import cfg_and_seed, load_processed, P, proc_dir, LOG
from rdtabxai.visualization import figures as F
from rdtabxai.evaluation import harness
from rdtabxai import robustness

def main():
    cfg = cfg_and_seed(); tables = P(cfg, "results_tables"); figs = P(cfg, "results_figures")
    A = load_processed(cfg); made = []

    cb = tables / "classification_baselines.csv"
    if cb.exists():
        df = pd.read_csv(cb)
        sub = df[df["split"] == "disease_holdout"].set_index("model")
        if len(sub):
            made.append(F.fig_performance_bars(sub, figs, "macro_f1",
                        "T-C disease-group (disease holdout)"))

    rr = tables / "ranking_results.csv"
    if rr.exists():
        df = pd.read_csv(rr); df = df[df["split"] == "disease_holdout"]
        rank_res = {r["method"]: r.to_dict() for _, r in df.iterrows()}
        if rank_res:
            made.append(F.fig_topk(rank_res, figs, ks=tuple(cfg["ranking"]["k_values"])))

    ab = tables / "ablation_ranking.csv"
    if ab.exists():
        made.append(F.fig_ablation(pd.read_csv(ab), figs, "mrr"))

    pi = tables / "explain_permutation_importance.csv"
    if pi.exists():
        made.append(F.fig_importance(pd.read_csv(pi), figs, "importance_mean",
                    "F_perm_importance.png", title="Permutation importance (T-B)"))

    # robustness (compute here; also persist table for stage 13 re-runs)
    rob = robustness.robustness_curve(A["profiles"], A["disease_terms"], A["anc"], cfg)
    rob.to_csv(tables / "robustness_curve.csv", index=False)
    made.append(F.fig_robustness(rob, figs, "macro_f1"))

    LOG.info("Stage14 done: %d figures -> %s", len([m for m in made if m]), figs)

if __name__ == "__main__":
    main()
