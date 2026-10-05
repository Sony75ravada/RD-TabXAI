#!/usr/bin/env python3
"""Stage 11 — CLUSTER-AWARE statistics (independent unit = source disease).

T-C classification:
  - clustered bootstrap 95% CI on macro-F1 per model (resample diseases)
  - disease-blocked Friedman omnibus + Nemenyi critical difference
  - cluster-level paired Wilcoxon vs the top model, Holm-corrected (HEADLINE test)
  - item-level McNemar vs top model (secondary DIAGNOSTIC; anticonservative)
T-B ranking:
  - clustered bootstrap 95% CI on MRR per method (resample diseases over per-query RRs)
"""
import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from _common import cfg_and_seed, load_processed, P, LOG
from rdtabxai.evaluation import harness, stats

def main():
    cfg = cfg_and_seed(); A = load_processed(cfg); out = P(cfg, "results_tables")
    res = harness.run_classification(A["profiles"], A["disease_terms"], A["anc"], cfg,
                                     target_col="Disease_Group", strategy="disease_holdout")
    preds = {m: p for m, p in res.get("preds", {}).items()}
    models = list(preds)
    macro_f1 = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)

    # (1) clustered bootstrap CI per model
    boot_rows = []
    for m in models:
        ci = stats.clustered_bootstrap_ci(preds[m]["y_true"], preds[m]["y_pred"],
                                          preds[m]["clusters"], macro_f1,
                                          n_boot=cfg["stats"]["n_bootstrap"],
                                          alpha=cfg["stats"]["alpha"], seed=cfg["seed"])
        boot_rows.append({"model": m, **ci})
    boot = pd.DataFrame(boot_rows).sort_values("estimate", ascending=False)
    boot.to_csv(out / "stats_bootstrap_ci.csv", index=False)
    best = boot.iloc[0]["model"]

    # (2) disease-blocked Friedman + Nemenyi
    M, blocks = stats.block_score_matrix(preds, models)
    fr = stats.friedman_nemenyi(M, models)
    (out / "stats_friedman_nemenyi.json").write_text(json.dumps(fr, indent=2))

    # (3) cluster-level Wilcoxon vs best (HEADLINE), Holm-corrected
    acc = {m: stats.per_cluster_accuracy(preds[m]["y_true"], preds[m]["y_pred"],
                                         preds[m]["clusters"]) for m in models}
    wil_rows, pvals = [], []
    for m in models:
        if m == best: continue
        w = stats.cluster_wilcoxon(acc[best], acc[m])
        wil_rows.append({"vs_best": best, "model": m, "test": "cluster_wilcoxon", **w})
        pvals.append(w["p_value"])
    adj, rej = stats.correct_pvalues(pvals, cfg["stats"]["correction"])
    for i, r in enumerate(wil_rows):
        r["p_adj_holm"] = adj[i]; r["significant"] = bool(rej[i])
    pd.DataFrame(wil_rows).to_csv(out / "stats_cluster_wilcoxon_vs_best.csv", index=False)

    # (4) item-level McNemar vs best (DIAGNOSTIC only)
    cb = (np.array(preds[best]["y_true"]) == np.array(preds[best]["y_pred"]))
    mc_rows = []
    for m in models:
        if m == best: continue
        cm = (np.array(preds[m]["y_true"]) == np.array(preds[m]["y_pred"]))
        mc_rows.append({"vs_best": best, "model": m, "test": "mcnemar_item_DIAGNOSTIC",
                        **stats.mcnemar_test(cb, cm)})
    pd.DataFrame(mc_rows).to_csv(out / "stats_mcnemar_diagnostic.csv", index=False)

    # (5) clustered bootstrap CI on ranking MRR (resample diseases over per-query RRs)
    pq_path = out / "ranking_per_query.json"
    if pq_path.exists():
        pq = json.loads(pq_path.read_text())
        rrows = []
        for key, recs in pq.items():
            if not recs: continue
            df = pd.DataFrame(recs)
            # treat rr as the "prediction"; metric = mean rr; cluster = ORPHA_ID
            mean_rr = lambda yt, yp: float(np.mean(yp))
            ci = stats.clustered_bootstrap_ci(np.zeros(len(df)), df["rr"].values,
                                              df["ORPHA_ID"].astype(str).values, mean_rr,
                                              n_boot=cfg["stats"]["n_bootstrap"],
                                              alpha=cfg["stats"]["alpha"], seed=cfg["seed"])
            method, split = key.split("|")
            rrows.append({"method": method, "split": split, "metric": "MRR", **ci})
        pd.DataFrame(rrows).to_csv(out / "stats_ranking_mrr_ci.csv", index=False)

    LOG.info("Stage11 done: clustered CIs + Friedman/Nemenyi + cluster Wilcoxon (best=%s, %d disease blocks)",
             best, len(blocks))

if __name__ == "__main__":
    main()
