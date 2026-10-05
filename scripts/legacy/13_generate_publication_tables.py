#!/usr/bin/env python3
"""Stage 13 — assemble publication tables into results/tables and a tables.md index.
Tables are built only from result CSVs produced by earlier stages (no fabrication)."""
import pandas as pd
from _common import cfg_and_seed, load_processed, P, proc_dir, LOG

def _load(out, name):
    p = out / name
    return pd.read_csv(p) if p.exists() else None

def main():
    cfg = cfg_and_seed(); out = P(cfg, "results_tables"); pdir = proc_dir(cfg)
    A = load_processed(cfg)
    md = ["# RD-TabXAI — Publication Tables", "",
          "_All numbers below come from running the pipeline; on fixtures they are a "
          "demonstration only, not scientific results._", ""]

    # Table 1 — dataset summary
    prof, pairs, meta = A["profiles"], A["pairs"], A["meta"]
    t1 = pd.DataFrame([
        {"item": "diseases (ORPHA)", "count": meta["ORPHA_ID"].nunique()},
        {"item": "disease groups", "count": meta["Disease_Group"].nunique()},
        {"item": "simulated profiles", "count": len(prof)},
        {"item": "candidate genes (evidence table)", "count": pairs["candidate_gene"].nunique()},
        {"item": "ranking pairs", "count": len(pairs)},
        {"item": "ranking positive rate", "count": round(pairs["causal"].mean(), 4)},
    ])
    t1.to_csv(out / "table1_dataset_summary.csv", index=False)
    md += ["## Table 1 — Dataset summary", t1.to_markdown(index=False), ""]

    # Table 2 — join audit (data quality)
    ja = _load(pdir, "join_audit.csv")
    if ja is not None:
        md += ["## Table 2 — Join audit (unmatched rates)", ja.to_markdown(index=False), ""]

    # Table 3 — classification baselines
    cb = _load(out, "classification_baselines.csv")
    if cb is not None:
        keep = ["split", "model", "macro_f1", "balanced_accuracy", "mcc", "auroc_ovr", "ece", "brier"]
        cb3 = cb[[c for c in keep if c in cb.columns]].round(3)
        cb3.to_csv(out / "table3_classification.csv", index=False)
        md += ["## Table 3 — T-C disease-group classification", cb3.to_markdown(index=False), ""]

    # Table 4 — proposed vs ablated encoder
    pa = _load(out, "proposed_vs_ablated.csv")
    if pa is not None:
        keep = ["config", "split", "model", "macro_f1", "balanced_accuracy", "ece"]
        pa4 = pa[[c for c in keep if c in pa.columns]].round(3)
        md += ["## Table 4 — Proposed (ontology+IC) vs flat-binary ablation", pa4.to_markdown(index=False), ""]

    # Table 5 — ranking results (with method type + clustered MRR CI)
    rr = _load(out, "ranking_results.csv")
    if rr is not None:
        keep = ["method", "method_type", "split", "mrr", "hit@1", "hit@5", "hit@10", "n_queries"]
        rr5 = rr[[c for c in keep if c in rr.columns]].copy()
        mrrci = _load(out, "stats_ranking_mrr_ci.csv")
        if mrrci is not None:
            mrrci = mrrci.rename(columns={"ci_low": "mrr_ci_low", "ci_high": "mrr_ci_high"})
            rr5 = rr5.merge(mrrci[["method", "split", "mrr_ci_low", "mrr_ci_high"]],
                            on=["method", "split"], how="left")
        rr5 = rr5.round(3).sort_values(["split", "mrr"], ascending=[True, False])
        rr5.to_csv(out / "table5_ranking.csv", index=False)
        md += ["## Table 5 — T-B causal-gene ranking (clustered 95% CI on MRR)",
               rr5.to_markdown(index=False), ""]

    # Table 6 — ablation
    ab = _load(out, "ablation_ranking.csv")
    if ab is not None:
        keep = ["feature_set", "n_features", "mrr", "hit@1", "hit@5", "hit@10"]
        ab6 = ab[[c for c in keep if c in ab.columns]].round(3)
        md += ["## Table 6 — Evidence ablation (T-B)", ab6.to_markdown(index=False), ""]

    # Table 7 — CLUSTERED bootstrap CIs
    bs = _load(out, "stats_bootstrap_ci.csv")
    if bs is not None:
        md += ["## Table 7 — Clustered bootstrap 95% CI on macro-F1 (resampled by disease)",
               bs.round(3).to_markdown(index=False), ""]

    # Table 8 — cluster-level Wilcoxon vs best (HEADLINE significance test)
    cw = _load(out, "stats_cluster_wilcoxon_vs_best.csv")
    if cw is not None:
        md += ["## Table 8 — Cluster-level Wilcoxon vs best model (Holm-corrected; HEADLINE)",
               cw.round(4).to_markdown(index=False), ""]

    # Friedman / Nemenyi omnibus
    import json as _json
    fn = out / "stats_friedman_nemenyi.json"
    if fn.exists():
        fr = _json.loads(fn.read_text())
        if "friedman_p" in fr:
            ranks = pd.DataFrame(sorted(fr["mean_rank"].items(), key=lambda x: x[1]),
                                 columns=["model", "mean_rank"])
            md += [f"## Table 8b — Friedman omnibus (disease-blocked): chi2={fr['friedman_stat']:.2f}, "
                   f"p={fr['friedman_p']:.4f}, blocks={fr['n_blocks']}, Nemenyi CD={fr['nemenyi_cd']:.3f}",
                   ranks.round(3).to_markdown(index=False), ""]

    # Table 8c — item-level McNemar (DIAGNOSTIC only)
    mc = _load(out, "stats_mcnemar_diagnostic.csv")
    if mc is not None:
        md += ["## Table 8c — Item-level McNemar vs best (DIAGNOSTIC; anticonservative under clustering)",
               mc.round(4).to_markdown(index=False), ""]

    # Table 9 — permutation importance
    pi = _load(out, "explain_permutation_importance.csv")
    if pi is not None:
        md += ["## Table 9 — Permutation importance (T-B ranker)", pi.round(4).to_markdown(index=False), ""]

    # Table 10 — robustness
    rob = _load(out, "robustness_curve.csv")
    if rob is not None:
        md += ["## Table 10 — Robustness to missing/noisy phenotypes", rob.round(3).to_markdown(index=False), ""]

    (out / "tables.md").write_text("\n".join(md))
    LOG.info("Stage13 done: tables.md + table*.csv in %s", out)

if __name__ == "__main__":
    main()
