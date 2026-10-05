#!/usr/bin/env python3
"""Proposed learning-to-rank vs literature baselines on REAL HPO data — fair, leakage-safe.

    PYTHONPATH=src python scripts/run_real_ltr.py

Produces: proposed-vs-baseline ranking table (with clustered 95% CI), ablation over
feature groups, and cluster-level Wilcoxon significance (proposed vs best baseline,
Holm-corrected). Honest scope note: simulated patients from real annotations; variant
evidence (ClinVar/ClinGen) deferred (unreachable here).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdtabxai.io_utils import load_config, set_seed, ensure_dir, LOG
from rdtabxai.data.hpo import ancestor_map
from rdtabxai.evaluation import ltr, stats


def main():
    cfg = load_config(ROOT / "configs/real_hpo.yaml"); set_seed(cfg["seed"])
    pdir = ROOT / cfg["paths"]["data_processed"]
    tdir = ensure_dir(ROOT / cfg["paths"]["results_tables"])
    fdir = ensure_dir(ROOT / cfg["paths"]["results_figures"])

    profiles = pd.read_csv(pdir / "profiles.csv")
    pairs = pd.read_csv(pdir / "ranking_pairs.csv")
    disease_terms = json.loads((pdir / "disease_terms.json").read_text())
    hterms = pd.read_csv(pdir / "hpo_terms.csv")
    anc = ancestor_map(hterms)
    g2p = pd.read_csv(pdir / "genes_to_phenotype.csv")
    gcol = "gene_symbol" if "gene_symbol" in g2p.columns else g2p.columns[1]
    hcol = "hpo_id" if "hpo_id" in g2p.columns else "HPO_ID"
    gene_to_terms = g2p.groupby(gcol)[hcol].apply(set).to_dict()

    summary = {}
    feat_cache = {}
    for strat in ["disease_holdout", "gene_holdout"]:
        LOG.info("=== LTR on REAL data, split=%s ===", strat)
        precomp = ltr.compute_features(pairs, profiles, gene_to_terms, disease_terms, anc,
                                       cfg, strategy=strat)
        feat_cache[strat] = precomp
        res = ltr.run_ltr(pairs, profiles, gene_to_terms, disease_terms, anc, cfg,
                          strategy=strat, precomputed=precomp)
        prop = res["proposed_ltr"]; bl = res["baselines"]

        # table: proposed vs each baseline with clustered MRR CI
        rows, pqs = [], {}
        def add(name, m, mtype):
            pq = m.get("per_query", [])
            pqs[name] = pq
            ci = stats.clustered_bootstrap_ci(np.zeros(len(pq)),
                    np.array([r["rr"] for r in pq]),
                    np.array([str(r["ORPHA_ID"]) for r in pq]),
                    lambda yt, yp: float(np.mean(yp)),
                    n_boot=cfg["stats"]["n_bootstrap"], alpha=cfg["stats"]["alpha"], seed=cfg["seed"])
            rows.append({"method": name, "type": mtype, "split": strat,
                         "mrr": m["mrr"], "hit@1": m["hit@1"], "hit@5": m["hit@5"],
                         "hit@10": m["hit@10"], "n_queries": m["n_queries"],
                         "mrr_ci_low": ci["ci_low"], "mrr_ci_high": ci["ci_high"]})
        add("proposed_LTR", prop, "proposed")
        for b, m in bl.items():
            if "_skipped" not in m:
                add(f"baseline_{b}", m, "literature_baseline")
        tab = pd.DataFrame(rows).sort_values("mrr", ascending=False)
        tab.to_csv(tdir / f"real_ltr_{strat}.csv", index=False)

        # significance: proposed vs BEST baseline (cluster-level Wilcoxon on per-disease RR)
        base_only = [r for r in rows if r["type"] != "proposed"]
        best_base = max(base_only, key=lambda r: r["mrr"])["method"] if base_only else None
        sig = {}
        if best_base:
            def per_disease_rr(pq):
                d = pd.DataFrame(pq)
                return d.groupby("ORPHA_ID")["rr"].mean().to_dict()
            a = per_disease_rr(pqs["proposed_LTR"]); b = per_disease_rr(pqs[best_base])
            sig = stats.cluster_wilcoxon(a, b)
            sig["proposed"] = "proposed_LTR"; sig["vs_best_baseline"] = best_base
        (tdir / f"real_ltr_significance_{strat}.json").write_text(json.dumps(sig, indent=2, default=str))
        summary[strat] = {"table": tab.round(4).to_dict("records"),
                          "best_baseline": best_base, "significance": sig,
                          "feature_importance": prop.get("feature_importance", {})}

    # ablation (disease_holdout): cumulative feature groups — reuse cached features
    LOG.info("=== ablation (feature groups) ===")
    abl_rows = []
    cumulative = []
    dh_precomp = feat_cache["disease_holdout"]
    for gname, gfeats in ltr.FEATURE_GROUPS.items():
        cumulative += gfeats
        r = ltr.run_ltr(pairs, profiles, gene_to_terms, disease_terms, anc, cfg,
                        strategy="disease_holdout", feature_set=list(cumulative),
                        precomputed=dh_precomp)
        m = r["proposed_ltr"]
        abl_rows.append({"groups_included": "+".join([k for k in ltr.FEATURE_GROUPS
                                                      if all(f in cumulative for f in ltr.FEATURE_GROUPS[k])]),
                         "n_features": len(cumulative), "mrr": m["mrr"],
                         "hit@1": m["hit@1"], "hit@10": m["hit@10"]})
    pd.DataFrame(abl_rows).to_csv(tdir / "real_ltr_ablation.csv", index=False)

    (tdir / "real_ltr_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- print honest summary
    print("\n=========== PROPOSED LTR vs BASELINES — REAL HPO DATA ===========")
    for strat in ["disease_holdout", "gene_holdout"]:
        print(f"\n[{strat}]")
        t = pd.DataFrame(summary[strat]["table"])
        print(t[["method", "mrr", "mrr_ci_low", "mrr_ci_high", "hit@1", "hit@10", "n_queries"]].to_string(index=False))
        s = summary[strat]["significance"]
        if s:
            print(f"  proposed vs best baseline ({s.get('vs_best_baseline')}): "
                  f"Wilcoxon p={s.get('p_value'):.4f}, mean ΔMRR(disease)={s.get('mean_diff'):.4f}, "
                  f"n_diseases={s.get('n_clusters')}")
    print("\n[ablation — cumulative feature groups, disease_holdout]")
    print(pd.read_csv(tdir / "real_ltr_ablation.csv").to_string(index=False))
    print("\nCAVEAT: simulated patients from real annotations; candidate scoring shares the")
    print("annotation source -> benchmark with an annotation-recovery ceiling, NOT clinical")
    print("validation. ClinVar/ClinGen independent evidence deferred (unreachable here).")


if __name__ == "__main__":
    main()
