#!/usr/bin/env python3
"""Run the RD-TabXAI phenotype->causal-gene ranking task on REAL HPO data.

    PYTHONPATH=src python scripts/run_real_hpo.py

Uses configs/real_hpo.yaml. Real ontology + disease-phenotype + gene-phenotype +
disease-gene from the genuine HPO release (via pyhpo or files in data_raw/).
ClinVar/ClinGen variant evidence and Orphadata groups are deferred (not reachable
here) and reported as such — this run does NOT fabricate them.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdtabxai.io_utils import load_config, set_seed, ensure_dir, LOG
from rdtabxai.data import hpo_real
from rdtabxai.features import build_matrices
from rdtabxai.evaluation import harness, stats
from rdtabxai.baselines.external import SIMILARITY_METHODS


def main():
    cfg = load_config(ROOT / "configs/real_hpo.yaml"); set_seed(cfg["seed"])
    pdir = ensure_dir(ROOT / cfg["paths"]["data_processed"])
    tdir = ensure_dir(ROOT / cfg["paths"]["results_tables"])

    # 1) ingest REAL HPO into intermediate tables
    info = hpo_real.ingest(cfg, pdir)

    # 2) build matrices (real) — reuse the exact same builder as the rest of the project
    inter = {k: pd.read_csv(pdir / f"{k}.csv") if (pdir / f"{k}.csv").stat().st_size > 1 else pd.DataFrame()
             for k in ["disease_hpo", "disease_gene", "classification", "hpo_terms",
                       "genes_to_phenotype", "clinvar_gene", "clingen_gene"]}
    inter["genes_to_pheno"] = inter.pop("genes_to_phenotype")
    # ensure empty evidence frames have the right columns for the builder
    if inter["clinvar_gene"].empty:
        inter["clinvar_gene"] = pd.DataFrame(columns=["Gene_Symbol", "n_variants", "n_path_lp",
            "path_fraction", "star_weighted_path", "max_review_stars", "mean_submitters"])
    if inter["clingen_gene"].empty:
        inter["clingen_gene"] = pd.DataFrame(columns=["Gene_Symbol", "clingen_validity_ord",
            "Haploinsufficiency_Score", "Triplosensitivity_Score"])
    if inter["classification"].empty:
        inter["classification"] = pd.DataFrame(columns=["ORPHA_ID", "Disease_Group"])
    built = build_matrices.build(inter, cfg, cfg["seed"], pdir)
    pairs = built["pairs"]

    # 3) T-B causal-gene ranking on REAL data: learned ranker vs literature baselines
    #    (phenotype features only — variant evidence deferred), leakage-safe splits.
    rows, pq_store = [], {}
    sim_cols = [f"sim_{m}" for m in cfg["ranking"].get("similarity_methods", []) if f"sim_{m}" in pairs.columns]
    heur_cols = [c for c in ["pheno_gene_overlap"] if c in pairs.columns]
    for strat in ["disease_holdout", "gene_holdout"]:
        learned = harness.run_ranking(pairs, cfg, strategy=strat, model_name="lightgbm")
        if "_skipped" not in learned:
            pq = learned.pop("per_query", None)
            rows.append({"method": "learned_lightgbm", "method_type": "proposed", "split": strat, **learned})
            pq_store[f"learned_lightgbm|{strat}"] = pq
        for col in heur_cols + sim_cols:
            r = harness.run_ranking_heuristic(pairs, cfg, strat, col)
            if "_skipped" in r: continue
            pq = r.pop("per_query", None)
            name = ("lit_" + col[4:]) if col.startswith("sim_") else f"heuristic_{col}"
            mtype = "literature_baseline" if col.startswith("sim_") else "single_evidence"
            rows.append({"method": name, "method_type": mtype, "split": strat, **r})
            pq_store[f"{name}|{strat}"] = pq
    res = pd.DataFrame(rows)
    res.to_csv(tdir / "real_ranking_results.csv", index=False)

    # 4) clustered (by disease) bootstrap CI on MRR
    mean_rr = lambda yt, yp: float(np.mean(yp))
    ci_rows = []
    for key, recs in pq_store.items():
        if not recs: continue
        df = pd.DataFrame(recs)
        ci = stats.clustered_bootstrap_ci(np.zeros(len(df)), df["rr"].values,
                                          df["ORPHA_ID"].astype(str).values, mean_rr,
                                          n_boot=cfg["stats"]["n_bootstrap"],
                                          alpha=cfg["stats"]["alpha"], seed=cfg["seed"])
        method, split = key.split("|")
        ci_rows.append({"method": method, "split": split, "metric": "MRR", **ci})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(tdir / "real_ranking_mrr_ci.csv", index=False)

    # 5) honest summary
    summary = {"task": "phenotype->causal-gene ranking (T-B) on REAL HPO data",
               "data": info,
               "deferred_unavailable_here": ["ClinVar variant evidence",
                   "ClinGen gene-disease validity / dosage", "Orphadata ORDO disease groups"],
               "caveat": "patients are SIMULATED from real disease annotations; this is a "
                         "benchmark on real data, not a real-patient clinical evaluation"}
    (tdir / "real_run_summary.json").write_text(json.dumps(summary, indent=2))

    LOG.info("REAL RUN COMPLETE")
    print("\n================ REAL-DATA RESULTS (phenotype -> causal gene) ================")
    print(f"HPO release {info['hpo_release']} | cohort {info['cohort_diseases_used']} single-gene "
          f"OMIM diseases (>= {info['min_terms']} HPO terms) | {info['n_disease_gene_edges']} disease-gene links")
    show = res.merge(ci_df.rename(columns={"ci_low": "mrr_lo", "ci_high": "mrr_hi"}),
                     on=["method", "split"], how="left")
    cols = ["method", "method_type", "split", "mrr", "hit@1", "hit@5", "hit@10", "n_queries", "mrr_lo", "mrr_hi"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].round(3).to_string(index=False))
    print("\nDeferred (not downloadable in this environment, hence not used):")
    print("  ClinVar, ClinGen, Orphadata ORDO groups — see real_run_summary.json")
    print("Caveat: simulated patients from real annotations -> benchmark, not clinical validation.")


if __name__ == "__main__":
    main()
