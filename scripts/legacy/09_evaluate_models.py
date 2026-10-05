#!/usr/bin/env python3
"""Stage 09 — T-B causal-gene ranking across leakage-safe splits.

Compares, on the SAME holdout splits:
  - learned tabular ranker (LightGBM over fused evidence)
  - single-evidence heuristics (phenotype overlap, ClinVar, ClinGen)
  - reimplemented literature baselines (Phrank-style, Resnik-BMA, Lin-BMA)
  - [optional] real external tools (Exomiser/GADO/Phen2Gene) if their output files
    are present in data_raw/external/ — ingested via the adapter, never fabricated.

Per-query reciprocal ranks (with source-disease cluster ids) are saved for the
cluster-aware confidence intervals computed in Stage 11.
"""
import json
import pandas as pd
from pathlib import Path
from _common import cfg_and_seed, load_processed, P, raw_dir, LOG
from rdtabxai.evaluation import harness
from rdtabxai.baselines.external import load_external_ranking

SIM_COLS = ["sim_phrank_style", "sim_resnik_bma", "sim_lin_bma"]
HEUR_COLS = ["pheno_gene_overlap", "clinvar_path_fraction", "clingen_validity_ord"]

def _strip(res):
    pq = res.pop("per_query", None)
    return res, pq

def main():
    cfg = cfg_and_seed(); A = load_processed(cfg); pairs = A["pairs"]
    out = P(cfg, "results_tables"); out.mkdir(parents=True, exist_ok=True)
    rows, per_query_store = [], {}
    for strat in ["disease_holdout", "gene_holdout"]:
        learned = harness.run_ranking(pairs, cfg, strategy=strat, model_name="lightgbm")
        if "_skipped" not in learned:
            m, pq = _strip(learned)
            rows.append({"method": "learned_lightgbm", "method_type": "proposed", "split": strat, **m})
            per_query_store[f"learned_lightgbm|{strat}"] = pq
        for col in HEUR_COLS:
            if col not in pairs.columns: continue
            r = harness.run_ranking_heuristic(pairs, cfg, strat, col)
            if "_skipped" in r: continue
            m, pq = _strip(r)
            rows.append({"method": f"heuristic_{col}", "method_type": "single_evidence", "split": strat, **m})
            per_query_store[f"heuristic_{col}|{strat}"] = pq
        for col in SIM_COLS:
            if col not in pairs.columns: continue
            r = harness.run_ranking_heuristic(pairs, cfg, strat, col)
            if "_skipped" in r: continue
            m, pq = _strip(r)
            name = col.replace("sim_", "lit_")
            rows.append({"method": name, "method_type": "literature_baseline",
                         "split": strat, **m})
            per_query_store[f"{name}|{strat}"] = pq

    # optional: real external tool outputs
    ext_dir = raw_dir(cfg) / "external"
    if ext_dir.exists():
        for f in ext_dir.glob("*.tsv"):
            tool = f.stem.split("_")[0]
            try:
                ext = load_external_ranking(f, tool)
                merged = pairs.merge(ext[["profile_id", "candidate_gene", "score"]],
                                     on=["profile_id", "candidate_gene"], how="left")
                merged["score"] = merged["score"].fillna(merged["score"].min() - 1 if merged["score"].notna().any() else 0)
                for strat in ["disease_holdout", "gene_holdout"]:
                    r = harness.run_ranking_precomputed(merged, cfg, strat)
                    if "_skipped" in r: continue
                    m, pq = _strip(r)
                    rows.append({"method": f"external_{tool}", "method_type": "external_tool",
                                 "split": strat, **m})
            except Exception as e:
                LOG.warning("external tool %s failed: %s", f.name, e)

    pd.DataFrame(rows).to_csv(out / "ranking_results.csv", index=False)
    (out / "ranking_per_query.json").write_text(json.dumps(per_query_store, default=str))
    LOG.info("Stage09 done: %d method-rows -> ranking_results.csv (+ per_query for stats)", len(rows))

if __name__ == "__main__":
    main()
