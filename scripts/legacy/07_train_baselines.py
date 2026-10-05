#!/usr/bin/env python3
"""Stage 07 — T-C disease-group classification baselines across all leakage-safe splits."""
import pandas as pd
from _common import cfg_and_seed, load_processed, P, LOG
from rdtabxai.evaluation import harness

def main():
    cfg = cfg_and_seed(); A = load_processed(cfg)
    # group_holdout is INCOHERENT for group classification (cannot predict an unseen
    # group), so it is excluded by design. 'random' is reported only as a leakage probe.
    TC_SPLITS = {"random": "leakage_probe (optimistic upper bound)",
                 "disease_holdout": "HEADLINE (no disease shared train/test)",
                 "gene_holdout": "generalization (unseen causal genes)"}
    rows = []
    for strat, role in TC_SPLITS.items():
        res = harness.run_classification(A["profiles"], A["disease_terms"], A["anc"], cfg,
                                         target_col="Disease_Group", strategy=strat)
        if "_skipped" in res:
            LOG.info("Stage07 %s skipped: %s", strat, res["_skipped"]); continue
        for model, m in res["metrics"].items():
            if "_error" in m: continue
            rows.append({"task": "T-C", "split": strat, "split_role": role, "model": model,
                         "n_train": res["n_train"], "n_test": res["n_test"], **m})
    df = pd.DataFrame(rows)
    out = P(cfg, "results_tables"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "classification_baselines.csv", index=False)
    LOG.info("Stage07 done: %d rows -> classification_baselines.csv", len(df))

if __name__ == "__main__":
    main()
