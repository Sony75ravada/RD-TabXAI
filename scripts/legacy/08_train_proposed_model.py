#!/usr/bin/env python3
"""Stage 08 — proposed RD-TabXAI configuration for T-C: ontology-aware + IC-weighted
encoding + calibrated gradient boosting. Compared against an ablated flat-binary encoder.
NOTE: proposed is NOT forced to win; the table reports whatever the data shows."""
import pandas as pd
from _common import cfg_and_seed, load_processed, P, LOG
from rdtabxai.evaluation import harness

PROPOSED = {"ontology_aware": True, "ic_weighting": True}
ABLATED = {"ontology_aware": False, "ic_weighting": False}

def main():
    cfg = cfg_and_seed(); A = load_processed(cfg)
    rows = []
    for tag, ek in [("proposed_ontology+IC", PROPOSED), ("ablated_flatbinary", ABLATED)]:
        for strat in ["disease_holdout", "gene_holdout"]:
            res = harness.run_classification(A["profiles"], A["disease_terms"], A["anc"], cfg,
                                             target_col="Disease_Group", strategy=strat,
                                             encoder_kwargs=ek)
            if "_skipped" in res: continue
            for model, m in res["metrics"].items():
                if "_error" in m: continue
                rows.append({"config": tag, "split": strat, "model": model, **m})
    df = pd.DataFrame(rows)
    out = P(cfg, "results_tables"); df.to_csv(out / "proposed_vs_ablated.csv", index=False)
    LOG.info("Stage08 done: %d rows -> proposed_vs_ablated.csv", len(df))

if __name__ == "__main__":
    main()
