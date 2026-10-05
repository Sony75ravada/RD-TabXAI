#!/usr/bin/env python3
"""Stage 10 — evidence ablation for T-B ranking: which evidence sources matter?"""
import pandas as pd
from _common import cfg_and_seed, load_processed, P, LOG
from rdtabxai.evaluation import harness

def main():
    cfg = cfg_and_seed(); A = load_processed(cfg); pairs = A["pairs"]
    rows = []
    for name, feats in harness.ABLATION_FEATURE_SETS.items():
        res = harness.run_ranking(pairs, cfg, strategy="disease_holdout",
                                  feature_cols=feats, model_name="lightgbm")
        if "_skipped" in res: continue
        rows.append({"feature_set": name, "n_features": len(feats), **res})
    df = pd.DataFrame(rows)
    out = P(cfg, "results_tables"); df.to_csv(out / "ablation_ranking.csv", index=False)
    LOG.info("Stage10 done: %d rows -> ablation_ranking.csv", len(df))

if __name__ == "__main__":
    main()
