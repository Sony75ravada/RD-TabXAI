#!/usr/bin/env python3
"""Determinism check: run the rigorous leave-target-out benchmark (disease holdout) and
print a hash of the feature matrix plus every method's MRR. Running this twice with
different PYTHONHASHSEED values must print identical output.

    PYTHONHASHSEED=1 PYTHONPATH=src python scripts/check_determinism.py
    PYTHONHASHSEED=2 PYTHONPATH=src python scripts/check_determinism.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdtabxai.io_utils import load_config, set_seed  # noqa: E402
from rdtabxai.data.hpo import ancestor_map  # noqa: E402
from rdtabxai.evaluation import ltr  # noqa: E402

BASELINES = ["resnik_bma", "phrank", "lin_bma", "cosine_bin", "jaccard"]


def main():
    cfg = load_config(ROOT / "configs/real_hpo.yaml")
    set_seed(cfg["seed"])
    pdir = ROOT / cfg["paths"]["data_processed"]
    profiles = pd.read_csv(pdir / "profiles.csv")
    pairs = pd.read_csv(pdir / "ranking_pairs.csv")
    disease_terms = json.loads((pdir / "disease_terms.json").read_text())
    anc = ancestor_map(pd.read_csv(pdir / "hpo_terms.csv"))
    g2p = pd.read_csv(pdir / "genes_to_phenotype.csv")
    gene_to_terms = g2p.groupby("gene_symbol")["hpo_id"].apply(set).to_dict()
    gdt = {}
    for (g, d), sub in g2p.groupby(["gene_symbol", "disease_id"]):
        gdt.setdefault(g, {})[d] = set(sub["hpo_id"])

    df, tr, te = ltr.compute_features(pairs, profiles, gene_to_terms, disease_terms, anc, cfg,
                                      strategy="disease_holdout", gene_disease_terms=gdt,
                                      leave_target_out=True)
    feat_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df[ltr.ALL_FEATURES].round(12), index=True).values.tobytes()
    ).hexdigest()[:16]
    print("feature-matrix hash:", feat_hash)
    prop = ltr._fit_lambdarank(df, tr, te, ltr.ALL_FEATURES, cfg["seed"])
    print(f"PhenoFuse-LTR  MRR={prop['mrr']:.6f}")
    for b in BASELINES:
        m = ltr._rank_by_column(df.loc[te], b, cfg)
        print(f"{b:14s} MRR={m['mrr']:.6f}")


if __name__ == "__main__":
    main()
