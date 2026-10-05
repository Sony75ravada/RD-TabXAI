#!/usr/bin/env python3
"""Stage 06 — assemble disease terms, gene evidence, simulated profiles, ranking pairs."""
import pandas as pd
from _common import cfg_and_seed, proc_dir, LOG
from rdtabxai.features import build_matrices

def main():
    cfg = cfg_and_seed(); out = proc_dir(cfg)
    inter = {
        "disease_hpo": pd.read_csv(out / "disease_hpo.csv"),
        "disease_gene": pd.read_csv(out / "disease_gene.csv"),
        "classification": pd.read_csv(out / "classification.csv"),
        "hpo_terms": pd.read_csv(out / "hpo_terms.csv"),
        "genes_to_pheno": pd.read_csv(out / "genes_to_phenotype.csv"),
        "clinvar_gene": pd.read_csv(out / "clinvar_gene.csv"),
        "clingen_gene": pd.read_csv(out / "clingen_gene.csv"),
    }
    build_matrices.build(inter, cfg, cfg["seed"], out)
    LOG.info("Stage06 done: matrices written to %s", out)

if __name__ == "__main__":
    main()
