#!/usr/bin/env python3
"""Stage 02 — parse HPO: hp.obo, phenotype.hpoa, genes_to_phenotype.txt."""
from _common import cfg_and_seed, raw_dir, proc_dir, LOG
from rdtabxai.data import hpo

def main():
    cfg = cfg_and_seed(); raw = raw_dir(cfg); out = proc_dir(cfg)
    terms = hpo.parse_obo(raw / "hp.obo"); terms.to_csv(out / "hpo_terms.csv", index=False)
    hpoa = hpo.parse_hpoa(raw / "phenotype.hpoa"); hpoa.to_csv(out / "disease_hpo_hpoa.csv", index=False)
    g2p = hpo.parse_genes_to_phenotype(raw / "genes_to_phenotype.txt")
    g2p.to_csv(out / "genes_to_phenotype.csv", index=False)
    LOG.info("Stage02 done: %d terms, %d hpoa rows, %d gene-pheno rows",
             len(terms), len(hpoa), len(g2p))

if __name__ == "__main__":
    main()
