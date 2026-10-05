#!/usr/bin/env python3
"""Stage 05 — parse ClinGen gene-disease validity + dosage -> gene evidence."""
from _common import cfg_and_seed, raw_dir, proc_dir, LOG
from rdtabxai.data import clingen

def main():
    cfg = cfg_and_seed(); raw = raw_dir(cfg); out = proc_dir(cfg)
    val = clingen.parse_validity(raw / "clingen_validity.csv")
    dos = clingen.parse_dosage(raw / "clingen_dosage.tsv")
    ev = clingen.build_gene_evidence(val, dos); ev.to_csv(out / "clingen_gene.csv", index=False)
    LOG.info("Stage05 done: %d genes with ClinGen evidence", len(ev))

if __name__ == "__main__":
    main()
