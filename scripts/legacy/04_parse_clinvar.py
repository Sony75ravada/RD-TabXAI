#!/usr/bin/env python3
"""Stage 04 — parse ClinVar variant_summary -> gene-level evidence (no per-variant leakage)."""
from _common import cfg_and_seed, raw_dir, proc_dir, LOG
from rdtabxai.data import clinvar

def main():
    cfg = cfg_and_seed(); raw = raw_dir(cfg); out = proc_dir(cfg)
    v = clinvar.parse_variant_summary(raw / "variant_summary.txt.gz")
    g = clinvar.aggregate_gene_level(v); g.to_csv(out / "clinvar_gene.csv", index=False)
    LOG.info("Stage04 done: %d genes with ClinVar evidence", len(g))

if __name__ == "__main__":
    main()
