#!/usr/bin/env python3
"""Stage 03 — parse Orphadata product4 (disease-HPO), product6 (disease-gene),
classification (groups)."""
from _common import cfg_and_seed, raw_dir, proc_dir, LOG
from rdtabxai.data import orphadata

def main():
    cfg = cfg_and_seed(); raw = raw_dir(cfg); out = proc_dir(cfg)
    p4 = orphadata.parse_product4(raw / "en_product4.xml"); p4.to_csv(out / "disease_hpo.csv", index=False)
    p6 = orphadata.parse_product6(raw / "en_product6.xml"); p6.to_csv(out / "disease_gene.csv", index=False)
    cls = orphadata.parse_classification(raw / "en_product1.xml"); cls.to_csv(out / "classification.csv", index=False)
    LOG.info("Stage03 done: %d disease-HPO, %d disease-gene, %d group rows",
             len(p4), len(p6), len(cls))

if __name__ == "__main__":
    main()
