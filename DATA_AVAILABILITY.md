# Data availability

All data analysed in this study are publicly available. No new primary data were
collected. No human participants or identifiable personal data are involved.

## Source dataset (permanent identifier)

**Human Phenotype Ontology (HPO), release 2025-01-16**

- Archived release (DOI): https://doi.org/10.5281/zenodo.14674339
  (Zenodo record 14674339, "obophenotype/human-phenotype-ontology: 2025-01-16 Release",
  licence CC BY 4.0)
- Persistent OBO PURL: http://purl.obolibrary.org/obo/hp/releases/2025-01-16/hp.obo
- Source repository: https://github.com/obophenotype/human-phenotype-ontology
- Reference: Gargano et al., Nucleic Acids Research 52(D1):D1333-D1346 (2024),
  https://doi.org/10.1093/nar/gkad1005

Files used from this release, with the exact checksums of the copies analysed:

| File | Content | SHA-256 |
|------|---------|---------|
| `hp.obo` | Ontology (19,034 terms); `data-version: hp/releases/2025-01-16` | `6b77de067eecc838319ce7650ed5bab0f92a502eabb160e6bc7c0238bc1548c5` |
| `phenotype.hpoa` | Disease-phenotype annotations; `#version: 2025-01-16` | `8180403e2f5de0d8f41890e587d95077ce7f8bb8228d5d7b29dd358b70f0938c` |
| `genes_to_phenotype.txt` | Gene-phenotype annotations with disease attribution | `77d4c616780ac048a6766f958ec8f6f194cd216e2edd3944c1a0756b6f3e9a36` |

The files were obtained through the `pyhpo` Python package, version 4.0.0
(https://pypi.org/project/pyhpo/4.0.0/), which bundles this release unchanged.
`python scripts/00_verify_data_release.py` checks the release stamp and all three checksums.

**Disease identifiers.** The 400 disorders in the cohort are single-gene disorders
identified by their Online Mendelian Inheritance in Man (OMIM) accession numbers
(for example `OMIM:104200`), taken from the HPO annotation files above
(https://omim.org; Amberger et al., Nucleic Acids Research 47(D1):D1038-D1043 (2019),
https://doi.org/10.1093/nar/gky1151). The full list of the 400 OMIM accessions used, with disease names, causal genes and
OMIM links, is in `data_processed_real/cohort_omim_disorders.csv`.

## Derived data generated in this study

The processed cohort (1,600 simulated phenotype profiles over 400 OMIM disorders), the
candidate-gene pools (19,048 profile-gene pairs), and every result table are included in
this package under `data_processed_real/` and `results_real/`. These are the exact files
behind the reported results.

## Not used

ClinVar, ClinGen, and Orphadata were not used in any reported analysis. They are named in
the manuscript only as future work.
