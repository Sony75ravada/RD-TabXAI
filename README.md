# RD-TabXAI / PhenoFuse-LTR: code and results for peer review

This package contains the complete code, the exact processed data, and every result table
and figure for the manuscript *"RD-TabXAI and PhenoFuse-LTR: A Leakage-Audited Benchmark
and Multi-View Fusion Learning-to-Rank Model for Phenotype-Driven Rare-Disease Gene
Prioritisation"*. It is provided for the Editorial Board and reviewers.

## 1. Setup (about 5 minutes)

Requires Python 3.10 or newer (results were produced with Python 3.12.3 on Linux).

```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements-lock.txt  # exact versions used; or requirements.txt for ranges
export PYTHONPATH=src                 # Windows (PowerShell): $env:PYTHONPATH="src"
```

No data download is needed. The `pyhpo` package (version 4.0.0) bundles the Human
Phenotype Ontology release 2025-01-16 used in the study.

## 2. Verify the data release and the code (under 1 minute)

```bash
python scripts/00_verify_data_release.py   # checks HPO release stamp + SHA-256 of all 3 files
python -m pytest -q tests                  # 7 unit tests
```

## 3. Reproduce the results

Run from the package root, in this order. Each step overwrites the corresponding files in
`results_real/`, so the shipped copies can be compared against freshly computed ones.

| Step | Command | Produces | Approx. time* |
|------|---------|----------|---------------|
| 1 | `python scripts/make_publication_artifacts.py` | Tables T1-T12 (naive + leave-target-out benchmarks, significance, ablation, importance, confound, permutation, stability) | see below |
| 2 | `python scripts/run_extensions.py` | E1-E4 (stronger baselines, noise robustness, Friedman-Nemenyi, SHAP) | see below |
| 3 | `python scripts/cache_perdisease.py` | E5 (per-disorder reciprocal ranks) | see below |
| 4 | `python scripts/make_figures_v2.py` | All data figures (PDF + PNG) | < 1 min |

*Single-threaded LightGBM on one CPU core; measured times are in `RUN_TIMES.md`.

Optional:

- `python scripts/run_real_hpo.py` rebuilds the processed cohort in `data_processed_real/`
  from the HPO release (1,600 simulated profiles over 400 OMIM disorders, 19,048 pairs).
- `python scripts/check_determinism.py` prints a feature-matrix hash and every method's MRR
  for the headline benchmark. Running it under two different `PYTHONHASHSEED` values gives
  identical output.

## 4. Determinism

All randomness is seeded (`seed: 20251229` in `configs/real_hpo.yaml`). Sampling and
ranking do not depend on Python's per-process string-hash order: candidate sets are sorted
before sampling, floating-point sums use exactly rounded `math.fsum`, ties are broken by
stable sorting, and LightGBM runs with `deterministic=True` on a single thread. Re-running
the pipeline therefore reproduces the shipped tables exactly.

## 5. Where things are

```
src/rdtabxai/              the library
  evaluation/ltr.py        PhenoFuse-LTR, the 11 features, leave-target-out construction
  evaluation/splits.py     group-disjoint disease / gene holdouts (with a leakage assertion)
  evaluation/stats.py      clustered bootstrap CIs, disorder-level Wilcoxon + Holm
  data/hpo_real.py         loader for the HPO release files
  features/                profile simulation and candidate pools
scripts/                   entry points (Section 3)
configs/real_hpo.yaml      every setting used (seed, cohort size, pool size, splits, bootstrap)
data_processed_real/       the exact processed cohort behind the reported results
  cohort_omim_disorders.csv  the 400 OMIM disorders (accession, name, causal gene, link)
results_real/tables/       all result tables (CSV/JSON)
results_real/figures/      all figures (PDF + 300-dpi PNG)
DATA_AVAILABILITY.md       dataset identifiers (DOI, PURL) and file checksums
PACKAGE_CONTENTS.md        maps every result file to its table or figure in the manuscript
```

## 6. Scope

Patient profiles are simulated from real HPO disease annotations. The study is a
benchmark of ranking methods under a leakage-audited protocol, not a clinical validation.
ClinVar, ClinGen and Orphadata are not used in any reported analysis.
