# Package contents: result files mapped to the manuscript

Headline result (rigorous leave-target-out benchmark, disease holdout):
**PhenoFuse-LTR MRR = 0.830 [0.781, 0.880]** against 0.655 for the strongest
single-similarity baseline (Resnik-BMA); Holm-corrected p <= 0.0013 against every
baseline on both splits. Every value below is reproduced exactly by the scripts in
`README.md` (the pipeline is deterministic).

## Tables (`results_real/tables/`)

| File | Manuscript | Content |
|------|-----------|---------|
| `T1_cohort_summary.csv` | Table 4 | Cohort: 400 OMIM disorders, 1,600 profiles, 19,048 pairs, pool size 12 |
| `T2_provenance.csv` | Table 5 | Data sources and their roles |
| `T3_feature_definitions.csv` | Table 3 | The 11 features in 3 views |
| `T10_hyperparameters.csv` | Table 6 | Ranker settings and protocol |
| `T4_naive_benchmark.csv` | Table 7 | Naive (annotation-recovery) benchmark |
| `T5_rigorous_benchmark.csv` | Table 8, Figs 3-5 | Rigorous leave-target-out benchmark |
| `T6_significance.csv` | Table 9 | Disorder-level Wilcoxon tests, Holm-corrected |
| `T4` + `T5` (derived) | Table 10 | Naive-to-rigorous drop, top-10 miss rate, generalisation gap |
| `T7_ablation.csv` | Table 11, Fig 6 | Cumulative and leave-one-view-out ablation |
| `T8_feature_importance.csv` | Fig 7 | LightGBM split counts |
| `T11_permutation_validity.csv` | Table 12, Fig 8 | Label-permutation leakage test |
| `T9_sensitivity_confound.csv` | Table 13 | Gene-term-count confound test |
| `T12_test_subset_stability.csv` | Table 14 | Stability over random test-disorder subsets |
| `E5_per_disease_rr.csv` | Fig 9 | Per-disorder reciprocal ranks (80 disorders) |
| `T5_rigorous_benchmark.csv` (mean of both splits) | Fig 10 | Overall ranking across disease and gene holdout |
| `T0_main_results.csv` | - | All benchmark rows in one long-format table |
| `E1_stronger_baselines.csv` | Table 15, Fig 11 | Random forest, logistic regression, rank aggregation |
| `E3_avg_ranks.csv`, `E3_friedman_nemenyi.json` | Fig 12 | Friedman chi2 = 97.8, p = 1.2e-17; Nemenyi CD = 1.34 |
| `E2_noise_robustness.csv` | Fig 13 | Term dropout, spurious terms, generalisation |
| `E4_shap_global.csv` | Fig 14 | Mean absolute SHAP per feature |
| `publication_tables.tex` | - | LaTeX versions of the main tables |

## Figures (`results_real/figures/`, PDF + 300-dpi PNG)

| File | Manuscript |
|------|-----------|
| `S1_architecture` | Fig 1 (author-drawn diagram) |
| `S2_leave_target_out` | Fig 2 (author-drawn diagram) |
| `F2_headline_naive_vs_lto` | Fig 3 |
| `F3_mrr_ci` | Fig 4 |
| `F4_hitk_curves` | Fig 5 |
| `F5_ablation` | Fig 6 |
| `F6_feature_importance` | Fig 7 |
| `F9_permutation_validity` | Fig 8 |
| `F7_per_disease_delta` | Fig 9 |
| `F8_overall_ranking` | Fig 10 |
| `E1_stronger_baselines` | Fig 11 |
| `E3_cd_diagram` | Fig 12 |
| `E2_noise_robustness` | Fig 13 |
| `E4_shap_global` | Fig 14 |

## Data (`data_processed_real/`)

The exact processed cohort behind every result, derived from HPO release 2025-01-16
(doi:10.5281/zenodo.14674339). `cohort_omim_disorders.csv` lists the 400 OMIM disorders.
See `DATA_AVAILABILITY.md` for identifiers and file checksums.
