#!/usr/bin/env python3
"""Stage 15 — orchestrate stages 01..14 from a single config + seed.

    python scripts/15_main_reproducible_pipeline.py
Each stage is importable; failures are logged but do not silently pass."""
import importlib, time
from _common import cfg_and_seed, LOG

STAGES = [
    "01_download_official_sources", "02_parse_hpo", "03_parse_orphadata",
    "04_parse_clinvar", "05_parse_clingen", "06_build_tabular_matrices",
    "07_train_baselines", "08_train_proposed_model", "09_evaluate_models",
    "10_run_ablation", "11_run_statistical_tests", "12_generate_explainability",
    "13_generate_publication_tables", "14_generate_publication_figures",
]

def main():
    cfg_and_seed()
    t0 = time.time()
    for s in STAGES:
        LOG.info("===== %s =====", s)
        mod = importlib.import_module(s)
        mod.main()
    LOG.info("PIPELINE COMPLETE in %.1fs", time.time() - t0)

if __name__ == "__main__":
    main()
