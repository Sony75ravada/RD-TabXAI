# Measured run times

Linux, Python 3.12.3, one CPU core for LightGBM (`deterministic=True`, `num_threads=1`).

| Step | Script | Time |
|------|--------|------|
| Verify data release | `scripts/00_verify_data_release.py` | < 5 s |
| Unit tests | `python -m pytest -q tests` | about 1 s |
| Rebuild cohort (optional) | `scripts/run_real_hpo.py` | about 30 s |
| Tables T1-T12 | `scripts/make_publication_artifacts.py` | about 2.5 min |
| Extensions E1-E4 | `scripts/run_extensions.py` | about 2 min |
| Per-disorder E5 | `scripts/cache_perdisease.py` | about 0.5 min |
| Figures | `scripts/make_figures_v2.py` | < 1 min |
| Determinism check | `scripts/check_determinism.py` | about 2 min |

Full reproduction of all reported results: about 6 minutes.
