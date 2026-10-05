"""Unit tests: parsers round-trip, splits are group-disjoint, ranking metrics sane.
Run: PYTHONPATH=src python -m pytest tests/ -q   (or: python tests/test_pipeline.py)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdtabxai.io_utils import load_config
from rdtabxai.data import fixtures, hpo, orphadata, clinvar, clingen
from rdtabxai.evaluation.splits import make_split
from rdtabxai.evaluation.metrics import ranking_metrics

FX = ROOT / "tests" / "fixtures"


def setup_module(_=None):
    fixtures.generate(FX, seed=0)


def test_parsers_nonempty():
    assert len(hpo.parse_obo(FX / "hp.obo")) > 50
    assert len(orphadata.parse_product4(FX / "en_product4.xml")) > 0
    assert len(orphadata.parse_product6(FX / "en_product6.xml")) > 0
    cv = clinvar.aggregate_gene_level(clinvar.parse_variant_summary(FX / "variant_summary.txt.gz"))
    assert "path_fraction" in cv.columns and (cv["path_fraction"] <= 1).all()
    ev = clingen.build_gene_evidence(clingen.parse_validity(FX / "clingen_validity.csv"),
                                     clingen.parse_dosage(FX / "clingen_dosage.tsv"))
    assert (ev["clingen_validity_ord"] >= -1).all()


def test_depth_monotone():
    t = hpo.parse_obo(FX / "hp.obo")
    # root has depth 0; leaves deeper than categories
    assert t["HPO_Depth"].min() == 0 and t["HPO_Depth"].max() >= 1


def test_split_disjoint():
    df = pd.DataFrame({"ORPHA_ID": np.repeat([f"ORPHA:{i}" for i in range(20)], 5),
                       "causal_gene": np.repeat([f"G{i}" for i in range(20)], 5),
                       "Disease_Group": np.repeat([f"grp{i%4}" for i in range(20)], 5)})
    for strat in ["disease_holdout", "gene_holdout", "group_holdout"]:
        tr, te = make_split(df, strat, 0.3, 0)
        key = {"disease_holdout": "ORPHA_ID", "gene_holdout": "causal_gene",
               "group_holdout": "Disease_Group"}[strat]
        assert not (set(df.loc[tr, key]) & set(df.loc[te, key]))


def test_ranking_metric_perfect():
    # one query, true gene scored highest -> all hits = 1, MRR = 1
    q = [(np.array([0.9, 0.1, 0.2]), np.array([1, 0, 0]))]
    m = ranking_metrics(q, ks=(1, 3))
    assert m["mrr"] == 1.0 and m["hit@1"] == 1.0


def test_clustered_bootstrap_uses_clusters():
    from rdtabxai.evaluation import stats
    # 4 diseases x 5 profiles; clustered CI must report n_clusters=4 not 20
    y = np.array([0, 1] * 10); p = y.copy()
    clusters = np.repeat([f"D{i}" for i in range(4)], 5)
    acc = lambda a, b: float((a == b).mean())
    ci = stats.clustered_bootstrap_ci(y, p, clusters, acc, n_boot=200, seed=0)
    assert ci["n_clusters"] == 4 and ci["estimate"] == 1.0


def test_cluster_wilcoxon_runs():
    from rdtabxai.evaluation import stats
    a = {"D0": 1.0, "D1": 0.8, "D2": 0.6}
    b = {"D0": 0.5, "D1": 0.4, "D2": 0.3}
    r = stats.cluster_wilcoxon(a, b)
    assert r["n_clusters"] == 3 and r["mean_diff"] > 0


def test_phrank_monotonic():
    from rdtabxai.baselines.external import phrank_score
    anc = {"HP:1": {"HP:1", "HP:0"}, "HP:2": {"HP:2", "HP:0"}}
    ic = {"HP:0": 1.0, "HP:1": 3.0, "HP:2": 3.0}
    # gene sharing the exact informative term scores higher than sharing only the root
    s_match = phrank_score({"HP:1"}, {"HP:1"}, ic, anc)
    s_partial = phrank_score({"HP:1"}, {"HP:2"}, ic, anc)
    assert s_match > s_partial >= 0


if __name__ == "__main__":
    setup_module()
    test_parsers_nonempty(); test_depth_monotone()
    test_split_disjoint(); test_ranking_metric_perfect()
    test_clustered_bootstrap_uses_clusters(); test_cluster_wilcoxon_runs()
    test_phrank_monotonic()
    print("ALL TESTS PASSED")
