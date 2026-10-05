"""Statistical significance + uncertainty — CLUSTER-AWARE.

The independent statistical unit is the SOURCE DISEASE, not the simulated profile
(many profiles are generated from one disease and are therefore correlated). All
inference here resamples / blocks by disease so intervals are not anticonservative.

  - clustered_bootstrap_ci : resample whole diseases (clusters) with replacement
  - per_cluster_accuracy   : mean correctness per disease, for one model
  - cluster_wilcoxon       : paired Wilcoxon across diseases (independent units)
  - friedman_nemenyi       : omnibus across >=3 models with DISEASES as blocks
  - mcnemar_test           : item-level paired test (kept as a DIAGNOSTIC only;
                             anticonservative under clustering — not the headline test)
  - correct_pvalues        : holm | bonferroni | fdr_bh

We report effect sizes and CIs, not just p-values, and correct for multiple
comparisons whenever >2 models are compared.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------- clustered bootstrap
def clustered_bootstrap_ci(y_true, y_pred, clusters, metric_fn,
                           n_boot=1000, alpha=0.05, seed=0):
    """Percentile CI by resampling CLUSTERS (diseases) with replacement.

    On each draw we sample len(unique clusters) clusters with replacement and pool
    all their items (with multiplicity), then recompute the metric. This propagates
    between-disease variability that an item-level bootstrap would ignore.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    clusters = np.asarray(clusters)
    rng = np.random.RandomState(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    point = float(metric_fn(y_true, y_pred))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in chosen])
        try:
            boots.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": len(boots), "n_clusters": int(len(uniq))}


# --------------------------------------------------------- cluster-level paired
def per_cluster_accuracy(y_true, y_pred, clusters) -> dict:
    """Mean correctness within each cluster (disease) -> {cluster: accuracy}."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); clusters = np.asarray(clusters)
    correct = (y_true == y_pred).astype(float)
    out = {}
    for c in np.unique(clusters):
        m = clusters == c
        out[str(c)] = float(correct[m].mean())
    return out


def cluster_wilcoxon(acc_a: dict, acc_b: dict):
    """Paired Wilcoxon signed-rank across diseases shared by both models.
    Diseases are the independent units, so this is the valid paired test."""
    keys = sorted(set(acc_a) & set(acc_b))
    a = np.array([acc_a[k] for k in keys]); b = np.array([acc_b[k] for k in keys])
    diff = a - b
    n_eff = int(np.sum(diff != 0))
    if n_eff == 0:
        return {"n_clusters": len(keys), "n_nonzero": 0, "p_value": 1.0,
                "median_diff": 0.0, "mean_diff": 0.0}
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        stat, p = float("nan"), 1.0
    return {"n_clusters": len(keys), "n_nonzero": n_eff, "statistic": float(stat),
            "p_value": float(p), "median_diff": float(np.median(diff)),
            "mean_diff": float(np.mean(diff))}


# --------------------------------------------------------- omnibus (disease blocks)
def friedman_nemenyi(score_matrix, model_names):
    """score_matrix: rows = blocks (DISEASES), cols = models. Independent blocks."""
    M = np.asarray(score_matrix, dtype=float)
    if M.shape[1] < 3 or M.shape[0] < 2:
        return {"_note": "Friedman requires >=3 models and >=2 blocks",
                "n_blocks": int(M.shape[0]), "n_models": int(M.shape[1])}
    stat, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)  # rank 1 = best
    out = {"friedman_stat": float(stat), "friedman_p": float(p),
           "n_blocks": int(M.shape[0]), "mean_rank": {}}
    for j, nm in enumerate(model_names):
        out["mean_rank"][nm] = float(ranks[:, j].mean())
    k, n = M.shape[1], M.shape[0]
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}.get(k, 3.164)
    out["nemenyi_cd"] = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n)))
    return out


def block_score_matrix(preds: dict, models: list):
    """Build a [disease x model] accuracy matrix from per-model preds that share a
    test set (same split/seed). Only diseases present for all models are kept."""
    acc = {m: per_cluster_accuracy(preds[m]["y_true"], preds[m]["y_pred"],
                                   preds[m]["clusters"]) for m in models}
    common = sorted(set.intersection(*[set(acc[m]) for m in models])) if models else []
    M = np.array([[acc[m][c] for m in models] for c in common])
    return M, common


# --------------------------------------------------------- item-level (diagnostic)
def mcnemar_test(correct_a, correct_b):
    """Item-level McNemar. NOTE: anticonservative under clustering; reported only as
    a secondary diagnostic, never as the headline significance test."""
    a = np.asarray(correct_a).astype(bool); b = np.asarray(correct_b).astype(bool)
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "statistic": 0.0}
    if n01 + n10 < 25:
        p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
        stat = float(min(n01, n10))
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p = stats.chi2.sf(stat, 1)
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": float(p)}


def correct_pvalues(pvals, method="holm"):
    from statsmodels.stats.multitest import multipletests
    if not pvals:
        return [], []
    rej, p_adj, _, _ = multipletests(pvals, method=method)
    return list(p_adj), list(rej)
