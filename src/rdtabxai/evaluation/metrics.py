"""Evaluation metrics for classification, ranking, and calibration."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             roc_auc_score, average_precision_score, brier_score_loss)


def classification_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and labels is not None:
        try:
            yb = _onehot(y_true, labels)
            m["auroc_ovr"] = roc_auc_score(yb, y_proba, average="macro", multi_class="ovr")
            m["auprc_macro"] = average_precision_score(yb, y_proba, average="macro")
        except Exception:
            m["auroc_ovr"] = float("nan"); m["auprc_macro"] = float("nan")
    return m


def _onehot(y, labels):
    idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(y), len(labels)))
    for r, v in enumerate(y):
        if v in idx:
            Y[r, idx[v]] = 1
    return Y


def expected_calibration_error(y_true_bin, p_conf, n_bins=10) -> float:
    """ECE for the predicted (max-prob) class correctness."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true_bin)
    for i in range(n_bins):
        m = (p_conf > bins[i]) & (p_conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc = y_true_bin[m].mean()
        conf = p_conf[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_metrics(y_true, y_pred, y_proba, labels) -> dict:
    conf = y_proba.max(axis=1)
    correct = (np.array(y_pred) == np.array(y_true)).astype(float)
    out = {"ece": expected_calibration_error(correct, conf)}
    # multiclass Brier = mean squared error vs one-hot
    Y = _onehot(y_true, labels)
    out["brier"] = float(np.mean(np.sum((y_proba - Y) ** 2, axis=1)))
    return out


# --------------------------------------------------------------- ranking
def ranking_metrics(scores_by_query: list[tuple[np.ndarray, np.ndarray]],
                    ks=(1, 3, 5, 10)) -> dict:
    """Each element: (scores, relevance) for one query's candidates.
    relevance is 0/1 (exactly one positive for causal-gene ranking)."""
    hits = {k: [] for k in ks}
    recall = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    rr = []
    for scores, rel in scores_by_query:
        order = np.argsort(-scores, kind="stable")
        rel_sorted = rel[order]
        # reciprocal rank of first relevant
        pos = np.where(rel_sorted == 1)[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
        n_rel = rel.sum()
        for k in ks:
            topk = rel_sorted[:k]
            hits[k].append(1.0 if topk.sum() > 0 else 0.0)
            recall[k].append(topk.sum() / n_rel if n_rel else 0.0)
            dcg = np.sum(topk / np.log2(np.arange(2, len(topk) + 2)))
            ideal = np.sort(rel)[::-1][:k]
            idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
            ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)
    out = {"mrr": float(np.mean(rr))}
    for k in ks:
        out[f"hit@{k}"] = float(np.mean(hits[k]))
        out[f"recall@{k}"] = float(np.mean(recall[k]))
        out[f"ndcg@{k}"] = float(np.mean(ndcg[k]))
    return out
