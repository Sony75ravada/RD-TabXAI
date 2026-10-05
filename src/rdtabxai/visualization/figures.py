"""Publication figures. All save to results/figures as PNG (300 dpi) + return path.
Figures are generated from real result frames; nothing is hand-drawn or faked.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..io_utils import ensure_dir

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out_dir, name):
    ensure_dir(out_dir)
    p = Path(out_dir) / name
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return str(p)


def fig_performance_bars(metrics_df, out_dir, metric="macro_f1", title=None):
    d = metrics_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(d))))
    ax.barh(d.index, d[metric], color="#3b6ea5")
    ax.set_xlabel(metric); ax.set_title(title or f"Model comparison — {metric}")
    for i, v in enumerate(d[metric]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    return _save(fig, out_dir, f"F_performance_{metric}.png")


def fig_calibration(y_true, y_proba, classes, out_dir, n_bins=10, name="F_calibration.png"):
    conf = y_proba.max(axis=1)
    pred = np.array(classes)[y_proba.argmax(1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(xs, ys, "o-", color="#c0504d", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram"); ax.legend()
    return _save(fig, out_dir, name)


def fig_confusion(y_true, y_pred, labels, out_dir, name="F_confusion.png"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Normalized confusion")
    fig.colorbar(im, fraction=0.046)
    return _save(fig, out_dir, name)


def fig_topk(ranking_results: dict[str, dict], out_dir, ks=(1, 3, 5, 10), name="F_topk.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, res in ranking_results.items():
        ys = [res.get(f"hit@{k}", np.nan) for k in ks]
        ax.plot(ks, ys, "o-", label=label)
    ax.set_xlabel("k"); ax.set_ylabel("Hit@k"); ax.set_title("Causal-gene ranking — Hit@k")
    ax.set_xticks(ks); ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


def fig_ablation(ablation_df, out_dir, metric="mrr", name="F_ablation.png"):
    d = ablation_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.5 * len(d))))
    ax.barh(d["feature_set"], d[metric], color="#4f8a5b")
    ax.set_xlabel(metric); ax.set_title(f"Ablation — {metric}")
    for i, v in enumerate(d[metric]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    return _save(fig, out_dir, name)


def fig_robustness(rob_df, out_dir, metric="macro_f1", name="F_robustness.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for kind, g in rob_df.groupby("perturbation"):
        ax.plot(g["fraction"], g[metric], "o-", label=kind)
    ax.set_xlabel("perturbation fraction"); ax.set_ylabel(metric)
    ax.set_title("Robustness to missing / noisy phenotypes"); ax.legend()
    return _save(fig, out_dir, name)


def fig_importance(imp_df, out_dir, value_col, name, top=15, title=None):
    d = imp_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(d))))
    ax.barh(d["feature"].astype(str), d[value_col], color="#8064a2")
    ax.set_xlabel(value_col); ax.set_title(title or "Feature importance")
    return _save(fig, out_dir, name)
