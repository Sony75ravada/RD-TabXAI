"""Leakage-safe splitting.

The unit of analysis is a simulated profile, but the *grouping* key prevents leakage:
  - random:          profile-level random (BASELINE ONLY; leaks disease identity)
  - disease_holdout: no ORPHA_ID appears in both train and test
  - gene_holdout:    no causal_gene appears in both (for T-B)
  - group_holdout:   whole disease groups held out (hardest generalization)
Returns boolean train/test masks aligned to the dataframe index.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_split(df: pd.DataFrame, strategy: str, test_size: float, seed: int):
    if strategy == "random":
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    else:
        keymap = {"disease_holdout": "ORPHA_ID", "gene_holdout": "causal_gene",
                  "group_holdout": "Disease_Group"}
        key = keymap[strategy]
        groups = df[key].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=groups))
    train_mask = np.zeros(len(df), dtype=bool); train_mask[tr] = True
    test_mask = np.zeros(len(df), dtype=bool); test_mask[te] = True
    # assert disjointness on the grouping key for holdouts
    if strategy != "random":
        key = {"disease_holdout": "ORPHA_ID", "gene_holdout": "causal_gene",
               "group_holdout": "Disease_Group"}[strategy]
        overlap = set(df.loc[train_mask, key]) & set(df.loc[test_mask, key])
        assert not overlap, f"LEAKAGE: {len(overlap)} {key} in both splits"
    return train_mask, test_mask
