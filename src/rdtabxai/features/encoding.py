"""Phenotype encoding: turn an HPO term set into a feature vector.

Supports flat binary, ancestor-augmented (true-path propagation), IC weighting,
and Orphadata frequency weighting. Vocabulary and IC are FIT ON TRAIN ONLY to
prevent leakage; transform() reuses the frozen vocabulary/weights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse


FREQ_TO_WEIGHT = {
    "obligate": 1.0, "very frequent": 0.9, "frequent": 0.55,
    "occasional": 0.17, "very rare": 0.025, "excluded": 0.0,
}


def freq_weight(label) -> float:
    if not isinstance(label, str):
        return 0.5
    l = label.lower()
    for k, w in FREQ_TO_WEIGHT.items():
        if k in l:
            return w
    return 0.5


class PhenotypeEncoder:
    def __init__(self, ancestors: dict[str, set] | None = None,
                 ontology_aware=True, ic_weighting=True, frequency_weighting=False,
                 min_doc_freq=1):
        self.ancestors = ancestors or {}
        self.ontology_aware = ontology_aware
        self.ic_weighting = ic_weighting
        self.frequency_weighting = frequency_weighting
        self.min_doc_freq = min_doc_freq
        self.vocab_: list[str] = []
        self.idx_: dict[str, int] = {}
        self.ic_: dict[str, float] = {}

    def _expand(self, terms):
        if not self.ontology_aware:
            return set(terms)
        out = set()
        for t in terms:
            out |= self.ancestors.get(t, {t})
        return out

    def fit(self, term_sets, ic: dict[str, float] | None = None):
        from collections import Counter
        self.ic_ = ic or {}
        df = Counter()
        for ts in term_sets:
            for t in self._expand(ts):
                df[t] += 1
        self.vocab_ = sorted([t for t, c in df.items() if c >= self.min_doc_freq])
        self.idx_ = {t: i for i, t in enumerate(self.vocab_)}
        return self

    def transform(self, term_sets, freq_maps=None):
        rows, cols, data = [], [], []
        for r, ts in enumerate(term_sets):
            expanded = self._expand(ts)
            fmap = (freq_maps[r] if freq_maps else None)
            for t in expanded:
                j = self.idx_.get(t)
                if j is None:
                    continue
                w = 1.0
                if self.ic_weighting and t in self.ic_:
                    w *= (1.0 + self.ic_[t])
                if self.frequency_weighting and fmap is not None:
                    w *= max(fmap.get(t, 0.5), 0.05)
                rows.append(r); cols.append(j); data.append(w)
        X = sparse.csr_matrix((data, (rows, cols)),
                              shape=(len(term_sets), len(self.vocab_)), dtype=np.float32)
        return X

    def fit_transform(self, term_sets, ic=None, freq_maps=None):
        return self.fit(term_sets, ic).transform(term_sets, freq_maps)
