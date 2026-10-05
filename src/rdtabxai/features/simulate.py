"""Leakage-safe simulation of patient profiles + phenotype similarity.

Simulation samples a STRICT SUBSET of a disease's HPO terms (controlled masking),
applies dropout and IC-weighted noise. Each profile records its source disease so
that splits can keep all profiles of a disease on one side (no disease appears in
both train and test). This is the masking the design's Stage 8.3 requires.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .encoding import freq_weight


def simulate_profiles(disease_terms: dict[str, list[str]],
                      disease_meta: pd.DataFrame,
                      all_terms: list[str],
                      ic: dict[str, float],
                      cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = cfg["simulation"]["profiles_per_disease"]
    kmin = cfg["simulation"]["min_terms"]
    keep_lo, keep_hi = cfg["simulation"]["term_keep_fraction"]
    lam = cfg["simulation"]["noise_terms_lambda"]
    drop = cfg["simulation"]["dropout_extra"]
    meta = disease_meta.set_index("ORPHA_ID")
    ic_terms = np.array(sorted(ic.keys())) if ic else np.array(sorted(all_terms))  # hash-seed independent
    ic_p = None
    if ic:
        w = np.array([ic[t] for t in ic_terms]); w = w / w.sum() if w.sum() else None
        ic_p = w
    rows = []
    pid = 0
    for orpha, terms in disease_terms.items():
        terms = sorted(terms)
        if len(terms) < kmin:
            continue
        grp = meta.loc[orpha, "Disease_Group"] if orpha in meta.index and "Disease_Group" in meta.columns else "UNK"
        causal = meta.loc[orpha, "causal_gene"] if orpha in meta.index and "causal_gene" in meta.columns else None
        for _ in range(n):
            frac = rng.uniform(keep_lo, keep_hi)
            k = max(kmin, int(round(frac * len(terms))))
            chosen = list(rng.choice(terms, size=min(k, len(terms)), replace=False))
            chosen = [t for t in chosen if rng.rand() > drop] or chosen[:kmin]
            n_noise = rng.poisson(lam)
            if n_noise > 0:
                noise = rng.choice(ic_terms, size=n_noise, replace=False, p=ic_p)
                chosen = list(set(chosen) | set(map(str, noise)))
            rows.append({"profile_id": f"P{pid:06d}", "ORPHA_ID": orpha,
                         "Disease_Group": grp, "causal_gene": causal,
                         "terms": "|".join(sorted(chosen)), "n_terms": len(chosen)})
            pid += 1
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------- similarity
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def cosine_binary(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / np.sqrt(len(a) * len(b))


def resnik_bma(a: set, b: set, ic: dict[str, float], ancestors: dict[str, set]) -> float:
    """Best-match-average Resnik similarity between two term sets."""
    if not a or not b:
        return 0.0

    def pair(t1, t2):
        common = ancestors.get(t1, {t1}) & ancestors.get(t2, {t2})
        return max((ic.get(c, 0.0) for c in common), default=0.0)

    sa = np.mean([max((pair(t1, t2) for t2 in b), default=0.0) for t1 in a])
    sb = np.mean([max((pair(t2, t1) for t1 in a), default=0.0) for t2 in b])
    return float((sa + sb) / 2)
