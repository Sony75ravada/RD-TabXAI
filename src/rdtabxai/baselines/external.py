"""Literature / external baseline methods for the T-B causal-gene ranking task.

Two categories, kept strictly separate for honesty:

1. REIMPLEMENTED published algorithms (run here, no external resources needed):
     - phrank_score      : Phrank-style shared-information-content scoring
                           (reimplementation per Jagadeesh et al., Genet Med 2019).
                           Labelled "phrank_style" — NOT the official Phrank binary.
     - resnik_bma_score  : Resnik best-match-average semantic similarity
     - lin_bma_score     : Lin best-match-average semantic similarity
   These give a fair, reproducible comparison using only HPO + gene-phenotype data.

2. EXTERNAL TOOL ADAPTERS (you run the real tool, we ingest its output):
     - load_external_ranking : parse Exomiser TSV / GADO / Phen2Gene output into a
                               common (profile_id, gene, score) frame to merge on the
                               SAME holdout split. We never fabricate tool outputs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _closure(terms, ancestors):
    out = set()
    for t in terms:
        out |= ancestors.get(t, {t})
    return out


def phrank_score(patient_terms, gene_terms, ic, ancestors) -> float:
    """Phrank-style: sum of information content over the shared ancestral closure.

    Reimplementation of the core idea in Jagadeesh et al. (2019): the match between a
    patient's phenotypes and a gene's annotated phenotypes is scored by the total
    information content of the terms they jointly imply (their common ancestor set).
    IC must be derived from the gene-phenotype annotation frequencies (train-only).
    """
    pc = _closure(patient_terms, ancestors)
    gc = _closure(gene_terms, ancestors)
    shared = pc & gc
    return float(sum(ic.get(t, 0.0) for t in shared))


def _mica_ic(t1, t2, ic, ancestors):
    common = ancestors.get(t1, {t1}) & ancestors.get(t2, {t2})
    return max((ic.get(c, 0.0) for c in common), default=0.0)


def resnik_bma_score(patient_terms, gene_terms, ic, ancestors) -> float:
    a, b = list(patient_terms), list(gene_terms)
    if not a or not b:
        return 0.0
    sa = np.mean([max((_mica_ic(x, y, ic, ancestors) for y in b), default=0.0) for x in a])
    sb = np.mean([max((_mica_ic(y, x, ic, ancestors) for x in a), default=0.0) for y in b])
    return float((sa + sb) / 2)


def lin_bma_score(patient_terms, gene_terms, ic, ancestors) -> float:
    a, b = list(patient_terms), list(gene_terms)
    if not a or not b:
        return 0.0

    def lin(x, y):
        mica = _mica_ic(x, y, ic, ancestors)
        denom = ic.get(x, 0.0) + ic.get(y, 0.0)
        return (2 * mica / denom) if denom > 0 else 0.0

    sa = np.mean([max((lin(x, y) for y in b), default=0.0) for x in a])
    sb = np.mean([max((lin(y, x) for x in a), default=0.0) for y in b])
    return float((sa + sb) / 2)


SIMILARITY_METHODS = {
    "phrank_style": phrank_score,
    "resnik_bma": resnik_bma_score,
    "lin_bma": lin_bma_score,
}


def add_similarity_columns(pairs: pd.DataFrame, profiles: pd.DataFrame,
                           gene_to_terms: dict, ic: dict, ancestors: dict,
                           methods=None) -> pd.DataFrame:
    """Add sim_<method> columns to the ranking_pairs frame (computed for every pair).

    `methods` selects a subset of SIMILARITY_METHODS (e.g. ["phrank_style"] for large
    real cohorts where the O(n^2) Resnik/Lin BMA scores are too slow). Default = all.

    IMPORTANT: pass an IC dict fit on TRAIN diseases only when using this for the
    headline comparison. For the demo we use the global IC; the harness recomputes
    train-only IC when wiring this into a leakage-safe split (see scripts/09b)."""
    use = methods or list(SIMILARITY_METHODS)
    fns = {f"sim_{m}": SIMILARITY_METHODS[m] for m in use if m in SIMILARITY_METHODS}
    pterms = profiles.set_index("profile_id")["terms"].apply(lambda s: set(s.split("|"))).to_dict()
    cols = {c: [] for c in fns}
    for _, r in pairs.iterrows():
        P = pterms.get(r["profile_id"], set())
        G = gene_to_terms.get(r["candidate_gene"], set())
        for c, fn in fns.items():
            cols[c].append(fn(P, G, ic, ancestors))
    out = pairs.copy()
    for c, v in cols.items():
        out[c] = v
    return out


# ------------------------------------------------------- external tool adapters
def load_external_ranking(path: str | Path, tool: str) -> pd.DataFrame:
    """Read a real external tool's output into (profile_id, candidate_gene, score).

    Expected formats (run the tool yourself, then point here):
      - exomiser : TSV with columns including GENE_SYMBOL and EXOMISER_GENE_COMBINED_SCORE
      - gado     : TSV with columns Hgnc / Zscore (per sample)
      - phen2gene: TSV with columns Gene / Score (rank file per sample)
    The 'profile_id' (sample) is taken from a 'sample'/'profile_id' column if present,
    else from the file stem. Returns a tidy frame; nothing is invented.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    tool = tool.lower()
    if tool == "exomiser":
        gcol = next((c for c in df.columns if "gene_symbol" in c or c == "gene"), None)
        scol = next((c for c in df.columns if "combined_score" in c or "exomiser" in c and "score" in c), None)
    elif tool == "gado":
        gcol = next((c for c in df.columns if c in ("hgnc", "gene", "ensg")), None)
        scol = next((c for c in df.columns if "zscore" in c or c == "z"), None)
    elif tool == "phen2gene":
        gcol = next((c for c in df.columns if c == "gene"), None)
        scol = next((c for c in df.columns if c == "score"), None)
    else:
        raise ValueError(f"unknown tool {tool}")
    if gcol is None or scol is None:
        raise ValueError(f"could not locate gene/score columns for {tool}: {list(df.columns)}")
    pcol = next((c for c in df.columns if c in ("sample", "profile_id")), None)
    out = pd.DataFrame({
        "profile_id": df[pcol] if pcol else path.stem,
        "candidate_gene": df[gcol],
        "score": pd.to_numeric(df[scol], errors="coerce"),
        "tool": tool,
    })
    return out.dropna(subset=["score"])
