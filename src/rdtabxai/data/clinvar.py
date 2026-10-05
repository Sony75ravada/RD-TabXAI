"""Parse ClinVar variant_summary.txt.gz and aggregate to GENE level.

Per-variant data is never exposed to the model (a specific causal variant would
trivially identify a disease). We keep gene-level pathogenicity evidence only.

Official variant_summary columns include (subset used):
  #AlleleID, Type, GeneSymbol, ClinicalSignificance, ClinSigSimple, ReviewStatus,
  NumberSubmitters, PhenotypeIDS, PhenotypeList, VariationID
"""
from __future__ import annotations
import gzip, io
from pathlib import Path
import pandas as pd
from ..io_utils import LOG

REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
}

USECOLS = ["#AlleleID", "Type", "GeneSymbol", "ClinicalSignificance", "ClinSigSimple",
           "ReviewStatus", "NumberSubmitters", "PhenotypeIDS", "PhenotypeList", "VariationID"]


def parse_variant_summary(path: str | Path) -> pd.DataFrame:
    op = (lambda p: io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8", errors="replace")) \
        if str(path).endswith(".gz") else (lambda p: open(p, encoding="utf-8", errors="replace"))
    with op(path) as f:
        df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols = [c for c in USECOLS if c in df.columns]
    df = df[cols].copy()
    df = df.rename(columns={"#AlleleID": "Allele_ID", "GeneSymbol": "Gene_Symbol",
                            "ClinicalSignificance": "Clinical_Significance",
                            "ReviewStatus": "Review_Status",
                            "NumberSubmitters": "Number_of_Submitters",
                            "PhenotypeIDS": "Phenotype_IDs", "PhenotypeList": "Phenotype_Names",
                            "VariationID": "Variation_ID", "Type": "Variant_Type"})
    LOG.info("parse_variant_summary: %d variant records (%d genes)",
             len(df), df["Gene_Symbol"].nunique())
    return df


def aggregate_gene_level(variants: pd.DataFrame) -> pd.DataFrame:
    """Collapse variant rows to one row per gene with pathogenicity evidence features."""
    v = variants.copy()
    v["ClinSigSimple"] = pd.to_numeric(v.get("ClinSigSimple"), errors="coerce")
    v["Number_of_Submitters"] = pd.to_numeric(v.get("Number_of_Submitters"), errors="coerce").fillna(0)
    v["stars"] = v["Review_Status"].str.strip().str.lower().map(REVIEW_STARS).fillna(0)
    sig = v["Clinical_Significance"].fillna("").str.lower()
    v["is_path"] = sig.str.contains("pathogenic") & ~sig.str.contains("benign")
    v["is_benign"] = sig.str.contains("benign")

    def agg(g):
        n_total = len(g)
        n_path = int(g["is_path"].sum())
        return pd.Series({
            "n_variants": n_total,
            "n_path_lp": n_path,
            "path_fraction": n_path / n_total if n_total else 0.0,
            "star_weighted_path": float((g["is_path"] * g["stars"]).sum()),
            "max_review_stars": float(g["stars"].max()),
            "mean_submitters": float(g["Number_of_Submitters"].mean()),
        })

    out = v.groupby("Gene_Symbol").apply(agg, include_groups=False).reset_index()
    LOG.info("aggregate_gene_level: %d genes with ClinVar evidence", len(out))
    return out
