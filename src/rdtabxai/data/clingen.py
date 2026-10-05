"""Parse ClinGen gene-disease validity and dosage sensitivity downloads.

Validity CSV has several preamble lines before the real header; dosage TSV likewise.
We sniff the header row defensively. Output -> clingen_gene_evidence schema.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..io_utils import LOG

VALIDITY_ORDINAL = {
    "refuted": 0, "disputed": 1, "limited": 2, "moderate": 3,
    "strong": 4, "definitive": 5, "no known disease relationship": 0,
}


def _read_with_preamble(path, key_token):
    """Find the header line containing key_token, then read from there."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    hdr_idx = None
    for i, ln in enumerate(lines):
        if key_token.lower() in ln.lower():
            hdr_idx = i
            break
    if hdr_idx is None:
        hdr_idx = 0
    sep = "\t" if "\t" in lines[hdr_idx] else ","
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(lines[hdr_idx:])), sep=sep, dtype=str,
                       engine="python", on_bad_lines="skip")


def parse_validity(path: str | Path) -> pd.DataFrame:
    df = _read_with_preamble(path, "GENE SYMBOL")
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {}
    for c in df.columns:
        if c.startswith("gene symbol") or c == "gene_symbol":
            ren[c] = "Gene_Symbol"
        elif "disease label" in c:
            ren[c] = "Disease_Name"
        elif "disease id" in c or "mondo" in c:
            ren[c] = "Disease_ID"
        elif c == "classification" or (c.startswith("classification") and "date" not in c):
            ren[c] = "Gene_Disease_Validity"
        elif c == "moi" or "inheritance" in c:
            ren[c] = "Inheritance_Mode"
    df = df.rename(columns=ren)
    keep = [c for c in ["Gene_Symbol", "Disease_Name", "Disease_ID",
                        "Gene_Disease_Validity", "Inheritance_Mode"] if c in df.columns]
    df = df[keep].dropna(subset=["Gene_Symbol"])
    LOG.info("parse_validity: %d ClinGen validity curations (%d genes)",
             len(df), df["Gene_Symbol"].nunique())
    return df


def parse_dosage(path: str | Path) -> pd.DataFrame:
    df = _read_with_preamble(path, "Gene Symbol")
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {}
    for c in df.columns:
        if c.startswith("gene symbol") or c == "gene_symbol":
            ren[c] = "Gene_Symbol"
        elif "haploinsufficiency score" in c:
            ren[c] = "Haploinsufficiency_Score"
        elif "triplosensitivity score" in c:
            ren[c] = "Triplosensitivity_Score"
    df = df.rename(columns=ren)
    keep = [c for c in ["Gene_Symbol", "Haploinsufficiency_Score",
                        "Triplosensitivity_Score"] if c in df.columns]
    df = df[keep].dropna(subset=["Gene_Symbol"])
    LOG.info("parse_dosage: %d ClinGen dosage rows", len(df))
    return df


def build_gene_evidence(validity: pd.DataFrame, dosage: pd.DataFrame) -> pd.DataFrame:
    """Merge validity + dosage into the clingen_gene_evidence table (gene-level)."""
    v = validity.copy()
    v["Gene_Disease_Validity_ord"] = (v.get("Gene_Disease_Validity", "")
                                      .fillna("").str.strip().str.lower().map(VALIDITY_ORDINAL))
    # best (max) validity per gene
    vmax = (v.groupby("Gene_Symbol")["Gene_Disease_Validity_ord"]
            .max().reset_index().rename(columns={"Gene_Disease_Validity_ord": "clingen_validity_ord"}))
    out = vmax
    if not dosage.empty:
        d = dosage.copy()
        for col in ["Haploinsufficiency_Score", "Triplosensitivity_Score"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        dg = d.groupby("Gene_Symbol")[[c for c in ["Haploinsufficiency_Score", "Triplosensitivity_Score"]
                                       if c in d.columns]].max().reset_index()
        out = out.merge(dg, on="Gene_Symbol", how="outer")
    out["clingen_validity_ord"] = out["clingen_validity_ord"].fillna(-1)  # -1 = no curation
    LOG.info("build_gene_evidence: %d genes with ClinGen evidence", len(out))
    return out
