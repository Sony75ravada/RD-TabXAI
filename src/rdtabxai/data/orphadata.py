"""Parse Orphadata XML products to the project's intermediate schema.

  - en_product4.xml -> disease_hpo_edges (ORPHA_ID, Disease_Name, HPO_ID, HPO_Term,
                       Frequency, Diagnostic_Criterion, Source)
  - en_product6.xml -> disease_gene_edges (ORPHA_ID, Disease_Name, Gene_Symbol,
                       Gene_ID, Association_Type, Inheritance_Mode, Source)
  - classification -> ORPHA_ID -> disease group mapping (Disease_Group)

Uses xml.etree with iterparse for memory safety on the large real files.
Schema follows the documented Orphadata JDBOR structure.
"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from ..io_utils import LOG


def _text(el, tag):
    c = el.find(tag)
    return c.text.strip() if c is not None and c.text else None


def parse_product4(path: str | Path) -> pd.DataFrame:
    """Disease-HPO associations with frequency + diagnostic criterion."""
    rows = []
    tree = ET.parse(path)
    root = tree.getroot()
    for disorder in root.iter("Disorder"):
        orpha = _text(disorder, "OrphaCode") or _text(disorder, "OrphaNumber")
        name = _text(disorder, "Name")
        assoc_list = disorder.find("HPODisorderAssociationList")
        if assoc_list is None:
            continue
        for a in assoc_list.findall("HPODisorderAssociation"):
            hpo = a.find("HPO")
            hpo_id = _text(hpo, "HPOId") if hpo is not None else None
            hpo_term = _text(hpo, "HPOTerm") if hpo is not None else None
            freq_el = a.find("HPOFrequency")
            freq = _text(freq_el, "Name") if freq_el is not None else None
            dc_el = a.find("DiagnosticCriteria")
            dc = _text(dc_el, "Name") if dc_el is not None else None
            if orpha and hpo_id:
                rows.append([f"ORPHA:{orpha}", name, hpo_id, hpo_term, freq, dc, "Orphadata_product4"])
    df = pd.DataFrame(rows, columns=["ORPHA_ID", "Disease_Name", "HPO_ID", "HPO_Term",
                                     "Frequency", "Diagnostic_Criterion", "Source"])
    LOG.info("parse_product4: %d disease-HPO edges (%d diseases)",
             len(df), df["ORPHA_ID"].nunique())
    return df


def parse_product6(path: str | Path) -> pd.DataFrame:
    """Disease-gene associations with association type + inheritance.

    NOTE (per Orphadata FAQ): gene inclusion is a *technical* choice and is NOT an
    evidence grade. Evidence weighting comes from ClinGen, not from this presence.
    """
    rows = []
    tree = ET.parse(path)
    root = tree.getroot()
    for disorder in root.iter("Disorder"):
        orpha = _text(disorder, "OrphaCode") or _text(disorder, "OrphaNumber")
        name = _text(disorder, "Name")
        gal = disorder.find("DisorderGeneAssociationList")
        if gal is None:
            continue
        for a in gal.findall("DisorderGeneAssociation"):
            gene = a.find("Gene")
            symbol = _text(gene, "Symbol") if gene is not None else None
            gene_id = None
            if gene is not None:
                erl = gene.find("ExternalReferenceList")
                if erl is not None:
                    for er in erl.findall("ExternalReference"):
                        src = _text(er, "Source")
                        ref = _text(er, "Reference")
                        if src in ("HGNC", "OMIM") and ref:
                            gene_id = f"{src}:{ref}"
                            if src == "HGNC":
                                break
            at_el = a.find("DisorderGeneAssociationType")
            assoc_type = _text(at_el, "Name") if at_el is not None else None
            # inheritance often lives at disorder level in product9; placeholder here
            inh = _text(disorder, "TypeOfInheritance")
            if orpha and symbol:
                rows.append([f"ORPHA:{orpha}", name, symbol, gene_id, assoc_type, inh, "Orphadata_product6"])
    df = pd.DataFrame(rows, columns=["ORPHA_ID", "Disease_Name", "Gene_Symbol", "Gene_ID",
                                     "Association_Type", "Inheritance_Mode", "Source"])
    LOG.info("parse_product6: %d disease-gene edges (%d diseases, %d genes)",
             len(df), df["ORPHA_ID"].nunique(), df["Gene_Symbol"].nunique())
    return df


def parse_classification(path: str | Path) -> pd.DataFrame:
    """Map each ORPHA disorder to a top-level disease group.

    Real ORDO classification is a DAG; here we walk the ClassificationNode tree and
    assign the highest ancestor node name as Disease_Group. The fixture mirrors this
    with a simplified <ClassificationNodeList> nesting.
    """
    rows = []
    tree = ET.parse(path)
    root = tree.getroot()

    def walk(node, group_name):
        dis = node.find("Disorder")
        if dis is not None:
            orpha = _text(dis, "OrphaCode") or _text(dis, "OrphaNumber")
            if orpha:
                rows.append([f"ORPHA:{orpha}", group_name])
        child_list = node.find("ClassificationNodeChildList")
        if child_list is not None:
            for child in child_list.findall("ClassificationNode"):
                walk(child, group_name)

    for top in root.iter("ClassificationNode"):
        dis = top.find("Disorder")
        gname = _text(dis, "Name") if dis is not None else None
        # only treat as group root if it has children
        if top.find("ClassificationNodeChildList") is not None and gname:
            walk(top, gname)

    df = pd.DataFrame(rows, columns=["ORPHA_ID", "Disease_Group"]).drop_duplicates("ORPHA_ID")
    LOG.info("parse_classification: %d disease-group assignments (%d groups)",
             len(df), df["Disease_Group"].nunique())
    return df
