"""Parse HPO sources to the project's intermediate schema.

Handles:
  - hp.obo  -> hpo_terms.csv (HPO_ID, HPO_Name, Definition, Parent_Terms, Ancestor_Terms, HPO_Depth)
  - phenotype.hpoa -> disease_hpo_edges rows (OMIM/ORPHA disease -> HPO + frequency)
  - genes_to_phenotype.txt -> gene<->HPO mapping (used for T-B gene-HPO evidence)

Parsers are dependency-free (no obonet/pronto) and defensive: unknown columns are
ignored, malformed lines are counted and logged, never silently dropped.
"""
from __future__ import annotations
import gzip, io, re
from collections import defaultdict, deque
from pathlib import Path
import math
import pandas as pd
from ..io_utils import LOG


def _open(path):
    path = str(path)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- hp.obo
def parse_obo(path: str | Path) -> pd.DataFrame:
    """Parse hp.obo into a term table with is_a parents."""
    terms, cur = [], None
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if cur is not None:
                    terms.append(cur)
                cur = {"HPO_ID": None, "HPO_Name": None, "Definition": "", "parents": [], "obsolete": False}
            elif line.startswith("[") and line.endswith("]"):
                # another stanza type (e.g. [Typedef]) -> close current term
                if cur is not None:
                    terms.append(cur); cur = None
            elif cur is not None:
                if line.startswith("id: HP:"):
                    cur["HPO_ID"] = line[4:].strip()
                elif line.startswith("name:"):
                    cur["HPO_Name"] = line[5:].strip()
                elif line.startswith("def:"):
                    m = re.search(r'"(.*?)"', line)
                    cur["Definition"] = m.group(1) if m else ""
                elif line.startswith("is_a:"):
                    pid = line.split("is_a:")[1].strip().split("!")[0].strip()
                    if pid.startswith("HP:"):
                        cur["parents"].append(pid)
                elif line.startswith("is_obsolete: true"):
                    cur["obsolete"] = True
    if cur is not None:
        terms.append(cur)

    df = pd.DataFrame([t for t in terms if t["HPO_ID"] and not t["obsolete"]])
    if df.empty:
        LOG.warning("parse_obo: no terms parsed from %s", path)
        return pd.DataFrame(columns=["HPO_ID", "HPO_Name", "Definition",
                                     "Parent_Terms", "Ancestor_Terms", "HPO_Depth"])
    parents = dict(zip(df["HPO_ID"], df["parents"]))
    ancestors, depth = _compute_ancestors_depth(parents)
    df["Parent_Terms"] = df["HPO_ID"].map(lambda i: "|".join(parents.get(i, [])))
    df["Ancestor_Terms"] = df["HPO_ID"].map(lambda i: "|".join(sorted(ancestors.get(i, set()))))
    df["HPO_Depth"] = df["HPO_ID"].map(lambda i: depth.get(i, 0))
    LOG.info("parse_obo: %d terms", len(df))
    return df[["HPO_ID", "HPO_Name", "Definition", "Parent_Terms", "Ancestor_Terms", "HPO_Depth"]]


def _compute_ancestors_depth(parents: dict[str, list[str]]):
    ancestors: dict[str, set] = {}
    depth: dict[str, int] = {}

    def anc(node, seen):
        if node in ancestors:
            return ancestors[node]
        acc = set()
        for p in parents.get(node, []):
            if p in seen:
                continue
            acc.add(p)
            acc |= anc(p, seen | {node})
        ancestors[node] = acc
        return acc

    for n in parents:
        anc(n, set())

    # depth = longest path (in edges) to a root term (no parents); memoized
    def dep(node, seen):
        if node in depth:
            return depth[node]
        ps = [p for p in parents.get(node, []) if p not in seen]
        d = 0 if not ps else 1 + max(dep(p, seen | {node}) for p in ps)
        depth[node] = d
        return d

    for n in parents:
        dep(n, set())
    return ancestors, depth


def ancestor_map(hpo_terms: pd.DataFrame) -> dict[str, set]:
    """HPO_ID -> set(ancestor ids) including the term itself (for true-path propagation)."""
    out = {}
    for _, r in hpo_terms.iterrows():
        anc = set(str(r["Ancestor_Terms"]).split("|")) if r["Ancestor_Terms"] else set()
        anc.discard("")
        anc.add(r["HPO_ID"])
        out[r["HPO_ID"]] = anc
    return out


# ---------------------------------------------------------------- phenotype.hpoa
# Official columns (HPOA >= 2021): database_id, disease_name, qualifier, hpo_id,
# reference, evidence, onset, frequency, sex, modifier, aspect, biocuration
HPOA_COLS = ["database_id", "disease_name", "qualifier", "hpo_id", "reference",
             "evidence", "onset", "frequency", "sex", "modifier", "aspect", "biocuration"]


def parse_hpoa(path: str | Path) -> pd.DataFrame:
    rows, bad = [], 0
    header = None
    with _open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                bad += 1
                continue
            rows.append(parts)
    if bad:
        LOG.warning("parse_hpoa: skipped %d malformed lines", bad)
    df = pd.DataFrame(rows, columns=header)
    # normalise to expected names if header differs in case
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in HPOA_COLS if c in df.columns]
    df = df[keep].copy()
    df = df[df.get("qualifier", "").astype(str).str.upper() != "NOT"]  # drop explicit negatives
    LOG.info("parse_hpoa: %d positive disease-HPO annotations", len(df))
    return df


# ---------------------------------------------------------------- genes_to_phenotype
def parse_genes_to_phenotype(path: str | Path) -> pd.DataFrame:
    """Tolerant parser for genes_to_phenotype.txt (column order has changed across releases)."""
    df = pd.read_csv(path, sep="\t", dtype=str, comment=None)
    df.columns = [c.strip().lstrip("#").lower() for c in df.columns]
    # canonical names we want
    rename = {}
    for c in df.columns:
        if "gene_symbol" in c or c == "gene-symbol" or c == "genesymbol":
            rename[c] = "gene_symbol"
        elif "ncbi" in c or c in ("gene_id", "entrez-gene-id", "entrez_gene_id"):
            rename[c] = "ncbi_gene_id"
        elif c in ("hpo_id", "hpo-term-id"):
            rename[c] = "hpo_id"
        elif c in ("hpo_name", "hpo-term-name"):
            rename[c] = "hpo_name"
        elif "disease" in c:
            rename[c] = "disease_id"
        elif "frequency" in c:
            rename[c] = "frequency"
    df = df.rename(columns=rename)
    LOG.info("parse_genes_to_phenotype: %d gene-HPO rows", len(df))
    return df


# ---------------------------------------------------------------- information content
def information_content(disease_hpo_long: pd.DataFrame,
                        ancestors: dict[str, set],
                        disease_col: str = "ORPHA_ID",
                        hpo_col: str = "HPO_ID") -> dict[str, float]:
    """Resnik IC computed from disease annotation frequency, propagated to ancestors.

    IC(t) = -log( freq(t) / N ) where freq(t) counts diseases annotated with t OR a descendant.
    MUST be called on TRAIN diseases only to avoid leakage.
    """
    dis_terms = disease_hpo_long.groupby(disease_col)[hpo_col].apply(set)
    N = max(len(dis_terms), 1)
    term_disease_count = defaultdict(int)
    for terms in dis_terms:
        propagated = set()
        for t in terms:
            propagated |= ancestors.get(t, {t})
        for t in propagated:
            term_disease_count[t] += 1
    ic = {}
    for t, c in term_disease_count.items():
        ic[t] = -math.log(max(c, 1) / N)
    return ic
