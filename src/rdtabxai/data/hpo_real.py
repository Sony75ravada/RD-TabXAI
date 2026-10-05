"""Ingest the REAL Human Phenotype Ontology release into the pipeline's intermediate
schema. Source files are the genuine HPO project downloads (hp.obo, phenotype.hpoa,
genes_to_phenotype.txt, phenotype_to_genes.txt), shipped with the `pyhpo` package or
placed in data_raw/ by the user.

Scope (honest): this gives REAL ontology + REAL disease-phenotype + REAL gene-phenotype
+ REAL disease-gene links. It does NOT include ClinVar/ClinGen variant evidence or the
Orphadata ORDO group tree (not reachable here), so the variant-evidence fusion and the
disease-GROUP task are deferred — evidence tables are written empty and clearly marked.

Inclusion criteria (configurable, documented in the paper's Methods):
  - disease must have >= min_terms annotated HPO terms
  - disease must map to exactly one causal gene (single-gene / Mendelian) when
    single_gene=True  (removes label ambiguity for the causal-gene ranking task)
  - optional random subsample (seeded) for compute-limited environments; this is a
    runtime constraint, disclosed, NOT a cherry-pick.
"""
from __future__ import annotations
import os, shutil, random
from pathlib import Path
import pandas as pd
from ..io_utils import LOG, ensure_dir, write_provenance


def _pyhpo_data_dir():
    try:
        import pyhpo
        d = Path(os.path.dirname(pyhpo.__file__)) / "data"
        if (d / "hp.obo").exists():
            return d
    except Exception:
        pass
    return None


def locate_sources(raw_dir) -> dict:
    """Prefer user-provided files in data_raw/; fall back to the pyhpo bundle."""
    raw = Path(raw_dir)
    need = {"hp.obo": "hp.obo", "phenotype.hpoa": "phenotype.hpoa",
            "genes_to_phenotype.txt": "genes_to_phenotype.txt",
            "phenotype_to_genes.txt": "phenotype_to_genes.txt"}
    found = {}
    bundle = _pyhpo_data_dir()
    for key, fname in need.items():
        if (raw / fname).exists():
            found[key] = raw / fname
        elif bundle and (bundle / fname).exists():
            found[key] = bundle / fname
        else:
            found[key] = None
    return found


def ingest(cfg, out_dir):
    """Build intermediate tables (real) and return inclusion stats."""
    from .hpo import parse_obo
    out = ensure_dir(out_dir)
    raw = ensure_dir(Path(out).parent / cfg["paths"]["data_raw"]) if "paths" in cfg else Path("data_raw")
    src = locate_sources(raw)
    missing = [k for k, v in src.items() if v is None]
    if missing:
        raise FileNotFoundError(f"missing real HPO files {missing}; place them in {raw} "
                                f"or `pip install pyhpo` to use the bundled release")
    rc = cfg.get("real", {})
    min_terms = rc.get("min_terms", 5)
    single_gene = rc.get("single_gene", True)
    subsample = rc.get("subsample", None)
    db_prefix = tuple(rc.get("db_prefix", ["OMIM", "ORPHA"]))
    seed = cfg["seed"]

    # ---- hpo terms (real ontology)
    hterms = parse_obo(src["hp.obo"])
    hterms.to_csv(out / "hpo_terms.csv", index=False)

    # ---- disease -> HPO (real, with frequency); drop NOT qualifiers
    hpoa = pd.read_csv(src["phenotype.hpoa"], sep="\t", comment="#", dtype=str)
    hpoa.columns = [c.strip().lower() for c in hpoa.columns]
    hpoa = hpoa[hpoa["qualifier"].fillna("") != "NOT"]
    hpoa = hpoa[hpoa["database_id"].str.startswith(db_prefix)]
    dhe = hpoa.rename(columns={"database_id": "ORPHA_ID", "disease_name": "Disease_Name",
                               "hpo_id": "HPO_ID", "frequency": "Frequency"})
    dhe = dhe[["ORPHA_ID", "Disease_Name", "HPO_ID", "Frequency"]].dropna(subset=["HPO_ID"])
    dhe["HPO_Term"] = None; dhe["Diagnostic_Criterion"] = None; dhe["Source"] = "HPOA"

    # ---- disease -> gene (real) from phenotype_to_genes
    p2g = pd.read_csv(src["phenotype_to_genes.txt"], sep="\t", dtype=str)
    p2g.columns = [c.strip().lower() for c in p2g.columns]
    dg = (p2g[["disease_id", "gene_symbol"]].dropna().drop_duplicates()
          .rename(columns={"disease_id": "ORPHA_ID", "gene_symbol": "Gene_Symbol"}))
    dg = dg[dg["ORPHA_ID"].str.startswith(db_prefix)]
    dg["Disease_Name"] = None; dg["Gene_ID"] = None
    dg["Association_Type"] = "Disease-causing (HPO gene-to-disease)"
    dg["Inheritance_Mode"] = None; dg["Source"] = "HPO_phenotype_to_genes"

    # ---- gene -> HPO (real)
    g2p = pd.read_csv(src["genes_to_phenotype.txt"], sep="\t", dtype=str)
    g2p.columns = [c.strip().lower() for c in g2p.columns]
    g2p = g2p.rename(columns={"gene_symbol": "gene_symbol", "hpo_id": "hpo_id"})
    g2p.to_csv(out / "genes_to_phenotype.csv", index=False)

    # ---- apply inclusion criteria
    terms_per = dhe.groupby("ORPHA_ID")["HPO_ID"].nunique()
    genes_per = dg.groupby("ORPHA_ID")["Gene_Symbol"].nunique()
    keep = set(terms_per[terms_per >= min_terms].index)
    keep &= set(genes_per[genes_per >= 1].index)
    if single_gene:
        keep &= set(genes_per[genes_per == 1].index)
    keep = sorted(keep)
    n_full = len(keep)
    if subsample and subsample < len(keep):
        rng = random.Random(seed)
        keep = sorted(rng.sample(keep, subsample))
    keep = set(keep)

    dhe = dhe[dhe["ORPHA_ID"].isin(keep)]
    dg = dg[dg["ORPHA_ID"].isin(keep)]
    dhe.to_csv(out / "disease_hpo.csv", index=False)
    dg.to_csv(out / "disease_gene.csv", index=False)

    # ---- deferred sources: write EMPTY, correctly-typed tables (honest placeholders)
    pd.DataFrame(columns=["ORPHA_ID", "Disease_Group"]).to_csv(out / "classification.csv", index=False)
    pd.DataFrame(columns=["Gene_Symbol", "n_variants", "n_path_lp", "path_fraction",
                          "star_weighted_path", "max_review_stars", "mean_submitters"]
                 ).to_csv(out / "clinvar_gene.csv", index=False)
    pd.DataFrame(columns=["Gene_Symbol", "clingen_validity_ord",
                          "Haploinsufficiency_Score", "Triplosensitivity_Score"]
                 ).to_csv(out / "clingen_gene.csv", index=False)

    stats = {"hpo_release": _hpo_version(src["phenotype.hpoa"]),
             "n_terms": int(len(hterms)),
             "cohort_diseases_full": int(n_full),
             "cohort_diseases_used": int(len(keep)),
             "subsampled": bool(subsample and subsample < n_full),
             "n_disease_hpo_edges": int(len(dhe)),
             "n_disease_gene_edges": int(len(dg)),
             "min_terms": min_terms, "single_gene": single_gene}
    write_provenance(out, {"stage": "real_ingest", **stats,
                           "sources": {k: str(v) for k, v in src.items()}})
    LOG.info("real ingest: HPO %s | %d terms | cohort %d/%d diseases (single_gene=%s, min_terms=%d)",
             stats["hpo_release"], stats["n_terms"], stats["cohort_diseases_used"],
             stats["cohort_diseases_full"], single_gene, min_terms)
    return stats


def _hpo_version(hpoa_path):
    try:
        for line in Path(hpoa_path).read_text().splitlines():
            if line.lower().startswith("#version"):
                return line.split(":", 1)[1].strip()
            if line.startswith("#hpo-version"):
                return line.split("releases/")[-1].split("/")[0]
    except Exception:
        pass
    return "unknown"
