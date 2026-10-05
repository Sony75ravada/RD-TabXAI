"""Assemble ML-ready artifacts from the intermediate tables.

Outputs to data_processed/:
  - disease_terms.json            ORPHA_ID -> sorted HPO term list (Orphadata ∪ HPOA)
  - disease_meta.csv              ORPHA_ID, Disease_Name, Disease_Group, causal_gene
  - gene_evidence.csv             Gene_Symbol + Orphadata/ClinGen/ClinVar evidence features
  - profiles.csv                  simulated patient profiles (T-A probe / T-C)
  - ranking_pairs.csv             (profile, candidate_gene, features..., causal) for T-B
  - join_audit.csv                unmatched rates for every join (no silent drops)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..io_utils import LOG, ensure_dir
from .encoding import freq_weight
from .simulate import simulate_profiles, jaccard


def _audit(name, left_keys, matched_keys):
    left = set(left_keys); m = set(matched_keys)
    unmatched = left - m
    return {"join": name, "n_left": len(left), "n_matched": len(left & m),
            "n_unmatched": len(unmatched),
            "unmatched_rate": round(len(unmatched) / max(len(left), 1), 4)}


def build(inter: dict[str, pd.DataFrame], cfg: dict, seed: int, out_dir):
    out = ensure_dir(out_dir)
    audit = []

    dhe, dge = inter["disease_hpo"], inter["disease_gene"]
    cls, hterms = inter["classification"], inter["hpo_terms"]
    clinvar_g, clingen_g = inter["clinvar_gene"], inter["clingen_gene"]
    g2p = inter.get("genes_to_pheno", pd.DataFrame(columns=["gene_symbol", "hpo_id"]))

    # ---- disease -> HPO term set (Orphadata ∪ HPOA), with frequency map
    dhe = dhe.copy()
    dhe["w"] = dhe["Frequency"].map(freq_weight)
    disease_terms = dhe.groupby("ORPHA_ID")["HPO_ID"].apply(lambda s: sorted(set(s))).to_dict()
    freq_map = (dhe.groupby(["ORPHA_ID", "HPO_ID"])["w"].max()
                .reset_index().groupby("ORPHA_ID")
                .apply(lambda g: dict(zip(g.HPO_ID, g.w)), include_groups=False).to_dict())

    # ---- disease group
    grp = cls.set_index("ORPHA_ID")["Disease_Group"].to_dict()
    audit.append(_audit("ORPHA->group", disease_terms.keys(), grp.keys()))

    # ---- causal gene per disease, defined from the Orphadata gene-disease ASSOCIATION
    # TYPE only (e.g. "Disease-causing germline mutation(s) in"). This is INDEPENDENT of
    # ClinGen validity, which the T-B ranker uses as a feature; defining the label from a
    # model feature would be target leakage. Tie-break is deterministic by gene symbol and
    # also independent of any model feature.
    dge = dge.copy()
    at = dge.get("Association_Type", pd.Series([""] * len(dge))).fillna("").str.lower()
    dge["_causal_rank"] = np.where(at.str.contains("disease-causing") & at.str.contains("germline"), 0,
                          np.where(at.str.contains("disease-causing"), 1,
                          np.where(at.str.contains("susceptibility|part of a fusion|role in"), 2, 3)))
    causal = (dge.sort_values(["_causal_rank", "Gene_Symbol"])
              .groupby("ORPHA_ID")["Gene_Symbol"].first().to_dict())
    audit.append(_audit("ORPHA->gene", disease_terms.keys(), causal.keys()))

    names = dhe.groupby("ORPHA_ID")["Disease_Name"].first().to_dict()
    meta = pd.DataFrame({
        "ORPHA_ID": list(disease_terms.keys()),
        "Disease_Name": [names.get(o) for o in disease_terms],
        "Disease_Group": [grp.get(o, "UNGROUPED") for o in disease_terms],
        "causal_gene": [causal.get(o) for o in disease_terms],
    })
    meta.to_csv(out / "disease_meta.csv", index=False)
    Path(out / "disease_terms.json").write_text(json.dumps(disease_terms))

    # ---- gene evidence table (Orphadata presence ∪ ClinGen ∪ ClinVar)
    all_genes = sorted(set(dge["Gene_Symbol"]) | set(clinvar_g["Gene_Symbol"]) |
                       set(clingen_g["Gene_Symbol"]))
    ge = pd.DataFrame({"Gene_Symbol": all_genes})
    ge["assoc_present"] = ge["Gene_Symbol"].isin(set(dge["Gene_Symbol"])).astype(int)
    ge = ge.merge(clinvar_g, on="Gene_Symbol", how="left")
    ge = ge.merge(clingen_g, on="Gene_Symbol", how="left")
    # explicit "no evidence" encodings (not silent NaN)
    for c in ["n_path_lp", "path_fraction", "star_weighted_path", "max_review_stars",
              "mean_submitters", "n_variants"]:
        if c in ge.columns:
            ge[c] = ge[c].fillna(0.0)
    if "clingen_validity_ord" in ge.columns:
        ge["clingen_validity_ord"] = ge["clingen_validity_ord"].fillna(-1.0)
    for c in ["Haploinsufficiency_Score", "Triplosensitivity_Score"]:
        if c in ge.columns:
            ge[c] = ge[c].fillna(0.0)
    ge.to_csv(out / "gene_evidence.csv", index=False)
    audit.append(_audit("gene->ClinVar", all_genes, clinvar_g["Gene_Symbol"]))
    audit.append(_audit("gene->ClinGen", all_genes, clingen_g["Gene_Symbol"]))

    # ---- IC for simulation noise (documented: global; model IC is refit train-only)
    from ..data.hpo import information_content, ancestor_map
    anc = ancestor_map(hterms)
    dh_long = dhe[["ORPHA_ID", "HPO_ID"]]
    ic = information_content(dh_long, anc)

    all_terms = sorted(set(dhe["HPO_ID"]))
    profiles = simulate_profiles(disease_terms, meta, all_terms, ic, cfg, seed)
    profiles.to_csv(out / "profiles.csv", index=False)
    profiles.to_csv(out / "phenotype_disease_matrix.csv", index=False)  # deliverable name
    profiles.rename(columns={}).to_csv(out / "disease_group_classification_matrix.csv", index=False)
    LOG.info("build: %d simulated profiles across %d diseases", len(profiles),
             profiles["ORPHA_ID"].nunique())

    # ---- T-B ranking pairs
    gene_to_terms = (g2p.groupby("gene_symbol")["hpo_id"].apply(set).to_dict()
                     if not g2p.empty else {})
    gene_to_diseases = dge.groupby("Gene_Symbol")["ORPHA_ID"].apply(set).to_dict()
    term_to_diseases = dh_long.groupby("HPO_ID")["ORPHA_ID"].apply(set).to_dict()
    ge_idx = ge.set_index("Gene_Symbol")
    pool = cfg["ranking"]["candidate_pool_size"]
    rng = np.random.RandomState(seed)

    pair_rows = []
    for _, p in profiles.iterrows():
        pterms = set(p["terms"].split("|"))
        # candidate genes: genes of diseases sharing >=1 term with the profile
        cand_dis = set().union(*[term_to_diseases.get(t, set()) for t in pterms]) if pterms else set()
        cand_genes = set().union(*[ {causal.get(d)} | gene_to_diseases.get(d, set())
                                    for d in cand_dis ]) if cand_dis else set()
        cand_genes = {g for g in cand_genes if g}
        cand_genes.add(p["causal_gene"])
        # sorted() makes sampling independent of Python's per-process string-hash
        # seed, so a fixed `seed` reproduces the same candidate pools on every run.
        cand_genes = sorted(cand_genes)
        if len(cand_genes) > pool:
            keep = {p["causal_gene"]}
            others = [g for g in cand_genes if g != p["causal_gene"]]
            keep |= set(rng.choice(others, size=pool - 1, replace=False))
            cand_genes = sorted(keep)
        for g in cand_genes:
            gt = gene_to_terms.get(g, set())
            row = {
                "profile_id": p["profile_id"], "ORPHA_ID": p["ORPHA_ID"],
                "candidate_gene": g,
                "pheno_gene_overlap": jaccard(pterms, gt),
                "assoc_present": int(ge_idx.loc[g, "assoc_present"]) if g in ge_idx.index else 0,
                "clingen_validity_ord": float(ge_idx.loc[g, "clingen_validity_ord"]) if g in ge_idx.index and "clingen_validity_ord" in ge_idx.columns else -1.0,
                "clinvar_path_fraction": float(ge_idx.loc[g, "path_fraction"]) if g in ge_idx.index and "path_fraction" in ge_idx.columns else 0.0,
                "clinvar_star_path": float(ge_idx.loc[g, "star_weighted_path"]) if g in ge_idx.index and "star_weighted_path" in ge_idx.columns else 0.0,
                "HI": float(ge_idx.loc[g, "Haploinsufficiency_Score"]) if g in ge_idx.index and "Haploinsufficiency_Score" in ge_idx.columns else 0.0,
                "TS": float(ge_idx.loc[g, "Triplosensitivity_Score"]) if g in ge_idx.index and "Triplosensitivity_Score" in ge_idx.columns else 0.0,
                "causal": int(g == p["causal_gene"]),
            }
            pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    # add reimplemented literature-baseline similarity scores (Phrank-style, Resnik, Lin)
    from ..baselines.external import add_similarity_columns
    sim_methods = cfg.get("ranking", {}).get("similarity_methods", None)
    pairs = add_similarity_columns(pairs, profiles, gene_to_terms, ic, anc, methods=sim_methods)
    pairs.to_csv(out / "ranking_pairs.csv", index=False)
    pairs.to_csv(out / "phenotype_gene_ranking_matrix.csv", index=False)  # deliverable name
    LOG.info("build: %d ranking pairs (%.1f avg candidates/profile, %.3f positive rate)",
             len(pairs), len(pairs) / max(len(profiles), 1), pairs["causal"].mean())

    pd.DataFrame(audit).to_csv(out / "join_audit.csv", index=False)
    LOG.info("build: join audit ->\n%s", pd.DataFrame(audit).to_string(index=False))
    return {"profiles": profiles, "pairs": pairs, "meta": meta, "gene_evidence": ge, "ic": ic, "anc": anc}
