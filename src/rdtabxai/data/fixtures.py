"""Generate schema-faithful SYNTHETIC raw files for offline testing/demo.

These mimic the real file *formats* (hp.obo, phenotype.hpoa, Orphadata product4/6 XML,
ClinVar variant_summary.txt.gz, ClinGen CSV/TSV) so the entire pipeline runs end-to-end
without network access. Replace data_raw/ with real downloads for scientific results.

The synthetic biology is internally consistent: G disease groups, diseases per group
sharing some group-level phenotypes (signal) plus private phenotypes (specificity),
each disease with a causal gene, gene-level ClinVar/ClinGen evidence correlated (noisily)
with being causal. NOTHING here should be reported as a scientific result.
"""
from __future__ import annotations
import gzip, random
from pathlib import Path
from xml.sax.saxutils import escape as _xesc
import numpy as np
from ..io_utils import ensure_dir, LOG


def esc(s):
    return _xesc(str(s))


def generate(out_dir, seed=0, n_groups=6, diseases_per_group=12,
             n_leaf_terms=120, n_genes=140):
    rng = random.Random(seed)
    npr = np.random.RandomState(seed)
    out = ensure_dir(out_dir)

    # ---- ontology: root -> categories -> leaves
    root = "HP:0000118"
    categories = [f"HP:00010{str(i).zfill(2)}" for i in range(n_groups)]
    leaves = [f"HP:00{str(20000 + i).zfill(5)}" for i in range(n_leaf_terms)]
    leaf_parent = {lf: categories[i % n_groups] for i, lf in enumerate(leaves)}

    # ---- write hp.obo
    obo = ["format-version: 1.2", "ontology: hp", ""]
    def term(tid, name, parent=None):
        s = ["[Term]", f"id: {tid}", f"name: {name}",
             f'def: "Synthetic definition of {name}." []']
        if parent:
            s.append(f"is_a: {parent} ! {parent}")
        return "\n".join(s) + "\n"
    obo.append(term(root, "Phenotypic abnormality"))
    for c in categories:
        obo.append(term(c, f"Abnormality category {c}", root))
    for lf in leaves:
        obo.append(term(lf, f"Phenotype {lf}", leaf_parent[lf]))
    (out / "hp.obo").write_text("\n".join(obo))

    # ---- genes
    genes = [f"GENE{str(i).zfill(3)}" for i in range(n_genes)]

    # ---- diseases
    freq_labels = ["Obligate (100%)", "Very frequent (99-80%)", "Frequent (79-30%)",
                   "Occasional (29-5%)", "Very rare (<4-1%)"]
    diseases = []
    orpha_counter = 100000
    for g in range(n_groups):
        group_terms = rng.sample([lf for lf in leaves if leaf_parent[lf] == categories[g]],
                                 k=min(6, sum(1 for lf in leaves if leaf_parent[lf] == categories[g])))
        for d in range(diseases_per_group):
            orpha_counter += rng.randint(1, 9)
            shared = rng.sample(group_terms, k=min(3, len(group_terms)))
            private = rng.sample(leaves, k=rng.randint(3, 7))
            terms = sorted(set(shared + private))
            causal = genes[(g * diseases_per_group + d) % n_genes]
            diseases.append({
                "orpha": orpha_counter,
                "name": f"Rare disorder {orpha_counter}",
                "group": f"Group {g}",
                "terms": terms,
                "causal_gene": causal,
            })

    # ---- en_product4.xml (disease-HPO)
    leaf_name = {lf: f"Phenotype {lf}" for lf in leaves}
    p4 = ['<?xml version="1.0" encoding="UTF-8"?>', "<JDBOR>", "<HPODisorderSetStatusList>"]
    for dis in diseases:
        p4 += ["<HPODisorderSetStatus>", "<Disorder>",
               f"<OrphaCode>{dis['orpha']}</OrphaCode>",
               f'<Name lang="en">{esc(dis["name"])}</Name>',
               f'<HPODisorderAssociationList count="{len(dis["terms"])}">']
        for t in dis["terms"]:
            fr = rng.choice(freq_labels)
            p4 += ["<HPODisorderAssociation>", "<HPO>",
                   f"<HPOId>{t}</HPOId>", f"<HPOTerm>{esc(leaf_name[t])}</HPOTerm>", "</HPO>",
                   f'<HPOFrequency><Name lang="en">{esc(fr)}</Name></HPOFrequency>',
                   "</HPODisorderAssociation>"]
        p4 += ["</HPODisorderAssociationList>", "</Disorder>", "</HPODisorderSetStatus>"]
    p4 += ["</HPODisorderSetStatusList>", "</JDBOR>"]
    (out / "en_product4.xml").write_text("\n".join(p4))

    # ---- en_product6.xml (disease-gene)
    assoc_types = ["Disease-causing germline mutation(s) in",
                   "Disease-causing germline mutation(s) (loss of function) in"]
    p6 = ['<?xml version="1.0" encoding="UTF-8"?>', "<JDBOR>", "<DisorderList>"]
    for dis in diseases:
        # causal gene + sometimes a decoy gene also linked (technical inclusion)
        glist = [dis["causal_gene"]]
        if rng.random() < 0.3:
            glist.append(rng.choice(genes))
        p6 += ["<Disorder>", f"<OrphaCode>{dis['orpha']}</OrphaCode>",
               f'<Name lang="en">{esc(dis["name"])}</Name>',
               f'<TypeOfInheritance>Autosomal {rng.choice(["dominant", "recessive"])}</TypeOfInheritance>',
               f'<DisorderGeneAssociationList count="{len(glist)}">']
        for gi, gsym in enumerate(glist):
            p6 += ["<DisorderGeneAssociation>", "<Gene>",
                   f"<Symbol>{gsym}</Symbol>",
                   "<ExternalReferenceList>",
                   f"<ExternalReference><Source>HGNC</Source><Reference>{1000+genes.index(gsym)}</Reference></ExternalReference>",
                   "</ExternalReferenceList>", "</Gene>",
                   f'<DisorderGeneAssociationType><Name lang="en">{rng.choice(assoc_types)}</Name></DisorderGeneAssociationType>',
                   "</DisorderGeneAssociation>"]
        p6 += ["</DisorderGeneAssociationList>", "</Disorder>"]
    p6 += ["</DisorderList>", "</JDBOR>"]
    (out / "en_product6.xml").write_text("\n".join(p6))

    # ---- classification xml (groups)
    pc = ['<?xml version="1.0" encoding="UTF-8"?>', "<JDBOR>", "<ClassificationNodeList>"]
    for g in range(n_groups):
        pc += ["<ClassificationNode>", "<Disorder>",
               f"<OrphaCode>9{g:05d}</OrphaCode>", f'<Name lang="en">Group {g}</Name>',
               "</Disorder>", "<ClassificationNodeChildList>"]
        for dis in [d for d in diseases if d["group"] == f"Group {g}"]:
            pc += ["<ClassificationNode>", "<Disorder>",
                   f"<OrphaCode>{dis['orpha']}</OrphaCode>",
                   f'<Name lang="en">{esc(dis["name"])}</Name>', "</Disorder>",
                   "</ClassificationNode>"]
        pc += ["</ClassificationNodeChildList>", "</ClassificationNode>"]
    pc += ["</ClassificationNodeList>", "</JDBOR>"]
    (out / "en_product1.xml").write_text("\n".join(pc))

    # ---- phenotype.hpoa (subset overlaps Orphadata; adds OMIM-style rows)
    hpoa = ["#description: synthetic", "#date: 2099-01-01"]
    hpoa.append("\t".join(["database_id", "disease_name", "qualifier", "hpo_id", "reference",
                           "evidence", "onset", "frequency", "sex", "modifier", "aspect", "biocuration"]))
    for dis in diseases:
        for t in dis["terms"]:
            if rng.random() < 0.85:
                hpoa.append("\t".join([f"ORPHA:{dis['orpha']}", dis["name"], "", t,
                                       "PMID:0", "PCS", "", rng.choice(freq_labels),
                                       "", "", "P", "HPO:synthetic"]))
    (out / "phenotype.hpoa").write_text("\n".join(hpoa))

    # ---- genes_to_phenotype.txt
    g2p = ["\t".join(["ncbi_gene_id", "gene_symbol", "hpo_id", "hpo_name",
                      "frequency", "disease_id"])]
    for dis in diseases:
        for t in dis["terms"]:
            if rng.random() < 0.7:
                g2p.append("\t".join([str(1000 + genes.index(dis["causal_gene"])),
                                      dis["causal_gene"], t, leaf_name[t], "",
                                      f"ORPHA:{dis['orpha']}"]))
    (out / "genes_to_phenotype.txt").write_text("\n".join(g2p))

    # ---- ClinVar variant_summary.txt.gz (gene-level signal noisier for non-causal)
    review = ["criteria provided, single submitter",
              "criteria provided, multiple submitters, no conflicts",
              "reviewed by expert panel", "no assertion criteria provided"]
    causal_genes = {d["causal_gene"] for d in diseases}
    header = ["#AlleleID", "Type", "Name", "GeneID", "GeneSymbol", "HGNC_ID",
              "ClinicalSignificance", "ClinSigSimple", "LastEvaluated", "RS# (dbSNP)",
              "nsv/esv (dbVar)", "RCVaccession", "PhenotypeIDS", "PhenotypeList",
              "Origin", "OriginSimple", "Assembly", "ReviewStatus", "NumberSubmitters",
              "VariationID"]
    cv = ["\t".join(header)]
    aid = 100000
    for gi, gsym in enumerate(genes):
        is_causal = gsym in causal_genes
        nvar = npr.randint(2, 12)
        for _ in range(nvar):
            aid += 1
            p_path = 0.55 if is_causal else 0.12
            sig = "Pathogenic" if npr.rand() < p_path else npr.choice(
                ["Benign", "Likely benign", "Uncertain significance"])
            simple = 1 if "athogenic" in sig else 0
            cv.append("\t".join([str(aid), "single nucleotide variant", f"{gsym} variant",
                                  str(1000 + gi), gsym, str(2000 + gi), sig, str(simple),
                                  "2024", "0", "-", f"RCV{aid}", "", "synthetic condition",
                                  "germline", "germline", "GRCh38",
                                  npr.choice(review), str(npr.randint(1, 6)), str(aid)]))
    raw = ("\n".join(cv)).encode()
    with gzip.open(out / "variant_summary.txt.gz", "wb") as f:
        f.write(raw)

    # ---- ClinGen validity CSV (with preamble)
    classes = ["Definitive", "Strong", "Moderate", "Limited", "Disputed", "Refuted"]
    val = ["ClinGen Gene-Disease Validity (synthetic)", "++++ preamble line ++++",
           "GENE SYMBOL,GENE ID (HGNC),DISEASE LABEL,DISEASE ID (MONDO),MOI,SOP,CLASSIFICATION,CLASSIFICATION DATE"]
    for gi, gsym in enumerate(genes):
        if npr.rand() < 0.6:  # only ~60% of genes curated (realistic sparsity)
            cls = npr.choice(classes, p=[0.3, 0.15, 0.2, 0.2, 0.1, 0.05]) \
                if gsym in causal_genes else npr.choice(classes, p=[0.05, 0.1, 0.2, 0.35, 0.2, 0.1])
            val.append(",".join([gsym, f"HGNC:{1000+gi}", f"disease of {gsym}",
                                  f"MONDO:{gi:07d}", "AD", "SOP9", cls, "2024-01-01"]))
    (out / "clingen_validity.csv").write_text("\n".join(val))

    # ---- ClinGen dosage TSV (with preamble)
    dos = ["#ClinGen Dosage (synthetic)",
           "Gene Symbol\tGene ID\tHaploinsufficiency Score\tTriplosensitivity Score"]
    for gi, gsym in enumerate(genes):
        if npr.rand() < 0.4:
            hi = npr.choice([0, 1, 2, 3, 30])
            ts = npr.choice([0, 1, 2, 3, 30])
            dos.append(f"{gsym}\t{1000+gi}\t{hi}\t{ts}")
    (out / "clingen_dosage.tsv").write_text("\n".join(dos))

    LOG.info("fixtures: generated %d diseases / %d genes / %d groups in %s",
             len(diseases), n_genes, n_groups, out)
    return {"n_diseases": len(diseases), "n_genes": n_genes, "n_groups": n_groups,
            "out_dir": str(out)}
