"""Robustness probes: how does performance degrade as phenotype profiles get
sparser (missing terms) or noisier (spurious terms added)?

This perturbs only the TEST profiles (the model is fixed), matching the design's
Stage 10 stress test. Returns a tidy frame: perturbation, fraction, metric values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .evaluation.splits import make_split
from .evaluation.metrics import classification_metrics
from .data.hpo import information_content
from .features.encoding import PhenotypeEncoder
from .models.registry import available_models


def _perturb(term_set, kind, frac, all_terms, rng):
    terms = list(term_set)
    if kind == "missing" and terms:
        keep = [t for t in terms if rng.rand() > frac]
        return set(keep) if keep else set(terms[:1])
    if kind == "noise":
        n_add = int(round(frac * max(len(terms), 1)))
        if n_add > 0:
            add = rng.choice(all_terms, size=n_add, replace=True)
            return set(terms) | set(map(str, add))
    return set(terms)


def robustness_curve(profiles, disease_terms, ancestors, cfg,
                     model_name="random_forest", target_col="Disease_Group",
                     strategy="disease_holdout"):
    seed = cfg["seed"]
    rng = np.random.RandomState(seed)
    df = profiles.reset_index(drop=True)
    tr_mask, te_mask = make_split(df, strategy, cfg["splits"]["test_size"], seed)

    train_orpha = set(df.loc[tr_mask, "ORPHA_ID"])
    dh_long = pd.DataFrame([(o, t) for o, ts in disease_terms.items() if o in train_orpha
                            for t in ts], columns=["ORPHA_ID", "HPO_ID"])
    ic = information_content(dh_long, ancestors)
    term_sets = df["terms"].apply(lambda s: set(s.split("|"))).tolist()
    enc = PhenotypeEncoder(ancestors=ancestors, ontology_aware=cfg["features"]["ontology_aware"],
                           ic_weighting=cfg["features"]["ic_weighting"],
                           min_doc_freq=cfg["features"]["min_term_document_freq"])
    enc.fit([term_sets[i] for i in np.where(tr_mask)[0]], ic=ic)
    all_terms = enc.vocab_
    y_raw = df[target_col].astype(str).values
    labels = sorted(set(y_raw[tr_mask]))
    code = {l: i for i, l in enumerate(labels)}

    Xtr = enc.transform([term_sets[i] for i in np.where(tr_mask)[0]]).toarray().astype(np.float64)
    ytr = np.array([code[v] for v in y_raw[tr_mask]])
    reg = available_models(cfg)
    model = reg.get(model_name, reg["random_forest"])[0]().fit(Xtr, ytr)

    te_idx = np.where(te_mask)[0]
    rows = []
    for kind, fracs in [("missing", cfg["robustness"]["missing_fractions"]),
                        ("noise", cfg["robustness"]["noise_fractions"])]:
        for frac in fracs:
            pert = [_perturb(term_sets[i], kind, frac, all_terms, rng) for i in te_idx]
            Xte = enc.transform(pert).toarray().astype(np.float64)
            seen = [j for j, i in enumerate(te_idx) if y_raw[i] in code]
            yte = np.array([code[y_raw[te_idx[j]]] for j in seen])
            ypred = model.predict(Xte[seen])
            m = classification_metrics(yte, ypred)
            rows.append({"perturbation": kind, "fraction": frac,
                         "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"], "mcc": m["mcc"]})
    return pd.DataFrame(rows)
