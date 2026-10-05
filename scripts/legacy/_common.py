"""Shared helpers for the numbered scripts so each stays thin and chainable.
Every script reads/writes CSVs under data_processed/ and results/ so the stages
compose, and the whole thing is reproducible from configs/default.yaml + seed.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdtabxai.io_utils import load_config, set_seed, ensure_dir, LOG, write_provenance  # noqa


def cfg_and_seed(config_path=None):
    cfg = load_config(config_path or (ROOT / "configs/default.yaml"))
    set_seed(cfg["seed"])
    return cfg


def P(cfg, key):
    return ROOT / cfg["paths"][key]


def raw_dir(cfg):
    return ensure_dir(ROOT / cfg["paths"]["data_raw"])


def proc_dir(cfg):
    return ensure_dir(ROOT / cfg["paths"]["data_processed"])


def load_processed(cfg):
    """Load the assembled artifacts produced by stage 06."""
    pd_dir = proc_dir(cfg)
    profiles = pd.read_csv(pd_dir / "profiles.csv")
    pairs = pd.read_csv(pd_dir / "ranking_pairs.csv")
    disease_terms = json.loads((pd_dir / "disease_terms.json").read_text())
    meta = pd.read_csv(pd_dir / "disease_meta.csv")
    from rdtabxai.data.hpo import parse_obo, ancestor_map
    hterms = pd.read_csv(pd_dir / "hpo_terms.csv")
    anc = ancestor_map(hterms)
    return {"profiles": profiles, "pairs": pairs, "disease_terms": disease_terms,
            "meta": meta, "anc": anc, "hpo_terms": hterms}
