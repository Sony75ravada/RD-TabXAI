"""Shared utilities: config, deterministic seeding, logging, safe IO, provenance."""
from __future__ import annotations
import os, sys, json, random, logging, hashlib, datetime
from pathlib import Path
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def get_logger(name: str = "rdtabxai") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


LOG = get_logger()


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    raise RuntimeError("PyYAML not available; install pyyaml.")


def set_seed(seed: int) -> None:
    """Deterministic seeding across stdlib, numpy, and (if present) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def repo_root() -> Path:
    # src/rdtabxai/io_utils.py -> repo root is three levels up
    return Path(__file__).resolve().parents[2]


def resolve(cfg: dict, key: str) -> Path:
    root = repo_root()
    return root / cfg["paths"][key]


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(out_dir: str | Path, record: dict) -> None:
    """Append a provenance record (source, url, timestamp, checksum, rowcount)."""
    out_dir = ensure_dir(out_dir)
    prov = out_dir / "provenance.jsonl"
    record = {"ts": datetime.datetime.utcnow().isoformat() + "Z", **record}
    with open(prov, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_json(obj, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
