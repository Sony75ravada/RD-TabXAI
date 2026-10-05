#!/usr/bin/env python3
"""Verify that the exact public data release used in the study is installed.

The study uses the Human Phenotype Ontology (HPO) release 2025-01-16
(archived DOI: 10.5281/zenodo.14674339), obtained through the `pyhpo` 4.0.0
package, which bundles that release unchanged. This script checks the release
version stamped inside each file and its SHA-256 checksum.

    python scripts/00_verify_data_release.py
"""
import hashlib
import sys
from pathlib import Path

EXPECTED = {
    "hp.obo": "6b77de067eecc838319ce7650ed5bab0f92a502eabb160e6bc7c0238bc1548c5",
    "phenotype.hpoa": "8180403e2f5de0d8f41890e587d95077ce7f8bb8228d5d7b29dd358b70f0938c",
    "genes_to_phenotype.txt": "77d4c616780ac048a6766f958ec8f6f194cd216e2edd3944c1a0756b6f3e9a36",
}
RELEASE = "2025-01-16"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    try:
        import pyhpo
    except ImportError:
        print("pyhpo is not installed. Run: pip install pyhpo==4.0.0")
        return 1
    data = Path(pyhpo.__file__).parent / "data"
    ok = True

    obo_head = (data / "hp.obo").read_text(encoding="utf-8", errors="ignore")[:2000]
    stamp = f"data-version: hp/releases/{RELEASE}"
    print(f"[{'OK ' if stamp in obo_head else 'FAIL'}] hp.obo release stamp: {stamp}")
    ok &= stamp in obo_head

    for name, want in EXPECTED.items():
        got = sha256(data / name)
        match = got == want
        ok &= match
        print(f"[{'OK ' if match else 'FAIL'}] {name:24s} sha256={got}")

    print("\nAll checks passed: HPO release 2025-01-16 (doi:10.5281/zenodo.14674339)."
          if ok else "\nMismatch: install pyhpo==4.0.0 to obtain the exact release used.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
