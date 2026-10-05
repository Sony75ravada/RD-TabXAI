#!/usr/bin/env python3
"""Stage 01 — obtain raw sources.

If cfg['use_fixtures'] is True, generate schema-faithful synthetic files into
data_raw/ (offline demo). Otherwise download the real official files from the URLs
in cfg['sources'] (run this in an environment with outbound network access).
Records provenance (url, checksum, rows) for every file.
"""
import urllib.request
from _common import cfg_and_seed, raw_dir, LOG, write_provenance
from rdtabxai.io_utils import sha256

def main():
    cfg = cfg_and_seed()
    raw = raw_dir(cfg)
    if cfg.get("use_fixtures", True):
        from rdtabxai.data import fixtures
        info = fixtures.generate(raw, seed=cfg["seed"])
        LOG.info("Stage01: generated fixtures %s", info)
        write_provenance(raw, {"stage": "01", "mode": "fixtures", **info})
        return
    for name, url in cfg["sources"].items():
        dest = raw / url.split("/")[-1]
        LOG.info("downloading %s -> %s", url, dest)
        try:
            urllib.request.urlretrieve(url, dest)
            write_provenance(raw, {"stage": "01", "source": name, "url": url,
                                   "file": dest.name, "sha256": sha256(dest)})
        except Exception as e:
            LOG.error("FAILED %s: %s (download manually to %s)", name, e, dest)

if __name__ == "__main__":
    main()
