"""00 - Check that every required input is present and identical to the version
used in the article (SHA-256 checksums in config/input_checksums.json).

Run with --write to (re)create the checksum file from the inputs on disk.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, ROOT  # noqa: E402

KEYS = ["climrisk_h5", "climrisk_grid_crosswalk", "climrisk_country_crosswalk",
        "eth_languages", "eth_lics", "eth_countries", "eth_polygons_gdb",
        "ne_land", "ne_coast", "ne_countries", "owid_co2"]
EXTRA = ["data/raw/climrisk/termination_country_PV_by_member_p005.csv",
         "data/raw/climrisk/termination_country_PV_by_member_p01.csv",
         "data/raw/climrisk/termination_country_PV_by_member_p02.csv"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


targets = {k: P(k) for k in KEYS}
targets.update({e: ROOT / e for e in EXTRA})
ck_file = ROOT / "config" / "input_checksums.json"
if "--write" in sys.argv:
    out = {k: sha(p) for k, p in targets.items()}
    ck_file.write_text(json.dumps(out, indent=1))
    print("checksums written:", ck_file)
    sys.exit(0)

ref = json.loads(ck_file.read_text()) if ck_file.exists() else {}
ok = True
for k, p in targets.items():
    if not p.exists():
        print(f"MISSING  {k:32s} {p}")
        ok = False
        continue
    if k in ref:
        same = sha(p) == ref[k]
        print(f"{'OK      ' if same else 'CHANGED '} {k:32s} {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
        ok &= same
    else:
        print(f"PRESENT  {k:32s} (no reference checksum)")
if not ok:
    print("\nSome inputs are missing or differ from the article version. See data/raw/README.md.")
    sys.exit(1)
print("\nAll inputs present and identical to the article version.")
