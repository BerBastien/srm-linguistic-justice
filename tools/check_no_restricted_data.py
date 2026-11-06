"""Run before every `git push`: fail if the repository holds licensed or restricted data.

The Ethnologue 28 datasets (SIL International) and the CLIMRISK-SRM inputs are licensed and must never
be committed. This repository may contain code, configuration, the public Glottolog 5.3 tables
(CC BY 4.0), Natural Earth and OWID files, and aggregated results only.

Standard library only.  Usage:  python tools/check_no_restricted_data.py   (or: make check-public)
Exit code 1 if anything restricted is found.
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_EXT = {".parquet", ".npz", ".npy", ".h5", ".hdf5", ".nc", ".gdb", ".gpkg", ".shp", ".shx", ".dbf",
            ".tab", ".xlsx", ".xls", ".zip", ".gz", ".7z", ".tif", ".tiff", ".feather", ".pkl", ".sqlite", ".db"}
LANG_COLS = {"iso", "iso_639", "iso639p3code_eth", "language_code", "language_name", "uninverted_name",
             "lic_id", "l1_users", "all_users", "l2_users", "is_indigenous", "is_primary", "is_established"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}

# only files git would see, if this is a git repository; otherwise every file
try:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    files = [ROOT / x for x in out if x]
    mode = "files git would commit"
except Exception:
    files = [f for f in ROOT.rglob("*") if (f.is_file() or f.is_symlink()) and not (SKIP_DIRS & set(f.relative_to(ROOT).parts))]
    mode = "all files"

problems, warnings = [], []
for f in files:
    rel = f.relative_to(ROOT).as_posix()
    if f.is_symlink():
        problems.append(f"symbolic link (may point to licensed data): {rel}")
        continue
    if not f.is_file():
        continue
    if rel.startswith(("data/raw/ethnologue/", "data/raw/climrisk/")) and f.name != ".gitkeep":
        problems.append(f"licensed input: {rel}")
    if rel.startswith("data/interim/") and f.name != "README.md":
        problems.append(f"restricted intermediate product: {rel}")
    low = f.name.lower()
    if low.startswith(("table_of_", "e28_")) or "ethnologue" in low and f.suffix.lower() not in (".py", ".md"):
        problems.append(f"Ethnologue-like file name: {rel}")
    if f.suffix.lower() in DATA_EXT:
        problems.append(f"data file type {f.suffix} not allowed here: {rel}")
    if f.stat().st_size > 50e6:
        problems.append(f"file larger than 50 MB: {rel}")
    if f.name == ".DS_Store":
        warnings.append(f"macOS metadata file (add to .gitignore): {rel}")
    if f.suffix.lower() == ".csv" and not rel.startswith("data/external/"):
        with open(f, newline="", encoding="utf-8", errors="ignore") as fh:
            rows = csv.reader(fh)
            header = [h.strip().lower() for h in next(rows, [])]
            bad = LANG_COLS & set(header)
            if bad:
                problems.append(f"language-level columns {sorted(bad)} in {rel}")
            if "n" in header:
                i = header.index("n")
                small = 0
                for r in rows:
                    try:
                        small += float(r[i]) < 2
                    except (ValueError, IndexError):
                        pass
                if small:
                    problems.append(f"{small} rows describe fewer than two languages in {rel}")

print(f"checked {len(files)} {mode} in {ROOT.name}")
for w in warnings:
    print("  warning:", w)
if problems:
    print("RESTRICTED CONTENT FOUND - do not push:")
    for p in problems:
        print("  ", p)
    sys.exit(1)
print("OK: no licensed or restricted data found")
