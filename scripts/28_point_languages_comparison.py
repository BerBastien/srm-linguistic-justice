"""28 - Robustness: languages without an Ethnologue polygon added at their Glottolog point (v5).

The variant is the full pipeline (scripts 02a-08, 15-17, 19, 21, 23, 27) run in a separate copy of
the repository with `add_point_languages: true` in config/config.yaml. Each language listed in
Ethnologue 28 without a polygon (and neither sign language nor dormant) enters at its Glottolog 5.3
point: the 0.5-degree cell containing the point, or the nearest land cell within 150 km if the point
lies at sea. This script copies the variant's key results into the main ledger under `points.*`
and writes results/tables/v5_point_languages.csv (main vs variant).

Usage:  python scripts/28_point_languages_comparison.py <variant repository root>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P, record, ledger  # noqa: E402

var_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("TMPDIR", "/tmp")) / "srm_points_variant"
V = {k: x["value"] for k, x in json.loads((var_root / "results/tables/numbers.json").read_text()).items()}
M = {k: x["value"] for k, x in ledger().items()}

KEYS = {
    "languages": "overlay.n_languages_living_nonsign",
    "noninst_languages": "v5.sample.n_noninst",
    "noninst_drier": "base.drier.non_institutional_languages.ensemble_mean_climate",
    "noninst_wetter": "base.wetter.non_institutional_languages.ensemble_mean_climate",
    "noninst_dry_to_wet": "base.odds.non_institutional_languages.dry_to_wet",
    "noninst_members_gt_land": "base.drier.non_institutional_languages.n_members_ratio_gt1",
    "hotspots_drier": "base.drier.linguistic_diversity_hotspots.ensemble_mean_climate",
    "people_drier": "base.drier.people.ensemble_mean_climate",
    "lineage_languages": "v5.class.glottolog.languages",
    "lineages": "v5.class.glottolog.lineages",
    "families": "v5.class.glottolog.families",
    "isolates": "v5.class.glottolog.isolates",
    "lineages_drier": "v5.class.glottolog.drier",
    "lineages_wetter": "v5.class.glottolog.wetter",
    "lineages_dry_to_wet": "v5.class.glottolog.dry_to_wet",
    "lineages_mostly_drier": "v5.class.glottolog.mostly_drier",
    "lineages_mostly_drier_p17": "v5.class.glottolog.mostly_drier_p17",
    "lineages_mostly_drier_p83": "v5.class.glottolog.mostly_drier_p83",
    "isolates_mostly_drier": "v5.class.glottolog.isolates_mostly_drier",
    "lineages_members_gt_land": "v5.class.glottolog.n_members_gt_land",
    "lineages_members_gt_people": "v5.class.glottolog.n_members_gt_people",
    "lineages_members_gt_economy": "v5.class.glottolog.n_members_gt_economy",
    "deep_tropics_share_languages": "v3.deep_tropics_share.languages",
    "ED_weighted_share_drier": "heritage.tree.all.ED_weighted_share_drier",
}
rows = []
for name, k in KEYS.items():
    rows.append({"quantity": name, "key": k, "main": M.get(k), "with_point_languages": V.get(k)})
    record(f"points.{name}", V.get(k))
for k in ("points.n_candidates_without_polygon", "points.n_without_glottolog_point", "points.n_point_languages_added"):
    record(k, V.get(k))
record("points.n_point_languages_in_sample", V["overlay.n_languages_living_nonsign"] - M["overlay.n_languages_living_nonsign"])
pd.DataFrame(rows).to_csv(P("results") / "v5_point_languages.csv", index=False)
print(pd.DataFrame(rows).to_string())

# where the added languages are, and how they fare (mean share of homeland drier / wetter, ensemble mean)
hl = pd.read_parquet(var_root / "data/interim/heritage_lang.parquet")
cwv = pd.read_parquet(var_root / "data/interim/glottolog_crosswalk.parquet").set_index("ISO")
pt = hl[hl.point_language.astype(bool)].copy()
pt["macro"] = pt.ISO.map(cwv.g_macroarea).fillna("none").str.split(";").str[0]
record("points.added_by_macroarea", pt.macro.value_counts().to_dict())
record("points.added_mean_f_dry", float(pt.f_dry.mean()))
record("points.added_mean_f_dry_by_macroarea", pt.groupby("macro").f_dry.mean().round(4).to_dict())
record("points.added_noninst_share", float(pt.noninst.mean()))
print(pt.groupby("macro").f_dry.agg(["size", "mean"]).round(3))
