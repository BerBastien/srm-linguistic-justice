"""02 - Prepare Ethnologue 28 language polygons and attributes.

Unit of observation: one polygon per language-in-country (LIC), identified by
``ID = <ISO 639-3>-<ISO 3166 alpha-3>``. Polygons represent the *current home
areas of indigenous languages* (Ethnologue GIS documentation).

Steps
  1. read the File Geodatabase layer ``Lang_Polys`` (EPSG:4326)
  2. repair invalid geometries (shapely.make_valid), keep polygonal parts only
  3. attach attributes: Table_of_LICs (by ISO + country alpha-2) and
     Table_of_Languages (by ISO)
  4. derive classifications (vitality, endangered, Annex I, size class, family)
  5. flag sign languages (country-wide polygons, not territorial communities)

Output (RESTRICTED - Ethnologue licence forbids redistribution of raw data):
  data/interim/eth_polygons.parquet
  data/interim/eth_languages.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import (CFG, EGIDS_RANK, P, annex_i, ensure_dirs, endangered,  # noqa: E402
                            record, vitality)

ensure_dirs()

# ---------------------------------------------------------------- attribute tables
rd = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
lang = pd.read_csv(P("eth_languages"), **rd)
lics = pd.read_csv(P("eth_lics"), **rd)
ctry = pd.read_csv(P("eth_countries"), **rd)
for c in ("L1_Users", "All_Users", "Countries", "Latitude", "Longitude", "Dls_Score", "YLSD"):
    lang[c] = pd.to_numeric(lang[c], errors="coerce")
for c in ("L1_Users", "L2_Users", "All_Users"):
    lics[c] = pd.to_numeric(lics[c], errors="coerce")
ctry["Population"] = pd.to_numeric(ctry["Population"], errors="coerce")

record("eth.n_languages_table", len(lang))
record("eth.n_lics_table", len(lics))
record("eth.n_living_languages", int((lang.EGIDS != "x10").sum()))
for v in ("Institutional", "Stable", "In trouble", "Dying", "Extinct"):
    record(f"eth.n_languages_{v.replace(' ', '_')}",
           int((lang.EGIDS.map(vitality) == v).sum()))

# ---------------------------------------------------------------- polygons
gdf = gpd.read_file(P("eth_polygons_gdb"), layer="Lang_Polys", engine="pyogrio")
record("eth.n_polygons_raw", len(gdf))
n_invalid = int((~gdf.is_valid).sum())
record("eth.n_polygons_invalid_repaired", n_invalid)
geom = shapely.make_valid(gdf.geometry.values)


def polygonal(g):
    """Keep only (multi)polygon parts of a repaired geometry."""
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        return None
    return shapely.unary_union(parts)


gdf["geometry"] = [polygonal(g) for g in geom]
gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

gdf["ISO"] = gdf["ID"].str.split("-").str[0].str.lower().str.strip()
gdf["CTRY3"] = gdf["ID"].str.split("-").str[1].str.upper().str.strip()
gdf["CTRY2"] = gdf["COUNTRY_ISO_ALPHA_2"].str.upper().str.strip()

# geodesic polygon area (km2) on the WGS84 ellipsoid via an equal-area projection
gdf["poly_area_km2"] = gdf.to_crs("+proj=eck4 +datum=WGS84").area / 1e6
gdf["point_language"] = False

# ---------------------------------------------------------------- v5: point languages (robustness)
# Living spoken languages listed in Ethnologue 28 without any polygon enter, if requested, at their
# Glottolog 5.3 point (script 02a): a tiny square around the point, so that script 03 assigns the
# 0.5-degree cell containing it, or the nearest land cell within nearest_cell_max_km if it lies at sea.
# Their homeland has no area (poly_area_km2 = NaN); they are left out of homeland-size analyses.
EXCL = set(CFG["analysis"].get("exclude_egids", []))
if CFG["analysis"].get("add_point_languages", False):
    cw = pd.read_parquet(P("interim") / "glottolog_crosswalk.parquet")
    has_poly = set(gdf["ISO"])
    sign = lang.Family.fillna("").str.contains("Sign language", case=False)
    cand = lang[(lang.EGIDS != "x10") & ~lang.EGIDS.isin(EXCL) & ~sign &
                ~lang.ISO_639.str.lower().isin(has_poly)][["ISO_639", "Country_Code"]].copy()
    cand["ISO"] = cand.ISO_639.str.lower()
    cand = cand.merge(cw[["ISO", "g_lat", "g_lon"]], on="ISO", how="left")
    record("points.n_candidates_without_polygon", len(cand))
    record("points.n_without_glottolog_point", int(cand.g_lat.isna().sum()))
    cand = cand[cand.g_lat.notna() & cand.g_lon.notna()]
    a2a3 = gdf.drop_duplicates("COUNTRY_ISO_ALPHA_2").set_index("COUNTRY_ISO_ALPHA_2").COUNTRY_ISO_3DIGIT
    ne = gpd.read_file(P("ne_countries"))[["ISO_A2", "ISO_A3"]].drop_duplicates("ISO_A2").set_index("ISO_A2").ISO_A3
    c3 = cand.Country_Code.map(a2a3).fillna(cand.Country_Code.map(ne)).fillna("UNK")
    d = 1e-4                                            # degrees; about 10 m
    pts = gpd.GeoDataFrame({
        "ID": cand.ISO.str.upper() + "-" + c3, "ISO": cand.ISO.to_numpy(), "CTRY3": c3.to_numpy(),
        "CTRY2": cand.Country_Code.str.upper().to_numpy(), "COUNTRY_ISO_ALPHA_2": cand.Country_Code.to_numpy(),
        "COUNTRY_ISO_3DIGIT": c3.to_numpy(), "ETH_LG_R": "", "poly_area_km2": np.nan, "point_language": True},
        geometry=[shapely.box(x - d, y - d, x + d, y + d) for x, y in zip(cand.g_lon, cand.g_lat)], crs=gdf.crs)
    gdf = pd.concat([gdf, pts], ignore_index=True)
    record("points.n_point_languages_added", len(pts))

# ---------------------------------------------------------------- joins
lic_cols = ["ISO_639", "Country_Code", "Is_Primary", "Is_Indigenous", "Is_Established",
            "L1_Users", "L2_Users", "All_Users", "EGIDS", "Function_Code"]
L = lics[lic_cols].rename(columns={"ISO_639": "ISO", "Country_Code": "CTRY2",
                                    "L1_Users": "L1_LIC", "L2_Users": "L2_LIC",
                                    "All_Users": "All_LIC", "EGIDS": "EGIDS_LIC"})
L = L.drop_duplicates(["ISO", "CTRY2"])
gdf = gdf.merge(L, on=["ISO", "CTRY2"], how="left")
lang_cols = ["ISO_639", "Language_Name", "Country_Code", "Region_Name", "Area", "L1_Users",
             "All_Users", "Countries", "Family", "Classification", "Latitude", "Longitude",
             "EGIDS", "Is_Written", "Dls_Level", "Dls_Score"]
LG = lang[lang_cols].rename(columns={"ISO_639": "ISO", "Country_Code": "PrimaryCountry",
                                     "L1_Users": "L1_total", "All_Users": "All_total"})
gdf = gdf.merge(LG, on="ISO", how="left")
record("eth.n_polygons_without_language_record", int(gdf.Language_Name.isna().sum()))
record("eth.n_polygons_without_lic_record", int(gdf.Is_Indigenous.isna().sum()))

# ---------------------------------------------------------------- classifications
gdf["vitality"] = gdf.EGIDS.map(vitality)
gdf["vitality_LIC"] = gdf.EGIDS_LIC.map(vitality)
gdf["endangered"] = gdf.EGIDS.map(endangered)
gdf["egids_rank"] = gdf.EGIDS.map(EGIDS_RANK)
gdf["annex_i"] = gdf.CTRY2.map(lambda c: annex_i(c, True))
gdf["annex_i_strict"] = gdf.CTRY2.map(lambda c: annex_i(c, False))
fam = gdf.Family.fillna("")
gdf["sign_language"] = fam.str.contains("Sign language", case=False) | \
    gdf.ETH_LG_R.fillna("").str.contains("Sign Language", case=False)

gdf = gdf.drop(columns=["Shape_Length", "Shape_Area"], errors="ignore")
# v5: exclude dormant languages (EGIDS 9: no first-language speakers) from the analysed polygons
excl = gdf.EGIDS.isin(EXCL)
record("eth.n_polygons_excluded_egids", int(excl.sum()))
record("eth.languages_excluded_egids_with_polygon", sorted(gdf[excl].ISO.unique().tolist()))
gdf = gdf[~excl].copy()
gdf.to_parquet(P("interim") / "eth_polygons.parquet", index=False)

# language-level attribute table (one row per ISO 639-3)
lang["vitality"] = lang.EGIDS.map(vitality)
lang["endangered"] = lang.EGIDS.map(endangered)
lang["egids_rank"] = lang.EGIDS.map(EGIDS_RANK)
lang["sign_language"] = lang.Family.fillna("").str.contains("Sign language", case=False)
lang = lang.rename(columns={"ISO_639": "ISO"})
lang.to_parquet(P("interim") / "eth_languages.parquet", index=False)

# indigenous status of polygon LICs (Ethnologue polygons should all be indigenous)
record("eth.polys_is_indigenous_T", int((gdf.Is_Indigenous == "T").sum()))
record("eth.polys_is_indigenous_F", int((gdf.Is_Indigenous == "F").sum()))
record("eth.polys_sign_language", int(gdf.sign_language.sum()))
record("eth.n_polygons_clean", len(gdf))
record("eth.n_languages_with_polygon", int(gdf.ISO.nunique()))
print(gdf[["vitality", "Is_Indigenous", "annex_i", "sign_language"]].value_counts(dropna=False)
      .head(30))
print("polygons:", len(gdf), "languages:", gdf.ISO.nunique())
