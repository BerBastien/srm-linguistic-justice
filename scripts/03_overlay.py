"""03 - Exact polygon x grid overlay and language-level exposure.

For every LIC polygon p and every 0.5-degree CLIMRISK cell i we compute the exact
fraction f_pi of the cell covered by the polygon (exactextract). The area weight
is  a_pi = f_pi * A_i  with A_i the spherical cell area (km2) from CLIMRISK.
Only CLIMRISK land cells carry climate values; ocean parts of a cell are not
represented, so weights are renormalised over land cells:

      X_p = sum_i a_pi X_i / sum_i a_pi        (i in land cells)

Polygons with no land-cell coverage (tiny islands absent from the 0.5-degree land
mask) are assigned the nearest land cell to their representative point, if it
lies within ``nearest_cell_max_km``; otherwise they are dropped and counted.

Language level (ISO 639-3): area-weighted mean over the language's LIC polygons.

Outputs (RESTRICTED):
  data/interim/poly_cell_weights.parquet   long table (polygon, cell, weight)
  data/interim/poly_exposure.parquet       polygon-level exposures
  data/interim/lang_exposure.parquet       language-level exposures + attributes
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, ensure_dirs, record  # noqa: E402

ensure_dirs()
I = P("interim")
grid = pd.read_parquet(I / "grid_fields.parquet")
polys = gpd.read_parquet(I / "eth_polygons.parquet")
polys["pid"] = np.arange(len(polys))

# ---------------------------------------------------------------- id raster
# CLIMRISK cell centres: lon = -180.0, -179.5, ..., 179.5 ; lat = -90.0, ..., 89.5
# grid_id = row_from_south * 720 + col + 1  (verified against the crosswalk table)
# We add a 721st column (centre 180.0) that duplicates column 0 so that polygon
# parts between 179.75 and 180 are captured, then fold it back.
ncol, nrow = 721, 360
rows_from_south = np.arange(nrow)[::-1]                 # tif row 0 is the northern row
cols = np.arange(ncol) % 720
ids = rows_from_south[:, None] * 720 + cols[None, :] + 1
transform = from_origin(-180.25, 89.75, 0.5, 0.5)
tif = I / "grid_id_0p5.tif"
with rasterio.open(tif, "w", driver="GTiff", height=nrow, width=ncol, count=1, dtype="float64",
                   crs="EPSG:4326", transform=transform) as dst:
    dst.write(ids.astype("float64"), 1)

# sanity check: centre of id 721 must be (-89.5, -180.0)
cw = grid.set_index("grid_id")
assert abs(cw.loc[721, "lat"] + 89.5) < 1e-9 and abs(cw.loc[721, "lon"] + 180.0) < 1e-9

# ---------------------------------------------------------------- exact coverage
print("running exactextract on", len(polys), "polygons ...")
res = exact_extract(str(tif), polys[["pid", "geometry"]], ["values", "coverage"],
                    include_cols=["pid"], output="pandas", strategy="feature-sequential")
long = res.explode(["values", "coverage"], ignore_index=True)
long = long.dropna(subset=["values"])
long["grid_id"] = long["values"].astype(float).astype(int)
long["coverage"] = long["coverage"].astype(float)
long = long.groupby(["pid", "grid_id"], as_index=False)["coverage"].sum()
area = cw["area_km2"]
long = long[long.grid_id.isin(area.index)]                # keep land cells only
long["w_km2"] = long["coverage"] * long["grid_id"].map(area)
long = long[long.w_km2 > 0]

covered = set(long.pid.unique())
missing = polys[~polys.pid.isin(covered)].copy()
record("overlay.n_polys_with_land_coverage", len(covered))
record("overlay.n_polys_without_land_coverage", len(missing))

# nearest-land-cell fallback for small islands / slivers
if len(missing):
    R = 6371.0
    lat0, lon0 = np.radians(cw["lat"].to_numpy()), np.radians(cw["lon"].to_numpy())
    xyz = np.c_[np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]
    tree = cKDTree(xyz)
    rp = missing.geometry.representative_point()
    la, lo = np.radians(rp.y.to_numpy()), np.radians(rp.x.to_numpy())
    q = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    d, k = tree.query(q)
    dist_km = 2 * R * np.arcsin(np.clip(d / 2, 0, 1))
    ok = dist_km <= CFG["analysis"]["nearest_cell_max_km"]
    fb = pd.DataFrame({"pid": missing.pid.to_numpy()[ok], "grid_id": cw.index.to_numpy()[k[ok]],
                       "coverage": np.nan, "w_km2": 1.0})
    long = pd.concat([long, fb], ignore_index=True)
    record("overlay.n_polys_nearest_cell_fallback", int(ok.sum()))
    record("overlay.n_polys_dropped_no_land", int((~ok).sum()))
    polys["nearest_cell_km"] = np.nan
    polys.loc[missing.index[ok], "nearest_cell_km"] = dist_km[ok]
polys["fallback_nearest_cell"] = polys["nearest_cell_km"].notna()
long.to_parquet(I / "poly_cell_weights.parquet", index=False)

# ---------------------------------------------------------------- polygon exposures
vars_ = []
for h in CFG["analysis"]["horizons"]:
    vars_ += [f"{v}_{h}" for v in ("TH", "TS", "TM", "R", "CS", "CM", "PH", "PS", "PM", "RP")]
    vars_ += [f"TM_A1_{h}", f"dT_{h}_p17", f"dT_{h}_p50", f"dT_{h}_p83",
              f"dP_{h}_p17", f"dP_{h}_p50", f"dP_{h}_p83"]
X = cw[vars_]
L = long.join(X, on="grid_id")
wsum = L.groupby("pid")["w_km2"].sum()
agg = L[vars_].multiply(L["w_km2"], axis=0).groupby(L["pid"]).sum().div(wsum, axis=0)
agg["land_w_km2"] = wsum
# share of the polygon's land area where SRM leaves residual warming / drying
for h in CFG["analysis"]["horizons"]:
    agg[f"share_R_pos_{h}"] = (L[f"R_{h}"] > 0).astype(float).mul(L.w_km2).groupby(L.pid).sum() / wsum
    agg[f"share_RP_neg_{h}"] = (L[f"RP_{h}"] < 0).astype(float).mul(L.w_km2).groupby(L.pid).sum() / wsum
# area-weighted mean latitude (used to describe the mechanism)
agg["lat_mean"] = L["grid_id"].map(cw["lat"]).mul(L.w_km2).groupby(L.pid).sum() / wsum
agg["n_cells"] = L.groupby("pid")["grid_id"].nunique()
pe = polys.drop(columns="geometry").merge(agg, left_on="pid", right_index=True, how="left")
pe["rep_lon"] = polys.geometry.representative_point().x.to_numpy()
pe["rep_lat"] = polys.geometry.representative_point().y.to_numpy()
pe.to_parquet(I / "poly_exposure.parquet", index=False)

# ---------------------------------------------------------------- language level
pe_ok = pe[pe.land_w_km2.notna()].copy()
num = vars_ + [c for c in pe_ok.columns if c.startswith("share_")] + ["lat_mean"]
wl = pe_ok["poly_area_km2"].fillna(1.0).clip(lower=1e-6)          # point languages (v5 robustness): no area
lang_x = pe_ok[num].multiply(wl, axis=0).groupby(pe_ok["ISO"]).sum().div(
    wl.groupby(pe_ok["ISO"]).sum(), axis=0)
lang_x["territory_km2"] = pe_ok.groupby("ISO")["poly_area_km2"].sum(min_count=1)
lang_x["point_language"] = pe_ok.groupby("ISO")["point_language"].any()
lang_x["n_polys"] = pe_ok.groupby("ISO").size()
# Annex I: share of the language's territory in Annex I parties
lang_x["annex_i_share"] = pe_ok.annex_i.astype(float).mul(wl).groupby(pe_ok.ISO).sum() / \
    wl.groupby(pe_ok.ISO).sum()
lang_x["any_fallback"] = pe_ok.groupby("ISO")["fallback_nearest_cell"].any()
# countries of the language (for country-level joins): area share per country
cshare = (pe_ok.assign(w=wl).groupby(["ISO", "CTRY3"])["w"].sum() /
          wl.groupby(pe_ok["ISO"]).sum())
cshare.rename("country_share").reset_index().to_parquet(I / "lang_country_shares.parquet",
                                                         index=False)
# indigeneity: share of territory in LICs flagged Is_Indigenous == "T"
lang_x["indigenous_share"] = (pe_ok.Is_Indigenous == "T").astype(float).mul(wl).groupby(
    pe_ok.ISO).sum() / wl.groupby(pe_ok.ISO).sum()
lang_attr = pd.read_parquet(I / "eth_languages.parquet")
lx = lang_attr.merge(lang_x, left_on="ISO", right_index=True, how="inner")
lx["annex_i"] = lx["annex_i_share"] >= 0.5
lx.to_parquet(I / "lang_exposure.parquet", index=False)

record("overlay.n_languages_with_exposure", len(lx))
record("overlay.n_languages_living_nonsign",
       int(((lx.vitality != "Extinct") & lx.vitality.notna() & ~lx.sign_language).sum()))
print(lx.groupby("vitality")[["R_2100", "RP_2100", "CS_2100", "CM_2100"]].mean().round(3))
print("languages with exposure:", len(lx))
