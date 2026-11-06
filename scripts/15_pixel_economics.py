"""15 - Pixel-level (0.5 degree) CLIMRISK economics on language areas.

Replaces the national-average economics of script 05 by the cell-level present values
that CLIMRISK exports for its 47,506 economic-support cells.

Inputs (CLIMRISK HDF5)
  /maps/termination_pv/positive_loss_quantiles_support  [6 pathways, 3 quantiles, cells]
      pathways: full_no_srm (H), full_srm_no_failure (S),
                full_srm_termination_restricted_ku, full_srm_termination_local_ku (S+T),
                termination_premium_restricted_ku, termination_premium_local_ku (premium)
      quantiles: p17, p50, p83 across the 60 members, computed cell by cell
  /maps/economic_pv/{no_srm,srm}_{positive_loss,pct_pv_gdp}_quantiles  [3, 720, 360]
      used only to recover each cell's PV GDP (2026-2100, 1.5%) = 100 * loss / pct
  /grid/economic/{lat,lon,country_m49}

Unit conversions
  cell PV GDP g_c (billion USD 2005 PPP); cell loss L_c; intensity = 100 * L / g  (% of local PV GDP)
  a language-in-country polygon p covering fraction f_pc of cell c is given f_pc of the
  cell's GDP and loss (uniform within the cell); nearest-cell fallback polygons use f = 1.

Metrics (per polygon, per language, per country), for each quantile map:
  economy-weighted   I = 100 * sum_c f L_c / sum_c f g_c        (share of the area's economy)
  territory-weighted I = sum_c w_km2 (100 L_c / g_c) / sum_c w_km2   (typical place)
  outcomes: H, S, ST (local and restricted), premium (local, restricted),
            benefit of uninterrupted SRM over no SRM  B = I_H - I_S

Caveats: only cell-wise quantiles are available (no member-level cells); sums of cell
medians reproduce CLIMRISK country medians (checked below). No mitigation pathway exists
at cell level.

Outputs
  data/interim/pixel_cells.parquet     cell table (CLIMRISK-derived)
  data/interim/pixel_poly.parquet      polygon metrics (restricted, Ethnologue-derived)
  data/interim/pixel_lang.parquet      language metrics (restricted)
  results/tables/pixel_country_intensity.csv   national intensities (public)
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P, ensure_dirs, record  # noqa: E402

ensure_dirs()
I = P("interim")
T = P("results")
Q = ["p17", "p50", "p83"]

# ------------------------------------------------------------------ cells
f = h5py.File(P("climrisk_h5"), "r")
lat = f["/grid/economic/lat"][:]
lon = f["/grid/economic/lon"][:]
m49 = f["/grid/economic/country_m49"][:]
col = np.round((lon + 180) / 0.5).astype(int)
row = np.round((lat + 90) / 0.5).astype(int)
grid_id = row * 720 + col + 1                      # same convention as script 03

names = [p.decode() if isinstance(p, bytes) else p for p in f["/maps/termination_pv/pathway"][:]]
tp = f["/maps/termination_pv/positive_loss_quantiles_support"][:].astype(float)   # 6x3xN
key = {"full_no_srm": "H", "full_srm_no_failure": "S",
       "full_srm_termination_local_ku": "STloc", "full_srm_termination_restricted_ku": "STres",
       "termination_premium_local_ku": "Ploc", "termination_premium_restricted_ku": "Pres"}

# cell PV GDP from the two %-of-GDP maps (identical up to rounding; median of 6 estimates)
est = []
for tag in ["no_srm", "srm"]:
    L = f[f"/maps/economic_pv/{tag}_positive_loss_quantiles"][:][:, col, row]
    pct = f[f"/maps/economic_pv/{tag}_pct_pv_gdp_quantiles"][:][:, col, row]
    with np.errstate(divide="ignore", invalid="ignore"):
        est.append(np.where(pct > 0, 100 * L / pct, np.nan))
gdp = np.nanmedian(np.vstack(est), axis=0)

# country lookup
iso = np.array([x.decode() if isinstance(x, bytes) else x for x in f["/country/iso3"][:]])
cm49 = f["/country/m49"][:]
iso_of = dict(zip(cm49, iso))
cells = pd.DataFrame({"grid_id": grid_id, "lat": lat, "lon": lon,
                      "iso3": [iso_of.get(m, None) for m in m49], "gdp": gdp})
for j, n in enumerate(names):
    for q in range(3):
        cells[f"{key[n]}_{Q[q]}"] = tp[j, q]

# land cell areas
def cell_area(lat_c):
    """Exact area (km2) of a 0.5 x 0.5 degree cell centred at lat_c on a 6371 km sphere."""
    p1, p2 = np.radians(lat_c - 0.25), np.radians(lat_c + 0.25)
    return 6371.0 ** 2 * np.radians(0.5) * (np.sin(p2) - np.sin(p1))


cells["area_km2"] = cell_area(cells.lat.to_numpy())
la = pd.Series(f["/grid/land/cell_area_km2"][:], index=f["/grid/land/grid_id_0p5"][:])
record("pixel.cell_area_formula_vs_climrisk_max_rel_diff",
       float(np.nanmax(np.abs(cells.grid_id.map(la) / cells.area_km2 - 1))))

# validation: cell GDP and median losses add up to CLIMRISK country totals
G = pd.Series(f["/country/economic_pv/gdp_2026_2100_billion_usd2005_ppp"][:], index=iso)
Lc = pd.Series(np.median(f["/country/termination_pv/full_no_srm/positive_loss_by_member"][:], 0), index=iso)
chk = cells.groupby("iso3").agg(gdp=("gdp", "sum"), L=("H_p50", "sum"))
r_gdp = (chk.gdp / G.reindex(chk.index)).replace([np.inf, -np.inf], np.nan)
r_L = (chk.L / Lc.reindex(chk.index)).replace([np.inf, -np.inf], np.nan)
record("pixel.n_cells", len(cells))
record("pixel.share_cells_gdp_recovered", float(np.isfinite(cells.gdp).mean()))
record("pixel.share_world_gdp_recovered", float(cells.gdp.sum() / G.sum()))
record("pixel.country_gdp_ratio_median", float(r_gdp.median()))
record("pixel.country_gdp_ratio_p5_p95", [float(r_gdp.quantile(.05)), float(r_gdp.quantile(.95))])
record("pixel.country_loss_sum_of_cell_medians_ratio_median", float(r_L.median()))
record("pixel.country_loss_sum_of_cell_medians_ratio_p25_p75",
       [float(r_L.quantile(.25)), float(r_L.quantile(.75))])

cells = cells[np.isfinite(cells.gdp) & (cells.gdp > 0)].copy()
OUT = ["H", "S", "STloc", "STres", "Ploc", "Pres"]
for o in OUT:
    for q in Q:
        cells[f"I_{o}_{q}"] = 100 * cells[f"{o}_{q}"] / cells.gdp
for q in Q:
    cells[f"I_B_{q}"] = cells[f"I_H_{q}"] - cells[f"I_S_{q}"]
cells.to_parquet(I / "pixel_cells.parquet", index=False)

# ------------------------------------------------------------------ national intensities
nat = {}
for q in Q:
    g = cells.groupby("iso3")
    for o in OUT:
        nat[f"I_{o}_{q}"] = 100 * g[f"{o}_{q}"].sum() / g.gdp.sum()
    nat[f"I_B_{q}"] = nat[f"I_H_{q}"] - nat[f"I_S_{q}"]
    for o in ["Ploc", "B"]:
        x = cells[f"I_{o}_{q}"] * cells.area_km2
        nat[f"Iterr_{o}_{q}"] = x.groupby(cells.iso3).sum() / cells.area_km2.groupby(cells.iso3).sum()
nat = pd.DataFrame(nat)
nat["gdp_pv"] = cells.groupby("iso3").gdp.sum()
nat.index.name = "iso3"
nat.to_csv(T / "pixel_country_intensity.csv")

# ------------------------------------------------------------------ polygons and languages
pe = pd.read_parquet(I / "poly_exposure.parquet")
lm = pd.read_parquet(I / "lang_metrics.parquet")
pe = pe[pe.ISO.isin(lm.ISO) & pe.land_w_km2.notna() & ~pe.sign_language].copy()
w = pd.read_parquet(I / "poly_cell_weights.parquet")
w = w[w.pid.isin(pe.pid)].copy()
w["f"] = w.coverage.fillna(1.0)                    # nearest-cell fallback: use the cell's intensity
w["CTRY3"] = w.pid.map(pe.set_index("pid").CTRY3)
cols = ["grid_id", "gdp", "area_km2"] + [f"{o}_{q}" for o in OUT for q in Q] \
       + [f"I_{o}_{q}" for o in OUT + ["B"] for q in Q]
record("pixel.share_polygon_land_weight_on_gdp_cells",
       float(w[w.grid_id.isin(cells.grid_id)].w_km2.sum() / w.w_km2.sum()))

# Cells outside CLIMRISK's economic support (mostly coastal and island cells that CLIMRISK
# left without a country code, -9999) carry no GDP. They take the intensities of the nearest
# supported cell in the polygon's own country (any country if none), within 150 km, with a
# GDP equal to the donor's GDP density times the cell's land area.
land = pd.DataFrame({"grid_id": f["/grid/land/grid_id_0p5"][:], "lat": f["/grid/land/lat"][:],
                     "lon": f["/grid/land/lon"][:]}).set_index("grid_id")
land["area"] = cell_area(land.lat.to_numpy())


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


on = w[w.grid_id.isin(cells.grid_id)].merge(cells[cols], on="grid_id")
on["imputed"] = False
off = w[~w.grid_id.isin(cells.grid_id)].copy()
off = off.join(land, on="grid_id")
donor_rows = []
C = cells.set_index("grid_id")
for ctry, part in off.groupby(off.CTRY3.fillna("?")):
    pool = cells[cells.iso3 == ctry]
    if len(pool) == 0:
        pool = cells
    tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(xyz(pool.lat.values, pool.lon.values))
    d, k = tree.query(xyz(part.lat.values, part.lon.values))
    km = 2 * 6371.0 * np.arcsin(np.clip(d / 2, 0, 1))
    donor_rows.append(pd.DataFrame({"idx": part.index, "donor": pool.grid_id.values[k], "km": km}))
dn = pd.concat(donor_rows).set_index("idx")
off = off.join(dn)
record("pixel.n_polycell_rows_imputed", int(len(off)))
record("pixel.share_polygon_land_weight_imputed", float(off.w_km2.sum() / w.w_km2.sum()))
record("pixel.imputation_km_p50_p95_max", [float(off.km.median()), float(off.km.quantile(.95)), float(off.km.max())])
off = off[off.km <= 150].copy()
D = C.loc[off.donor.values]
scale = (D.gdp.values / D.area_km2.values) * off.area.values / D.gdp.values   # gdp_sub / gdp_donor
imp = pd.DataFrame({"grid_id": off.grid_id.values, "gdp": D.gdp.values * scale, "area_km2": off.area.values})
for o in OUT:
    for q in Q:
        imp[f"{o}_{q}"] = D[f"{o}_{q}"].values * scale
for o in OUT + ["B"]:
    for q in Q:
        imp[f"I_{o}_{q}"] = D[f"I_{o}_{q}"].values
off = pd.concat([off[["pid", "grid_id", "coverage", "w_km2", "f", "CTRY3"]].reset_index(drop=True),
                 imp.drop(columns="grid_id")], axis=1)
off["imputed"] = True
X = pd.concat([on, off], ignore_index=True)
X["fg"] = X.f * X.gdp


def aggregate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)
    res = {"gdp_local": g.fg.sum(), "terr_km2": g.w_km2.sum(), "n_cells": g.grid_id.nunique(),
           "share_imputed": (df.w_km2 * df.imputed).groupby(df[by]).sum() / g.w_km2.sum()}
    for q in Q:
        for o in OUT:
            res[f"I_{o}_{q}"] = 100 * (df[f"{o}_{q}"] * df.f).groupby(df[by]).sum() / res["gdp_local"]
        res[f"I_B_{q}"] = res[f"I_H_{q}"] - res[f"I_S_{q}"]
        for o in ["Ploc", "Pres", "B"]:
            res[f"Iterr_{o}_{q}"] = (df[f"I_{o}_{q}"] * df.w_km2).groupby(df[by]).sum() / res["terr_km2"]
    return pd.DataFrame(res)


pp = aggregate(X, "pid")
pp = pe.set_index("pid")[["ISO", "CTRY3", "vitality", "vitality_LIC", "endangered", "egids_rank",
                          "Is_Indigenous", "annex_i", "L1_LIC"]].join(pp, how="inner")
pp.to_parquet(I / "pixel_poly.parquet")

X["ISO"] = X.pid.map(pe.set_index("pid").ISO)
pl = aggregate(X, "ISO")
pl = lm.set_index("ISO")[["Country_Code", "vitality", "end", "EGIDS", "annex_i", "L1_Users",
                          "econ_premium"]].join(pl, how="inner")
pl.to_parquet(I / "pixel_lang.parquet")

record("pixel.n_polygons", len(pp))
record("pixel.n_languages", len(pl))
record("pixel.lang_mean_premium_local_econ_p50", float(pl.I_Ploc_p50.mean()))
record("pixel.lang_mean_premium_local_terr_p50", float(pl.Iterr_Ploc_p50.mean()))
record("pixel.lang_mean_premium_national_from_script05", float(pl.econ_premium.mean()))
record("pixel.lang_mean_benefit_econ_p50", float(pl.I_B_p50.mean()))
print(f"cells {len(cells):,}  polygons {len(pp):,}  languages {len(pl):,}")
print(pl[["I_Ploc_p50", "Iterr_Ploc_p50", "econ_premium", "I_B_p50"]].describe().round(3))
