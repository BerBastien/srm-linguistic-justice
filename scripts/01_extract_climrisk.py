"""01 - Extract CLIMRISK-SRM fields and build the iso-GMST mitigation counterfactual.

Inputs (CLIMRISK-SRM research package, MAGICC7 SCEN7-H x CESM2-WACCM G6sulfur, N=60):
  * global temperature trajectories by member, no SRM (H) and SRM (S)
  * country climate by member (60 x 81 years x 187 countries)
  * 0.5-degree land maps: ensemble mean and p17/p50/p83 of T and P for 2050, 2100

The three worlds compared (m = ensemble member, k = pattern index, x = location):
  H  high-emission world, no SRM        T_H = beta_k(x) G_H,m(t)
  S  same emissions + SRM (G6sulfur)    T_S = beta_k(x) G_H,m + gamma_k(x) [G_H,m - G_S,m]
  M  mitigation to the SAME global mean T_M = beta_k(x) G_S,m(t)
     temperature as S, member by member (constructed here by pattern scaling)

Residual of SRM relative to mitigation:  R = T_S - T_M = (beta_k + gamma_k) * dG,
with dG = G_H - G_S the global cooling delivered by SRM.

Outputs
  data/interim/grid_fields.parquet           0.5-degree land fields (both horizons)
  data/interim/country_member_fields.npz     member-level country quantities
  results/tables/climrisk_validation.csv     checks of the pattern-scaling identities
  results/tables/global_trajectories.csv     member quantiles of G_H, G_S, dG
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, ensure_dirs, record  # noqa: E402

ensure_dirs()
H5 = P("climrisk_h5")
f = h5py.File(H5, "r")
dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]  # noqa: E731

years = f["time/year"][:]
GH = f["global/tas_no_srm_c_ref1900_by_member"][:]  # (60, 81)
GS = f["global/tas_srm_c_ref1900_by_member"][:]
dG = GH - GS
pidx = f["ensemble/pattern_index"][:]
n_mem, n_yr = GH.shape
horizons = CFG["analysis"]["horizons"]
hidx = {h: int(np.where(years == h)[0][0]) for h in horizons}

# ---------------------------------------------------------------- global
rows = []
for j, y in enumerate(years):
    r = {"year": int(y)}
    for name, arr in (("G_H", GH), ("G_S", GS), ("dG", dG)):
        q = np.percentile(arr[:, j], [17, 50, 83])
        r.update({f"{name}_p17": q[0], f"{name}_p50": q[1], f"{name}_p83": q[2],
                  f"{name}_mean": arr[:, j].mean()})
    rows.append(r)
glob = pd.DataFrame(rows)
glob.to_csv(P("results") / "global_trajectories.csv", index=False)
for h in horizons:
    j = hidx[h]
    for name, arr in (("G_H", GH), ("G_S", GS), ("dG", dG)):
        q = np.percentile(arr[:, j], [17, 50, 83])
        record(f"global.{name}.{h}.p17_p50_p83", q)
        record(f"global.{name}.{h}.mean", arr[:, j].mean())

# ---------------------------------------------------------------- country, member level
iso3 = dec(f["country/iso3"][:])
m49 = f["country/m49"][:]
TcH = f["country/climate/tas_no_srm_by_member"][:]  # (60, 81, 187)
TcS = f["country/climate/tas_srm_by_member"][:]
PcH = f["country/climate/pr_no_srm_by_member"][:]
PcS = f["country/climate/pr_srm_by_member"][:]
n_c = len(iso3)

# Pattern coefficients implied by the CLIMRISK construction. They must be
# constant in time for a given member: this is checked below.
with np.errstate(invalid="ignore", divide="ignore"):
    beta_t = TcH / GH[:, :, None]
    gamma_t = (TcS - TcH) / dG[:, :, None]      # T_S - T_H = gamma * dG; gamma < 0 where SRM cools
    pi_t = PcH / GH[:, :, None]

# use years from 2030 on, where |dG| is large enough for a stable ratio
late = years >= 2030
val = []
beta_dev = np.nanmax(np.abs(beta_t[:, late, :] - beta_t[:, [-1], :]))
gamma_dev = np.nanmax(np.abs(gamma_t[:, late, :] - gamma_t[:, [-1], :]))
pi_dev = np.nanmax(np.abs(pi_t[:, late, :] - pi_t[:, [-1], :]))
val += [("max |beta(t)-beta(2100)|, 2030-2100", beta_dev),
        ("max |gamma(t)-gamma(2100)|, 2030-2100", gamma_dev),
        ("max |pi(t)-pi(2100)|, 2030-2100", pi_dev)]

beta = beta_t[:, -1, :]    # (60, 187): one value per member; identical within pattern
gamma = gamma_t[:, -1, :]
pi = pi_t[:, -1, :]
# identical within pattern?
within = max(np.nanmax(np.nanstd(beta[pidx == k], axis=0)) for k in np.unique(pidx))
val.append(("max within-pattern std of beta across members", within))
has_clim = np.isfinite(beta).all(axis=0)
record("country.n_with_climate", int(has_clim.sum()))
record("country.n_total", n_c)

# member-level country residuals at each horizon
cm = {}
for h, j in hidx.items():
    TM = beta * GS[:, [j]]                    # (60, 187)
    PM = pi * GS[:, [j]]
    cm[f"TH_{h}"] = TcH[:, j, :]
    cm[f"TS_{h}"] = TcS[:, j, :]
    cm[f"TM_{h}"] = TM
    cm[f"R_{h}"] = TcS[:, j, :] - TM
    cm[f"PH_{h}"] = PcH[:, j, :]
    cm[f"PS_{h}"] = PcS[:, j, :]
    cm[f"PM_{h}"] = PM
    cm[f"RP_{h}"] = PcS[:, j, :] - PM
np.savez_compressed(P("interim") / "country_member_fields.npz", iso3=np.array(iso3), m49=m49,
                    pattern_index=pidx, beta=beta, gamma=gamma, pi=pi, GH=GH, GS=GS,
                    years=years, **cm)

# ---------------------------------------------------------------- grid level (ensemble means)
lat = f["grid/land/lat"][:]
lon = f["grid/land/lon"][:]
area = f["grid/land/cell_area_km2"][:]
gid = f["grid/land/grid_id_0p5"][:]
cm49 = f["grid/land/country_m49"][:]
reg = f["grid/land/region_id"][:]
hz = list(f["horizon/year"][:])
q = f["summary/quantile"][:]

grid = pd.DataFrame({"grid_id": gid, "lat": lat, "lon": lon, "area_km2": area,
                     "country_m49": cm49, "region_id": reg})
for h in horizons:
    ih = hz.index(h)
    j = hidx[h]
    # ratio of ensemble-mean global temperatures used for the mitigation map
    rbar = GS[:, j].mean() / GH[:, j].mean()
    record(f"grid.ratio_GS_over_GH.{h}", rbar)
    TH = f["maps/climate_land/tas_no_srm_mean"][ih].astype(float)
    TS = f["maps/climate_land/tas_srm_mean"][ih].astype(float)
    PH = f["maps/climate_land/pr_no_srm_mean"][ih].astype(float)
    PS = f["maps/climate_land/pr_srm_mean"][ih].astype(float)
    # Approximation A1: E[T_M(x)] ~ E[T_H(x)] * mean(G_S)/mean(G_H).
    TM = TH * rbar
    PM = PH * rbar
    # Approximation A2 (used in all results): add a country-specific shift so that
    # the area-weighted country mean of the map equals the EXACT member-level
    # country value E_m[beta_{c,k(m)} G_S,m]. Country values are exact area means
    # of the grid (checked below), so A2 is exact at country level and only the
    # within-country distribution of pattern weights remains approximate.
    TM_c_exact = (beta * GS[:, [j]]).mean(axis=0)
    PM_c_exact = (pi * GS[:, [j]]).mean(axis=0)
    TH_c = TcH[:, j, :].mean(axis=0)
    PH_c = PcH[:, j, :].mean(axis=0)
    shiftT = pd.Series(TM_c_exact - TH_c * rbar, index=m49)
    shiftP = pd.Series(PM_c_exact - PH_c * rbar, index=m49)
    sT = pd.Series(cm49).map(shiftT).fillna(0.0).to_numpy()
    sP = pd.Series(cm49).map(shiftP).fillna(0.0).to_numpy()
    grid[f"TM_A1_{h}"] = TM
    TM = TM + sT
    PM = PM + sP
    record(f"grid.A2_shiftT_absmax.{h}", float(np.nanmax(np.abs(shiftT.values))))
    grid[f"TH_{h}"] = TH
    grid[f"TS_{h}"] = TS
    grid[f"TM_{h}"] = TM
    grid[f"R_{h}"] = TS - TM                     # residual: SRM minus mitigation (C)
    grid[f"CS_{h}"] = TH - TS                    # cooling delivered by SRM (C)
    grid[f"CM_{h}"] = TH - TM                    # cooling delivered by mitigation (C)
    grid[f"PH_{h}"] = PH
    grid[f"PS_{h}"] = PS
    grid[f"PM_{h}"] = PM
    grid[f"RP_{h}"] = PS - PM                    # precipitation residual (percentage points)
    for qi, qq in enumerate(q):
        tag = f"p{int(round(qq * 100))}"
        grid[f"TH_{h}_{tag}"] = f["maps/climate_land/tas_no_srm_quantiles"][ih, qi].astype(float)
        grid[f"TS_{h}_{tag}"] = f["maps/climate_land/tas_srm_quantiles"][ih, qi].astype(float)
        grid[f"dT_{h}_{tag}"] = f["maps/climate_land/tas_delta_quantiles"][ih, qi].astype(float)
        grid[f"dP_{h}_{tag}"] = f["maps/climate_land/pr_delta_quantiles"][ih, qi].astype(float)

    # --- validation of approximation A1 at country level, where the exact value is known
    TM_exact = (beta * GS[:, [j]]).mean(axis=0)             # E_m[beta_k G_S,m]
    TM_A1 = TcH[:, j, :].mean(axis=0) * rbar                # E_m[T_H] * rbar
    err = np.abs(TM_exact - TM_A1)[has_clim]
    R_exact = (TcS[:, j, :] - beta * GS[:, [j]]).mean(axis=0)[has_clim]
    val += [(f"A1 abs error in E[T_M], country level, {h}: mean", err.mean()),
            (f"A1 abs error in E[T_M], country level, {h}: p95", np.percentile(err, 95)),
            (f"A1 abs error in E[T_M], country level, {h}: max", err.max()),
            (f"|E[R]| country level exact, {h}: median", np.median(np.abs(R_exact)))]
    record(f"validation.A1_err_mean.{h}", err.mean())
    record(f"validation.A1_err_p95.{h}", np.percentile(err, 95))
    record(f"validation.A1_err_max.{h}", err.max())

    # land-area-weighted global summaries of the grid fields
    w = area / area.sum()
    for v in ("TH", "TS", "TM", "R", "RP", "PH", "PS", "PM"):
        record(f"grid.landmean.{v}.{h}", float(np.sum(w * grid[f"{v}_{h}"])))
    record(f"grid.landshare_R_pos.{h}", float(np.sum(w * (grid[f"R_{h}"] > 0))))
    record(f"grid.landshare_RP_neg.{h}", float(np.sum(w * (grid[f"RP_{h}"] < 0))))

# country aggregation check: is the country value an area-weighted mean of grid cells?
cw = pd.read_csv(P("climrisk_grid_crosswalk"))
grid = grid.merge(cw[["grid_id_0p5", "country_iso3", "is_coastal_rsl_proxy", "economic_support"]]
                  .rename(columns={"grid_id_0p5": "grid_id"}), on="grid_id", how="left")
chk = []
for c3 in ["MEX", "BRA", "USA", "IND", "PNG", "AUS", "NGA", "RUS"]:
    sub = grid[grid.country_iso3 == c3]
    if sub.empty:
        continue
    ci = iso3.index(c3)
    a = np.average(sub["TH_2100"], weights=sub["area_km2"])
    b = TcH[:, hidx[2100], ci].mean()
    chk.append(abs(a - b))
val.append(("max |area-wtd grid mean - country mean| T_H 2100 (8 countries)", max(chk)))

grid.to_parquet(P("interim") / "grid_fields.parquet", index=False)
pd.DataFrame(val, columns=["check", "value"]).to_csv(P("results") / "climrisk_validation.csv",
                                                     index=False)
print(pd.DataFrame(val, columns=["check", "value"]).to_string())
print("grid cells:", len(grid))
