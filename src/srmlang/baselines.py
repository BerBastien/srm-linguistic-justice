"""Shared set-up of the cell-level member fields and the unit weights (scripts 23 and 26).

`setup()` reproduces exactly what script 23 computed inline before it was moved here:
  - grid cells north of 60 S, their areas, and member-level fields for 2100
      Rm  (cells, 60): temperature difference SAI minus mitigation, R = T_S - T_M (deg C)
      RPm (cells, 60): rainfall difference R_P = P_S - P_M (pp of 1900 rainfall)
    built with the cell-level hybrid
      R_c,m  = (Rbar_c / mean dG) dG_m + v_{country(c),k(m)} dG_m
      RP_c,m = RPbar_c + (RP_country,m - mean_m RP_country)
  - outcome masks with the dead-bands of config.yaml (drier, wetter, warmer, cooler)
  - sparse operators: languages (range shares), polygons (people), lineages
  - weights: area, populated land, GDP, first-language speakers, hotspot endemism
  - units(X): every unit's weighted share (boolean X) or weighted mean (float X), for the
    ensemble-mean field (C,) or for each member (C, 60)
It returns a dict of all these names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import CFG, P

NONINST = ["Stable", "In trouble", "Dying"]


def setup() -> dict:
    I = P("interim")  # noqa: E741
    DBP, DBT = CFG["analysis"]["precip_deadband_pct"], CFG["analysis"]["temp_deadband_c"]
    grid = pd.read_parquet(I / "grid_fields.parquet")
    grid = grid[grid.lat > -60].reset_index(drop=True)
    cidx = pd.Series(np.arange(len(grid)), index=grid.grid_id)
    area = grid.area_km2.to_numpy()

    # ---------------------------------------------------------------- member-level cell fields
    cm = np.load(I / "country_member_fields.npz", allow_pickle=True)
    iso3 = list(cm["iso3"])
    pidx = cm["pattern_index"]
    GH, GS, years = cm["GH"], cm["GS"], cm["years"]
    j = int(np.where(years == 2100)[0][0])
    dG = GH[:, j] - GS[:, j]
    unit = cm["beta"] + cm["gamma"]
    unit_k = np.stack([unit[pidx == k][0] for k in np.unique(pidx)])
    v = np.nan_to_num(unit - unit_k.mean(0, keepdims=True))             # (60, 187)
    RPc = cm["RP_2100"]
    dev = np.nan_to_num(RPc - np.nanmean(RPc, 0, keepdims=True))
    ci = grid.country_iso3.map({c: i for i, c in enumerate(iso3)})
    has = ci.notna().to_numpy()
    cc = ci[has].astype(int).to_numpy()
    Rm = (grid.R_2100.to_numpy() / dG.mean())[:, None] * dG[None, :]
    Rm[has] += v[:, cc].T * dG[None, :]
    RPm = np.repeat(grid.RP_2100.to_numpy()[:, None], 60, 1)
    RPm[has] += dev[:, cc].T
    R0, RP0 = grid.R_2100.to_numpy(), grid.RP_2100.to_numpy()

    OUT = {"drier": (RP0 < -DBP, RPm < -DBP), "wetter": (RP0 > DBP, RPm > DBP),
           "warmer": (R0 > DBT, Rm > DBT), "cooler": (R0 < -DBT, Rm < -DBT)}

    # ---------------------------------------------------------------- weights / operators
    lm = pd.read_parquet(I / "lang_metrics.parquet")
    lm["noninst"] = lm.vitality.isin(NONINST)
    pe = pd.read_parquet(I / "poly_exposure.parquet")
    pe = pe[pe.ISO.isin(lm.ISO) & pe.land_w_km2.notna() & ~pe.sign_language]
    w = pd.read_parquet(I / "poly_cell_weights.parquet")
    w = w[w.pid.isin(pe.pid) & w.grid_id.isin(cidx.index)].copy()
    w["ISO"] = w.pid.map(pe.set_index("pid").ISO)
    # language operator (rows sum to one)
    lw = w.groupby(["ISO", "grid_id"], as_index=False).w_km2.sum()
    lid = pd.Series(np.arange(len(lm)), index=lm.ISO)
    r_, c_ = lid[lw.ISO].to_numpy(), cidx[lw.grid_id].to_numpy()
    tot = np.bincount(r_, weights=lw.w_km2.to_numpy(), minlength=len(lm))
    Lop = sp.csr_matrix((lw.w_km2.to_numpy() / tot[r_], (r_, c_)), shape=(len(lm), len(grid)))
    # polygon operator for people
    pid_ids = pe.pid.to_numpy()
    pidx_s = pd.Series(np.arange(len(pid_ids)), index=pid_ids)
    r2, c2 = pidx_s[w.pid].to_numpy(), cidx[w.grid_id].to_numpy()
    tot2 = np.bincount(r2, weights=w.w_km2.to_numpy(), minlength=len(pid_ids))
    Pop = sp.csr_matrix((w.w_km2.to_numpy() / np.where(tot2 > 0, tot2, 1)[r2], (r2, c2)), shape=(len(pid_ids), len(grid)))
    L1 = pe.L1_LIC.fillna(0).to_numpy().astype(float)
    # lineage operator
    hl = pd.read_parquet(I / "heritage_lang.parquet").set_index("ISO")
    lin = lm.ISO.map(hl.lineage)
    codes = pd.Categorical(lin)                      # v5: Glottolog lineages; NaN = not in the lineage sample
    okl = codes.codes >= 0
    Gop = sp.csr_matrix((np.ones(int(okl.sum())), (codes.codes[okl], np.arange(len(lm))[okl])),
                        shape=(len(codes.categories), len(lm)))
    gn = np.asarray(Gop.sum(1)).ravel()
    # hotspot cells and economy
    hc = pd.read_parquet(I / "heritage_cells.parquet").set_index("grid_id")
    hot = grid.grid_id.map(hc.hotspot_half).fillna(False).to_numpy()
    lwe = grid.grid_id.map(hc.LWE_noninst).fillna(0).to_numpy()
    gdp = grid.grid_id.map(pd.read_parquet(I / "pixel_cells.parquet").set_index("grid_id").gdp).fillna(0).to_numpy()
    popl = gdp > 0
    non = lm.noninst.to_numpy()


    def units(Xc):
        """Xc: (C,) or (C, 60) boolean/float. Returns dict unit -> value(s)."""
        X = Xc.astype(float)
        if X.ndim == 1:
            X = X[:, None]
        f_lang = Lop @ X
        f_poly = Pop @ X
        out = {
            "land": (area[:, None] * X).sum(0) / area.sum(),
            "populated land": (area[popl, None] * X[popl]).sum(0) / area[popl].sum(),
            "economy (GDP)": (gdp[:, None] * X).sum(0) / gdp.sum(),
            "people (L1 speakers)": (L1[:, None] * f_poly).sum(0) / L1.sum(),
            "non-institutional languages": f_lang[non].mean(0),
            "institutional languages": f_lang[~non].mean(0),
            "linguistic diversity hotspots": (lwe[hot, None] * X[hot]).sum(0) / lwe[hot].sum(),
            "language lineages": ((Gop @ f_lang) / gn[:, None]).mean(0),
        }
        return {k: np.asarray(v).ravel() for k, v in out.items()}

    env = dict(locals())
    env.pop("I", None)
    return env
