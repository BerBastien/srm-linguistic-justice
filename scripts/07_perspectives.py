"""07 - Whose world? Burdens seen per dollar, per person, per language.

The same country-level burden (percent of present-value GDP) is averaged with four
different weights:
  per dollar             w_c = PV GDP_c (2026-2100)            -> the economy's view
  per person             w_c = population_c (OWID, latest year)
  per language           w_c = sum_l s_lc  (living languages, territory shares)
  per endangered language w_c = sum_{l endangered} s_lc

Burden ratio (BR) = perspective-weighted mean / dollar-weighted mean.
BR > 1 means the burden looks larger from that perspective than from the economy's.

Also:
  * termination-hazard sensitivity (annual failure probability p = 0.5%, 1%, 2%),
  * historical responsibility (OWID cumulative fossil+cement CO2) versus burden,
  * equity-weighted world totals (income elasticity of marginal utility eta = 1, 1.5).

Outputs: results/tables/perspectives.csv, termination_sensitivity.csv,
         responsibility_country.csv (public; no Ethnologue raw data, counts only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import ANNEX_I_ISO2, LIVING_ORDER, P, ROOT, ensure_dirs, record  # noqa: E402

ensure_dirs()
I, T = P("interim"), P("results")
pvz = np.load(I / "country_member_pv.npz", allow_pickle=True)
iso3 = list(pvz["iso3"])
pv_gdp = pvz["pv_gdp"]
ok_c = pv_gdp > 0
pct = {k: 100 * np.nan_to_num(pvz[k]) / np.where(ok_c, pv_gdp, np.nan)[None, :]
       for k in ("H", "S", "M", "ST", "premium")}
pct["ST_minus_M"] = pct["ST"] - pct["M"]
pct["S_minus_M"] = pct["S"] - pct["M"]
pct["H_minus_M"] = pct["H"] - pct["M"]

# ---------------------------------------------------------------- weights
lm = pd.read_parquet(I / "lang_exposure.parquet")
cs = pd.read_parquet(I / "lang_country_shares.parquet")
living = lm[lm.vitality.isin(LIVING_ORDER) & ~lm.sign_language]
endg = set(living.ISO[living.vitality.isin(["In trouble", "Dying"])])
cs = cs[cs.ISO.isin(set(living.ISO))]
w_lang = cs.groupby("CTRY3")["country_share"].sum()
w_end = cs[cs.ISO.isin(endg)].groupby("CTRY3")["country_share"].sum()

owid = pd.read_csv(P("owid_co2"), usecols=["iso_code", "year", "population", "cumulative_co2",
                                             "co2", "gdp"])
owid = owid[owid.iso_code.notna() & ~owid.iso_code.str.startswith("OWID")]
latest = owid.dropna(subset=["cumulative_co2", "population"]).sort_values("year") \
    .groupby("iso_code").tail(1).set_index("iso_code")
record("owid.latest_year_mode", int(latest.year.mode().iloc[0]))
pop = latest["population"]
cum = latest["cumulative_co2"]          # million tonnes CO2

W = pd.DataFrame(index=iso3)
W["gdp"] = np.where(ok_c, pv_gdp, 0.0)
W["pop"] = pop.reindex(iso3).fillna(0).to_numpy()
W["lang"] = w_lang.reindex(iso3).fillna(0).to_numpy()
W["end"] = w_end.reindex(iso3).fillna(0).to_numpy()
W.loc[~ok_c, :] = 0.0
record("persp.share_languages_with_econ", float(W["lang"].sum() / w_lang.sum()))

# ---------------------------------------------------------------- perspective means
rows = []
for k in ("premium", "ST_minus_M", "S_minus_M", "H_minus_M", "M", "ST"):
    X = np.nan_to_num(pct[k])                      # (60, nc)
    res = {}
    for wname in ("gdp", "pop", "lang", "end"):
        w = W[wname].to_numpy()
        res[wname] = (X * w[None, :]).sum(1) / w.sum()       # (60,)
    for wname in ("gdp", "pop", "lang", "end"):
        q = np.percentile(res[wname], [17, 50, 83])
        br = res[wname] / res["gdp"] if k in ("premium", "ST_minus_M", "M", "ST") else \
            np.full(60, np.nan)
        qb = np.nanpercentile(br, [17, 50, 83]) if np.isfinite(br).any() else [np.nan] * 3
        rows.append({"metric": k, "perspective": wname, "p17": q[0], "p50": q[1], "p83": q[2],
                     "BR_p17": qb[0], "BR_p50": qb[1], "BR_p83": qb[2]})
        record(f"persp.{k}.{wname}.p17_p50_p83", q)
        record(f"persp.{k}.{wname}.BR_p17_p50_p83", qb)
pers = pd.DataFrame(rows)
pers.to_csv(T / "perspectives.csv", index=False)

# ---------------------------------------------------------------- termination hazard sensitivity
sens = []
Mpct = pct["M"]
for tag, fn in (("p005", "termination_country_PV_by_member_p005.csv"),
                ("p01", "termination_country_PV_by_member_p01.csv"),
                ("p02", "termination_country_PV_by_member_p02.csv")):
    d = pd.read_csv(ROOT / "data" / "raw" / "climrisk" / fn)
    prem = d.pivot_table(index="run_id", columns="ISO3", values="KULocalShockPremiumNet_pctPVGDP")
    full = d.pivot_table(index="run_id", columns="ISO3", values="KULocalShockFullPositive_pctPVGDP")
    prem = prem.reindex(columns=iso3).to_numpy()
    full = full.reindex(columns=iso3).to_numpy()
    # run_id order equals member order in the HDF5 (both sorted ascending; checked)
    for k, X in (("premium", prem), ("ST_minus_M", full - Mpct)):
        X = np.nan_to_num(X)
        res = {wn: (X * W[wn].to_numpy()[None, :]).sum(1) / W[wn].sum()
               for wn in ("gdp", "pop", "lang", "end")}
        for wn in ("gdp", "pop", "lang", "end"):
            q = np.percentile(res[wn], [17, 50, 83])
            qb = np.percentile(res[wn] / res["gdp"], [17, 50, 83])
            sens.append({"hazard": tag, "metric": k, "perspective": wn, "p17": q[0], "p50": q[1],
                         "p83": q[2], "BR_p50": qb[1]})
            record(f"sens.{tag}.{k}.{wn}.p50", q[1])
        # share of the language-weighted burden carried outside Annex I
    sens_df = pd.DataFrame(sens)
sens_df.to_csv(T / "termination_sensitivity.csv", index=False)
# consistency check: p01 CSV premium equals HDF5 premium
d01 = pd.read_csv(ROOT / "data" / "raw" / "climrisk" / "termination_country_PV_by_member_p01.csv")
chk = d01.pivot_table(index="run_id", columns="ISO3",
                      values="KULocalShockPremiumNet_pctPVGDP").reindex(columns=iso3).to_numpy()
record("validation.csv_vs_h5_premium_maxabs",
       float(np.nanmax(np.abs(np.nan_to_num(chk) - np.nan_to_num(pct["premium"])))))

# ---------------------------------------------------------------- responsibility
import json  # noqa: E402
with open(P("ne_countries")) as fh:
    ne = json.load(fh)
a2a3 = {ft["properties"]["ISO_A2_EH"]: ft["properties"]["ISO_A3_EH"] for ft in ne["features"]}
annex3 = {a2a3.get(a2) for a2 in ANNEX_I_ISO2 if a2a3.get(a2)}
resp = pd.DataFrame({"iso3": iso3, "pv_gdp": pv_gdp,
                     "premium_p50": np.nanmedian(pct["premium"], 0),
                     "ST_minus_M_p50": np.nanmedian(pct["ST_minus_M"], 0),
                     "S_minus_M_p50": np.nanmedian(pct["S_minus_M"], 0),
                     "n_lang": W["lang"].to_numpy(), "n_end": W["end"].to_numpy()})
resp["population"] = resp.iso3.map(pop)
resp["cum_co2_mt"] = resp.iso3.map(cum)
resp["cum_co2_t_per_cap"] = 1e6 * resp.cum_co2_mt / resp.population
resp["annex_i"] = resp.iso3.isin(annex3)
resp = resp[resp.pv_gdp > 0]
resp.to_csv(T / "responsibility_country.csv", index=False)
r_ok = resp.dropna(subset=["cum_co2_t_per_cap"])
rho, pval = sps.spearmanr(r_ok.cum_co2_t_per_cap, r_ok.premium_p50)
record("resp.spearman_cumco2pc_vs_premium", [rho, pval])
rho2, p2 = sps.spearmanr(r_ok.cum_co2_t_per_cap, r_ok.n_lang)
record("resp.spearman_cumco2pc_vs_nlang", [rho2, p2])
world_cum = cum.reindex(resp.iso3).sum()
record("resp.share_cumco2_annexI", float(resp[resp.annex_i].cum_co2_mt.sum() / world_cum))
record("resp.share_languages_annexI", float(resp[resp.annex_i].n_lang.sum() / resp.n_lang.sum()))
record("resp.share_endangered_annexI", float(resp[resp.annex_i].n_end.sum() / resp.n_end.sum()))
record("resp.share_pop_annexI", float(resp[resp.annex_i].population.sum() /
                                      resp.population.sum()))
# premium terciles of countries by cumulative per-capita CO2
r_ok = r_ok.assign(tercile=pd.qcut(r_ok.cum_co2_t_per_cap, 3, labels=["low", "mid", "high"]))
for t_, d in r_ok.groupby("tercile", observed=True):
    record(f"resp.tercile_{t_}.premium_median", float(d.premium_p50.median()))
    record(f"resp.tercile_{t_}.share_languages", float(d.n_lang.sum() / r_ok.n_lang.sum()))
    record(f"resp.tercile_{t_}.share_cumco2", float(d.cum_co2_mt.sum() / r_ok.cum_co2_mt.sum()))

# ---------------------------------------------------------------- equity weights (SI)
gdp26 = pd.Series(pvz["pv_gdp"], index=iso3)  # placeholder overwritten below
import h5py  # noqa: E402
with h5py.File(P("climrisk_h5"), "r") as f:
    g26 = f["country/economic_pv/gdp_2026_billion_usd2005_ppp"][:]
ypc = pd.Series(1e9 * g26 / W["pop"].replace(0, np.nan).to_numpy(), index=iso3)
ybar = np.nansum(1e9 * g26 * (W["pop"] > 0)) / W["pop"].sum()
for eta in (0.0, 1.0, 1.5):
    ew = (ybar / ypc) ** eta
    ew = ew.fillna(0).to_numpy() * (W["pop"].to_numpy() > 0)
    for k in ("premium", "ST_minus_M"):
        X = np.nan_to_num(pvz[k] if k == "premium" else (pvz["ST"] - pvz["M"]))  # billions
        tot = (X * ew[None, :]).sum(1) / (pv_gdp * (W["pop"].to_numpy() > 0)).sum() * 100
        record(f"equity.eta{eta}.{k}.p17_p50_p83", np.percentile(tot, [17, 50, 83]))

print(pers[pers.metric.isin(["premium", "ST_minus_M", "S_minus_M"])].round(3).to_string())
print(sens_df[sens_df.metric == "premium"].round(3).to_string())
