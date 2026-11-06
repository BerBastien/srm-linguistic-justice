"""06 - Distributional metrics across language vitality classes.

Estimands (kept separate, never mixed):
  * language-weighted   each living language counts once (linguistic diversity)
  * speaker-weighted    weights = L1 speakers (people)
  * territory-weighted  weights = polygon area (land)

Exposures (2100 unless stated; S = SRM, M = mitigation with the same global mean
temperature, H = no-SRM high-emission world):
  R    residual temperature  T_S - T_M (C)            >0: SRM leaves extra heat
  rho  cooling ratio  (T_H - T_S)/(T_H - T_M)         >1: SRM cools more than M
  RP   residual precipitation  P_S - P_M (pp)         <0: SRM drier than M
  econ present-value loss, % of PV GDP of host countries (territory-share composite):
       S-M (ordinary KU), ST-M (SRM with termination risk minus M), premium (ST - S)

Metrics
  group means / medians, bootstrap 95% CIs (languages resampled within groups),
  ensemble p17-p83 of group statistics (60 members), Wilcoxon rank-sum tests,
  disproportionality index  DI_g = mean_g(B) / mean_all(B)  for burdens B >= 0,
  concentration indices by speaker-community size and by endangerment rank,
  relative risks of exceeding dead-bands, and Sankey flow tables.

Outputs (public, aggregated): results/tables/*.csv and numbers.json entries.
Restricted: data/interim/lang_metrics.parquet (language-level, Ethnologue-derived).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, LIVING_ORDER, P, ensure_dirs, record  # noqa: E402
from srmlang import stats as st  # noqa: E402

ensure_dirs()
I = P("interim")
T = P("results")
A = CFG["analysis"]
DB_T, DB_P = A["temp_deadband_c"], A["precip_deadband_pct"]
NB, SEED = A["n_bootstrap"], A["random_seed"]

lx = pd.read_parquet(I / "lang_exposure.parquet")
cs = pd.read_parquet(I / "lang_country_shares.parquet")
mem = np.load(I / "lang_member_R.npz", allow_pickle=True)
pvz = np.load(I / "country_member_pv.npz", allow_pickle=True)
iso3 = list(pvz["iso3"])
pv_gdp = pvz["pv_gdp"]
ci = {c: i for i, c in enumerate(iso3)}

# ------------------------------------------------------------------ language-level economics
# E_l,m = sum_c s_lc * 100 * PV_k[m,c] / PVGDP_c, shares renormalised over countries
# that have CLIMRISK economics.
econ_keys = ["H", "S", "M", "M_noT", "ST", "ST_restricted", "premium"]
isos = lx.ISO.to_numpy()
pos = {iso: i for i, iso in enumerate(isos)}
Smat = np.zeros((len(isos), len(iso3)))
for r in cs.itertuples():
    if r.ISO in pos and r.CTRY3 in ci and pv_gdp[ci[r.CTRY3]] > 0:
        Smat[pos[r.ISO], ci[r.CTRY3]] += r.country_share
rows_ok = Smat.sum(1) > 0
Smat[rows_ok] /= Smat[rows_ok].sum(1, keepdims=True)
econ_m = {}
for k in econ_keys:
    pct = 100 * np.nan_to_num(pvz[k]) / np.where(pv_gdp > 0, pv_gdp, np.nan)[None, :]
    pct = np.nan_to_num(pct)
    E = Smat @ pct.T                                   # (L, 60)
    E[~rows_ok] = np.nan
    econ_m[k] = E
econ_m["S_minus_M"] = econ_m["S"] - econ_m["M"]
econ_m["ST_minus_M"] = econ_m["ST"] - econ_m["M"]
econ_m["H_minus_ST"] = econ_m["H"] - econ_m["ST"]
for k, E in econ_m.items():
    lx[f"econ_{k}"] = np.nanmedian(E, axis=1)          # member median per language
np.savez_compressed(I / "lang_member_econ.npz", ISO=isos,
                    **{k: econ_m[k] for k in ("premium", "ST_minus_M", "S_minus_M", "M", "ST")})
record("econ.n_languages_with_econ", int(rows_ok.sum()))

# member arrays aligned to lx
mem_idx = pd.Series(np.arange(len(mem["ISO"])), index=mem["ISO"])
lx["_m"] = lx.ISO.map(mem_idx)

for h in A["horizons"]:
    lx[f"rho_{h}"] = lx[f"CS_{h}"] / lx[f"CM_{h}"]

# ------------------------------------------------------------------ samples


def sample(variant="main"):
    d = lx[lx.vitality.isin(LIVING_ORDER)].copy()
    if variant != "with_sign":
        d = d[~d.sign_language]
    if variant == "no_fallback":
        d = d[~d.any_fallback]
    d["w_lang"] = 1.0
    d["w_spk"] = d.L1_Users.fillna(0).clip(lower=0)
    d["w_area"] = d.territory_km2
    d["end"] = d.vitality.isin(["In trouble", "Dying"])
    return d


S0 = sample("main")
for h in A["horizons"]:
    S0[f"drier_{h}"] = (S0[f"RP_{h}"] < -DB_P).astype(float) * 100     # percent indicator
record("sample.n_languages_main", len(S0))
record("sample.n_endangered_main", int(S0.end.sum()))
record("sample.share_endangered_main", float(S0.end.mean()))
for g in LIVING_ORDER:
    record(f"sample.n_{g.replace(' ', '_')}", int((S0.vitality == g).sum()))
record("sample.L1_total_main", float(S0.w_spk.sum()))
record("sample.n_annexI", int(S0.annex_i.sum()))
for g in LIVING_ORDER:
    d = S0[S0.vitality == g]
    record(f"sample.share_annexI.{g.replace(' ', '_')}", float(d.annex_i.mean()))
    record(f"sample.mean_abs_lat.{g.replace(' ', '_')}", float(d.lat_mean.abs().mean()))

# ------------------------------------------------------------------ main group table


def group_stats(d, wcol="w_lang", h=2100, tag=""):
    """One row per vitality class (+ endangered / non-endangered / all)."""
    groups = [(g, d.vitality == g) for g in LIVING_ORDER] + \
             [("Endangered", d.end), ("Not endangered", ~d.end), ("All living", d.end | ~d.end)]
    out = []
    for name, sel in groups:
        x = d[sel]
        w = x[wcol].to_numpy()
        rr = {"group": name, "n_languages": len(x), "L1_speakers": float(x.w_spk.sum()),
              "territory_km2": float(x.w_area.sum()), "share_annexI": float(x.annex_i.mean()),
              "mean_abs_lat": st.wmean(x.lat_mean.abs(), w)}
        R = x[f"R_{h}"].to_numpy()
        RP = x[f"RP_{h}"].to_numpy()
        rr["R_mean"] = st.wmean(R, w)
        reps = st.bootstrap(lambda idx: st.wmean(R[idx], w[idx]), len(x), NB, SEED)
        rr["R_mean_lo"], rr["R_mean_hi"] = st.ci(reps)
        rr["R_median"] = st.wquantile(R, 0.5, w)
        rr["share_R_warmer"] = st.wshare(R > DB_T, w)
        rr["share_R_similar"] = st.wshare(np.abs(R) <= DB_T, w)
        rr["share_R_cooler"] = st.wshare(R < -DB_T, w)
        rr["rho"] = st.wmean(x[f"CS_{h}"], w) / st.wmean(x[f"CM_{h}"], w)
        rr["CS_mean"] = st.wmean(x[f"CS_{h}"], w)
        rr["CM_mean"] = st.wmean(x[f"CM_{h}"], w)
        rr["RP_mean"] = st.wmean(RP, w)
        rr["RP_median"] = st.wquantile(RP, 0.5, w)
        reps = st.bootstrap(lambda idx: st.wquantile(RP[idx], 0.5, w[idx]), len(x), NB // 4, SEED)
        rr["RP_median_lo"], rr["RP_median_hi"] = st.ci(reps)
        rr["share_RP_drier"] = st.wshare(RP < -DB_P, w)
        rr["share_RP_wetter"] = st.wshare(RP > DB_P, w)
        # ensemble spread of the group statistic (member level)
        mi = x["_m"].dropna().astype(int).to_numpy()
        wm = x.loc[x["_m"].notna(), wcol].to_numpy()
        if mi.size and h in (2050, 2100):
            Rm = mem[f"R_{h}"][mi]
            RPm = mem[f"RP_{h}"][mi]
            gm = (Rm * wm[:, None]).sum(0) / wm.sum()
            rr["R_mean_ens_p17"], rr["R_mean_ens_p50"], rr["R_mean_ens_p83"] = \
                np.percentile(gm, [17, 50, 83])
            med = np.array([st.wquantile(RPm[:, m], 0.5, wm) for m in range(RPm.shape[1])])
            rr["RP_median_ens_p17"], rr["RP_median_ens_p50"], rr["RP_median_ens_p83"] = \
                np.percentile(med, [17, 50, 83])
            rr["RP_median_ens_frac_neg"] = float((med < 0).mean())
        # economics (2100 horizon only; PV covers 2026-2100)
        for k in ("H", "M", "S", "ST", "premium", "S_minus_M", "ST_minus_M", "H_minus_ST"):
            rows = x.index.to_numpy()
            E = econ_m[k][lx.index.get_indexer(rows)]
            ok = np.isfinite(E[:, 0])
            if ok.any():
                gm = (E[ok] * w[ok, None]).sum(0) / w[ok].sum()
                rr[f"econ_{k}_p50"] = float(np.median(gm))
                rr[f"econ_{k}_p17"], rr[f"econ_{k}_p83"] = np.percentile(gm, [17, 83])
                if k in ("S_minus_M", "ST_minus_M", "H_minus_ST"):
                    rr[f"econ_{k}_frac_members_pos"] = float((gm > 0).mean())
        out.append(rr)
    t = pd.DataFrame(out)
    t.insert(0, "weighting", wcol)
    t.insert(1, "horizon", h)
    t.insert(2, "variant", tag)
    return t


main = group_stats(S0, "w_lang", 2100, "main")
main.to_csv(T / "table1_group_stats_main.csv", index=False)
for _, r in main.iterrows():
    g = r.group.replace(" ", "_")
    for c in ("n_languages", "R_mean", "R_mean_lo", "R_mean_hi", "R_mean_ens_p17", "R_mean_ens_p83",
              "share_R_warmer", "share_R_cooler", "share_R_similar", "rho", "RP_median",
              "RP_median_lo", "RP_median_hi", "RP_median_ens_p17", "RP_median_ens_p83",
              "RP_median_ens_frac_neg", "share_RP_drier", "share_RP_wetter",
              "econ_M_p50", "econ_S_p50", "econ_ST_p50", "econ_H_p50",
              "econ_premium_p50", "econ_premium_p17", "econ_premium_p83",
              "econ_S_minus_M_p50", "econ_S_minus_M_p17", "econ_S_minus_M_p83",
              "econ_ST_minus_M_p50", "econ_ST_minus_M_p17", "econ_ST_minus_M_p83",
              "econ_ST_minus_M_frac_members_pos", "econ_H_minus_ST_p50",
              "econ_H_minus_ST_frac_members_pos", "share_annexI", "mean_abs_lat", "CS_mean",
              "CM_mean", "L1_speakers"):
        if c in r and pd.notna(r[c]):
            record(f"t1.{g}.{c}", r[c])

# ------------------------------------------------------------------ gaps & tests
gap_rows = []
for wcol in ("w_lang", "w_spk", "w_area"):
    d = S0[S0[wcol] > 0]
    for var, h in (("R", 2100), ("RP", 2100), ("drier", 2100), ("R", 2050), ("RP", 2050),
                   ("econ_ST_minus_M", None), ("econ_premium", None), ("econ_S_minus_M", None)):
        col = f"{var}_{h}" if h else var
        x = d[col].to_numpy()
        w = d[wcol].to_numpy()
        e = d.end.to_numpy()
        ok = np.isfinite(x)
        x, w, e = x[ok], w[ok], e[ok]
        dying = (d.vitality.to_numpy()[ok] == "Dying")
        inst = (d.vitality.to_numpy()[ok] == "Institutional")

        def gap(idx, x=x, w=w, e=e):
            return st.wmean(x[idx][e[idx]], w[idx][e[idx]]) - st.wmean(x[idx][~e[idx]], w[idx][~e[idx]])

        g0 = gap(np.arange(x.size))
        reps = st.bootstrap(gap, x.size, NB, SEED, strata=e)
        lo, hi = st.ci(reps)
        U, p, rb = st.ranksum(x[e], x[~e])
        gdi = st.wmean(x[dying], w[dying]) - st.wmean(x[inst], w[inst])
        gap_rows.append({"weighting": wcol, "variable": col, "gap_end_minus_non": g0, "ci_lo": lo,
                         "ci_hi": hi, "wilcoxon_U": U, "wilcoxon_p": p, "rank_biserial": rb,
                         "gap_dying_minus_institutional": gdi, "n_end": int(e.sum()),
                         "n_non": int((~e).sum())})
gaps = pd.DataFrame(gap_rows)
gaps.to_csv(T / "table_gaps_tests.csv", index=False)
for _, r in gaps.iterrows():
    k = f"gap.{r.weighting}.{r.variable}"
    record(k + ".value", r.gap_end_minus_non)
    record(k + ".ci", [r.ci_lo, r.ci_hi])
    record(k + ".p", r.wilcoxon_p)
    record(k + ".rank_biserial", r.rank_biserial)
    record(k + ".dying_minus_inst", r.gap_dying_minus_institutional)

# EGIDS gradient: OLS slope of R on EGIDS rank (0 = international ... 11 = dormant)
d = S0
for var in ("R_2100", "RP_2100", "econ_ST_minus_M", "econ_premium"):
    x = d.egids_rank.to_numpy(float)
    y = d[var].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slope = np.polyfit(x, y, 1)[0]
    reps = st.bootstrap(lambda idx: np.polyfit(x[idx], y[idx], 1)[0], x.size, NB, SEED)
    record(f"gradient.{var}.slope_per_egids_step", slope)
    record(f"gradient.{var}.ci", st.ci(reps))

# ------------------------------------------------------------------ disproportionality & concentration
burdens = {
    "residual_warming": np.maximum(S0["R_2100"], 0),
    "residual_drying": np.maximum(-S0["RP_2100"], 0),
    "termination_premium": S0["econ_premium"],
    "excess_loss_ST_vs_M": np.maximum(S0["econ_ST_minus_M"], 0),
    "srm_cooling_bonus": np.maximum(-S0["R_2100"], 0),   # a benefit, for contrast
}
dis_rows = []
for bname, B in burdens.items():
    B = B.to_numpy(float)
    ok = np.isfinite(B)
    for wcol in ("w_lang", "w_spk", "w_area"):
        w = S0[wcol].to_numpy(float)
        allm = st.wmean(B[ok], w[ok])
        for g in LIVING_ORDER + ["Endangered"]:
            sel = (S0.vitality == g).to_numpy() if g != "Endangered" else S0.end.to_numpy()
            s = sel & ok
            di = st.wmean(B[s], w[s]) / allm if allm > 0 else np.nan

            def f_di(idx, s=s, B=B, w=w, ok=ok):
                a = st.wmean(B[idx][ok[idx]], w[idx][ok[idx]])
                return st.wmean(B[idx][s[idx]], w[idx][s[idx]]) / a if a > 0 else np.nan

            reps = st.bootstrap(f_di, len(B), NB // 2, SEED)
            lo, hi = st.ci(reps)
            dis_rows.append({"burden": bname, "weighting": wcol, "group": g, "DI": di,
                             "ci_lo": lo, "ci_hi": hi})
        # concentration indices
        size = np.log10(S0["L1_Users"].fillna(0).clip(lower=0) + 1).to_numpy(float)
        rank_e = S0["egids_rank"].to_numpy(float)
        for rname, rv in (("by_speakers", size), ("by_endangerment", rank_e)):
            okk = ok & np.isfinite(rv)
            cidx = st.concentration_index(B[okk], rv[okk], w[okk])
            reps = st.bootstrap(lambda idx: st.concentration_index(B[okk][idx], rv[okk][idx],
                                                                   w[okk][idx]),
                                int(okk.sum()), NB // 2, SEED)
            lo, hi = st.ci(reps)
            dis_rows.append({"burden": bname, "weighting": wcol, "group": f"CI_{rname}",
                             "DI": cidx, "ci_lo": lo, "ci_hi": hi})
dis = pd.DataFrame(dis_rows)
dis.to_csv(T / "table_disproportionality.csv", index=False)
for _, r in dis.iterrows():
    k = f"DI.{r.burden}.{r.weighting}.{r.group.replace(' ', '_')}"
    record(k, [r.DI, r.ci_lo, r.ci_hi])

# relative risks of dead-band exceedance (endangered vs not; dying vs institutional)
rr_rows = []
for cond_name, cond in (("R_warmer", S0["R_2100"] > DB_T), ("R_cooler", S0["R_2100"] < -DB_T),
                        ("RP_drier", S0["RP_2100"] < -DB_P), ("RP_wetter", S0["RP_2100"] > DB_P),
                        ("premium_gt3", S0["econ_premium"] > 3),
                        ("STminusM_gt3", S0["econ_ST_minus_M"] > 3)):
    c = cond.to_numpy()
    e = S0.end.to_numpy()
    dy = (S0.vitality == "Dying").to_numpy()
    ins = (S0.vitality == "Institutional").to_numpy()
    rr1 = c[e].mean() / max(c[~e].mean(), 1e-12)
    rr2 = c[dy].mean() / max(c[ins].mean(), 1e-12)
    reps = st.bootstrap(lambda idx: c[idx][e[idx]].mean() / max(c[idx][~e[idx]].mean(), 1e-12),
                        len(c), NB, SEED, strata=e)
    rr_rows.append({"condition": cond_name, "share_end": c[e].mean(), "share_non": c[~e].mean(),
                    "RR_end_vs_non": rr1, "RR_ci_lo": st.ci(reps)[0], "RR_ci_hi": st.ci(reps)[1],
                    "share_dying": c[dy].mean(), "share_inst": c[ins].mean(),
                    "RR_dying_vs_inst": rr2})
rrt = pd.DataFrame(rr_rows)
rrt.to_csv(T / "table_relative_risks.csv", index=False)
for _, r in rrt.iterrows():
    for c in ("share_end", "share_non", "RR_end_vs_non", "RR_ci_lo", "RR_ci_hi", "share_dying",
              "share_inst", "RR_dying_vs_inst"):
        record(f"RR.{r.condition}.{c}", r[c])

# ------------------------------------------------------------------ Annex I split
ax_rows = []
for ax in (False, True):
    d = S0[S0.annex_i == ax]
    t = group_stats(d, "w_lang", 2100, f"annexI={ax}")
    ax_rows.append(t)
    for _, r in t.iterrows():
        record(f"annex.{ax}.{r.group.replace(' ', '_')}.R_mean", r.R_mean)
        record(f"annex.{ax}.{r.group.replace(' ', '_')}.n", r.n_languages)
        record(f"annex.{ax}.{r.group.replace(' ', '_')}.RP_median", r.RP_median)
        record(f"annex.{ax}.{r.group.replace(' ', '_')}.econ_premium_p50", r.get("econ_premium_p50"))
        record(f"annex.{ax}.{r.group.replace(' ', '_')}.econ_ST_minus_M_p50",
               r.get("econ_ST_minus_M_p50"))
pd.concat(ax_rows).to_csv(T / "table_annexI_split.csv", index=False)

# share of all endangered languages located (>=50% territory) outside Annex I
record("annex.share_endangered_outside_annexI", float((~S0[S0.end].annex_i).mean()))
record("annex.share_all_outside_annexI", float((~S0.annex_i).mean()))

# ------------------------------------------------------------------ latitude bands
bands = pd.cut(S0.lat_mean, [-60, -35, -23.5, 0, 23.5, 35, 55, 90],
               labels=["35-60S", "23-35S", "0-23S", "0-23N", "23-35N", "35-55N", ">55N"])
lb = S0.groupby(bands, observed=True).agg(
    n=("ISO", "size"), share_end=("end", "mean"), R=("R_2100", "mean"),
    RP_median=("RP_2100", "median"), premium=("econ_premium", "mean"),
    ST_minus_M=("econ_ST_minus_M", "mean")).reset_index().rename(columns={"lat_mean": "band"})
lb.to_csv(T / "table_latitude_bands.csv", index=False)
for _, r in lb.iterrows():
    record(f"latband.{r.iloc[0]}", {"n": r.n, "share_end": r.share_end, "R": r.R,
                                    "RP_median": r.RP_median, "premium": r.premium})

# ------------------------------------------------------------------ Sankey flows (LIC polygons)
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.land_w_km2.notna() & ~pe.sign_language & pe.vitality_LIC.isin(LIVING_ORDER)].copy()
# v5: communities whose community-level EGIDS is excluded (9, dormant: no first-language speakers)
pe = pe[~pe.EGIDS_LIC.isin(CFG["analysis"].get("exclude_egids", []))].copy()
pe["origin"] = np.where(pe.Is_Indigenous == "T", "Indigenous to country",
                        "Non-indigenous (colonial or immigrant origin)")
pe["T_out"] = np.select([pe.R_2100 > DB_T, pe.R_2100 < -DB_T],
                        ["SRM warmer than mitigation", "SRM cooler than mitigation"],
                        "Similar (within 0.1 C)")
pe["P_out"] = np.select([pe.RP_2100 < -DB_P, pe.RP_2100 > DB_P],
                        ["SRM drier than mitigation", "SRM wetter than mitigation"],
                        "Similar (within 2 pp)")
pe["L1_LIC"] = pd.to_numeric(pe.L1_LIC, errors="coerce").fillna(0)
flows = pe.groupby(["origin", "vitality_LIC", "T_out", "P_out"]).agg(
    n_lic=("ID", "size"), L1=("L1_LIC", "sum"), area_km2=("poly_area_km2", "sum")).reset_index()
flows.to_csv(T / "sankey_flows_lic.csv", index=False)
record("sankey.n_lic_polygons", len(pe))
record("sankey.n_lic_indigenous", int((pe.Is_Indigenous == "T").sum()))
record("sankey.n_lic_nonindigenous", int((pe.Is_Indigenous != "T").sum()))
for o in pe.origin.unique():
    d = pe[pe.origin == o]
    for out in pe.T_out.unique():
        record(f"sankey.{o[:10]}.T.{out[:10]}", float((d.T_out == out).mean()))
    for out in pe.P_out.unique():
        record(f"sankey.{o[:10]}.P.{out[:10]}", float((d.P_out == out).mean()))
    record(f"sankey.{o[:10]}.R_mean", float(d.R_2100.mean()))
    record(f"sankey.{o[:10]}.RP_median", float(d.RP_2100.median()))
    record(f"sankey.{o[:10]}.share_endangered",
           float(d.vitality_LIC.isin(["In trouble", "Dying"]).mean()))

# ------------------------------------------------------------------ robustness variants (SI)
rob = []
for wcol in ("w_lang", "w_spk", "w_area"):
    for h in (2050, 2100):
        rob.append(group_stats(S0[S0[wcol] > 0], wcol, h, "main"))
for var in ("no_fallback", "with_sign"):
    rob.append(group_stats(sample(var), "w_lang", 2100, var))
# LIC-level units with LIC-level EGIDS
pl = pe.copy()
pl["vitality"] = pl.vitality_LIC
pl["end"] = pl.vitality.isin(["In trouble", "Dying"])
pl["w_lang"] = 1.0
pl["w_spk"] = pl.L1_LIC
pl["w_area"] = pl.poly_area_km2
pl["annex_i"] = pl.annex_i.astype(bool)
pl["_m"] = np.nan
pl["lat_mean"] = pl.lat_mean
for k in ("premium", "ST_minus_M", "S_minus_M"):
    pass
rob_lic = []
for g in LIVING_ORDER + ["Endangered", "Not endangered"]:
    sel = (pl.vitality == g) if g in LIVING_ORDER else (pl.end if g == "Endangered" else ~pl.end)
    x = pl[sel]
    rob_lic.append({"weighting": "w_lang", "horizon": 2100, "variant": "LIC_units", "group": g,
                    "n_languages": len(x), "R_mean": x.R_2100.mean(),
                    "share_R_warmer": (x.R_2100 > DB_T).mean(),
                    "RP_median": x.RP_2100.median(), "share_RP_drier": (x.RP_2100 < -DB_P).mean()})
robust = pd.concat(rob + [pd.DataFrame(rob_lic)], ignore_index=True)
robust.to_csv(T / "table_robustness.csv", index=False)

# restricted language-level table (Ethnologue-derived): kept out of the public repo
keep = ["ISO", "Language_Name", "Country_Code", "Area", "Family", "EGIDS", "vitality", "end",
        "L1_Users", "territory_km2", "annex_i", "lat_mean", "R_2050", "R_2100", "RP_2050",
        "RP_2100", "CS_2100", "CM_2100", "rho_2100", "TS_2100", "TM_2100", "TH_2100",
        "econ_H", "econ_M", "econ_S", "econ_ST", "econ_premium", "econ_S_minus_M",
        "econ_ST_minus_M", "any_fallback"]
S0[[c for c in keep if c in S0.columns]].to_parquet(I / "lang_metrics.parquet", index=False)
print(main[["group", "n_languages", "R_mean", "R_mean_lo", "R_mean_hi", "R_mean_ens_p17",
            "R_mean_ens_p83", "rho", "RP_median", "share_RP_drier", "econ_ST_minus_M_p50",
            "econ_premium_p50"]].round(3).to_string())
print(gaps.round(4).to_string())
print(rrt.round(3).to_string())
