"""17 - Drying under SAI in the homelands of non-institutional (community) languages.

Population
  5,937 living spoken languages (main sample of scripts 02-06).
  Institutional  = EGIDS 0-4 (international, national, provincial, wider communication, educational)
  Non-institutional (community) = EGIDS 5-8b (Stable, In trouble, Dying)

Precipitation variables (CLIMRISK; percent change of annual precipitation relative to 1900)
  P_H  no-SAI high-emission world        P_S  same emissions + G6sulfur SAI
  P_M  mitigation world with the same global mean temperature as S: P_M = pi_k * G_S
       (pattern scaling, so P_M has the same sign as P_H and a smaller magnitude)
  R_P  = P_S - P_M  residual precipitation (percentage points, pp, of the 1900 baseline)

Classification (dead-band 2 pp, config precip_deadband_pct)
  drier   R_P < -2      wetter   R_P > +2      similar otherwise

Member-level language values use the hybrid of script 04:
  X_l,m = Xbar_l + sum_c s_lc (Xc_c,m - mean_m Xc_c), for X in {P_H, P_S, P_M, R_P}.

Outputs
  results/tables/dry_*.csv            aggregated tables (public)
  data/interim/dry_lang.parquet       language-level (RESTRICTED)
  data/interim/dry_member.npz         language x member arrays (RESTRICTED)
  numbers.json entries  dry.*
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, record  # noqa: E402

I = P("interim")
T = P("results")
DB = CFG["analysis"]["precip_deadband_pct"]
NB = CFG["analysis"]["n_bootstrap"]
rng = np.random.default_rng(CFG["analysis"]["random_seed"])
NONINST = ["Stable", "In trouble", "Dying"]

lm = pd.read_parquet(I / "lang_metrics.parquet")
lx = pd.read_parquet(I / "lang_exposure.parquet").set_index("ISO")
for c in ["PH_2100", "PS_2100", "PM_2100", "PH_2050", "PS_2050", "PM_2050", "Area"]:
    if c not in lm:
        lm[c] = lm.ISO.map(lx[c])
lm["noninst"] = lm.vitality.isin(NONINST)
lm["drier"] = lm.RP_2100 < -DB
lm["wetter"] = lm.RP_2100 > DB
lm["drier_2050"] = lm.RP_2050 < -DB
n_all, n_non = len(lm), int(lm.noninst.sum())
record("dry.n_languages", n_all)
record("dry.n_noninst", n_non)
record("dry.n_inst", n_all - n_non)
record("dry.share_noninst_of_all", n_non / n_all)

# ------------------------------------------------------------------ member-level arrays
z = np.load(I / "lang_member_R.npz", allow_pickle=True)
assert (z["ISO"] == lm.ISO.to_numpy()).all()
cm = np.load(I / "country_member_fields.npz", allow_pickle=True)
iso3 = list(cm["iso3"])
pidx = cm["pattern_index"]
cs = pd.read_parquet(I / "lang_country_shares.parquet")
S = np.zeros((n_all, len(iso3)))
pos = {iso: i for i, iso in enumerate(lm.ISO)}
ci = {c: i for i, c in enumerate(iso3)}
for r in cs.itertuples():
    if r.ISO in pos and r.CTRY3 in ci:
        S[pos[r.ISO], ci[r.CTRY3]] += r.country_share
mem = {}
for h in [2050, 2100]:
    mem[f"RP_{h}"] = z[f"RP_{h}"]
    for v in ["PH", "PS", "PM"]:
        Xc = cm[f"{v}_{h}"]
        dev = np.nan_to_num(Xc - np.nanmean(Xc, axis=0, keepdims=True))
        mem[f"{v}_{h}"] = lm[f"{v}_{h}"].to_numpy()[:, None] + S @ dev.T
# consistency: member-mean of P_S - P_M equals the stored R_P member array mean
record("dry.check_member_PS_minus_PM_vs_RP_maxabs",
       float(np.nanmax(np.abs((mem["PS_2100"] - mem["PM_2100"]).mean(1) - mem["RP_2100"].mean(1)))))
np.savez_compressed(I / "dry_member.npz", ISO=lm.ISO.to_numpy(), pattern=pidx, **mem)

non = lm.noninst.to_numpy()
ins = ~non


def boot_share(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (NB, len(x)), replace=True).mean(1)
    return np.percentile(b, [2.5, 97.5])


# ------------------------------------------------------------------ D1-D3 headline (ensemble mean)
for lab, m in [("noninst", non), ("inst", ins), ("all", np.ones(n_all, bool))]:
    d = lm[m]
    record(f"dry.{lab}.share_drier", float(d.drier.mean()))
    record(f"dry.{lab}.share_drier_ci", boot_share(d.drier))
    record(f"dry.{lab}.share_wetter", float(d.wetter.mean()))
    record(f"dry.{lab}.share_similar", float(1 - d.drier.mean() - d.wetter.mean()))
    record(f"dry.{lab}.n_drier", int(d.drier.sum()))
    record(f"dry.{lab}.RP_median", float(d.RP_2100.median()))
    record(f"dry.{lab}.RP_p25_p75", [float(d.RP_2100.quantile(.25)), float(d.RP_2100.quantile(.75))])
    for thr in [5, 10]:
        record(f"dry.{lab}.share_RP_below_minus{thr}", float((d.RP_2100 < -thr).mean()))
    record(f"dry.{lab}.share_drier_2050", float(d.drier_2050.mean()))
    record(f"dry.{lab}.RP_median_2050", float(d.RP_2050.median()))
for g in NONINST + ["Institutional"]:
    d = lm[lm.vitality == g]
    record(f"dry.class.{g.replace(' ', '_')}.share_drier", float(d.drier.mean()))
    record(f"dry.class.{g.replace(' ', '_')}.n", len(d))
d = lm[non]
record("dry.noninst.share_drier_endangered", float(d[d.vitality != "Stable"].drier.mean()))
record("dry.noninst.share_drier_stable", float(d[d.vitality == "Stable"].drier.mean()))
record("dry.gap_noninst_minus_inst_share_drier", float(lm[non].drier.mean() - lm[ins].drier.mean()))
diffs = []
xn, xi = lm[non].drier.to_numpy(float), lm[ins].drier.to_numpy(float)
for _ in range(NB):
    diffs.append(rng.choice(xn, len(xn)).mean() - rng.choice(xi, len(xi)).mean())
record("dry.gap_noninst_minus_inst_share_drier_ci", np.percentile(diffs, [2.5, 97.5]))

# speakers and communities
dn = lm[non & lm.drier.to_numpy()]
record("dry.noninst.drier.L1_total", float(dn.L1_Users.sum()))
record("dry.noninst.drier.L1_median_per_language", float(dn.L1_Users.median()))
record("dry.noninst.L1_total", float(lm[non].L1_Users.sum()))
record("dry.noninst.share_speakers_drier", float(dn.L1_Users.sum() / lm[non].L1_Users.sum()))
record("dry.noninst.share_territory_drier",
       float(lm[non].territory_km2.where(lm[non].drier, 0).sum() / lm[non].territory_km2.sum()))

# ------------------------------------------------------------------ D4 members and patterns
RPm = mem["RP_2100"]
sh_non = (RPm[non] < -DB).mean(0)
sh_ins = (RPm[ins] < -DB).mean(0)
med_non = np.median(RPm[non], 0)
record("dry.member.noninst.share_drier.p17_p50_p83", np.percentile(sh_non, [17, 50, 83]))
record("dry.member.noninst.share_drier.min_max", [float(sh_non.min()), float(sh_non.max())])
record("dry.member.noninst.n_members_majority_drier", int((sh_non > 0.5).sum()))
record("dry.member.noninst.median_RP.p17_p50_p83", np.percentile(med_non, [17, 50, 83]))
record("dry.member.noninst.n_members_median_negative", int((med_non < 0).sum()))
record("dry.member.inst.share_drier.p17_p50_p83", np.percentile(sh_ins, [17, 50, 83]))
record("dry.member.gap_noninst_minus_inst.p17_p50_p83", np.percentile(sh_non - sh_ins, [17, 50, 83]))
record("dry.member.gap_noninst_minus_inst.n_pos", int((sh_non - sh_ins > 0).sum()))
# per-language robustness: fraction of the 60 members in which the language is drier
fr = (RPm < -DB).mean(1)
lm["frac_members_drier"] = fr
record("dry.noninst.share_drier_in_ge_5of6_patterns",
       float(((np.stack([(RPm[:, pidx == k] < -DB).mean(1) > 0.5 for k in np.unique(pidx)], 1)).sum(1) >= 5)[non].mean()))
record("dry.noninst.share_drier_in_ge_two_thirds_members", float((fr[non] >= 2 / 3).mean()))
pat_rows = []
for k in np.unique(pidx):
    sel = pidx == k
    pat_rows.append({"pattern": int(k), "dG_mean": None,
                     "share_drier_noninst": float(sh_non[sel].mean()),
                     "share_drier_inst": float(sh_ins[sel].mean()),
                     "median_RP_noninst": float(med_non[sel].mean())})
pat = pd.DataFrame(pat_rows).drop(columns="dG_mean")
pat.to_csv(T / "dry_by_pattern.csv", index=False)
record("dry.pattern.noninst.share_drier_min_max", [float(pat.share_drier_noninst.min()),
                                                    float(pat.share_drier_noninst.max())])
pd.DataFrame({"member": np.arange(60), "pattern": pidx, "share_drier_noninst": sh_non,
              "share_drier_inst": sh_ins, "median_RP_noninst": med_non}).to_csv(
    T / "dry_by_member.csv", index=False)

# thresholds
thr_rows = []
for thr in np.arange(0, 20.5, 0.5):
    thr_rows.append({"threshold_pp": thr,
                     "share_noninst": float((lm[non].RP_2100 < -thr).mean()),
                     "share_inst": float((lm[ins].RP_2100 < -thr).mean()),
                     "member_p17_noninst": float(np.percentile((RPm[non] < -thr).mean(0), 17)),
                     "member_p83_noninst": float(np.percentile((RPm[non] < -thr).mean(0), 83))})
pd.DataFrame(thr_rows).to_csv(T / "dry_thresholds.csv", index=False)

# ------------------------------------------------------------------ D6-D7 absolute rainfall
PH, PS, PM = (lm[f"{v}_2100"].to_numpy() for v in ["PH", "PS", "PM"])
for lab, m in [("noninst", non), ("inst", ins)]:
    record(f"dry.{lab}.share_below1900.S", float((PS[m] < 0).mean()))
    record(f"dry.{lab}.share_below1900.M", float((PM[m] < 0).mean()))
    record(f"dry.{lab}.share_below1900.H", float((PH[m] < 0).mean()))
    flip = (PM[m] > 0) & (PS[m] < 0)
    record(f"dry.{lab}.n_flip_wetter_under_M_to_below1900_under_S", int(flip.sum()))
    record(f"dry.{lab}.share_flip", float(flip.mean()))
    record(f"dry.{lab}.share_S_drier_than_H_by_2pp", float((PS[m] - PH[m] < -DB).mean()))
    record(f"dry.{lab}.share_S_below_both_H_and_M", float(((PS[m] < PH[m] - DB) & (PS[m] < PM[m] - DB)).mean()))
# member-level robustness of the 1900 comparison
for lab, m in [("noninst", non)]:
    s_S = (mem["PS_2100"][m] < 0).mean(0)
    s_M = (mem["PM_2100"][m] < 0).mean(0)
    record(f"dry.member.{lab}.share_below1900.S.p17_p50_p83", np.percentile(s_S, [17, 50, 83]))
    record(f"dry.member.{lab}.share_below1900.M.p17_p50_p83", np.percentile(s_M, [17, 50, 83]))
    record(f"dry.member.{lab}.n_members_S_more_below1900_than_M", int((s_S > s_M).sum()))
    fl = ((mem["PM_2100"][m] > 0) & (mem["PS_2100"][m] < 0)).sum(0)
    record(f"dry.member.{lab}.n_flip.p17_p50_p83", np.percentile(fl, [17, 50, 83]))
    sh = (mem["PS_2100"][m] - mem["PH_2100"][m] < -DB).mean(0)
    record(f"dry.member.{lab}.share_S_drier_than_H.p17_p50_p83", np.percentile(sh, [17, 50, 83]))
    record(f"dry.member.{lab}.n_members_majority_S_drier_than_H", int((sh > 0.5).sum()))
    pd.DataFrame({"member": np.arange(60), "pattern": pidx, "share_below1900_S": s_S,
                  "share_below1900_M": s_M, "n_flip": fl, "share_S_drier_than_H": sh}).to_csv(
        T / "dry_absolute_by_member.csv", index=False)

# ------------------------------------------------------------------ D8 temperature context
cool = lm.R_2100 < -CFG["analysis"]["temp_deadband_c"]
record("dry.noninst.share_cooler_and_drier", float((cool & lm.drier)[non].mean()))
record("dry.noninst.share_of_drier_that_are_cooler", float(cool[non & lm.drier.to_numpy()].mean()))
record("dry.noninst.share_cooler", float(cool[non].mean()))

# ------------------------------------------------------------------ D9 geography
geo = (lm[non].groupby("Area").agg(n=("ISO", "size"), n_drier=("drier", "sum"),
                                   share_drier=("drier", "mean"), median_RP=("RP_2100", "median"))
       .sort_values("n", ascending=False))
geo.to_csv(T / "dry_by_area.csv")
for a, r in geo.iterrows():
    record(f"dry.area.{a}.share_drier", float(r.share_drier))
    record(f"dry.area.{a}.n_drier", int(r.n_drier))
    record(f"dry.area.{a}.n", int(r.n))
top = (lm[non].groupby("Country_Code").agg(n=("ISO", "size"), n_drier=("drier", "sum"),
                                           share_drier=("drier", "mean"))
       .sort_values("n_drier", ascending=False))
top.to_csv(T / "dry_by_country.csv")
record("dry.top10_countries_share_of_drier_noninst", float(top.n_drier.head(10).sum() / top.n_drier.sum()))
lat = lm.lat_mean.abs() < 23.44
record("dry.noninst.share_in_tropics", float(lat[non].mean()))
record("dry.noninst.share_of_drier_in_tropics", float(lat[non & lm.drier.to_numpy()].mean()))

# ------------------------------------------------------------------ D10 within-country check (LIC polygons)
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.ISO.isin(lm.ISO) & pe.land_w_km2.notna() & ~pe.sign_language].copy()
pe["non"] = pe.vitality.isin(NONINST).astype(float)
pe["dr"] = (pe.RP_2100 < -DB).astype(float)
pe["w"] = 1.0 / pe.groupby("ISO").pid.transform("size")


def fe(y, x, w, g):
    d = pd.DataFrame({"y": y, "x": x, "w": w, "g": g}).dropna()
    ws = d.w.groupby(d.g).transform("sum")
    yd = d.y - (d.y * d.w).groupby(d.g).transform("sum") / ws
    xd = d.x - (d.x * d.w).groupby(d.g).transform("sum") / ws
    return (d.w * xd * yd).sum() / (d.w * xd ** 2).sum()


b = fe(pe.dr, pe.non, pe.w, pe.CTRY3)
ug = pe.CTRY3.unique()
idx = {k: np.flatnonzero(pe.CTRY3.to_numpy() == k) for k in ug}
bs = []
for _ in range(1000):
    pick = rng.choice(ug, len(ug))
    rows = np.concatenate([idx[k] for k in pick])
    grp = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(pick)])
    sub = pe.iloc[rows]
    bs.append(fe(sub.dr.to_numpy(), sub.non.to_numpy(), sub.w.to_numpy(), grp))
record("dry.within_country.gap_noninst_minus_inst_share_drier", float(b))
record("dry.within_country.gap_ci", np.percentile(bs, [2.5, 97.5]))
record("dry.lic.noninst.n_units", int((pe.non == 1).sum()))
record("dry.lic.noninst.share_indigenous", float((pe[pe.non == 1].Is_Indigenous == "T").mean()))
record("dry.lang.noninst.share_indigenous_somewhere",
       float(pe[pe.non == 1].groupby("ISO").Is_Indigenous.agg(lambda x: (x == "T").any()).mean()))
record("dry.lic.indigenous_noninst.share_drier",
       float(pe[(pe.non == 1) & (pe.Is_Indigenous == "T")].dr.mean()))
record("dry.lic.indigenous_share_of_noninst_drier_lics",
       float((pe[(pe.non == 1) & (pe.dr == 1)].Is_Indigenous == "T").mean()))

# ------------------------------------------------------------------ D11 sensitivity: hyper-arid, weights
ok = (np.abs(PH) <= 50)
record("dry.noninst.share_drier_excl_hyperarid", float(lm[non & ok].drier.mean()))
record("dry.noninst.n_excluded_hyperarid", int((non & ~ok).sum()))
record("dry.noninst.share_drier_speaker_weighted",
       float(np.average(lm[non].drier, weights=lm[non].L1_Users.fillna(0) + 1e-9)))
record("dry.noninst.share_drier_territory_weighted",
       float(np.average(lm[non].drier, weights=lm[non].territory_km2)))

# ------------------------------------------------------------------ save restricted language table
keep = ["ISO", "Country_Code", "Area", "vitality", "noninst", "EGIDS", "L1_Users", "territory_km2",
        "lat_mean", "R_2100", "RP_2100", "RP_2050", "PH_2100", "PS_2100", "PM_2100", "drier", "wetter",
        "frac_members_drier"]
lm[keep].to_parquet(I / "dry_lang.parquet", index=False)

summ = pd.DataFrame({
    "group": ["Non-institutional", "Institutional", "All"],
    "n": [n_non, n_all - n_non, n_all],
    "share_drier": [lm[non].drier.mean(), lm[ins].drier.mean(), lm.drier.mean()],
    "share_wetter": [lm[non].wetter.mean(), lm[ins].wetter.mean(), lm.wetter.mean()],
    "median_RP": [lm[non].RP_2100.median(), lm[ins].RP_2100.median(), lm.RP_2100.median()],
    "share_PS_below_1900": [(PS[non] < 0).mean(), (PS[ins] < 0).mean(), (PS < 0).mean()],
    "share_PM_below_1900": [(PM[non] < 0).mean(), (PM[ins] < 0).mean(), (PM < 0).mean()],
})
summ.to_csv(T / "dry_summary.csv", index=False)
print(summ.round(3).to_string())
print(geo.round(3).to_string())
print(pat.round(3).to_string())
print(top.head(12).round(3).to_string())
