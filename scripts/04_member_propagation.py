"""04 - Propagate the 60-member ensemble to language-level residuals.

The 0.5-degree maps carry only ensemble statistics, while country aggregates carry
every member. We therefore build member-level language residuals with a hybrid
that keeps the spatial detail of the map and the member spread of the countries:

  Temperature (pattern-scaling algebra, per unit of global cooling dG_m):
    R_l,m = u_l * dG_m + sum_c s_lc * v_c,k(m) * dG_m
      u_l      = Rbar_l / mean(dG)              (ensemble-mean map, per unit cooling)
      v_c,k    = (beta_c,k + gamma_c,k) - mean_k'(beta_c,k' + gamma_c,k')
                 pattern-specific deviation of the per-unit residual in country c
      s_lc     = share of language l's territory in country c

  Precipitation (multiplicative composition is not separable, so additive):
    RP_l,m = RPbar_l + sum_c s_lc * (RPc_c,m - mean_m RPc_c,m)

Outputs
  data/interim/lang_member_R.npz           (RESTRICTED) language x member arrays
  results/tables/member_group_stats.csv    group statistics for each member
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, LIVING_ORDER, P, ensure_dirs, record  # noqa: E402

ensure_dirs()
I = P("interim")
cm = np.load(I / "country_member_fields.npz", allow_pickle=True)
iso3 = list(cm["iso3"])
pidx = cm["pattern_index"]
beta, gamma = cm["beta"], cm["gamma"]          # (60, 187)
GH, GS, years = cm["GH"], cm["GS"], cm["years"]

lx = pd.read_parquet(I / "lang_exposure.parquet")
cs = pd.read_parquet(I / "lang_country_shares.parquet")
sample = lx[lx.vitality.isin(LIVING_ORDER) & ~lx.sign_language].reset_index(drop=True)
iso_index = {c: i for i, c in enumerate(iso3)}

# sparse language x country share matrix
S = np.zeros((len(sample), len(iso3)))
pos = {iso: i for i, iso in enumerate(sample.ISO)}
for r in cs.itertuples():
    if r.ISO in pos and r.CTRY3 in iso_index:
        S[pos[r.ISO], iso_index[r.CTRY3]] += r.country_share

out_rows = []
store = {}
for h in CFG["analysis"]["horizons"]:
    j = int(np.where(years == h)[0][0])
    dG = GH[:, j] - GS[:, j]                          # (60,)
    unit = beta + gamma                               # per-unit residual, (60, 187)
    # pattern-mean per-unit residual in each country (equal weight on 6 patterns)
    pats = np.unique(pidx)
    unit_k = np.stack([unit[pidx == k][0] for k in pats])   # (6, 187)
    v = unit - unit_k.mean(axis=0, keepdims=True)           # (60, 187)
    v = np.nan_to_num(v)
    u = sample[f"R_{h}"].to_numpy() / dG.mean()             # (L,)
    R_lm = u[:, None] * dG[None, :] + (S @ v.T) * dG[None, :]   # (L, 60)

    RPc = cm[f"RP_{h}"]                                     # (60, 187)
    dev = np.nan_to_num(RPc - np.nanmean(RPc, axis=0, keepdims=True))
    RP_lm = sample[f"RP_{h}"].to_numpy()[:, None] + S @ dev.T   # (L, 60)
    store[f"R_{h}"] = R_lm
    store[f"RP_{h}"] = RP_lm

    vit = sample.vitality.to_numpy()
    end = vit == "In trouble"
    end = np.isin(vit, ["In trouble", "Dying"])
    for m in range(R_lm.shape[1]):
        r = {"horizon": h, "member": m, "pattern": int(pidx[m]), "dG": dG[m]}
        for g in LIVING_ORDER:
            sel = vit == g
            r[f"R_mean_{g}"] = R_lm[sel, m].mean()
            r[f"RP_mean_{g}"] = RP_lm[sel, m].mean()
            r[f"RP_median_{g}"] = np.median(RP_lm[sel, m])
            r[f"share_Rpos_{g}"] = (R_lm[sel, m] > CFG["analysis"]["temp_deadband_c"]).mean()
            r[f"share_RPneg_{g}"] = (RP_lm[sel, m] < -CFG["analysis"]["precip_deadband_pct"]).mean()
        r["gap_R_end_minus_non"] = R_lm[end, m].mean() - R_lm[~end, m].mean()
        r["gap_R_dying_minus_inst"] = r["R_mean_Dying"] - r["R_mean_Institutional"]
        r["gap_RP_end_minus_non"] = RP_lm[end, m].mean() - RP_lm[~end, m].mean()
        r["RR_Rpos_dying_vs_inst"] = r["share_Rpos_Dying"] / max(r["share_Rpos_Institutional"], 1e-9)
        r["RR_Rpos_end_vs_non"] = (R_lm[end, m] > 0.1).mean() / max((R_lm[~end, m] > 0.1).mean(), 1e-9)
        out_rows.append(r)

np.savez_compressed(I / "lang_member_R.npz", ISO=sample.ISO.to_numpy(), **store)
ms = pd.DataFrame(out_rows)
ms.to_csv(P("results") / "member_group_stats.csv", index=False)

for h in CFG["analysis"]["horizons"]:
    d = ms[ms.horizon == h]
    for col in ["gap_R_end_minus_non", "gap_R_dying_minus_inst", "gap_RP_end_minus_non",
                "RR_Rpos_dying_vs_inst", "RR_Rpos_end_vs_non"] + \
               [f"R_mean_{g}" for g in LIVING_ORDER] + [f"RP_median_{g}" for g in LIVING_ORDER]:
        q = np.percentile(d[col], [17, 50, 83])
        record(f"member.{col}.{h}.p17_p50_p83", q)
        record(f"member.{col}.{h}.frac_pos", float((d[col] > 0).mean()))
    print(h, d[["gap_R_end_minus_non", "gap_R_dying_minus_inst", "gap_RP_end_minus_non",
                "RR_Rpos_dying_vs_inst"]].describe(percentiles=[.17, .5, .83]).round(3))

# ---------------------------------------------------------------- structural (pattern) spread
# Per language: mean residual within each of the six response patterns, and its range.
for h in CFG["analysis"]["horizons"]:
    Rk = np.stack([store[f"R_{h}"][:, pidx == k].mean(1) for k in np.unique(pidx)], 1)
    RPk = np.stack([store[f"RP_{h}"][:, pidx == k].mean(1) for k in np.unique(pidx)], 1)
    rng_R = Rk.max(1) - Rk.min(1)
    record(f"pattern.R_range_median.{h}", float(np.median(rng_R)))
    record(f"pattern.RP_range_median.{h}", float(np.median(RPk.max(1) - RPk.min(1))))
    for g in LIVING_ORDER:
        sel = sample.vitality.to_numpy() == g
        record(f"pattern.R_range_median.{h}.{g.replace(' ', '_')}", float(np.median(rng_R[sel])))
    hi = (sample.lat_mean > 55).to_numpy()
    record(f"pattern.highlat_R_by_pattern.{h}", Rk[hi].mean(0))
    record(f"pattern.highlat_n", int(hi.sum()))
    record(f"pattern.highlat_share_end", float(np.isin(sample.vitality[hi], ["In trouble", "Dying"]).mean()))
    trop = (sample.lat_mean.abs() < 23.44).to_numpy()
    record(f"pattern.tropics_R_by_pattern.{h}", Rk[trop].mean(0))
    # share of languages for which >= 5 of 6 patterns agree on the sign of R and of RP
    agreeR = np.maximum((Rk > 0).sum(1), (Rk < 0).sum(1)) >= 5
    agreeP = ((RPk < 0).sum(1) >= 5)
    record(f"pattern.share_lang_R_sign_agree5of6.{h}", float(agreeR.mean()))
    record(f"pattern.share_lang_RP_neg_5of6.{h}", float(agreeP.mean()))
print("pattern spread recorded")
