"""23 - Against what? Exposure of linguistic heritage to SAI drying AND residual warming,
compared with land, populated land, people and the economy (2100, SAI vs iso-temperature mitigation).

Outcomes (dead-bands as in the paper)
  drier   R_P < -2 pp       wetter  R_P > +2 pp
  warmer  R   > +0.1 C      cooler  R   < -0.1 C

Units (all from the same cell-level member fields, so they are directly comparable)
  land            area-weighted, all land north of 60 S
  populated land  area-weighted, CLIMRISK economic cells
  economy         GDP-weighted (CLIMRISK cell PV GDP, script 15)
  people          first-language speakers, each language-in-country polygon weighted by its
                  L1 speakers (all living languages), polygon value = area-weighted cell share
  non-institutional languages   mean over non-institutional languages of the fraction of the
                  homeland in the outcome (each language counted once; script 21)
  linguistic diversity hotspots   cells holding half of non-institutional-language weighted endemism
  lineages        mean over top-level lineages (families; isolates each their own) of the mean
                  homeland fraction of their languages (each lineage counted once)

Cell values per member: hybrid of script 04 at cell level.
  R_c,m  = (Rbar_c / mean dG) dG_m + v_{country(c),k(m)} dG_m
  RP_c,m = RPbar_c + (RP_country,m - mean_m RP_country)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, record  # noqa: E402
from srmlang import heritage as hm  # noqa: E402

I, T = P("interim"), P("results")
DBP, DBT = CFG["analysis"]["precip_deadband_pct"], CFG["analysis"]["temp_deadband_c"]
NONINST = ["Stable", "In trouble", "Dying"]

# cell-level member fields, outcome masks, unit weights and units() live in srmlang.baselines
# (shared with script 26); the names below (grid, area, Rm, RPm, R0, RP0, OUT, Lop, Pop, Gop,
# gn, codes, lin, hl, lm, non, gdp, popl, hot, lwe, L1, pidx, units, ...) come from there.
from srmlang.baselines import setup  # noqa: E402
globals().update(setup())
record("base.check_member_mean_R_vs_map_maxabs", float(np.nanmax(np.abs(Rm.mean(1) - R0))))  # noqa: F821


# self-check: the sparse operators reproduce the plain definitions in srmlang.heritage
_x = OUT["drier"][0]
_u = units(_x)
_ok = lin.notna().to_numpy()
record("base.check_units_vs_module_maxabs", float(max(
    abs(_u["land"][0] - hm.weighted_share(area, _x)),
    abs(_u["economy (GDP)"][0] - hm.weighted_share(gdp, _x)),
    abs(_u["non-institutional languages"][0] - hm.homeland_fraction(Lop, _x)[non].mean()),
    abs(_u["language lineages"][0] - hm.lineage_mean(hm.homeland_fraction(Lop, _x)[_ok], lin[_ok].to_numpy())))))

rows = []
member_rows = []
for oc, (x0, xm) in OUT.items():
    u0, um = units(x0), units(xm)
    for k_, arr in um.items():
        for mbr, val in enumerate(arr):
            member_rows.append({"outcome": oc, "unit": k_, "member": mbr, "pattern": int(pidx[mbr]), "share": float(val)})
    land_m = um["land"]
    for k in u0:
        rr = um[k] / land_m
        row = {"outcome": oc, "unit": k, "ensemble_mean_climate": float(u0[k][0]),
               "member_p17": np.percentile(um[k], 17), "member_p50": np.percentile(um[k], 50),
               "member_p83": np.percentile(um[k], 83),
               "ratio_vs_land": float(u0[k][0] / u0["land"][0]),
               "ratio_member_p17": np.percentile(rr, 17), "ratio_member_p50": np.percentile(rr, 50),
               "ratio_member_p83": np.percentile(rr, 83), "n_members_ratio_gt1": int((rr > 1).sum())}
        rows.append(row)
        key = f"base.{oc}.{k.split(' (')[0].replace(' ', '_').replace('-', '_')}"
        for kk in ["ensemble_mean_climate", "ratio_vs_land", "n_members_ratio_gt1"]:
            record(f"{key}.{kk}", row[kk])
        record(f"{key}.member_p17_p50_p83", [row["member_p17"], row["member_p50"], row["member_p83"]])
        record(f"{key}.ratio_member_p17_p50_p83", [row["ratio_member_p17"], row["ratio_member_p50"],
                                                   row["ratio_member_p83"]])
tab = pd.DataFrame(rows)
pd.DataFrame(member_rows).to_csv(T / "baselines_by_member.csv", index=False)
tab.to_csv(T / "baselines_rain_heat.csv", index=False)

# lineage table for temperature (which lineages are mostly warmer / cooler)
fw = Lop @ OUT["warmer"][0].astype(float)
fc = Lop @ OUT["cooler"][0].astype(float)
fwm = Lop @ OUT["warmer"][1].astype(float)
lt = pd.DataFrame({"lineage": codes.categories, "name": pd.Series(codes.categories).map(
                       hl.lineage_name.groupby(hl.lineage).first()).to_numpy(), "n": gn,
                   "mean_f_warmer": (Gop @ fw) / gn, "mean_f_cooler": (Gop @ fc) / gn,
                   "frac_members_majority_warmer": (((Gop @ fwm) / gn[:, None]) > 0.5).mean(1)})
lt = lt.sort_values("n", ascending=False)
lt.to_parquet(I / "heritage_lineages_temperature_full.parquet")           # RESTRICTED: includes isolates
lt[lt.n >= 2].to_csv(T / "heritage_families_temperature.csv", index=False)  # public: lineages with >= 2 languages
record("base.lineages.n_mostly_warmer", int((lt.mean_f_warmer > 0.5).sum()))
record("base.lineages.n_mostly_cooler", int((lt.mean_f_cooler > 0.5).sum()))
# store per-language temperature fractions for the tree figure
hl2 = hl.copy()
hl2["f_warmer"] = pd.Series(np.asarray(fw).ravel(), index=lm.ISO)
hl2["f_cooler"] = pd.Series(np.asarray(fc).ravel(), index=lm.ISO)
hl2.reset_index().to_parquet(I / "heritage_lang.parquet", index=False)
# member-level homeland fractions for the tree triptych (drier)
np.savez_compressed(I / "heritage_member_fdry.npz", ISO=lm.ISO.to_numpy(),
                    f=np.asarray(Lop @ OUT["drier"][1].astype(float)))
pd.set_option("display.width", 220)
print(tab.round(3).to_string())
print(lt.sort_values("mean_f_warmer", ascending=False).head(12).round(3).to_string())

# ---------------------------------------------------------------- odds of losing vs gaining rain
d_m, w_m = units(OUT["drier"][1]), units(OUT["wetter"][1])
d_0, w_0 = units(OUT["drier"][0]), units(OUT["wetter"][0])
odds_rows = []
for k in d_0:
    o0 = d_0[k][0] / w_0[k][0]
    om = d_m[k] / w_m[k]
    ol = d_m["land"] / w_m["land"]
    odds_rows.append({"unit": k, "dry_to_wet_ratio": o0, "member_p17": np.percentile(om, 17),
                      "member_p50": np.percentile(om, 50), "member_p83": np.percentile(om, 83),
                      "relative_to_land": o0 / (d_0["land"][0] / w_0["land"][0]),
                      "n_members_above_land": int((om > ol).sum())})
    key = f"base.odds.{k.split(' (')[0].replace(' ', '_').replace('-', '_')}"
    record(f"{key}.dry_to_wet", o0)
    record(f"{key}.member_p17_p50_p83", np.percentile(om, [17, 50, 83]))
    record(f"{key}.n_members_above_land", int((om > ol).sum()))
od = pd.DataFrame(odds_rows)
od.to_csv(T / "baselines_dry_to_wet.csv", index=False)
# non-institutional languages and lineages vs people and economy, member by member (drier)
for a in ["non-institutional languages", "language lineages", "linguistic diversity hotspots"]:
    for b in ["people (L1 speakers)", "economy (GDP)", "populated land"]:
        ka = a.split(' (')[0].replace(' ', '_').replace('-', '_')
        kb = b.split(' (')[0].replace(' ', '_')
        record(f"base.drier.{ka}_vs_{kb}.n_members_gt", int((d_m[a] > d_m[b]).sum()))
        record(f"base.drier.{ka}_vs_{kb}.ratio_member_p17_p50_p83", np.percentile(d_m[a] / d_m[b], [17, 50, 83]))
print(od.round(2).to_string())
