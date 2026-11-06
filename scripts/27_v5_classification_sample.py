"""27 - Version 5 (Zariquiey's review): sample accounting and genealogical robustness (2100).

1. Sample accounting from Ethnologue 28's Table_of_Languages: living languages, sign languages,
   dormant languages (EGIDS 9, no first-language speakers), spoken languages without a mapped
   homeland (no polygon, or a polygon more than 150 km from any CLIMRISK land cell), and the
   analysed sample; the macro-area (Glottolog) and EGIDS composition of the languages left out;
   the share of endangered languages (EGIDS 6b-9) in Ethnologue 28 as a whole and in the sample;
   whether "Maipurean" is Ethnologue 28's own top-level label.
2. Lineage results under four classifications (Extended Data Table 2), from the same member-level
   homeland fractions as the main analysis (srmlang.baselines.setup):
     Glottolog 5.3 (main; script 21): spoken L1 languages, creoles with their lexifier family
     Glottolog 5.3, creoles excluded (creole = Ethnologue family label "Creole")
     Glottolog 5.3, each creole its own lineage
     Ethnologue 28, genealogical units only: Creole, Unclassified, Mixed language and Pidgin
       excluded; each "Language isolate" its own lineage
     Ethnologue 28, all labels: Creole counted as one family and each unclassified, mixed and
       pidgin language as its own lineage (the rule of our earlier drafts)
   For each: languages, lineages, families, isolates, mean over lineages of the share of homeland
   drier and wetter, dry:wet ratio, share of lineages mostly drier (ensemble-mean climate, member
   p17-p83, min and max), share of isolates mostly drier, members in which lineages are more often
   drier than land, people and the economy.

Outputs: results/tables/v5_classification_robustness.csv, results/tables/v5_sample_accounting.csv,
results/tables/latex/tabED2_classification.tex; ledger keys v5.*
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P, record  # noqa: E402
from srmlang.baselines import setup  # noqa: E402

I, T, LX = P("interim"), P("results"), P("results") / "latex"
LX.mkdir(parents=True, exist_ok=True)
S = setup()
lm, Lop, OUT, units = S["lm"], S["Lop"], S["OUT"], S["units"]
cw = pd.read_parquet(I / "glottolog_crosswalk.parquet").set_index("ISO")

# =================================================================== 1. sample accounting
rd = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
tl = pd.read_csv(P("eth_languages"), **rd).rename(columns={"ISO_639": "ISO"})
tl["ISO"] = tl.ISO.str.lower()
living = tl[tl.EGIDS != "x10"].copy()
living["sign"] = living.Family.fillna("").str.contains("Sign language", case=False)
living["dormant"] = (living.EGIDS == "9") & ~living.sign
living["mapped"] = living.ISO.isin(lm.ISO)
polys_raw = set(pd.read_parquet(I / "eth_polygons.parquet", columns=["ISO"]).ISO) | {"omg"}
living["has_polygon"] = living.ISO.isin(polys_raw)
spoken = living[~living.sign & ~living.dormant]
unm = spoken[~spoken.mapped]
assert spoken.mapped.sum() == len(lm), (spoken.mapped.sum(), len(lm))
acc = {"n_table": len(tl), "n_living": len(living), "n_sign": int(living.sign.sum()),
       "n_dormant": int(living.dormant.sum()),
       "n_dormant_with_polygon": int((living.dormant & living.ISO.isin(polys_raw)).sum()),
       "n_spoken_nondormant": len(spoken), "n_sample": int(spoken.mapped.sum()),
       "n_unmapped": len(unm), "n_unmapped_no_polygon": int((~unm.has_polygon).sum()),
       "n_unmapped_polygon_off_grid": int(unm.has_polygon.sum()),
       "n_unmapped_with_glottolog_point": int(unm.ISO.map(cw.g_lat).notna().sum())}
for k, v in acc.items():
    record(f"v5.sample.{k}", v)
END = ["6b", "7", "8a", "8b", "9"]
record("v5.sample.eth28_endangered_n", int(living.EGIDS.isin(END).sum()))
record("v5.sample.eth28_endangered_share", float(living.EGIDS.isin(END).mean()))
record("v5.sample.eth28_endangered_share_spoken", float(living[~living.sign].EGIDS.isin(END).mean()))
record("v5.sample.sample_endangered_n", int(lm.EGIDS.isin(END).sum()))
record("v5.sample.sample_endangered_share", float(lm.EGIDS.isin(END).mean()))
record("v5.sample.n_noninst", int(S["non"].sum()))
# composition of the unmapped spoken languages against the sample (Glottolog macro-area, EGIDS)
macro = lambda d: d.ISO.map(cw.g_macroarea).fillna("(no Glottolog match)").str.split(";").str[0]  # noqa: E731
comp = pd.concat({"unmapped": macro(unm).value_counts(), "sample": macro(lm).value_counts()}, axis=1).fillna(0)
comp_e = pd.concat({"unmapped": unm.EGIDS.value_counts(), "sample": lm.EGIDS.value_counts()}, axis=1).fillna(0)
comp.rename_axis("macroarea").to_csv(T / "v5_unmapped_by_macroarea.csv")
comp_e.to_csv(T / "v5_unmapped_by_egids.csv")
record("v5.sample.unmapped_by_macroarea", {k: int(v) for k, v in comp.unmapped.items()})
record("v5.sample.sample_by_macroarea", {k: int(v) for k, v in comp["sample"].items()})
record("v5.sample.unmapped_share_by_macroarea", (comp.unmapped / (comp.unmapped + comp["sample"])).round(4).to_dict())
record("v5.sample.unmapped_by_egids", {k: int(v) for k, v in comp_e.unmapped.items()})
record("v5.sample.unmapped_endangered_share", float(unm.EGIDS.isin(END).mean()))
record("v5.sample.unmapped_noninst_share", float((~unm.EGIDS.isin(["0", "1", "2", "3", "4"])).mean()))
record("v5.sample.unmapped_by_eth_area", unm.Area.value_counts().to_dict())
record("v5.sample.dormant_by_eth_area", living[living.dormant].Area.value_counts().to_dict())
# "Maipurean": Ethnologue 28's own top-level label?
mai = tl[tl.Family == "Maipurean"]
record("v5.maipurean.n_table", len(mai))
record("v5.maipurean.top_level_of_classification",
       sorted(mai.Classification.fillna("").str.split(",").str[0].str.strip().unique().tolist()))
mai_s = lm[lm.Family == "Maipurean"]
record("v5.maipurean.n_sample", len(mai_s))
record("v5.maipurean.glottolog_lineages", mai_s.ISO.map(cw.g_lineage_name).value_counts().to_dict())
pd.DataFrame([acc]).to_csv(T / "v5_sample_accounting.csv", index=False)

# =================================================================== 2. classification robustness
f0d = np.asarray(Lop @ OUT["drier"][0].astype(float)).ravel()
f0w = np.asarray(Lop @ OUT["wetter"][0].astype(float)).ravel()
fmd = np.asarray(Lop @ OUT["drier"][1].astype(float))
fmw = np.asarray(Lop @ OUT["wetter"][1].astype(float))
u_m = units(OUT["drier"][1])
land_m, ppl_m, gdp_m = u_m["land"], u_m["people (L1 speakers)"], u_m["economy (GDP)"]
land_w = units(OUT["wetter"][1])["land"]
fam_e = lm.Family.fillna("")
creole = (fam_e == "Creole").to_numpy()
g_lin = lm.ISO.map(cw.g_lineage).where(lm.ISO.map(cw.g_category) == "Spoken_L1_Language")
g_iso = lm.ISO.map(cw.g_is_isolate).astype(bool)
NONGEN = {"Creole", "Unclassified", "Mixed language", "Pidgin"}


def variant(labels: pd.Series, isolate: pd.Series, name: str, key: str) -> dict:
    ok = labels.notna().to_numpy()
    lab = labels[ok].to_numpy()
    iso_flag = pd.Series(isolate[ok].to_numpy(), index=lab).groupby(level=0).first()

    def lmean(x):
        return pd.DataFrame(x[ok]).groupby(lab).mean()
    d0, w0 = lmean(f0d)[0], lmean(f0w)[0]
    dm, wm = lmean(fmd), lmean(fmw)
    mostly_m = (dm > 0.5).mean(0).to_numpy()
    r = {"classification": name, "languages": int(ok.sum()), "lineages": int(d0.size),
         "families": int((~iso_flag).sum()), "isolates": int(iso_flag.sum()),
         "drier": float(d0.mean()), "wetter": float(w0.mean()), "dry_to_wet": float(d0.mean() / w0.mean()),
         "mostly_drier": float((d0 > 0.5).mean()),
         "mostly_drier_p17": float(np.percentile(mostly_m, 17)), "mostly_drier_p83": float(np.percentile(mostly_m, 83)),
         "mostly_drier_min": float(mostly_m.min()), "mostly_drier_max": float(mostly_m.max()),
         "n_members_mostly_drier_gt_half": int((mostly_m > 0.5).sum()),
         "isolates_mostly_drier": float((d0[iso_flag.reindex(d0.index).to_numpy()] > 0.5).mean()),
         "drier_member_p17": float(np.percentile(dm.mean(0), 17)), "drier_member_p83": float(np.percentile(dm.mean(0), 83)),
         "n_members_gt_land": int((dm.mean(0).to_numpy() > land_m).sum()),
         "n_members_gt_people": int((dm.mean(0).to_numpy() > ppl_m).sum()),
         "n_members_gt_economy": int((dm.mean(0).to_numpy() > gdp_m).sum()),
         "n_members_ratio_gt_land": int(((dm.mean(0) / wm.mean(0)).to_numpy() > land_m / land_w).sum())}
    for k, v in r.items():
        if k != "classification":
            record(f"v5.class.{key}.{k}", v)
    return r


rows = [variant(g_lin, g_iso, "Glottolog 5.3 (main)", "glottolog")]
rows.append(variant(g_lin.where(~creole), g_iso, "Glottolog 5.3, creoles excluded", "glottolog_no_creoles"))
g_cre = g_lin.where(~creole | g_lin.isna(), "CREOLE:" + lm.ISO)
rows.append(variant(g_cre, g_iso | pd.Series(creole, index=lm.index), "Glottolog 5.3, each creole a lineage",
                    "glottolog_creole_lineages"))
e_iso = fam_e == "Language isolate"
e_lin = pd.Series(np.where(e_iso, "ISO:" + lm.ISO, fam_e), index=lm.index).where(~fam_e.isin(NONGEN) & (fam_e != ""))
rows.append(variant(e_lin, e_iso, "Ethnologue 28, genealogical units only", "ethnologue_genealogical"))
single = fam_e.isin({"Language isolate", "Unclassified", "Mixed language", "Pidgin"})
e_all = pd.Series(np.where(single, "ISO:" + lm.ISO, fam_e), index=lm.index).where(fam_e != "")
rows.append(variant(e_all, single, "Ethnologue 28, all labels", "ethnologue_all_labels"))
rob = pd.DataFrame(rows)
record("v5.class.ethnologue_nongenealogical_labels_in_sample",
       {k: int((fam_e == k).sum()) for k in ["Creole", "Unclassified", "Mixed language", "Pidgin"]})
rob.to_csv(T / "v5_classification_robustness.csv", index=False)
record("v5.class.ethnologue_n_families_mapping_to_one_glottolog_lineage", int(
    lm[g_lin.notna() & ~fam_e.isin(NONGEN | {"Language isolate"})].assign(g=g_lin).groupby("Family").g.nunique().eq(1).sum()))
record("v5.class.ethnologue_isolates_also_glottolog_isolates", int((e_iso & g_iso & g_lin.notna()).sum()))

pc = lambda v: f"{100 * v:.0f}"  # noqa: E731
lines = []
for r in rob.itertuples():
    lab = {"Glottolog 5.3 (main)": "Glottolog 5.3 (main)",
           "Glottolog 5.3, creoles excluded": "\\quad creoles excluded",
           "Glottolog 5.3, each creole a lineage": "\\quad each creole a lineage",
           "Ethnologue 28, genealogical units only": "Ethnologue 28, genealogical units only",
           "Ethnologue 28, all labels": "Ethnologue 28, all labels"}[r.classification]
    lines.append(f"{lab} & {r.languages:,} & {r.lineages} & {r.families} & {r.isolates} & {pc(r.drier)} & "
                 f"{pc(r.wetter)} & {r.dry_to_wet:.1f} & {pc(r.mostly_drier)} ({pc(r.mostly_drier_p17)}--{pc(r.mostly_drier_p83)}) & "
                 f"{pc(r.isolates_mostly_drier)} \\\\")
(LX / "tabED2_classification.tex").write_text("\n".join(lines) + "\n")
pd.set_option("display.width", 250)
print(rob.round(3).to_string())
print(comp, comp_e, sep="\n")
