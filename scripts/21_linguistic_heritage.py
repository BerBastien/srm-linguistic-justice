"""21 - Rarity- and lineage-aware exposure of linguistic heritage to SAI drying (2100).

Idea (Bernie, 27 Oct 2026): one dry pixel of Spanish matters little because Spanish has
thousands of pixels; the only pixel of a small language is the whole language. We borrow
three tools from biodiversity science.

1. Linguistic weighted endemism (LWE). Range share r_lc = w_lc / sum_c w_lc of language l in
   land cell c (w = exact coverage x cell area, from script 03). Each language sums to one,
   so every language counts once whatever its size. The fraction of a language's homeland
   that is drier (R_P < -2 pp) is f_l = sum_c r_lc 1[dry_c]. Endemism-weighted share of
   linguistic heritage that is drier = mean_l f_l. Cell LWE_c = sum_l r_lc (sums to N).
   Baselines: share of land area drier (excluding Antarctica), of populated land, of GDP.
2. No refuge. Languages whose whole homeland is drier (f_l >= 0.999), by range-size quintile.
3. Language lineages (v5: Glottolog 5.3, script 02a). A lineage is a Glottolog top-level family or
   an isolate; only languages Glottolog counts as spoken L1 languages enter (bookkeeping,
   unattested, unclassifiable, pidgin and mixed codes are left out). Glottolog's classification
   paths, with unit branch lengths and every lineage joined to a common root by one branch (a
   computational convention, not a claim of relatedness), give an approximate topological
   distinctiveness (fair proportion): ED_l = sum over edges on the path to l of 1/(languages below
   the edge). The Ethnologue classification is kept as a robustness check (script 27).
   Phylogenetic-diversity analogue PD(S) = number of distinct edges needed to connect set S.
   We report the ED-weighted share drier and the share of PD spanned by languages whose
   homeland is mostly drier (f_l > 0.5), against random sets of the same size.

Climate uncertainty: cell values per member use the hybrid of script 04 at cell level,
RP_c,m = RPbar_c + (RP_country,m - mean_m RP_country), for cells with a CLIMRISK country.

Outputs: results/tables/heritage_*.csv (aggregated), numbers.json entries heritage.*,
data/interim/heritage_lang.parquet (RESTRICTED), data/interim/heritage_cells.parquet.
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
DB = CFG["analysis"]["precip_deadband_pct"]
rng = np.random.default_rng(CFG["analysis"]["random_seed"])
NONINST = ["Stable", "In trouble", "Dying"]

# ------------------------------------------------------------------ inputs
lm = pd.read_parquet(I / "lang_metrics.parquet")
lx = pd.read_parquet(I / "lang_exposure.parquet").set_index("ISO")
lm["Classification"] = lm.ISO.map(lx.Classification)
lm["Family"] = lm.ISO.map(lx.Family)
lm["noninst"] = lm.vitality.isin(NONINST)
lm["point_language"] = lm.ISO.map(lx.point_language).fillna(False).astype(bool) if "point_language" in lx else False
cw = pd.read_parquet(I / "glottolog_crosswalk.parquet").set_index("ISO")
lm["g_lineage"] = lm.ISO.map(cw.g_lineage).where(lm.ISO.map(cw.g_category) == "Spoken_L1_Language")
lm["g_lineage_name"] = lm.ISO.map(cw.g_lineage_name)
lm["g_classification"] = lm.ISO.map(cw.g_classification).fillna("")
lm["g_is_isolate"] = lm.ISO.map(cw.g_is_isolate).astype(bool)
record("heritage.glottolog.n_languages_in_lineage_sample", int(lm.g_lineage.notna().sum()))
record("heritage.glottolog.n_excluded_by_category", lm.ISO.map(cw.g_category)[lm.g_lineage.isna()].value_counts().to_dict())
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.ISO.isin(lm.ISO) & pe.land_w_km2.notna() & ~pe.sign_language]
w = pd.read_parquet(I / "poly_cell_weights.parquet")
w = w[w.pid.isin(pe.pid)].copy()
w["ISO"] = w.pid.map(pe.set_index("pid").ISO)
lw = w.groupby(["ISO", "grid_id"], as_index=False).w_km2.sum()

grid = pd.read_parquet(I / "grid_fields.parquet")
grid = grid[grid.lat > -60].reset_index(drop=True)                 # exclude Antarctica
cidx = pd.Series(np.arange(len(grid)), index=grid.grid_id)
lw = lw[lw.grid_id.isin(cidx.index)]
lang_ids = lm.ISO.to_numpy()
lidx = pd.Series(np.arange(len(lang_ids)), index=lang_ids)
rows = lidx[lw.ISO].to_numpy()
cols = cidx[lw.grid_id].to_numpy()
range_km2 = np.bincount(rows, weights=lw.w_km2.to_numpy(), minlength=len(lang_ids))
Rmat = sp.csr_matrix((lw.w_km2.to_numpy() / range_km2[rows], (rows, cols)), shape=(len(lang_ids), len(grid)))
covered = np.asarray(Rmat.sum(1)).ravel() > 0.999
record("heritage.n_languages_with_cells", int(covered.sum()))

# ------------------------------------------------------------------ cell RP per member (hybrid)
cm = np.load(I / "country_member_fields.npz", allow_pickle=True)
iso3 = list(cm["iso3"])
RPc = cm["RP_2100"]                                                   # (60, 187)
dev = np.nan_to_num(RPc - np.nanmean(RPc, axis=0, keepdims=True))     # (60, 187)
ci = grid.country_iso3.map({c: i for i, c in enumerate(iso3)})
has = ci.notna().to_numpy()
RPbar = grid.RP_2100.to_numpy()
RPm = np.repeat(RPbar[:, None], 60, axis=1)
RPm[has] += dev[:, ci[has].astype(int).to_numpy()].T
DRYm = (RPm < -DB).astype(float)                                      # (C, 60)
dry0 = (RPbar < -DB).astype(float)
area = grid.area_km2.to_numpy()
pop_mask = grid.economic_support.fillna(0).to_numpy() > 0 if "economic_support" in grid else np.ones(len(grid), bool)

# ------------------------------------------------------------------ 1. endemism-weighted exposure
f0 = Rmat @ dry0                                                       # (L,)
fm = Rmat @ DRYm                                                       # (L, 60)
non = lm.noninst.to_numpy() & covered
ins = ~lm.noninst.to_numpy() & covered
land0 = np.average(dry0, weights=area)
landm = (area[:, None] * DRYm).sum(0) / area.sum()
popl0 = np.average(dry0[pop_mask], weights=area[pop_mask])
poplm = (area[pop_mask, None] * DRYm[pop_mask]).sum(0) / area[pop_mask].sum()
record("heritage.land_share_drier", float(land0))
record("heritage.land_share_drier.member_p17_p50_p83", np.percentile(landm, [17, 50, 83]))
record("heritage.populated_land_share_drier", float(popl0))
for lab, m in [("noninst", non), ("inst", ins), ("all", covered)]:
    e0 = f0[m].mean()
    em = fm[m].mean(0)
    record(f"heritage.{lab}.endemism_share_drier", float(e0))
    record(f"heritage.{lab}.endemism_share_drier.member_p17_p50_p83", np.percentile(em, [17, 50, 83]))
    record(f"heritage.{lab}.endemism_ratio_vs_land", float(e0 / land0))
    record(f"heritage.{lab}.endemism_ratio_vs_land.member_p17_p50_p83", np.percentile(em / landm, [17, 50, 83]))
    record(f"heritage.{lab}.endemism_ratio_vs_land.n_members_gt1", int((em / landm > 1).sum()))
    record(f"heritage.{lab}.endemism_ratio_vs_populated_land.member_p17_p50_p83",
           np.percentile(em / poplm, [17, 50, 83]))
    record(f"heritage.{lab}.endemism_ratio_vs_populated_land.n_members_gt1", int((em / poplm > 1).sum()))
# binary language share vs land, per member (for the paper's baseline comparison)
z = np.load(I / "lang_member_R.npz", allow_pickle=True)
assert (z["ISO"] == lang_ids).all()
bin_m = (z["RP_2100"][lm.noninst.to_numpy()] < -DB).mean(0)
record("heritage.noninst.binary_ratio_vs_land.member_p17_p50_p83", np.percentile(bin_m / landm, [17, 50, 83]))
record("heritage.noninst.binary_ratio_vs_land.n_members_gt1", int((bin_m / landm > 1).sum()))
record("heritage.noninst.binary_ratio_vs_land", float(lm[lm.noninst].RP_2100.lt(-DB).mean() / land0))
# GDP and people baselines
cells = pd.read_parquet(I / "pixel_cells.parquet")[["grid_id", "gdp"]]
g2 = grid.merge(cells, on="grid_id", how="inner")
record("heritage.gdp_share_drier", float(np.average(g2.RP_2100 < -DB, weights=g2.gdp)))

# ------------------------------------------------------------------ 2. no refuge, range size
lm["range_km2"] = range_km2
lm["f_dry"] = f0
lm["f_dry_member_median"] = np.median(fm, 1)
lm["no_refuge"] = f0 >= 0.999
lm["no_refuge_frac_members"] = (fm >= 0.999).mean(1)
d = lm[non & ~lm.point_language.to_numpy()].copy()          # point languages have no homeland area
record("heritage.noninst.n_no_refuge", int(d.no_refuge.sum()))
record("heritage.noninst.share_no_refuge", float(d.no_refuge.mean()))
record("heritage.noninst.n_no_refuge_ge_two_thirds_members", int((d.no_refuge_frac_members >= 2 / 3).sum()))
record("heritage.noninst.share_no_refuge.member_p17_p50_p83",
       np.percentile((fm[non] >= 0.999).mean(0), [17, 50, 83]))
record("heritage.noninst.n_fully_wetter_or_similar", int((d.f_dry <= 0.001).sum()))
record("heritage.noninst.range_km2_median", float(d.range_km2.median()))
d["range_q"] = pd.qcut(d.range_km2, 5, labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"])
rq = d.groupby("range_q", observed=True).agg(
    n=("ISO", "size"), range_km2_median=("range_km2", "median"), mean_f_dry=("f_dry", "mean"),
    share_majority_drier=("f_dry", lambda x: (x > 0.5).mean()), share_no_refuge=("no_refuge", "mean"),
    share_fully_not_drier=("f_dry", lambda x: (x <= 0.001).mean()), L1_median=("L1_Users", "median"))
# member bands for the no-refuge share by quintile
qlab = d.range_q.to_numpy()
fmn = fm[non & ~lm.point_language.to_numpy()]
for qn in rq.index:
    sel = qlab == qn
    rq.loc[qn, "no_refuge_p17"], rq.loc[qn, "no_refuge_p83"] = np.percentile((fmn[sel] >= 0.999).mean(0), [17, 83])
    rq.loc[qn, "mean_f_p17"], rq.loc[qn, "mean_f_p83"] = np.percentile(fmn[sel].mean(0), [17, 83])
rq.to_csv(T / "heritage_range_quintiles.csv")
for qn, r in rq.iterrows():
    k = str(qn).split()[0]
    record(f"heritage.range.{k}.share_no_refuge", float(r.share_no_refuge))
    record(f"heritage.range.{k}.mean_f_dry", float(r.mean_f_dry))
    record(f"heritage.range.{k}.range_km2_median", float(r.range_km2_median))

# ------------------------------------------------------------------ 3. the family tree
cov = lm[covered & lm.g_lineage.notna().to_numpy()].copy()              # Glottolog lineage sample
paths = {r.ISO: hm.glottolog_path(r.ISO, r.g_lineage, r.g_classification) for r in cov.itertuples()}
desc = hm.descendant_counts(paths)
ED = hm.fair_proportion(paths)
cov["ED"] = cov.ISO.map(ED)
cov["lineage"] = [paths[i][0] for i in cov.ISO]
lm["ED"] = lm.ISO.map(ED)
lm["lineage"] = lm.ISO.map({i: p[0] for i, p in paths.items()})
lm["lineage_name"] = lm.lineage.map(dict(zip(cov.lineage, cov.g_lineage_name)))
record("heritage.tree.n_edges_total", len(desc))
record("heritage.tree.n_top_lineages", int(cov.lineage.nunique()))
record("heritage.tree.ED_median", float(cov.ED.median()))
record("heritage.tree.ED_max", float(cov.ED.max()))


def pd_of(isos):
    return hm.pd_count(paths, isos)


PD_all = pd_of(cov.ISO)
for lab, m in [("noninst", cov.noninst), ("all", cov.noninst | ~cov.noninst)]:
    sub = cov[m]
    fsub = lm.set_index('ISO').f_dry[sub.ISO].to_numpy()
    record(f"heritage.tree.{lab}.ED_weighted_share_drier", float(np.average(fsub, weights=sub.ED)))
    record(f"heritage.tree.{lab}.unweighted_share_drier", float(fsub.mean()))
cov["f_dry"] = cov.ISO.map(lm.set_index("ISO").f_dry)
mostly = cov[cov.f_dry > 0.5].ISO.to_numpy()
pd_share = pd_of(mostly) / PD_all
lang_share = len(mostly) / len(cov)
null = np.array([pd_of(rng.choice(cov.ISO.to_numpy(), len(mostly), replace=False)) / PD_all for _ in range(500)])
record("heritage.tree.lang_share_mostly_drier", float(lang_share))
record("heritage.tree.PD_share_mostly_drier", float(pd_share))
record("heritage.tree.PD_share_null_p2.5_p50_p97.5", np.percentile(null, [2.5, 50, 97.5]))
record("heritage.tree.PD_share_percentile_vs_null", float((null < pd_share).mean()))
# per member PD share and ED-weighted share
cov_idx = lidx[cov.ISO].to_numpy()
pd_m, ed_m = [], []
for mbr in range(60):
    fmm = fm[cov_idx, mbr]
    pd_m.append(pd_of(cov.ISO.to_numpy()[fmm > 0.5]) / PD_all)
    ed_m.append(np.average(fmm, weights=cov.ED.to_numpy()))
pd_m, ed_m = np.array(pd_m), np.array(ed_m)
record("heritage.tree.PD_share_mostly_drier.member_p17_p50_p83", np.percentile(pd_m, [17, 50, 83]))
record("heritage.tree.ED_weighted_share_drier.member_p17_p50_p83", np.percentile(ed_m, [17, 50, 83]))
record("heritage.tree.ED_ratio_vs_land.member_p17_p50_p83", np.percentile(ed_m / landm, [17, 50, 83]))
record("heritage.tree.ED_ratio_vs_land.n_members_gt1", int((ed_m / landm > 1).sum()))
# lineages (top-level families, isolates each their own)
lin = cov.groupby("lineage").agg(n=("ISO", "size"), mean_f_dry=("f_dry", "mean"),
                                 all_mostly_drier=("f_dry", lambda x: bool((x > 0.5).all())),
                                 none_mostly_drier=("f_dry", lambda x: bool((x <= 0.5).all())))
lin["is_isolate"] = cov.groupby("lineage").g_is_isolate.first().reindex(lin.index).to_numpy()
lin["name"] = cov.groupby("lineage").g_lineage_name.first().reindex(lin.index).to_numpy()
record("heritage.lineage.n_families", int((~lin.is_isolate).sum()))
record("heritage.lineage.n_families_single_language", int(((~lin.is_isolate) & (lin.n == 1)).sum()))
record("heritage.lineage.n", len(lin))
record("heritage.lineage.share_majority_homeland_drier", float((lin.mean_f_dry > 0.5).mean()))
record("heritage.lineage.n_all_languages_mostly_drier", int(lin.all_mostly_drier.sum()))
record("heritage.lineage.n_all_languages_mostly_drier_multi", int((lin.all_mostly_drier & (lin.n > 1)).sum()))
record("heritage.lineage.isolates_n", int(lin.is_isolate.sum()))
record("heritage.lineage.isolates_share_mostly_drier", float((lin[lin.is_isolate].mean_f_dry > 0.5).mean()))
record("heritage.lineage.lineage_weighted_mean_f_dry", float(lin.mean_f_dry.mean()))
# member robustness of lineage results
lin_codes = pd.Categorical(cov.lineage)
L_onehot = sp.csr_matrix((np.ones(len(cov)), (lin_codes.codes, np.arange(len(cov)))),
                         shape=(len(lin_codes.categories), len(cov)))
nper = np.asarray(L_onehot.sum(1)).ravel()
lin_f_m = (L_onehot @ fm[cov_idx]) / nper[:, None]                      # (lineages, 60) mean f per member
lin_all_m = (L_onehot @ (fm[cov_idx] > 0.5).astype(float)) == nper[:, None]
lm_df = pd.DataFrame(lin_f_m, index=lin_codes.categories)
lin["mean_f_p17"] = lm_df.quantile(0.17, axis=1).reindex(lin.index)
lin["mean_f_p83"] = lm_df.quantile(0.83, axis=1).reindex(lin.index)
lin["frac_members_majority_drier"] = (lm_df > 0.5).mean(1).reindex(lin.index)
n_all_m = lin_all_m.sum(0)
record("heritage.lineage.n_all_languages_mostly_drier.member_p17_p50_p83", np.percentile(n_all_m, [17, 50, 83]))
record("heritage.lineage.share_majority_homeland_drier.member_p17_p50_p83",
       np.percentile((lin_f_m > 0.5).mean(0), [17, 50, 83]))
# Glottolog families quoted in the text, and the 25 largest (aggregates over >= 2 languages only)
fams = lin[~lin.is_isolate & (lin.n >= 2)]
named = set(fams.sort_values("n", ascending=False).head(25).name) | {
    "Otomanguean", "Nuclear Trans New Guinea", "Austronesian", "Atlantic-Congo", "Mayan", "Uto-Aztecan", "Sepik",
    "Nuclear Torricelli", "Indo-European", "Afro-Asiatic", "Sino-Tibetan", "Dravidian", "Arawakan"}
for gc, r in fams[fams.name.isin(named)].iterrows():
    if True:
        key = r["name"].replace(" ", "_").replace("-", "_")
        record(f"heritage.family.{key}.n", int(r.n))
        record(f"heritage.family.{key}.mean_f_dry", float(r.mean_f_dry))
        record(f"heritage.family.{key}.mean_f_p17_p83", [float(r.mean_f_p17), float(r.mean_f_p83)])
        record(f"heritage.family.{key}.frac_members_majority_drier", float(r.frac_members_majority_drier))
fam_tab = lin[~lin.is_isolate].sort_values("n", ascending=False)
fam_tab.reset_index().to_parquet(I / "heritage_families_full.parquet")      # RESTRICTED: includes single-language families
# public table: families with at least two mapped living languages (no language-level values; Ethnologue licence)
fam_tab[fam_tab.n >= 2].to_csv(T / "heritage_families.csv")
iso_lin = lin[lin.is_isolate]
record("heritage.lineage.isolates_by_macroarea", cov[cov.g_is_isolate].ISO.map(cw.g_macroarea).value_counts().to_dict())

# ------------------------------------------------------------------ cell map products
LWE = np.asarray(Rmat[covered].sum(0)).ravel()
LWE_non = np.asarray(Rmat[non].sum(0)).ravel()
cells_out = pd.DataFrame({"grid_id": grid.grid_id, "lat": grid.lat, "lon": grid.lon, "area_km2": area,
                          "RP_2100": RPbar, "dry": dry0, "frac_members_dry": DRYm.mean(1),
                          "LWE": LWE, "LWE_noninst": LWE_non})

# share of total endemism in the top-1% / top-10% LWE-density cells, and their dryness
hot = hm.hotspot_cells(LWE_non, area, 0.5)              # ranked by endemism density
k10 = len(hot)
o = hot
record("heritage.cells.n_cells_holding_half_of_noninst_endemism", int(k10))
record("heritage.cells.land_share_holding_half_of_noninst_endemism", float(area[o[:k10]].sum() / area.sum()))
record("heritage.cells.share_dry_of_cells_holding_half", float(np.average(dry0[o[:k10]], weights=LWE_non[o[:k10]])))
hot_m = (LWE_non[hot, None] * DRYm[hot]).sum(0) / LWE_non[hot].sum()
record("heritage.cells.share_dry_of_cells_holding_half.member_p17_p50_p83", np.percentile(hot_m, [17, 50, 83]))
record("heritage.cells.hotspot_ratio_vs_land.member_p17_p50_p83", np.percentile(hot_m / landm, [17, 50, 83]))
record("heritage.cells.hotspot_ratio_vs_land.n_members_gt1", int((hot_m / landm > 1).sum()))
cells_hot = np.zeros(len(grid), bool)
cells_hot[hot] = True
cells_out["hotspot_half"] = cells_hot
cells_out.to_parquet(I / "heritage_cells.parquet", index=False)

lm[["ISO", "Country_Code", "Family", "lineage", "lineage_name", "vitality", "noninst", "point_language", "L1_Users",
    "range_km2", "f_dry", "f_dry_member_median", "no_refuge", "no_refuge_frac_members", "ED"]].to_parquet(I / "heritage_lang.parquet",
                                                                                    index=False)
print(rq.round(3).to_string())
print(fam_tab.head(15).round(3).to_string())
