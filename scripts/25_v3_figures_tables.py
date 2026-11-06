"""25 - Figures and tables of the v3 paper ("Less rain where languages live").

Main figures
  Fig1_v3_map      A: cells holding half of community-language heritage, coloured by their rainfall
                   outcome, on a faint map of the rainfall difference. B: zonal rainfall difference.
                   C: latitude distribution of land, people and community languages.
  Fig2             = Fig_against_what (script 24)
  Fig3             = Fig_tree_rain_range (script 24)
  Fig4_v3_robust   A: share with less rain, member by member, against land (1:1 line).
                   B: ratio of homeland losing rain to homeland gaining rain, by unit and member.
                   C: the 15 largest language families (Glottolog 5.3) with member ranges.
SI figure
  FigS_range_refuge  homeland size and the share of languages with no part of the homeland spared.
Tables (LaTeX rows in results/tables/latex/)
  tab1_v3, tabS_v3_families, tabS_v3_drytowet, tabS_v3_range, tabS_v3_warm_lineages
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P, record  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
I, T, F, LX = P("interim"), P("results"), P("figures"), P("results") / "latex"
NB = ps.numbers(T / "numbers.json")
DRY, WET, SIM = "#8c510a", "#01665e", "#9a9891"
C_LANG, C_FAM, C_BASE = ps.VIT_COLORS["Stable"], ps.VIT_COLORS["Dying"], "#898781"
NONINST = ["Stable", "In trouble", "Dying"]

tab = pd.read_csv(T / "baselines_rain_heat.csv")
bym = pd.read_csv(T / "baselines_by_member.csv")
od = pd.read_csv(T / "baselines_dry_to_wet.csv")
fam = pd.read_parquet(I / "heritage_families_full.parquet")
famT = pd.read_parquet(I / "heritage_lineages_temperature_full.parquet")
rq = pd.read_csv(T / "heritage_range_quintiles.csv")
LABEL = {"land": "Land (area)", "populated land": "Populated land", "people (L1 speakers)": "People",
         "economy (GDP)": "Economy (GDP)", "institutional languages": "Institutional languages",
         "non-institutional languages": "Non-institutional languages", "linguistic diversity hotspots": "Linguistic diversity hotspots",
         "language lineages": "Language lineages"}
ORDER = list(LABEL)
DISPLAY = {"Otomanguean": "Oto-Manguean"}
fam["name"] = fam["name"].map(lambda x: DISPLAY.get(x, x))
famT["name"] = famT["name"].map(lambda x: DISPLAY.get(x, x))

# =================================================================== Fig 1
hc = pd.read_parquet(I / "heritage_cells.parquet")
gf = pd.read_parquet(I / "grid_fields.parquet")
hc = hc.merge(gf[["grid_id", "PH_2100"]], on="grid_id", how="left")
fig = plt.figure(figsize=(ps.W2, 3.15))
ax = fig.add_axes([0.0, 0.0, 0.74, 1.0])
ps.draw_basemap(ax, P("ne_land"))
X, Y = ps.grid_mesh()
light = LinearSegmentedColormap.from_list("lightBrBG", ["#e2c590", "#efdcb4", "#f7efde", "#f6f5f2", "#ddf0ec",
                                                        "#b9e0d9", "#94cdc4"])
Z = ps.to_grid(hc.assign(v=hc.RP_2100.where(hc.PH_2100.abs() <= 50)), "v")
ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=light, norm=TwoSlopeNorm(0, -30, 30), shading="flat", zorder=1,
              rasterized=True)
cls = np.where(hc.RP_2100 < -2, 2, np.where(hc.RP_2100 > 2, 0, 1)).astype(float)
cls[~hc.hotspot_half.to_numpy()] = np.nan
ax.pcolormesh(X, Y, np.ma.masked_invalid(ps.to_grid(hc.assign(c=cls), "c")), cmap=ListedColormap([WET, SIM, DRY]),
              vmin=-0.5, vmax=2.5, shading="flat", zorder=2, rasterized=True)
gpd.read_file(P("ne_coast")).to_crs(ps.ROBIN).plot(ax=ax, color="#6f6e69", lw=0.18, zorder=3)
share_land = 100 * NB["heritage.cells.land_share_holding_half_of_noninst_endemism"]
ax.legend(handles=[Patch(color=DRY, label="less rain under SAI (>2 pp)"), Patch(color=SIM, label="similar (±2 pp)"),
                   Patch(color=WET, label="more rain under SAI (>2 pp)")],
          title=f"Cells holding half of all non-institutional-\nlanguage diversity ({share_land:.1f}% of land):",
          title_fontsize=5.7, fontsize=5.5, loc="lower left", bbox_to_anchor=(0.0, 0.07), alignment="left",
          handlelength=1.0)
ax.text(0.0, 0.99, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax.text(0.02, 0.025, "Background: rainfall, SAI minus mitigation, 2100 (faint; hyper-arid blank)",
        transform=ax.transAxes, fontsize=5.2, color=ps.INK2)
axZ = fig.add_axes([0.775, 0.12, 0.085, 0.80])
axD = fig.add_axes([0.875, 0.12, 0.12, 0.80], sharey=axZ)
g = gf[(gf.PH_2100.abs() <= 50) & (gf.lat > -60)].copy()
g["band"] = pd.cut(g.lat, np.arange(-56, 84, 4))
zm = g.groupby("band", observed=True).apply(lambda d: np.average(d.RP_2100, weights=d.area_km2))
mid = np.array([b.mid for b in zm.index])
axZ.axvline(0, color=ps.AXIS, lw=0.6)
axZ.fill_betweenx(mid, 0, zm.values, where=zm.values < 0, color="#dfc27d", lw=0, interpolate=True)
axZ.fill_betweenx(mid, 0, zm.values, where=zm.values > 0, color="#80cdc1", lw=0, interpolate=True)
axZ.plot(zm.values, mid, color=ps.INK, lw=0.9)
axZ.set_xlabel("Rainfall\ndifference (pp)", fontsize=6.0)
axZ.set_ylabel("Latitude (°)", labelpad=1)
axZ.set_ylim(-56, 82)
yy = np.linspace(-56, 82, 300)
lm = pd.read_parquet(I / "lang_metrics.parquet")
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.land_w_km2.notna() & ~pe.sign_language & pe.ISO.isin(lm.ISO)]
x_l = lm[lm.vitality.isin(NONINST)].lat_mean.dropna().to_numpy()
k_l = gaussian_kde(x_l, bw_method=3.0 / x_l.std())
pp = pe[pe.L1_LIC.fillna(0) > 0]
x_p, w_p = pp.rep_lat.to_numpy(), pp.L1_LIC.to_numpy()
k_p = gaussian_kde(x_p, weights=w_p, bw_method=3.0 / np.sqrt(np.cov(x_p, aweights=w_p)))
lat_land = gf[gf.lat > -60]
dens_land = np.histogram(lat_land.lat, bins=np.arange(-60, 84.5, 1), weights=lat_land.area_km2, density=True)
axD.plot(dens_land[0] * 100, dens_land[1][:-1] + 0.5, color=C_BASE, lw=0.9, label="Land")
axD.plot(k_p(yy) * 100, yy, color=ps.INK2, lw=1.0, ls=(0, (3, 1.5)), label="People")
axD.plot(k_l(yy) * 100, yy, color=C_FAM, lw=1.3, label="Non-institutional\nlanguages")
axD.set_xlabel("Distribution\n(% per °)", fontsize=6.0)
axD.tick_params(labelleft=False)
axD.set_xlim(0, None)
axD.legend(loc="lower right", fontsize=5.2, handlelength=1.4, bbox_to_anchor=(1.07, 0.0))
for a_ in (axZ, axD):
    a_.grid(axis="y", color=ps.GRID, lw=0.4)
    for y0 in (-23.44, 23.44):
        a_.axhline(y0, color=ps.MUTED, lw=0.4, ls=(0, (2, 2)))
axZ.text(-0.62, 1.07, "B", transform=axZ.transAxes, fontsize=9, fontweight="bold", va="top")
axD.text(-0.02, 1.07, "C", transform=axD.transAxes, fontsize=9, fontweight="bold", va="top", ha="right")
ps.save(fig, F / "Fig1_v3_map")
# share of each distribution between 15 S and 15 N (deep tropics)
record("v3.deep_tropics_share.languages", float((np.abs(x_l) < 15).mean()))
record("v3.deep_tropics_share.people", float(w_p[np.abs(x_p) < 15].sum() / w_p.sum()))
lt = lat_land
record("v3.deep_tropics_share.land", float(lt.area_km2[(lt.lat > -15) & (lt.lat < 15)].sum() / lt.area_km2.sum()))
zz = g[(g.lat > -15) & (g.lat < 15)]
record("v3.deep_tropics_land_mean_RP", float(np.average(zz.RP_2100, weights=zz.area_km2)))
zo = g[(g.lat.abs() >= 15)]
record("v3.outside_deep_tropics_land_mean_RP", float(np.average(zo.RP_2100, weights=zo.area_km2)))

# =================================================================== Fig 4
fig = plt.figure(figsize=(ps.W2, 2.7))
axA = fig.add_axes([0.055, 0.17, 0.235, 0.73])
dr = bym[bym.outcome == "drier"].pivot(index="member", columns="unit", values="share") * 100
for u, col, mk, lab in [("people (L1 speakers)", C_BASE, "o", "People"),
                        ("non-institutional languages", C_LANG, "o", "Non-inst. lang."),
                        ("language lineages", C_FAM, "o", "Lineages")]:
    n_above = int((dr[u] > dr["land"]).sum())
    axA.scatter(dr["land"], dr[u], s=9, color=col, lw=0.3, ec="white", zorder=3, label=f"{lab} {n_above}/60")
lo, hi = 30, 85
axA.plot([lo, hi], [lo, hi], color=ps.INK2, lw=0.7, ls=(0, (2, 2)))
axA.text(81, 78.5, "1:1", fontsize=5.4, color=ps.INK2, rotation=40)
axA.set_xlim(lo, 100)
axA.set_ylim(lo, hi)
axA.set_xlabel("Land with less rain (%)")
axA.set_ylabel("With less rain (%)")
axA.legend(loc="lower right", fontsize=4.9, handletextpad=0.1, borderaxespad=0.2, title="Members above 1:1",
           title_fontsize=4.9, framealpha=0.95, edgecolor="none", alignment="left")
axA.grid(color=ps.GRID, lw=0.4)
ps.panel_label(axA, "A", x=-0.24)
axB = fig.add_axes([0.435, 0.17, 0.19, 0.73])
units = ["land", "people (L1 speakers)", "economy (GDP)", "non-institutional languages", "language lineages"]
d_ = bym[bym.outcome == "drier"].pivot(index="member", columns="unit", values="share")
w_ = bym[bym.outcome == "wetter"].pivot(index="member", columns="unit", values="share")
rng = np.random.default_rng(5)
for yi, u in enumerate(units):
    r = (d_[u] / w_[u]).to_numpy()
    col = C_FAM if u == "language lineages" else (C_LANG if u.startswith("non-institutional") else C_BASE)
    axB.scatter(r, yi + rng.uniform(-0.18, 0.18, len(r)), s=5, color=col, alpha=0.6, lw=0)
    r0 = od.set_index("unit").loc[u, "dry_to_wet_ratio"]
    axB.plot([r0, r0], [yi - 0.3, yi + 0.3], color=ps.INK, lw=1.4)
    axB.text(r0 + 0.15, yi - 0.30, f"{r0:.1f}", va="bottom", ha="left", fontsize=5.8,
             fontweight="bold" if col != C_BASE else "normal")
axB.set_yticks(range(len(units)))
axB.set_yticklabels([LABEL[u].replace(" languages", "\nlanguages").replace("Language lineages", "Language\nlineages") for u in units], fontsize=5.8)
axB.invert_yaxis()
axB.set_xlim(0, 10)
axB.axvline(1, color=ps.AXIS, lw=0.6)
axB.set_xlabel("Homeland losing rain ÷\nhomeland gaining rain")
axB.grid(axis="x", color=ps.GRID, lw=0.4)
ps.panel_label(axB, "B", x=-0.5)
axC = fig.add_axes([0.77, 0.17, 0.22, 0.73])
big = fam.sort_values("n", ascending=False).head(15).sort_values("mean_f_dry")
for yi, r in enumerate(big.itertuples()):
    col = DRY if r.frac_members_majority_drier >= 0.9 else SIM
    axC.plot([r.mean_f_p17 * 100, r.mean_f_p83 * 100], [yi, yi], color=col, lw=1.4, alpha=0.6,
             solid_capstyle="round")
    axC.plot(r.mean_f_dry * 100, yi, "o", color=col, ms=3.4, mec="white", mew=0.4, zorder=4)
axC.axvline(100 * NB["heritage.land_share_drier"], color=C_BASE, lw=0.7, ls=(0, (2, 2)))
axC.set_yticks(range(len(big)))
axC.set_yticklabels([f"{r.name} ({r.n:,})" for r in big.itertuples()], fontsize=5.3)
axC.set_xlim(-2, 102)
axC.set_xlabel("Homeland with less rain (%)")
axC.grid(axis="x", color=ps.GRID, lw=0.4)
axC.legend(handles=[Line2D([], [], color=DRY, marker="o", lw=1.4, ms=3.2, label="≥90% of members"),
                    Line2D([], [], color=SIM, marker="o", lw=1.4, ms=3.2, label="mixed")],
           title="Mostly drier in:", title_fontsize=5.0, loc="lower right", fontsize=4.9, handlelength=1.3,
           bbox_to_anchor=(1.04, -0.03), framealpha=1, edgecolor="none")
ps.panel_label(axC, "C", x=-0.78)
ps.save(fig, F / "Fig4_v3_robust")

# =================================================================== SI: range size and refuge
fig, axB = plt.subplots(figsize=(ps.W1 * 1.25, 2.4))
x = np.arange(len(rq))
axB.bar(x, rq.share_no_refuge * 100, width=0.62, color=C_LANG, lw=0, label="Whole homeland with less rain")
axB.vlines(x, rq.no_refuge_p17 * 100, rq.no_refuge_p83 * 100, color=ps.INK, lw=0.9)
axB.plot(x, rq.mean_f_dry * 100, "o", color="white", mec=ps.INK, mew=0.9, ms=4.2, zorder=5,
         label="Mean share of homeland with less rain")
axB.vlines(x + 0.18, rq.mean_f_p17 * 100, rq.mean_f_p83 * 100, color=ps.MUTED, lw=0.9)
axB.axhline(100 * NB["heritage.land_share_drier"], color=C_BASE, lw=0.7, ls=(0, (2, 2)))
axB.set_xticks(x)
axB.set_xticklabels([f"{['Smallest', '2nd', '3rd', '4th', 'Largest'][i]}\n{v:,.0f} km²" for i, v in
                     enumerate(rq.range_km2_median)], fontsize=5.6)
axB.set_xlabel("Homeland size (fifths; median area)")
axB.set_ylabel("Non-institutional languages (%)")
axB.set_ylim(0, 95)
axB.grid(axis="y", color=ps.GRID, lw=0.4)
h_, l_ = axB.get_legend_handles_labels()
h_.append(Line2D([], [], color=C_BASE, lw=0.7, ls=(0, (2, 2))))
l_.append(f"Land with less rain ({100 * NB['heritage.land_share_drier']:.0f}%)")
axB.legend(h_, l_, loc="upper right", fontsize=5.2, handlelength=1.6)
ps.save(fig, F / "FigS_range_refuge")

# =================================================================== tables
pc = lambda v: f"{100 * v:.0f}"  # noqa: E731
rows = []
for u in ORDER:
    d = tab[(tab.outcome == "drier") & (tab.unit == u)].iloc[0]
    w = tab[(tab.outcome == "wetter") & (tab.unit == u)].iloc[0]
    h = tab[(tab.outcome == "warmer") & (tab.unit == u)].iloc[0]
    o = od[od.unit == u].iloc[0]
    land_row = u == "land"
    rows.append(f"{LABEL[u]} & {pc(d.ensemble_mean_climate)} & {pc(d.member_p17)}--{pc(d.member_p83)} & "
                f"{'--' if land_row else f'{d.ratio_vs_land:.2f} ({int(d.n_members_ratio_gt1)})'} & "
                f"{pc(w.ensemble_mean_climate)} & {o.dry_to_wet_ratio:.1f} & "
                f"{'--' if land_row else str(int(o.n_members_above_land))} & {pc(h.ensemble_mean_climate)} \\\\")
    if u == "economy (GDP)":
        rows.append("\\midrule")
(LX / "tab1_v3.tex").write_text("\n".join(rows) + "\n")

fb = fam.sort_values("n", ascending=False).head(20)
fT = famT.set_index("lineage")
rows = [f"{r.name} & {r.n:,} & {pc(r.mean_f_dry)} & {pc(r.mean_f_p17)}--{pc(r.mean_f_p83)} & "
        f"{pc(r.frac_members_majority_drier)} & {pc(fT.loc[r.lineage, 'mean_f_warmer'])} \\\\" for r in fb.itertuples()]
iso_ = fam[fam.lineage.str.startswith("[")] if "lineage" in fam else None
(LX / "tabS_v3_families.tex").write_text("\n".join(rows) + "\n")

rows = [f"{LABEL[r.unit]} & {r.dry_to_wet_ratio:.2f} & {r.member_p17:.2f}--{r.member_p83:.2f} & "
        f"{'--' if r.unit == 'land' else int(r.n_members_above_land)} \\\\" for r in od.set_index('unit').loc[ORDER].reset_index().itertuples()]
(LX / "tabS_v3_drytowet.tex").write_text("\n".join(rows) + "\n")

rows = [f"{r.range_q} & {int(r.n):,} & {r.range_km2_median:,.0f} & {r.L1_median:,.0f} & {pc(r.mean_f_dry)} & "
        f"{pc(r.share_no_refuge)} ({pc(r.no_refuge_p17)}--{pc(r.no_refuge_p83)}) & {pc(r.share_fully_not_drier)} \\\\"
        for r in rq.itertuples()]
(LX / "tabS_v3_range.tex").write_text("\n".join(rows) + "\n")

ww = famT[famT.mean_f_warmer > 0.5].sort_values("n", ascending=False)
# single-language lineages are pooled so that no language-level value is published (Ethnologue licence)
w_multi, w_single = ww[ww.n >= 2], ww[ww.n < 2]
rows = [f"{r.name} & {int(r.n)} & {pc(r.mean_f_warmer)} & {pc(r.frac_members_majority_warmer)} \\\\"
        for r in w_multi.itertuples()]
if len(w_single):
    rows.append(f"Single-language lineages (pooled, {len(w_single)}) & {len(w_single)} & "
                f"{pc(w_single.mean_f_warmer.mean())} & {pc(w_single.frac_members_majority_warmer.mean())} \\\\")
record("v3.warm_lineages.n_multi", len(w_multi))
record("v3.warm_lineages.n_single", len(w_single))
(LX / "tabS_v3_warm_lineages.tex").write_text("\n".join(rows) + "\n")
print("v3 figures and tables written")

# =================================================================== extra numbers quoted in the v3 text
lmn = lm[lm.vitality.isin(NONINST)]
record("v3.temp.noninst.R_median", float(lmn.R_2100.median()))
record("v3.temp.noninst.R_p25_p75", np.percentile(lmn.R_2100.dropna(), [25, 75]))
# heritage_families_full.parquet holds all Glottolog families incl. single-language ones (isolates are separate lineages)
record("v3.family.n", len(fam))
record("v3.family.n_isolates_in_table", int(fam.is_isolate.sum()))
record("v3.family.n_mostly_drier", int((fam.mean_f_dry > 0.5).sum()))
record("v3.family.n_robust_drier_ge90", int((fam.frac_members_majority_drier >= 0.9).sum()))
record("v3.family.n_robust_not_drier_le10", int((fam.frac_members_majority_drier <= 0.1).sum()))
record("v3.family.n_coinflip_40_60", int(fam.frac_members_majority_drier.between(0.4, 0.6).sum()))
record("v3.family.n_languages_in_robust_drier", int(fam[fam.frac_members_majority_drier >= 0.9].n.sum()))
record("v3.family.n_languages_total", int(fam.n.sum()))
print("extra numbers recorded")
# New Guinea families whose whole homeland is drier, member by member
zz = np.load(I / "heritage_member_fdry.npz", allow_pickle=True)
hl_ = pd.read_parquet(I / "heritage_lang.parquet").set_index("ISO")
lin_ = pd.Series(zz["ISO"]).map(hl_.lineage).to_numpy()
for fam_name in ["Sepik", "Nuclear Torricelli", "Nuclear Trans New Guinea", "Austronesian", "Oto-Manguean"]:
    gc_ = fam.loc[fam.name == fam_name, "lineage"].iloc[0]
    fam_name = fam_name.replace(" ", "_").replace("-", "_")
    fmean = zz["f"][lin_ == gc_].mean(0)
    record(f"v3.family.{fam_name}.share_members_whole_homeland_drier_ge99", float((fmean >= 0.99).mean()))
# where the heritage hotspots are
hh = hc[hc.hotspot_half].merge(gf[["grid_id", "country_iso3"]], on="grid_id", how="left")
vc = hh.country_iso3.value_counts()
record("v3.hotspots.n_cells", int(len(hh)))
record("v3.hotspots.top_countries", {k: int(v) for k, v in vc.head(10).items()})
record("v3.hotspots.share_cells_top6", float(vc.head(6).sum() / len(hh)))
record("v3.populated_land.n_cells", int((gdp_cells := pd.read_parquet(I / "pixel_cells.parquet")).gdp.gt(0).sum()))
print("v3 extra records done")
# land north of 60 S (the land baseline of script 23): mean rainfall difference and any-negative share
ln = gf[gf.lat > -60]
record("v3.land.mean_RP", float(np.average(ln.RP_2100, weights=ln.area_km2)))
record("v3.land.share_RP_negative", float(np.average(ln.RP_2100 < 0, weights=ln.area_km2)))
record("v3.land.mean_R", float(np.average(ln.R_2100, weights=ln.area_km2)))
record("v3.land.share_R_positive", float(np.average(ln.R_2100 > 0, weights=ln.area_km2)))
record("v3.tree.n_languages_mostly_drier", int(round(NB["heritage.tree.lang_share_mostly_drier"] * NB["heritage.n_languages_with_cells"])))
print("land records done")
record("v3.family.n_single_language", int((fam.n == 1).sum()))
# share of world GDP (CLIMRISK present value) in the deep tropics, for the Discussion
_g = pd.read_parquet(I / "pixel_cells.parquet")[["grid_id", "gdp"]].merge(gf[["grid_id", "lat"]], on="grid_id")
record("v3.deep_tropics_share.gdp", float(_g.gdp[_g.lat.abs() < 15].sum() / _g.gdp.sum()))
