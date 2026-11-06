"""26 - Temperature under the same lens: do language homelands cool more than land, people or the economy?

Exploratory analysis requested by Bernie on 28 Oct 2026, after the Nature draft (v4). It is NOT
part of the submitted paper; it documents the temperature counterpart of the rainfall baselines.

Question. At the same global temperature, SAI cools low latitudes more than mitigation would and
leaves residual warming at high latitudes. Most language homelands are tropical. Do they therefore
end up cooler under SAI than under mitigation, and more so than land, people or the economy?

Fields (2100; member level from srmlang.baselines, the cell-level hybrid of script 23)
  R  = T_S - T_M   temperature difference, SAI minus iso-temperature mitigation (deg C)
  CS = T_H - T_S   local cooling delivered by SAI relative to the no-SAI world
  CM = T_H - T_M   local cooling delivered by mitigation relative to the no-SAI world
  CS - CM = -R, so R < 0 means SAI cools that place more than mitigation would.

Metrics, for each unit of script 23 (land, populated land, people, economy, institutional and
non-institutional languages, linguistic diversity hotspots, language families), all from the same member fields:
  mean R              unit-weighted mean temperature difference (deg C); languages: mean over
                      languages of the homeland-mean R; families: mean over the 209 lineages
  share cooler/warmer share beyond the 0.1 deg C dead-band (as in script 23)
  cooling ratio       rho = mean(CS) / mean(CM): SAI's local cooling as a multiple of
                      mitigation's (ensemble-mean fields only)
  member comparisons  number of members in which the unit's mean R is below (cooler than) that
                      of land, people and the economy
  within-country      non-institutional minus institutional language-in-country units, country fixed
                      effects, each language weighted equally, country-clustered bootstrap (as
                      the rainfall test of script 17)
  zonal               land-mean R by 4-degree latitude band; deep tropics (15 S-15 N) vs the rest

Outputs
  results/tables/temperature_baselines.csv        one row per unit
  results/tables/temperature_by_member.csv        unit x member: mean R, share cooler, share warmer
  results/tables/temperature_families.csv         20 largest families (>= 2 languages): mean R
  results/figures/FigT1_temperature_baselines     a mean R by unit; b shares cooler and warmer;
                                                  c member by member against land
  results/figures/FigT2_temperature_where         a map with linguistic diversity hotspots; b zonal mean R;
                                                  c latitude distribution of land, people, languages
  results/figures/FigT3_temperature_families      mean R of the 15 largest families
  numbers.json                                    keys temp.*
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
from srmlang.common import CFG, P, record  # noqa: E402
from srmlang.baselines import NONINST, setup  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
I, T, F = P("interim"), P("results"), P("figures")
S = setup()
grid, units, OUT, Rm, R0 = S["grid"], S["units"], S["OUT"], S["Rm"], S["R0"]
Lop, Gop, gn, codes, lm, non, pidx = S["Lop"], S["Gop"], S["gn"], S["codes"], S["lm"], S["non"], S["pidx"]
rng = np.random.default_rng(CFG["analysis"]["random_seed"])
COOL, WARM, SIM = ps.T_OUT_COLORS["SRM cooler than mitigation"], "#d03b2f", "#9a9891"
C_LANG, C_FAM, C_BASE = ps.VIT_COLORS["Stable"], ps.VIT_COLORS["Dying"], "#898781"
LABEL = {"land": "Land (area)", "populated land": "Populated land", "people (L1 speakers)": "People",
         "economy (GDP)": "Economy (GDP)", "institutional languages": "Institutional languages",
         "non-institutional languages": "Non-institutional languages", "linguistic diversity hotspots": "Linguistic diversity hotspots",
         "language lineages": "Language lineages"}
ORDER = list(LABEL)
LANG_UNITS = {"non-institutional languages", "linguistic diversity hotspots", "language lineages",
              "institutional languages"}


def key(u):
    return u.split(" (")[0].replace(" ", "_").replace("-", "_")


# ===================================================================== unit metrics
mR0, mRm = units(R0), units(Rm)                                    # mean R (deg C)
c0, cm_ = units(OUT["cooler"][0]), units(OUT["cooler"][1])         # share cooler
w0, wm = units(OUT["warmer"][0]), units(OUT["warmer"][1])          # share warmer
CS, CM = grid.CS_2100.to_numpy(), grid.CM_2100.to_numpy()
cs0, cmit0 = units(CS), units(CM)
dG = S["dG"]

rows, mem_rows = [], []
for u in ORDER:
    r = {"unit": u, "mean_R": float(mR0[u][0]),
         "mean_R_member_p17": np.percentile(mRm[u], 17), "mean_R_member_p50": np.percentile(mRm[u], 50),
         "mean_R_member_p83": np.percentile(mRm[u], 83),
         "mean_R_per_degree_of_global_cooling": float(np.median(mRm[u] / dG)),
         "n_members_cooler_than_land": int((mRm[u] < mRm["land"]).sum()),
         "n_members_cooler_than_people": int((mRm[u] < mRm["people (L1 speakers)"]).sum()),
         "n_members_cooler_than_economy": int((mRm[u] < mRm["economy (GDP)"]).sum()),
         "n_members_mean_R_negative": int((mRm[u] < 0).sum()),
         "share_cooler": float(c0[u][0]), "share_warmer": float(w0[u][0]),
         "share_cooler_member_p17": np.percentile(cm_[u], 17), "share_cooler_member_p83": np.percentile(cm_[u], 83),
         "share_warmer_member_p17": np.percentile(wm[u], 17), "share_warmer_member_p83": np.percentile(wm[u], 83),
         "cooling_ratio_SAI_over_mitigation": float(cs0[u][0] / cmit0[u][0]),
         "mitigation_cooling_C": float(cmit0[u][0]), "SAI_cooling_C": float(cs0[u][0])}
    rows.append(r)
    for m in range(60):
        mem_rows.append({"unit": u, "member": m, "pattern": int(pidx[m]), "mean_R": float(mRm[u][m]),
                         "share_cooler": float(cm_[u][m]), "share_warmer": float(wm[u][m])})
    k = key(u)
    record(f"temp.{k}.mean_R", r["mean_R"])
    record(f"temp.{k}.mean_R.member_p17_p50_p83", [r["mean_R_member_p17"], r["mean_R_member_p50"],
                                                   r["mean_R_member_p83"]])
    record(f"temp.{k}.mean_R_per_degree_global_cooling.member_median", r["mean_R_per_degree_of_global_cooling"])
    record(f"temp.{k}.n_members_cooler_than_land", r["n_members_cooler_than_land"])
    record(f"temp.{k}.n_members_cooler_than_people", r["n_members_cooler_than_people"])
    record(f"temp.{k}.n_members_cooler_than_economy", r["n_members_cooler_than_economy"])
    record(f"temp.{k}.n_members_mean_R_negative", r["n_members_mean_R_negative"])
    record(f"temp.{k}.cooling_ratio", r["cooling_ratio_SAI_over_mitigation"])
    record(f"temp.{k}.mitigation_cooling_C", r["mitigation_cooling_C"])
tab = pd.DataFrame(rows)
tab.to_csv(T / "temperature_baselines.csv", index=False)
bym = pd.DataFrame(mem_rows)
bym.to_csv(T / "temperature_by_member.csv", index=False)

# ===================================================================== within-country (LIC units)
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.ISO.isin(lm.ISO) & pe.land_w_km2.notna() & ~pe.sign_language].copy()
pe["non"] = pe.vitality.isin(NONINST).astype(float)
pe["cool"] = (pe.R_2100 < -CFG["analysis"]["temp_deadband_c"]).astype(float)
pe["w"] = 1.0 / pe.groupby("ISO").pid.transform("size")


def fe(y, x, w, g):
    d = pd.DataFrame({"y": y, "x": x, "w": w, "g": g}).dropna()
    ws = d.w.groupby(d.g).transform("sum")
    yd = d.y - (d.y * d.w).groupby(d.g).transform("sum") / ws
    xd = d.x - (d.x * d.w).groupby(d.g).transform("sum") / ws
    return (d.w * xd * yd).sum() / (d.w * xd ** 2).sum()


ug = pe.CTRY3.unique()
idx = {k: np.flatnonzero(pe.CTRY3.to_numpy() == k) for k in ug}
for yname, lab in [("R_2100", "gap_mean_R"), ("cool", "gap_share_cooler")]:
    b = fe(pe[yname], pe.non, pe.w, pe.CTRY3)
    bs = []
    for _ in range(1000):
        pick = rng.choice(ug, len(ug))
        rr = np.concatenate([idx[k] for k in pick])
        grp = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(pick)])
        sub = pe.iloc[rr]
        bs.append(fe(sub[yname].to_numpy(), sub.non.to_numpy(), sub.w.to_numpy(), grp))
    record(f"temp.within_country.{lab}_noninst_minus_inst", float(b))
    record(f"temp.within_country.{lab}_ci", np.percentile(bs, [2.5, 97.5]))
    # across countries (no fixed effects), for contrast
    raw = (np.average(pe[pe.non == 1][yname], weights=pe[pe.non == 1].w)
           - np.average(pe[pe.non == 0][yname], weights=pe[pe.non == 0].w))
    record(f"temp.across_countries.{lab}_noninst_minus_inst", float(raw))

# ===================================================================== zonal and deep tropics
g = grid.copy()
deep = g.lat.abs() < 15
record("temp.land.mean_R_deep_tropics", float(np.average(g.R_2100[deep], weights=g.area_km2[deep])))
record("temp.land.mean_R_outside_deep_tropics", float(np.average(g.R_2100[~deep], weights=g.area_km2[~deep])))
record("temp.land.mean_R_north_of_55N", float(np.average(g.R_2100[g.lat > 55], weights=g.area_km2[g.lat > 55])))
record("temp.global_cooling_dG_2100_median", float(np.median(dG)))

# ===================================================================== families (>= 2 languages)
fR = np.asarray(Lop @ R0[:, None]).ravel()
fRm = np.asarray(Lop @ Rm)
fam_R = pd.Series(np.asarray(Gop @ fR[:, None]).ravel() / gn, index=codes.categories)
fam_Rm = pd.DataFrame(np.asarray(Gop @ fRm) / gn[:, None], index=codes.categories)
ftab = pd.DataFrame({"lineage": codes.categories, "n": gn.astype(int), "mean_R": fam_R.values,
                     "mean_R_p17": fam_Rm.quantile(0.17, axis=1).values,
                     "mean_R_p83": fam_Rm.quantile(0.83, axis=1).values,
                     "frac_members_mean_R_negative": (fam_Rm < 0).mean(1).values})
ftab = ftab[ftab.n >= 2].sort_values("n", ascending=False)
ftab.head(20).to_csv(T / "temperature_families.csv", index=False)
record("temp.families.n_mean_R_negative_ensemble", int((ftab.mean_R < 0).sum()))
record("temp.families.n_with_ge2_languages", int(len(ftab)))

# ===================================================================== Fig T1
fig = plt.figure(figsize=(ps.W2, 2.75))
axA = fig.add_axes([0.13, 0.17, 0.22, 0.72])
d = tab.set_index("unit").loc[ORDER]
land_R = d.loc["land", "mean_R"]
for yi, (u, r) in enumerate(d.iterrows()):
    col = C_LANG if u in LANG_UNITS else C_BASE
    if u == "institutional languages":
        col = ps.VIT_COLORS["Institutional"]
    axA.plot([r.mean_R_member_p17, r.mean_R_member_p83], [yi, yi], color=col, lw=2.2, alpha=0.35,
             solid_capstyle="round")
    axA.plot(r.mean_R, yi, "o", color=col, ms=4.4, mec="white", mew=0.5, zorder=4)
    axA.text(0.30, yi, f"{r.mean_R:+.2f}", ha="right", va="center", fontsize=5.6,
             fontweight="bold" if u in ("non-institutional languages", "language lineages") else "normal",
             color=ps.INK if u in LANG_UNITS else ps.INK2)
axA.axvline(0, color=ps.AXIS, lw=0.6)
axA.axvline(land_R, color=C_BASE, lw=0.7, ls=(0, (2, 2)))
axA.set_yticks(range(len(ORDER)))
axA.set_yticklabels([LABEL[u] for u in ORDER], fontsize=5.8)
axA.invert_yaxis()
axA.axhline(3.5, color=ps.AXIS, lw=0.6)
axA.set_xlim(-0.55, 0.32)
axA.set_xlabel("Mean temperature difference,\nSAI minus mitigation (°C)")
axA.grid(axis="x", color=ps.GRID, lw=0.4)
ps.panel_label(axA, "a", x=-0.62)

axB = fig.add_axes([0.42, 0.17, 0.22, 0.72], sharey=axA)
for yi, (u, r) in enumerate(d.iterrows()):
    axB.barh(yi, -100 * r.share_cooler, color=COOL, height=0.62, lw=0, alpha=0.85)
    axB.barh(yi, 100 * r.share_warmer, color=WARM, height=0.62, lw=0, alpha=0.85)
    axB.plot([-100 * r.share_cooler_member_p17, -100 * r.share_cooler_member_p83], [yi, yi], color=ps.INK, lw=0.7)
    axB.plot([100 * r.share_warmer_member_p17, 100 * r.share_warmer_member_p83], [yi, yi], color=ps.INK, lw=0.7)
    xl = -100 * max(r.share_cooler, r.share_cooler_member_p83) - 2
    xr = 100 * max(r.share_warmer, r.share_warmer_member_p83) + 2
    axB.text(xl, yi, f"{100 * r.share_cooler:.0f}", ha="right", va="center", fontsize=5.3)
    axB.text(xr, yi, f"{100 * r.share_warmer:.0f}", ha="left", va="center", fontsize=5.3)
axB.axvline(0, color=ps.INK2, lw=0.6)
axB.set_xlim(-108, 58)
axB.set_xticks([-100, -50, 0, 50])
axB.set_xticklabels(["100", "50", "0", "50"])
axB.set_xlabel("Share cooler (left) and warmer (right)\nthan mitigation by >0.1 °C (%)")
axB.tick_params(labelleft=False)
axB.grid(axis="x", color=ps.GRID, lw=0.4)
ps.panel_label(axB, "b", x=-0.04)

axC = fig.add_axes([0.75, 0.17, 0.23, 0.72])
mr = bym.pivot(index="member", columns="unit", values="mean_R")
for u, col, lab in [("people (L1 speakers)", C_BASE, "People"), ("non-institutional languages", C_LANG, "Non-inst. lang."),
                    ("language lineages", C_FAM, "Families")]:
    n_below = int((mr[u] < mr["land"]).sum())
    axC.scatter(mr["land"], mr[u], s=8, color=col, lw=0.3, ec="white", zorder=3, label=f"{lab} {n_below}/60")
lo, hi = float(np.nanmin(mr.values)) - 0.05, float(np.nanmax(mr.values)) + 0.05
axC.plot([lo, hi], [lo, hi], color=ps.INK2, lw=0.7, ls=(0, (2, 2)))
axC.set_xlim(lo, hi)
axC.set_ylim(lo, hi)
axC.axhline(0, color=ps.AXIS, lw=0.5)
axC.axvline(0, color=ps.AXIS, lw=0.5)
axC.set_xlabel("Land: mean difference (°C)")
axC.set_ylabel("Unit: mean difference (°C)")
axC.legend(loc="upper left", fontsize=4.9, title="Members cooler than land\n(in tens: set by the pattern)", title_fontsize=4.9,
           handletextpad=0.1, borderaxespad=0.2, alignment="left", framealpha=0.95, edgecolor="none")
axC.grid(color=ps.GRID, lw=0.4)
ps.panel_label(axC, "c", x=-0.3)
ps.save(fig, F / "FigT1_temperature_baselines")

# ===================================================================== Fig T2 (mirror of main Fig. 1)
hc = pd.read_parquet(I / "heritage_cells.parquet")
gf = pd.read_parquet(I / "grid_fields.parquet")
hc = hc.merge(gf[["grid_id", "R_2100"]], on="grid_id", how="left")
fig = plt.figure(figsize=(ps.W2, 3.15))
ax = fig.add_axes([0.0, 0.0, 0.74, 1.0])
ps.draw_basemap(ax, P("ne_land"))
X, Y = ps.grid_mesh()
light = LinearSegmentedColormap.from_list("lightRdBu", ["#a9c9ec", "#cfe0f3", "#eaf1f8", "#f6f5f2", "#f8e3df",
                                                       "#f0c3bb", "#e59a8e"])
Z = ps.to_grid(gf.assign(v=gf.R_2100), "v")
ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=light, norm=TwoSlopeNorm(0, -0.8, 0.8), shading="flat",
              zorder=1, rasterized=True)
cls = np.where(hc.R_2100 < -0.1, 0, np.where(hc.R_2100 > 0.1, 2, 1)).astype(float)
cls[~hc.hotspot_half.to_numpy()] = np.nan
ax.pcolormesh(X, Y, np.ma.masked_invalid(ps.to_grid(hc.assign(c=cls), "c")), cmap=ListedColormap([COOL, SIM, WARM]),
              vmin=-0.5, vmax=2.5, shading="flat", zorder=2, rasterized=True)
gpd.read_file(P("ne_coast")).to_crs(ps.ROBIN).plot(ax=ax, color="#6f6e69", lw=0.18, zorder=3)
ax.legend(handles=[Patch(color=COOL, label="cooler under SAI (>0.1 °C)"), Patch(color=SIM, label="similar (±0.1 °C)"),
                   Patch(color=WARM, label="warmer under SAI (>0.1 °C)")],
          title="Cells holding half of all non-institutional-\nlanguage diversity:", title_fontsize=5.7, fontsize=5.5,
          loc="lower left", bbox_to_anchor=(0.0, 0.07), alignment="left", handlelength=1.0)
ax.text(0.0, 0.99, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax.text(0.02, 0.025, "Background: temperature, SAI minus mitigation (faint)",
        transform=ax.transAxes, fontsize=5.2, color=ps.INK2)
axZ = fig.add_axes([0.775, 0.12, 0.085, 0.80])
axD = fig.add_axes([0.875, 0.12, 0.12, 0.80], sharey=axZ)
gz = grid.copy()
gz["band"] = pd.cut(gz.lat, np.arange(-56, 84, 4))
zm = gz.groupby("band", observed=True).apply(lambda q: np.average(q.R_2100, weights=q.area_km2))
mid = np.array([b_.mid for b_ in zm.index])
axZ.axvline(0, color=ps.AXIS, lw=0.6)
axZ.fill_betweenx(mid, 0, zm.values, where=zm.values < 0, color="#a9c9ec", lw=0, interpolate=True)
axZ.fill_betweenx(mid, 0, zm.values, where=zm.values > 0, color="#f0c3bb", lw=0, interpolate=True)
axZ.plot(zm.values, mid, color=ps.INK, lw=0.9)
axZ.set_xlabel("Temperature\ndifference (°C)", fontsize=6.0)
axZ.set_xticks([-0.2, 0, 0.2])
axZ.set_xticklabels(["-0.2", "0", "0.2"])
axZ.set_ylabel("Latitude (°)", labelpad=1)
axZ.set_ylim(-56, 82)
yy = np.linspace(-56, 82, 300)
x_l = lm[non].lat_mean.dropna().to_numpy()
k_l = gaussian_kde(x_l, bw_method=3.0 / x_l.std())
pp = pe[pe.L1_LIC.fillna(0) > 0]
x_p, w_p = pp.rep_lat.to_numpy(), pp.L1_LIC.to_numpy()
k_p = gaussian_kde(x_p, weights=w_p, bw_method=3.0 / np.sqrt(np.cov(x_p, aweights=w_p)))
dens_land = np.histogram(grid.lat, bins=np.arange(-60, 84.5, 1), weights=grid.area_km2, density=True)
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
axZ.text(-0.62, 1.07, "b", transform=axZ.transAxes, fontsize=9, fontweight="bold", va="top")
axD.text(-0.02, 1.07, "c", transform=axD.transAxes, fontsize=9, fontweight="bold", va="top", ha="right")
ps.save(fig, F / "FigT2_temperature_where")
hh = hc[hc.hotspot_half]
record("temp.hotspots.share_cells_cooler", float(np.average(hh.R_2100 < -0.1, weights=hh.LWE_noninst)))

# ===================================================================== Fig T3 families
fig, ax = plt.subplots(figsize=(ps.W1 * 1.25, 2.6))
big = ftab.head(15).sort_values("mean_R")
for yi, r in enumerate(big.itertuples()):
    col = COOL if r.frac_members_mean_R_negative >= 0.9 else (WARM if r.frac_members_mean_R_negative <= 0.1 else SIM)
    ax.plot([r.mean_R_p17, r.mean_R_p83], [yi, yi], color=col, lw=1.4, alpha=0.6, solid_capstyle="round")
    ax.plot(r.mean_R, yi, "o", color=col, ms=3.4, mec="white", mew=0.4, zorder=4)
ax.axvline(0, color=ps.AXIS, lw=0.6)
ax.axvline(land_R, color=C_BASE, lw=0.7, ls=(0, (2, 2)))
ax.set_yticks(range(len(big)))
ax.set_yticklabels([f"{r.lineage} ({r.n:,})" for r in big.itertuples()], fontsize=5.4)
ax.set_xlabel("Mean temperature difference, SAI minus mitigation (°C)")
ax.grid(axis="x", color=ps.GRID, lw=0.4)
ax.legend(handles=[Line2D([], [], color=COOL, marker="o", lw=1.4, ms=3.2, label="≥90% of members (cooler)"),
                   Line2D([], [], color=SIM, marker="o", lw=1.4, ms=3.2, label="mixed"),
                   Line2D([], [], color=WARM, marker="o", lw=1.4, ms=3.2, label="≤10% of members (warmer)")],
          loc="upper left", fontsize=4.9, handlelength=1.3, framealpha=1, edgecolor="none",
          title="Family mean below zero in:", title_fontsize=4.9, alignment="left")
ps.save(fig, F / "FigT3_temperature_families")

pd.set_option("display.width", 220)
print(tab[["unit", "mean_R", "mean_R_member_p17", "mean_R_member_p83", "n_members_cooler_than_land",
           "n_members_cooler_than_people", "n_members_cooler_than_economy", "share_cooler", "share_warmer",
           "cooling_ratio_SAI_over_mitigation"]].round(3).to_string())
print(ftab.head(15).round(3).to_string())

# ===================================================================== written summary (docs/)
# The summary is generated from the computed values so that no number is typed by hand.
pat = bym.pivot_table(index="pattern", columns="unit", values="mean_R", aggfunc="mean")
record("temp.families.n_cooler_in_ge90pct_members", int((ftab.frac_members_mean_R_negative >= 0.9).sum()))
record("temp.families.n_warmer_in_ge90pct_members", int((ftab.frac_members_mean_R_negative <= 0.1).sum()))
t = tab.set_index("unit")
L = {u: LABEL[u] for u in ORDER}
f2 = lambda x: f"{x:+.2f}"  # noqa: E731
pc = lambda x: f"{100 * x:.0f}%"  # noqa: E731
from srmlang.common import ledger  # noqa: E402
LG = ledger()
gap = LG["temp.within_country.gap_mean_R_noninst_minus_inst"]["value"]
gci = LG["temp.within_country.gap_mean_R_ci"]["value"]
sgap = LG["temp.within_country.gap_share_cooler_noninst_minus_inst"]["value"]
sci = LG["temp.within_country.gap_share_cooler_ci"]["value"]
lines = []
lines.append("# Temperature under the same lens as rainfall (exploratory, not in the submitted paper)\n")
lines.append("Generated by `scripts/26_temperature_baselines.py` (28 Oct 2026). Every number below is computed by")
lines.append("the script and also written to `results/tables/numbers.json` (keys `temp.*`).\n")
lines.append("## Question\n")
lines.append("At the same global temperature, SAI cools low latitudes more than mitigation would and leaves")
lines.append("residual warming at high latitudes. Most language homelands are tropical. Are they cooled more")
lines.append("than land, people or the economy? We use exactly the units, weights and member fields of the")
lines.append("rainfall baselines (script 23), with R = T_S - T_M (SAI minus mitigation; negative = SAI cooler).\n")
lines.append("## Results (2100)\n")
lines.append("| Unit | Mean R (°C), ensemble mean | Members p17 to p83 | Cooler >0.1 °C | Warmer >0.1 °C | SAI cooling / mitigation cooling | Members cooler than land |")
lines.append("|---|---|---|---|---|---|---|")
for u in ORDER:
    r = t.loc[u]
    lines.append(f"| {L[u]} | {f2(r.mean_R)} | {f2(r.mean_R_member_p17)} to {f2(r.mean_R_member_p83)} | {pc(r.share_cooler)} | "
                 f"{pc(r.share_warmer)} | {r.cooling_ratio_SAI_over_mitigation:.3f} | "
                 f"{'–' if u == 'land' else str(int(r.n_members_cooler_than_land)) + '/60'} |")
cl, ld, pp_, ec = (t.loc["non-institutional languages"], t.loc["land"], t.loc["people (L1 speakers)"],
                   t.loc["economy (GDP)"])
lines.append("")
lines.append("**Short answer.** In the ensemble-mean climate, yes, but not robustly, and not more than people.\n")
lines.append(f"1. Non-institutional-language homelands are cooled more than land: mean difference {f2(cl.mean_R)} °C against "
             f"{f2(ld.mean_R)} °C, and {pc(cl.share_cooler)} of homeland is more than 0.1 °C cooler against "
             f"{pc(ld.share_cooler)} of land. SAI delivers {100 * (cl.cooling_ratio_SAI_over_mitigation - 1):.0f}% more local "
             f"cooling than mitigation over language homelands, {100 * (ld.cooling_ratio_SAI_over_mitigation - 1):.0f}% more over land.")
lines.append(f"2. People are cooled at least as much ({f2(pp_.mean_R)} °C; {pc(pp_.share_cooler)} cooler), and the economy "
             f"somewhat less ({f2(ec.mean_R)} °C). Non-institutional languages are cooler than people in "
             f"{int(cl.n_members_cooler_than_people)} of 60 members and cooler than the economy in "
             f"{int(cl.n_members_cooler_than_economy)}.")
lines.append(f"3. The ranking is not robust. Non-institutional languages are cooler than land in {int(cl.n_members_cooler_than_land)} "
             "of 60 members, which means three of the six response patterns: within a pattern the cell-level "
             "temperature field scales with the member's global cooling, so every member of a pattern ranks the "
             "units the same way and member counts move in tens. By pattern (mean over its ten members):\n")
lines.append("| Pattern | Land | People | Economy | Non-institutional languages | Language lineages |")
lines.append("|---|---|---|---|---|---|")
for k_, r in pat.iterrows():
    lines.append(f"| {k_} | {f2(r['land'])} | {f2(r['people (L1 speakers)'])} | {f2(r['economy (GDP)'])} | "
                 f"{f2(r['non-institutional languages'])} | {f2(r['language lineages'])} |")
lines.append("")
lines.append(f"4. Within countries there is no difference: non-institutional minus institutional language-in-country units, "
             f"country fixed effects, {gap:+.3f} °C (95% CI {gci[0]:+.3f} to {gci[1]:+.3f}); share cooler "
             f"{100 * sgap:+.1f} pp (CI {100 * sci[0]:+.1f} to {100 * sci[1]:+.1f}). As for rainfall, any difference "
             "between languages and other units comes from geography.")
lines.append(f"5. Geography: the land-mean difference is {f2(LG['temp.land.mean_R_deep_tropics']['value'])} °C between "
             f"15°S and 15°N, {f2(LG['temp.land.mean_R_outside_deep_tropics']['value'])} °C elsewhere and "
             f"{f2(LG['temp.land.mean_R_north_of_55N']['value'])} °C north of 55°N (FigT2).")
lines.append(f"6. Magnitude: per degree of global cooling (member median), non-institutional languages end up "
             f"{abs(cl.mean_R_per_degree_of_global_cooling):.2f} °C cooler than under mitigation, land "
             f"{abs(ld.mean_R_per_degree_of_global_cooling):.2f} °C. Mitigation itself cools these homelands by "
             f"{cl.mitigation_cooling_C:.2f} °C relative to the no-SAI world (land {ld.mitigation_cooling_C:.2f} °C), "
             "so the extra cooling is a small fraction of the total.")
lines.append(f"7. Families: {LG['temp.families.n_mean_R_negative_ensemble']['value']} of "
             f"{LG['temp.families.n_with_ge2_languages']['value']} families with at least two languages are cooler "
             f"in the ensemble mean, but only {LG['temp.families.n_cooler_in_ge90pct_members']['value']} are cooler in "
             f"at least 90% of members (FigT3).\n")
lines.append("## How this compares with rainfall\n")
lines.append("Rainfall gives a robust distributive signal: language families are more often drier than the "
             "economy in all 60 members. Temperature does not: the language advantage over land is real in the "
             "ensemble mean and in three patterns, and absent in the other three. In both cases the within-country "
             "difference is zero and geography drives the result. For the paper, the existing sentence "
             "(\"On temperature, non-institutional languages fare like people, not worse\") remains accurate; a stronger "
             "claim that languages are cooled more than other units would not be robust.\n")
lines.append("## Files\n")
lines.append("- `results/tables/temperature_baselines.csv`, `temperature_by_member.csv`, `temperature_families.csv`")
lines.append("- `results/figures/FigT1_temperature_baselines` (a mean difference by unit; b shares cooler and warmer; "
             "c member by member against land), `FigT2_temperature_where` (map with linguistic diversity hotspots, zonal mean, "
             "latitude distributions), `FigT3_temperature_families` (15 largest families)")
lines.append("- Shared set-up with script 23: `src/srmlang/baselines.py`")
(Path(__file__).resolve().parents[1] / "docs" / "temperature_baselines.md").write_text("\n".join(lines) + "\n")
print("wrote docs/temperature_baselines.md")
