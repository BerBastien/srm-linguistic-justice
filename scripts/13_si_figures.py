"""Supplementary figures S1-S10 (methods and robustness).

S1  global trajectories and the iso-GMST design; cooling-ratio map
S2  validation: pattern-scaling identities, approximation A1/A2, damage emulator
S3  overlay illustration (southern Mexico): language polygons on the 0.5-degree grid
S4  structural (pattern) uncertainty: group statistics for each response pattern
S5  robustness of the endangered-minus-not gaps across estimands and data choices
S6  concentration curves by speaker-community size
S7  country maps of S-M, S+T-M and the termination premium
S8  (speaker-weighted alluvial; produced by 11_fig3_sankey.py L1)
S9  scenario lens: cultural-loss weights and language-survival pairings
S10 where languages are: vitality composition by world area and Annex I status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, LIVING_ORDER, P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402
from srmlang import stats as st  # noqa: E402

ps.setup()
I, T, F = P("interim"), P("results"), P("figures")
grid = pd.read_parquet(I / "grid_fields.parquet")
lm = pd.read_parquet(I / "lang_metrics.parquet")
glob = pd.read_csv(T / "global_trajectories.csv")


def lab(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold")


# =============================================================== S1
fig = plt.figure(figsize=(ps.W2, 4.6))
a1 = fig.add_axes([0.07, 0.58, 0.38, 0.38])
a2 = fig.add_axes([0.57, 0.58, 0.38, 0.38])
a3 = fig.add_axes([0.0, 0.02, 1.0, 0.48])
yr = glob.year
for nm, col, lb in (("G_H", "#d03b3b", "H: SCEN7-H, no SAI"),
                    ("G_S", "#2a78d6", "S: SCEN7-H + SAI  (= M: mitigation, by construction)")):
    a1.fill_between(yr, glob[f"{nm}_p17"], glob[f"{nm}_p83"], color=col, alpha=0.18, lw=0)
    a1.plot(yr, glob[f"{nm}_p50"], color=col, lw=1.3, label=lb)
a1.axhline(2.0, color=ps.MUTED, lw=0.5, ls=(0, (2, 2)))
a1.set_ylabel("Global mean temperature\nrelative to 1900 (°C)")
a1.legend(loc="upper left", fontsize=5.5)
a1.grid(color=ps.GRID, lw=0.4)
lab(a1, "A")
a2.fill_between(yr, glob.dG_p17, glob.dG_p83, color=ps.INK2, alpha=0.18, lw=0)
a2.plot(yr, glob.dG_p50, color=ps.INK, lw=1.3)
a2.set_ylabel("Global cooling delivered\nΔG = G_H − G_S (°C)")
a2.grid(color=ps.GRID, lw=0.4)
lab(a2, "B")
X, Y = ps.grid_mesh()
ps.draw_basemap(a3, P("ne_land"))
rho = grid.CS_2100 / grid.CM_2100
Z = ps.to_grid(grid.assign(rho=rho.where(grid.CM_2100.abs() > 0.3)), "rho")
m = a3.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=ps.CMAP_T.reversed(),
                  norm=TwoSlopeNorm(1.0, 0.7, 1.3), shading="flat", rasterized=True)
cax = a3.inset_axes([0.04, 0.12, 0.22, 0.035])
cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both")
cb.ax.tick_params(labelsize=5.6, length=2)
cb.outline.set_visible(False)
cb.ax.set_title("Cooling ratio ρ = (T_H − T_S)/(T_H − T_M), 2100\n>1: SAI cools more than mitigation",
                fontsize=5.8, loc="left", fontweight="normal")
a3.text(0.0, 0.98, "C", transform=a3.transAxes, fontsize=9, fontweight="bold", va="top")
ps.save(fig, F / "FigS1_design_trajectories_rho")

# =============================================================== S2 validation
cm = np.load(I / "country_member_fields.npz", allow_pickle=True)
cv = pd.read_csv(T / "emulator_cv.csv")
f5 = h5py.File(P("climrisk_h5"), "r")
fig, ax = plt.subplots(1, 3, figsize=(ps.W2, 2.4))
GS, GH = cm["GS"], cm["GH"]
j = -1
beta = cm["beta"]
exact = (beta * GS[:, [j]]).mean(0)
A1 = cm["TH_2100"].mean(0) * (GS[:, j].mean() / GH[:, j].mean())
ok = np.isfinite(exact)
ax[0].scatter(exact[ok], A1[ok], s=4, color=ps.INK2, lw=0)
lim = [np.nanmin(exact) - 0.1, np.nanmax(exact) + 0.1]
ax[0].plot(lim, lim, color="#d03b3b", lw=0.6)
ax[0].set_xlabel("Exact E[T_M], country (°C)")
ax[0].set_ylabel("Approximation A1 (°C)")
nb = ps.numbers(T / "numbers.json")
ax[0].text(0.04, 0.9, f"mean |error| {nb['validation.A1_err_mean.2100']:.3f} °C\n"
           f"max {nb['validation.A1_err_max.2100']:.3f} °C\nA2 removes it at country level",
           transform=ax[0].transAxes, fontsize=5.5, va="top")
lab(ax[0], "A", x=-0.25)
ax[1].hist(np.log10(cv.rel_rmse.clip(lower=1e-5)), bins=40, color=ps.INK2)
ax[1].set_xlabel("log$_{10}$ relative CV RMSE of the\nmitigation-damage emulator")
ax[1].set_ylabel("Country × pattern fits")
ax[1].text(0.04, 0.9, f"median R² = {nb['emulator.median_r2']:.5f}\n"
           f"median rel. RMSE = {100 * nb['emulator.median_rel_rmse']:.2f}%",
           transform=ax[1].transAxes, fontsize=5.5, va="top")
lab(ax[1], "B", x=-0.25)
iso = list(cm["iso3"])
gdp = f5["country/economic_annual/gdp_billion_usd2005_ppp"][:]
LH = f5["country/economic_annual/ordinary_no_srm/positive_loss_by_member"][:]
LS = f5["country/economic_annual/ordinary_srm/positive_loss_by_member"][:]
pidx = cm["pattern_index"]
cols = {"NGA": "#2a78d6", "PNG": "#eb6834", "MEX": "#1baf7a"}
for c3, colr in cols.items():
    c = iso.index(c3)
    ms = pidx == 1
    ax[2].scatter(GH[ms].ravel(), 100 * (LH[ms, :, c] / gdp[None, :, c]).ravel(), s=1.2,
                  color=colr, alpha=0.4, lw=0, rasterized=True)
    ax[2].scatter(GS[ms].ravel(), 100 * (LS[ms, :, c] / gdp[None, :, c]).ravel(), s=1.2,
                  color=colr, alpha=0.4, lw=0, marker="x", rasterized=True)
    ax[2].text(4.0, 100 * np.nanmax(LH[ms, -1, c] / gdp[-1, c]), c3, color=colr, fontsize=6)
ax[2].set_xlabel("Global mean temperature (°C)")
ax[2].set_ylabel("Annual KU loss (% of GDP)")
ax[2].text(0.03, 0.92, "dots: no SRM (fitting data)\ncrosses: SRM (different pattern)",
           transform=ax[2].transAxes, fontsize=5.3, va="top")
lab(ax[2], "C", x=-0.25)
for a in ax:
    a.grid(color=ps.GRID, lw=0.4)
fig.tight_layout()
ps.save(fig, F / "FigS2_validation")

# =============================================================== S3 overlay illustration
polys = gpd.read_parquet(I / "eth_polygons.parquet")
box = (-99.5, 15.0, -91.5, 19.5)
sub = polys.cx[box[0]:box[2], box[1]:box[3]]
sub = sub[~sub.sign_language & sub.vitality.isin(LIVING_ORDER)]
fig, ax = plt.subplots(1, 2, figsize=(ps.W2, 3.0))
g = grid[(grid.lon >= box[0] - 1) & (grid.lon <= box[2] + 1) & (grid.lat >= box[1] - 1) &
         (grid.lat <= box[3] + 1)]
for a, colname, cmap, v, title in ((ax[0], "R_2100", ps.CMAP_T, 0.4, "Temperature, SAI minus mitigation (°C)"),
                                   (ax[1], "RP_2100", ps.CMAP_P, 20, "Rainfall, SAI minus mitigation (pp)")):
    for _, r_ in g.iterrows():
        pass
    lon_e = np.arange(box[0] - 0.25, box[2] + 0.75, 0.5)
    lat_e = np.arange(box[1] - 0.25, box[3] + 0.75, 0.5)
    Zg = np.full((lat_e.size - 1, lon_e.size - 1), np.nan)
    for _, r_ in g.iterrows():
        ii = int(round((r_.lat - (box[1])) / 0.5))
        jj = int(round((r_.lon - (box[0])) / 0.5))
        if 0 <= ii < Zg.shape[0] and 0 <= jj < Zg.shape[1]:
            Zg[ii, jj] = r_[colname]
    mm = a.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(Zg), cmap=cmap,
                      norm=TwoSlopeNorm(0, -v, v), edgecolors="white", lw=0.3)
    # Ethnologue polygons are licensed: draw their outlines as raster layers, never as vector paths
    sub.boundary.plot(ax=a, color=ps.INK, lw=0.35, rasterized=True)
    end = sub[sub.vitality.isin(["In trouble", "Dying"])]
    end.boundary.plot(ax=a, color="#5240b8", lw=0.9, rasterized=True)
    a.set_xlim(box[0], box[2])
    a.set_ylim(box[1], box[3])
    a.set_aspect("equal")
    a.set_title(title, fontsize=7)
    fig.colorbar(mm, ax=a, shrink=0.6, pad=0.02)
    a.tick_params(labelsize=5.5)
    a.set_xlabel("Longitude (°)")
    a.set_ylabel("Latitude (°)")
ax[0].text(0.01, 0.01, f"{len(sub)} LIC polygons (black); endangered outlined in violet",
           transform=ax[0].transAxes, fontsize=5.4, va="bottom",
           bbox=dict(fc="white", ec="none", alpha=0.8))
lab(ax[0], "A", x=-0.08)
lab(ax[1], "B", x=-0.08)
fig.tight_layout()
ps.save(fig, F / "FigS3_overlay_mexico")

# =============================================================== S4 pattern uncertainty
ms_ = pd.read_csv(T / "member_group_stats.csv")
d = ms_[ms_.horizon == 2100]
fig, ax = plt.subplots(1, 3, figsize=(ps.W2, 2.3))
for a, prefix, ttl, cmap, v in ((ax[0], "R_mean_", "Mean residual T (°C)", ps.CMAP_T, 0.4),
                                (ax[1], "RP_median_", "Median residual P (pp)", ps.CMAP_P, 8),
                                (ax[2], "share_RPneg_", "Share drier > 2 pp", "Purples", None)):
    M = d.groupby("pattern")[[prefix + g for g in LIVING_ORDER]].mean().to_numpy()
    if v is None:
        im = a.imshow(M * 100, cmap=cmap, vmin=40, vmax=90, aspect="auto")
        fmt = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    else:
        im = a.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
        fmt = (lambda x: f"{x:+.2f}") if "R_" in prefix else (lambda x: f"{x:+.1f}")  # noqa
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            a.text(k, i, fmt(M[i, k]), ha="center", va="center", fontsize=5.3)
    a.set_xticks(range(4))
    a.set_xticklabels(["Instit.", "Stable", "In trouble", "Dying"], fontsize=5.6)
    a.set_yticks(range(6))
    a.set_yticklabels([f"Pattern {k}" for k in range(1, 7)], fontsize=5.6)
    a.set_title(ttl, fontsize=6.6)
for a, s in zip(ax, "abc"):
    lab(a, s, x=-0.32)
fig.tight_layout()
ps.save(fig, F / "FigS4_pattern_uncertainty")

# =============================================================== S5 robustness forest
gaps = pd.read_csv(T / "table_gaps_tests.csv")
rob = pd.read_csv(T / "table_robustness.csv")
rows = []
for _, r_ in gaps.iterrows():
    rows.append((r_.weighting, r_.variable, r_.gap_end_minus_non, r_.ci_lo, r_.ci_hi))
fig, ax = plt.subplots(1, 3, figsize=(ps.W2, 2.6), sharey=True)
wl = {"w_lang": "language-weighted", "w_spk": "speaker-weighted", "w_area": "territory-weighted"}
for a, var, ttl in ((ax[0], "R_2100", "Residual T 2100 (°C)"),
                    (ax[1], "drier_2100", "Share with drying > 2 pp (pp)"),
                    (ax[2], "econ_premium", "Termination premium (pp)")):
    sel = gaps[gaps.variable == var].reset_index(drop=True)
    for i, r_ in sel.iterrows():
        a.plot([r_.ci_lo, r_.ci_hi], [i, i], color=ps.INK, lw=0.8)
        a.scatter(r_.gap_end_minus_non, i, s=12, color="#5240b8", zorder=3)
    # variants (language-weighted) from robustness table
    k0 = len(sel)
    for jv, var_name in enumerate(["no_fallback", "with_sign", "LIC_units"]):
        rr = rob[(rob.variant == var_name) & (rob.horizon == 2100)].set_index("group")
        col = {"R_2100": "R_mean", "drier_2100": "share_RP_drier",
               "econ_premium": "econ_premium_p50"}[var]
        if col in rr and pd.notna(rr.loc["Endangered", col]):
            gval = rr.loc["Endangered", col] - rr.loc["Not endangered", col]
            if col == "share_RP_drier":
                gval *= 100
            a.scatter(gval, k0 + jv, s=12, marker="s", color=ps.INK2, zorder=3)
    a.axvline(0, color=ps.AXIS, lw=0.7)
    a.set_title(ttl, fontsize=6.6)
    a.grid(axis="x", color=ps.GRID, lw=0.4)
ylabs = [wl[w] for w in gaps[gaps.variable == "R_2100"].weighting] + \
    ["no nearest-cell fallback", "including sign languages", "LIC units (community EGIDS)"]
ax[0].set_yticks(range(len(ylabs)))
ax[0].set_yticklabels(ylabs, fontsize=5.6)
ax[0].invert_yaxis()
fig.supxlabel("Endangered minus not endangered (circles: mean with bootstrap 95% CI; squares: "
              "alternative samples, language-weighted)", fontsize=6)
for a, s in zip(ax, "abc"):
    lab(a, s, x=-0.1)
fig.tight_layout()
ps.save(fig, F / "FigS5_robustness")

# =============================================================== S6 concentration curves
fig, ax = plt.subplots(1, 3, figsize=(ps.W2, 2.4))
size = np.log10(lm.L1_Users.fillna(0).clip(lower=0) + 1)
burd = {"Residual warming  max(R,0)": np.maximum(lm.R_2100, 0),
        "Residual drying  max(−R_P,0)": np.maximum(-lm.RP_2100, 0),
        "Termination premium": lm.econ_premium}
for a, (name, B) in zip(ax, burd.items()):
    okk = np.isfinite(B) & np.isfinite(size)
    x, y = st.concentration_curve(B[okk], size[okk])
    ci_ = st.concentration_index(B[okk], size[okk])
    a.plot([0, 1], [0, 1], color=ps.AXIS, lw=0.7)
    a.plot(x, y, color="#5240b8", lw=1.3)
    a.set_title(name, fontsize=6.4)
    a.text(0.05, 0.9, f"CI = {ci_:+.3f}", transform=a.transAxes, fontsize=6)
    a.set_xlabel("Cumulative share of languages,\nsmallest speaker communities first")
    a.grid(color=ps.GRID, lw=0.4)
ax[0].set_ylabel("Cumulative share of burden")
for a, s in zip(ax, "abc"):
    lab(a, s, x=-0.2)
fig.tight_layout()
ps.save(fig, F / "FigS6_concentration_curves")

# =============================================================== S7 country maps
cty = pd.read_csv(T / "country_pv_by_pathway.csv")
ne = gpd.read_file(P("ne_countries")).to_crs(ps.ROBIN)
ne["iso3"] = ne["ISO_A3_EH"]
ne = ne.merge(cty, on="iso3", how="left")
fig, ax = plt.subplots(3, 1, figsize=(ps.W15, 6.0))
for a, colname, cmap, norm, ttl in (
        (ax[0], "S_minus_M_p50", ps.CMAP_T, TwoSlopeNorm(0, -1.5, 1.5),
         "A  S − M: continuous SRM minus mitigation (pp of PV GDP)"),
        (ax[1], "ST_minus_M_p50", ps.CMAP_T, TwoSlopeNorm(0, -10, 10),
         "B  S+T − M: SRM with termination risk minus mitigation (pp)"),
        (ax[2], "premium_p50", "Reds", plt.Normalize(0, 10),
         "C  Termination premium (% of PV GDP)")):
    ne.plot(column=colname, ax=a, cmap=cmap, norm=norm, edgecolor="white", lw=0.15,
            missing_kwds=dict(color="#e1e0d9"))
    a.set_axis_off()
    a.set_title(ttl, fontsize=6.8, loc="left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=a, shrink=0.6, pad=0.01)
fig.tight_layout()
ps.save(fig, F / "FigS7_country_maps")

# =============================================================== S9 scenario lens
sc = pd.read_csv(T / "scenario_cultural_weights.csv")
fig, ax = plt.subplots(1, 2, figsize=(ps.W2, 2.4))
styles = {("L0", "L0"): ("#9a9891", "persistence in both worlds"),
          ("L1", "L0"): ("#7b67d9", "SRM world loses Dying languages"),
          ("L2", "L1"): ("#5240b8", "SRM world loses all endangered; M loses Dying"),
          ("L2", "L0"): ("#2b1d78", "SRM world loses all endangered; M none")}
for (s_, m_), (col, lab_) in styles.items():
    dd = sc[(sc.survival_SRM_world == s_) & (sc.survival_M_world == m_)]
    ax[0].plot(dd.kappa, dd.CW_diff_mean, "o-", color=col, ms=2.5, lw=1, label=lab_)
    ax[1].plot(dd.kappa, 100 * dd.share_diff_endangered, "o-", color=col, ms=2.5, lw=1)
ax[1].axhline(100 * sc.share_languages_endangered.iloc[0], color=ps.MUTED, lw=0.6,
              ls=(0, (2, 2)))
ax[1].text(3.9, 100 * sc.share_languages_endangered.iloc[0] + 1.5,
           "share of languages\nthat are endangered", fontsize=5.3, ha="right")
ax[0].set_xlabel("Cultural-loss weight κ")
ax[1].set_xlabel("Cultural-loss weight κ")
ax[0].set_ylabel("Culturally weighted excess loss,\nS+T − M (pp, language mean)")
ax[1].set_ylabel("Share of the excess borne by\nendangered-language communities (%)")
ax[0].legend(fontsize=5.2, loc="upper left")
for a, s in zip(ax, "AB"):
    a.grid(color=ps.GRID, lw=0.4)
    lab(a, s, x=-0.2)
fig.tight_layout()
ps.save(fig, F / "FigS9_scenario_lens")

# =============================================================== S10 composition
fig, ax = plt.subplots(1, 2, figsize=(ps.W2, 2.4))
comp = lm.groupby(["Area", "vitality"]).size().unstack().reindex(columns=LIVING_ORDER).fillna(0)
left = np.zeros(len(comp))
for v in LIVING_ORDER:
    ax[0].barh(comp.index, comp[v], left=left, color=ps.VIT_COLORS[v], label=v,
               edgecolor="white", lw=0.4)
    left += comp[v].to_numpy()
ax[0].set_xlabel("Living languages (n)")
ax[0].legend(fontsize=5.4, loc="lower right")
comp2 = lm.groupby(["annex_i", "vitality"]).size().unstack().reindex(columns=LIVING_ORDER)
comp2 = comp2.div(comp2.sum(1), axis=0) * 100
left = np.zeros(2)
for v in LIVING_ORDER:
    ax[1].barh(["Non-Annex I", "Annex I"], comp2[v].to_numpy(), left=left,
               color=ps.VIT_COLORS[v], edgecolor="white", lw=0.4)
    for i_, (l0, w0) in enumerate(zip(left, comp2[v].to_numpy())):
        if w0 > 6:
            ax[1].text(l0 + w0 / 2, i_, f"{w0:.0f}%", ha="center", va="center", fontsize=5.6,
                       color="white" if v != "Institutional" else ps.INK)
    left += comp2[v].to_numpy()
ax[1].set_xlabel("Share of living languages (%)")
for a, s in zip(ax, "AB"):
    lab(a, s, x=-0.2)
    a.grid(axis="x", color=ps.GRID, lw=0.4)
fig.tight_layout()
ps.save(fig, F / "FigS10_composition")
print("SI figures saved")
