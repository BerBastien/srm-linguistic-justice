"""18 - Main figures for the rainfall framing.

Fig1_drying_map           residual precipitation map + community-language home areas; zonal profile
Fig2_drying_robustness    60 members x 6 patterns; threshold curve
Fig3_sankey_drying        origin -> vitality -> rainfall outcome -> temperature outcome (LIC units)
Fig4_drying_where         rainfall relative to 1900 in three worlds; macro-areas; countries
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, LIVING_ORDER, P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
I, T, F = P("interim"), P("results"), P("figures")
DB = CFG["analysis"]["precip_deadband_pct"]
NONINST = ["Stable", "In trouble", "Dying"]
C_NON, C_INS = ps.VIT_COLORS["In trouble"], ps.VIT_COLORS["Institutional"]
DRY, WET = ps.P_OUT_COLORS["SRM drier than mitigation"], ps.P_OUT_COLORS["SRM wetter than mitigation"]
WORLD = {"H": "#eda100", "M": "#2a78d6", "S": "#eb6834"}     # validated (dataviz validator, light)
NB = ps.numbers(T / "numbers.json")

grid = pd.read_parquet(I / "grid_fields.parquet")
dl = pd.read_parquet(I / "dry_lang.parquet")
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.land_w_km2.notna() & ~pe.sign_language & pe.ISO.isin(dl.ISO)]
non_pts = pe[pe.vitality.isin(NONINST)]

# =================================================================== Figure 1
X, Y = ps.grid_mesh()
coast = gpd.read_file(P("ne_coast")).to_crs(ps.ROBIN)
fig = plt.figure(figsize=(ps.W2, 3.05))
ax = fig.add_axes([0.0, 0.0, 0.735, 1.0])
ps.draw_basemap(ax, P("ne_land"))
Z = ps.to_grid(grid, "RP_2100")
arid = ps.to_grid(grid.assign(a=(grid.PH_2100.abs() > 50).astype(float)), "a") > 0
ax.pcolormesh(X, Y, np.ma.masked_invalid(np.where(arid, 1.0, np.nan)), cmap=ListedColormap(["#d9d8d3"]),
              shading="flat", zorder=1, rasterized=True)
m = ax.pcolormesh(X, Y, np.ma.masked_invalid(np.where(arid, np.nan, Z)), cmap=ps.CMAP_P,
                  norm=TwoSlopeNorm(0, -30, 30), shading="flat", zorder=1, rasterized=True)
coast.plot(ax=ax, color="#6f6e69", lw=0.18, zorder=3)
xe, ye = ps.project(non_pts.rep_lon.to_numpy(), non_pts.rep_lat.to_numpy())
ax.scatter(xe, ye, s=0.8, c=ps.INK, lw=0, alpha=0.7, zorder=5, rasterized=True)
cax = ax.inset_axes([0.035, 0.12, 0.25, 0.035])
cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both", ticks=[-30, -15, 0, 15, 30])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.6, length=2, pad=1)
cb.ax.set_title("Rainfall, SAI minus mitigation, 2100\n(percentage points of 1900 rainfall)",
                fontsize=5.9, loc="left", pad=2, fontweight="normal")
ax.scatter([], [], s=5, c=ps.INK, label=f"Non-institutional language\nhome areas ({len(non_pts):,})")
ax.fill_between([], [], color="#d9d8d3", label="Hyper-arid (masked)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.30), fontsize=5.6, handletextpad=0.3, handlelength=1.0)
ax.text(0.0, 0.99, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

axZ = fig.add_axes([0.775, 0.10, 0.095, 0.84])
axD = fig.add_axes([0.885, 0.10, 0.11, 0.84], sharey=axZ)
lat_bins = np.arange(-56, 84, 4)
g = grid[grid.PH_2100.abs() <= 50].copy()       # same hyper-arid mask as the map
g["band"] = pd.cut(g.lat, lat_bins)
zm = g.groupby("band", observed=True).apply(lambda d: np.average(d.RP_2100, weights=d.area_km2))
mid = np.array([b.mid for b in zm.index])
axZ.axvline(0, color=ps.AXIS, lw=0.6)
axZ.fill_betweenx(mid, 0, zm.values, where=zm.values < 0, color="#dfc27d", lw=0, interpolate=True)
axZ.fill_betweenx(mid, 0, zm.values, where=zm.values > 0, color="#80cdc1", lw=0, interpolate=True)
axZ.plot(zm.values, mid, color=ps.INK, lw=0.9)
axZ.set_xlabel("Zonal land mean\n(pp, excl. hyper-arid)", fontsize=6.0)
axZ.set_ylabel("Latitude (°)", labelpad=1)
axZ.set_ylim(-56, 82)
axZ.text(-0.62, 1.0, "B", transform=axZ.transAxes, fontsize=9, fontweight="bold", va="top")
yy = np.linspace(-56, 82, 300)
for lab, sel, col in [("Non-institutional", dl.noninst, C_NON), ("Institutional", ~dl.noninst, C_INS)]:
    x = dl[sel].lat_mean.dropna().to_numpy()
    k = gaussian_kde(x, bw_method=3.0 / x.std())
    axD.plot(k(yy) * 100, yy, color=col, lw=1.1, label=f"{lab}\n(n = {len(x):,})")
axD.set_xlabel("Languages\n(density, % per °)", fontsize=6.2)
axD.tick_params(labelleft=False)
axD.set_xlim(0, None)
axD.legend(loc="lower right", fontsize=5.3, handlelength=1.0, bbox_to_anchor=(1.06, 0.0))
for a in (axZ, axD):
    a.grid(axis="y", color=ps.GRID, lw=0.4)
    for y0 in (-23.44, 23.44):
        a.axhline(y0, color=ps.MUTED, lw=0.4, ls=(0, (2, 2)))
ps.save(fig, F / "Fig1_drying_map")

# =================================================================== Figure 2
bm = pd.read_csv(T / "dry_by_member.csv")
thr = pd.read_csv(T / "dry_thresholds.csv")
rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 3, figsize=(ps.W2, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.42})
for a, col, ylab, tag in [(axs[0], "share_drier_noninst", "Non-institutional languages\n>2 pp drier than mitigation (%)", "A"),
                          (axs[1], "median_RP_noninst", "Median rainfall difference,\nnon-institutional languages (pp)", "B")]:
    for k in range(1, 7):
        d = bm[bm.pattern == k]
        v = d[col] * (100 if col.startswith("share") else 1)
        a.scatter(k + rng.uniform(-0.17, 0.17, len(d)), v, s=9, color=C_NON, alpha=0.75, lw=0.4, ec="white", zorder=3)
        a.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color=ps.INK, lw=1.4, zorder=4)
        if col.startswith("share"):
            vi = d.share_drier_inst.mean() * 100
            a.plot([k - 0.28, k + 0.28], [vi] * 2, color=C_INS, lw=1.4, ls=(0, (2, 1)), zorder=4)
    a.set_xticks(range(1, 7))
    a.set_xlabel("SAI response pattern")
    a.set_ylabel(ylab)
    a.grid(axis="y", color=ps.GRID, lw=0.4)
    ps.panel_label(a, tag, x=-0.2)
axs[0].axhline(50, color=ps.MUTED, lw=0.6, ls=(0, (2, 2)))
axs[0].set_ylim(30, 85)
axs[0].text(0.6, 82, f"Majority drier in {NB['dry.member.noninst.n_members_majority_drier']} of 60 members",
            fontsize=5.6, color=ps.INK2, va="top")
axs[0].plot([], [], color=ps.INK, lw=1.4, label="Pattern mean")
axs[0].plot([], [], color=C_INS, lw=1.4, ls=(0, (2, 1)), label="Institutional languages")
axs[0].legend(loc="lower right", fontsize=5.4, handlelength=1.6)
axs[1].axhline(0, color=ps.AXIS, lw=0.8)
axs[1].set_ylim(-12, 1.5)
axs[1].text(0.6, 1.2, f"Below zero in {NB['dry.member.noninst.n_members_median_negative']} of 60 members",
            fontsize=5.6, color=ps.INK2, va="top")
a = axs[2]
a.fill_between(thr.threshold_pp, thr.member_p17_noninst * 100, thr.member_p83_noninst * 100, color=C_NON,
               alpha=0.18, lw=0)
a.plot(thr.threshold_pp, thr.share_noninst * 100, color=C_NON, lw=1.6)
a.plot(thr.threshold_pp, thr.share_inst * 100, color=C_INS, lw=1.3, ls=(0, (2, 1)))
for th in (2, 5, 10):
    v = thr.loc[thr.threshold_pp == th, "share_noninst"].iloc[0] * 100
    a.plot([th], [v], "o", ms=3.2, color=C_NON, mec="white", mew=0.5, zorder=5)
    a.annotate(f"{v:.0f}%", (th, v), xytext=(4, 3) if th < 10 else (-3, -10), textcoords="offset points",
               fontsize=5.8, color=ps.INK, ha="left" if th < 10 else "right")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([], [], color=C_NON, lw=1.6, label="Non-institutional"),
                  Patch(color=C_NON, alpha=0.18, lw=0, label="  members p17–p83"),
                  Line2D([], [], color=C_INS, lw=1.3, ls=(0, (2, 1)), label="Institutional")],
         loc="upper right", fontsize=5.4, handlelength=1.6)
a.set_xlim(0, 15.5)
a.set_ylim(0, 85)
a.set_xlabel("Drier than mitigation by more than (pp)")
a.set_ylabel("Languages (%)")
a.grid(color=ps.GRID, lw=0.4)
ps.panel_label(a, "C", x=-0.2)
ps.save(fig, F / "Fig2_drying_robustness")

# =================================================================== Figure 3 (Sankey)
flows = pd.read_csv(T / "sankey_flows_lic.csv")
ORIG = ["Indigenous to country", "Non-indigenous (colonial or immigrant origin)"]
TOUT = ["SRM cooler than mitigation", "Similar (within 0.1 C)", "SRM warmer than mitigation"]
POUT = ["SRM drier than mitigation", "Similar (within 2 pp)", "SRM wetter than mitigation"]
stages = [("origin", ORIG), ("vitality_LIC", LIVING_ORDER), ("P_out", POUT), ("T_out", TOUT)]
W = "n_lic"
N = flows[W].sum()
GAP = 0.035 * N
XW = 0.06


def node_layout(col, cats):
    tot = flows.groupby(col)[W].sum().reindex(cats).fillna(0)
    y = (tot.sum() + GAP * (len(cats) - 1)) / 2
    pos = {}
    for c in cats:
        pos[c] = [y, y - tot[c]]
        y -= tot[c] + GAP
    return pos, tot


def band(ax, x0, x1, a0, b0, a1, b1, color, alpha=0.58):
    t = np.linspace(0, 1, 60)
    s = 3 * t ** 2 - 2 * t ** 3
    ax.fill_between(x0 + (x1 - x0) * t, b0 + (b1 - b0) * s, a0 + (a1 - a0) * s, facecolor=color, alpha=alpha,
                    edgecolor="white", lw=0.25)


layouts = [node_layout(c, cats) for c, cats in stages]
fig, ax = plt.subplots(figsize=(ps.W2, 3.55))
xs = [0, 1, 2, 3]
for si in range(3):
    (ca, cats_a), (cbb, cats_b) = stages[si], stages[si + 1]
    pos_a, pos_b = layouts[si][0], layouts[si + 1][0]
    f = flows.groupby([ca, cbb, "vitality_LIC"])[W].sum()
    oc = {a_: pos_a[a_][0] for a_ in cats_a}
    ic = {b_: pos_b[b_][0] for b_ in cats_b}
    so, sn = {}, {}
    for a_ in cats_a:
        for b_ in cats_b:
            for v in LIVING_ORDER:
                w = f.get((a_, b_, v), 0)
                if w > 0:
                    so[(a_, b_, v)] = (oc[a_], oc[a_] - w)
                    oc[a_] -= w
    for b_ in cats_b:
        for a_ in cats_a:
            for v in LIVING_ORDER:
                w = f.get((a_, b_, v), 0)
                if w > 0:
                    sn[(a_, b_, v)] = (ic[b_], ic[b_] - w)
                    ic[b_] -= w
    for key, (a0, b0) in so.items():
        a1, b1 = sn[key]
        band(ax, xs[si] + XW / 2, xs[si + 1] - XW / 2, a0, b0, a1, b1, ps.VIT_COLORS[key[2]])
node_col = {**{o: "#52514e" for o in ORIG}, **ps.VIT_COLORS, **ps.T_OUT_COLORS, **ps.P_OUT_COLORS}
short = {ORIG[0]: "Indigenous\nto country", ORIG[1]: "Non-indigenous\n(colonial/immigrant)",
         TOUT[0]: "SAI cooler", TOUT[1]: "Similar\n(±0.1 °C)", TOUT[2]: "SAI warmer",
         POUT[0]: "SAI drier", POUT[1]: "Similar\n(±2 pp)", POUT[2]: "SAI wetter",
         "Institutional": "Institutional", "Stable": "Stable", "In trouble": "In trouble", "Dying": "Dying"}
for si, ((col, cats), (pos, tot)) in enumerate(zip(stages, layouts)):
    for c in cats:
        top, bot = pos[c]
        if tot[c] <= 0:
            continue
        ax.add_patch(plt.Rectangle((xs[si] - XW / 2, bot), XW, top - bot, color=node_col[c], lw=0, zorder=3))
        lab = f"{short.get(c, c)}\n{tot[c]:,.0f} ({100 * tot[c] / N:.0f}%)"
        if si == 0:
            ax.text(xs[si] - XW / 2 - 0.03, (top + bot) / 2, lab, ha="right", va="center", fontsize=6)
        elif si == 3:
            ax.text(xs[si] + XW / 2 + 0.03, (top + bot) / 2, lab, ha="left", va="center", fontsize=6)
        else:
            ax.text(xs[si] + XW / 2 + 0.02, (top + bot) / 2, lab, ha="left", va="center", fontsize=5.6, zorder=4,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))
heads = ["Origin of the language\nin the country", "Vitality\n(community level;\nour EGIDS grouping)",
         "Rainfall, 2100\nSAI vs. mitigation", "Temperature, 2100\nSAI vs. mitigation"]
ytop = max(p[0] for lay, _ in layouts for p in lay.values())
for x, h in zip(xs, heads):
    ax.text(x, ytop + 0.06 * N, h, ha="center", va="bottom", fontsize=6.4, fontweight="bold")
non_lic = flows[flows.vitality_LIC.isin(NONINST)]
pn = non_lic.groupby("P_out")[W].sum() / non_lic[W].sum() * 100
dr = non_lic[non_lic.P_out == POUT[0]]
cool_of_dry = dr[dr.T_out == TOUT[0]][W].sum() / dr[W].sum() * 100
ind_of_dry = dr[dr.origin == ORIG[0]][W].sum() / dr[W].sum() * 100
from srmlang.common import record  # noqa: E402
record("dry.sankey.n_lic", float(N))
record("dry.sankey.noninst_lic.n", float(non_lic[W].sum()))
record("dry.sankey.noninst_lic.share_drier", float(pn[POUT[0]] / 100))
record("dry.sankey.noninst_lic.share_wetter", float(pn[POUT[2]] / 100))
record("dry.sankey.noninst_lic.drier_share_indigenous", float(ind_of_dry / 100))
record("dry.sankey.noninst_lic.drier_share_cooler", float(cool_of_dry / 100))
for o in ORIG:
    sub = flows[flows.origin == o]
    record(f"dry.sankey.{o.split()[0]}.share_drier", float(sub[sub.P_out == POUT[0]][W].sum() / sub[W].sum()))
ax.text(0.0, -0.035,
        f"Non-institutional communities: {pn[POUT[0]]:.0f}% drier, {pn[POUT[2]]:.0f}% wetter. "
        f"Of those left drier, {ind_of_dry:.0f}% are indigenous to their country and {cool_of_dry:.0f}% are also cooler.",
        transform=ax.transAxes, fontsize=5.8, color=ps.INK2, va="top")
ax.set_xlim(-0.75, 3.75)
ax.set_axis_off()
ps.save(fig, F / "Fig3_sankey_drying")

# =================================================================== Figure 4
ab = pd.read_csv(T / "dry_absolute_by_member.csv")
z = np.load(I / "dry_member.npz", allow_pickle=True)
nonm = dl.noninst.to_numpy()
sH = (z["PH_2100"][nonm] < 0).mean(0)
area = pd.read_csv(T / "dry_by_area.csv").set_index("Area")
cty = pd.read_csv(T / "dry_by_country.csv").set_index("Country_Code").head(10)
lx = pd.read_parquet(I / "lang_exposure.parquet")
cname = lx.drop_duplicates("Country_Code").set_index("Country_Code").Country_Name
nice = {"CD": "DR Congo", "PG": "Papua New Guinea"}
fig, axs = plt.subplots(1, 3, figsize=(ps.W2, 2.45), gridspec_kw={"width_ratios": [1.05, 0.9, 1.1], "wspace": 0.95})
a = axs[0]
vals = [("No SAI", "H", NB["dry.noninst.share_below1900.H"], sH),
        ("Mitigation", "M", NB["dry.noninst.share_below1900.M"], ab.share_below1900_M.to_numpy()),
        ("SAI", "S", NB["dry.noninst.share_below1900.S"], ab.share_below1900_S.to_numpy())]
for i, (lab, key, v, mem) in enumerate(vals):
    lo, hi = np.percentile(mem, [17, 83]) * 100
    a.bar(i, v * 100, width=0.62, color=WORLD[key], lw=0)
    a.plot([i, i], [lo, hi], color=ps.INK, lw=0.9)
    a.text(i, max(hi, v * 100) + 2, f"{v * 100:.0f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
a.set_xticks(range(3))
a.set_xticklabels([v[0] for v in vals], fontsize=5.7)
a.set_ylabel("Non-institutional languages with\nless rain in 2100 than in 1900 (%)")
a.set_ylim(0, 72)
a.grid(axis="y", color=ps.GRID, lw=0.4)
ps.panel_label(a, "A", x=-0.28)

a = axs[1]
ar = area.sort_values("share_drier")
yy_ = np.arange(len(ar))
a.barh(yy_, ar.share_drier * 100, color=DRY, height=0.62, lw=0)
for y_, (nm, r) in zip(yy_, ar.iterrows()):
    a.text(r.share_drier * 100 + 1.5, y_, f"{int(r.n_drier):,} of {int(r.n):,}", va="center", fontsize=5.7)
a.set_yticks(yy_)
a.set_yticklabels(ar.index)
a.set_xlim(0, 125)
a.set_xticks([0, 25, 50, 75, 100])
a.set_xlabel(">2 pp drier than mitigation (%)")
a.grid(axis="x", color=ps.GRID, lw=0.4)
ps.panel_label(a, "B", x=-0.26)
a = axs[2]
cc = cty.iloc[::-1]
yy_ = np.arange(len(cc))
a.barh(yy_, cc.n, color="#e8e6df", height=0.66, lw=0, label="All non-institutional")
a.barh(yy_, cc.n_drier, color=DRY, height=0.66, lw=0, label=">2 pp drier")
for y_, (code, r) in zip(yy_, cc.iterrows()):
    a.text(r.n + 12, y_, f"{int(r.n_drier)}", va="center", fontsize=5.7)
a.set_yticks(yy_)
a.set_yticklabels([nice.get(c, cname.get(c, c)) for c in cc.index], fontsize=5.9)
a.set_xlabel("Non-institutional languages (number)")
a.grid(axis="x", color=ps.GRID, lw=0.4)
a.legend(loc="lower right", fontsize=5.4, handlelength=1.0, bbox_to_anchor=(1.03, 0.0))
ps.panel_label(a, "C", x=-0.45)
ps.save(fig, F / "Fig4_drying_where")
print("figures saved")
