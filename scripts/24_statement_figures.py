"""24 - Figures for the final statements.

Fig_against_what   Shares of land, people, economy and linguistic heritage left drier, wetter and
                   warmer under SAI than under iso-temperature mitigation (2100), with member ranges.
Fig_tree_rain_heat The human language tree coloured by SAI's rainfall (left) and temperature (right)
                   outcome, ensemble-mean climate. Branch width ~ sqrt(number of languages).
Fig_tree_rain_range The rainfall tree in the least-dry member, the ensemble-mean climate and the
                   driest member (same layout).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
I, T, F = P("interim"), P("results"), P("figures")
NB = ps.numbers(T / "numbers.json")
C_LANG, C_BASE = ps.VIT_COLORS["In trouble"], "#898781"

# =================================================================== Fig_against_what
tab = pd.read_csv(T / "baselines_rain_heat.csv")
order = ["land", "populated land", "people (L1 speakers)", "economy (GDP)", "institutional languages",
         "non-institutional languages", "linguistic diversity hotspots", "language lineages"]
labels = {"land": "Land (area)", "populated land": "Populated land", "people (L1 speakers)": "People",
          "economy (GDP)": "Economy (GDP)", "institutional languages": "Institutional languages",
          "non-institutional languages": "Non-institutional languages", "linguistic diversity hotspots": "Linguistic diversity hotspots",
          "language lineages": "Language lineages"}
lang_units = {"non-institutional languages", "linguistic diversity hotspots", "language lineages",
              "institutional languages"}
fig, axs = plt.subplots(1, 3, figsize=(ps.W2, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, oc, title, xmax in zip(axs, ["drier", "wetter", "warmer"],
                               ["Less rain than mitigation (>2 pp)", "More rain than mitigation (>2 pp)",
                                "Warmer than mitigation (>0.1 °C)"], [93, 55, 45]):
    d = tab[tab.outcome == oc].set_index("unit").loc[order]
    land = d.loc["land", "ensemble_mean_climate"] * 100
    ax.axvspan(0, land, color="#f1f0ec", lw=0, zorder=0)
    ax.axvline(land, color=C_BASE, lw=0.7, ls=(0, (2, 2)), zorder=1)
    for yi, (u, r) in enumerate(d.iterrows()):
        col = C_LANG if u in lang_units else C_BASE
        if u == "institutional languages":
            col = ps.VIT_COLORS["Institutional"]
        ax.plot([r.member_p17 * 100, r.member_p83 * 100], [yi, yi], color=col, lw=2.2, alpha=0.35,
                solid_capstyle="round")
        ax.plot(r.ensemble_mean_climate * 100, yi, "o", color=col, ms=4.6, mec="white", mew=0.5, zorder=4)
        ax.text(xmax * 0.995, yi, f"{r.ensemble_mean_climate * 100:.0f}%", ha="right", va="center", fontsize=5.8,
                color=ps.INK if u in lang_units else ps.INK2,
                fontweight="bold" if u in ("non-institutional languages", "language lineages") else "normal")
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=6.6, loc="left", fontweight="bold")
    ax.grid(axis="x", color=ps.GRID, lw=0.4)
    ax.set_xlabel("Share in 2100 (%)")
axs[0].set_yticks(range(len(order)))
axs[0].set_yticklabels([labels[u] for u in order])
axs[0].invert_yaxis()
axs[0].axhline(3.5, color=ps.AXIS, lw=0.6)
for ax, tag in zip(axs, "abc"):
    ax.text(-0.62 if tag == "a" else -0.04, 1.12, tag.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
fig.text(0.0, -0.06, "Dots: ensemble-mean climate. Bars: p17–p83 across 60 members. Dashed line and shading: "
         "land. Non-institutional languages and lineages are each counted once (fraction of homeland).",
         fontsize=5.6, color=ps.INK2)
ps.save(fig, F / "Fig_against_what")

# =================================================================== the language lineages (Glottolog 5.3)
# Each lineage (a family, or an isolate) is an independent tree. In Fig_tree_* they are drawn joined at a
# common base for display only; Fig_forest draws each lineage as its own tree on the ground.
cw = pd.read_parquet(I / "glottolog_crosswalk.parquet").set_index("ISO")
GN = pd.read_csv(P("results") / "glottolog_lineage_names.csv").set_index("glottocode").name.to_dict()
DISPLAY = {"Otomanguean": "Oto-Manguean"}
hl = pd.read_parquet(I / "heritage_lang.parquet").set_index("ISO")
hl = hl[hl.f_dry.notna() & hl.lineage.notna()]
MAXD = 4                                   # lineage + 3 sub-levels, then the languages


def path_of(iso):
    lin = hl.lineage[iso]
    parts = [q for q in str(cw.g_classification.get(iso) or "").split("/") if q]
    if not parts:                          # isolate: its own lineage, a single branch
        return (lin,)
    return tuple(parts[:MAXD])


def lname(gc):
    nm = GN.get(gc, hl.lineage_name[hl.lineage == gc].iloc[0] if (hl.lineage == gc).any() else gc)
    return DISPLAY.get(nm, nm)


paths = {i: path_of(i) for i in hl.index}
fam_of = {i: p[0] for i, p in paths.items()}
fam_dry = hl.f_dry.groupby(pd.Series(fam_of)).mean()
fam_n = pd.Series(fam_of).value_counts()
# lineage order around the crown: least dry on the left, driest on the right
fam_order = fam_dry.sort_values().index.tolist()
fam_rank = {f: k for k, f in enumerate(fam_order)}
leaves = sorted(hl.index, key=lambda i: (fam_rank[fam_of[i]], paths[i], hl.f_dry[i]))
N = len(leaves)
ang = {iso: np.radians(162 - 144 * k / (N - 1)) for k, iso in enumerate(leaves)}   # 162 deg (left) -> 18 deg
# nodes: every path prefix
node_leaves = {}
for iso in leaves:
    p = paths[iso]
    for k in range(1, len(p) + 1):
        node_leaves.setdefault(p[:k], []).append(iso)
CROWN = np.array([0.0, 0.75])
RAD = [0.0, 0.55, 1.0, 1.36, 1.66]          # radius by depth (depth 1 = lineage)
SX = 1.30                                    # crown wider than tall
rng0 = np.random.default_rng(11)
tip_members = {}
for i in leaves:
    tip_members.setdefault(paths[i], []).append(i)
nmax_tip = max(len(v) for v in tip_members.values())
JIT = {}
for tip, members in tip_members.items():                      # round clumps of foliage at each branch tip
    rho = min(0.07 + 0.42 * np.sqrt(len(members) / nmax_tip), 0.48)
    for i in members:
        u, th = rng0.uniform(), rng0.uniform(0, 2 * np.pi)
        JIT[i] = (rho, np.sqrt(u) * rho * np.cos(th), np.sqrt(u) * rho * np.sin(th))
LABELLED = [f for f in fam_order if fam_n[f] >= 130 or lname(f) in ("Sepik", "Nuclear Torricelli", "Oto-Manguean",
                                                                     "Dravidian")]


def pos(angle, r):
    return CROWN + r * np.array([SX * np.cos(angle), np.sin(angle)])


def node_angle(node):
    return float(np.mean([ang[i] for i in node_leaves[node]]))


def draw_tree(ax, value, cmap, vmin, vmax, label_fams=True, title=None, title_top=False):
    ax.set_aspect("equal")
    ax.set_axis_off()
    nm = lambda v: cmap((np.clip(v, vmin, vmax) - vmin) / (vmax - vmin))  # noqa: E731
    allv = float(np.mean([value[i] for i in leaves]))
    ax.fill([-0.12, -0.05, 0.05, 0.12], [0.0, CROWN[1] + 0.02, CROWN[1] + 0.02, 0.0], color=nm(allv), lw=0, zorder=1)
    ax.plot([-0.8, 0.8], [0, 0], color="#d6d4cc", lw=0.9, zorder=0)
    nmax = max(len(v) for k, v in node_leaves.items() if len(k) == 1)
    for node, lv in sorted(node_leaves.items(), key=lambda kv: -len(kv[1])):
        d = len(node)
        a = node_angle(node)
        if d == 1:
            p0, ctrl = CROWN, CROWN + np.array([0.0, 0.32])
            r0 = 0.0
        else:
            a0 = node_angle(node[:-1])
            r0 = RAD[d - 1]
            p0 = pos(a0, r0)
            ctrl = pos(0.6 * a + 0.4 * a0, (r0 + RAD[d]) / 2)
        p1 = pos(a, RAD[d])
        t_ = np.linspace(0, 1, 28)[:, None]
        curve = (1 - t_) ** 2 * p0 + 2 * (1 - t_) * t_ * ctrl + t_ ** 2 * p1
        lw = 0.2 + 4.6 * np.sqrt(len(lv) / nmax)
        ax.plot(curve[:, 0], curve[:, 1], color=nm(np.mean([value[i] for i in lv])), lw=lw,
                solid_capstyle="round", zorder=2 + 0.01 * d, rasterized=True)
    # foliage: each language is a leaf clustered around the tip of its deepest branch
    xy, cols, sz = [], [], []
    for i in leaves:
        tip_node = paths[i]
        rho, dx, dy = JIT[i]
        a = node_angle(tip_node)
        cen = pos(a, RAD[len(tip_node)] + 0.85 * rho)
        xy.append(cen + np.array([dx, dy]))
        cols.append(nm(value[i]))
        sz.append(2.2 if value[i] < 0.5 else 1.4)
    xy = np.array(xy)
    ax.scatter(xy[:, 0], xy[:, 1], s=sz, c=cols, lw=0, zorder=3, alpha=0.9, rasterized=True)
    if label_fams:
        fams = LABELLED
        items = []
        for fam in fams:
            a = node_angle((fam,))
            x, y = pos(a, RAD[4] + 0.95)
            items.append([fam, a, x, y])
        for side in (-1, 1):
            sel = sorted([it for it in items if np.sign(it[2]) == side or (side == 1 and it[2] == 0)],
                         key=lambda it: it[3])
            for k in range(1, len(sel)):
                if sel[k][3] - sel[k - 1][3] < 0.17:
                    sel[k][3] = sel[k - 1][3] + 0.17
        fam_arr = np.array([fam_of[i] for i in leaves])
        for fam, a, x, y in items:
            pts = xy[fam_arr == fam]
            k = np.argmin(((pts - np.array([x, y])) ** 2).sum(1))
            x0, y0 = pts[k]
            ax.plot([x0, x], [y0, y], color="#9d9b93", lw=0.4, zorder=4)
            ax.text(x + (0.03 if x >= 0 else -0.03), y, f"{lname(fam)} ({fam_n[fam]:,})", ha="left" if x >= 0 else "right",
                    va="center", fontsize=5.3, color=ps.INK2)
    ax.set_xlim(-3.9, 3.9)
    ax.set_ylim(-0.1, 3.05)
    if title:
        ax.text(0, 3.25 if title_top else -0.08, title, ha="center", va="top" if not title_top else "top",
                fontsize=6.2, fontweight="bold")


RAIN = LinearSegmentedColormap.from_list("rain", ["#2f7d3a", "#7fa35a", "#c9b77a", "#b0793a", "#7a4310"])
HEAT = LinearSegmentedColormap.from_list("heat", ["#1c5cab", "#86b6ef", "#e1e0d9", "#f2a39d", "#c0392b"])
fdry = hl.f_dry.to_dict()
net_heat = (hl.f_warmer - hl.f_cooler).to_dict()                      # -1 cooler ... +1 warmer


def cbar(fig, rect, cmap, ticks, labels):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.set_xticklabels(labels)
    cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
    cb.outline.set_visible(False)
    return cb


# ---- heat and rain pair (no labels)
fig, axs = plt.subplots(1, 2, figsize=(ps.W2, 2.55), gridspec_kw={"wspace": 0.0})
draw_tree(axs[0], {k: (v + 1) / 2 for k, v in net_heat.items()}, HEAT, 0, 1, label_fams=False,
          title="Heat: cooler or warmer than mitigation")
draw_tree(axs[1], fdry, RAIN, 0, 1, label_fams=False, title="Rain: share of homeland with less rain")
for ax in axs:
    ax.set_xlim(-2.75, 2.75)
for ax, tag in zip(axs, "ab"):
    ax.text(-2.7, 3.2, tag.upper(), fontsize=9, fontweight="bold", va="top")
cbar(fig, [0.06, 0.93, 0.17, 0.022], HEAT, [0, 0.5, 1], ["cooler", "same", "warmer"])
cbar(fig, [0.56, 0.93, 0.17, 0.022], RAIN, [0, 0.5, 1], ["none", "half", "all"])
fig.text(0.01, 0.0, "Lineages are drawn joined at a common base for display only; this does not imply that they are related.", fontsize=5.2, color=ps.INK2)
ps.save(fig, F / "Fig_tree_rain_heat")

# ---- triptych: least dry member, ensemble mean, driest member
zz = np.load(I / "heritage_member_fdry.npz", allow_pickle=True)
fm = pd.DataFrame(zz["f"], index=zz["ISO"]).loc[hl.index]
fam_s = hl.lineage                       # lineages as in script 21 (each isolate its own lineage)
lin_share = fm.groupby(fam_s.reindex(fm.index).to_numpy()).mean().gt(0.5).mean(0)   # share of lineages mostly drier
m_lo, m_hi = int(lin_share.idxmin()), int(lin_share.idxmax())
fig = plt.figure(figsize=(ps.W2, 5.4))
axM = fig.add_axes([0.0, 0.34, 1.0, 0.66])
fam_mean = hl.f_dry.groupby(fam_s.reindex(hl.index).to_numpy()).mean()
draw_tree(axM, fdry, RAIN, 0, 1, label_fams=True)
axM.text(-3.85, 3.2, "A", fontsize=9, fontweight="bold", va="top")
axM.text(-3.85, -0.06, "Lineages are drawn joined at a common base for display only; this does not imply that they are related.", fontsize=5.2, color=ps.INK2, va="top")
cbar(fig, [0.74, 0.37, 0.2, 0.012], RAIN, [0, 0.5, 1], ["none", "half", "all"]).ax.set_title(
    "Share of homeland with less rain", fontsize=5.6, pad=2)
for k, (vals, ttl) in enumerate([(fm[m_lo].to_dict(), f"Least-dry member: {100 * lin_share[m_lo]:.0f}% of lineages"),
                                 (fdry, f"Ensemble mean: {100 * (fam_mean > 0.5).mean():.0f}%"),
                                 (fm[m_hi].to_dict(), f"Driest member: {100 * lin_share[m_hi]:.0f}% of lineages")]):
    ax = fig.add_axes([k / 3, 0.0, 1 / 3, 0.31])
    draw_tree(ax, vals, RAIN, 0, 1, label_fams=False, title=ttl)
    ax.set_xlim(-2.75, 2.75)
    if k == 0:
        ax.text(-2.7, 3.2, "B", fontsize=9, fontweight="bold", va="top")
ps.save(fig, F / "Fig_tree_rain_range")
from srmlang.common import record  # noqa: E402
record("tree.member_least_dry", m_lo)
record("tree.member_driest", m_hi)
record("tree.share_families_mostly_drier.least_dry", float(lin_share[m_lo]))
record("tree.share_families_mostly_drier.driest", float(lin_share[m_hi]))
record("tree.share_families_mostly_drier.ensemble_mean", float((fam_mean > 0.5).mean()))
print("saved")

# =================================================================== the forest (no common base)
# Science version (v5): Fig. 3 is the forest. Each lineage is its own tree standing on the ground: families
# branch into up to three levels of subgroups and end in one dot per language; isolates are single stems.
# Lineages are ordered from the least dry (top left) to the driest (bottom right) and wrapped into rows.
# Tree width ~ sqrt(languages), height ~ log(languages). The common-base tree (Fig_tree_rain_range) and
# the heat-rain pair move to the Supplementary Materials; FigS_forest_temperature is the forest for heat.


def draw_forest(fig, value, cmap, y0=0.0, y1=1.0, n_rows=5, label_min=130):
    nm = lambda v: cmap(np.clip(v, 0, 1))  # noqa: E731
    lin_list = fam_order
    width = {f: 0.9 + 1.25 * np.sqrt(fam_n[f]) for f in lin_list}
    total = sum(width.values()) + 0.6 * len(lin_list)
    per_row = total / n_rows
    rows_, cur, acc_ = [], [], 0.0
    for f in lin_list:
        if acc_ + width[f] > per_row * 1.02 and cur:
            rows_.append(cur)
            cur, acc_ = [], 0.0
        cur.append(f)
        acc_ += width[f] + 0.6
    rows_.append(cur)
    row_h = (y1 - y0) / len(rows_)
    th = lambda n: 1.2 + 2.2 * np.log10(n + 1)  # noqa: E731  tree height grows with log(languages)
    hmax_all = th(fam_n.max())
    rng_f = np.random.default_rng(5)
    axes = []
    for r_i, row in enumerate(rows_):
        ax = fig.add_axes([0.01, y1 - (r_i + 1) * row_h + 0.006, 0.98, row_h - 0.01])
        ax.set_axis_off()
        axes.append(ax)
        span = sum(width[f] + 0.6 for f in row)
        x0 = 0.0
        last_end = [-1e9]
        for f in row:
            w_ = width[f]
            lv = [i for i in leaves if fam_of[i] == f]
            h_tree = th(len(lv))
            levels = [0.0, 0.42, 0.62, 0.78, 0.9]
            xs_ = {i: x0 + w_ * (k + 0.5) / len(lv) for k, i in enumerate(lv)}
            nodes = {}
            for i in lv:
                pth = paths[i]
                for k in range(1, len(pth) + 1):
                    nodes.setdefault(pth[:k], []).append(i)
            for node, ll in sorted(nodes.items(), key=lambda kv: len(kv[0])):
                d = len(node)
                xn = float(np.mean([xs_[i] for i in ll]))
                yn = levels[d] * h_tree
                if d == 1:
                    xp, yp = xn, 0.0
                else:
                    pl = nodes[node[:-1]]
                    xp, yp = float(np.mean([xs_[i] for i in pl])), levels[d - 1] * h_tree
                lw = 0.25 + 2.6 * np.sqrt(len(ll) / fam_n.max())
                ym = yp + 0.55 * (yn - yp)        # rectangular (dendrogram) elbow
                ax.plot([xp, xp, xn, xn], [yp, ym, ym, yn], color=nm(np.mean([value[i] for i in ll])),
                        lw=lw, solid_capstyle="butt", solid_joinstyle="miter", rasterized=True)
            ytop = levels[MAXD] * h_tree
            for tip in {paths[i] for i in lv}:   # shallow tips grow up to the crown, so all foliage sits on top
                yt = levels[len(tip)] * h_tree
                if yt < ytop - 1e-9:
                    xt = float(np.mean([xs_[j] for j in nodes[tip]]))
                    ax.plot([xt, xt], [yt, ytop], color=nm(np.mean([value[j] for j in nodes[tip]])),
                            lw=0.25 + 2.6 * np.sqrt(len(nodes[tip]) / fam_n.max()), rasterized=True)
            for i in lv:                       # foliage: one dot per language above the tip of its branch
                tip = paths[i]
                xt = float(np.mean([xs_[j] for j in nodes[tip]]))
                ax.scatter(xs_[i] * 0.5 + xt * 0.5, ytop + 0.12 + 0.3 * rng_f.uniform(), s=1.1, color=nm(value[i]),
                           lw=0, rasterized=True)
            if fam_n[f] >= label_min or lname(f) in ("Sepik", "Nuclear Torricelli", "Oto-Manguean", "Dravidian"):
                lab = f"{lname(f)} ({fam_n[f]:,})"
                half = len(lab) * 0.55 * 5.2 / 72 * (per_row * 1.04 + 0.5) / (0.98 * fig.get_figwidth()) / 2 + 1.0
                ylab = -0.3 if x0 + w_ / 2 - half > last_end[0] else -0.95
                ax.text(x0 + w_ / 2, ylab, lab, ha="center", va="top", fontsize=5.2, color=ps.INK2)
                last_end[0] = x0 + w_ / 2 + half if ylab == -0.3 else last_end[0]
            x0 += w_ + 0.6
        ax.plot([-0.3, span], [0, 0], color="#bdbbb2", lw=0.6)
        ax.set_xlim(-0.5, per_row * 1.04)
        ax.set_ylim(-1.6, hmax_all + 0.6)
    return axes


# ---- Fig. 3: forest (ensemble-mean climate) and, below, the share of lineages mostly drier in each member
pattern = np.arange(fm.shape[1]) % 6 + 1          # CLIMRISK cycles the six patterns across members
fig = plt.figure(figsize=(ps.W2, 7.0))
axes = draw_forest(fig, fdry, RAIN, y0=0.17, y1=0.965)
axes[0].text(-0.4, axes[0].get_ylim()[1], "A", fontsize=9, fontweight="bold", va="top")
fig.text(0.04, 0.985, "Each language lineage drawn as its own tree (Glottolog 5.3, 278 lineages); "
         "least dry top left, driest bottom right", fontsize=5.8, color=ps.INK2, va="top")
cbar(fig, [0.70, 0.972, 0.25, 0.008], RAIN, [0, 0.5, 1], ["none", "half", "all"]).ax.set_title(
    "Share of homeland with less rain", fontsize=5.8, pad=2)
axB = fig.add_axes([0.13, 0.045, 0.55, 0.095])
jit = np.random.default_rng(3).uniform(-0.25, 0.25, len(pattern))
axB.scatter(100 * lin_share.to_numpy(), pattern + jit, s=7, color="#7a4310", alpha=0.75, lw=0)
axB.axvline(100 * (fam_mean > 0.5).mean(), color=ps.INK, lw=1.0)
axB.axvline(50, color=ps.INK2, lw=0.6, ls=":")
axB.text(100 * (fam_mean > 0.5).mean() + 0.6, 6.9, f"ensemble-mean climate: {100 * (fam_mean > 0.5).mean():.0f}%",
         fontsize=5.6, va="top")
axB.set_yticks(range(1, 7))
axB.set_yticklabels([str(k) for k in range(1, 7)], fontsize=5.6)
axB.set_ylabel("Response\npattern", fontsize=5.8)
axB.set_xlim(40, 100)
axB.set_ylim(0.4, 7.0)
axB.set_xlabel("Lineages whose homeland is mostly drier, by ensemble member (%)", fontsize=6.0)
axB.tick_params(labelsize=5.6)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.text(-0.16, 1.12, "B", transform=axB.transAxes, fontsize=9, fontweight="bold", va="top")
ps.save(fig, F / "Fig3_forest_rain")

# ---- Supplementary: forest coloured by temperature (cooler or warmer than mitigation)
fig = plt.figure(figsize=(ps.W2, 6.0))
draw_forest(fig, {k: (v + 1) / 2 for k, v in net_heat.items()}, HEAT, y0=0.0, y1=0.955)
fig.text(0.01, 0.995, "Each language lineage drawn as its own tree; colour: share of homeland >0.1 \u00b0C warmer minus "
         "share >0.1 \u00b0C cooler than mitigation (same order as Fig. 3)", fontsize=5.8, color=ps.INK2, va="top")
cbar(fig, [0.70, 0.962, 0.25, 0.008], HEAT, [0, 0.5, 1], ["cooler", "same", "warmer"])
ps.save(fig, F / "FigS_forest_temperature")
print("forest saved")
