"""Figure 3 - Where does SRM put the residual heat and the drying? An alluvial
(Sankey) diagram of 7,079 language-in-country communities (Ethnologue LIC polygons):

  origin in country (indigenous / non-indigenous)  ->  vitality (LIC-level EGIDS)
  ->  temperature outcome of SRM relative to iso-GMST mitigation (2100)
  ->  precipitation outcome (2100)

Ribbon width = number of communities; ribbon colour = vitality class, so endangered
communities can be followed into the outcomes. Dead-bands: +/-0.1 C and +/-2 pp.
A speaker-weighted version is in the SI (variant='L1').
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import LIVING_ORDER, P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
WEIGHT = sys.argv[1] if len(sys.argv) > 1 else "n_lic"   # or "L1"
flows = pd.read_csv(P("results") / "sankey_flows_lic.csv")
ORIG = ["Indigenous to country", "Non-indigenous (colonial or immigrant origin)"]
TOUT = ["SRM cooler than mitigation", "Similar (within 0.1 C)", "SRM warmer than mitigation"]
POUT = ["SRM drier than mitigation", "Similar (within 2 pp)", "SRM wetter than mitigation"]
stages = [("origin", ORIG), ("vitality_LIC", LIVING_ORDER), ("T_out", TOUT), ("P_out", POUT)]
N = flows[WEIGHT].sum()
GAP = 0.035 * N
XW = 0.06                                     # node width (stage units)


def node_layout(col, cats):
    tot = flows.groupby(col)[WEIGHT].sum().reindex(cats).fillna(0)
    span = tot.sum() + GAP * (len(cats) - 1)
    y = span / 2 + 0.0
    pos = {}
    for c in cats:
        pos[c] = [y, y - tot[c]]               # top, bottom
        y -= tot[c] + GAP
    return pos, tot


layouts = [node_layout(c, cats) for c, cats in stages]


def band(ax, x0, x1, a0, b0, a1, b1, color, alpha=0.58):
    t = np.linspace(0, 1, 60)
    s = 3 * t ** 2 - 2 * t ** 3                    # smooth step
    x = x0 + (x1 - x0) * t
    top = a0 + (a1 - a0) * s
    bot = b0 + (b1 - b0) * s
    ax.fill_between(x, bot, top, facecolor=color, alpha=alpha, edgecolor="white", lw=0.25)


fig, ax = plt.subplots(figsize=(ps.W2, 3.55))
xs = [0, 1, 2, 3]
for si in range(3):
    (ca, cats_a), (cb, cats_b) = stages[si], stages[si + 1]
    pos_a, _ = layouts[si]
    pos_b, _ = layouts[si + 1]
    f = flows.groupby([ca, cb, "vitality_LIC"])[WEIGHT].sum()
    out_cursor = {a: pos_a[a][0] for a in cats_a}
    in_cursor = {b: pos_b[b][0] for b in cats_b}
    # outgoing order: by destination then vitality; incoming order: by source then vitality
    seg_out, seg_in = {}, {}
    for a in cats_a:
        for b in cats_b:
            for v in LIVING_ORDER:
                w = f.get((a, b, v), 0)
                if w > 0:
                    seg_out[(a, b, v)] = (out_cursor[a], out_cursor[a] - w)
                    out_cursor[a] -= w
    for b in cats_b:
        for a in cats_a:
            for v in LIVING_ORDER:
                w = f.get((a, b, v), 0)
                if w > 0:
                    seg_in[(a, b, v)] = (in_cursor[b], in_cursor[b] - w)
                    in_cursor[b] -= w
    for key, (a0, b0) in seg_out.items():
        a1, b1 = seg_in[key]
        band(ax, xs[si] + XW / 2, xs[si + 1] - XW / 2, a0, b0, a1, b1, ps.VIT_COLORS[key[2]])

node_col = {**{o: "#52514e" for o in ORIG}, **ps.VIT_COLORS, **ps.T_OUT_COLORS,
            **ps.P_OUT_COLORS}
short = {ORIG[0]: "Indigenous\nto country", ORIG[1]: "Non-indigenous\n(colonial/immigrant)",
         TOUT[0]: "SRM cooler", TOUT[1]: "Similar\n(±0.1 °C)", TOUT[2]: "SRM warmer",
         POUT[0]: "SRM drier", POUT[1]: "Similar\n(±2 pp)", POUT[2]: "SRM wetter"}
for si, ((col, cats), (pos, tot)) in enumerate(zip(stages, layouts)):
    for c in cats:
        top, bot = pos[c]
        if tot[c] <= 0:
            continue
        ax.add_patch(plt.Rectangle((xs[si] - XW / 2, bot), XW, top - bot, color=node_col[c],
                                   lw=0, zorder=3))
        pct = 100 * tot[c] / N
        lab = f"{short.get(c, c)}\n{tot[c]:,.0f} ({pct:.0f}%)" if WEIGHT == "n_lic" else \
            f"{short.get(c, c)}\n{pct:.0f}%"
        if si == 0:
            ax.text(xs[si] - XW / 2 - 0.03, (top + bot) / 2, lab, ha="right", va="center",
                    fontsize=6)
        elif si == 3:
            ax.text(xs[si] + XW / 2 + 0.03, (top + bot) / 2, lab, ha="left", va="center",
                    fontsize=6)
        else:
            ax.text(xs[si] + XW / 2 + 0.02, (top + bot) / 2, lab, ha="left", va="center",
                    fontsize=5.6, zorder=4,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))

heads = ["Origin of the language\nin the country", "Vitality\n(EGIDS, community level)",
         "Temperature, 2100\nSRM vs. mitigation", "Precipitation, 2100\nSRM vs. mitigation"]
ytop = max(p[0] for lay, _ in layouts for p in lay.values())
for x, h in zip(xs, heads):
    ax.text(x, ytop + 0.06 * N, h, ha="center", va="bottom", fontsize=6.4, fontweight="bold")

# per-origin outcome summary
d = flows.groupby(["origin", "T_out"])[WEIGHT].sum().unstack().reindex(ORIG)
e = flows.groupby(["origin", "P_out"])[WEIGHT].sum().unstack().reindex(ORIG)
txt = []
for o, lab in zip(ORIG, ["Indigenous", "Non-indigenous"]):
    t = d.loc[o] / d.loc[o].sum() * 100
    p = e.loc[o] / e.loc[o].sum() * 100
    txt.append(f"{lab}: {t[TOUT[0]]:.0f}% cooler, {t[TOUT[2]]:.0f}% warmer; "
               f"{p[POUT[0]]:.0f}% drier, {p[POUT[2]]:.0f}% wetter")
ax.text(0.0, -0.035, "\n".join(txt), transform=ax.transAxes, fontsize=5.8, color=ps.INK2,
        va="top")
ax.set_xlim(-0.75, 3.75)
ax.set_axis_off()
suffix = "" if WEIGHT == "n_lic" else "_speakers"
ps.save(fig, P("figures") / f"Fig3_sankey{suffix}")
print("saved")
