"""22 - Figure: where the world's community-language heritage meets SAI drying.

A  Land coloured lightly by the rainfall difference (SAI minus mitigation, 2100). Solid cells:
   the 0.5-degree cells that together hold half of all community-language weighted endemism,
   coloured by their rainfall outcome.
B  Range size and refuge: by quintile of homeland area, the mean fraction of a language's
   homeland that is drier, and the share of languages whose whole homeland is drier.
C  Language lineages (the 15 largest families and the isolates): mean fraction of homeland
   drier, ensemble-mean climate (dot) and p17-p83 across members (bar); colour = robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
I, T, F = P("interim"), P("results"), P("figures")
NB = ps.numbers(T / "numbers.json")
DRY, WET, SIM = "#8c510a", "#01665e", "#9a9891"
C_NON = ps.VIT_COLORS["In trouble"]

hc = pd.read_parquet(I / "heritage_cells.parquet")
hc = hc.merge(pd.read_parquet(I / "grid_fields.parquet")[["grid_id", "PH_2100"]], on="grid_id", how="left")
rq = pd.read_csv(T / "heritage_range_quintiles.csv")
fam = pd.read_parquet(I / "heritage_families_full.parquet")

fig = plt.figure(figsize=(ps.W2, 5.6))
# ------------------------------------------------------------------ A
ax = fig.add_axes([0.0, 0.43, 1.0, 0.57])
ps.draw_basemap(ax, P("ne_land"))
X, Y = ps.grid_mesh()
Z = ps.to_grid(hc.assign(v=hc.RP_2100.where(hc.PH_2100.abs() <= 50)), "v")       # hyper-arid masked
from matplotlib.colors import LinearSegmentedColormap
light = LinearSegmentedColormap.from_list("lightBrBG", ["#e2c590", "#efdcb4", "#f7efde", "#f6f5f2", "#ddf0ec", "#b9e0d9", "#94cdc4"])
ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=light, norm=TwoSlopeNorm(0, -30, 30), shading="flat",
              zorder=1, rasterized=True)
cls = np.where(hc.RP_2100 < -2, 2, np.where(hc.RP_2100 > 2, 0, 1)).astype(float)
cls[~hc.hotspot_half.to_numpy()] = np.nan
Zc = ps.to_grid(hc.assign(c=cls), "c")
ax.pcolormesh(X, Y, np.ma.masked_invalid(Zc), cmap=ListedColormap([WET, SIM, DRY]), vmin=-0.5, vmax=2.5,
              shading="flat", zorder=2, rasterized=True)
gpd.read_file(P("ne_coast")).to_crs(ps.ROBIN).plot(ax=ax, color="#6f6e69", lw=0.18, zorder=3)
share_land = 100 * NB["heritage.cells.land_share_holding_half_of_noninst_endemism"]
ax.legend(handles=[Patch(color=DRY, label="drier under SAI (>2 pp)"), Patch(color=SIM, label="similar (±2 pp)"),
                   Patch(color=WET, label="wetter under SAI (>2 pp)")],
          title=f"Cells holding half of all community-language\nheritage ({share_land:.1f}% of land), 2100:",
          title_fontsize=5.8, fontsize=5.6, loc="lower left", bbox_to_anchor=(0.0, 0.10), alignment="left",
          handlelength=1.0)
ax.text(0.0, 0.99, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax.text(0.30, 0.13, "Background: rainfall, SAI minus mitigation (faint; brown drier, green wetter; hyper-arid blank)",
        transform=ax.transAxes, fontsize=5.4, color=ps.INK2)

# ------------------------------------------------------------------ B
axB = fig.add_axes([0.07, 0.07, 0.36, 0.30])
x = np.arange(len(rq))
axB.bar(x, rq.share_no_refuge * 100, width=0.62, color=C_NON, lw=0, label="Whole homeland drier (no refuge)")
axB.vlines(x, rq.no_refuge_p17 * 100, rq.no_refuge_p83 * 100, color=ps.INK, lw=0.9)
axB.plot(x + 0.0, rq.mean_f_dry * 100, "o", color="white", mec=ps.INK, mew=0.9, ms=4.2, zorder=5,
         label="Mean share of homeland drier")
axB.vlines(x + 0.18, rq.mean_f_p17 * 100, rq.mean_f_p83 * 100, color=ps.MUTED, lw=0.9)
axB.axhline(100 * NB["heritage.land_share_drier"], color=ps.MUTED, lw=0.7, ls=(0, (2, 2)))


def km(v):
    return f"{v:,.0f}" if v >= 10 else f"{v:.0f}"


axB.set_xticks(x)
axB.set_xticklabels([f"{['Smallest', '2nd', '3rd', '4th', 'Largest'][i]}\n{km(v)} km²" for i, v in
                     enumerate(rq.range_km2_median)], fontsize=5.6)
axB.set_xlabel("Homeland size (fifths of community languages; median area)")
axB.set_ylabel("Non-institutional languages (%)")
axB.set_ylim(0, 95)
axB.grid(axis="y", color=ps.GRID, lw=0.4)
h_, l_ = axB.get_legend_handles_labels()
h_.append(Line2D([], [], color=ps.MUTED, lw=0.7, ls=(0, (2, 2))))
l_.append(f"All land drier ({100 * NB['heritage.land_share_drier']:.0f}%)")
axB.legend(h_, l_, loc="upper right", fontsize=5.4, handlelength=1.6)
ps.panel_label(axB, "B", x=-0.16)

# ------------------------------------------------------------------ C
axC = fig.add_axes([0.62, 0.07, 0.36, 0.30])
big = fam.sort_values("n", ascending=False).head(15)
iso = pd.read_parquet(I / "heritage_lang.parquet")
iso_f = iso[iso.lineage.str.startswith("[", na=False)]
rows = [(r.lineage, r.n, r.mean_f_dry, r.mean_f_p17, r.mean_f_p83, r.frac_members_majority_drier)
        for r in big.itertuples()]
rows = sorted(rows, key=lambda t: t[2])
y = np.arange(len(rows))
for yi, (nm, n, mf, lo, hi, fr) in zip(y, rows):
    col = DRY if fr >= 0.9 else (WET if fr <= 0.1 else SIM)
    axC.plot([lo * 100, hi * 100], [yi, yi], color=col, lw=1.4, alpha=0.6, solid_capstyle="round")
    axC.plot(mf * 100, yi, "o", color=col, ms=3.6, mec="white", mew=0.4, zorder=4)
axC.axvline(100 * NB["heritage.land_share_drier"], color=ps.MUTED, lw=0.7, ls=(0, (2, 2)))
axC.set_yticks(y)
axC.set_yticklabels([f"{nm} ({n:,})" for nm, n, *_ in rows], fontsize=5.5)
axC.set_xlim(-2, 102)
axC.set_xlabel("Share of homeland drier (%)")
axC.grid(axis="x", color=ps.GRID, lw=0.4)
axC.legend(handles=[Line2D([], [], color=DRY, marker="o", lw=1.4, ms=3.5, label="≥90% of members"),
                    Line2D([], [], color=SIM, marker="o", lw=1.4, ms=3.5, label="mixed")],
           title="Mostly drier in:", title_fontsize=5.2, loc="lower right", fontsize=5.0, handlelength=1.4,
           bbox_to_anchor=(1.03, -0.03))
ps.panel_label(axC, "C", x=-0.55)
ps.save(fig, F / "Fig_heritage")
print("saved")
