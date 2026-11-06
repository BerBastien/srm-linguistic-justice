"""Figure 1 - Same global temperature, different local climates.

A  residual temperature in 2100, SRM minus iso-GMST mitigation (C), with the home
   areas (polygon representative points) of endangered languages (In trouble + Dying)
B  residual precipitation in 2100 (percentage points of the 1900 baseline)
C  zonal land means of the temperature residual, and the latitude distribution of
   languages by vitality class (kernel density, 3-degree bandwidth)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import LIVING_ORDER, P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
grid = pd.read_parquet(P("interim") / "grid_fields.parquet")
pe = pd.read_parquet(P("interim") / "poly_exposure.parquet")
lm = pd.read_parquet(P("interim") / "lang_metrics.parquet")
pe = pe[pe.land_w_km2.notna() & ~pe.sign_language & pe.vitality.isin(LIVING_ORDER)]
endp = pe[pe.vitality.isin(["In trouble", "Dying"])]
coast = gpd.read_file(P("ne_coast")).to_crs(ps.ROBIN)

X, Y = ps.grid_mesh()
fig = plt.figure(figsize=(ps.W2, 4.35))
axA = fig.add_axes([0.0, 0.52, 0.72, 0.47])
axB = fig.add_axes([0.0, 0.03, 0.72, 0.47])

for ax, col, cmap, vlim, lab, tag, ticks in (
        (axA, "R_2100", ps.CMAP_T, 0.6, "Residual temperature (°C)\nSRM minus mitigation, 2100",
         "A", [-0.6, -0.3, 0, 0.3, 0.6]),
        (axB, "RP_2100", ps.CMAP_P, 30, "Residual precipitation (pp)\nSRM minus mitigation, 2100",
         "B", [-30, -15, 0, 15, 30])):
    ps.draw_basemap(ax, P("ne_land"))
    Z = ps.to_grid(grid, col)
    if col == "RP_2100":
        # map-only mask: hyper-arid cells where the no-SRM change exceeds +/-50% of a
        # very small baseline make percentage residuals visually dominant
        arid = ps.to_grid(grid.assign(a=(grid.PH_2100.abs() > 50).astype(float)), "a") > 0
        Za = np.where(arid, 1.0, np.nan)
        ax.pcolormesh(X, Y, np.ma.masked_invalid(Za), cmap=plt.matplotlib.colors.ListedColormap(["#d9d8d3"]),
                      shading="flat", zorder=1, rasterized=True)
        Z = np.where(arid, np.nan, Z)
    m = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=cmap,
                      norm=TwoSlopeNorm(0, -vlim, vlim), shading="flat", zorder=1,
                      rasterized=True)
    coast.plot(ax=ax, color="#6f6e69", lw=0.18, zorder=3)
    xe, ye = ps.project(endp.rep_lon.to_numpy(), endp.rep_lat.to_numpy())
    ax.scatter(xe, ye, s=0.9, c="#0b0b0b", lw=0, alpha=0.75, zorder=5, rasterized=True)
    cax = ax.inset_axes([0.035, 0.10, 0.25, 0.035])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1)
    cb.ax.set_title(lab, fontsize=5.9, loc="left", pad=2, fontweight="normal")
    ax.text(0.0, 0.98, tag.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

axA.scatter([], [], s=5, c="#0b0b0b", label=f"Endangered-language\nhome areas (n = {len(endp):,})")
axA.legend(loc="lower left", bbox_to_anchor=(0.0, 0.21), fontsize=5.6, handletextpad=0.2,
           markerscale=1.0)
axB.fill_between([], [], color="#d9d8d3", label="Hyper-arid cells\n(masked on map)")
axB.legend(loc="lower left", bbox_to_anchor=(0.0, 0.21), fontsize=5.6, handlelength=1.0)

# ------------------------------------------------------------------ panel C
axC1 = fig.add_axes([0.765, 0.07, 0.095, 0.88])
axC2 = fig.add_axes([0.875, 0.07, 0.115, 0.88], sharey=axC1)
lat_bins = np.arange(-56, 84, 4)
g = grid.copy()
g["band"] = pd.cut(g.lat, lat_bins)
zm = g.groupby("band", observed=True).apply(
    lambda d: pd.Series({"R": np.average(d.R_2100, weights=d.area_km2)}))
mid = np.array([b.mid for b in zm.index])
axC1.axvline(0, color=ps.AXIS, lw=0.6)
axC1.fill_betweenx(mid, 0, zm.R, where=zm.R > 0, color="#f2a39d", lw=0, interpolate=True)
axC1.fill_betweenx(mid, 0, zm.R, where=zm.R < 0, color="#86b6ef", lw=0, interpolate=True)
axC1.plot(zm.R, mid, color=ps.INK, lw=0.9)
axC1.set_xlabel("Zonal land-mean\nresidual T (°C)", fontsize=6.2)
axC1.set_ylabel("Latitude (°)", labelpad=1)
axC1.set_xlim(-0.4, 0.4)
axC1.set_xticks([-0.3, 0, 0.3])
axC1.set_ylim(-56, 82)
axC1.text(-0.62, 1.0, "C", transform=axC1.transAxes, fontsize=9, fontweight="bold", va="top")
axC1.text(-0.38, 76, "SRM\nwarmer", fontsize=5.3, color="#d03b3b", va="top")
axC1.text(0.03, -8, "SRM\ncooler", fontsize=5.3, color="#1c5cab", va="top")

yy = np.linspace(-56, 82, 300)
for v in LIVING_ORDER:
    x = lm[lm.vitality == v].lat_mean.dropna().to_numpy()
    k = gaussian_kde(x, bw_method=3.0 / x.std())
    axC2.plot(k(yy) * 100, yy, color=ps.VIT_COLORS[v], lw=1.1, label=f"{v}\n(n = {len(x):,})")
axC2.set_xlabel("Languages\n(density, % per °)", fontsize=6.2)
axC2.tick_params(labelleft=False)
axC2.set_xlim(0, None)
axC2.legend(loc="upper right", fontsize=5.3, handlelength=1.0, bbox_to_anchor=(1.04, 1.0),
            labelspacing=0.5)
for ax in (axC1, axC2):
    ax.grid(axis="y", color=ps.GRID, lw=0.4)
    for y0 in (-23.44, 23.44, 66.56):
        ax.axhline(y0, color=ps.MUTED, lw=0.4, ls=(0, (2, 2)))

out = P("figures") / "Fig1_residual_map"
ps.save(fig, out)
print("saved", out)
