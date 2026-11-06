"""Shared figure style (PNAS column widths, validated palettes, map helpers)."""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pyproj import Transformer

# PNAS: single column 8.7 cm, 1.5 column 11.4 cm, double column 17.8 cm
CM = 1 / 2.54
W1, W15, W2 = 8.7 * CM, 11.4 * CM, 17.8 * CM

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

# Ordinal vitality ramp (one hue, light -> dark = stronger -> weaker vitality);
# validated: monotone lightness, adjacent dL >= 0.06, light end 2.46:1 on surface.
VIT_COLORS = {"Institutional": "#a898e6", "Stable": "#7b67d9", "In trouble": "#5240b8",
              "Dying": "#2b1d78"}
# Categorical pair (validated): non-Annex I / Annex I
ANNEX_COLORS = {False: "#2a78d6", True: "#eb6834"}
ANNEX_LABEL = {False: "Non-Annex I", True: "Annex I"}
# Diverging outcome classes (polarity, gray midpoint)
T_OUT_COLORS = {"SRM cooler than mitigation": "#2a78d6", "Similar (within 0.1 C)": "#c3c2b7",
                "SRM warmer than mitigation": "#e34948"}
P_OUT_COLORS = {"SRM drier than mitigation": "#a6611a", "Similar (within 2 pp)": "#c3c2b7",
                "SRM wetter than mitigation": "#1b9e77"}

CMAP_T = LinearSegmentedColormap.from_list(
    "bgr", ["#0d366b", "#256abf", "#86b6ef", NEUTRAL, "#f2a39d", "#d03b3b", "#7a1414"])
CMAP_P = LinearSegmentedColormap.from_list(
    "brg", ["#5a3206", "#a6611a", "#dfc27d", NEUTRAL, "#80cdc1", "#1b9e77", "#01463f"])


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 400, "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def panel_label(ax, s, x=-0.02, y=1.02):
    """Science style (v5 Science submission): bold upper-case panel letters."""
    ax.text(x, y, s.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom",
            ha="right")


# ------------------------------------------------------------------ maps
ROBIN = "+proj=robin +lon_0=0 +datum=WGS84"
_tr = Transformer.from_crs("EPSG:4326", ROBIN, always_xy=True)


def project(lon, lat):
    return _tr.transform(lon, lat)


def grid_mesh():
    """Projected corner mesh for the CLIMRISK 0.5-degree grid (360 x 720 cells)."""
    lon_e = np.clip(np.arange(-180.25, 180.0, 0.5), -180, 180)          # 721 edges
    lat_e = np.clip(np.arange(-90.25, 90.0, 0.5), -90, 90)              # 361 edges
    LON, LAT = np.meshgrid(lon_e, lat_e)
    X, Y = project(LON, LAT)
    return X, Y


def to_grid(df, col):
    """Fill a (360, 720) array from a land-cell table with grid_id (row-from-south)."""
    Z = np.full((360, 720), np.nan)
    gid = df["grid_id"].to_numpy() - 1
    Z[gid // 720, gid % 720] = df[col].to_numpy()
    return Z


def draw_basemap(ax, land_geojson, coast_geojson=None):
    import geopandas as gpd
    land = gpd.read_file(land_geojson).to_crs(ROBIN)
    land.plot(ax=ax, color="#f6f5f2", edgecolor="none", zorder=0)
    if coast_geojson:
        coast = gpd.read_file(coast_geojson).to_crs(ROBIN)
        coast.plot(ax=ax, color="#9a9891", linewidth=0.25, zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_ylim(project(0, -57)[1], project(0, 84)[1])
    ax.set_xlim(project(-180, 0)[0] * 0.985, project(180, 0)[0] * 0.985)


def numbers(path):
    with open(path) as fh:
        d = json.load(fh)
    return {k: v["value"] for k, v in d.items()}


def save(fig, path_noext, also=()):
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_noext}.png", bbox_inches="tight", dpi=400)
    for extra in also:
        fig.savefig(extra, bbox_inches="tight", dpi=400)
    plt.close(fig)
