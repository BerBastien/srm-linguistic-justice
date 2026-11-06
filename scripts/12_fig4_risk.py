"""Figure 4 - The risk SRM adds, and whose world it falls on.

A  Termination premium (expected extra loss from a 1%/yr chance that deployment
   stops, % of PV GDP 2026-2100) averaged per dollar, per person, per language and
   per endangered language. Bars: ensemble median; whiskers: p17-p83; open and
   filled diamonds: medians for 0.5% and 2% annual hazards.
B  Historical responsibility versus termination exposure: countries' cumulative
   CO2 per person (OWID) against the median termination premium; bubble area
   proportional to the number of living languages (territory-weighted).
C  The gamble: change in present-value loss relative to iso-GMST mitigation if SRM
   is never interrupted (S - M, open circles) and including termination risk
   (S+T - M, filled), language-weighted, ensemble p17-p83.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

ps.setup()
T = P("results")
pers = pd.read_csv(T / "perspectives.csv")
sens = pd.read_csv(T / "termination_sensitivity.csv")
resp = pd.read_csv(T / "responsibility_country.csv")
ann = pd.read_csv(T / "table_annexI_split.csv")
main = pd.read_csv(T / "table1_group_stats_main.csv")

fig = plt.figure(figsize=(ps.W2, 3.3))
axA = fig.add_axes([0.055, 0.17, 0.20, 0.72])
axB = fig.add_axes([0.335, 0.17, 0.30, 0.72])
axC = fig.add_axes([0.835, 0.17, 0.16, 0.72])

# ------------------------------------------------------------------ A
order = [("gdp", "Dollar\n(GDP)"), ("pop", "Person"), ("lang", "Language"),
         ("end", "Endangered\nlanguage")]
d = pers[pers.metric == "premium"].set_index("perspective")
for i, (k, lab) in enumerate(order):
    r = d.loc[k]
    col = "#9a9891" if k == "gdp" else ("#5240b8" if k in ("lang", "end") else "#52514e")
    axA.bar(i, r.p50, width=0.62, color=col, lw=0)
    axA.plot([i, i], [r.p17, r.p83], color=ps.INK, lw=0.8)
    for hz, mk in (("p005", "D"), ("p02", "D")):
        v = sens[(sens.hazard == hz) & (sens.metric == "premium") & (sens.perspective == k)].p50
        axA.scatter(i + 0.22, v, marker=mk, s=9, facecolor="white" if hz == "p005" else ps.INK,
                    edgecolor=ps.INK, lw=0.5, zorder=4)
    if k != "gdp":
        axA.text(i, r.p83 + 0.15, f"×{r.BR_p50:.2f}", ha="center", fontsize=5.8, color=ps.INK2)
axA.set_xticks(range(4))
axA.set_xticklabels([l.replace("\n", " ") for _, l in order], fontsize=5.6, rotation=28,
                    ha="right", rotation_mode="anchor")
axA.set_xlabel("Averaged per", fontsize=6.2, labelpad=2)
axA.set_ylabel("Termination premium (% of PV GDP)")
axA.set_ylim(0, 9.2)
axA.grid(axis="y", color=ps.GRID, lw=0.4)
axA.scatter([], [], marker="D", s=9, facecolor="white", edgecolor=ps.INK, lw=0.5,
            label="hazard 0.5%/yr")
axA.scatter([], [], marker="D", s=9, facecolor=ps.INK, edgecolor=ps.INK, lw=0.5,
            label="hazard 2%/yr")
axA.legend(loc="upper left", fontsize=5.4, handletextpad=0.1)
axA.text(-0.33, 1.04, "A", transform=axA.transAxes, fontsize=9, fontweight="bold")

# ------------------------------------------------------------------ B
r = resp.dropna(subset=["cum_co2_t_per_cap"]).copy()
r = r[r.cum_co2_t_per_cap > 0]
for flag in (False, True):
    s = r[r.annex_i == flag]
    axB.scatter(s.cum_co2_t_per_cap, s.premium_p50, s=3 + 0.3 * s.n_lang, alpha=0.6,
                color=ps.ANNEX_COLORS[flag], edgecolor="white", lw=0.3,
                label=ps.ANNEX_LABEL[flag])
axB.set_xscale("log")
axB.set_xlabel("Cumulative CO$_2$ emitted per person (t, 1750–2024)")
axB.set_ylabel("Termination premium (% of PV GDP)")
lab_iso = {"PNG": "Papua New Guinea", "NGA": "Nigeria", "IDN": "Indonesia", "IND": "India",
           "CMR": "Cameroon", "USA": "USA", "AUS": "Australia",
           "MEX": "Mexico", "BRA": "Brazil", "RUS": "Russia", "ETH": "Ethiopia",
           "CHN": "China", }
off = {"MEX": (4, 4), "CHN": (4, -7), "USA": (-22, 5), "CAN": (2, 6), "AUS": (-30, -9),
       "RUS": (4, 3), "BRA": (-4, -9), "IDN": (5, 3), "PNG": (-38, 9), "VUT": (-30, -7),
       "IND": (6, 0), "NGA": (6, 2), "CMR": (5, 2), "COD": (-30, 6), "ETH": (-10, 7),
       "TCD": (-18, -8)}
lab_iso.pop("CHN")
for _, row in r[r.iso3.isin(lab_iso)].iterrows():
    axB.annotate(lab_iso[row.iso3], (row.cum_co2_t_per_cap, row.premium_p50),
                 xytext=off.get(row.iso3, (3, 2)), textcoords="offset points", fontsize=5.2,
                 color=ps.INK)
leg1 = axB.legend(loc="upper right", fontsize=5.4, handletextpad=0.2, markerscale=0.6)
axB.add_artist(leg1)
hs = [axB.scatter([], [], s=3 + 0.3 * nl, color="#c3c2b7") for nl in (10, 100, 500)]
axB.legend(hs, ["10", "100", "500 languages"], loc="lower left", fontsize=5.3,
           labelspacing=1.0, borderpad=0.8, handletextpad=0.5, ncol=1)
axB.grid(color=ps.GRID, lw=0.4)
axB.set_ylim(-0.3, 11.5)
axB.text(-0.16, 1.04, "B", transform=axB.transAxes, fontsize=9, fontweight="bold")

# ------------------------------------------------------------------ C
rows = [("All living languages", main[main.group == "All living"].iloc[0]),
        ("Non-Annex I, endangered", ann[(ann.variant == "annexI=False") & (ann.group == "Endangered")].iloc[0]),
        ("Non-Annex I, not endangered", ann[(ann.variant == "annexI=False") & (ann.group == "Not endangered")].iloc[0]),
        ("Annex I, endangered", ann[(ann.variant == "annexI=True") & (ann.group == "Endangered")].iloc[0]),
        ("Annex I, not endangered", ann[(ann.variant == "annexI=True") & (ann.group == "Not endangered")].iloc[0])]
for i, (lab, rr) in enumerate(rows):
    y = len(rows) - 1 - i
    axC.plot([rr.econ_S_minus_M_p17, rr.econ_S_minus_M_p83], [y + 0.12] * 2, color=ps.INK2,
             lw=0.6)
    axC.plot([rr.econ_ST_minus_M_p17, rr.econ_ST_minus_M_p83], [y - 0.12] * 2,
             color="#d03b3b", lw=0.6)
    axC.scatter(rr.econ_S_minus_M_p50, y + 0.12, s=14, facecolor="white", edgecolor=ps.INK,
                lw=0.7, zorder=3)
    axC.scatter(rr.econ_ST_minus_M_p50, y - 0.12, s=14, color="#d03b3b", zorder=3)
axC.axvline(0, color=ps.INK2, lw=0.6)
axC.set_yticks(range(len(rows)))
axC.set_yticklabels([l for l, _ in rows][::-1], fontsize=5.6)
axC.set_xlabel("Change in loss vs. mitigation\n(pp of PV GDP)")
axC.scatter([], [], s=14, facecolor="white", edgecolor=ps.INK, lw=0.7,
            label="SRM never interrupted")
axC.scatter([], [], s=14, color="#d03b3b", label="SRM with 1%/yr\ntermination hazard")
axC.legend(loc="lower right", fontsize=5.2, bbox_to_anchor=(1.03, 1.0), handletextpad=0.1,
           ncol=1)
axC.grid(axis="x", color=ps.GRID, lw=0.4)
axC.set_ylim(-0.6, len(rows) - 0.4)
axC.text(-1.15, 1.04, "C", transform=axC.transAxes, fontsize=9, fontweight="bold")

ps.save(fig, P("figures") / "Fig4_risk_responsibility")
print("saved")
