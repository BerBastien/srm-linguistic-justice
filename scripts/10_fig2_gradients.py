"""Figure 2 - Exposure along the endangerment gradient, split by governance geography.

Top row (A-C): language-weighted means by EGIDS level (0 = international ...
9 = dormant) for languages whose territory lies mostly outside (blue) or inside
(orange) UNFCCC Annex I parties. Error bars: bootstrap 95% CI over languages.
  A residual temperature (SRM - mitigation, C, 2100)
  B share of languages with residual drying beyond 2 pp
  C SRM termination premium (% of host-country PV GDP, 2026-2100)

Bottom row (D-F): structural uncertainty. Gap between endangered and
non-endangered languages (mean, endangered minus not) for each of the 60
ensemble members, grouped by the six CLIMRISK-SRM response patterns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, EGIDS_ORDER, P  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402
from srmlang import stats as st  # noqa: E402

ps.setup()
I = P("interim")
lm = pd.read_parquet(I / "lang_metrics.parquet")
memR = np.load(I / "lang_member_R.npz", allow_pickle=True)
memE = np.load(I / "lang_member_econ.npz", allow_pickle=True)
pidx = np.load(I / "country_member_fields.npz")["pattern_index"]
DB_P = CFG["analysis"]["precip_deadband_pct"]

levels = [e for e in EGIDS_ORDER if e not in ("x10", "0", "9")]  # n < 8 at level 0 and 9
lm["lev"] = pd.Categorical(lm.EGIDS, categories=levels, ordered=True)
xpos = {e: i for i, e in enumerate(levels)}

fig = plt.figure(figsize=(ps.W2, 4.1))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95], hspace=0.62, wspace=0.40)
axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
axes_bot = [fig.add_subplot(gs[1, i]) for i in range(3)]

metrics = [
    ("R_2100", lambda x: x, "Residual temperature (°C)", "A"),
    ("RP_2100", lambda x: (x < -DB_P).astype(float) * 100,
     "Languages with residual\ndrying > 2 pp (%)", "B"),
    ("econ_premium", lambda x: x, "Termination premium\n(% of PV GDP)", "C"),
]
spans = [("Institutional", 0, 3), ("Stable", 4, 5), ("In trouble", 6, 7), ("Dying", 8, 9)]
for ax, (col, fn, lab, tag) in zip(axes_top, metrics):
    for name, a, b in spans:
        ax.axvspan(a - 0.5, b + 0.5, color=ps.VIT_COLORS[name], alpha=0.10, lw=0)
    for ax_flag, off in ((False, -0.14), (True, 0.14)):
        d = lm[lm.annex_i == ax_flag]
        xs, ms, lo, hi = [], [], [], []
        for e in levels:
            v = fn(d.loc[d.EGIDS == e, col].dropna().to_numpy())
            if v.size < 8:
                continue
            reps = st.bootstrap(lambda idx: v[idx].mean(), v.size, 1000, 1)
            xs.append(xpos[e] + off)
            ms.append(v.mean())
            l_, h_ = st.ci(reps)
            lo.append(l_)
            hi.append(h_)
        ms, lo, hi = map(np.array, (ms, lo, hi))
        ax.errorbar(xs, ms, yerr=[ms - lo, hi - ms], fmt="o", ms=3.2, lw=0.8, capsize=0,
                    color=ps.ANNEX_COLORS[ax_flag], mec="white", mew=0.4,
                    label=ps.ANNEX_LABEL[ax_flag] + f" (n = {len(d):,})")
    allv = fn(lm[col].dropna().to_numpy()).mean()
    ax.axhline(allv, color=ps.INK2, lw=0.6, ls=(0, (3, 2)))
    if col == "R_2100":
        ax.axhline(0, color=ps.AXIS, lw=0.6)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=5.8)
    ax.set_xlabel("EGIDS level (stronger → weaker)", fontsize=6.2)
    ax.set_ylabel(lab)
    ax.grid(axis="y", color=ps.GRID, lw=0.4)
    ax.text(-0.30, 1.08, tag.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold")
for ax in axes_top:
    for j, (name, a, b) in enumerate(spans):
        ax.text((a + b) / 2, 1.0 + 0.075 * (j % 2), name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=4.9,
                color=ps.VIT_COLORS[name] if name != "Institutional" else "#7b67d9")
h, l = axes_top[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=3, fontsize=6,
           handletextpad=0.2, columnspacing=1.5)
fig.text(0.70, 1.012, "dashed line: all living languages", fontsize=5.6, color=ps.INK2)

# ------------------------------------------------------------------ bottom row
end = lm.vitality.isin(["In trouble", "Dying"]).to_numpy()
iso_R = pd.Series(np.arange(len(memR["ISO"])), index=memR["ISO"])
iso_E = pd.Series(np.arange(len(memE["ISO"])), index=memE["ISO"])
iR = lm.ISO.map(iso_R).to_numpy()
iE = lm.ISO.map(iso_E).to_numpy()
okR = ~np.isnan(iR)
R = memR["R_2100"][iR[okR].astype(int)]
RP = memR["RP_2100"][iR[okR].astype(int)]
PR = memE["premium"][iE.astype(int)]
eR = end[okR]
gaps = {
    "D": (R[eR].mean(0) - R[~eR].mean(0), "Gap in residual T (°C)"),
    "E": (((RP[eR] < -DB_P).mean(0) - (RP[~eR] < -DB_P).mean(0)) * 100,
          "Gap in share with drying (pp)"),
    "F": (np.nanmean(PR[end], 0) - np.nanmean(PR[~end], 0), "Gap in termination premium (pp)"),
}
rng = np.random.default_rng(3)
for ax, (tag, (g, lab)) in zip(axes_bot, gaps.items()):
    for k in range(1, 7):
        sel = pidx == k
        y = np.full(sel.sum(), k) + rng.uniform(-0.18, 0.18, sel.sum())
        ax.scatter(g[sel], y, s=5, color=ps.INK2, lw=0, alpha=0.75)
        ax.plot([g[sel].mean()] * 2, [k - 0.32, k + 0.32], color=ps.INK, lw=1.4)
    ax.axvline(0, color=ps.AXIS, lw=0.7)
    ax.axvline(g.mean(), color="#d03b3b", lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Pattern {k}" for k in range(1, 7)], fontsize=5.8)
    ax.set_ylim(6.5, -0.15)
    ax.set_xlabel(lab + "\nendangered minus not endangered", fontsize=6.2)
    frac = (g > 0).mean()
    ax.text(0.98, 0.985, f"{frac * 60:.0f} of 60 members > 0", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.6, color=ps.INK2,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.grid(axis="x", color=ps.GRID, lw=0.4)
    ax.text(-0.36, 1.06, tag.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold")

ps.save(fig, P("figures") / "Fig2_gradients")
print("saved")
