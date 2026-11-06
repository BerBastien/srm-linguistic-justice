"""20 - SI tables and figure for the rainfall paper.

Writes LaTeX row files to results/tables/latex/ (tabS_dry_*.tex, tabS_pixel.tex) and
FigS_drying_agreement (fraction of the 60 members in which each non-institutional
language is more than 2 pp drier under SAI than under mitigation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, record  # noqa: E402
from srmlang import plotstyle as ps  # noqa: E402

I, T, F = P("interim"), P("results"), P("figures")
L = T / "latex"
DB = CFG["analysis"]["precip_deadband_pct"]
NB = ps.numbers(T / "numbers.json")
dl = pd.read_parquet(I / "dry_lang.parquet")
z = np.load(I / "dry_member.npz", allow_pickle=True)
non = dl.noninst.to_numpy()
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731

# ---- thresholds
thr = pd.read_csv(T / "dry_thresholds.csv")
rows = []
for th in [0, 1, 2, 5, 10, 15]:
    r = thr[thr.threshold_pp == th].iloc[0]
    rows.append(f"{th:g} & {pct(r.share_noninst)} & {pct(r.member_p17_noninst)}--{pct(r.member_p83_noninst)} & "
                f"{pct(r.share_inst)} \\\\")
(L / "tabS_dry_thresholds.tex").write_text("\n".join(rows) + "\n")

# ---- patterns
pat = pd.read_csv(T / "dry_by_pattern.csv")
bm = pd.read_csv(T / "dry_by_member.csv")
dg = pd.read_csv(T / "member_group_stats.csv")
dg = dg[dg.horizon == 2100].groupby("pattern").dG.mean()
rows = [f"{int(r.pattern)} & {dg[r.pattern]:.2f} & {pct(r.share_drier_noninst)} & "
        f"{pct(bm[bm.pattern == r.pattern].share_drier_noninst.min())}--{pct(bm[bm.pattern == r.pattern].share_drier_noninst.max())} & "
        f"{pct(r.share_drier_inst)} & ${r.median_RP_noninst:+.1f}$ \\\\" for r in pat.itertuples()]
(L / "tabS_dry_patterns.tex").write_text("\n".join(rows) + "\n")

# ---- absolute rainfall
ab = pd.read_csv(T / "dry_absolute_by_member.csv")
sH = (z["PH_2100"][non] < 0).mean(0)
record("dry.member.noninst.share_below1900.H.p17_p50_p83", np.percentile(sH, [17, 50, 83]))
q = lambda a: f"{100 * np.percentile(a, 17):.1f}--{100 * np.percentile(a, 83):.1f}"  # noqa: E731
fl = ab.n_flip.to_numpy()
rows = [
    f"Less rain than in 1900, no SAI (H) & {pct(NB['dry.noninst.share_below1900.H'])} & {q(sH)} \\\\",
    f"Less rain than in 1900, mitigation (M) & {pct(NB['dry.noninst.share_below1900.M'])} & {q(ab.share_below1900_M)} \\\\",
    f"Less rain than in 1900, SAI (S) & {pct(NB['dry.noninst.share_below1900.S'])} & {q(ab.share_below1900_S)} \\\\",
    f"More rain than 1900 under M, less under S & {pct(NB['dry.noninst.share_flip'])} & "
    f"{100 * np.percentile(fl, 17) / non.sum():.1f}--{100 * np.percentile(fl, 83) / non.sum():.1f} \\\\",
    f"S more than 2\\,pp drier than H & {pct(NB['dry.noninst.share_S_drier_than_H_by_2pp'])} & {q(ab.share_S_drier_than_H)} \\\\",
    f"S more than 2\\,pp drier than both H and M & {pct(NB['dry.noninst.share_S_below_both_H_and_M'])} & -- \\\\",
]
(L / "tabS_dry_absolute.tex").write_text("\n".join(rows) + "\n")

# ---- sensitivities
c = NB
rows = [
    f"Main: ensemble-mean climate, language-weighted & {pct(c['dry.noninst.share_drier'])} & "
    f"{pct(c['dry.t1.Non-institutional.ci_lo'])}--{pct(c['dry.t1.Non-institutional.ci_hi'])} \\\\",
    f"Member median (p17--p83 in brackets) & {pct(c['dry.member.noninst.share_drier.p17_p50_p83'][1])} & "
    f"{pct(c['dry.member.noninst.share_drier.p17_p50_p83'][0])}--{pct(c['dry.member.noninst.share_drier.p17_p50_p83'][2])} \\\\",
    f"Territory-weighted & {pct(c['dry.noninst.share_drier_territory_weighted'])} & -- \\\\",
    f"Speaker-weighted (L1) & {pct(c['dry.noninst.share_drier_speaker_weighted'])} & -- \\\\",
    f"Excluding {c['dry.noninst.n_excluded_hyperarid']} languages in hyper-arid areas & "
    f"{pct(c['dry.noninst.share_drier_excl_hyperarid'])} & -- \\\\",
    f"Horizon 2050 & {pct(c['dry.noninst.share_drier_2050'])} & -- \\\\",
    f"Stable languages only & {pct(c['dry.noninst.share_drier_stable'])} & -- \\\\",
    f"Endangered (In trouble, Dying) only & {pct(c['dry.noninst.share_drier_endangered'])} & -- \\\\",
    f"LIC units indigenous to their country & {pct(c['dry.lic.indigenous_noninst.share_drier'])} & -- \\\\",
    f"Drier in $\\geq$ two-thirds of members & {pct(c['dry.noninst.share_drier_in_ge_two_thirds_members'])} & -- \\\\",
    f"Drier in $\\geq$ 5 of 6 patterns & {pct(c['dry.noninst.share_drier_in_ge_5of6_patterns'])} & -- \\\\",
    f"Institutional languages (comparison) & {pct(c['dry.inst.share_drier'])} & "
    f"{pct(c['dry.t1.Institutional.ci_lo'])}--{pct(c['dry.t1.Institutional.ci_hi'])} \\\\",
    f"Difference, non-institutional minus institutional (pp) & {100 * c['dry.gap_noninst_minus_inst_share_drier']:+.1f} & "
    f"{100 * c['dry.gap_noninst_minus_inst_share_drier_ci'][0]:+.1f} to {100 * c['dry.gap_noninst_minus_inst_share_drier_ci'][1]:+.1f} \\\\",
    f"Same, within countries (LIC units, country fixed effects; pp) & {100 * c['dry.within_country.gap_noninst_minus_inst_share_drier']:+.1f} & "
    f"{100 * c['dry.within_country.gap_ci'][0]:+.1f} to {100 * c['dry.within_country.gap_ci'][1]:+.1f} \\\\",
]
(L / "tabS_dry_sens.tex").write_text("\n".join(rows).replace("+-", "-") + "\n")

# ---- areas and countries
area = pd.read_csv(T / "dry_by_area.csv")
rows = [f"{r.Area} & {int(r.n):,} & {int(r.n_drier):,} & {pct(r.share_drier)} & ${r.median_RP:+.1f}$ \\\\"
        for r in area.sort_values("n", ascending=False).itertuples()]
(L / "tabS_dry_area.tex").write_text("\n".join(rows) + "\n")
cty = pd.read_csv(T / "dry_by_country.csv").head(15)
lx = pd.read_parquet(I / "lang_exposure.parquet")
cname = lx.drop_duplicates("Country_Code").set_index("Country_Code").Country_Name.to_dict()
rows = [f"{cname.get(r.Country_Code, r.Country_Code)} & {int(r.n):,} & {int(r.n_drier):,} & {pct(r.share_drier)} \\\\"
        for r in cty.itertuples()]
(L / "tabS_dry_countries.tex").write_text("\n".join(rows) + "\n")

# ---- pixel-level within-country economic tests (script 16)
pt = pd.read_csv(T / "pixel_within_country_tests.csv")
lab = {"T1_endangered_within_country": "Endangered vs.\\ not, same country",
       "T2_richness_all_within_country": "Language-rich vs.\\ language-poor cells (p90 vs.\\ p10)",
       "T2_richness_endangered_within_country": "Endangered-language-rich vs.\\ -poor cells"}
rows = []
for t_, nm in lab.items():
    for out in ["premium_local", "benefit"]:
        w = "economy" if t_.startswith("T1") else "area"
        r = pt[(pt.test == t_) & (pt.outcome == out) & (pt["map"] == "p50") & (pt.weighting == w)].iloc[0]
        rows.append(f"{nm} & {'Termination premium' if out.startswith('premium') else 'Benefit of uninterrupted SAI'} & "
                    f"${r.estimate:+.2f}$ & ${r.ci_lo:+.2f}$ to ${r.ci_hi:+.2f}$ & {100 * r.rel_to_mean:+.0f} \\\\")
(L / "tabS_pixel.tex").write_text("\n".join(rows) + "\n")

# ---- Figure: agreement across members
pe = pd.read_parquet(I / "poly_exposure.parquet")
pe = pe[pe.land_w_km2.notna() & ~pe.sign_language & pe.ISO.isin(dl[dl.noninst].ISO)]
pe = pe.merge(dl[["ISO", "frac_members_drier"]], on="ISO")
ps.setup()
fig = plt.figure(figsize=(ps.W2, 3.6))
ax = fig.add_axes([0, 0, 1, 1])
ps.draw_basemap(ax, P("ne_land"), P("ne_coast"))
cmap = LinearSegmentedColormap.from_list("br", ["#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005"])
x, y = ps.project(pe.rep_lon.to_numpy(), pe.rep_lat.to_numpy())
o = np.argsort(pe.frac_members_drier.to_numpy())
s = ax.scatter(x[o], y[o], c=pe.frac_members_drier.to_numpy()[o] * 60, cmap=cmap, vmin=0, vmax=60, s=2.2, lw=0,
               zorder=5, rasterized=True)
cax = ax.inset_axes([0.04, 0.12, 0.25, 0.035])
cb = fig.colorbar(s, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.6, length=2, pad=1)
cb.ax.set_title("Members (of 60) in which the language is\n>2 pp drier under SAI than mitigation, 2100",
                fontsize=5.9, loc="left", pad=2, fontweight="normal")
ps.save(fig, F / "FigS_drying_agreement")
print("SI tables and figure written")
