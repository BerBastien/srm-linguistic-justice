"""14 - LaTeX tables for the SI Appendix (aggregated results only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import EGIDS_LABEL, EGIDS_ORDER, LIVING_ORDER, P, VITALITY  # noqa: E402
from srmlang.plotstyle import numbers  # noqa: E402

T = P("results")
OUT = T / "latex"
OUT.mkdir(exist_ok=True)
I = P("interim")
nb = numbers(T / "numbers.json")


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


# ---------------------------------------------------------------- S2 EGIDS counts
lang = pd.read_parquet(I / "eth_languages.parquet")
lm = pd.read_parquet(I / "lang_metrics.parquet")
rows = []
for e in EGIDS_ORDER:
    n_all = int((lang.EGIDS == e).sum())
    n_s = int((lm.EGIDS == e).sum())
    rows.append(f"{e} & {EGIDS_LABEL[e]} & {VITALITY[e]} & {n_all:,} & {n_s:,} \\\\")
write("tabS_egids", "\n".join(rows))

# ---------------------------------------------------------------- S3 processing flow
flow = [
    ("Languages in Ethnologue 28 (Table of Languages)", nb["eth.n_languages_table"]),
    ("\\quad of which living (EGIDS 0--9)", nb["eth.n_living_languages"]),
    ("Language-in-country (LIC) polygons in the GIS dataset", nb["eth.n_polygons_raw"]),
    ("\\quad invalid geometries repaired", nb["eth.n_polygons_invalid_repaired"]),
    ("\\quad LIC polygons flagged indigenous to the country", nb["eth.polys_is_indigenous_T"]),
    ("\\quad LIC polygons flagged non-indigenous", nb["eth.polys_is_indigenous_F"]),
    ("\\quad sign-language polygons (excluded from main sample)", nb["eth.polys_sign_language"]),
    ("Polygons with exact land-cell coverage", nb["overlay.n_polys_with_land_coverage"]),
    ("Polygons assigned nearest land cell ($\\leq$150 km)", nb["overlay.n_polys_nearest_cell_fallback"]),
    ("Polygons dropped (no land cell within 150 km)", nb["overlay.n_polys_dropped_no_land"]),
    ("Languages with a polygon", nb["eth.n_languages_with_polygon"]),
    ("Languages with climate exposure", nb["overlay.n_languages_with_exposure"]),
    ("Main sample: living, non-sign languages", nb["sample.n_languages_main"]),
    ("\\quad endangered (In trouble + Dying)", nb["sample.n_endangered_main"]),
    ("\\quad mostly in Annex I parties", nb["sample.n_annexI"]),
    ("LIC communities in the alluvial analysis (Fig.~3)", nb["sankey.n_lic_polygons"]),
    ("Languages with economic exposure", nb["econ.n_languages_with_econ"]),
]
write("tabS_flow", "\n".join(f"{a} & {int(b):,} \\\\" for a, b in flow))

# ---------------------------------------------------------------- S4 validation
val = pd.read_csv(T / "climrisk_validation.csv")
extra = [("Emulator: median leave-one-member-out $R^2$", nb["emulator.median_r2"]),
         ("Emulator: median relative CV RMSE", nb["emulator.median_rel_rmse"]),
         ("Emulator: 95th percentile relative CV RMSE", nb["emulator.p95_rel_rmse"]),
         ("Emulator: share of member-years clamped to fitted support",
          nb["emulator.frac_member_years_clamped"]),
         ("Max relative error, recomputed vs.\\ stored PV (no SRM)",
          nb["validation.pv_recompute_max_rel_err_H"]),
         ("Max relative error, recomputed vs.\\ stored PV (SRM)",
          nb["validation.pv_recompute_max_rel_err_S"]),
         ("Max $|$CSV$-$HDF5$|$ termination premium (pp)", nb["validation.csv_vs_h5_premium_maxabs"])]
body = [f"{r.check.replace('|', '$|$').replace('E[T_M]', 'E[$T_M$]').replace(' T_H ', ' $T_H$ ')} & {r.value:.3g} \\\\"
        for r in val.itertuples()]
body += [f"{a} & {b:.3g} \\\\" for a, b in extra]
write("tabS_validation", "\n".join(body))

# ---------------------------------------------------------------- S5 gaps & tests
g = pd.read_csv(T / "table_gaps_tests.csv")
wl = {"w_lang": "languages", "w_spk": "speakers", "w_area": "territory"}
vl = {"R_2100": "Residual T 2100 (\\textdegree C)", "R_2050": "Residual T 2050 (\\textdegree C)",
      "RP_2100": "Residual P 2100, mean (pp)", "RP_2050": "Residual P 2050, mean (pp)",
      "drier_2100": "Drier than mitigation 2100 (\\% pts)",
      "econ_ST_minus_M": "S+T$-$M (pp PV GDP)", "econ_premium": "Termination premium (pp)",
      "econ_S_minus_M": "S$-$M (pp PV GDP)"}
body = []
for r in g.itertuples():
    p = "$<$0.0001" if r.wilcoxon_p < 1e-4 else f"{r.wilcoxon_p:.3f}"
    body.append(f"{vl[r.variable]} & {wl[r.weighting]} & {r.gap_end_minus_non:+.3f} & "
                f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {p} & {r.rank_biserial:+.3f} & "
                f"{r.gap_dying_minus_institutional:+.3f} \\\\")
write("tabS_gaps", "\n".join(body))

# ---------------------------------------------------------------- S6 relative risks
rr = pd.read_csv(T / "table_relative_risks.csv")
cl = {"R_warmer": "$R>0.1$\\textdegree C (warmer)", "R_cooler": "$R<-0.1$\\textdegree C (cooler)",
      "RP_drier": "$R_P<-2$ pp (drier)", "RP_wetter": "$R_P>2$ pp (wetter)",
      "premium_gt3": "Premium $>$3\\% PV GDP", "STminusM_gt3": "S+T$-$M $>$3 pp"}
body = [f"{cl[r.condition]} & {100 * r.share_end:.1f} & {100 * r.share_non:.1f} & {r.RR_end_vs_non:.2f} "
        f"[{r.RR_ci_lo:.2f}, {r.RR_ci_hi:.2f}] & {100 * r.share_dying:.1f} & {100 * r.share_inst:.1f} & "
        f"{r.RR_dying_vs_inst:.2f} \\\\" for r in rr.itertuples()]
write("tabS_rr", "\n".join(body))

# ---------------------------------------------------------------- S7 disproportionality
d = pd.read_csv(T / "table_disproportionality.csv")
bl = {"residual_warming": "Residual warming", "residual_drying": "Residual drying",
      "termination_premium": "Termination premium", "excess_loss_ST_vs_M": "Excess loss S+T vs M",
      "srm_cooling_bonus": "SAI cooling bonus (benefit)"}
body = []
for b in bl:
    for w in ("w_lang", "w_spk", "w_area"):
        sub = d[(d.burden == b) & (d.weighting == w)].set_index("group")
        cells = []
        for gname in LIVING_ORDER + ["Endangered", "CI_by_speakers", "CI_by_endangerment"]:
            r_ = sub.loc[gname]
            cells.append(f"{r_.DI:.2f}")
        body.append(f"{bl[b]} & {wl[w]} & " + " & ".join(cells) + " \\\\")
write("tabS_DI", "\n".join(body))

# ---------------------------------------------------------------- S8 robustness
rb = pd.read_csv(T / "table_robustness.csv")
body = []
lab_ = {("w_lang", 2100, "main"): "Main (languages, 2100)",
        ("w_lang", 2050, "main"): "Languages, 2050", ("w_spk", 2100, "main"): "Speakers, 2100",
        ("w_area", 2100, "main"): "Territory, 2100",
        ("w_lang", 2100, "no_fallback"): "No nearest-cell fallback",
        ("w_lang", 2100, "with_sign"): "Including sign languages",
        ("w_lang", 2100, "LIC_units"): "LIC units, community EGIDS"}
for key, name in lab_.items():
    sub = rb[(rb.weighting == key[0]) & (rb.horizon == key[1]) & (rb.variant == key[2])].set_index("group")
    e, n = sub.loc["Endangered"], sub.loc["Not endangered"]
    prem_e = e.get("econ_premium_p50", np.nan)
    prem_n = n.get("econ_premium_p50", np.nan)
    body.append(f"{name} & {e.R_mean:+.3f} & {n.R_mean:+.3f} & {100 * e.share_R_warmer:.1f} & "
                f"{100 * n.share_R_warmer:.1f} & {e.RP_median:+.1f} & {n.RP_median:+.1f} & "
                f"{100 * e.share_RP_drier:.1f} & {100 * n.share_RP_drier:.1f} & {fmt(prem_e, 2)} & {fmt(prem_n, 2)} \\\\")
write("tabS_robust", "\n".join(body))

# ---------------------------------------------------------------- S9 perspectives & hazard
pr = pd.read_csv(T / "perspectives.csv")
sens = pd.read_csv(T / "termination_sensitivity.csv")
pl = {"gdp": "per dollar (GDP)", "pop": "per person", "lang": "per language",
      "end": "per endangered language"}
ml = {"premium": "Termination premium", "ST_minus_M": "S+T$-$M", "S_minus_M": "S$-$M",
      "M": "Mitigation loss (M)", "ST": "SAI with termination (S+T)", "H_minus_M": "H$-$M"}
body = []
for m in ("premium", "ST_minus_M", "S_minus_M", "M", "ST"):
    for w in ("gdp", "pop", "lang", "end"):
        r_ = pr[(pr.metric == m) & (pr.perspective == w)].iloc[0]
        br = "--" if not np.isfinite(r_.BR_p50) else f"{r_.BR_p50:.2f} [{r_.BR_p17:.2f}, {r_.BR_p83:.2f}]"
        body.append(f"{ml[m]} & {pl[w]} & {r_.p50:.2f} [{r_.p17:.2f}, {r_.p83:.2f}] & {br} \\\\")
write("tabS_persp", "\n".join(body))
body = []
for hz, hl in (("p005", "0.5\\%"), ("p01", "1\\%"), ("p02", "2\\%")):
    for m in ("premium", "ST_minus_M"):
        cells = []
        for w in ("gdp", "pop", "lang", "end"):
            r_ = sens[(sens.hazard == hz) & (sens.metric == m) & (sens.perspective == w)].iloc[0]
            cells.append(f"{r_.p50:.2f}")
        br = sens[(sens.hazard == hz) & (sens.metric == m) & (sens.perspective == "lang")].BR_p50.iloc[0]
        body.append(f"{hl} & {ml[m]} & " + " & ".join(cells) + f" & {br:.2f} \\\\")
write("tabS_hazard", "\n".join(body))

# ---------------------------------------------------------------- S10 Annex I split
ax = pd.read_csv(T / "table_annexI_split.csv")
body = []
for v in ("annexI=False", "annexI=True"):
    for gname in LIVING_ORDER + ["Endangered", "Not endangered", "All living"]:
        r_ = ax[(ax.variant == v) & (ax.group == gname)].iloc[0]
        body.append(f"{'Annex I' if v.endswith('True') else 'Non-Annex I'} & {gname} & {int(r_.n_languages):,} & "
                    f"{r_.R_mean:+.3f} & {100 * r_.share_R_warmer:.1f} & {r_.RP_median:+.1f} & "
                    f"{100 * r_.share_RP_drier:.1f} & {r_.econ_S_minus_M_p50:+.2f} & "
                    f"{r_.econ_ST_minus_M_p50:.2f} & {r_.econ_premium_p50:.2f} \\\\")
write("tabS_annex", "\n".join(body))

# ---------------------------------------------------------------- S11 latitude bands
lb = pd.read_csv(T / "table_latitude_bands.csv")
body = [f"{r.band} & {int(r.n):,} & {100 * r.share_end:.0f} & {r.R:+.3f} & {r.RP_median:+.1f} & "
        f"{r.premium:.2f} \\\\" for r in lb.itertuples()]
write("tabS_lat", "\n".join(body))

# ---------------------------------------------------------------- S12 scenario lens
sc = pd.read_csv(T / "scenario_cultural_weights.csv")
body = []
for r in sc[sc.kappa.isin([0, 0.5, 1, 2, 4])].itertuples():
    body.append(f"{r.kappa:.2f} & {r.survival_SRM_world} & {r.survival_M_world} & {r.CW_ST_mean:.2f} & "
                f"{r.CW_M_mean:.2f} & {r.CW_diff_mean:.2f} & {100 * r.share_diff_endangered:.0f} \\\\")
write("tabS_scen", "\n".join(body))

# ---------------------------------------------------------------- S13 world totals & equity
body = []
for k, lab2 in (("H", "No SAI (H)"), ("M", "Mitigation (M), emulated"),
                ("M_noT", "Mitigation, time-invariant emulator"), ("S", "SAI, uninterrupted (S)"),
                ("ST", "SAI with termination risk (S+T)"), ("premium", "Termination premium")):
    q = nb[f"econ.world_pct_pvgdp.{k}.p17_p50_p83"]
    body.append(f"{lab2} & {q[1]:.2f} & [{q[0]:.2f}, {q[2]:.2f}] \\\\")
for eta in ("0.0", "1.0", "1.5"):
    for k, lab2 in (("premium", "premium"), ("ST_minus_M", "S+T$-$M")):
        q = nb[f"equity.eta{eta}.{k}.p17_p50_p83"]
        body.append(f"Equity-weighted {lab2}, $\\eta={eta}$ & {q[1]:.2f} & [{q[0]:.2f}, {q[2]:.2f}] \\\\")
write("tabS_world", "\n".join(body))

# ---------------------------------------------------------------- S14 pattern table
ms = pd.read_csv(T / "member_group_stats.csv")
d = ms[ms.horizon == 2100].groupby("pattern").mean(numeric_only=True)
body = [f"{k} & {r.dG:.2f} & {r['R_mean_Institutional']:+.3f} & {r['R_mean_Dying']:+.3f} & "
        f"{r.gap_R_end_minus_non:+.3f} & {r['RP_median_Institutional']:+.1f} & {r['RP_median_Dying']:+.1f} & "
        f"{r.gap_RP_end_minus_non:+.2f} \\\\" for k, r in d.iterrows()]
write("tabS_patterns", "\n".join(body))
print("tables written to", OUT)
