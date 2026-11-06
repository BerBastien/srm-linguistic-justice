"""29 - Supplementary Materials tables for the Science submission (aggregated results only).

Writes, in results/tables/latex/:
  tabS_science_flow.tex        sample accounting: from Ethnologue 28 to the analysis samples
  tabS_science_validation.tex  climate-side validation checks (no economic rows)
  tabS_science_points.tex      main sample vs. unmapped languages placed at their Glottolog points

All values come from the ledger (results/tables/numbers.json) and from public aggregated tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P  # noqa: E402
from srmlang.plotstyle import numbers  # noqa: E402

T = P("results")
OUT = T / "latex"
OUT.mkdir(exist_ok=True)
nb = numbers(T / "numbers.json")


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body + "\n")


def n(key: str) -> str:
    return f"{int(round(nb[key])):,}"


# ---------------------------------------------------------------- sample accounting
excl = nb["heritage.glottolog.n_excluded_by_category"]
n_excl = sum(int(v) for v in excl.values())
flow = [
    ("Languages in Ethnologue 28 (Table of Languages)", n("eth.n_languages_table")),
    ("\\quad living (EGIDS 0--9)", n("v5.sample.n_living")),
    ("\\qquad sign languages (excluded)", n("v5.sample.n_sign")),
    ("\\qquad dormant spoken languages, EGIDS 9 (excluded)", n("v5.sample.n_dormant")),
    ("\\qquad spoken languages with first-language speakers", n("v5.sample.n_spoken_nondormant")),
    ("\\qquad\\quad without a homeland on the land grid", n("v5.sample.n_unmapped")),
    ("\\qquad\\qquad no polygon", n("v5.sample.n_unmapped_no_polygon")),
    ("\\qquad\\qquad polygon more than 150\\,km from the land grid", n("v5.sample.n_unmapped_polygon_off_grid")),
    ("\\qquad\\quad \\textbf{analysis sample}", n("v5.sample.n_sample")),
    ("\\qquad\\qquad non-institutional (EGIDS 5--8b)", n("v5.sample.n_noninst")),
    ("\\qquad\\qquad institutional (EGIDS 0--4)", n("dry.n_inst")),
    ("\\qquad\\qquad endangered (EGIDS 6b--8b)", n("v5.sample.sample_endangered_n")),
    ("\\midrule Language-in-country polygons in the GIS dataset", n("eth.n_polygons_raw")),
    ("\\quad invalid geometries repaired", n("eth.n_polygons_invalid_repaired")),
    ("\\quad with exact land-cell coverage", n("overlay.n_polys_with_land_coverage")),
    ("\\quad assigned the nearest land cell ($\\leq$150\\,km)", n("overlay.n_polys_nearest_cell_fallback")),
    ("\\quad no land cell within 150\\,km", n("overlay.n_polys_dropped_no_land")),
    ("\\midrule Lineage sample (Glottolog 5.3): languages", n("heritage.glottolog.n_languages_in_lineage_sample")),
    ("\\quad excluded (bookkeeping, unattested, unclassifiable, pidgin, mixed)", f"{n_excl:,}"),
    ("\\quad lineages", n("heritage.lineage.n")),
    ("\\qquad families", n("heritage.lineage.n_families")),
    ("\\qquad isolates", n("heritage.lineage.isolates_n")),
    ("\\midrule Language-in-country communities in the alluvial diagram", n("sankey.n_lic_polygons")),
    ("Unmapped languages placed at their Glottolog point (text S1)", n("points.n_point_languages_in_sample")),
]
write("tabS_science_flow", "\n".join(f"{a} & {b} \\\\" for a, b in flow))

# ---------------------------------------------------------------- validation (climate only)
val = pd.read_csv(T / "climrisk_validation.csv")
rows = []
for r in val.itertuples():
    lab = (r.check.replace("|", "$|$").replace("E[T_M]", "E[$T_M$]").replace(" T_H ", " $T_H$ ")
           .replace("E[R]", "E[$R$]").replace("beta", "$\\beta$").replace("gamma", "$\\gamma$")
           .replace("pi(", "$\\pi$("))
    rows.append(f"{lab} & {r.value:.3g} \\\\")
rows.append(f"Member mean of $P_S-P_M$ vs.\\ stored $R_P$, max $|$difference$|$ (pp) & "
            f"{nb['dry.check_member_PS_minus_PM_vs_RP_maxabs']:.2g} \\\\")
write("tabS_science_validation", "\n".join(rows))

# ---------------------------------------------------------------- point-language robustness
if not (T / "v5_point_languages.csv").exists():  # written by `make points` (script 28)
    print("wrote tabS_science_flow, tabS_science_validation; run `make points` for tabS_science_points")
    sys.exit(0)
pts = pd.read_csv(T / "v5_point_languages.csv").set_index("quantity")


def pc(q):
    return f"{100 * pts.loc[q, 'main']:.0f} & {100 * pts.loc[q, 'with_point_languages']:.0f}"


def ct(q):
    return f"{int(pts.loc[q, 'main']):,} & {int(pts.loc[q, 'with_point_languages']):,}"


def rt(q):
    return f"{pts.loc[q, 'main']:.1f} & {pts.loc[q, 'with_point_languages']:.1f}"


def rg(q):
    a = (100 * pts.loc[q, "main"], 100 * pts.loc[f"{q}_p17", "main"], 100 * pts.loc[f"{q}_p83", "main"])
    b = (100 * pts.loc[q, "with_point_languages"], 100 * pts.loc[f"{q}_p17", "with_point_languages"],
         100 * pts.loc[f"{q}_p83", "with_point_languages"])
    return f"{a[0]:.0f} ({a[1]:.0f}--{a[2]:.0f}) & {b[0]:.0f} ({b[1]:.0f}--{b[2]:.0f})"


body = [
    f"Languages in the sample & {ct('languages')} \\\\",
    f"\\quad non-institutional & {ct('noninst_languages')} \\\\",
    f"Non-institutional languages: drier (\\%) & {pc('noninst_drier')} \\\\",
    f"\\quad wetter (\\%) & {pc('noninst_wetter')} \\\\",
    f"\\quad times as much drier as wetter (land: 2.1) & {rt('noninst_dry_to_wet')} \\\\",
    f"\\quad members more often drier than land & {ct('noninst_members_gt_land')} \\\\",
    f"Linguistic diversity hotspots: drier (\\%) & {pc('hotspots_drier')} \\\\",
    f"People: drier (\\%) & {pc('people_drier')} \\\\",
    f"Non-institutional languages between 15\\textdegree S and 15\\textdegree N (\\%) & {pc('deep_tropics_share_languages')} \\\\",
    "\\midrule",
    f"Languages in the lineage sample & {ct('lineage_languages')} \\\\",
    f"Lineages & {ct('lineages')} \\\\",
    f"\\quad families & {ct('families')} \\\\",
    f"\\quad isolates & {ct('isolates')} \\\\",
    f"Lineages: drier (\\%) & {pc('lineages_drier')} \\\\",
    f"\\quad wetter (\\%) & {pc('lineages_wetter')} \\\\",
    f"\\quad times as much drier as wetter & {rt('lineages_dry_to_wet')} \\\\",
    f"\\quad mostly drier (\\%; members p17--p83) & {rg('lineages_mostly_drier')} \\\\",
    f"\\quad isolates mostly drier (\\%) & {pc('isolates_mostly_drier')} \\\\",
    f"\\quad members more often drier than land & {ct('lineages_members_gt_land')} \\\\",
    f"\\quad members more often drier than people & {ct('lineages_members_gt_people')} \\\\",
    f"\\quad members more often drier than the economy & {ct('lineages_members_gt_economy')} \\\\",
    f"Distinctiveness-weighted share drier, all languages (\\%) & {pc('ED_weighted_share_drier')} \\\\",
]
write("tabS_science_points", "\n".join(body))
print("wrote tabS_science_flow, tabS_science_validation, tabS_science_points")
