"""16 - Within-country tests with pixel-level damages (go / no-go for a stronger paper).

PRE-REGISTERED DECISION RULE (written 27 Oct 2026, before any within-country estimate was run)
-----------------------------------------------------------------------------------------
Main outcome: termination premium of SAI (local-shock KU allocation), % of local PV GDP.
Secondary outcome: benefit of uninterrupted SAI over no SAI, B = I_H - I_S (pp).
Main specification: p50 cell map, economy-weighted language intensity, language-weighted.

T1  Languages. Within-country difference, endangered (In trouble + Dying) minus not endangered,
    language-in-country polygons as units, country fixed effects, each language weight 1
    (split equally across its polygons).
T2  Places. Within-country association between a cell's language richness
    (log2(1 + number of living languages present)) and its premium intensity,
    country fixed effects, cells weighted by area. Effect = slope x (p90 - p10) of the
    within-country-demeaned richness.

A finding counts as ROBUST only if ALL hold:
  (i)   same sign in the p17, p50 and p83 cell maps, under both termination allocations
        (local, restricted) and both weightings (economy, territory; T2: area, GDP);
  (ii)  the 95% country-clustered bootstrap CI of the main specification excludes zero;
  (iii) magnitude >= 10% of the language-weighted (T1) or area-weighted (T2) mean premium.
Reported regardless of outcome: all specifications, the share of variance that lies within
countries, and a sensitivity excluding polygons whose cells were mostly imputed.
-----------------------------------------------------------------------------------------
Outputs: results/tables/pixel_within_country_tests.csv (aggregated), numbers.json entries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, record  # noqa: E402

I = P("interim")
T = P("results")
B = 1000
rng = np.random.default_rng(CFG["analysis"]["random_seed"])
Q = ["p17", "p50", "p83"]


def fe_slope(y, x, w, g):
    """Weighted within-group (fixed-effects) slope of y on x."""
    d = pd.DataFrame({"y": y, "x": x, "w": w, "g": g}).dropna()
    gm = d.groupby("g")
    wsum = gm.w.transform("sum")
    yd = d.y - (d.y * d.w).groupby(d.g).transform("sum") / wsum
    xd = d.x - (d.x * d.w).groupby(d.g).transform("sum") / wsum
    den = (d.w * xd ** 2).sum()
    return np.nan if den == 0 else (d.w * xd * yd).sum() / den


def cluster_boot(fn, groups, n=B):
    ug = np.unique(groups)
    idx_by = {k: np.flatnonzero(groups == k) for k in ug}
    out = np.empty(n)
    for b in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        rows, newg = [], []
        for j, k in enumerate(pick):
            rows.append(idx_by[k])
            newg.append(np.full(len(idx_by[k]), j))
        out[b] = fn(np.concatenate(rows), np.concatenate(newg))
    return np.percentile(out, [2.5, 97.5])


def within_share(y, w, g):
    d = pd.DataFrame({"y": y, "w": w, "g": g}).dropna()
    mu = np.average(d.y, weights=d.w)
    gm = (d.y * d.w).groupby(d.g).transform("sum") / d.w.groupby(d.g).transform("sum")
    tot = np.average((d.y - mu) ** 2, weights=d.w)
    return np.average((d.y - gm) ** 2, weights=d.w) / tot


rows = []

# ------------------------------------------------------------------ T1 languages
pp = pd.read_parquet(I / "pixel_poly.parquet").reset_index()
pp = pp[pp.vitality.isin(["Institutional", "Stable", "In trouble", "Dying"])].copy()
pp["e"] = pp.vitality.isin(["In trouble", "Dying"]).astype(float)
pp["e_lic"] = pp.vitality_LIC.isin(["In trouble", "Dying"]).astype(float)
pp["w"] = 1.0 / pp.groupby("ISO").pid.transform("size")
g1 = pp.CTRY3.to_numpy()

mean_main = np.average(pp.I_Ploc_p50, weights=pp.w)
record("pixel.T1.mean_premium_lang_weighted", float(mean_main))
record("pixel.T1.within_country_variance_share_premium",
       float(within_share(pp.I_Ploc_p50, pp.w, g1)))
record("pixel.T1.n_polygons", len(pp))
record("pixel.T1.n_countries_with_both_groups",
       int((pp.groupby("CTRY3").e.agg(["min", "max"]).pipe(lambda d: (d["min"] == 0) & (d["max"] == 1))).sum()))

specs = []
for out_, lab in [("Ploc", "premium_local"), ("Pres", "premium_restricted"), ("B", "benefit")]:
    for wt, pre in [("economy", "I_"), ("territory", "Iterr_")]:
        for q in Q:
            specs.append((lab, wt, q, f"{pre}{out_}_{q}"))
for lab, wt, q, col in specs:
    b = fe_slope(pp[col], pp.e, pp.w, g1)
    ci = [np.nan, np.nan]
    if q == "p50":
        y, x, w = pp[col].to_numpy(), pp.e.to_numpy(), pp.w.to_numpy()
        ci = cluster_boot(lambda r, gg: fe_slope(y[r], x[r], w[r], gg), g1)
    rows.append({"test": "T1_endangered_within_country", "outcome": lab, "weighting": wt, "map": q,
                 "estimate": b, "ci_lo": ci[0], "ci_hi": ci[1],
                 "raw_gap": np.average(pp[col][pp.e == 1], weights=pp.w[pp.e == 1])
                 - np.average(pp[col][pp.e == 0], weights=pp.w[pp.e == 0]),
                 "rel_to_mean": b / np.average(pp[col], weights=pp.w)})

# sensitivities on the main specification
sens = {
    "exclude_mostly_imputed": pp.share_imputed <= 0.5,
    "all": pp.share_imputed >= 0,
}
for name, m in sens.items():
    d = pp[m]
    rows.append({"test": f"T1_sens_{name}", "outcome": "premium_local", "weighting": "economy", "map": "p50",
                 "estimate": fe_slope(d.I_Ploc_p50, d.e, d.w, d.CTRY3)})
rows.append({"test": "T1_sens_LIC_level_vitality", "outcome": "premium_local", "weighting": "economy",
             "map": "p50", "estimate": fe_slope(pp.I_Ploc_p50, pp.e_lic, pp.w, g1)})
rows.append({"test": "T1_sens_egids_slope_per_step", "outcome": "premium_local", "weighting": "economy",
             "map": "p50", "estimate": fe_slope(pp.I_Ploc_p50, pp.egids_rank, pp.w, g1)})
rows.append({"test": "T1_sens_non_annexI_only", "outcome": "premium_local", "weighting": "economy",
             "map": "p50", "estimate": fe_slope(pp.I_Ploc_p50[~pp.annex_i], pp.e[~pp.annex_i],
                                                  pp.w[~pp.annex_i], g1[~pp.annex_i.to_numpy()])})
# local minus national (descriptive)
nat = pd.read_csv(T / "pixel_country_intensity.csv", index_col="iso3")
pp["excess"] = pp.I_Ploc_p50 - pp.CTRY3.map(nat.I_Ploc_p50)
for e_, lab in [(1, "endangered"), (0, "not_endangered")]:
    m = (pp.e == e_) & pp.excess.notna()
    record(f"pixel.T1.local_minus_national_premium.{lab}", float(np.average(pp.excess[m], weights=pp.w[m])))

# ------------------------------------------------------------------ T2 places
cells = pd.read_parquet(I / "pixel_cells.parquet")
pe = pd.read_parquet(I / "poly_exposure.parquet")
lm = pd.read_parquet(I / "lang_metrics.parquet")
pe = pe[pe.ISO.isin(lm.ISO) & pe.land_w_km2.notna() & ~pe.sign_language]
w = pd.read_parquet(I / "poly_cell_weights.parquet")
w = w[w.pid.isin(pe.pid) & (w.coverage.fillna(1) > 0)]
w["ISO"] = w.pid.map(pe.set_index("pid").ISO)
w["end"] = w.pid.map(pe.set_index("pid").vitality.isin(["In trouble", "Dying"]))
rich = w.groupby("grid_id").ISO.nunique()
rich_e = w[w.end].groupby("grid_id").ISO.nunique()
cells["rich"] = cells.grid_id.map(rich).fillna(0)
cells["rich_e"] = cells.grid_id.map(rich_e).fillna(0)
cells["x"] = np.log2(1 + cells.rich)
cells["xe"] = np.log2(1 + cells.rich_e)
cells = cells[cells.iso3.notna()].copy()
g2 = cells.iso3.to_numpy()


def demeaned_spread(x, wt, g):
    d = pd.DataFrame({"x": x, "w": wt, "g": g})
    xd = d.x - (d.x * d.w).groupby(d.g).transform("sum") / d.w.groupby(d.g).transform("sum")
    o = np.argsort(xd.to_numpy())
    cw = np.cumsum(d.w.to_numpy()[o]) / d.w.sum()
    xs = xd.to_numpy()[o]
    return xs[np.searchsorted(cw, 0.9)] - xs[np.searchsorted(cw, 0.1)]


record("pixel.T2.n_cells", len(cells))
record("pixel.T2.mean_premium_area_weighted", float(np.average(cells.I_Ploc_p50, weights=cells.area_km2)))
record("pixel.T2.within_country_variance_share_premium",
       float(within_share(cells.I_Ploc_p50, cells.area_km2, g2)))
record("pixel.T2.within_country_variance_share_benefit",
       float(within_share(cells.I_B_p50, cells.area_km2, g2)))
for xvar, xl in [("x", "richness_all"), ("xe", "richness_endangered")]:
    for out_, lab in [("Ploc", "premium_local"), ("Pres", "premium_restricted"), ("B", "benefit")]:
        for wt, wcol in [("area", "area_km2"), ("gdp", "gdp")]:
            spread = demeaned_spread(cells[xvar], cells[wcol], g2)
            for q in Q:
                col = f"I_{out_}_{q}"
                b = fe_slope(cells[col], cells[xvar], cells[wcol], g2)
                ci = [np.nan, np.nan]
                if q == "p50" and wt == "area":
                    y, x, ww = cells[col].to_numpy(), cells[xvar].to_numpy(), cells[wcol].to_numpy()
                    ci = np.array(cluster_boot(lambda r, gg: fe_slope(y[r], x[r], ww[r], gg), g2)) * spread
                rows.append({"test": f"T2_{xl}_within_country", "outcome": lab, "weighting": wt, "map": q,
                             "estimate": b * spread, "ci_lo": ci[0], "ci_hi": ci[1],
                             "rel_to_mean": b * spread / np.average(cells[col], weights=cells[wcol]),
                             "slope_per_doubling": b, "p90_p10_spread": spread})

res = pd.DataFrame(rows)
res.to_csv(T / "pixel_within_country_tests.csv", index=False)


# ------------------------------------------------------------------ verdicts
def verdict(test, outcome_main="premium_local"):
    d = res[res.test == test]
    prem = d[d.outcome.isin(["premium_local", "premium_restricted"])]
    signs = np.sign(prem.estimate)
    same_sign = bool((signs == signs.iloc[0]).all())
    main = d[(d.outcome == outcome_main) & (d["map"] == "p50") &
             (d.weighting == ("economy" if test.startswith("T1") else "area"))].iloc[0]
    ci_ok = bool((main.ci_lo > 0) or (main.ci_hi < 0))
    mag_ok = bool(abs(main.rel_to_mean) >= 0.10)
    return {"estimate": float(main.estimate), "ci": [float(main.ci_lo), float(main.ci_hi)],
            "rel_to_mean": float(main.rel_to_mean), "same_sign_all_specs": same_sign,
            "ci_excludes_zero": ci_ok, "magnitude_ge_10pct": mag_ok,
            "ROBUST": same_sign and ci_ok and mag_ok}


for t in ["T1_endangered_within_country", "T2_richness_all_within_country",
          "T2_richness_endangered_within_country"]:
    v = verdict(t)
    for k, val in v.items():
        record(f"pixel.verdict.{t}.{k}", val)
    print(t, v)
print(res[res["map"] == "p50"].round(4).to_string())
