"""08 - Scenario lens: socioeconomic pathways, language survival and cultural-loss weights.

This is an ILLUSTRATIVE, assumption-driven extension (reported in the SI). It does not
estimate how climate or SRM changes the probability that a language is lost; no such
elasticity exists in the literature at global scale.

Pairing of climate pathways with socioeconomic narratives (assumption):
  S+T  SRM on top of the CLIMRISK SCEN7-H emissions path: fossil-intensive or
       fragmented development (SSP5-/SSP3-like narratives; CLIMRISK GDP is SSP3)
  M    mitigation to the same global temperature: SSP1-/SSP2-like narratives

Language-survival bounding scenarios to 2100 (by current EGIDS, not forecasts):
  L0  persistence          every living language is still spoken in 2100
  L1  dying languages lost  EGIDS 8a, 8b, 9 lose their last first-language speakers
  L2  endangered lost       EGIDS 6b, 7, 8a, 8b, 9 lose their last L1 speakers

Cultural-loss weight kappa: a community's economic loss counts (1 + kappa) times when
its language is lost (a reduced-form stand-in for the loss of place-based knowledge
and of a community's capacity to adapt in its own terms). kappa is NOT estimated.

Reported:
  * culturally weighted mean loss for each pathway and survival pairing,
  * the share of that weighted loss borne by endangered-language communities,
  * the break-even kappa* at which the weighted burden of endangered-language
    communities equals that of other communities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P, ensure_dirs, record  # noqa: E402

ensure_dirs()
lm = pd.read_parquet(P("interim") / "lang_metrics.parquet")
lm = lm.dropna(subset=["econ_ST", "econ_M"])
vit = lm.vitality
lost = {"L0": np.zeros(len(lm), bool),
        "L1": vit.eq("Dying").to_numpy(),
        "L2": vit.isin(["In trouble", "Dying"]).to_numpy()}
end = vit.isin(["In trouble", "Dying"]).to_numpy()
loss = {"M": lm.econ_M.to_numpy(), "ST": lm.econ_ST.to_numpy(),
        "excess": (lm.econ_ST - lm.econ_M).to_numpy()}

rows = []
for kappa in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    for pair in (("L0", "L0"), ("L1", "L0"), ("L2", "L1"), ("L2", "L0")):
        sS, sM = pair
        cwS = loss["ST"] * (1 + kappa * lost[sS])
        cwM = loss["M"] * (1 + kappa * lost[sM])
        diff = cwS - cwM
        rows.append({"kappa": kappa, "survival_SRM_world": sS, "survival_M_world": sM,
                     "CW_ST_mean": cwS.mean(), "CW_M_mean": cwM.mean(),
                     "CW_diff_mean": diff.mean(),
                     "share_diff_endangered": diff[end].sum() / diff.sum(),
                     "share_languages_endangered": end.mean(),
                     "CW_diff_end_mean": diff[end].mean(), "CW_diff_non_mean": diff[~end].mean()})
sc = pd.DataFrame(rows)
sc.to_csv(P("results") / "scenario_cultural_weights.csv", index=False)

# break-even kappa: (1+k) * mean_end(excess) = mean_non(excess), all endangered lost (L2)
for name in ("excess", "ST"):
    me, mn = loss[name][end].mean(), loss[name][~end].mean()
    kstar = mn / me - 1
    record(f"scenario.kappa_star.{name}", kstar)
    record(f"scenario.mean_end.{name}", me)
    record(f"scenario.mean_non.{name}", mn)
for _, r in sc[(sc.kappa.isin([0, 1])) & (sc.survival_SRM_world == "L2") &
               (sc.survival_M_world == "L1")].iterrows():
    record(f"scenario.k{r.kappa}.L2L1.share_diff_endangered", r.share_diff_endangered)
    record(f"scenario.k{r.kappa}.L2L1.CW_diff_mean", r.CW_diff_mean)
record("scenario.n_L1_lost", int(lost["L1"].sum()))
record("scenario.n_L2_lost", int(lost["L2"].sum()))
print(sc.round(3).to_string())
