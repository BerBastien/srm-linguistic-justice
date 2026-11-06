# srm-linguistic-justice

Code for the article **"Solar geoengineering shifts rainfall loss onto linguistic diversity"**
(Bastien-Olvera, Estrada & Zariquiey, submitted to *Science*, November 2026).

**Question.** If the world reaches the same global temperature either by cutting emissions or by
keeping emissions high and adding stratospheric aerosol injection (SAI), does SAI's rainfall
penalty fall more heavily on the homelands of the world's non-institutional languages (EGIDS 5–8b,
sustained mainly by families and communities rather than by institutions) than on land, people or
the economy?

**Short answer (2100, 60-member ensemble, ensemble-mean climate unless stated).** Four times as much
non-institutional-language homeland loses rain as gains it; for land, people and the economy, about twice as much.

| Unit | Share >2 pp drier | Share >2 pp wetter | Drier : wetter | More often drier than land in |
|---|---|---|---|---|
| Land (area) | 55% | 27% | 2.1 | – |
| People (first-language speakers) | 53% | 29% | 1.8 | 14 / 60 members |
| Economy (GDP) | 51% | 25% | 2.0 | 23 / 60 members |
| Non-institutional languages (each language once) | 65% | 16% | 4.0 | 50 / 60 members |
| Language lineages (Glottolog 5.3; each of 278 lineages once) | 71% | 13% | 5.4 | 60 / 60 members |

* 72% of the 278 Glottolog lineages (183 families, 95 isolates) have most of their homeland drier
  (54–90% between the least and most dry members). Ethnologue's classification, genealogical units
  only, gives 70%.
* Half of all non-institutional-language diversity (weighted endemism) lies on 1.3% of land; 69% of it is drier.
* 68% of non-institutional languages lie between 15°S and 15°N, where the deficit is concentrated (35% of people).
* Temperature gives no counterweight: 6% of non-institutional-language homeland is >0.1 °C warmer than under
  mitigation (people 8%, land 22%).
* Sample: 5,936 of Ethnologue 28's 7,159 living languages (158 sign languages, 337 dormant and 728 unmapped
  spoken languages left out; Ethnologue 28 has no polygons for Australia). Adding the unmapped languages at
  their Glottolog points (`make points`) gives 62% drier and a ratio of 3.3 for non-institutional languages,
  and 68% of 315 lineages mostly drier.

Annual rainfall is one of several variables that shape water availability; the results map exposure,
not impacts. The main analysis uses Ethnologue polygons only; the Glottolog-point variant is a robustness
check in the Supplementary Materials (text S1, table S7). Earlier exploratory analyses (temperature by
vitality class, termination-risk economics, scenario lens) remain in the pipeline but are not part of the
submitted paper.

---

## 1. Design in one paragraph

CLIMRISK-SRM provides 60 paired MAGICC7 members of the SCEN7-H scenario with and without SAI
(CESM2-WACCM G6sulfur schedule). Local fields are pattern-scaled: `P_H = π·G_H` (percent change of
rainfall relative to 1900) and `T_H = β·G_H`; the SAI fields `P_S`, `T_S` are provided by CLIMRISK.
For each member we build a mitigation world with the SAI world's global temperature,
`P_M = π·G_S`, `T_M = β·G_S`. The rainfall difference is `R_P = P_S − P_M` (percentage points). We
overlay these fields exactly on 7,482 Ethnologue language-in-country polygons. Every unit (land,
populated land, people, GDP, language homelands, linguistic diversity hotspots, language lineages) is computed
from the same cell-level member fields, so comparisons between units are paired, member by member.
Language homelands use range shares that sum to one per language (weighted endemism). Lineages
follow Glottolog 5.3 (ISO 639-3 crosswalk, script 02a); an approximate topological distinctiveness
(fair proportion) uses Glottolog's classification paths with a conventional common root.

## 2. Repository layout

```
config/                 config.yaml (paths, parameters); input_checksums.json
src/srmlang/            common.py (classes, Annex I list, numbers ledger), stats.py, plotstyle.py,
                        heritage.py (weighted endemism, tree metrics), baselines.py (member fields and units, scripts 23 and 26)
scripts/
  00_check_inputs.py        verify inputs are present and identical to the article version
  01_extract_climrisk.py    global/country/grid fields; iso-GMST mitigation counterfactual (A1/A2)
  02a_glottolog_crosswalk.py Ethnologue ISO 639-3 -> Glottolog 5.3 languages, lineages, AES, points (v5)
  02_prepare_ethnologue.py  repair polygons; join Ethnologue tables; classifications; drop EGIDS 9
  03_overlay.py             exact polygon × 0.5° grid coverage (exactextract); language exposures
  04_member_propagation.py  member- and pattern-level language residuals
  05_economics.py           PV losses; mitigation-damage emulator; termination products
  06_metrics.py             group statistics, gaps, tests, DI, concentration indices, Sankey flows
  07_perspectives.py        per dollar / person / language burdens; hazard sensitivity; responsibility
  08_scenarios.py           scenario lens: language survival and cultural-loss weights (SI)
  09–12_*.py                figures of the first version (SI)
  13_si_figures.py          SI figures S1–S10
  14_si_tables.py           SI tables (LaTeX), first-version analyses
  15_pixel_economics.py     0.5° CLIMRISK present values on language areas
  16_within_country_tests.py pre-registered within-country economic tests
  17_drought_metrics.py     language-level rainfall metrics for non-institutional languages
  18_drought_figures.py     main figures 1–4 (rainfall paper)
  19_drought_table.py       Table 1 (rainfall paper)
  20_drought_si.py          SI tables and agreement map (language-level rainfall)
  21_linguistic_heritage.py weighted endemism, hotspots, homeland size, Glottolog lineages, distinctiveness (ED, PD)
  22_heritage_figure.py     heritage overview figure (exploratory)
  23_baselines_rain_heat.py land / people / economy / linguistic-diversity baselines for rain and temperature
  24_statement_figures.py   Fig. 2 (against what), Fig. 3 (forest: each lineage its own tree), fig. S8 (common base,
                            display only), figs. S9-S10 (temperature)
  25_v3_figures_tables.py   Fig. 1 (map), Fig. 4 (member by member), Table 1 and SM tables
  26_temperature_baselines.py  exploratory: temperature under the same baselines (see docs/temperature_baselines.md)
  27_v5_classification_sample.py  sample accounting; lineage results under five classifications (table S6)
  28_point_languages_comparison.py, run_point_variant.sh  robustness: unmapped languages at Glottolog points
  29_science_sm_tables.py   SM tables: sample flow (S4), validation (S2), point-language robustness (S7)
results/tables/         aggregated outputs (public) + numbers.json (every number in the paper)
results/figures/        PDF and PNG figures
data/raw/               inputs (NOT included; see data/raw/README.md)
data/interim/           intermediate products (RESTRICTED; gitignored)
tests/                  unit tests for the statistics, identities and heritage metrics
```

## 3. Reproducing the results

```bash
conda env create -f environment.yml && conda activate srm-linguistic-justice
#   or: python -m pip install -r requirements.txt
# place the inputs as described in data/raw/README.md, then:
make all      # ~10 minutes on a laptop
make test
make points   # robustness run in a copy outside the repository (~10 minutes)
```

`make all` checks the inputs against `config/input_checksums.json` and regenerates every
table, figure and number. Each number quoted in the article is written to
`results/tables/numbers.json` by the script that computes it.

## 4. Data and licences

| Input | Source | Licence / access |
|---|---|---|
| CLIMRISK-SRM research package (c7 base v2) | CLIMRISK team, ICAyCC-UNAM | To be deposited by its authors; contact the corresponding author |
| Ethnologue 28 Global Dataset and Language GIS Dataset | SIL International | **Licensed; may not be redistributed.** Obtain a licence from Ethnologue |
| Glottolog 5.3 CLDF (`languages.csv`, `values.csv`) | glottolog/glottolog-cldf, tag v5.3 (in `data/external/glottolog`) | CC BY 4.0 |
| Natural Earth 1:50m | naturalearthdata.com | Public domain |
| CO₂ and population | Our World in Data (`owid/co2-data`) | CC BY 4.0 |

**Ethnologue licence.** The licence permits publishing visualizations and aggregated results,
but not raw or language-level data. This repository therefore contains **no** Ethnologue data
and no language-level outputs. Tables of language families list families with at least two mapped
languages; single-language lineages and isolates are pooled or kept in `data/interim/`
(gitignored), which holds the restricted products that the pipeline creates locally for licensed users.

**Without the licensed inputs the analysis does not run.** `make all` stops at `make check`, which
lists the missing Ethnologue and CLIMRISK files, and every analysis script needs them. Only the
figures that are redrawn from the aggregated tables in `results/` can be rendered without them.
The figures that show language locations are raster images; Ethnologue polygon outlines are never
stored as vector paths.

**Before every push**, run `make check-public` (`tools/check_no_restricted_data.py`, standard library
only). It fails if any licensed input, intermediate product, data file, symbolic link or language-level
table is in the files git would commit.

Code: MIT. Aggregated results and figures: CC BY 4.0 (see `LICENSE-results.md`).

## 5. Interpretation safeguards

* Ensemble fractions are empirical frequencies of a fixed 60-member subset, not calibrated
  probabilities.
* The mitigation world is an idealized iso-temperature counterfactual with the same SSP3 GDP.
* The 0.5° maps carry ensemble statistics. Member-level results use a documented hybrid of the
  maps and exact country-level identities (Supplementary Materials, M3).
* The ensemble-mean field is smoother than any member, so its shares and ratios are not typical
  member values; compare units member by member (`results/tables/baselines_by_member.csv`).
* Counting languages or lineages rather than people or dollars gives each a weight independent of its
  size; all weightings are reported side by side.
* Lineages are independent trees. The common root in the distinctiveness measure and the common base in
  the tree figure are conventions, not claims of relatedness.
* Economic burdens are shares of host-country GDP. They are intensities, not community-level
  welfare losses.
* The termination module is a reduced-form risk calculation, not a forecast of governance failure.

## 6. Citation

See `CITATION.cff`. Please also cite Ethnologue (Eberhard, Simons & Fennig 2025) and the
CLIMRISK-SRM report, and Glottolog (Hammarström, Forkel, Haspelmath & Bank 2026).
