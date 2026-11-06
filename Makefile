# Reproduce every result, table and figure.  Usage:  make all   (about 10 minutes)
PY ?= python
S = scripts

# v5 robustness (languages without a polygon at their Glottolog point): make points  (after make all)
.PHONY: points all check check-public climrisk ethnologue overlay members econ metrics perspectives scenarios figures si tables rain heritage temperature test clean

all: check climrisk ethnologue overlay members econ metrics perspectives scenarios figures si tables rain heritage temperature

check:
	$(PY) $(S)/00_check_inputs.py
# run before every git push: fails if licensed or restricted data are in the repository
check-public:
	$(PY) tools/check_no_restricted_data.py
climrisk:
	$(PY) $(S)/01_extract_climrisk.py
ethnologue:
	$(PY) $(S)/02a_glottolog_crosswalk.py
	$(PY) $(S)/02_prepare_ethnologue.py
overlay:
	$(PY) $(S)/03_overlay.py
members:
	$(PY) $(S)/04_member_propagation.py
econ:
	$(PY) $(S)/05_economics.py
metrics:
	$(PY) $(S)/06_metrics.py
perspectives:
	$(PY) $(S)/07_perspectives.py
scenarios:
	$(PY) $(S)/08_scenarios.py
figures:
	$(PY) $(S)/09_fig1_map.py
	$(PY) $(S)/10_fig2_gradients.py
	$(PY) $(S)/11_fig3_sankey.py
	$(PY) $(S)/11_fig3_sankey.py L1
	$(PY) $(S)/12_fig4_risk.py
si:
	$(PY) $(S)/13_si_figures.py
tables:
	$(PY) $(S)/14_si_tables.py
rain:
	$(PY) $(S)/15_pixel_economics.py
	$(PY) $(S)/16_within_country_tests.py
	$(PY) $(S)/17_drought_metrics.py
	$(PY) $(S)/18_drought_figures.py
	$(PY) $(S)/19_drought_table.py
	$(PY) $(S)/20_drought_si.py
heritage:
	$(PY) $(S)/21_linguistic_heritage.py
	$(PY) $(S)/22_heritage_figure.py
	$(PY) $(S)/23_baselines_rain_heat.py
	$(PY) $(S)/24_statement_figures.py
	$(PY) $(S)/25_v3_figures_tables.py
	$(PY) $(S)/27_v5_classification_sample.py
	$(PY) $(S)/29_science_sm_tables.py
# exploratory, not in the submitted paper: temperature under the rainfall baselines (docs/temperature_baselines.md)
temperature:
	$(PY) $(S)/26_temperature_baselines.py
points:
	bash $(S)/run_point_variant.sh
	$(PY) $(S)/29_science_sm_tables.py
test:
	$(PY) -m pytest -q tests
clean:
	find data/interim -mindepth 1 ! -name README.md -delete
	rm -rf results/tables/*.csv results/tables/numbers.json results/tables/latex results/figures/*
