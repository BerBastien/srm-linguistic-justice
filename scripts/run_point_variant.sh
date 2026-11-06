#!/usr/bin/env bash
# v5 robustness: rerun the analysis with languages that have no Ethnologue polygon added at their
# Glottolog point (config add_point_languages: true), in a separate copy of the repository, then copy
# the key results into the main ledger (script 28). Run after `make all`.  Usage: make points
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# the copy holds licensed (restricted) data: keep it outside the repository and the project folder
V="${POINTS_DIR:-${TMPDIR:-/tmp}/srm_points_variant}"
mkdir -p "$V/data/interim" "$V/results/tables" "$V/results/figures"
cp -r "$ROOT/scripts" "$ROOT/src" "$ROOT/config" "$ROOT/Makefile" "$V/"
cp -r "$ROOT/data/external" "$V/data/"
cp -a "$ROOT/data/raw" "$V/data/"
cp "$ROOT/data/interim/grid_fields.parquet" "$ROOT/data/interim/country_member_fields.npz" \
   "$ROOT/data/interim/country_member_pv.npz" "$V/data/interim/"
sed -i 's/add_point_languages: false/add_point_languages: true /' "$V/config/config.yaml"
rm -f "$V/results/tables/numbers.json"
cd "$V"
for s in 02a_glottolog_crosswalk 02_prepare_ethnologue 03_overlay 04_member_propagation 05_economics 06_metrics \
         07_perspectives 08_scenarios 15_pixel_economics 16_within_country_tests 17_drought_metrics 19_drought_table \
         21_linguistic_heritage 23_baselines_rain_heat 25_v3_figures_tables 27_v5_classification_sample; do
  echo "=== $s"; python "scripts/$s.py" > "log_$s.txt" 2>&1
done
cd "$ROOT"
python scripts/28_point_languages_comparison.py "$V"
