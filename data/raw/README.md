# Inputs (not distributed)

Place the inputs exactly as below. File names and paths can be changed in
`config/config.yaml`. Run `python scripts/00_check_inputs.py` to confirm that they are
identical to the versions used in the article (SHA-256 in `config/input_checksums.json`).

```
data/raw/
├── climrisk/
│   ├── CLIMRISK_SRM_SCEN7H_N60_LANGUAGE_READY.h5          (package data/)
│   ├── grid_0p5_country_region_crosswalk.csv               (package tables/)
│   ├── country_region_crosswalk.csv                        (package tables/)
│   ├── economic_pv_country_summary.csv                     (package tables/, optional)
│   ├── CLIMRISK_SRM_termination_country_PV_summary.csv     (package tables/, optional)
│   ├── termination_country_PV_by_member_p01.csv            (= tables/CLIMRISK_SRM_termination_country_PV_by_member.csv)
│   ├── termination_country_PV_by_member_p005.csv           (= tables/termination_sensitivities/p005/...)
│   └── termination_country_PV_by_member_p02.csv            (= tables/termination_sensitivities/p02/...)
└── ethnologue/
    ├── Table_of_Languages.tab                              (Ethnologue 28 Global Dataset)
    ├── Table_of_LICs.tab
    ├── Table_of_Countries.tab
    └── E28_lang_polys.gdb/                                  (Ethnologue 28 Language GIS Dataset; unzip)
```

Public data in `data/external/` (downloaded once):

```
naturalearth/ne_50m_land.geojson, ne_50m_coastline.geojson, ne_50m_admin_0_countries.geojson
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/<name>.geojson
owid/owid-co2-data.csv
  https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv
```

**Ethnologue licence.** You need your own licence (https://www.ethnologue.com). Do not
commit these files. The `.gitignore` excludes everything in this folder except this README.
