"""02a - Ethnologue 28 (ISO 639-3) -> Glottolog 5.3 crosswalk.

Every language in Ethnologue's Table_of_Languages is matched by its ISO 639-3 code to a
Glottolog languoid at the level "language":
  1. a language-level languoid carrying the code;
  2. otherwise a dialect-level languoid carrying the code, climbing to its language
     (Glottolog's Language_ID);
  3. manual cases: ylb (Yaleba) -> tawa1275 (Tawala; Glottolog lists Yaleba as a dialect of
     Tawala without an ISO code on the dialect).
ISO codes are read with keep_default_na=False because "nan" is a valid ISO 639-3 code.

For each match we keep the glottocode, Glottolog's category (Spoken_L1_Language, Bookkeeping,
Unattested, Unclassifiable, Pidgin, Mixed_Language, Sign_Language, ...), the top-level family
(Family_ID; empty for isolates), the lineage (= Family_ID, or the language's own glottocode if
it is an isolate), the classification path (ancestor glottocodes), the agglomerated
endangerment status (AES) and the point coordinates (used for languages without an
Ethnologue polygon, see config add_point_languages).

Glottolog 5.3 CLDF tables (CC BY 4.0) are in data/external/glottolog (SHA256SUMS there):
  https://raw.githubusercontent.com/glottolog/glottolog-cldf/v5.3/cldf/{languages,values}.csv

Output (RESTRICTED: joined to Ethnologue attributes): data/interim/glottolog_crosswalk.parquet
Public: results/tables/glottolog_lineage_names.csv (glottocode -> name, no Ethnologue data)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import P, ROOT, ensure_dirs, record  # noqa: E402

ensure_dirs()
G = ROOT / "data" / "external" / "glottolog"
MANUAL = {"ylb": "tawa1275"}

g = pd.read_csv(G / "languages.csv", dtype=str, keep_default_na=False, na_values=[""])
v = pd.read_csv(G / "values.csv", dtype=str, keep_default_na=False, na_values=[""],
                usecols=["Language_ID", "Parameter_ID", "Value", "Code_ID"])
for par, col in [("category", "Value"), ("aes", "Code_ID"), ("classification", "Value")]:
    s = v[v.Parameter_ID == par].drop_duplicates("Language_ID").set_index("Language_ID")[col]
    g[par] = g.ID.map(s)
gi = g.set_index("ID")
names = gi.Name
lang_level = g[g.Level == "language"]
iso_lang = lang_level[lang_level.ISO639P3code.notna()].drop_duplicates("ISO639P3code").set_index("ISO639P3code")
dial = g[(g.Level == "dialect") & g.ISO639P3code.notna()].drop_duplicates("ISO639P3code").set_index("ISO639P3code")

rd = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
eth = pd.read_csv(P("eth_languages"), **rd)
eth = eth.rename(columns={"ISO_639": "ISO"})
eth["ISO"] = eth.ISO.astype(str).str.strip().str.lower()


def match(iso: str):
    if iso in MANUAL:
        return MANUAL[iso], "manual"
    if iso in iso_lang.index:
        return iso_lang.loc[iso, "ID"], "language"
    if iso in dial.index:
        lid = dial.loc[iso, "Language_ID"]
        return (lid if isinstance(lid, str) else dial.loc[iso, "ID"]), "dialect->language"
    return None, None


rows = []
for iso in eth.ISO:
    gc, how = match(iso)
    if gc is None:
        rows.append({"ISO": iso})
        continue
    r = gi.loc[gc]
    fam = r.Family_ID if isinstance(r.Family_ID, str) else None
    rows.append({"ISO": iso, "glottocode": gc, "g_match": how, "g_name": r.Name, "g_level": r.Level,
                 "g_category": r.category, "g_family_id": fam,
                 "g_family_name": names.get(fam) if fam else None,
                 "g_is_isolate": fam is None,
                 "g_lineage": fam if fam else gc,
                 "g_classification": r.classification if isinstance(r.classification, str) else "",
                 "g_aes": r.aes, "g_macroarea": r.Macroarea,
                 "g_lat": pd.to_numeric(r.Latitude, errors="coerce"),
                 "g_lon": pd.to_numeric(r.Longitude, errors="coerce")})
cw = pd.DataFrame(rows)
cw["g_lineage_name"] = cw.g_lineage.map(names)
cw = eth[["ISO", "Language_Name", "Country_Code", "Area", "Family", "Classification", "EGIDS",
          "Latitude", "Longitude"]].merge(cw, on="ISO", how="left")
cw.to_parquet(P("interim") / "glottolog_crosswalk.parquet", index=False)

living = cw.EGIDS != "x10"
record("glottolog.version", "5.3")
record("glottolog.n_eth_languages_matched", int(cw.glottocode.notna().sum()))
record("glottolog.n_eth_living_matched", int((cw.glottocode.notna() & living).sum()))
record("glottolog.n_eth_living", int(living.sum()))
record("glottolog.n_matched_via_dialect", int((cw.g_match == "dialect->language").sum()))
record("glottolog.n_matched_manual", int((cw.g_match == "manual").sum()))
# Glottolog worldwide: top-level families and isolates among spoken L1 languages
sl1 = g[(g.Level == "language") & (g.category == "Spoken_L1_Language")]
record("glottolog.world.n_families_spoken_l1", int(sl1.Family_ID.dropna().nunique()))
record("glottolog.world.n_isolates_spoken_l1", int(sl1.Family_ID.isna().sum()))
# names of ALL Glottolog top-level lineages (families and language-level isolates), from Glottolog alone:
# the public lookup table carries no information about which languages Ethnologue maps
lineage_ids = sorted(set(g.Family_ID.dropna()) | set(g.loc[(g.Level == "language") & g.Family_ID.isna(), "ID"]))
pd.DataFrame({"glottocode": lineage_ids, "name": names.reindex(lineage_ids).values}).to_csv(
    P("results") / "glottolog_lineage_names.csv", index=False)
print(cw.g_match.value_counts(dropna=False))
print(cw[living].g_category.value_counts(dropna=False))
