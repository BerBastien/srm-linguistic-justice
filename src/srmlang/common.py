"""Shared configuration, classification tables and a ledger of reported numbers.

Every number quoted in the manuscript is written to ``results/tables/numbers.json``
by the script that computes it (``record(key, value)``). The manuscript and the
lab notebook read from that ledger, so text and code cannot drift apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


CFG = load_config()


def P(key: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    p = Path(CFG["paths"][key])
    return p if p.is_absolute() else ROOT / p


def ensure_dirs() -> None:
    for k in ("interim", "results", "figures"):
        P(k).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ethnologue vitality classification
# ---------------------------------------------------------------------------
# EGIDS = Expanded Graded Intergenerational Disruption Scale (Lewis & Simons 2010).
# Ethnologue codes level 10 (extinct) as "x10" so it sorts last.
EGIDS_ORDER = ["0", "1", "2", "3", "4", "5", "6a", "6b", "7", "8a", "8b", "9", "x10"]
EGIDS_RANK = {e: i for i, e in enumerate(EGIDS_ORDER)}  # 0 = strongest

EGIDS_LABEL = {
    "0": "International", "1": "National", "2": "Provincial", "3": "Wider communication",
    "4": "Educational", "5": "Developing", "6a": "Vigorous", "6b": "Threatened",
    "7": "Shifting", "8a": "Moribund", "8b": "Nearly extinct", "9": "Dormant",
    "x10": "Extinct",
}

# Five summary classes of EGIDS levels. Institutional (0-4) follows Ethnologue; the grouping of the
# other levels into Stable (5, 6a), In trouble (6b, 7) and Dying (8a, 8b, 9) is ours (v5; Ethnologue 28
# itself groups 6b-9 as Endangered). EGIDS 9 (dormant) is excluded from the analysed sample.
VITALITY = {
    "0": "Institutional", "1": "Institutional", "2": "Institutional", "3": "Institutional",
    "4": "Institutional", "5": "Stable", "6a": "Stable", "6b": "In trouble", "7": "In trouble",
    "8a": "Dying", "8b": "Dying", "9": "Dying", "x10": "Extinct",
}
VITALITY_ORDER = ["Institutional", "Stable", "In trouble", "Dying", "Extinct"]
LIVING_ORDER = VITALITY_ORDER[:4]


def vitality(egids) -> str | None:
    if egids is None or (isinstance(egids, float) and np.isnan(egids)):
        return None
    return VITALITY.get(str(egids).strip())


def endangered(egids) -> bool | None:
    """Ethnologue 'endangered' = In trouble (6b, 7) + Dying (8a, 8b, 9)."""
    v = vitality(egids)
    if v is None:
        return None
    if v == "Extinct":
        return None
    return v in ("In trouble", "Dying")


# ---------------------------------------------------------------------------
# Global North / South proxy: UNFCCC Annex I parties (ISO 3166 alpha-2).
# The European Union is a party in its own right but has no territory code.
# ---------------------------------------------------------------------------
ANNEX_I_ISO2 = {
    "AU", "AT", "BY", "BE", "BG", "CA", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "GR", "HU", "IS", "IE", "IT", "JP", "LV", "LI", "LT", "LU", "MT", "MC", "NL", "NZ",
    "NO", "PL", "PT", "RO", "RU", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "GB", "US",
}

# Dependent territories of Annex I parties whose languages Ethnologue lists under
# their own ISO code. They are assigned to the administering party for the
# Annex I proxy (governance, not geography), and flagged so it can be undone.
TERRITORY_OF = {
    "GL": "DK", "FO": "DK", "PF": "FR", "NC": "FR", "WF": "FR", "GF": "FR", "GP": "FR",
    "MQ": "FR", "RE": "FR", "YT": "FR", "PM": "FR", "BL": "FR", "MF": "FR",
    "AS": "US", "GU": "US", "MP": "US", "PR": "US", "VI": "US", "UM": "US",
    "CK": "NZ", "NU": "NZ", "TK": "NZ", "NF": "AU", "CX": "AU", "CC": "AU",
    "AW": "NL", "CW": "NL", "SX": "NL", "BQ": "NL", "SJ": "NO",
    "GI": "GB", "FK": "GB", "BM": "GB", "KY": "GB", "VG": "GB", "MS": "GB", "TC": "GB",
    "AI": "GB", "SH": "GB", "PN": "GB", "IM": "GB", "JE": "GB", "GG": "GB", "AX": "FI",
}


def annex_i(iso2: str, include_territories: bool = True) -> bool:
    iso2 = (iso2 or "").upper()
    if iso2 in ANNEX_I_ISO2:
        return True
    if include_territories and TERRITORY_OF.get(iso2) in ANNEX_I_ISO2:
        return True
    return False


# ---------------------------------------------------------------------------
# Numbers ledger
# ---------------------------------------------------------------------------
_LEDGER = ROOT / "results" / "tables" / "numbers.json"


def _to_py(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_to_py(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_py(x) for k, x in v.items()}
    return v


def record(key: str, value, note: str | None = None) -> None:
    """Store a reported number in the shared JSON ledger."""
    _LEDGER.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if _LEDGER.exists():
        with open(_LEDGER) as fh:
            data = json.load(fh)
    data[key] = {"value": _to_py(value)}
    if note:
        data[key]["note"] = note
    with open(_LEDGER, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)


def ledger() -> dict:
    if not _LEDGER.exists():
        return {}
    with open(_LEDGER) as fh:
        return json.load(fh)
