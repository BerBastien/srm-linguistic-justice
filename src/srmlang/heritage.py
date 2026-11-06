"""Heritage metrics borrowed from biodiversity science (used by scripts 21, 23 and 25).

A language's homeland is a set of grid cells with range shares r_lc that sum to one, so every
language counts once whatever its size. The language tree comes from Ethnologue classification
strings: each language is a path (family, branch, sub-branch, ..., language).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

# Ethnologue "families" that are not genealogical units: each such language is its own lineage
SINGLE = {"Language isolate", "Unclassified", "Mixed language", "Deaf sign language", "Pidgin",
          "Artificial language", "Spurious", "Unclassified language"}


def path_of(iso: str, family: str | None, classification: str | None) -> tuple:
    """Path of a language on the classification tree, ending in the language itself.

    Isolates, unclassified, mixed, pidgin and artificial languages become their own lineage,
    labelled "[<family>:<iso>]".
    """
    if family in SINGLE or not isinstance(classification, str) or classification.strip() in SINGLE:
        return (f"[{family}:{iso}]", iso)
    parts = [p.strip() for p in classification.split(",") if p.strip()]
    return tuple(parts) + (iso,)


def glottolog_path(iso: str, lineage: str, classification: str | None) -> tuple:
    """Path of a language on Glottolog's classification (v5), ending in the language itself.

    `classification` is Glottolog's slash-separated list of ancestor glottocodes, from the top-level
    family down to the language's parent; it is empty for isolates. An isolate is its own lineage:
    path (lineage, iso). Every lineage hangs from a common root by one branch (the first element of
    the path); the root is a computational convention, not a claim of relatedness.
    """
    parts = [p for p in (classification or "").split("/") if p]
    if not parts:
        return (lineage, iso)
    if parts[0] != lineage:
        raise ValueError(f"{iso}: classification does not start with its lineage {lineage}")
    return tuple(parts) + (iso,)


def descendant_counts(paths: Mapping[str, tuple]) -> dict:
    """Number of languages below every branch (every prefix of every path)."""
    desc: dict = {}
    for pth in paths.values():
        for k in range(1, len(pth) + 1):
            desc[pth[:k]] = desc.get(pth[:k], 0) + 1
    return desc


def fair_proportion(paths: Mapping[str, tuple]) -> dict:
    """Evolutionary distinctiveness by fair proportion (Isaac et al. 2007) with unit branch lengths.

    ED_l = sum over the branches on l's path of 1 / (number of languages below the branch).
    The ED values of all languages sum to the number of branches of the tree.
    """
    desc = descendant_counts(paths)
    return {iso: sum(1.0 / desc[pth[:k]] for k in range(1, len(pth) + 1)) for iso, pth in paths.items()}


def pd_count(paths: Mapping[str, tuple], isos: Iterable[str]) -> int:
    """Phylogenetic-diversity analogue (Faith 1992): number of distinct branches connecting a set."""
    s = set()
    for i in isos:
        pth = paths[i]
        for k in range(1, len(pth) + 1):
            s.add(pth[:k])
    return len(s)


def homeland_fraction(R, X):
    """Fraction of each homeland in an outcome: rows of R are range shares (sum to one), X is (C,) or (C, M)."""
    return R @ np.asarray(X, dtype=float)


def lineage_mean(values, labels) -> np.ndarray:
    """Mean over lineages of the within-lineage mean (each lineage counted once). values: (L,) or (L, M)."""
    v = np.asarray(values, dtype=float)
    v2 = v[:, None] if v.ndim == 1 else v
    df = pd.DataFrame(v2)
    out = df.groupby(np.asarray(labels)).mean().mean(axis=0).to_numpy()
    return out[0] if v.ndim == 1 else out


def weighted_share(weights, X) -> np.ndarray | float:
    """Weighted share of cells (or units) in an outcome X (boolean or fraction), X: (C,) or (C, M)."""
    w = np.asarray(weights, dtype=float)
    x = np.asarray(X, dtype=float)
    if x.ndim == 1:
        return float((w * x).sum() / w.sum())
    return (w[:, None] * x).sum(0) / w.sum()


def hotspot_cells(endemism, area=None, share: float = 0.5) -> np.ndarray:
    """Indices of hotspot cells: ranked by endemism density (endemism / area, or endemism if no
    area is given), the top cells that together hold `share` of total endemism."""
    e = np.asarray(endemism, dtype=float)
    dens = e if area is None else e / np.asarray(area, dtype=float)
    order = np.argsort(-dens)
    cum = np.cumsum(e[order]) / e.sum()
    k = int(np.searchsorted(cum, share) + 1)
    return order[:k]
