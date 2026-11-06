"""Unit tests for the heritage metrics (weighted endemism, tree metrics, baselines) on toy data."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang import heritage as hm  # noqa: E402

# toy tree: family F1 with branches B1 (a, b) and B2 (c); d is an isolate
PATHS = {"a": hm.path_of("a", "F1", "F1, B1"), "b": hm.path_of("b", "F1", "F1, B1"),
         "c": hm.path_of("c", "F1", "F1, B2"), "d": hm.path_of("d", "Language isolate", None)}


def test_path_of_isolate_is_own_lineage():
    assert PATHS["a"] == ("F1", "B1", "a")
    assert PATHS["d"] == ("[Language isolate:d]", "d")


def test_fair_proportion_values_and_sum():
    ed = hm.fair_proportion(PATHS)
    assert np.isclose(ed["a"], 1 / 3 + 1 / 2 + 1)
    assert np.isclose(ed["c"], 1 / 3 + 1 + 1)
    assert np.isclose(ed["d"], 2.0)
    # fair proportion distributes every branch among the languages below it
    assert np.isclose(sum(ed.values()), len(hm.descendant_counts(PATHS)))


def test_pd_count():
    assert hm.pd_count(PATHS, ["a", "b"]) == 4          # F1, F1-B1, a, b
    assert hm.pd_count(PATHS, ["a", "c"]) == 5          # F1, F1-B1, a, F1-B2, c
    assert hm.pd_count(PATHS, list(PATHS)) == 8


def test_rarity_weighting_spanish_vs_small_language():
    # a widespread language over cells 0-8 and a small one only in cell 9; only cell 9 is drier
    area = np.ones(10)
    dry = np.zeros(10)
    dry[9] = 1
    w = np.zeros((2, 10))
    w[0, :9] = 1.0
    w[1, 9] = 1.0
    R = sp.csr_matrix(w / w.sum(1, keepdims=True))
    f = hm.homeland_fraction(R, dry)
    assert np.allclose(f, [0.0, 1.0])
    assert hm.weighted_share(area, dry) == 0.1          # land: 10% drier
    assert f.mean() == 0.5                              # languages, each counted once: 50%


def test_lineage_mean_counts_each_lineage_once():
    vals = np.array([1.0, 1.0, 1.0, 0.0])               # three languages of one family, one isolate
    labels = np.array(["F1", "F1", "F1", "iso"])
    assert hm.lineage_mean(vals, labels) == 0.5
    two = np.c_[vals, 1 - vals]                         # works member by member
    assert np.allclose(hm.lineage_mean(two, labels), [0.5, 0.5])


def test_weighted_share_members():
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    assert np.allclose(hm.weighted_share([1, 1, 2], X), [0.75, 0.75])


def test_hotspot_cells_by_density():
    endemism = np.array([5.0, 1.0, 4.0, 0.5])
    area = np.array([10.0, 1.0, 1.0, 2.0])               # cell 0 has most endemism but low density
    hot = hm.hotspot_cells(endemism, area, 0.5)
    assert list(hot) == [2, 1, 0]                        # densities 4, 1, 0.5, 0.25
    assert endemism[hot].sum() / endemism.sum() >= 0.5
    assert list(hm.hotspot_cells(endemism, None, 0.5)) == [0, 2]


def test_glottolog_path_family_and_isolate():
    assert hm.glottolog_path("abc", "fam1", "fam1/sub1/sub2") == ("fam1", "sub1", "sub2", "abc")
    assert hm.glottolog_path("xyz", "isol1", "") == ("isol1", "xyz")
    assert hm.glottolog_path("xyz", "isol1", None) == ("isol1", "xyz")
    paths = {"a": hm.glottolog_path("a", "F", "F/B1"), "b": hm.glottolog_path("b", "F", "F/B1"),
             "c": hm.glottolog_path("c", "F", "F/B2"), "d": hm.glottolog_path("d", "I", "")}
    ed = hm.fair_proportion(paths)
    assert np.isclose(ed["a"], 1 / 3 + 1 / 2 + 1)
    assert np.isclose(ed["d"], 2.0)


def test_glottolog_path_rejects_mismatch():
    import pytest
    with pytest.raises(ValueError):
        hm.glottolog_path("a", "F", "G/B1")
