"""Unit tests for the statistics and for the residual algebra."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang import stats as st  # noqa: E402
from srmlang.common import annex_i, endangered, vitality  # noqa: E402


def test_wmean_and_share():
    assert st.wmean([1, 2, 3]) == 2
    assert st.wmean([1, 3], [3, 1]) == 1.5
    assert st.wshare([True, False, False, True]) == 0.5


def test_concentration_index_limits():
    # burden carried only by the highest-ranked observation -> CI close to +1
    y = np.r_[np.zeros(999), 1.0]
    z = np.arange(1000)
    assert st.concentration_index(y, z) > 0.99
    # equal burden -> CI = 0
    assert abs(st.concentration_index(np.ones(100), np.arange(100))) < 1e-12
    # burden decreasing in rank -> CI < 0
    assert st.concentration_index(np.arange(100)[::-1].astype(float), np.arange(100)) < 0


def test_ranksum_effect_size_sign():
    U, p, r = st.ranksum(np.arange(10, 20), np.arange(0, 10))
    assert r == 1.0 and p < 0.001


def test_residual_algebra():
    # R = T_S - T_M = (beta + gamma) dG ; rho = -gamma / beta
    beta, gamma, GH, GS = 1.2, -1.4, 3.5, 1.9
    TH, TS, TM = beta * GH, beta * GH + gamma * (GH - GS), beta * GS
    assert np.isclose(TS - TM, (beta + gamma) * (GH - GS))
    assert np.isclose((TH - TS) / (TH - TM), -gamma / beta)


def test_classifications():
    assert vitality("6b") == "In trouble" and vitality("8b") == "Dying"
    assert endangered("7") is True and endangered("6a") is False and endangered("x10") is None
    assert annex_i("US") and annex_i("GL") and not annex_i("GL", include_territories=False)
    assert not annex_i("MX")
