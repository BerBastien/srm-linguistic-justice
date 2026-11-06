"""Small, dependency-light statistics used throughout the analysis.

All functions accept NumPy arrays and optional non-negative weights.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def wmean(x, w=None):
    x = np.asarray(x, float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def wquantile(x, q, w=None):
    """Weighted quantile (inverse of the weighted empirical CDF, midpoint rule)."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.nan
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(q, cw, x))


def wshare(cond, w=None):
    """Weighted share of observations for which ``cond`` is True."""
    cond = np.asarray(cond, bool)
    w = np.ones(cond.shape) if w is None else np.asarray(w, float)
    m = np.isfinite(w) & (w > 0)
    return float(np.sum(w[m] * cond[m]) / np.sum(w[m]))


def bootstrap(fn, n_obs, n_boot=2000, seed=0, strata=None):
    """Non-parametric bootstrap over observation indices.

    ``fn(idx)`` must return a scalar computed on the resampled indices.
    If ``strata`` (array of group labels) is given, resampling is done within
    strata, so group sizes are preserved (used for between-group gaps).
    Returns the array of bootstrap replicates.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    if strata is None:
        for b in range(n_boot):
            out[b] = fn(rng.integers(0, n_obs, n_obs))
    else:
        strata = np.asarray(strata)
        groups = [np.where(strata == s)[0] for s in np.unique(strata)]
        for b in range(n_boot):
            idx = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])
            out[b] = fn(idx)
    return out


def ci(reps, level=0.95):
    a = (1 - level) / 2
    return float(np.nanquantile(reps, a)), float(np.nanquantile(reps, 1 - a))


def concentration_index(burden, rank_var, w=None):
    """Weighted concentration index (Wagstaff, van Doorslaer & Watanabe 1991;
    Kakwani, Wagstaff & van Doorslaer 1997).

    CI = 2 * cov_w(burden, r) / mean_w(burden), where r is the weighted
    fractional rank of ``rank_var`` (ascending). CI lies in [-1, 1] for a
    non-negative burden. CI < 0: burden concentrated among observations with
    LOW values of ``rank_var`` (e.g. small speaker communities); CI > 0: among
    HIGH values. Ties in ``rank_var`` receive their average fractional rank.
    """
    y = np.asarray(burden, float)
    z = np.asarray(rank_var, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    y, z, w = y[m], z[m], w[m]
    o = np.argsort(z, kind="mergesort")
    y, z, w = y[o], z[o], w[o]
    W = w.sum()
    cw = np.cumsum(w)
    r = (cw - 0.5 * w) / W
    # average ranks over ties
    _, inv, counts = np.unique(z, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=r * w)
        wts = np.bincount(inv, weights=w)
        r = (sums / wts)[inv]
    mu = np.sum(w * y) / W
    if mu == 0:
        return np.nan
    cov = np.sum(w * (y - mu) * (r - np.sum(w * r) / W)) / W
    return float(2 * cov / mu)


def concentration_curve(burden, rank_var, w=None, n=101):
    """Return (cumulative population share, cumulative burden share) points."""
    y = np.asarray(burden, float)
    z = np.asarray(rank_var, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    y, z, w = y[m], z[m], w[m]
    o = np.argsort(z, kind="mergesort")
    y, w = y[o], w[o]
    cp = np.concatenate([[0], np.cumsum(w) / w.sum()])
    cb = np.concatenate([[0], np.cumsum(w * y) / np.sum(w * y)])
    grid = np.linspace(0, 1, n)
    return grid, np.interp(grid, cp, cb)


def ranksum(x, y):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) test; returns (U, p, r_rb).

    r_rb is the rank-biserial correlation, an effect size in [-1, 1]
    (positive: values in ``x`` tend to exceed values in ``y``).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    U, p = sps.mannwhitneyu(x, y, alternative="two-sided")
    r_rb = 2 * U / (x.size * y.size) - 1
    return float(U), float(p), float(r_rb)


def relative_risk(cond_a, cond_b, w_a=None, w_b=None):
    """Ratio of (weighted) shares: P(cond | group a) / P(cond | group b)."""
    pa = wshare(cond_a, w_a)
    pb = wshare(cond_b, w_b)
    return pa / pb if pb > 0 else np.nan
