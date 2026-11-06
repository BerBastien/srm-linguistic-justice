"""05 - Economic damages: no SRM (H), SRM (S), SRM with termination risk (S+T),
and an emulated mitigation world (M) with the same global temperature as S.

CLIMRISK reports annual ordinary KU positive losses by member and country for
H and S (60 x 81 x 187). M is not simulated, so we emulate it.

Emulator. In H, local temperature is beta_k(x) G_H(t), so for a fixed pattern k
the country loss share s = L / GDP depends on the global temperature G and,
weakly, on time (within-country GDP composition evolves). For every country c
and pattern k we fit, on the 10 no-SRM members that use pattern k:

   s_ckt = sum_{a=0..3} theta_a G^a + tau * sum_{b=0..2} phi_b G^b,
   tau = (year - 2060) / 40

and evaluate it at G = G_S,m(t), which places M on the same pattern manifold as H
but at the global temperature of S. Leave-one-member-out cross-validation is
reported. A time-invariant variant (phi = 0) is reported as a sensitivity.

Present value: 2026-2100, base year 2026, 1.5 percent per year (as CLIMRISK).

Outputs
  results/tables/country_pv_by_pathway.csv          country medians (public)
  results/tables/emulator_cv.csv                    cross-validation diagnostics
  data/interim/country_member_pv.npz                member x country PV (restricted
                                                    only because derived from CLIMRISK)
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, ensure_dirs, record  # noqa: E402

ensure_dirs()
f = h5py.File(P("climrisk_h5"), "r")
dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]  # noqa: E731
iso3 = dec(f["country/iso3"][:])
names = dec(f["country/name"][:])
years = f["time/year"][:]
GH = f["global/tas_no_srm_c_ref1900_by_member"][:]
GS = f["global/tas_srm_c_ref1900_by_member"][:]
pidx = f["ensemble/pattern_index"][:]
gdp = f["country/economic_annual/gdp_billion_usd2005_ppp"][:]            # (81, 187)
LH = f["country/economic_annual/ordinary_no_srm/positive_loss_by_member"][:]
LS = f["country/economic_annual/ordinary_srm/positive_loss_by_member"][:]
nm, ny, nc = LH.shape

r = CFG["analysis"]["discount_rate"]
t0, t1 = CFG["analysis"]["pv_start"], CFG["analysis"]["pv_end"]
win = (years >= t0) & (years <= t1)
disc = np.where(win, (1 + r) ** -(years - t0), 0.0)                   # (81,)
pv_gdp = (gdp * disc[:, None]).sum(axis=0)                            # (187,)

tau = (years - 2060) / 40.0


def design(G, T, with_time=True):
    cols = [np.ones_like(G), G, G ** 2, G ** 3]
    if with_time:
        cols += [T, T * G, T * G ** 2]
    return np.stack(cols, axis=-1)


share_H = LH / gdp[None, :, :]
LM = np.full_like(LH, np.nan)
LM_noT = np.full_like(LH, np.nan)
cv_rows = []
fit_years = np.ones_like(win, dtype=bool)  # fit on all years 2020-2100 (widest G support)
n_clamped = 0
n_pred = 0
for c in range(nc):
    if np.isnan(share_H[:, :, c]).all():
        continue
    for k in np.unique(pidx):
        ms = np.where(pidx == k)[0]
        G = GH[ms][:, fit_years].ravel()
        T = np.broadcast_to(tau[fit_years], (ms.size, fit_years.sum())).ravel()
        y = share_H[ms][:, fit_years, c].ravel()
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        for with_time, store in ((True, LM), (False, LM_noT)):
            X = design(G[ok], T[ok], with_time)
            theta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
            # predict mitigation shares for members using pattern k. G is clamped to
            # the fitted support [min G_H, max G_H]; below the support this holds the
            # damage share at its lowest fitted level, which can only OVERstate
            # mitigation damages (conservative for the S+T versus M comparison).
            Gm = np.clip(GS[ms], G.min(), G.max())
            if with_time:
                n_clamped += int(((GS[ms] < G.min()) & win[None, :]).sum())
                n_pred += int(win.sum() * ms.size)
            Tm = np.broadcast_to(tau, Gm.shape)
            pred = design(Gm, Tm, with_time) @ theta
            store[ms, :, c] = np.clip(pred, 0, 1) * gdp[None, :, c]
            if with_time:
                # leave-one-member-out CV on the no-SRM data
                err = []
                ybar = y[ok].mean()
                memb = np.repeat(np.arange(ms.size), fit_years.sum())[ok]
                for mo in range(ms.size):
                    tr = memb != mo
                    th, *_ = np.linalg.lstsq(X[tr], y[ok][tr], rcond=None)
                    err.append((X[~tr] @ th - y[ok][~tr]))
                err = np.concatenate(err)
                cv_rows.append({"iso3": iso3[c], "pattern": int(k), "mean_share": ybar,
                                "rmse": float(np.sqrt(np.mean(err ** 2))),
                                "rel_rmse": float(np.sqrt(np.mean(err ** 2)) / max(ybar, 1e-12)),
                                "r2": float(1 - np.sum(err ** 2) / np.sum((y[ok] - ybar) ** 2)),
                                "G_min_fit": float(G.min()), "G_max_fit": float(G.max()),
                                "G_min_pred": float(GS[ms][:, win].min()),
                                "G_max_pred": float(GS[ms][:, win].max())})
cv = pd.DataFrame(cv_rows)
record("emulator.frac_member_years_clamped", n_clamped / max(n_pred, 1))
cv.to_csv(P("results") / "emulator_cv.csv", index=False)
record("emulator.median_rel_rmse", float(cv.rel_rmse.median()))
record("emulator.p95_rel_rmse", float(cv.rel_rmse.quantile(0.95)))
record("emulator.median_r2", float(cv.r2.median()))
record("emulator.min_r2", float(cv.r2.min()))
record("emulator.n_country_patterns", len(cv))


def pv(L):
    return np.nansum(L * disc[None, :, None], axis=1)                 # (60, 187)


PV = {"H": pv(LH), "S": pv(LS), "M": pv(LM), "M_noT": pv(LM_noT)}
# validate our PV of H and S against CLIMRISK's stored PV
PVH_ref = f["country/economic_pv/ordinary_no_srm/positive_loss_by_member"][:]
PVS_ref = f["country/economic_pv/ordinary_srm/positive_loss_by_member"][:]
okc = np.isfinite(PVH_ref).all(axis=0) & (PVH_ref.sum(axis=0) > 0)
relH = np.nanmax(np.abs(PV["H"][:, okc] - PVH_ref[:, okc]) / np.maximum(PVH_ref[:, okc], 1e-9))
relS = np.nanmax(np.abs(PV["S"][:, okc] - PVS_ref[:, okc]) / np.maximum(PVS_ref[:, okc], 1e-9))
record("validation.pv_recompute_max_rel_err_H", float(relH))
record("validation.pv_recompute_max_rel_err_S", float(relS))

# full-damage pathways from the termination module (positive loss PV, member level)
PV["H_full"] = f["country/termination_pv/full_no_srm/positive_loss_by_member"][:]
PV["S_full"] = f["country/termination_pv/full_srm_no_failure/positive_loss_by_member"][:]
PV["ST"] = f["country/termination_pv/full_srm_termination_local_ku/positive_loss_by_member"][:]
PV["ST_restricted"] = f["country/termination_pv/full_srm_termination_restricted_ku/positive_loss_by_member"][:]
PV["premium"] = f["country/termination_pv/termination_premium_local_ku/net_damage_by_member"][:]

np.savez_compressed(P("interim") / "country_member_pv.npz", iso3=np.array(iso3),
                    pv_gdp=pv_gdp, pattern_index=pidx, **PV)

# country table of medians (percent of PV GDP)
rows = []
for c in range(nc):
    if pv_gdp[c] <= 0:
        continue
    r_ = {"iso3": iso3[c], "country": names[c], "pv_gdp_2026_2100": pv_gdp[c]}
    for k, A in PV.items():
        x = 100 * A[:, c] / pv_gdp[c]
        r_[f"{k}_p17"], r_[f"{k}_p50"], r_[f"{k}_p83"] = np.nanpercentile(x, [17, 50, 83])
    d1 = 100 * (PV["S"][:, c] - PV["M"][:, c]) / pv_gdp[c]
    d2 = 100 * (PV["ST"][:, c] - PV["M"][:, c]) / pv_gdp[c]
    r_["S_minus_M_p50"] = np.nanmedian(d1)
    r_["S_minus_M_frac_members_pos"] = np.mean(d1 > 0)
    r_["ST_minus_M_p50"] = np.nanmedian(d2)
    r_["ST_minus_M_frac_members_pos"] = np.mean(d2 > 0)
    rows.append(r_)
ct = pd.DataFrame(rows)
ct.to_csv(P("results") / "country_pv_by_pathway.csv", index=False)

# world totals
for k in ("H", "S", "M", "M_noT", "ST", "premium"):
    w = 100 * np.nansum(PV[k], axis=1) / pv_gdp.sum()
    record(f"econ.world_pct_pvgdp.{k}.p17_p50_p83", np.percentile(w, [17, 50, 83]))
print(ct[["iso3", "H_p50", "M_p50", "S_p50", "ST_p50", "premium_p50", "S_minus_M_p50",
          "ST_minus_M_p50"]].sort_values("premium_p50", ascending=False).head(25).round(2))
print(cv.describe().round(4))
