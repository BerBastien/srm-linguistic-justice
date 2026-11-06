"""19 - Table 1 of the rainfall paper: exposure to SAI drying by vitality class (2100).

Columns: n languages; share >2 pp drier than mitigation [95% bootstrap CI over languages];
member p17-p83 of that share; share >2 pp wetter; median rainfall difference (pp);
share >5 pp drier; share with less rain than in 1900 under SAI and under mitigation;
L1 speakers of the languages left drier (millions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srmlang.common import CFG, P, record  # noqa: E402

I, T = P("interim"), P("results")
DB = CFG["analysis"]["precip_deadband_pct"]
rng = np.random.default_rng(CFG["analysis"]["random_seed"])
dl = pd.read_parquet(I / "dry_lang.parquet")
z = np.load(I / "dry_member.npz", allow_pickle=True)
RP, PS, PM = z["RP_2100"], z["PS_2100"], z["PM_2100"]
groups = [("Stable", dl.vitality == "Stable"), ("In trouble", dl.vitality == "In trouble"),
          ("Dying", dl.vitality == "Dying"), ("Non-institutional", dl.noninst),
          ("Institutional", ~dl.noninst), ("All living languages", dl.vitality.notna())]
# v5 (Zariquiey): the same rows by Glottolog 5.3's agglomerated endangerment status (AES)
cw = pd.read_parquet(I / "glottolog_crosswalk.parquet").set_index("ISO")
aes = dl.ISO.map(cw.g_aes)
AES = [("Not endangered", "aes-not_endangered"), ("Threatened", "aes-threatened"), ("Shifting", "aes-shifting"),
       ("Moribund", "aes-moribund"), ("Nearly extinct", "aes-nearly_extinct"), ("Extinct in Glottolog", "aes-extinct")]
groups += [(f"AES: {lab}", aes == code) for lab, code in AES]
record("dry.t1.aes.n_without_value", int(aes.isna().sum()))
rows = []
for name, m in groups:
    m = m.to_numpy()
    d = dl[m]
    x = d.drier.to_numpy(float)
    boot = rng.choice(x, (CFG["analysis"]["n_bootstrap"], len(x))).mean(1)
    mem = (RP[m] < -DB).mean(0)
    r = {"group": name, "n": int(m.sum()),
         "share_drier": x.mean(), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
         "mem_p17": np.percentile(mem, 17), "mem_p83": np.percentile(mem, 83),
         "share_wetter": d.wetter.mean(), "median_RP": d.RP_2100.median(),
         "share_lt_minus5": (d.RP_2100 < -5).mean(),
         "below1900_S": (d.PS_2100 < 0).mean(), "below1900_M": (d.PM_2100 < 0).mean(),
         "L1_drier_millions": d.L1_Users.where(d.drier, 0).sum() / 1e6}
    rows.append(r)
    key = name.replace(" ", "_")
    for k, v in r.items():
        if k != "group":
            record(f"dry.t1.{key}.{k}", v)
tab = pd.DataFrame(rows)
tab.to_csv(T / "dry_table1.csv", index=False)

LAB = {"Stable": "Stable (EGIDS 5--6a)", "In trouble": "In trouble (6b--7)", "Dying": "Dying (8a--8b)"}
lines = []
for r in tab.itertuples():
    if r.group == "AES: Not endangered":
        lines.append("\\multicolumn{10}{l}{\\textit{Glottolog agglomerated endangerment status}}\\\\")
    lines.append(f"{LAB.get(r.group, r.group.replace('AES: ', ''))} & {r.n:,} & {100 * r.share_drier:.1f} [{100 * r.ci_lo:.1f}, {100 * r.ci_hi:.1f}] & "
                 f"{100 * r.mem_p17:.0f}--{100 * r.mem_p83:.0f} & {100 * r.share_wetter:.1f} & "
                 f"${r.median_RP:+.1f}$ & {100 * r.share_lt_minus5:.1f} & {100 * r.below1900_S:.1f} & "
                 f"{100 * r.below1900_M:.1f} & {r.L1_drier_millions:,.0f} \\\\")
    if r.group == "Dying":
        lines.append("\\midrule")
    if r.group in ("Institutional", "All living languages"):
        lines.append("\\midrule")
txt = "\n".join(lines).replace("$+", "$+").replace("$-", "$-")
(T / "latex").mkdir(exist_ok=True)
(T / "latex" / "tab1_drying.tex").write_text(txt + "\n")
print(tab.round(3).to_string())
