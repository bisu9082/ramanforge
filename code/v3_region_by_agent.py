"""
Regenerates the per-agent region cosines under the current calibration.

The existing data/v3_region_by_agent.csv was a leftover file from before band pairing was
corrected to one-to-one (2026-09-18), and no v3 script rebuilt it. As a result Table S15
reported a different value from Table S20 for the same quantity (for example, GF 0.562 vs 0.548).
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, measured_peaks, fit_scale,
                                 synth_native, cos, AGENTS, JOB)

REG = [("250-600 skeletal/torsion", 250, 600), ("600-850", 600, 850),
       ("850-1150", 850, 1150), ("1150-1400", 1150, 1400),
       ("1400-1800 CH deform", 1400, 1800)]

X, acq = load_measured(); lib = load_library()
mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
CAL = [a for a in AGENTS if a in JOB]
_, SL, SH, _ = fit_scale(lib, mpk, CAL)
dft = {k: synth_native(v, X, SL, SH) for k, v in lib.items()}

ags = [a for a in AGENTS if a in JOB and a != "HN2"]
rows = {}
for a in ags:
    rows[a] = {n: cos(mean[a][(X >= lo) & (X < hi)], dft[JOB[a]][(X >= lo) & (X < hi)])
               for n, lo, hi in REG}
D = pd.DataFrame(rows).T.round(3)
D.to_csv(os.path.join(ROOT, "data", "v3_region_by_agent.csv"))
print(f"[calibration] s=({SL:.3f}, {SH:.3f})\n")
print(D.to_string())
P = {"GD", "GF", "VX", "A230", "A232", "A234", "A242"}
print("\nP-containing mean:", D.loc[[a for a in D.index if a in P]].mean().round(3).to_dict())
print("P-free       mean:", D.loc[[a for a in D.index if a not in P]].mean().round(3).to_dict())
