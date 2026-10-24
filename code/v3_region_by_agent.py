"""
작용제별 구간 코사인을 현재 보정에서 재생성한다.

기존 data/v3_region_by_agent.csv 는 밴드 짝짓기를 1대1로 고치기 이전(2026-09-18)의
잔존 파일이었고 어떤 v3 스크립트도 그것을 다시 만들지 않았다. 그 결과 Table S15가
Table S20과 같은 양에 대해 다른 값을 보고했다(예: GF 0.562 대 0.548).
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
print(f"[보정] s=({SL:.3f}, {SH:.3f})\n")
print(D.to_string())
P = {"GD", "GF", "VX", "A230", "A232", "A234", "A242"}
print("\n인 함유 평균:", D.loc[[a for a in D.index if a in P]].mean().round(3).to_dict())
print("비인   평균:", D.loc[[a for a in D.index if a not in P]].mean().round(3).to_dict())
