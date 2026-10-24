"""
T1-14b — 밴드 검출 기준과 짝짓기 허용창에 대한 보정 민감도.

심사 지적(R1-M2): 35 cm-1 허용창은 피크 검출의 최소 간격과 비슷해 배정이 거의
축퇴 상태이며, 관문을 통과한 생존자에 대해서만 계산한 RMSE는 최악의 밴드를
설계상 배제하므로 낙관적이다. 여기서 검출 기준 셋 × 허용창 둘을 모두 보고한다.
미짝 밴드 벌점은 비교 가능하도록 항상 35 cm-1로 고정한다.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import load_measured, load_library, fit_scale, AGENTS, JOB, CUT

PEN = 35.0
MODES = [("prominence $>$ 0.05, separation 2 points", dict(prominence=0.05, distance=2)),
         ("height $>$ 0.10 and prominence $>$ 0.05, separation 4 points",
          dict(height=0.10, prominence=0.05, distance=4)),
         ("height $>$ 0.12, separation 6 points", dict(height=0.12, distance=6))]

X, acq = load_measured(); lib = load_library()
mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
CAL = [a for a in AGENTS if a in JOB]
rows = []
for lab, kw in MODES:
    mpk = {a: X[find_peaks(mean[a], **kw)[0]] for a in AGENTS}
    _, SL, SH, _ = fit_scale(lib, mpk, CAL, min_pairs=20)
    for tol in (35.0, 20.0):
        d = []; nm = nc = npr = 0
        for ag in CAL:
            pk = lib[JOB[ag]]
            f = np.where(pk[:, 0] < CUT, pk[:, 0] * SL, pk[:, 0] * SH)
            k = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max()); f = f[k]
            w = np.asarray(mpk[ag], float)
            if not len(f) or not len(w): continue
            C = np.abs(w[:, None] - f[None, :]); C = np.where(C > tol, 1e6, C)
            ri, ci = linear_sum_assignment(C)
            m = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]
            d += [float(f[j] - w[i]) for i, j in m]
            nm += len(w); nc += len(f); npr += len(m)
        d = np.asarray(d)
        rows.append(dict(criterion=lab, tol=tol, s_low=SL, s_high=SH,
                         rmse=float(np.sqrt((d ** 2).mean())), n_pair=npr,
                         n_meas=nm, n_calc=nc,
                         rmse_pen=float(np.sqrt(np.concatenate(
                             [d ** 2, np.full(nm + nc - 2 * npr, PEN ** 2)]).mean()))))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(ROOT, "data", "v3_detect_sensitivity.csv"), index=False)
print(D.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
