"""
T1-19 — 600–850 cm-1 결손의 성분 분해.

심사 지적(R2-M10): 구간 코사인은 위치 오차·상대강도 오차·선폭 오차를 하나의 숫자로
뭉쳐 놓았으므로, 'P-F/P-N 신축의 계산 기술이 문제'라는 진단은 세 후보 중 하나일 뿐이다.

방법: 계산 라인목록의 한 속성만 실측값으로 교체하고 나머지는 그대로 둔 채 재합성하여
실측 스펙트럼과의 코사인이 얼마나 회복되는지 본다. 참조는 실측 스펙트럼 그 자체다.
  BASE  계산 위치 + 계산 강도 + Γ=22
  POS   **실측 위치** + 계산 강도 + Γ=22
  INT   계산 위치 + **실측 강도** + Γ=22
  WID   계산 위치 + 계산 강도 + **작용제별 실측 Γ**
  P+I   실측 위치 + 실측 강도 + Γ=22
  ALL   실측 위치 + 실측 강도 + 실측 Γ   (상한 — 짝 없는 밴드만 남는다)
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, measured_peaks, fit_scale,
                                 scattering_factor, cos, AGENTS, JOB, CUT)

DATA = os.path.join(ROOT, "data")
LO, HI, TOL = 600.0, 850.0, 35.0
PCONT = {"GD", "GF", "VX", "A230", "A232", "A234", "A242"}   # 인 함유


def lorentz(grid, pos, amp, gam):
    g = np.asarray(gam, float) / 2.0
    I = np.zeros_like(grid, float)
    for p, a, gg in zip(pos, amp, np.broadcast_to(g, np.shape(pos))):
        I += a * gg * gg / ((grid - p) ** 2 + gg * gg)
    return I


def main():
    X, acq = load_measured(); lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]
    _, SL, SH, _ = fit_scale(lib, mpk, calib)

    LW = pd.read_csv(f"{DATA}/v3_linewidths.csv")
    gper = LW.groupby("agent")["gamma"].mean().to_dict()
    gmed = float(LW["gamma"].median())

    win = (X >= LO) & (X <= HI)
    rows = []
    for ag in [a for a in AGENTS if a in JOB and len(acq[a]) >= 2]:
        pk = lib[JOB[ag]]
        f = np.where(pk[:, 0] < CUT, pk[:, 0] * SL, pk[:, 0] * SH)
        b = pk[:, 1].astype(float) * scattering_factor(f)
        keep = (f > LO - 150) & (f < HI + 150) & (b > 0.02 * b.max())
        f, b = f[keep], b[keep]
        gm = float(gper.get(ag, gmed))

        # 실측 밴드 (창 안, 여유 포함) 과 그 강도
        w = np.asarray([v for v in mpk[ag] if LO - 150 < v < HI + 150], float)
        a_m = np.asarray([mean[ag][np.argmin(np.abs(X - v))] for v in w], float)
        if not len(f) or not len(w):
            continue

        C = np.abs(w[:, None] - f[None, :]); C = np.where(C > TOL, 1e6, C)
        ri, ci = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]

        f_pos = f.copy(); b_int = b.copy()
        for i, j in pairs:
            f_pos[j] = w[i]
            b_int[j] = a_m[i] * (b.max() / (a_m.max() + 1e-12))

        M = mean[ag][win]
        def C_(pos, amp, gam):
            return cos(M, lorentz(X, pos, amp, gam)[win])

        rows.append(dict(
            agent=ag, phos=ag in PCONT, n_calc=len(f), n_meas=len(w), n_pair=len(pairs),
            gamma_meas=gm,
            BASE=C_(f, b, 22.0), POS=C_(f_pos, b, 22.0), INT=C_(f, b_int, 22.0),
            WID=C_(f, b, gm), PI=C_(f_pos, b_int, 22.0), ALL=C_(f_pos, b_int, gm)))

    D = pd.DataFrame(rows)
    D["d_POS"] = D.POS - D.BASE; D["d_INT"] = D.INT - D.BASE; D["d_WID"] = D.WID - D.BASE
    D.to_csv(f"{DATA}/v3_t119_decompose.csv", index=False)

    print("600–850 cm-1  한 속성만 실측값으로 교체했을 때의 코사인")
    print("=" * 96)
    print(f"{'작용제':<7}{'P':>3}{'짝/계산/실측':>14}{'BASE':>8}{'POS':>8}{'INT':>8}{'WID':>8}{'P+I':>8}{'ALL':>8}")
    print("-" * 96)
    for _, r in D.iterrows():
        print(f"{r.agent:<7}{'●' if r.phos else '○':>3}"
              f"{f'{int(r.n_pair)}/{int(r.n_calc)}/{int(r.n_meas)}':>14}"
              f"{r.BASE:>8.3f}{r.POS:>8.3f}{r.INT:>8.3f}{r.WID:>8.3f}{r.PI:>8.3f}{r.ALL:>8.3f}")
    print("-" * 96)
    for lab, sub in (("전체", D), ("인 함유", D[D.phos]), ("비인", D[~D.phos])):
        print(f"{lab:<7}{'':>3}{'':>14}{sub.BASE.mean():>8.3f}{sub.POS.mean():>8.3f}"
              f"{sub.INT.mean():>8.3f}{sub.WID.mean():>8.3f}{sub.PI.mean():>8.3f}{sub.ALL.mean():>8.3f}")
    print("\n회복량 (평균 Δ, BASE 대비)")
    for lab, sub in (("전체", D), ("인 함유", D[D.phos]), ("비인", D[~D.phos])):
        print(f"  {lab:<7} 위치 {sub.d_POS.mean():+.3f}   강도 {sub.d_INT.mean():+.3f}   "
              f"선폭 {sub.d_WID.mean():+.3f}   위치+강도 {(sub.PI-sub.BASE).mean():+.3f}   "
              f"전부 {(sub.ALL-sub.BASE).mean():+.3f}")
    print(f"\nsaved data/v3_t119_decompose.csv")


if __name__ == "__main__":
    main()
