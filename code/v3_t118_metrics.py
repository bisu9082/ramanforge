"""
T1-18 — 지표 비교. 동일 쿼리·동일 라이브러리·동일 전처리에서 '지표'만 바꾼다.

심사 지적(R2-M11, R3-M10): full-profile 코사인은 Γ=22 격자에서 위치 잔차 0.6Γ
구간에 있어 위치 오차에 최대로 민감하다. 따라서 보고된 결손이 '계산의 부정확'인지
'지표의 부적합'인지 구별되지 않는다. band-match fraction이 in-library 조건에서
0.833이라는 저자 자신의 수치가 후자를 시사한다.

공정성: 측정·계산 양쪽 모두 **동일한 피크 검출기**를 동일 native 격자에서 통과시킨다.
"""
import os, sys, json, collections
import numpy as np, pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, measured_peaks, fit_scale,
                                 synth_native, cos, identify, AGENTS, CLASS, JOB, CUT, GAMMA)

DATA = os.path.join(ROOT, "data")

# ─────────────────────────────── 지표 정의
def peaklist(spec, X, prom=0.05):
    pk, _ = find_peaks(spec, prominence=prom, distance=2)
    return X[pk], spec[pk]

def hqi(qp, qa, lp, la, tol):
    """대칭 피크리스트 hit-quality. 1대1 할당 + 위치 벌점 + 강도 가중.
       S = Σ min(a,b)·(1−|Δ|/tol) / max(Σa, Σb)   ∈ [0,1]
       짝 없는 쿼리 피크와 여분의 라이브러리 피크를 모두 벌점한다."""
    if not len(qp) or not len(lp): return 0.0
    C = np.abs(qp[:, None] - lp[None, :])
    C = np.where(C > tol, 1e6, C)
    ri, ci = linear_sum_assignment(C)
    num = 0.0
    for i, j in zip(ri, ci):
        if C[i, j] < 1e6:
            num += min(qa[i], la[j]) * (1.0 - C[i, j] / tol)
    return float(num / max(qa.sum(), la.sum()))

def dice(qp, lp, tol):
    """강도를 배제한 순수 밴드 위치 일치 (Dice). 위치만의 기여를 본다."""
    if not len(qp) or not len(lp): return 0.0
    C = np.abs(qp[:, None] - lp[None, :])
    C = np.where(C > tol, 1e6, C)
    ri, ci = linear_sum_assignment(C)
    m = sum(1 for i, j in zip(ri, ci) if C[i, j] < 1e6)
    return float(2 * m / (len(qp) + len(lp)))

def dcos(u, v, w=7):
    """1차 미분 코사인. 완만한 배경·포락선 차이를 제거하고 밴드 구조만 비교."""
    du = savgol_filter(u, w, 2, deriv=1); dv = savgol_filter(v, w, 2, deriv=1)
    return cos(du, dv)

def shiftcos(u, v, maxpx=3):
    """±maxpx 픽셀 정렬 허용 코사인 (≈±17 cm-1). 계통적 위치 오차를 흡수시킨다."""
    best = -1.0
    for k in range(-maxpx, maxpx + 1):
        vv = np.roll(v, k)
        if k > 0: vv[:k] = v[0]
        elif k < 0: vv[k:] = v[-1]
        best = max(best, cos(u, vv))
    return best

# ─────────────────────────────── main
def main():
    X, acq = load_measured(); lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]
    _, SL, SH, _ = fit_scale(lib, mpk, calib)
    print(f"[보정] s=({SL:.3f}, {SH:.3f})  Γ_base={GAMMA:.0f}\n")

    # 작용제별 실측 선폭 (Table S7) — 없으면 중앙값
    LW = pd.read_csv(f"{DATA}/v3_linewidths.csv")
    gcol = [c for c in LW.columns if "gamma" in c.lower() or c.lower() in ("g", "width", "fwhm")][0]
    gmed = float(LW[gcol].median())
    gper = LW.groupby("agent")[gcol].mean().to_dict()
    print(f"작용제별 실측 Γ: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(gper.items())))
    print(f"중앙값 {gmed:.1f}\n")

    JOB2AG = {j: a for a, j in JOB.items()}
    def gamma_for(job):
        a = JOB2AG.get(job)
        return float(gper.get(a, gmed)) if a else gmed

    # 계산 라이브러리: Γ 시나리오별 합성
    scen = {"G22": {k: synth_native(v, X, SL, SH, 22.0) for k, v in lib.items()},
            "G30": {k: synth_native(v, X, SL, SH, 30.0) for k, v in lib.items()},
            "G40": {k: synth_native(v, X, SL, SH, 40.0) for k, v in lib.items()},
            "G50": {k: synth_native(v, X, SL, SH, 50.0) for k, v in lib.items()},
            "Gper": {k: synth_native(v, X, SL, SH, gamma_for(k)) for k, v in lib.items()}}

    core = [JOB[a] for a in AGENTS if a in JOB]
    queryable = [a for a in AGENTS if len(acq[a]) >= 2]
    queries = [(a, r, s) for a in queryable for r, s in acq[a]]
    print(f"쿼리 {len(queries)}건 / {len(queryable)}종\n")

    # 피크리스트 사전계산 (라이브러리)
    PLK = {sk: {k: peaklist(v, X) for k, v in sp.items()} for sk, sp in scen.items()}

    METRICS = []   # (label, kind, param)
    for sk in ["G22", "G30", "G40", "G50", "Gper"]:
        METRICS.append((f"cosine, Γ={sk[1:] if sk!='Gper' else 'per-agent'}", "cos", sk))
    METRICS += [("1차 미분 코사인, Γ=22", "dcos", "G22"),
                ("±3px 이동허용 코사인, Γ=22", "shift", "G22"),
                ("피크리스트 HQI, tol=10", "hqi10", "G22"),
                ("피크리스트 HQI, tol=20", "hqi20", "G22"),
                ("피크리스트 HQI, tol=35", "hqi35", "G22"),
                ("밴드위치 Dice, tol=20", "dice20", "G22"),
                ("밴드위치 Dice, tol=35", "dice35", "G22")]

    out = []
    for label, kind, sk in METRICS:
        dft = scen[sk]; plk = PLK[sk]
        nmol = ncls = nmol53 = ncls53 = 0
        selfs, margs = [], []
        per = collections.defaultdict(lambda: [0, 0])
        for q_ag, q_rep, q in queries:
            gold = JOB.get(q_ag, "__none__")
            if kind == "cos":
                sc = {k: cos(q, dft[k]) for k in dft}
            elif kind == "dcos":
                sc = {k: dcos(q, dft[k]) for k in dft}
            elif kind == "shift":
                sc = {k: shiftcos(q, dft[k]) for k in dft}
            elif kind.startswith("hqi"):
                tol = float(kind[3:]); qp, qa = peaklist(q, X)
                sc = {k: hqi(qp, qa, plk[k][0], plk[k][1], tol) for k in dft}
            else:
                tol = float(kind[4:]); qp, _ = peaklist(q, X)
                sc = {k: dice(qp, plk[k][0], tol) for k in dft}

            for tag, pool in (("closed", core), ("open", list(dft.keys()))):
                s2 = {k: sc[k] for k in pool}
                o = sorted(s2.items(), key=lambda t: -t[1])
                top = o[0][0]
                ok = (top == gold)
                clsok = (CLASS.get(JOB2AG.get(top, ""), "Nerve") == CLASS[q_ag])
                if tag == "closed":
                    nmol += ok; ncls += clsok
                    selfs.append(s2.get(gold, np.nan))
                    margs.append(o[0][1] - (o[1][1] if len(o) > 1 else 0))
                    per[q_ag][0] += ok; per[q_ag][1] += 1
                else:
                    nmol53 += ok; ncls53 += clsok
        N = len(queries)
        agent_ok = sum(1 for a in per if per[a][0] * 2 > per[a][1])
        out.append(dict(metric=label, kind=kind, gamma=sk,
                        mol_closed=nmol, mol_open=nmol53, cls_closed=ncls, cls_open=ncls53,
                        n=N, agent_closed=agent_ok, n_agents=len(per),
                        self_mean=float(np.nanmean(selfs)), margin_mean=float(np.nanmean(margs))))
        print(f"{label:<34} 분자 {nmol:2d}/{N}  (53종 {nmol53:2d}/{N})  "
              f"클래스 {ncls:2d}/{N}  작용제단위 {agent_ok}/{len(per)}  "
              f"자기 {np.nanmean(selfs):.3f}  마진 {np.nanmean(margs):.3f}")

    D = pd.DataFrame(out)
    D.to_csv(f"{DATA}/v3_t118_metrics.csv", index=False)
    print(f"\nsaved data/v3_t118_metrics.csv")
    base = D[D.metric.str.startswith("cosine, Γ=22")].iloc[0]
    best = D.loc[D.mol_closed.idxmax()]
    print(f"\n기준선 (cosine Γ=22): 분자 {base.mol_closed}/{base.n}, 작용제단위 {base.agent_closed}/{base.n_agents}")
    print(f"최고    ({best.metric}): 분자 {best.mol_closed}/{best.n}, 작용제단위 {best.agent_closed}/{best.n_agents}")

if __name__ == "__main__":
    main()
