"""
v3 B2 판정 — P 기술 개선이 600–850 cm-1 결함을 고쳤는가.

A1 진단: 측정↔계산 갭이 600–850(P–F·P–N)에 집중, 인 함유 작용제에만 나타남.
         GF 0.562 / A-232 0.205  vs  비인 작용제 0.83–0.97.

이 스크립트는 orca/outputs_b2/ 에 있는 B2_*.out 을 전부 찾아
해당 작용제의 baseline 계산을 그것으로 갈아끼우고 다음을 비교한다.

  1) 600–850 구간 측정↔계산 코사인      ← 1차 판정 기준
  2) 전 구간 자기 코사인
  3) 구간별 분해 (baseline 대비 델타)
  4) 그 작용제의 분자 top-1이 뒤집히는가 (53종 open-set)

보정(s_low, s_high)은 두 조건에서 **각각 다시 적합**한다 —
기저·범함수·용매가 바뀌면 최적 스케일도 바뀌므로, baseline 스케일을
그대로 쓰면 개선을 과소평가한다. Γ=22는 실측값이라 양쪽 공통.
"""
import os, sys, glob, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.spectra import parse_raman
from v3_step3_headtohead import (load_measured, load_library, synth_native, fit_scale,
                                 measured_peaks, cos, cls_of, JOB, CLASS, AGENTS, GAMMA)

REG = [("250–600", 250, 600), ("600–850", 600, 850), ("850–1150", 850, 1150),
       ("1150–1400", 1150, 1400), ("1400–1800", 1400, 1800)]
TAG2AGENT = {"GF": "GF", "A232": "A232", "A230": "A230", "A234": "A234", "GD": "GD", "HD": "HD"}


def region_cos(X, meas, spec):
    out = {}
    for name, lo, hi in REG:
        m = (X >= lo) & (X < hi)
        out[name] = cos(meas[m], spec[m])
    return out


def main():
    outdir = os.path.join(ROOT, "orca", "outputs_b2")
    files = sorted(glob.glob(os.path.join(outdir, "B2_*.out")))
    done = []
    for f in files:
        txt_ok = False
        with open(f, "r", errors="ignore") as fh:
            txt_ok = "ORCA TERMINATED NORMALLY" in fh.read()
        if txt_ok:
            done.append(f)
        else:
            print(f"  (미완료 건너뜀: {os.path.basename(f)})")
    if not done:
        print("완료된 B2 계산이 없습니다. orca/outputs_b2/ 확인.")
        return
    print(f"완료된 B2 계산 {len(done)}개\n")

    X, acq = load_measured()
    base_lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]

    # ---------- baseline ----------
    rb, SLb, SHb, nb = fit_scale(base_lib, mpk, calib)
    base_spec = {k: synth_native(v, X, SLb, SHb) for k, v in base_lib.items()}
    print(f"[baseline] s=({SLb:.3f},{SHb:.3f})  위치 RMSE {rb:.1f} cm-1 (n={nb})")

    rows = []
    for f in done:
        name = os.path.basename(f)[:-4]            # B2_GF_V1_cpcm
        _, tag, *rest = name.split("_")
        variant = "_".join(rest)
        agent = TAG2AGENT.get(tag)
        if agent is None:
            print(f"  ! 작용제 매핑 없음: {name}"); continue
        job = JOB[agent]

        try:
            peaks = parse_raman(f)
        except Exception as e:
            print(f"  ! 파싱 실패 {name}: {e}"); continue

        # 라이브러리에서 해당 항목만 교체하고 스케일 재적합
        lib2 = dict(base_lib); lib2[job] = peaks
        r2, SL2, SH2, n2 = fit_scale(lib2, mpk, calib)
        spec2 = {k: synth_native(v, X, SL2, SH2) for k, v in lib2.items()}

        cb = region_cos(X, mean[agent], base_spec[job])
        cn = region_cos(X, mean[agent], spec2[job])
        self_b = cos(mean[agent], base_spec[job])
        self_n = cos(mean[agent], spec2[job])

        # open-set 분자 top-1 (53종)
        def top1(sp):
            sims = {k: cos(mean[agent], v) for k, v in sp.items()}
            return max(sims, key=sims.get)
        t_b, t_n = top1(base_spec), top1(spec2)

        rows.append(dict(job=name, agent=agent, variant=variant,
                         s_low=SL2, s_high=SH2, pos_rmse=r2, n_pairs=n2,
                         self_base=self_b, self_new=self_n,
                         **{f"base_{k}": v for k, v in cb.items()},
                         **{f"new_{k}": v for k, v in cn.items()},
                         top1_base=t_b, top1_new=t_n,
                         correct_base=(t_b == job), correct_new=(t_n == job)))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(ROOT, "data", "v3_b2_evaluation.csv"), index=False)

    print("\n" + "=" * 76)
    print("★ 1차 판정 — 600–850 cm-1 측정↔계산 코사인")
    print("=" * 76)
    print(f"{'계산':<24}{'baseline':>10}{'개선본':>10}{'Δ':>9}   판정")
    for _, r in R.iterrows():
        b, n = r["base_600–850"], r["new_600–850"]
        d = n - b
        verdict = "★ 유의한 개선" if d > 0.10 else ("개선" if d > 0.03 else
                  ("변화 없음" if d > -0.03 else "악화"))
        print(f"{r['job']:<24}{b:>10.3f}{n:>10.3f}{d:>+9.3f}   {verdict}")

    print("\n전 구간 자기 코사인:")
    for _, r in R.iterrows():
        print(f"  {r['job']:<24}{r['self_base']:>8.3f} → {r['self_new']:>6.3f}"
              f"  ({r['self_new']-r['self_base']:+.3f})")

    print("\n구간별 델타 (개선본 − baseline):")
    hdr = "  " + f"{'계산':<24}" + "".join(f"{n:>12}" for n, _, _ in REG)
    print(hdr)
    for _, r in R.iterrows():
        line = f"  {r['job']:<24}"
        for n, _, _ in REG:
            line += f"{r[f'new_{n}'] - r[f'base_{n}']:>+12.3f}"
        print(line)

    print("\nopen-set 분자 top-1 (53종):")
    for _, r in R.iterrows():
        mark = "✓" if r["correct_new"] else "✗"
        was = "✓" if r["correct_base"] else "✗"
        print(f"  {r['job']:<24} {was} {r['top1_base']:<18} → {mark} {r['top1_new']}")

    print("\n재적합된 스케일 인자:")
    for _, r in R.iterrows():
        print(f"  {r['job']:<24} s=({r['s_low']:.3f},{r['s_high']:.3f})  "
              f"위치 RMSE {r['pos_rmse']:.1f} (n={int(r['n_pairs'])})")
    print(f"  {'[baseline]':<24} s=({SLb:.3f},{SHb:.3f})  위치 RMSE {rb:.1f} (n={nb})")

    print("\n" + "=" * 76)
    best = R.assign(delta=R["new_600–850"] - R["base_600–850"]).sort_values("delta",
                                                                            ascending=False)
    b0 = best.iloc[0]
    if b0["delta"] > 0.10:
        print(f"결론: {b0['variant']} 가 600–850을 {b0['delta']:+.3f} 개선했다.")
        print("      → 이 방향으로 A-232 확인 후 전수 재계산 권고.")
    elif b0["delta"] > 0.03:
        print(f"결론: {b0['variant']} 가 소폭({b0['delta']:+.3f}) 개선. 다른 변형도 확인 필요.")
    else:
        print("결론: 시험한 변형 중 600–850을 의미 있게 고친 것이 없다.")
        print("      → 남은 후보: wB97X-D3(V3), P–F 비조화 보정, 또는 진단 재검토.")
    print("=" * 76)
    print("\nsaved data/v3_b2_evaluation.csv")


if __name__ == "__main__":
    main()
