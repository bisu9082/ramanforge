"""
T1-1 — 측정 라이브러리 leave-one-molecule-out (실제 절제).

이전 버전의 v3_measured_only_loo.csv는 절제가 적용되지 않은 출력이었다
(질의가 자기 자신에게 유사도 0.999로 배정됨). 여기서 실제로 삭제한다.

절제 정의: 질의 작용제의 **모든** 참조 획득을 라이브러리에서 제거한다.
따라서 남는 항목은 항상 다른 분자이며, before/after 항목 수를 함께 기록해
삭제가 실제로 일어났음을 검증 가능하게 만든다.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import load_measured, cos, AGENTS, CLASS

X, acq = load_measured()
queryable = [a for a in AGENTS if len(acq[a]) >= 2]
rows = []
for q_ag in queryable:
    for q_rep, q in acq[q_ag]:
        full = {a: np.mean([s for r, s in acq[a]], 0) for a in AGENTS}
        n_before = len(full)
        lib = {a: v for a, v in full.items() if a != q_ag}      # ← 실제 절제
        n_after = len(lib)
        assert q_ag not in lib, "절제 실패"
        sims = {k: cos(q, v) for k, v in lib.items()}
        o = sorted(sims.items(), key=lambda t: -t[1])
        top, ts = o[0]
        rows.append(dict(agent=q_ag, rep=q_rep, n_before=n_before, n_after=n_after,
                         top1=top, cls_ok=(CLASS[top] == CLASS[q_ag]),
                         top_cos=ts, margin=ts - (o[1][1] if len(o) > 1 else 0.0)))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(ROOT, "data", "v3_measured_only_loo.csv"), index=False)

print("측정 라이브러리 leave-one-molecule-out (자기 분자 전체 삭제)")
print(f"{'작용제':<7}{'n':>3}{'라이브러리 전/후':>16}{'낙하처':<22}{'클래스유지':>10}{'top cos':>9}{'마진':>8}")
print("-" * 78)
for a in queryable:
    s = D[D.agent == a]
    t = s.top1.value_counts()
    dest = ", ".join(f"{k}" + (f"×{v}" if v > 1 else "") for k, v in t.items())
    print(f"{a:<7}{len(s):>3}{f'{s.n_before.iloc[0]}→{s.n_after.iloc[0]}':>16}{dest:<22}"
          f"{int(s.cls_ok.sum())}/{len(s):<8}{s.top_cos.mean():>9.3f}{s.margin.mean():>8.3f}")
print("-" * 78)
print(f"{'전체':<7}{len(D):>3}{'':>16}{'':<22}{int(D.cls_ok.sum())}/{len(D):<8}"
      f"{D.top_cos.mean():>9.3f}{D.margin.mean():>8.3f}")
print(f"\n자기 자신으로 낙하한 건수: {(D.top1 == D.agent).sum()}  (0이어야 정상)")
print("saved data/v3_measured_only_loo.csv")
