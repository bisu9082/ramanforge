"""
v3 — 순환 보정 대안의 재현. 에디터가 인쇄본에 남기라고 요청한 시연.

식별 정확도를 목적함수로 (s, Gamma) 격자탐색을 하면 어떤 수가 나오고,
그 수가 얼마나 불안정한지를 보인다. 이 보정은 채택하지 않는다.
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, synth_native, cos,
                                 JOB, AGENTS, cls_of, CUT)

X, acq = load_measured(); lib = load_library()
queries = [(a, r, s) for a in AGENTS for r, s in acq[a] if len(acq[a]) >= 3]
print(f"쿼리 {len(queries)}건 / {len(set(q[0] for q in queries))}종")

S_GRID = np.round(np.arange(0.960, 1.0401, 0.004), 4)
G_GRID = np.arange(10, 51, 2)
res = np.zeros((len(G_GRID), len(S_GRID)), int)
for gi, g in enumerate(G_GRID):
    for si, s in enumerate(S_GRID):
        spec = {k: synth_native(v, X, s, s, gamma=float(g)) for k, v in lib.items()}
        n = 0
        for a, r, y in queries:
            job = JOB.get(a)
            if job is None: continue
            best = max(spec, key=lambda k: cos(y, spec[k]))
            n += int(best == job)
        res[gi, si] = n
best = np.unravel_index(res.argmax(), res.shape)
bg, bs, bn = G_GRID[best[0]], S_GRID[best[1]], res.max()
ncell = int((res == bn).sum())
print(f"\n최대 {bn}/{len(queries)}  at  s={bs:.3f}, Gamma={bg}")
print(f"  같은 최댓값을 내는 격자칸: {ncell} / {res.size}")
si = list(S_GRID).index(bs)
for d in (1, 2):
    for sgn in (-1, 1):
        j = si + sgn * d
        if 0 <= j < len(S_GRID):
            print(f"  s {S_GRID[j]:.3f} (Δ{(S_GRID[j]/bs-1)*100:+.1f}%) → {res[best[0], j]}/{len(queries)}")
# 채택 보정에서의 값
M = json.load(open(f"{ROOT}/data/v3_headtohead_meta.json"))
spec = {k: synth_native(v, X, M["s_low"], M["s_high"], gamma=22.0) for k, v in lib.items()}
n = sum(int(max(spec, key=lambda k: cos(y, spec[k])) == JOB[a])
        for a, r, y in queries if a in JOB)
print(f"\n채택 보정 (s={M['s_low']}/{M['s_high']}, Γ=22): {n}/{len(queries)}")
pd.DataFrame(res, index=[f"{g}" for g in G_GRID],
             columns=[f"{s:.3f}" for s in S_GRID]).to_csv(f"{ROOT}/data/v3_calibgrid.csv")
json.dump(dict(best_n=int(bn), best_s=float(bs), best_gamma=int(bg), n_cells=ncell,
               n_total=int(res.size), n_queries=len(queries), adopted_n=int(n)),
          open(f"{ROOT}/data/v3_calibgrid_meta.json", "w"), indent=1)
