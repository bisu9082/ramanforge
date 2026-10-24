"""
T1-1: measured library leave-one-molecule-out (actual ablation).

The earlier v3_measured_only_loo.csv was an output in which the ablation had not been
applied (the query was assigned to itself with a similarity of 0.999). Here the removal
is actually performed.

Definition of the ablation: **all** reference acquisitions of the query agent are removed
from the library. The remaining entries are therefore always other molecules, and the entry
counts before and after are recorded together so that the removal can be verified to have
taken place.
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
        lib = {a: v for a, v in full.items() if a != q_ag}      # <- the actual ablation
        n_after = len(lib)
        assert q_ag not in lib, "ablation failed"
        sims = {k: cos(q, v) for k, v in lib.items()}
        o = sorted(sims.items(), key=lambda t: -t[1])
        top, ts = o[0]
        rows.append(dict(agent=q_ag, rep=q_rep, n_before=n_before, n_after=n_after,
                         top1=top, cls_ok=(CLASS[top] == CLASS[q_ag]),
                         top_cos=ts, margin=ts - (o[1][1] if len(o) > 1 else 0.0)))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(ROOT, "data", "v3_measured_only_loo.csv"), index=False)

print("Measured library leave-one-molecule-out (own molecule fully removed)")
print(f"{'agent':<7}{'n':>3}{'lib n pre/post':>16}{'falls to':<22}{'class kept':>10}{'top cos':>9}{'margin':>8}")
print("-" * 78)
for a in queryable:
    s = D[D.agent == a]
    t = s.top1.value_counts()
    dest = ", ".join(f"{k}" + (f"×{v}" if v > 1 else "") for k, v in t.items())
    print(f"{a:<7}{len(s):>3}{f'{s.n_before.iloc[0]}→{s.n_after.iloc[0]}':>16}{dest:<22}"
          f"{int(s.cls_ok.sum())}/{len(s):<8}{s.top_cos.mean():>9.3f}{s.margin.mean():>8.3f}")
print("-" * 78)
print(f"{'total':<7}{len(D):>3}{'':>16}{'':<22}{int(D.cls_ok.sum())}/{len(D):<8}"
      f"{D.top_cos.mean():>9.3f}{D.margin.mean():>8.3f}")
print(f"\ncases falling to itself: {(D.top1 == D.agent).sum()}  (0 is expected)")
print("saved data/v3_measured_only_loo.csv")
