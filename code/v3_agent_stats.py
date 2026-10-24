"""
T1-8/T1-9/T1-10 — 작용제 단위 통계.

심사 지적(R1-M4, R3-M1, R3-M9): 30개 취득은 작용제당 바이알 1개를 1시간 내에
3–5회 재조회한 의사반복이므로 유효 표본은 작용제 수(HN2 제외 8종)다. 취득 단위로
계산한 정확도·AUC·p값은 유효 표본을 3–4배 부풀린 것이다.

여기서 (i) 작용제 단위 정확도와 이항 신뢰구간, (ii) 짝지은 McNemar 정확검정,
(iii) 작용제를 교환 단위로 한 AUC 부트스트랩 신뢰구간, (iv) 5개 창 다중비교
보정을 계산한다.
"""
import os, sys, json, itertools, math
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
rng = np.random.default_rng(20260925)

H = pd.read_csv(f"{DATA}/v3_headtohead.csv")
out = {}

# ── (i) 작용제 단위 정확도 + Clopper–Pearson
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else stats.beta.ppf(1-a/2, k+1, n-k)
    return float(lo), float(hi)

print("=" * 78); print("작용제 단위 정확도 (획득 과반 기준)"); print("=" * 78)
tab = []
for lab, D in (("HN2 포함 (9종)", H), ("HN2 제외 (8종)", H[H.agent != "HN2"])):
    g = D.groupby("agent")[["meas_mol", "dft_mol", "dft53_mol", "meas_cls", "dft_cls"]].mean()
    n = len(g)
    r = {}
    for c in g.columns:
        k = int((g[c] > 0.5).sum()); lo, hi = cp(k, n)
        r[c] = (k, n, lo, hi)
        tab.append(dict(subset=lab, quantity=c, k=k, n=n, lo=lo, hi=hi))
    print(f"\n{lab}")
    for c, (k, nn, lo, hi) in r.items():
        print(f"  {c:<12} {k}/{nn}  95% CI [{lo:.2f}, {hi:.2f}]")
    # McNemar (짝지은 분자 수준: 실측 vs 계산)
    b = int(((g.meas_mol > 0.5) & (g.dft_mol <= 0.5)).sum())
    c_ = int(((g.meas_mol <= 0.5) & (g.dft_mol > 0.5)).sum())
    p = float(stats.binomtest(min(b, c_), b + c_, 0.5).pvalue) if b + c_ else 1.0
    print(f"  McNemar (분자): b={b}, c={c_} → 정확검정 양측 p = {p:.3f}")
    bc = int(((g.meas_cls > 0.5) & (g.dft_cls <= 0.5)).sum())
    cc = int(((g.meas_cls <= 0.5) & (g.dft_cls > 0.5)).sum())
    pc = float(stats.binomtest(min(bc, cc), bc + cc, 0.5).pvalue) if bc + cc else 1.0
    print(f"  McNemar (클래스): b={bc}, c={cc} → 양측 p = {pc if bc+cc else float('nan'):.3f}"
          + ("  (불일치 없음)" if bc + cc == 0 else ""))
    out[lab] = dict(mcnemar_mol_b=b, mcnemar_mol_c=c_, mcnemar_mol_p=p,
                    mcnemar_cls_b=bc, mcnemar_cls_c=cc,
                    mcnemar_cls_p=(pc if bc + cc else None))
pd.DataFrame(tab).to_csv(f"{DATA}/v3_agent_accuracy.csv", index=False)

# ── (ii) HN2 포함/제외 headline
print("\n" + "=" * 78); print("HN2 포함 / 제외 headline"); print("=" * 78)
rows = []
for lab, D in (("all", H), ("noHN2", H[H.agent != "HN2"])):
    rows.append(dict(subset=lab, n=len(D),
                     meas_mol=int(D.meas_mol.sum()), dft_mol=int(D.dft_mol.sum()),
                     dft53_mol=int(D.dft53_mol.sum()),
                     meas_cls=int(D.meas_cls.sum()), dft_cls=int(D.dft_cls.sum()),
                     dft53_cls=int(D.dft53_cls.sum()),
                     meas_self=D.meas_self.mean(), dft_self=D.dft_self.mean(),
                     meas_margin=D.meas_margin.mean(), dft_margin=D.dft_margin.mean()))
HN = pd.DataFrame(rows); HN.to_csv(f"{DATA}/v3_hn2_effect.csv", index=False)
print(HN.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ── (iii) AUC 작용제 단위 부트스트랩
SRC = {"computed": f"{DATA}/v3_abstention_scores.csv",
       "measured": f"{DATA}/v3_abstention_measured.csv"}
SC = ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]
def auc(x, y):
    r = stats.rankdata(np.concatenate([x, y])); n1 = len(x)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(y)))
print("\n" + "=" * 78); print("AUC — 작용제 단위 부트스트랩 95% CI (B=5000)"); print("=" * 78)
arows = []
for libtag, path in SRC.items():
    if not os.path.exists(path): continue
    S = pd.read_csv(path)
    have = [c for c in SC if c in S.columns]
    ags = sorted(S.agent.unique())
    print(f"\n[{libtag} library]  작용제 {len(ags)}종, 취득 {len(S)}건")
    for c in have:
        IN = S[S.cond == "IN"]; OU = S[S.cond == "LOCO"]
        if not len(IN) or not len(OU): continue
        a0 = auc(IN[c].values, OU[c].values); flip = a0 < 0.5
        if flip: a0 = 1 - a0
        Ia = {a: IN[IN.agent == a][c].values for a in ags}
        Oa = {a: OU[OU.agent == a][c].values for a in ags}
        bs = []
        for _ in range(5000):
            pick = rng.choice(ags, len(ags), replace=True)
            xi = np.concatenate([Ia[a] for a in pick if len(Ia[a])])
            yo = np.concatenate([Oa[a] for a in pick if len(Oa[a])])
            if len(xi) and len(yo):
                v = auc(xi, yo); bs.append(1 - v if flip else v)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        arows.append(dict(library=libtag, score=c, auc=a0, lo=float(lo), hi=float(hi),
                          includes_half=bool(lo <= 0.5 <= hi), n_agents=len(ags)))
        print(f"  {c:<14} AUC {a0:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              + ("   ← 0.5 포함" if lo <= 0.5 <= hi else ""))
pd.DataFrame(arows).to_csv(f"{DATA}/v3_auc_ci.csv", index=False)

# ── (iv) 5개 창 다중비교 — 양측 검정을 직접 계산한다.
#    (이전 코드는 단측 p에서 양측을 유도했는데, 이산 순위검정에서 단측 p = 1.000은
#     반대 방향 단측 p가 최소값 1/C(9,3) = 0.0119라는 뜻이므로 그 유도는 틀린다.)
RBA = pd.read_csv(f"{DATA}/v3_region_by_agent.csv", index_col=0)
PCONT = {"GD", "GF", "VX", "A230", "A232", "A234", "A242"}
prow = []
for col in RBA.columns:
    pf_ = RBA.loc[[a for a in RBA.index if a in PCONT], col].dropna().values
    np_ = RBA.loc[[a for a in RBA.index if a not in PCONT], col].dropna().values
    u = stats.mannwhitneyu(pf_, np_, alternative="two-sided")
    prow.append(dict(region=col, n_p=len(pf_), n_np=len(np_),
                     mean_p=float(pf_.mean()), mean_np=float(np_.mean()),
                     delta=float(np_.mean() - pf_.mean()), p_two=float(u.pvalue)))
pf = pd.DataFrame(prow)
m = len(pf)
pf["p_bonf"] = np.minimum(1.0, pf.p_two * m)
o = np.argsort(pf.p_two.values); holm = np.empty(m); run = 0.0
for rank, i2 in enumerate(o):
    run = max(run, (m - rank) * pf.p_two.values[i2]); holm[i2] = min(1.0, run)
pf["p_holm"] = holm
pf.to_csv(f"{DATA}/v3_pf_multiplicity.csv", index=False)
print("\n" + "=" * 86); print("인 함유 대비 — 양측 Mann--Whitney + 다중비교 보정 (5개 창, 6 대 3 분자)")
print("=" * 86)
print(f"{'구간':<26}{'P계열':>8}{'비인':>8}{'Δ':>9}{'p 양측':>9}{'Bonf':>8}{'Holm':>8}")
for _, r in pf.iterrows():
    print(f"{r.region:<26}{r.mean_p:>8.3f}{r.mean_np:>8.3f}{r.delta:>+9.3f}"
          f"{r.p_two:>9.3f}{r.p_bonf:>8.3f}{r.p_holm:>8.3f}")
print(f"\n보정 후 유의한 구간: {int((pf.p_holm < 0.05).sum())}개 "
      f"(6 대 3 분자에서 양측 최소 달성 가능 p = {2/math.comb(9,3):.4f})")

json.dump(out, open(f"{DATA}/v3_agent_stats_meta.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved v3_agent_accuracy.csv, v3_hn2_effect.csv, v3_pf_multiplicity.csv")
