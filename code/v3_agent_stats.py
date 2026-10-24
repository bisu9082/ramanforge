"""
T1-8/T1-9/T1-10: agent-level statistics.

Reviewer comment (R1-M4, R3-M1, R3-M9): the 30 acquisitions are pseudo-replicates obtained by
re-interrogating a single vial per agent 3–5 times within one hour, so the effective sample is
the number of agents (8 species, excluding HN2). Accuracies, AUCs and p values computed per
acquisition inflate the effective sample by a factor of 3 to 4.

Here we compute (i) agent-level accuracy with binomial confidence intervals, (ii) the paired
exact McNemar test, (iii) bootstrap confidence intervals for the AUC with the agent as the
exchangeable unit, and (iv) multiple-comparison correction over the 5 windows.
"""
import os, sys, json, itertools, math
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
rng = np.random.default_rng(20260925)

H = pd.read_csv(f"{DATA}/v3_headtohead.csv")
out = {}

# -- (i) per-agent accuracy with Clopper-Pearson intervals
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else stats.beta.ppf(1-a/2, k+1, n-k)
    return float(lo), float(hi)

print("=" * 78); print("Agent-level accuracy (majority of acquisitions)"); print("=" * 78)
tab = []
for lab, D in (("HN2 included (9 agents)", H), ("HN2 excluded (8 agents)", H[H.agent != "HN2"])):
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
    # McNemar (paired at the molecule level: measured vs computed)
    b = int(((g.meas_mol > 0.5) & (g.dft_mol <= 0.5)).sum())
    c_ = int(((g.meas_mol <= 0.5) & (g.dft_mol > 0.5)).sum())
    p = float(stats.binomtest(min(b, c_), b + c_, 0.5).pvalue) if b + c_ else 1.0
    print(f"  McNemar (molecule): b={b}, c={c_} → exact test two-sided p = {p:.3f}")
    bc = int(((g.meas_cls > 0.5) & (g.dft_cls <= 0.5)).sum())
    cc = int(((g.meas_cls <= 0.5) & (g.dft_cls > 0.5)).sum())
    pc = float(stats.binomtest(min(bc, cc), bc + cc, 0.5).pvalue) if bc + cc else 1.0
    print(f"  McNemar (class): b={bc}, c={cc} → two-sided p = {pc if bc+cc else float('nan'):.3f}"
          + ("  (no mismatches)" if bc + cc == 0 else ""))
    out[lab] = dict(mcnemar_mol_b=b, mcnemar_mol_c=c_, mcnemar_mol_p=p,
                    mcnemar_cls_b=bc, mcnemar_cls_c=cc,
                    mcnemar_cls_p=(pc if bc + cc else None))
pd.DataFrame(tab).to_csv(f"{DATA}/v3_agent_accuracy.csv", index=False)

# -- (ii) headline with and without HN2
print("\n" + "=" * 78); print("HN2 included / excluded headline"); print("=" * 78)
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

# -- (iii) agent-level bootstrap of the AUC
SRC = {"computed": f"{DATA}/v3_abstention_scores.csv",
       "measured": f"{DATA}/v3_abstention_measured.csv"}
SC = ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]
def auc(x, y):
    r = stats.rankdata(np.concatenate([x, y])); n1 = len(x)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(y)))
print("\n" + "=" * 78); print("AUC: agent-level bootstrap 95% CI (B=5000)"); print("=" * 78)
arows = []
for libtag, path in SRC.items():
    if not os.path.exists(path): continue
    S = pd.read_csv(path)
    have = [c for c in SC if c in S.columns]
    ags = sorted(S.agent.unique())
    print(f"\n[{libtag} library]  {len(ags)} agents, {len(S)} acquisitions")
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
              + ("   ← includes 0.5" if lo <= 0.5 <= hi else ""))
pd.DataFrame(arows).to_csv(f"{DATA}/v3_auc_ci.csv", index=False)

# -- (iv) multiple comparison over the five windows; the two-sided test is computed directly
#     (the earlier code derived the two-sided p from the one-sided one, but in a discrete rank test
#      a one-sided p of 1.000 means the opposite-tail p is at its minimum of 1/C(9,3) = 0.0119, so that derivation is wrong)
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
print("\n" + "=" * 86); print("Phosphorus contrast: two-sided Mann--Whitney + multiple-comparison correction (5 windows, 6 vs 3 molecules)")
print("=" * 86)
print(f"{'region':<26}{'P-series':>8}{'P-free':>8}{'Δ':>9}{'p':>9}{'Bonf':>8}{'Holm':>8}")
for _, r in pf.iterrows():
    print(f"{r.region:<26}{r.mean_p:>8.3f}{r.mean_np:>8.3f}{r.delta:>+9.3f}"
          f"{r.p_two:>9.3f}{r.p_bonf:>8.3f}{r.p_holm:>8.3f}")
print(f"\nsignificant regions after correction: {int((pf.p_holm < 0.05).sum())} "
      f"(smallest attainable two-sided p for 6 vs 3 molecules = {2/math.comb(9,3):.4f})")

json.dump(out, open(f"{DATA}/v3_agent_stats_meta.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved v3_agent_accuracy.csv, v3_hn2_effect.csv, v3_pf_multiplicity.csv")
