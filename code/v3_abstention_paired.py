"""
Paired analysis of the class-absent condition.

The class-present and class-absent conditions are the SAME acquisitions under two
treatments, not two independent samples: every acquisition appears under cond=IN and
again under cond=LOCO with its own toxicological class deleted from the library. The
pooled AUC of Table tab:si-absten discards that pairing. This script reports the paired
statistic instead -- for each score, how many acquisitions move in the expected direction
and the exact two-sided sign test -- and recomputes the AUC on the agent as the unit so
that the dependence of the rank statistic on the resampling unit is stated empirically
rather than asserted.

Produces data/v3_abstention_paired.csv and SI table \\label{tab:si-abstpaired}.
"""
import os
from math import comb
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES = ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]
NICE = {"S1_top1": "top-1 cosine", "S2_margin": "margin",
        "S3_psr": "peak-to-sidelobe", "S4_libz": "library-normalized $z$",
        "S5_resid": "unmixing residual", "S6_bandfrac": "band-match fraction"}
# Direction in which the score is expected to be LARGER when the class is present.
# The unmixing residual is a misfit, so it runs the other way.
HIGHER_IS_IN = {"S1_top1": True, "S2_margin": True, "S3_psr": True,
                "S4_libz": True, "S5_resid": False, "S6_bandfrac": True}


def sign_test(k, n):
    """Exact two-sided sign test, k successes of n non-tied pairs."""
    if n == 0:
        return float("nan")
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main():
    D = pd.read_csv(os.path.join(ROOT, "data", "v3_abstention_scores.csv"))
    D = D[D.agent != "HN2"]                      # a condition of its own, as elsewhere
    wide = D.pivot_table(index=["agent", "rep"], columns="cond", values=SCORES)

    rows = []
    for s in SCORES:
        a, b = wide[(s, "IN")], wide[(s, "LOCO")]
        keep = a.notna() & b.notna()
        d = (a[keep] - b[keep]) * (1 if HIGHER_IS_IN[s] else -1)
        n_nontied = int((d != 0).sum())
        k = int((d > 0).sum())
        p = sign_test(k, n_nontied)
        # unpaired AUC on the acquisition, and on the agent mean
        iv = D[D.cond == "IN"]; lv = D[D.cond == "LOCO"]
        sgn = 1 if HIGHER_IS_IN[s] else -1
        a_acq = auc(sgn * iv[s].dropna(), sgn * lv[s].dropna())
        ia = iv.groupby("agent")[s].mean(); la = lv.groupby("agent")[s].mean()
        a_ag = auc(sgn * ia, sgn * la)
        rows.append(dict(score=s, k=k, n=n_nontied, p_sign=p,
                         auc_acquisition=round(a_acq, 3), auc_agent=round(a_ag, 3),
                         n_agents=int(ia.index.size)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(ROOT, "data", "v3_abstention_paired.csv"), index=False)

    def fmt_p(p):
        return "$<10^{-7}$" if p < 1e-7 else (f"{p:.4f}" if p < 0.001 else f"{p:.3f}")

    body = [f"{NICE[r.score]} & {r.k}/{r.n} & {fmt_p(r.p_sign)} & "
            f"{r.auc_acquisition:.3f} & {r.auc_agent:.2f} \\\\" for r in R.itertuples()]
    nsig = int((R.p_sign < 0.05).sum())
    spread = (R.auc_agent - R.auc_acquisition).abs().max()

    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{The class-absent test is paired, and is reported here as such. Each "
           "of the %d acquisitions is scored twice, once against the full computed library "
           "and once with its own toxicological class deleted, so the two conditions are "
           "one sample under two treatments. The second column counts the acquisitions "
           "that move in the expected direction; the third is the exact two-sided sign "
           "test over the non-tied pairs. The two AUC columns are the same rank statistic "
           "computed on the acquisition and on the agent mean. The vial labeled HN2 is a "
           "condition of its own and is excluded, as elsewhere.}\n"
           % int(R.n.max()) +
           "\\label{tab:si-abstpaired}\n\\small\n"
           "\\begin{tabular}{lrrrr}\n\\toprule\n"
           "Score & Moves as expected & Sign test & AUC (acquisition) & AUC (agent) \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize On the paired statistic %d of the six scores "
           "separate the two conditions at the 5\\,\\%% level, and three do so without a "
           "single acquisition out of order. The paired test is the stronger statement and "
           "we report it as the primary one; the AUC is retained as an effect size because "
           "it is what the open-set literature quotes. The two AUC columns differ by at "
           "most %.3f, so the rank statistic is close to but not independent of the "
           "resampling unit, which is an empirical property of these data and not a "
           "theorem. None of this changes the conclusion of "
           "Table~\\ref{tab:si-absten}: the scores order the two conditions, and no "
           "threshold separates them without error. Produced by "
           "\\texttt{code/v3\\_abstention\\_paired.py}.\\end{flushleft}\n\\end{table}\n"
           % (nsig, spread))
    open(os.path.join(ROOT, "latex", "sitables", "abstpaired.tex"), "w").write(tex)
    print(R.to_string(index=False))
    print("saved data/v3_abstention_paired.csv + latex/sitables/abstpaired.tex")


if __name__ == "__main__":
    main()
