"""
Abstention AUC stratified by the deleted toxicological class.

The pooled AUC of Section S7 mixes three deletions that are not equally hard. Under the
leave-one-class-out condition the class removed from the library is the query's own, so
stratifying by the query's class stratifies by which deletion was performed. Written in
response to the observation that the pooled figure hides a large difference between them.

Reads data/v3_abstention_scores.csv (written by v3_step5_a2_abstention.py) and writes
data/v3_abstention_bycls.csv and latex/sitables/abstbycls.tex.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]
NAME = {"S1_top1": "top-1 cosine", "S2_margin": "margin", "S3_psr": "peak-to-sidelobe",
        "S4_libz": "library $z$", "S5_resid": "residual", "S6_bandfrac": "band fraction"}
# S5 is a residual: a smaller value means a better match, so its orientation is reversed.
HIGHER = {s: (s != "S5_resid") for s in SC}


def auc(pos, neg, higher=True):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if not higher:
        pos, neg = -pos, -neg
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main():
    D = pd.read_csv(os.path.join(ROOT, "data", "v3_abstention_scores.csv"))
    # Same convention as Section S7: the vial labeled HN2 is a condition of its own.
    D = D[D.agent != "HN2"]
    rows = []
    groups = [("all", D)] + [(c, g) for c, g in D.groupby("cls")]
    for cls, g in groups:
        i = g[g.cond == "IN"]; l = g[g.cond == "LOCO"]
        r = dict(deleted_class=cls, n_in=len(i), n_loco=len(l),
                 agents=", ".join(sorted(set(g.agent))))
        for s in SC:
            r[s] = auc(i[s], l[s], HIGHER[s])
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(ROOT, "data", "v3_abstention_bycls.csv"), index=False)

    print(f"{'deleted class':<12}{'n':>4}   " + "".join(f"{NAME[s].split()[0]:>11}" for s in SC))
    for r in T.itertuples():
        print(f"{r.deleted_class:<12}{r.n_in:>4}   " +
              "".join(f"{getattr(r, s):>11.2f}" for s in SC))

    body = []
    for r in T.itertuples():
        lab = "All three (pooled)" if r.deleted_class == "all" else r.deleted_class
        body.append(f"{lab} & {r.n_in} & " +
                    " & ".join(f"{getattr(r, s):.2f}" for s in SC) + " \\\\")
        if r.deleted_class == "all":
            body.append("\\addlinespace")
    nov = T[T.deleted_class == "Novichok"].iloc[0]
    ner = T[T.deleted_class == "Nerve"].iloc[0]
    tex = """\\begin{table}[htbp]
\\centering
\\caption{Abstention AUC stratified by the class deleted from the library. Under the
leave-one-class-out condition the class removed is the query's own, so this is a
stratification by deletion. $n$ is the number of in-library queries in the stratum, each
compared against the same number of class-absent queries. Produced by
\\texttt{code/v3\\_abstention\\_bycls.py}.}
\\label{tab:si-abstbycls}
\\small
\\begin{tabular}{lrrrrrrr}
\\toprule
Deleted class & $n$ & """ + " & ".join(NAME[s] for s in SC) + """ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}

\\begin{flushleft}\\footnotesize The pooled figure averages strata that behave differently.
Deleting the nerve agents is detected: three of the six scores separate that deletion
perfectly on the """ + str(int(ner.n_in)) + """ queries available. Deleting the Novichok
agents is not: no score exceeds """ + f"{max(getattr(nov, s) for s in SC):.2f}" + """, and the
A-series supplies """ + str(int(nov.n_in)) + """ of the thirty queries, so it dominates the
pooled value. The abstention failure reported in Section~\\ref{si:absten} is therefore
specific to the A-series rather than general, which is consistent with the identification
result: the A-series is also where the computed library is least accurate. Each stratum
rests on a few queries and these are point estimates without intervals.\\end{flushleft}
\\end{table}
"""
    open(os.path.join(ROOT, "latex", "sitables", "abstbycls.tex"), "w").write(tex)
    print("\nsaved data/v3_abstention_bycls.csv + latex/sitables/abstbycls.tex")


if __name__ == "__main__":
    main()
