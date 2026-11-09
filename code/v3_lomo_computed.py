"""The condition the method exists to serve: the query's own molecule is not in the library.

Section 2 declares this test -- each agent's own computed entry is deleted before that
agent's spectrum is queried, and the class of the best remaining match is recorded -- and
it is the condition a computed library is built for, because the reason to compute a
reference is that no measured one exists. The scores for it are already produced by
code/v3_step5_a2_abstention.py, which writes data/v3_abstention_scores.csv with three
conditions per acquisition: IN (the whole library), LOMO (the query's own molecule
deleted) and LOCO (its whole class deleted).

This script reports the LOMO condition against IN at the class level, which is the only
level at which LOMO has an answer: with the molecule deleted there is no correct molecular
entry to find. It also asks whether the top-1 cosine separates the acquisitions the
deleted-molecule condition assigns to the wrong class from those it still assigns to the
right one, because an operational recommendation that rests on class accuracy needs a
confidence score that flags the class errors.

Writes data/v3_lomo_computed.csv and latex/sitables/lomocomputed.tex.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS = {"GD": "Nerve", "GF": "Nerve", "VX": "Nerve",
         "HD": "Blister", "HN1": "Blister", "HN2": "Blister", "HN3": "Blister",
         "A230": "Novichok", "A232": "Novichok", "A234": "Novichok", "A242": "Novichok"}
NICE = {"GD": "GD", "GF": "GF", "HN1": "HN1", "HN3": "HN3",
        "A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}
# the library entry names are not agent names; this is only for the report
ENTRY = {"EXP_HN2": "computed HN2", "OP_nPr_iBu": "an OP analog",
         "OP_Me_Me": "an OP analog", "OP_Me_sBu": "an OP analog"}
ENTRY_CLASS = {"EXP_HN2": "Blister", "OP_nPr_iBu": "Nerve",
               "OP_Me_Me": "Nerve", "OP_Me_sBu": "Nerve"}


def main():
    S = pd.read_csv(os.path.join(ROOT, "data", "v3_abstention_scores.csv"))
    S = S[S.agent != "HN2"]                      # a condition of its own, as elsewhere
    rows = []
    for a in sorted(S.agent.unique(), key=lambda x: (CLASS[x], x)):
        gin = S[(S.agent == a) & (S.cond == "IN")]
        glo = S[(S.agent == a) & (S.cond == "LOMO")]
        wrong = glo[~glo.cls_correct]
        dest = wrong.top1_name.value_counts()
        rows.append(dict(
            agent=NICE[a], cls=CLASS[a], n=len(glo),
            in_correct=int(gin.cls_correct.sum()),
            lomo_correct=int(glo.cls_correct.sum()),
            dest="--" if dest.empty else
                 "; ".join(f"{ENTRY.get(k, k)} ({ENTRY_CLASS.get(k, '?')}) $\\times${v}"
                           for k, v in dest.items())))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_lomo_computed.csv"), index=False)

    lo = S[S.cond == "LOMO"]
    ok = lo[lo.cls_correct].S1_top1.to_numpy()
    bad = lo[~lo.cls_correct].S1_top1.to_numpy()
    above = int((bad > ok.min()).sum())
    tot_in, tot_lo, tot_n = D.in_correct.sum(), D.lomo_correct.sum(), D.n.sum()
    ag_in = int((D.in_correct == D.n).sum())
    ag_lo = int((D.lomo_correct == D.n).sum())

    body = [f"{r.agent} & {r.cls} & {r.n} & {r.in_correct}/{r.n} & "
            f"{r.lomo_correct}/{r.n} & {r.dest} \\\\" for r in D.itertuples()]
    cap = ("The computed library when the query's own molecule is absent from it. "
           "This is the condition declared in Section 2 and the one a computed library "
           "exists to serve: a reference is computed because none has been measured, so "
           "the molecule in front of the instrument is not in the library. Each agent's "
           "own computed entry is deleted before its acquisitions are queried, and the "
           "class of the best remaining match is recorded. Only the class level has an "
           "answer here, since no correct molecular entry remains. The \\emph{in library} "
           "column is the same quantity with the entry present, i.e. the closed-set "
           "figure the main text reports. The vial labeled HN2 is excluded, as elsewhere; "
           "the library entry named \\emph{computed HN2} is the computed nitrogen-mustard "
           "structure and not that vial's measurement.")
    note = (f"With the molecule present the computed library is class-correct for "
            f"{tot_in} of {tot_n} acquisitions and for {ag_in} of {len(D)} agents; with it "
            f"absent, {tot_lo} of {tot_n} and {ag_lo} of {len(D)}. The loss is not spread "
            "evenly and it is not in a harmless direction. All four acquisitions of soman "
            "are assigned to the computed nitrogen-mustard entry, a nerve-agent-to-"
            "vesicant crossing of the kind that changes a prophylaxis decision, and the "
            "three Novichok errors go to organophosphorus analogs, i.e. to the nerve "
            "class. A paired exact test at the agent level gives $b$ = 4, $c$ = 0, "
            "two-sided $p$ = 0.125, the same power limit as the molecular-level test. "
            "The top-1 cosine does not flag the errors: over the deleted-molecule "
            "condition the class-correct acquisitions run "
            f"{ok.min():.3f}--{ok.max():.3f} with a median of {np.median(ok):.3f} and the "
            f"class-incorrect ones {bad.min():.3f}--{bad.max():.3f} with a median of "
            f"{np.median(bad):.3f}, so the wrong answers are if anything the more "
            f"confident, and {above} of the {len(bad)} sit above the lowest "
            "class-correct value. No threshold on that score separates them. Produced by "
            "\\texttt{code/v3\\_lomo\\_computed.py}.")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{" + cap + "}\n"
           "\\label{tab:si-lomocomputed}\n\\small\n"
           "\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{llrrrl}\n\\toprule\n"
           " & & & \\multicolumn{2}{c}{Class-correct} & \\\\\n"
           "\\cmidrule(lr){4-5}\n"
           "Agent & Class & Acq. & In library & Molecule deleted & "
           "Best remaining match when wrong \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\midrule\n"
           f"\\textbf{{Total}} & & {tot_n} & \\textbf{{{tot_in}/{tot_n}}} & "
           f"\\textbf{{{tot_lo}/{tot_n}}} & \\\\\n"
           "\\bottomrule\n\\end{tabular}\n}\n\n"
           "\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}\n"
           "\\end{table}\n")
    open(os.path.join(ROOT, "latex", "sitables", "lomocomputed.tex"), "w").write(tex)
    print(D.to_string(index=False))
    print(f"\nIN {tot_in}/{tot_n} ({ag_in}/{len(D)} agents)   "
          f"LOMO {tot_lo}/{tot_n} ({ag_lo}/{len(D)} agents)")
    print(f"top-1 cosine: correct {ok.min():.3f}-{ok.max():.3f} med {np.median(ok):.3f} | "
          f"incorrect {bad.min():.3f}-{bad.max():.3f} med {np.median(bad):.3f} | "
          f"{above}/{len(bad)} above the lowest correct")
    print("saved data/v3_lomo_computed.csv + latex/sitables/lomocomputed.tex")


if __name__ == "__main__":
    main()
