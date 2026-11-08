"""
Continuous scan of the two frequency scale factors over the leave-one-agent-out envelope.

The eleven held-out refits of Table tab:si-loao land on only four distinct values of
s_low, so they do not sample the interior of their own range. This scans the envelope on
the 0.002 grid the calibration is reported to and records the molecular top-1 at every
point, together with the agents whose assignment changes.

Produces data/v3_scalescan.csv and SI table \\label{tab:si-scalescan}.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
import v3_step3_headtohead as H

SLOW  = [1.006, 1.008, 1.010, 1.012, 1.014, 1.016, 1.018, 1.020, 1.022, 1.024]
SHIGH = [0.980, 0.982, 0.984]
ADOPTED = (1.014, 0.980)


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    query = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]

    def evaluate(sl, sh):
        dft = {k: H.synth_native(v, X, sl, sh) for k, v in lib.items()}
        mol, per = 0, {}
        for a in query:
            hits = 0
            for _, spec in acq[a]:
                nrm = np.linalg.norm(spec)
                sims = {k: float(np.dot(spec, d) / (nrm * np.linalg.norm(d)))
                        for k, d in dft.items()}
                if max(sims, key=sims.get) == H.JOB[a]:
                    mol += 1; hits += 1
            per[a] = hits
        return mol, per

    rows = []
    base_mol, base_per = evaluate(*ADOPTED)
    for sl in SLOW:
        for sh in SHIGH:
            mol, per = evaluate(sl, sh)
            moved = ",".join(f"{a}{per[a] - base_per[a]:+d}"
                             for a in query if per[a] != base_per[a])
            rows.append(dict(s_low=sl, s_high=sh, molecular=mol, n=30,
                             adopted=(sl, sh) == ADOPTED, agents_changed=moved))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_scalescan.csv"), index=False)

    lo, hi = int(D.molecular.min()), int(D.molecular.max())
    body = []
    for sl in SLOW:
        cells = []
        for sh in SHIGH:
            v = int(D[(D.s_low == sl) & (D.s_high == sh)].molecular.iloc[0])
            cells.append(f"\\textbf{{{v}}}" if (sl, sh) == ADOPTED else str(v))
        tag = "\\textbf{%.3f}" % sl if sl == ADOPTED[0] else "%.3f" % sl
        body.append(f"{tag} & " + " & ".join(cells) + " \\\\")

    header = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Molecular top-1 of 30 over the leave-one-agent-out calibration envelope,\n"
        "scanned continuously on the 0.002 grid. The eleven held-out refits of\n"
        "Table~\\ref{tab:si-loao} span $s_\\mathrm{low}$ = 1.006--1.024 and\n"
        "$s_\\mathrm{high}$ = 0.980--0.984 but land on only four distinct values of\n"
        "$s_\\mathrm{low}$, leaving the interior of their own range unsampled. The adopted\n"
        "calibration is in bold. Produced by \\texttt{code/v3\\_scalescan.py}.}\n"
        "\\label{tab:si-scalescan}\n\\small\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{$s_\\mathrm{high}$} \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        "$s_\\mathrm{low}$ & 0.980 & 0.982 & 0.984 \\\\\n\\midrule\n")
    footer = (
        "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\begin{flushleft}\\footnotesize The molecular figure runs from "
        f"{lo} to {hi} of 30 across\n"
        "the envelope and the adopted calibration returns "
        f"{base_mol}. $s_\\mathrm{{high}}$ moves it at\n"
        "one point only; the variation is carried entirely by $s_\\mathrm{low}$. The maximum at\n"
        "$s_\\mathrm{low}$ = 1.016 and 1.018 is reached because all three HN1 acquisitions move\n"
        "from HN3 to HN1; those values lie 0.2 and 0.4\\,\\% from the adopted 1.014 and inside\n"
        "the held-out range, so the adopted figure is not the most favorable one available\n"
        "within the calibration's own uncertainty. We report it because it is the value the\n"
        "non-circular objective selects, and we state the envelope rather than a robustness\n"
        "claim the eleven-point check cannot support.\\end{flushleft}\n"
        "\\end{table}\n")
    tex = header + "\n".join(body) + footer
    open(os.path.join(ROOT, "latex", "sitables", "scalescan.tex"), "w").write(tex)
    print(D.to_string(index=False))
    print(f"\nmolecular top-1 over the envelope: {lo}-{hi} of 30, adopted {base_mol}")
    print("saved data/v3_scalescan.csv + latex/sitables/scalescan.tex")


if __name__ == "__main__":
    main()
