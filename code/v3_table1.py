"""
Main-text Table 1: the head-to-head result in one place, with and without the benzene vial.

Every figure here appears in the Results prose as well; the table exists because the
manuscript otherwise carries none and a reader has no single place to find them.

Produces latex/table1.tex.
"""
import json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    H = pd.read_csv(os.path.join(ROOT, "data", "v3_hn2_effect.csv"))
    a = H[H.subset == "all"].iloc[0]
    b = H[H.subset == "noHN2"].iloc[0]
    na, nb = int(a.n), int(b.n)

    rows = [
        ("Molecular top-1", f"{int(a.meas_mol)}/{na}", f"{int(a.dft_mol)}/{na}",
         f"{int(b.meas_mol)}/{nb}", f"{int(b.dft_mol)}/{nb}"),
        ("Class top-1", f"{int(a.meas_cls)}/{na}", f"{int(a.dft_cls)}/{na}",
         f"{int(b.meas_cls)}/{nb}", f"{int(b.dft_cls)}/{nb}"),
        ("Similarity to the correct entry", f"{a.meas_self:.3f}", f"{a.dft_self:.3f}",
         f"{b.meas_self:.3f}", f"{b.dft_self:.3f}"),
        ("Decision margin", f"{a.meas_margin:.3f}", f"{a.dft_margin:.3f}",
         f"{b.meas_margin:.3f}", f"{b.dft_margin:.3f}"),
    ]
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{The same thirty acquisitions against a measured and a calibrated "
           "computed library, under one pipeline, one native measurement grid and one "
           "metric, with no parameter fitted to identification. The right-hand pair sets "
           "aside the three acquisitions of the vial labeled HN2, whose contents are "
           "benzene and therefore not established; excluding it raises the computed "
           "library's similarity to the correct entry and leaves its molecular count "
           "unchanged. Similarities and margins are means over the queries. Produced by "
           "\\texttt{code/v3\\_table1.py}.}\n"
           "\\label{tab:headline}\n\\small\n"
           "\\begin{tabular}{lcccc}\n\\toprule\n"
           f" & \\multicolumn{{2}}{{c}}{{All {na} acquisitions}} & "
           f"\\multicolumn{{2}}{{c}}{{{nb}, benzene vial set aside}} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
           " & Measured & Computed & Measured & Computed \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize The measured arm compares acquisitions of one "
           "vial recorded in one session, so it is a within-vial repeatability ceiling "
           "rather than a transferable identification rate; on another laboratory's "
           "published band lists the two libraries reach three of five and two of five "
           "(Section~S11.2 of the Supplementary Material). The molecular figure runs from "
           "12 to 16 of 30 across the leave-one-agent-out calibration envelope "
           "(Table~S-scalescan).\\end{flushleft}\n\\end{table}\n")
    open(os.path.join(ROOT, "latex", "table1.tex"), "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
