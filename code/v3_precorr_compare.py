"""
The geometry correction, under both pre-correction calibrations.

Correcting the GF and A-232 geometries and supplying the missing VX calculation changed
the computed library. How large that change is depends on which calibration the
pre-correction state is scored under, and the two defensible choices disagree in sign for
the open set, so both are reported.

  held   the calibration in force before the correction, s = (1.018, 0.988), carried over
         unchanged. This is what the manuscript quoted in earlier versions.
  refit  the same non-circular objective refitted on the pre-correction library, which
         returns s = (1.022, 0.980). This isolates the geometry change with the
         calibration treated identically on both sides, and it is what this deposit
         reproduces.

Produces data/v3_precorr_compare.csv and SI table \\label{tab:si-precorr}.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))

CONDITIONS = [("corrected (adopted)", False, 1.014, 0.980),
              ("pre-correction, held calibration", True, 1.018, 0.988),
              ("pre-correction, refitted calibration", True, 1.022, 0.980)]


def evaluate(precorr, sl, sh):
    for m in [k for k in list(sys.modules) if "headtohead" in k]:
        del sys.modules[m]
    os.environ["RF_PRECORR"] = "1" if precorr else ""
    if not precorr:
        os.environ.pop("RF_PRECORR", None)
    import v3_step3_headtohead as H
    X, acq = H.load_measured(); lib = H.load_library()
    dft = {k: H.synth_native(v, X, sl, sh) for k, v in lib.items()}
    query = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]
    closed = {H.JOB[a] for a in H.AGENTS if a in H.JOB}
    cl = op = n = 0
    for a in query:
        for _, spec in acq[a]:
            n += 1
            nrm = np.linalg.norm(spec)
            o = {k: float(np.dot(spec, d) / (nrm * np.linalg.norm(d))) for k, d in dft.items()}
            c = {k: v for k, v in o.items() if k in closed}
            if max(o, key=o.get) == H.JOB[a]:
                op += 1
            if max(c, key=c.get) == H.JOB[a]:
                cl += 1
    return len(lib), cl, op, n


def main():
    rows = []
    for name, pre, sl, sh in CONDITIONS:
        size, cl, op, n = evaluate(pre, sl, sh)
        rows.append(dict(condition=name, library=size, s_low=sl, s_high=sh,
                         closed=cl, open=op, n=n))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_precorr_compare.csv"), index=False)

    adopted = D.iloc[0]
    held, refit = D.iloc[1], D.iloc[2]
    body = [f"{r.condition} & {r.s_low:.3f}, {r.s_high:.3f} & {r.closed}/{r.n} & {r.open}/{r.n} \\\\"
            for r in D.itertuples()]
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{The effect of correcting the GF and A-232 geometries and supplying "
           "the missing VX calculation, scored under both defensible treatments of the "
           "pre-correction calibration. Held carries the calibration that was in force "
           "before the correction; refitted applies the same non-circular objective to the "
           "pre-correction library. The closed set is the eleven measured agents, the open "
           "set the full library.}\n"
           "\\label{tab:si-precorr}\n\\small\n"
           "\\begin{tabular}{lccc}\n\\toprule\n"
           "Condition & $s_\\mathrm{low}$, $s_\\mathrm{high}$ & Closed set & Open set \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize Under the held calibration the correction "
           f"lowers both figures, from {held.closed} to {adopted.closed} in the closed set "
           f"and from {held.open} to {adopted.open} in the open set. Under the refitted "
           f"calibration it lowers the closed set, from {refit.closed} to {adopted.closed}, "
           f"and raises the open set by one, from {refit.open} to {adopted.open}. The two "
           "differ because the held comparison changes the geometry and the calibration "
           "together while the refitted one changes only the geometry. We report both "
           "rather than choose: what survives either treatment is that correcting a "
           "verified structural error did not recover the identification, and that the "
           "closed-set figure fell. Removing VX from the pre-correction library changes no "
           "assignment under either calibration. Produced by "
           "\\texttt{code/v3\\_precorr\\_compare.py}.\\end{flushleft}\n\\end{table}\n")
    open(os.path.join(ROOT, "latex", "sitables", "precorr.tex"), "w").write(tex)
    print(D.to_string(index=False))
    print("saved data/v3_precorr_compare.csv + latex/sitables/precorr.tex")


if __name__ == "__main__":
    main()
