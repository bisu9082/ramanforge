"""
Detector configuration per agent, from the deposited acquisition headers.

The instrument's automatic exposure control sets a gain stage as well as an integration
time. Both, and the detector temperature, are recorded per acquisition and were not
reported in earlier versions of this work. Two vials, A-234 and A-242, span more than one
setting; they are also the two whose same-vial cosine is lowest, so the configuration has
to be on the record before that spread is interpreted.

Produces data/v3_detector_config.csv and SI table \\label{tab:si-detconfig}.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NICE = {"GD": "GD", "GF": "GF", "VX": "VX", "HD": "HD", "HN1": "HN1", "HN2": "HN2",
        "HN3": "HN3", "A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}
ORDER = ["GD", "GF", "VX", "HD", "HN1", "HN2", "HN3", "A230", "A232", "A234", "A242"]


def main():
    M = pd.read_csv(os.path.join(ROOT, "data", "v3_raw_meta.csv"))
    rows = []
    for ag in ORDER:
        A = M[M.agent == ag]
        settings = (A.groupby(["Gain", "Offset"], sort=False)
                     .agg(n=("file", "size"), it_lo=("IntTime", "min"),
                          it_hi=("IntTime", "max")).reset_index())
        for _, r in settings.iterrows():
            it = (f"{r.it_lo:.0f}" if r.it_lo == r.it_hi
                  else f"{r.it_lo:.0f}--{r.it_hi:.0f}")
            rows.append(dict(agent=NICE[ag], files=int(r.n), integration_ms=it,
                             gain=int(r.Gain), offset=int(r.Offset),
                             ccdt_lo=A.CCDT.min(), ccdt_hi=A.CCDT.max(),
                             n_settings=len(settings)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_detector_config.csv"), index=False)

    split = sorted({r["agent"] for r in rows if r["n_settings"] > 1})
    body, seen = [], set()
    for r in rows:
        lab = r["agent"] if r["agent"] not in seen else ""
        seen.add(r["agent"])
        ccdt = (f"{r['ccdt_lo']:.2f} to {r['ccdt_hi']:.2f}" if lab else "")
        body.append(f"{lab} & {r['files']} & {r['integration_ms']} & {r['gain']} & "
                    f"{r['offset']} & {ccdt} \\\\")
    note = (f"Two vials, {' and '.join(split)}, span more than one detector setting; the "
            "other nine are at a single setting throughout. Both are among the three "
            "whose same-vial cosine falls below 0.99. Neither change can account for that "
            "spread: the gain difference is multiplicative and 3.2\\,\\%, to which both the "
            "normalization and the cosine are invariant, and the offset is a constant "
            "pedestal removed by the baseline step. The detector temperature varies over "
            "0.52\\,K across the whole campaign and is not correlated with agent. Counts "
            "are not gain-normalized anywhere in the pipeline, because no quantity reported "
            "in this work depends on absolute intensity. Produced by "
            "\\texttt{code/v3\\_detector\\_config.py}.")

    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Detector configuration recorded per acquisition. The automatic "
           "exposure control sets the gain stage as well as the integration time. One row "
           "per distinct (gain, offset) setting per vial, over all 55 delivered files; the "
           "derived files inherit their parent's header.}\n"
           "\\label{tab:si-detconfig}\n\\small\n"
           "\\begin{tabular}{lrrrrl}\n\\toprule\n"
           "Agent & Files & Integration (ms) & Gain & Offset & CCD temperature ($^{\\circ}$C) \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}\n\\end{table}\n")
    open(os.path.join(ROOT, "latex", "sitables", "detconfig.tex"), "w").write(tex)
    print(D.to_string(index=False))
    print(f"\nvials spanning more than one setting: {split}")
    print("saved data/v3_detector_config.csv + latex/sitables/detconfig.tex")


if __name__ == "__main__":
    main()
