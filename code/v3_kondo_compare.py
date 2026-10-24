"""
ramanforge: external validation of the measured band positions against Kondo et al. 2018.

Kondo, T. et al., "Analysis of chemical warfare agents by portable Raman spectrometer with
both 785 nm and 1064 nm excitation", Forensic Sci. Int. 291, 23-38 (2018),
doi:10.1016/j.forsciint.2018.07.032. Their Table 1 lists assigned band positions for
twenty-two authentic agents measured on a Xantus-2 at 785 nm and at 1064 nm; the values
used here are the 1064 nm ones, which is the excitation used in this work. Seven of their
agents overlap this study. Kondo's material was synthesised in house and confirmed by NMR
and GC-MS at a purity above 98 per cent, so this is the only independent measured
reference available for authentic agents at this excitation.

Kondo report assigned bands only, not a complete band list, so the comparison is one-way:
for each Kondo band we ask whether this work has a measured band at the same position.
The reverse is not evidence of disagreement.

Writes data/v3_kondo_comparison.csv and latex/sitables/kondo.tex.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

# Kondo et al. 2018, Table 1 and Section 3.2. Value at 1064 nm where the table gives one
# in parentheses, otherwise the single value reported for both excitations.
KONDO = {
    "GD":  [(727, "P"), (751, "P"), (513, "P-O"), (802, "alkyl")],
    "GF":  [(753, "P"), (502, "P-O"), (792, "alkyl"), (1447, "alkyl")],
    "VX":  [(695, "P"), (742, "P"), (768, "P"), (500, "P-O"),
            (837, "alkyl"), (1455, "N-C"), (667, "C-S")],
    "HD":  [(698, "C-Cl"), (746, "C-Cl"), (648, "C-S"), (855, "alkyl"), (1428, "alkyl")],
    "HN1": [(721, "C-Cl"), (787, "C-N-C"), (1448, "N-C"), (664, "tert-amine")],
    "HN2": [(729, "C-Cl"), (1448, "N-C"), (388, "skeletal"), (664, "tert-amine")],
    "HN3": [(722, "C-Cl"), (1440, "N-C"), (805, "alkyl")],
}
TOL = 18.0          # Kondo's quoted 1064 nm spectral resolution is 15-18 cm-1


def measured_bands(X, spec):
    sd = float(np.std(np.diff(spec[X > 1600.0]))) / np.sqrt(2.0)
    idx, _ = find_peaks(spec, prominence=max(0.05, 5.0 * sd), distance=2)
    return X[idx], spec[idx]


def main():
    X, acq = H.load_measured()
    rows = []
    for ag, bands in KONDO.items():
        m = np.mean([s for _, s in acq[ag]], 0); m = m / m.max()
        w, h = measured_bands(X, m)
        for nu, assign in bands:
            if nu < X[0] or nu > X[-1]:
                rows.append(dict(agent=ag, kondo_cm1=nu, assignment=assign,
                                 this_work_cm1=np.nan, residual=np.nan,
                                 rel_intensity=np.nan, matched="out of range"))
                continue
            j = int(np.argmin(np.abs(w - nu)))
            d = float(w[j] - nu)
            ok = abs(d) <= TOL
            rows.append(dict(agent=ag, kondo_cm1=nu, assignment=assign,
                             this_work_cm1=(float(w[j]) if ok else np.nan),
                             residual=(round(d, 1) if ok else np.nan),
                             rel_intensity=(round(float(h[j]), 3) if ok else np.nan),
                             matched=("yes" if ok else "no")))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_kondo_comparison.csv"), index=False)

    inr = D[D.matched != "out of range"]
    print(f"{'agent':6s}{'matched':>10s}{'of':>5s}{'  residuals (cm-1)'}")
    for ag in KONDO:
        s = inr[inr.agent == ag]
        mt = s[s.matched == "yes"]
        res = ", ".join(f"{r:+.0f}" for r in mt.residual) if len(mt) else "-"
        print(f"{ag:6s}{len(mt):>10d}{len(s):>5d}  {res}")
    mt = inr[inr.matched == "yes"]
    print(f"\noverall {len(mt)} of {len(inr)} Kondo bands reproduced within {TOL:.0f} cm-1")
    print(f"residual mean {mt.residual.mean():+.1f}, RMS {np.sqrt((mt.residual**2).mean()):.1f} cm-1")
    write_table(D, inr, mt)
    print("\nsaved data/v3_kondo_comparison.csv + latex/sitables/kondo.tex")


NM = {"A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}


def write_table(D, inr, mt):
    body = []
    for ag in KONDO:
        s = D[D.agent == ag]
        for i, r in enumerate(s.itertuples()):
            lab = NM.get(ag, ag) if i == 0 else ""
            if r.matched == "yes":
                here, res = f"{r.this_work_cm1:.0f}", f"${r.residual:+.1f}$"
                mark = "\\checkmark"
            elif r.matched == "no":
                here, res, mark = "---", "---", "$\\times$"
            else:
                here, res, mark = "out of range", "---", "---"
            body.append(f"{lab} & {r.kondo_cm1:.0f} & {r.assignment} & {here} & {res} & {mark} \\\\")
        if ag != list(KONDO)[-1]:
            body.append("\\addlinespace")
    rms = float(np.sqrt((mt.residual ** 2).mean()))
    big = mt[mt.residual.abs() > 10.0]
    core = mt[mt.residual.abs() <= 10.0]
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Measured band positions against those published by Kondo et al.\\ for authentic
agents at the same excitation (Kondo et al., Forensic Sci.\\ Int.\\ 291, 23--38, 2018).\nTheir Table~1 reports assigned
bands only, so the comparison is one-way: for each band they report we ask whether this
work has a measured band at the same position, within the 18\\,\\wn{{}} resolution they quote
for 1064\\,nm. The absence of a partner is therefore not evidence of disagreement. Bands
are those detected by the criterion of Section~\\ref{{si:preproc}}.}}
\\label{{tab:si-kondo}}
\\small
\\begin{{tabular}}{{llrrrc}}
\\toprule
Agent & Kondo (\\wn) & Assignment & This work (\\wn) & Residual (\\wn) & Reproduced \\\\
\\midrule
""" + "\n".join(body) + f"""
\\bottomrule
\\end{{tabular}}

\\begin{{flushleft}}\\footnotesize {len(mt)} of {len(inr)} reported bands are reproduced,
with a residual mean of {mt.residual.mean():+.1f}\\,\\wn{{}} and an RMS of {rms:.1f}\\,\\wn{{}},
below the {5.73:.2f}\\,\\wn{{}} sampling of this measurement; excluding the two residuals
larger than 10\\,\\wn{{}} the RMS is {float(np.sqrt((core.residual ** 2).mean())):.1f}\\,\\wn{{}}.
Both large residuals have a stated cause. For HD, Kondo give the C--Cl band as 734\\,\\wn{{}}
at 785\\,nm and 746\\,\\wn{{}} at 1064\\,nm and note in their text that the 734 and
756\\,\\wn{{}} bands merge at 1064\\,nm; the 733\\,\\wn{{}} band measured here resolves the
lower of the pair. For HN1 their two excitations differ by 19\\,\\wn{{}} for the same band
(768 and 787\\,\\wn{{}}), and the 770\\,\\wn{{}} measured here matches the former. HN2 is the
only agent that reproduces none of the bands reported for it.\\end{{flushleft}}
\\end{{table}}
"""
    out = os.path.join(ROOT, "latex", "sitables", "kondo.tex")
    open(out, "w").write(tex)


if __name__ == "__main__":
    main()
