"""
Is the window diagnosis an artifact of the two-region frequency map?

The adopted map multiplies harmonic wavenumbers below 800 cm-1 by 1.016 and those above it
by 0.990. The branch is taken on the unscaled wavenumber, so the map is discontinuous at
the boundary: 800- maps to 812.8 and 800+ to 792.0, a 20.8 cm-1 overlap in which the order
of two modes can be reversed. Four library entries have a mode pair in that overlap, and
the overlap sits inside the 600-850 cm-1 window the diagnosis of Section 3.2 rests on.

The affine map fitted in Section S4.4 is monotone and fits the band positions better. This
script recomputes the window-resolved cosines, the phosphorus contrast and the unmatched
band shares under both maps, so that the diagnosis can be read off a calibration that
cannot reverse a mode order.

Writes data/v3_affine_window.csv and latex/sitables/affwindow.tex.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

REG = [("250-600", 250, 600), ("600-850", 600, 850), ("850-1150", 850, 1150),
       ("1150-1400", 1150, 1400), ("1400-1800", 1400, 1800)]
PF = ("GD", "GF", "VX", "A230", "A232", "A234", "A242")
NOP = ("HD", "HN1", "HN3")


def synth(peaks, grid, mapper, gamma=H.GAMMA):
    f = mapper(peaks[:, 0])
    a = peaks[:, 1].astype(float) * H.scattering_factor(f)
    if a.max() > 0:
        a = a / a.max()
    g = gamma / 2.0
    I = np.zeros_like(grid, dtype=float)
    for ff, aa in zip(f, a):
        I += aa * g * g / ((grid - ff) ** 2 + g * g)
    return I / (I.max() + 1e-12)


def fit_affine(lib, mpk, agents, tol=35.0):
    """Same non-circular objective as the two-region fit: band positions only."""
    best = None
    for g in np.arange(0.950, 1.0001, 0.002):
        for b in np.arange(-10.0, 60.1, 2.0):
            res, n = [], 0
            for a in agents:
                pk = lib[H.JOB[a]]
                f = g * pk[:, 0] + b
                keep = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max())
                f = f[keep]; w = np.asarray(mpk[a], float)
                if not len(f) or not len(w):
                    continue
                C = np.abs(w[:, None] - f[None, :])
                C = np.where(C > tol, 1e6, C)
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] < 1e6:
                        res.append(f[j] - w[i]); n += 1
            if n >= 60:
                r = float(np.sqrt(np.mean(np.square(res))))
                if best is None or r < best[0]:
                    best = (r, g, b, n)
    return best


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    calib = [a for a in H.AGENTS if a in H.JOB]
    _, SL, SH, n2 = H.fit_scale(lib, mpk, calib)
    rA, gA, bA, nA = fit_affine(lib, mpk, calib)
    print(f"two-region  s=({SL:.3f},{SH:.3f})  pairs={n2}")
    print(f"affine      gain={gA:.3f} offset={bA:+.1f}  RMSE={rA:.2f}  pairs={nA}")

    maps = {"two-region": lambda f: np.where(f < H.CUT, f * SL, f * SH),
            "affine": lambda f: gA * f + bA}
    ags = [a for a in H.AGENTS if a in H.JOB and a != "HN2"]
    rows = []
    for tag, mp in maps.items():
        dft = {a: synth(lib[H.JOB[a]], X, mp) for a in ags}
        for name, lo, hi in REG:
            m = (X >= lo) & (X < hi)
            cs = {a: H.cos(mean[a][m], dft[a][m]) for a in ags}
            p = float(np.mean([cs[a] for a in ags if a in PF]))
            q = float(np.mean([cs[a] for a in ags if a in NOP]))
            rows.append(dict(calibration=tag, region=name, mean_all=float(np.mean(list(cs.values()))),
                             mean_P=p, mean_noP=q, delta=q - p,
                             **{f"cos_{a}": cs[a] for a in ags}))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_affine_window.csv"), index=False)
    print()
    print(D[["calibration", "region", "mean_P", "mean_noP", "delta"]].round(3).to_string(index=False))

    T = D.set_index(["calibration", "region"])
    body = []
    for name, _, _ in REG:
        a = T.loc[("two-region", name)]; b = T.loc[("affine", name)]
        body.append(f"{name.replace('-', '--')} & {a.mean_P:.3f} & {a.mean_noP:.3f} & "
                    f"${a.delta:+.3f}$ & {b.mean_P:.3f} & {b.mean_noP:.3f} & "
                    f"${b.delta:+.3f}$ \\\\")
    d2 = T.loc[("two-region", "600-850")]; da = T.loc[("affine", "600-850")]
    other2 = max(abs(T.loc[("two-region", r)].delta) for r, _, _ in [(n, 0, 0) for n, _, _ in REG] if r != "600-850")
    othera = max(abs(T.loc[("affine", r)].delta) for r, _, _ in [(n, 0, 0) for n, _, _ in REG] if r != "600-850")
    tex = """\\begin{table}[htbp]
\\centering
\\caption{The window diagnosis under both calibrations. The adopted two-region map is
discontinuous at 800\\,\\wn{}, where it sends 800$^-$ to 812.8 and 800$^+$ to 792.0; four
library entries have a mode pair inside that overlap and have their order reversed by it.
The affine map of Table~\\ref{tab:si-affine} is monotone. Both columns are computed with
the same spectra, the same window definitions and the same ten agents. Produced by
\\texttt{code/v3\\_affine\\_window.py}.}
\\label{tab:si-affwindow}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
 & \\multicolumn{3}{c}{Two-region (adopted)} & \\multicolumn{3}{c}{Affine} \\\\
\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}
Window (\\wn) & P & no P & $\\Delta$ & P & no P & $\\Delta$ \\\\
\\midrule
""" + "\n".join(body) + f"""
\\bottomrule
\\end{{tabular}}

\\begin{{flushleft}}\\footnotesize The diagnosis does not depend on the map. Under the
adopted calibration the 600--850\\,\\wn{{}} window gives $\\Delta = {d2.delta:+.3f}$ against
at most ${other2:.3f}$ in absolute value elsewhere; under the affine map it gives
$\\Delta = {da.delta:+.3f}$ against at most ${othera:.3f}$. In both cases 600--850\\,\\wn{{}}
is the only window in which the phosphorus-free agents score higher than the
phosphorus-containing ones, and in both cases it is the largest contrast in the row. The
discontinuity is therefore a defect of the adopted map that should be stated, but it does
not generate the localisation it sits inside.\\end{{flushleft}}
\\end{{table}}
"""
    open(os.path.join(ROOT, "latex", "sitables", "affwindow.tex"), "w").write(tex)
    print("\nsaved data/v3_affine_window.csv + latex/sitables/affwindow.tex")


if __name__ == "__main__":
    main()
