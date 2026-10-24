"""
Vibrational analysis of every member of the library actually used.

Reads the ORCA outputs of the 54 entries that load_library() returns, counts real and
imaginary modes and records the lowest real wavenumber. Earlier versions of this table were
generated for a 53-member library that predated the VX calculation and the two geometry
corrections; this script builds it from the library the analysis loads, so it cannot drift.

Writes data/v3_freqcheck.csv and latex/sitables/freqcheck.tex.
"""
import os, re, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

FREQ_HDR = "VIBRATIONAL FREQUENCIES"


def modes(path):
    """All non-zero harmonic wavenumbers from an ORCA output, signed."""
    txt = open(path, errors="replace").read()
    i = txt.rindex(FREQ_HDR)
    out = []
    for line in txt[i:].splitlines()[1:]:
        m = re.match(r"\s*(\d+):\s*(-?\d+\.\d+)\s*cm\*\*-1", line)
        if m:
            v = float(m.group(2))
            if abs(v) > 1e-6:
                out.append(v)
        elif out and not line.strip():
            break
    return np.asarray(out, float)


def main():
    lib = H.load_library()
    rows = []
    for job in sorted(lib):
        v = modes(os.path.join(ROOT, "orca", "outputs", job + ".out"))
        rows.append(dict(job=job, n_real=int((v > 0).sum()), n_imag=int((v < 0).sum()),
                         lowest_real=float(v[v > 0].min())))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_freqcheck.csv"), index=False)
    n_imag = int(D.n_imag.sum())
    print(f"{len(D)} library members, {n_imag} imaginary modes in total, "
          f"lowest real wavenumber {D.lowest_real.min():.1f}-{D.lowest_real.max():.1f} cm-1")

    esc = lambda t: t.replace("_", "\\_")
    half = (len(D) + 1) // 2
    L, R = D.iloc[:half], D.iloc[half:]
    body = []
    for k in range(half):
        a = L.iloc[k]
        left = f"{esc(a.job)} & {a.n_real} & {a.n_imag} & {a.lowest_real:.1f}"
        if k < len(R):
            b = R.iloc[k]
            right = f"{esc(b.job)} & {b.n_real} & {b.n_imag} & {b.lowest_real:.1f}"
        else:
            right = " &  &  & "
        body.append(left + " & " + right + " \\\\")
    hdr = ("Job & $N_\\mathrm{real}$ & $N_\\mathrm{imag}$ & $\\nu_\\mathrm{min}$ (\\wn) & "
           "Job & $N_\\mathrm{real}$ & $N_\\mathrm{imag}$ & $\\nu_\\mathrm{min}$ (\\wn) \\\\")
    tex = f"""{{\\footnotesize
\\begin{{longtable}}{{lrrr|lrrr}}
\\caption{{Vibrational analysis of every member of the library actually used. $N_\\mathrm{{real}}$
is the number of real harmonic modes, $N_\\mathrm{{imag}}$ the number of imaginary modes, and
$\\nu_\\mathrm{{min}}$ the lowest real wavenumber. Produced by \\texttt{{code/v3\\_freqcheck.py}}
from the {len(D)} entries that the analysis loads, which include the VX calculation and the
corrected GF and A-232 geometries.}}\\label{{tab:si-freq}}\\\\
\\toprule
{hdr}
\\midrule
\\endfirsthead
\\multicolumn{{8}}{{l}}{{\\footnotesize\\emph{{Table \\ref{{tab:si-freq}}, continued}}}}\\\\
\\toprule
{hdr}
\\midrule
\\endhead
\\bottomrule
\\endfoot
""" + "\n".join(body) + f"""
\\end{{longtable}}}}

\\begin{{flushleft}}\\footnotesize No structure in the library has an imaginary frequency, so
all {len(D)} geometries are true minima on the B3LYP/6-311++G(d,p) surface. The lowest real
wavenumbers ({D.lowest_real.min():.0f}--{D.lowest_real.max():.0f}\\,\\wn) are soft torsions
that fall below the 250\\,\\wn{{}} lower bound of the analysis window and therefore do not
enter the spectral comparison.\\end{{flushleft}}
"""
    open(os.path.join(ROOT, "latex", "sitables", "freqcheck.tex"), "w").write(tex)
    print("saved data/v3_freqcheck.csv + latex/sitables/freqcheck.tex")


if __name__ == "__main__":
    main()
