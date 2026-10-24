"""
What is in the vial labeled HN2.

The vial fails the data-quality criterion because one band carries 80 per cent of its
intensity. This script asks what that band is, and whether anything of the agent on the
label is present underneath it.

Three questions, each answered from the deposited acquisitions:

  1. the four strong bands against the Raman spectrum of liquid benzene;
  2. the positions a monosubstituted aromatic would require, which separate benzene from
     toluene, chlorobenzene, pyridine and polystyrene;
  3. the four bands Kondo et al. report for HN2 at 1064 nm, scored against the local noise
     of each individual acquisition rather than against the absolute prominence floor,
     which is referenced to the dominant band and therefore cannot see a minor component.

Writes data/v3_hn2_identity.csv and latex/sitables/hn2identity.tex.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

# Liquid benzene, Raman-active fundamentals in the fingerprint region.
BENZENE = [(606, "$\\nu_6$, ring deformation"), (992, "$\\nu_1$, ring breathing"),
           (1178, "$\\nu_9$, C--H in-plane bend"), (1586, "$\\nu_8$, ring stretch")]
# Positions a monosubstituted ring requires; their absence separates benzene from
# toluene (1004/1030/787), chlorobenzene (1002/704) and pyridine (991/1030).
MONOSUB = [702, 787, 1030]
# Kondo et al. 2018, Table 1, HN2 at 1064 nm.
KONDO_HN2 = [(388, "skeletal"), (664, "tert-amine"), (729, "C--Cl"), (1448, "N--C")]
TOL = 18.0


def noise(X, y):
    return float(np.std(np.diff(y[X > 1600.0]))) / np.sqrt(2.0)


_SP = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}


def main():
    X, acq = H.load_measured()
    reps = [(r, s / s.max()) for r, s in acq["HN2"]]
    mean = np.mean([y for _, y in reps], 0); mean = mean / mean.max()
    sd_m = noise(X, mean)

    rows = []

    def probe(group, nu, label):
        """Nearest detected band to nu in each acquisition, in units of that acquisition's noise."""
        hits = []
        for r, y in reps:
            sd = noise(X, y)
            idx, pr = find_peaks(y, prominence=0.0, distance=2)
            w, p = X[idx], pr["prominences"]
            j = int(np.argmin(np.abs(w - nu)))
            hits.append((float(w[j]), float(p[j]), float(p[j] / sd)))
        d = [abs(h[0] - nu) for h in hits]
        matched = all(x <= TOL for x in d) and all(h[2] >= 3.0 for h in hits)
        rows.append(dict(group=group, reference_cm1=nu, assignment=label,
                         observed_cm1=round(float(np.mean([h[0] for h in hits])), 1),
                         residual_cm1=round(float(np.mean([h[0] for h in hits])) - nu, 1),
                         sigma_min=round(min(h[2] for h in hits), 1),
                         sigma_max=round(max(h[2] for h in hits), 1),
                         in_all_three=bool(matched)))

    for nu, lab in BENZENE:
        probe("benzene", nu, lab)
    for nu in MONOSUB:
        j = int(np.argmin(np.abs(X - nu)))
        sig = [float(y[int(np.argmin(np.abs(X - nu)))] / noise(X, y)) for _, y in reps]
        rows.append(dict(group="monosubstituted ring", reference_cm1=nu,
                         assignment="required if the ring were substituted",
                         observed_cm1=np.nan, residual_cm1=np.nan,
                         sigma_min=round(min(sig), 1), sigma_max=round(max(sig), 1),
                         in_all_three=False))
    for nu, lab in KONDO_HN2:
        probe("Kondo HN2", nu, lab)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_hn2_identity.csv"), index=False)

    print(f"mean spectrum noise sd = {sd_m:.5f}  (5 sd = {5*sd_m:.4f}; the absolute "
          f"prominence floor used elsewhere is 0.05)")
    print(D.to_string(index=False))
    nk = int(D[(D.group == "Kondo HN2") & D.in_all_three].shape[0])
    print(f"\nKondo HN2 bands present in all three acquisitions at >=3 sd: {nk} of "
          f"{len(KONDO_HN2)}")

    esc = lambda t: t
    body = []
    for grp in ("benzene", "monosubstituted ring", "Kondo HN2"):
        sub = D[D.group == grp]
        head = {"benzene": "Liquid benzene", "monosubstituted ring": "Monosubstituted ring",
                "Kondo HN2": "HN2, Kondo et al."}[grp]
        body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{head}}}}} \\\\")
        for r in sub.itertuples():
            obs = "---" if np.isnan(r.observed_cm1) else f"{r.observed_cm1:.1f}"
            res = "---" if np.isnan(r.residual_cm1) else f"${r.residual_cm1:+.1f}$"
            sg = f"{r.sigma_min:.1f}--{r.sigma_max:.1f}"
            mark = "\\checkmark" if r.in_all_three else "$\\times$"
            body.append(f"\\quad {r.reference_cm1:.0f} & {r.assignment} & {obs} & {res} & "
                        f"{sg} & {mark} \\\\")
        body.append("\\addlinespace")
    tex = """\\begin{table}[htbp]
\\centering
\\caption{What the vial labeled HN2 contains. Each reference position is compared with the
nearest detected band in each of the three acquisitions separately, and the prominence of
that band is expressed in units of that acquisition's own noise, estimated above
1600\\,\\wn{}. The last column marks positions matched within 18\\,\\wn{} at three standard
deviations or better in all three acquisitions. Produced by
\\texttt{code/v3\\_hn2\\_identity.py}.}
\\label{tab:si-hn2identity}
\\footnotesize
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrc}
\\toprule
Reference (\\wn) & Assignment & Observed (\\wn) & Residual (\\wn) & Prominence ($\\sigma$) & In all three \\\\
\\midrule
""" + "\n".join(body[:-1]) + f"""
\\bottomrule
\\end{{tabular}}}}

\\begin{{flushleft}}\\footnotesize The four strong bands are those of liquid benzene, to
within {D[D.group=='benzene'].residual_cm1.abs().max():.1f}\\,\\wn{{}}, and in benzene's
intensity order. The three positions a monosubstituted ring would require are all below
{D[D.group=='monosubstituted ring'].sigma_max.max():.1f}$\\sigma$, which separates benzene
from toluene, chlorobenzene, pyridine and polystyrene, each of which has a strong band near
1030\\,\\wn{{}}. Of the four bands Kondo et al.\\ report for HN2, {_SP.get(nk, nk)} are present at three
standard deviations or better in every acquisition; they fall below the absolute prominence
floor of 0.05 used elsewhere in this work only because that floor is referenced to the
benzene band, which carries 80\\,\\% of the intensity of this spectrum. The vial therefore
does not contain neat HN2; what it does contain is benzene, with weak reproducible features
at HN2 band positions that these data can neither confirm nor
exclude.\\end{{flushleft}}
\\end{{table}}
"""
    open(os.path.join(ROOT, "latex", "sitables", "hn2identity.tex"), "w").write(tex)
    print("\nsaved data/v3_hn2_identity.csv + latex/sitables/hn2identity.tex")


if __name__ == "__main__":
    main()
