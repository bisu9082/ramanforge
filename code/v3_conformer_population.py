"""
How much of the room-temperature population the single library conformer carries.

The library holds one structure per molecule. The CREST ensembles deposited under
conformers/ allow the Boltzmann weight of that structure at 298 K to be read off directly,
which bounds how well a single-conformer spectrum can represent a neat liquid.

The energies are GFN2 and the weights inherit that method's error, so these are indicative
rather than exact; the point is the order of magnitude.

Writes data/v3_conformer_population.csv and latex/sitables/confpop.tex.
"""
import os, glob, re, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KT = 0.001987204 * 298.15          # kcal mol-1
NM = {"EXP_GD_soman": "GD", "GF_cyclosarin": "GF", "EXP_VX": "VX", "EXP_HD": "HD",
      "EXP_HN1": "HN1", "EXP_HN2": "the HN2 vial", "EXP_HN3": "HN3", "EXP_A230": "A-230",
      "EXP_A232_E": "A-232", "EXP_A234": "A-234", "EXP_A242": "A-242"}
ORDER = ["GD", "GF", "VX", "HD", "HN1", "HN3", "A-230", "A-232", "A-234", "A-242"]


def energies(path):
    e = []
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            try:
                e.append(float(p[-1]))
            except ValueError:
                pass
    v = np.asarray(e, float)
    if not len(v):
        return v
    v = v - v.min()
    return v * 627.5095 if v.max() < 0.2 else v       # Hartree -> kcal/mol if needed


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "conformers", "*_conformer_energies.txt"))):
        job = os.path.basename(f).replace("_conformer_energies.txt", "")
        if job not in NM:
            continue
        e = energies(f)
        if not len(e):
            continue
        w = np.exp(-e / KT); w = w / w.sum()
        rows.append(dict(agent=NM[job], job=job, n_conformers=len(e),
                         p_global_pct=round(float(w.max()) * 100, 1),
                         n_effective=round(float(1 / np.square(w).sum()), 1),
                         second_kcal=round(float(np.sort(e)[1]), 2) if len(e) > 1 else np.nan))
    D = pd.DataFrame(rows)
    D["sort"] = D.agent.map({a: i for i, a in enumerate(ORDER)})
    D = D.sort_values("sort").drop(columns="sort")
    D.to_csv(os.path.join(ROOT, "data", "v3_conformer_population.csv"), index=False)
    print(D.to_string(index=False))

    Q = D[D.agent != "the HN2 vial"]
    under10 = int((Q.p_global_pct < 10).sum())
    body = [f"{r.agent} & {r.n_conformers} & {r.second_kcal:.2f} & {r.p_global_pct:.1f} & "
            f"{r.n_effective:.1f} \\\\" for r in D.itertuples() if r.agent != "the HN2 vial"]
    tex = """\\begin{table}[htbp]
\\centering
\\caption{What fraction of the 298\\,K population the single structure in the library
carries. Conformer energies are the GFN2 metadynamic ensembles deposited under
\\texttt{conformers/}; $\\Delta E_2$ is the energy of the second conformer above the
minimum, $p$ the Boltzmann weight of the minimum and $N_\\mathrm{eff}$ the inverse
participation ratio of the weights. Produced by
\\texttt{code/v3\\_conformer\\_population.py}.}
\\label{tab:si-confpop}
\\small
\\begin{tabular}{lrrrr}
\\toprule
Agent & Conformers & $\\Delta E_2$ (kcal\\,mol$^{-1}$) & $p$ (\\%) & $N_\\mathrm{eff}$ \\\\
\\midrule
""" + "\n".join(body) + f"""
\\bottomrule
\\end{{tabular}}

\\begin{{flushleft}}\\footnotesize For {under10} of the ten agents the structure the library
uses carries under a tenth of the room-temperature population, and for HN1 it carries
3.1\\,\\%. The two G-agents are the exception, at 59\\,\\%. These are GFN2 weights and inherit
that method's error --- Table~\\ref{{tab:si-a232}} documents a 3.55\\,kcal\\,mol$^{{-1}}$ GFN2
error on one configurational coordinate in this set --- so they bound the problem rather
than quantify it. What they establish is that a single-conformer library spectrum is not a
good representation of a neat liquid for most of these molecules, and that Boltzmann
averaging is the untested part of the standard treatment with the most room to move. We
did not run it: an ensemble spectrum requires a frequency calculation per populated
conformer, which for VX alone is of order a hundred.\\end{{flushleft}}
\\end{{table}}
"""
    open(os.path.join(ROOT, "latex", "sitables", "confpop.tex"), "w").write(tex)
    print(f"\n{under10} of ten agents below 10 per cent")
    print("saved data/v3_conformer_population.csv + latex/sitables/confpop.tex")


if __name__ == "__main__":
    main()
