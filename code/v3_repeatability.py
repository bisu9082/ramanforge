"""
Within-vial repeatability, per vial rather than pooled.

The repeatability of a measured library is quoted elsewhere in this work as a single mean
over all same-vial pairs. That mean conceals the quantity a reader needs: the worst case.
Two vials are not stationary across their own repeat acquisitions, and this script reports
the spread per vial together with the count rate of each acquisition, which is what moves.

Writes data/v3_repeatability.csv and latex/sitables/repeat.tex.
"""
import os, sys, itertools
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

NM = {"A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}


def main():
    X, acq = H.load_measured()
    A = pd.read_csv(os.path.join(ROOT, "data", "v3_acquisition_audit.csv"))
    rate = {}
    for r in A.itertuples():
        rate.setdefault(r.agent, {})[int(r.rep)] = float(r.counts_per_s)

    rows = []
    for a in H.AGENTS:
        pairs = [(r, s / s.max()) for r, s in acq[a]]
        n = len(pairs)
        cs = [H.cos(x, y) for (_, x), (_, y) in itertools.combinations(pairs, 2)]
        rr = [rate.get(a, {}).get(r, np.nan) for r, _ in pairs]
        rr = [x for x in rr if not np.isnan(x)]
        rows.append(dict(agent=a, n=n,
                         cos_min=(min(cs) if cs else np.nan),
                         cos_med=(float(np.median(cs)) if cs else np.nan),
                         cos_max=(max(cs) if cs else np.nan),
                         rate_min=(min(rr) if rr else np.nan),
                         rate_max=(max(rr) if rr else np.nan),
                         rate_ratio=((max(rr) / min(rr)) if rr and min(rr) > 0 else np.nan)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_repeatability.csv"), index=False)
    print(D.round(4).to_string(index=False))

    multi = D[D.n >= 2]
    worst = multi.loc[multi.cos_min.idxmin()]
    body = []
    for r in D.itertuples():
        if r.n < 2:
            body.append(f"{NM.get(r.agent, r.agent)} & {r.n} & \\multicolumn{{3}}{{c}}{{---}} & "
                        f"{r.rate_min:.0f} & --- \\\\")
        else:
            body.append(f"{NM.get(r.agent, r.agent)} & {r.n} & {r.cos_min:.3f} & "
                        f"{r.cos_med:.3f} & {r.cos_max:.3f} & "
                        f"{r.rate_min:.0f}--{r.rate_max:.0f} & {r.rate_ratio:.1f}$\\times$ \\\\")
    tex = """\\begin{table}[htbp]
\\centering
\\caption{Within-vial repeatability, per vial. Each entry is the cosine between two
acquisitions of the same sealed vial, taken minutes apart in one session; the count rate is
the detector rate the instrument's automatic exposure control settled on for each
acquisition. Produced by \\texttt{code/v3\\_repeatability.py}.}
\\label{tab:si-repeat}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
Vial & Acq. & Min & Median & Max & Count rate (s$^{-1}$) & Spread \\\\
\\midrule
""" + "\n".join(body) + f"""
\\bottomrule
\\end{{tabular}}

\\begin{{flushleft}}\\footnotesize The pooled mean over all same-vial pairs is 0.967, which
is not the useful number. Two vials are not stationary across their own repeats:
{NM.get(worst.agent, worst.agent)} falls to {worst.cos_min:.3f} and A-242 to
{float(D[D.agent=='A242'].cos_min.iloc[0]):.3f}, against 0.998 or better for the six vials
that are stationary. In both cases the count rate rises by more than an order of magnitude
between the first acquisition and the rest and then settles, so the spectra of the early
acquisitions are of a different state of the sample, not a noisier measurement of the same
one. A within-vial difference of {1 - worst.cos_min:.3f} is larger than the decision margin
of the measured library, 0.219, so these acquisitions are not interchangeable and are
treated separately in Section~\\ref{{si:ident}}.\\end{{flushleft}}
\\end{{table}}
"""
    open(os.path.join(ROOT, "latex", "sitables", "repeat.tex"), "w").write(tex)
    print("\nsaved data/v3_repeatability.csv + latex/sitables/repeat.tex")


if __name__ == "__main__":
    main()
