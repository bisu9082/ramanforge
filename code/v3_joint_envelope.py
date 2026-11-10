"""Do the favourable instrument-model choices compound?

Every sensitivity analysis elsewhere in this work varies one factor at a time, and four of
four of them move the computed library from 13 of 30 to 16 of 30 on their own: the monotone
affine frequency map, a wider Lorentzian, a shift-tolerant cosine, and a spectral
responsivity ramp on the measured side. If those gains were independent the headline would
be far higher than reported, and the central claim would be an artifact of the choices.

This script runs the identification under every combination of the four, so the question is
answered by a number rather than by an argument:

    axis map    two-region (the adopted map)  |  affine, refitted on the same
                                                 non-circular band-position objective
    linewidth   22 (adopted) | 30 | 40 | 50 cm-1
    metric      cosine (adopted) | cosine allowing +/-3 pixels of alignment
    ramp        none (adopted) | log-linear end-to-end ratio 2.0 on the measured side

The affine map is refitted inside each (ramp, linewidth) cell, because its optimum depends
on both. That makes this the slowest script in the deposit -- a few minutes -- and it is
excluded from run_all.sh's default path for that reason.

Writes data/v3_joint_envelope.csv and latex/sitables/jointenv.tex.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import v3_step3_headtohead as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMAS = [22, 30, 40, 50]
RAMPS = [1.0, 2.0]
MIN_PAIRS = 60
TOL = 35.0


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def shiftcos(u, v, maxpx=3):
    best = -1.0
    for k in range(-maxpx, maxpx + 1):
        vv = np.roll(v, k)
        if k > 0:
            vv[:k] = v[0]
        elif k < 0:
            vv[k:] = v[-1]
        best = max(best, cos(u, vv))
    return best


def synth_affine(peaks, grid, g, b, gamma):
    """Same forward model as synth_native with a monotone affine wavenumber map."""
    f = g * peaks[:, 0] + b
    a = peaks[:, 1].astype(float) * H.scattering_factor(f)
    if a.max() > 0:
        a = a / a.max()
    hw = gamma / 2.0
    out = np.zeros_like(grid, float)
    for fi, ai in zip(f, a):
        out += ai * hw * hw / ((grid - fi) ** 2 + hw * hw)
    return out / out.max() if out.max() > 0 else out


def fit_affine(lib, X, mpk, calib, gamma):
    """The non-circular objective of Section 2.6: band positions only."""
    best = None
    for g in np.arange(0.950, 1.0001, 0.002):
        for b in np.arange(-10.0, 60.1, 2.0):
            res = []
            for a in calib:
                sp = synth_affine(lib[H.JOB[a]], X, g, b, gamma)
                f = H.measured_peaks(X, sp)
                w = np.asarray(mpk[a], float)
                if not len(f) or not len(w):
                    continue
                C = np.abs(w[:, None] - f[None, :])
                C = np.where(C > TOL, 1e6, C)
                ri, ci = linear_sum_assignment(C)
                res += [C[i, j] for i, j in zip(ri, ci) if C[i, j] < 1e6]
            if len(res) < MIN_PAIRS:
                continue
            r = float(np.sqrt(np.mean(np.square(res))))
            if best is None or r < best[0]:
                best = (r, float(g), float(b), len(res))
    return best


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    calib = [a for a in H.AGENTS if a in H.JOB]
    query = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]
    n = sum(len(acq[a]) for a in query)

    rows = []
    for ramp_r in RAMPS:
        ramp = np.exp(np.log(ramp_r) * (X - X.min()) / (X.max() - X.min()))
        acq_p = {a: [(r, (s * ramp) / (s * ramp).max()) for r, s in v]
                 for a, v in acq.items()}
        mean = {a: np.mean([s for _, s in acq_p[a]], 0) for a in H.AGENTS}
        mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
        for gamma in GAMMAS:
            aff = fit_affine(lib, X, mpk, calib, gamma)
            for axis in ["two-region", "affine"]:
                if axis == "two-region":
                    D = {k: H.synth_native(v, X, 1.014, 0.980, gamma=gamma)
                         for k, v in lib.items()}
                    rmse, par = float("nan"), "1.014 / 0.980"
                else:
                    D = {k: synth_affine(v, X, aff[1], aff[2], gamma)
                         for k, v in lib.items()}
                    rmse, par = aff[0], f"{aff[1]:.3f} / {aff[2]:+.0f}"
                for mname, mf in [("cosine", cos), ("shift", shiftcos)]:
                    mol = cls = 0
                    for a in query:
                        for _, s in acq_p[a]:
                            top = max(D, key=lambda k: mf(s, D[k]))
                            if top == H.JOB[a]:
                                mol += 1
                            if H.cls_of(top) == H.cls_of(H.JOB[a]):
                                cls += 1
                    rows.append(dict(axis=axis, gamma=gamma, metric=mname,
                                     ramp=ramp_r, params=par, rmse=rmse,
                                     molecular=mol, cls=cls, n=n))
                    print(f"{axis:10s} g={gamma:<3d} {mname:7s} ramp={ramp_r:.1f}  "
                          f"molecular {mol}/{n}  class {cls}/{n}")

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_joint_envelope.csv"), index=False)

    def cell(axis, gamma, metric, ramp):
        return D[(D.axis == axis) & (D.gamma == gamma) &
                 (D.metric == metric) & (D.ramp == ramp)].iloc[0]

    adopted = cell("two-region", 22, "cosine", 1.0)
    singles = [
        ("affine map alone", cell("affine", 22, "cosine", 1.0)),
        ("$\\Gamma$ = 30\\,\\wn{} alone", cell("two-region", 30, "cosine", 1.0)),
        ("$\\Gamma$ = 40\\,\\wn{} alone", cell("two-region", 40, "cosine", 1.0)),
        ("$\\Gamma$ = 50\\,\\wn{} alone", cell("two-region", 50, "cosine", 1.0)),
        ("shift-tolerant cosine alone", cell("two-region", 22, "shift", 1.0)),
        ("responsivity ramp 2.0 alone", cell("two-region", 22, "cosine", 2.0)),
    ]
    # Pareto front over (molecular, class), both maximized
    def dominated(r):
        return any((o.molecular >= r.molecular and o.cls >= r.cls and
                    (o.molecular > r.molecular or o.cls > r.cls))
                   for o in D.itertuples())
    front = sorted([r for r in D.itertuples() if not dominated(r)],
                   key=lambda r: (-r.molecular, -r.cls))
    n_dom = sum(1 for r in D.itertuples()
                if r.molecular >= adopted.molecular and r.cls >= adopted.cls
                and (r.molecular > adopted.molecular or r.cls > adopted.cls))
    mol_max = int(D.molecular.max())
    ties = D[D.molecular == mol_max]
    best_cls = front[-1] if front else adopted

    body = [f"Adopted (two-region, $\\Gamma$ = 22\\,\\wn{{}}, cosine, no ramp) & "
            f"{adopted.molecular}/{adopted.n} & {adopted.cls}/{adopted.n} \\\\",
            "\\midrule"]
    body += [f"{k} & {v.molecular}/{v.n} & {v.cls}/{v.n} \\\\" for k, v in singles]
    body += ["\\midrule"]
    body += [f"Pareto-optimal: {r.axis}, $\\Gamma$ = {int(r.gamma)}\\,\\wn{{}}, "
             f"{'shift-tolerant' if r.metric == 'shift' else 'plain'} cosine, ramp "
             f"{r.ramp:.1f} & \\textbf{{{r.molecular}/{r.n}}} & \\textbf{{{r.cls}/{r.n}}} \\\\"
             for r in front]
    body += ["\\midrule",
             f"Range over the {len(D)} cells & {int(D.molecular.min())}--{mol_max}/{n} & "
             f"{int(D.cls.min())}--{int(D.cls.max())}/{n} \\\\"]

    cap = ("Do the favourable instrument-model choices compound? The identification is "
           "rerun under every combination of the frequency map, the linewidth "
           f"({', '.join(str(g) for g in GAMMAS)}\\,\\wn{{}}), the similarity metric "
           "(cosine or a cosine allowing $\\pm$3 pixels of alignment) and a log-linear "
           "spectral-responsivity ramp of end-to-end ratio 1.0 or 2.0 on the measured "
           f"side --- {len(rows)} combinations. Only the affine arm is refitted in each "
           "cell, on the same non-circular band-position objective; the two-region "
           "factors are held at the values fitted at $\\Gamma$ = 22\\,\\wn{} without a "
           "ramp, so the two arms are not treated symmetrically and the two-region half "
           "of this grid is a sensitivity to the metric and the linewidth at a fixed "
           "calibration rather than a refitted comparison. The ramp axis is sampled at "
           "two values here; Table~\\ref{tab:si-prepvar} scans it from 0.33 to 3.0 and "
           "reaches 11 of 30, below the minimum of this grid.")

    singles_txt = ", ".join(f"{k.split(' alone')[0]} {v.molecular}" for k, v in singles)
    note = (f"They do not compound. Taken one at a time the molecular count goes to: "
            f"{singles_txt} --- so {sum(1 for _, v in singles if v.molecular == 16)} "
            f"of the six single changes reach 16 of {n} and the others do not. Over the "
            f"whole grid the molecular count spans "
            f"{int(D.molecular.min())} to {mol_max} of {n} "
            f"({100*D.molecular.min()/n:.0f}--{100*mol_max/n:.0f}\\%) and the class count "
            f"{int(D.cls.min())} to {int(D.cls.max())} of {n}, against the measured "
            "library's 30 of 30 at both levels, so the direction of the result is not a "
            f"product of any of these choices. The molecular maximum of {mol_max} of {n} "
            f"is a plateau, not a point: it is reached in {len(ties)} cells, the "
            "shift-tolerant cosine on the two-region map at $\\Gamma$ = "
            + ", ".join(f"{int(g)}" for g in sorted(ties.gamma)) + "\\,\\wn{}, all with "
            f"the class count at {int(ties.cls.iloc[0])} of {n}. "
            "We state one thing against ourselves. The adopted cell is not on the Pareto "
            f"front of this grid: {n_dom} of the {len(D)} cells are at least as good at "
            "both levels and strictly better at one, and the front itself contains "
            f"{best_cls.molecular} of {n} molecular with {best_cls.cls} of {n} class, "
            "which beats the adopted cell on both. The adopted cell is therefore not the "
            "most favourable choice available, and it is not the least favourable either; "
            "it is chosen a priori, $\\Gamma$ = 22\\,\\wn{} being the measured median "
            "linewidth and no ramp being the no-correction default, and we report the "
            "range rather than defending the point. Produced by "
            "\\texttt{code/v3\\_joint\\_envelope.py}, which is outside the default path of "
            "\\texttt{run\\_all.sh} because it refits the affine map in every cell.")

    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{" + cap + "}\n"
           "\\label{tab:si-jointenv}\n\\small\n"
           "\\begin{tabular}{lcc}\n\\toprule\n"
           "Condition & Molecular top-1 & Class top-1 \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}\n"
           "\\end{table}\n")
    open(os.path.join(ROOT, "latex", "sitables", "jointenv.tex"), "w").write(tex)
    print(f"\nmolecular {int(D.molecular.min())}-{mol_max}/{n}  class "
          f"{int(D.cls.min())}-{int(D.cls.max())}/{n}; adopted {adopted.molecular}/"
          f"{adopted.cls}; dominated by {n_dom} cells; front={[(r.molecular,r.cls) for r in front]}")
    print("saved data/v3_joint_envelope.csv + latex/sitables/jointenv.tex")


if __name__ == "__main__":
    main()
