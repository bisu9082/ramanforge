"""
ramanforge: sensitivity of the level-of-theory trials to the band-pairing rule.

The calibration in this work pairs measured and computed bands by one-to-one assignment
(a linear assignment problem). Nearest-neighbour matching is the obvious alternative, and
it is the rule a reader is most likely to assume, so Table S-pairrule reports the ten
agent-correction trials of Section S8 under both rules.

Both columns are computed on the pre-correction library, so that the two rules are
compared on identical inputs; run with RF_PRECORR=1, which this script sets for itself.

Writes data/v3_pairrule.csv and latex/sitables/pairrule.tex.
"""
import os
os.environ["RF_PRECORR"] = "1"          # must precede the import below

import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_step3_headtohead import (ROOT, load_measured, load_library, synth_native,
                                 measured_peaks, cos, JOB, AGENTS, CUT)
from src.spectra import parse_raman

DATA = os.path.join(ROOT, "data")

TAG2AGENT = {"GF": "GF", "A232": "A232", "A230": "A230", "A234": "A234", "GD": "GD", "HD": "HD"}
VARLAB = {"V1_cpcm": "CPCM", "V2_tightP": "tight P/F/S/Cl", "V3_wb97xd3": "$\\omega$B97X-D3"}
WIN = (600.0, 850.0)


def pos_rmse(lib, mpk, agents, s_low, s_high, rule, tol=35.0):
    """Band-position residuals under either pairing rule."""
    d = []
    for ag in agents:
        job = JOB.get(ag)
        if job is None or job not in lib:
            continue
        pk = lib[job]
        f = np.where(pk[:, 0] < CUT, pk[:, 0] * s_low, pk[:, 0] * s_high)
        keep = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max())
        f = f[keep]
        w = np.asarray(mpk[ag], float)
        if not len(f) or not len(w):
            continue
        C = np.abs(w[:, None] - f[None, :])
        if rule == "one-to-one":
            from scipy.optimize import linear_sum_assignment
            Cm = np.where(C > tol, 1e6, C)
            ri, ci = linear_sum_assignment(Cm)
            d += [float(f[j] - w[i]) for i, j in zip(ri, ci) if Cm[i, j] < 1e6]
        else:                                    # nearest neighbour, with replacement
            j = C.argmin(1)
            d += [float(f[jj] - w[i]) for i, jj in enumerate(j) if C[i, jj] <= tol]
    d = np.asarray(d)
    return (float(np.sqrt((d ** 2).mean())) if len(d) else np.nan), len(d)


def fit(lib, mpk, agents, rule, min_pairs=60):
    best = None
    for sl in np.arange(0.960, 1.0401, 0.002):
        for sh in np.arange(0.950, 1.0201, 0.002):
            r, n = pos_rmse(lib, mpk, agents, float(sl), float(sh), rule)
            if n >= min_pairs and (best is None or r < best[0]):
                best = (r, float(sl), float(sh), n)
    return best


def win_cos(spec_a, spec_b, X):
    m = (X >= WIN[0]) & (X <= WIN[1])
    return cos(spec_a[m], spec_b[m])


def main():
    X, acq = load_measured()
    base_lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]

    b2 = os.path.join(ROOT, "orca", "outputs_b2")
    trials = sorted(f[:-4] for f in os.listdir(b2) if f.endswith(".out"))

    rows = []
    for rule in ("one-to-one", "nearest"):
        rb, SLb, SHb, nb = fit(base_lib, mpk, calib, rule)
        print(f"[{rule}] s=({SLb:.3f},{SHb:.3f})  position RMSE {rb:.1f} cm-1 over {nb} pairs")
        base_spec = {k: synth_native(v, X, SLb, SHb) for k, v in base_lib.items()}
        for t in trials:
            _, tag, variant = t.split("_", 2)
            ag = TAG2AGENT.get(tag)
            if ag is None:
                continue
            job = JOB[ag]
            peaks = parse_raman(os.path.join(b2, t + ".out"))
            lib = dict(base_lib); lib[job] = peaks
            r, sl, sh, n = fit(lib, mpk, calib, rule)
            spec = {k: synth_native(v, X, sl, sh) for k, v in lib.items()}
            q = mean[ag]
            c0 = win_cos(q, base_spec[job], X)
            c1 = win_cos(q, spec[job], X)
            top0 = max(base_spec, key=lambda k: cos(q, base_spec[k]))
            top1 = max(spec, key=lambda k: cos(q, spec[k]))
            rows.append(dict(rule=rule, agent=ag, variant=variant, base=c0, new=c1,
                             delta=c1 - c0, s_low=sl, s_high=sh, rmse=r, n_pairs=n,
                             correct_base=top0 == job, correct_new=top1 == job,
                             fit_rmse=rb, fit_s_low=SLb, fit_s_high=SHb, fit_n=nb))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(DATA, "v3_pairrule.csv"), index=False)
    print(f"\nsaved data/v3_pairrule.csv ({len(D)} rows)")
    write_table(D)


def nm(a):
    return {"A230": "A-230", "A232": "A-232", "A234": "A-234"}.get(a, a)


def mark(ok):
    return "\\checkmark" if ok else "$\\times$"


def write_table(D):
    one = D[D.rule == "one-to-one"].set_index(["agent", "variant"])
    nea = D[D.rule == "nearest"].set_index(["agent", "variant"])
    order = one.delta.sort_values(ascending=False).index
    body = []
    for k in order:
        o, n = one.loc[k], nea.loc[k]
        bo = (lambda v, x: f"\\textbf{{{x}}}" if abs(v) > 0.10 else x)
        body.append(f"{nm(k[0])} & {VARLAB[k[1]]} & {bo(o.delta, f'{o.delta:+.3f}')} & "
                    f"{bo(n.delta, f'{n.delta:+.3f}')} & "
                    f"{mark(o.correct_base)} $\\rightarrow$ {mark(o.correct_new)} & "
                    f"{mark(n.correct_base)} $\\rightarrow$ {mark(n.correct_new)} \\\\")
    f1 = one.iloc[0]; f2 = nea.iloc[0]
    flips = [k for k in order
             if (one.loc[k].correct_base, one.loc[k].correct_new)
             != (nea.loc[k].correct_base, nea.loc[k].correct_new)]
    diff = ("The two rules agree on every assignment."
            if not flips else
            "The only assignment the two rules disagree on is "
            + ", ".join(f"{nm(a)} under {VARLAB[v]}" for a, v in flips) + ".")
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{The ten trials under both band-pairing rules. $\\Delta$ is the change in the
600--850\\,\\wn{{}} cosine; the assignment columns give the open-set top-1 result against the
pre-correction {{}}-member library before and after the correction, with \\checkmark{{}} for a
correct assignment and $\\times$ for an incorrect one.}}
\\label{{tab:si-pairrule}}
\\footnotesize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrcc}}
\\toprule
& & \\multicolumn{{2}}{{c}}{{$\\Delta$ cosine, 600--850 \\wn}} & \\multicolumn{{2}}{{c}}{{Assignment}} \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}
Agent & Correction & One-to-one & Nearest & One-to-one & Nearest \\\\
\\midrule
""" + "\n".join(body) + f"""
\\bottomrule
\\end{{tabular}}
}}

\\begin{{flushleft}}\\footnotesize The baseline calibration itself differs between the rules,
shown here on the pre-correction library so that the two rules are compared on identical
inputs: one-to-one assignment gives $s_{{\\mathrm{{low}}}}$ = {f1.fit_s_low:.3f},
$s_{{\\mathrm{{high}}}}$ = {f1.fit_s_high:.3f} with a position RMSE of {f1.fit_rmse:.1f}\\,\\wn{{}}
over {int(f1.fit_n)} pairs, nearest-neighbor matching gives {f2.fit_s_low:.3f} and
{f2.fit_s_high:.3f} with {f2.fit_rmse:.1f}\\,\\wn{{}} over {int(f2.fit_n)} pairs. The one-to-one
values are the ones used throughout this work, including in the identification results of
Section~\\ref{{si:ident}}. {diff}\\end{{flushleft}}
\\end{{table}}
"""
    import json
    LIBN = len(load_library())
    tex = tex.replace("pre-correction {}-member", f"pre-correction {LIBN}-member")
    out = os.path.join(ROOT, "latex", "sitables", "pairrule.tex")
    open(out, "w").write(tex)
    print("saved latex/sitables/pairrule.tex")


if __name__ == "__main__":
    main()
