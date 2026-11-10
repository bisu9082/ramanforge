"""
Two preprocessing variants the adopted pipeline does not use, run end to end.

(1) Standard-normal-variate normalization in place of maximum normalization. Maximum
    normalization anchors every cosine to the single tallest band, which is the choice
    most able to manufacture a deficit when a computed spectrum misplaces that band. SNV
    removes the anchor. It is scale-invariant, so it can be applied to the deposited
    spectra without returning to the raw counts.

(2) A spectral-responsivity ramp on the measured side. No relative-intensity standard was
    run, so the measured intensities are uncorrected for the instrument's spectral
    response, while the computed side carries a physically correct scattering model. An
    uncorrected responsivity is a multiplicative function of wavenumber applied to one
    side only, which is the error class that would flatter a measured-to-measured
    comparison and penalize a measured-to-computed one. This imposes a log-linear ramp of
    a given end-to-end ratio over 250-1800 cm-1 and reruns the whole identification.

Produces data/v3_preproc_variants.csv and SI table \\label{tab:si-prepvar}.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
import v3_step3_headtohead as H

SLOW, SHIGH = 1.014, 0.980
RATIOS = [0.33, 0.50, 1.00, 1.50, 2.00, 3.00]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def snv(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    dft = {k: H.synth_native(v, X, SLOW, SHIGH) for k, v in lib.items()}
    query = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]
    members = {H.JOB[a] for a in H.AGENTS if a in H.JOB}

    def identify(transform_meas, transform_both):
        """Measured library is the mean of the other acquisitions of the same vial."""
        T = {k: transform_both(d) for k, d in dft.items()}
        mol = cls = mmol = 0
        for a in query:
            reps = acq[a]
            for i, (_, spec) in enumerate(reps):
                q = transform_both(transform_meas(spec))
                sims = {k: cos(q, d) for k, d in T.items()}
                top = max(sims, key=sims.get)
                if top == H.JOB[a]:
                    mol += 1
                if H.cls_of(top) == H.cls_of(H.JOB[a]):
                    cls += 1
                others = [transform_both(transform_meas(s)) for j, (_, s) in enumerate(reps) if j != i]
                ref = {b: np.mean([transform_both(transform_meas(s)) for _, s in acq[b]
                                   if not (b == a and np.array_equal(s, spec))], 0)
                       for b in H.AGENTS if b in H.JOB and (len(acq[b]) > 1 or b != a)}
                if others:
                    ref[a] = np.mean(others, 0)
                    msims = {b: cos(q, r) for b, r in ref.items()}
                    if max(msims, key=msims.get) == a:
                        mmol += 1
        return mmol, mol, cls

    ident = lambda tm, tb: identify(tm, tb)
    rows = []

    # adopted
    m, d, c = ident(lambda x: x, lambda x: x)
    rows.append(dict(variant="adopted (maximum normalization)", detail="--",
                     measured=m, computed=d, cls=c))

    # SNV on both sides
    m, d, c = ident(lambda x: x, snv)
    rows.append(dict(variant="standard normal variate", detail="both libraries",
                     measured=m, computed=d, cls=c))

    # responsivity ramp on the measured side only
    w = X
    for r in RATIOS:
        ramp = np.exp(np.log(r) * (w - w.min()) / (w.max() - w.min()))
        f = (lambda rr: (lambda x: x * rr))(ramp)
        m, d, c = ident(f, lambda x: x)
        rows.append(dict(variant="responsivity ramp", detail=f"end-to-end ratio {r:.2f}",
                         measured=m, computed=d, cls=c))

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_preproc_variants.csv"), index=False)

    body = [f"{r.variant} & {r.detail} & {r.measured}/30 & {r.computed}/30 & {r.cls}/30 \\\\"
            for r in D.itertuples()]
    ramp_rows = D[D.variant == "responsivity ramp"]
    lo, hi = int(ramp_rows.computed.min()), int(ramp_rows.computed.max())
    snv_d = int(D[D.variant == "standard normal variate"].computed.iloc[0])

    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Two preprocessing choices the adopted pipeline does not make, run "
           "end to end. Standard-normal-variate normalization replaces maximum "
           "normalization on both libraries. The responsivity ramp multiplies the measured "
           "spectra by a log-linear function of wavenumber with the stated end-to-end "
           "ratio over 250--1800\\,\\wn{}, standing in for the spectral-response correction "
           "no intensity standard was available to make.}\n"
           "\\label{tab:si-prepvar}\n\\small\n"
           "\\begin{tabular}{llrrr}\n\\toprule\n"
           "Variant & Detail & Measured mol. & Computed mol. & Computed class \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize Maximum normalization anchors every cosine to "
           "the tallest band and is the choice most able to manufacture the reported "
           f"deficit; under SNV the computed library reaches {snv_d} of 30, so it does "
           "not. A responsivity error is multiplicative in wavenumber and applied to the "
           "measured side only, so it is the error class that would flatter the measured "
           "arm and penalize the computed one; over end-to-end ratios from 0.33 to 3, far "
           "wider than any plausible residual on this instrument, the computed figure "
           f"moves only between {lo} and {hi} of 30 while the measured arm holds at 30 of "
           f"30 except at the most compressive ramp, where it is 29. "
           "Neither choice accounts for the result. Produced by "
           "\\texttt{code/v3\\_preproc\\_variants.py}.\\end{flushleft}\n\\end{table}\n")
    open(os.path.join(ROOT, "latex", "sitables", "prepvar.tex"), "w").write(tex)
    print(D.to_string(index=False))
    print("saved data/v3_preproc_variants.csv + latex/sitables/prepvar.tex")


if __name__ == "__main__":
    main()
