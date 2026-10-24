"""
T1-14b: calibration sensitivity to the band detection criterion and the pairing tolerance window.

Reviewer comment (R1-M2): the 35 cm-1 tolerance window is comparable to the minimum peak
detection spacing, so the assignment is nearly degenerate, and an RMSE computed only over
the survivors that pass the gate is optimistic because it excludes the worst bands by
design. Here all three detection criteria × both tolerance windows are reported.
The penalty for unmatched bands is always fixed at 35 cm-1 so that the results stay comparable.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import load_measured, load_library, fit_scale, AGENTS, JOB, CUT

PEN = 35.0
def _noise_sd(X, spec):
    """Noise estimate from the band-free window above 1600 cm-1."""
    d = np.diff(spec[X > 1600.0])
    return float(np.std(d)) / np.sqrt(2.0)


def _noise_scaled(X, spec):
    """Per-agent detection threshold: five times that spectrum's own noise, but never
    looser than the adopted criterion. The measured spectra differ in noise by more than
    an order of magnitude, so a single absolute prominence admits many more peaks from
    the noisiest spectrum than from the quietest; this criterion removes that imbalance."""
    return dict(prominence=max(0.05, 5.0 * _noise_sd(X, spec)), distance=2)


MODES = [("prominence $>$ 0.05, separation 2 points", dict(prominence=0.05, distance=2)),
         ("height $>$ 0.10 and prominence $>$ 0.05, separation 4 points",
          dict(height=0.10, prominence=0.05, distance=4)),
         ("height $>$ 0.12, separation 6 points", dict(height=0.12, distance=6)),
         ("prominence $>$ max(0.05, $5\\sigma$) per agent, separation 2 points", _noise_scaled)]

X, acq = load_measured(); lib = load_library()
mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
CAL = [a for a in AGENTS if a in JOB]
rows = []
for lab, kw in MODES:
    mpk = {a: X[find_peaks(mean[a],
                           **(kw(X, mean[a]) if callable(kw) else kw))[0]]
           for a in AGENTS}
    _, SL, SH, _ = fit_scale(lib, mpk, CAL, min_pairs=20)
    for tol in (35.0, 20.0):
        d = []; nm = nc = npr = 0
        for ag in CAL:
            pk = lib[JOB[ag]]
            f = np.where(pk[:, 0] < CUT, pk[:, 0] * SL, pk[:, 0] * SH)
            k = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max()); f = f[k]
            w = np.asarray(mpk[ag], float)
            if not len(f) or not len(w): continue
            C = np.abs(w[:, None] - f[None, :]); C = np.where(C > tol, 1e6, C)
            ri, ci = linear_sum_assignment(C)
            m = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]
            d += [float(f[j] - w[i]) for i, j in m]
            nm += len(w); nc += len(f); npr += len(m)
        d = np.asarray(d)
        rows.append(dict(criterion=lab, tol=tol, s_low=SL, s_high=SH,
                         rmse=float(np.sqrt((d ** 2).mean())), n_pair=npr,
                         n_meas=nm, n_calc=nc,
                         rmse_pen=float(np.sqrt(np.concatenate(
                             [d ** 2, np.full(nm + nc - 2 * npr, PEN ** 2)]).mean()))))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(ROOT, "data", "v3_detect_sensitivity.csv"), index=False)
print(D.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
