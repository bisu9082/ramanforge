"""
Bands inside the 600-850 cm-1 window that find no partner in the other library, expressed
as a share of the intensity carried in that window. Produces data/v3_t119_unmatched.csv,
which feeds Figure 4(d) and Table S-unmatched.

Round-4 revision. This script did not previously exist: the deposited CSV had been written
by hand from an earlier state of the analysis and run_all.sh could not regenerate it. Two
things were wrong with that file and are corrected here.

  1. It was computed on the PRE-CORRECTION library (the GF and A-232 geometries that
     Section 3.8 shows are not at the global minimum) and on the pre-correction scale
     factors s = (1.018, 0.988). The two agents the manuscript quoted as its headline
     examples, A-232 at 75% and GF at 50%, are precisely the two whose geometries were
     replaced. This script uses the adopted library and the adopted calibration.
  2. Computed bands were taken from the raw harmonic line list above 2% of maximum
     activity, while measured bands were resolved local maxima. Both sides are now
     detected with the one criterion stated in the Methods: synthesis onto the measured
     native grid at the adopted width, then local maxima at prominence > 0.05 with a
     minimum separation of two grid points.

Band intensity is the height of the detected maximum on the normalized spectrum, which is
the same quantity on both sides.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
DATA = os.path.join(ROOT, "data")
import v3_step3_headtohead as H

LO, HI, TOL = 600.0, 850.0, 35.0


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    calib = [a for a in H.AGENTS if a in H.JOB]
    _, SL, SH, _ = H.fit_scale(lib, mpk, calib)
    print(f"[calibration] s=({SL:.3f}, {SH:.3f}) — symmetric detection on both sides")

    rows = []
    for ag in [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]:
        # computed bands: the stated criterion, with the height of each detected maximum
        pk = lib[H.JOB[ag]]
        f_all = np.where(pk[:, 0] < H.CUT, pk[:, 0] * SL, pk[:, 0] * SH)
        fc, bc = H.peaks_of_lines(f_all, pk[:, 1], X, return_height=True)
        k = (fc >= LO) & (fc <= HI)
        fc, bc = fc[k], bc[k]

        # measured bands: the same criterion on the agent mean spectrum
        w = np.asarray([v for v in mpk[ag] if LO <= v <= HI], float)
        bm = np.asarray([mean[ag][np.argmin(np.abs(X - v))] for v in w], float)

        pairs = []
        if len(w) and len(fc):
            C = np.abs(w[:, None] - fc[None, :])
            C = np.where(C > TOL, 1e6, C)
            ri, ci = linear_sum_assignment(C)
            pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]
        pm = {i for i, _ in pairs}
        pc = {j for _, j in pairs}
        unm_meas = (bm[[i for i in range(len(w)) if i not in pm]].sum() / bm.sum()) if len(w) else 1.0
        unm_calc = (bc[[j for j in range(len(fc)) if j not in pc]].sum() / bc.sum()) if len(fc) else 1.0
        # how much of the unpaired computed count is forced by the one-to-one constraint
        forced = max(0, len(fc) - len(w))
        rows.append(dict(agent=ag, n_c=len(fc), n_m=len(w), pair=len(pairs),
                         unm_meas=float(unm_meas), unm_calc=float(unm_calc),
                         unpaired_c=len(fc) - len(pairs), forced_excess=forced))

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(DATA, "v3_t119_unmatched.csv"), index=False)
    print(D.to_string(index=False))
    nb = D[D.agent != "HN2"]
    print(f"\nmean unmatched computed intensity, benzene vial excluded: {nb.unm_calc.mean()*100:.0f}%")
    print(f"unpaired computed bands forced by n_calc > n_meas: "
          f"{int(nb.forced_excess.sum())} of {int(nb.unpaired_c.sum())}")
    print("wrote data/v3_t119_unmatched.csv")


if __name__ == "__main__":
    main()
