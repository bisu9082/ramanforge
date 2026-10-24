"""
T1-19: component decomposition of the 600–850 cm-1 deficit.

Reviewer comment (R2-M10): the region cosine lumps position error, relative intensity error
and linewidth error into a single number, so the diagnosis that 'the computational description
of the P-F/P-N stretches is the problem' is only one of three candidates.

Method: replace a single attribute of the computed line list with the measured value, leave the
rest unchanged, resynthesize, and see how much of the cosine against the measured spectrum is
recovered. The reference is the measured spectrum itself.
  BASE  computed positions + computed intensities + Γ=22
  POS   **measured positions** + computed intensities + Γ=22
  INT   computed positions + **measured intensities** + Γ=22
  WID   computed positions + computed intensities + **per-agent measured Γ**
  P+I   measured positions + measured intensities + Γ=22
  ALL   measured positions + measured intensities + measured Γ   (upper bound: only unmatched bands remain)
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, measured_peaks, fit_scale,
                                 scattering_factor, cos, AGENTS, JOB, CUT)

DATA = os.path.join(ROOT, "data")
LO, HI, TOL = 600.0, 850.0, 35.0
PCONT = {"GD", "GF", "VX", "A230", "A232", "A234", "A242"}   # contains phosphorus


def lorentz(grid, pos, amp, gam):
    g = np.asarray(gam, float) / 2.0
    I = np.zeros_like(grid, float)
    for p, a, gg in zip(pos, amp, np.broadcast_to(g, np.shape(pos))):
        I += a * gg * gg / ((grid - p) ** 2 + gg * gg)
    return I


def main():
    X, acq = load_measured(); lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]
    _, SL, SH, _ = fit_scale(lib, mpk, calib)

    LW = pd.read_csv(f"{DATA}/v3_linewidths.csv")
    gper = LW.groupby("agent")["gamma"].mean().to_dict()
    gmed = float(LW["gamma"].median())

    win = (X >= LO) & (X <= HI)
    rows = []
    for ag in [a for a in AGENTS if a in JOB and len(acq[a]) >= 2]:
        pk = lib[JOB[ag]]
        f = np.where(pk[:, 0] < CUT, pk[:, 0] * SL, pk[:, 0] * SH)
        b = pk[:, 1].astype(float) * scattering_factor(f)
        keep = (f > LO - 150) & (f < HI + 150) & (b > 0.02 * b.max())
        f, b = f[keep], b[keep]
        gm = float(gper.get(ag, gmed))

        # measured bands inside the window (with margin) and their intensities
        w = np.asarray([v for v in mpk[ag] if LO - 150 < v < HI + 150], float)
        a_m = np.asarray([mean[ag][np.argmin(np.abs(X - v))] for v in w], float)
        if not len(f) or not len(w):
            continue

        C = np.abs(w[:, None] - f[None, :]); C = np.where(C > TOL, 1e6, C)
        ri, ci = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]

        f_pos = f.copy(); b_int = b.copy()
        for i, j in pairs:
            f_pos[j] = w[i]
            b_int[j] = a_m[i] * (b.max() / (a_m.max() + 1e-12))

        M = mean[ag][win]
        def C_(pos, amp, gam):
            return cos(M, lorentz(X, pos, amp, gam)[win])

        rows.append(dict(
            agent=ag, phos=ag in PCONT, n_calc=len(f), n_meas=len(w), n_pair=len(pairs),
            gamma_meas=gm,
            BASE=C_(f, b, 22.0), POS=C_(f_pos, b, 22.0), INT=C_(f, b_int, 22.0),
            WID=C_(f, b, gm), PI=C_(f_pos, b_int, 22.0), ALL=C_(f_pos, b_int, gm)))

    D = pd.DataFrame(rows)
    D["d_POS"] = D.POS - D.BASE; D["d_INT"] = D.INT - D.BASE; D["d_WID"] = D.WID - D.BASE
    D.to_csv(f"{DATA}/v3_t119_decompose.csv", index=False)

    print("600–850 cm-1  cosine when only one attribute is replaced by the measured value")
    print("=" * 96)
    print(f"{'agent':<7}{'P':>3}{'pair/comp/meas':>14}{'BASE':>8}{'POS':>8}{'INT':>8}{'WID':>8}{'P+I':>8}{'ALL':>8}")
    print("-" * 96)
    for _, r in D.iterrows():
        print(f"{r.agent:<7}{'●' if r.phos else '○':>3}"
              f"{f'{int(r.n_pair)}/{int(r.n_calc)}/{int(r.n_meas)}':>14}"
              f"{r.BASE:>8.3f}{r.POS:>8.3f}{r.INT:>8.3f}{r.WID:>8.3f}{r.PI:>8.3f}{r.ALL:>8.3f}")
    print("-" * 96)
    for lab, sub in (("total", D), ("P-cont", D[D.phos]), ("P-free", D[~D.phos])):
        print(f"{lab:<7}{'':>3}{'':>14}{sub.BASE.mean():>8.3f}{sub.POS.mean():>8.3f}"
              f"{sub.INT.mean():>8.3f}{sub.WID.mean():>8.3f}{sub.PI.mean():>8.3f}{sub.ALL.mean():>8.3f}")
    print("\nRecovery (mean Δ vs BASE)")
    for lab, sub in (("total", D), ("P-cont", D[D.phos]), ("P-free", D[~D.phos])):
        print(f"  {lab:<7} position {sub.d_POS.mean():+.3f}   intensity {sub.d_INT.mean():+.3f}   "
              f"linewidth {sub.d_WID.mean():+.3f}   position+intensity {(sub.PI-sub.BASE).mean():+.3f}   "
              f"all {(sub.ALL-sub.BASE).mean():+.3f}")
    print(f"\nsaved data/v3_t119_decompose.csv")


if __name__ == "__main__":
    main()
