"""
v3 wavenumber-axis checks for the SI.

Produces SI table \\label{tab:si-affine} (affine vs two-region calibration models fitted
by the same non-circular, band-position-only objective, and the identification obtained
under each) and SI table \\label{tab:si-offsetscan} (identification under an imposed
additive offset on the measured wavenumber axis, with the calibration refitted at each
offset).
"""
import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H


def affine_vs_two_region():
    """Fit an affine axis model against the adopted two-region scaling and rerun the
    identification under each (SI table tab:si-affine)."""
    X, acq = H.load_measured()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    lib = H.load_library()
    CAL = [a for a in H.AGENTS if a in H.JOB]
    QUERY = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]

    def apply_two(f, sl, sh): return np.where(f < H.CUT, f * sl, f * sh)
    def apply_aff(f, g, b):   return g * f + b

    def pos_rmse(mapper, tol=35.0):
        d = []
        for ag in CAL:
            pk = lib[H.JOB[ag]]; f = mapper(pk[:, 0])
            keep = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max()); f = f[keep]
            w = np.asarray(mpk[ag], float)
            if not len(f) or not len(w): continue
            C = np.abs(w[:, None] - f[None, :]); C = np.where(C > tol, 1e6, C)
            ri, ci = linear_sum_assignment(C)
            d += [float(f[j] - w[i]) for i, j in zip(ri, ci) if C[i, j] < 1e6]
        d = np.asarray(d)
        return (float(np.sqrt((d ** 2).mean())) if len(d) else np.nan), len(d)

    # fit the affine parameters by the same non-circular procedure (band positions only)
    best = None
    for g in np.arange(0.940, 1.0201, 0.002):
        for b in np.arange(-10.0, 60.1, 2.0):
            r, n = pos_rmse(lambda f, g=g, b=b: apply_aff(f, g, b))
            if n >= 60 and (best is None or r < best[0]): best = (r, float(g), float(b), n)
    rA, gA, bA, nA = best
    r2, SL, SH, n2 = H.fit_scale(lib, mpk, CAL)
    print(f"[2-region]  s=({SL:.3f},{SH:.3f})   RMSE={r2:.2f}  pairs={n2}")
    print(f"[affine] gain={gA:.3f}, offset={bA:+.1f}   RMSE={rA:.2f}  pairs={nA}")

    def synth(peaks, mapper, gamma=22.0):
        f = mapper(peaks[:, 0]); a = peaks[:, 1].astype(float) * H.scattering_factor(f)
        if a.max() > 0: a = a / a.max()
        g = gamma / 2.0; I = np.zeros_like(X, dtype=float)
        for ff, aa in zip(f, a): I += aa * g * g / ((X - ff) ** 2 + g * g)
        return I / (I.max() + 1e-12)

    def run(mapper, label):
        dft = {k: synth(v, mapper) for k, v in lib.items()}
        n = mol = cls = 0; selfs = []; margins = []; wrong = {}
        for a in QUERY:
            for rep, spec in acq[a]:
                sims = {k: H.cos(spec, v) for k, v in dft.items()}
                o = sorted(sims.items(), key=lambda t: -t[1]); n += 1
                top = o[0][0]; mol += top == H.JOB[a]; cls += H.cls_of(top) == H.cls_of(H.JOB[a])
                selfs.append(sims[H.JOB[a]]); margins.append(o[0][1] - o[1][1])
                if top != H.JOB[a]: wrong.setdefault(a, set()).add(top)
        # per agent (majority of that agent's acquisitions)
        ag = 0
        for a in QUERY:
            c = sum(1 for rep, spec in acq[a]
                    if sorted({k: H.cos(spec, v) for k, v in dft.items()}.items(), key=lambda t: -t[1])[0][0] == H.JOB[a])
            ag += c > len(acq[a]) / 2
        print(f"\n=== {label} ===")
        print(f"  computed library molecule top-1 : {mol}/{n}   class {cls}/{n}   agents {ag}/{len(QUERY)}")
        print(f"  mean self-cosine {np.mean(selfs):.3f}   mean margin {np.mean(margins):.3f}")
        print(f"  errors: " + ", ".join(f"{a}->{'/'.join(sorted(v))}" for a, v in sorted(wrong.items())))
        return mol, cls, np.mean(selfs), np.mean(margins)

    run(lambda f: apply_two(f, SL, SH), f"2-region calibration (adopted, s={SL}/{SH})")
    run(lambda f: apply_aff(f, gA, bA), f"affine calibration (gain={gA:.3f}, offset={bA:+.1f})")
    # the measured library does not depend on the calibration - reported as a check
    mm = 0
    for a in QUERY:
        for rep, spec in acq[a]:
            o = sorted({b: H.cos(spec, np.mean([s for r, s in acq[b] if not (b == a and r == rep)], 0))
                        for b in H.AGENTS if len(acq[b]) >= 1}.items(), key=lambda t: -t[1])
            mm += o[0][0] == a
    print(f"\nmeasured library (calibration-independent): {mm}/30")


def offset_scan():
    """Impose an additive offset on the measured axis, refit the calibration at each
    offset and recompute the identification (SI table tab:si-offsetscan)."""
    X, acq = H.load_measured()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    lib = H.load_library()
    CAL = [a for a in H.AGENTS if a in H.JOB]
    QUERY = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]

    print("Impose an additive offset delta on the measured wavenumber axis; at each delta the calibration is refitted and the identification recomputed")
    print("(delta>0 = assume the measured axis read higher than the truth, and undo it)\n")
    print(f"{'delta(cm-1)':>9s} {'s_low':>7s} {'s_high':>7s} {'RMSE':>6s} {'pairs':>4s} {'computed':>9s} {'measured':>9s} {'self':>6s}")
    for d in [-30, -20, -10, -5, 0, 5, 10, 20, 30]:
        Xs = X - d
        mp = {a: H.measured_peaks(Xs, mean[a]) for a in H.AGENTS}
        r, sl, sh, n = H.fit_scale(lib, mp, CAL)
        dft = {}
        for k, v in lib.items():
            f = np.where(v[:, 0] < H.CUT, v[:, 0] * sl, v[:, 0] * sh)
            a_ = v[:, 1].astype(float) * H.scattering_factor(f)
            if a_.max() > 0: a_ = a_ / a_.max()
            g = 11.0; I = np.zeros_like(Xs, dtype=float)
            for ff, aa in zip(f, a_): I += aa * g * g / ((Xs - ff) ** 2 + g * g)
            dft[k] = I / (I.max() + 1e-12)
        mol = 0; tot = 0; selfs = []
        for a in QUERY:
            for rep, spec in acq[a]:
                sims = {k: H.cos(spec, v) for k, v in dft.items()}
                o = sorted(sims.items(), key=lambda t: -t[1]); tot += 1
                mol += o[0][0] == H.JOB[a]; selfs.append(sims[H.JOB[a]])
        mm = 0
        for a in QUERY:
            for rep, spec in acq[a]:
                o = sorted({b: H.cos(spec, np.mean([s for r, s in acq[b] if not (b == a and r == rep)], 0))
                            for b in H.AGENTS}.items(), key=lambda t: -t[1])
                mm += o[0][0] == a
        print(f"{d:>9d} {sl:7.3f} {sh:7.3f} {r:6.2f} {n:4d} {mol:>6d}/30 {mm:>6d}/30 {np.mean(selfs):6.3f}")


def main():
    print("=" * 68)
    print("tab:si-affine  affine vs two-region wavenumber calibration")
    print("=" * 68)
    affine_vs_two_region()
    print("\n" + "=" * 68)
    print("tab:si-offsetscan  imposed additive axis offset")
    print("=" * 68)
    offset_scan()


if __name__ == "__main__":
    main()
