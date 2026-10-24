"""
v3 symmetric-preprocessing check for the SI.

Produces SI table \\label{tab:si-aslssym}: the AsLS baseline correction and the dead-pixel
interpolation, which the adopted pipeline applies to the measurement only, are applied to
the synthesised computed spectra as well, and the per-region measured-vs-computed cosines,
the molecular top-1 and the self-cosine are recomputed under each condition.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H
from src.preprocess_v3 import asls


def asls_symmetry():
    """AsLS and dead-pixel treatment applied symmetrically: identification, the 600-850
    window and the unmatched computed intensity in that window."""
    X, acq = H.load_measured()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    lib = H.load_library()
    CAL = [a for a in H.AGENTS if a in H.JOB]
    QUERY = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]
    META = json.load(open(os.path.join(ROOT, "data", "v3_preproc_meta.json")))
    # v3_preproc_meta.json stores each dead pixel as [detector index, wavenumber]. The
    # detector index addresses the raw 512-pixel array, not the trimmed analysis grid X,
    # so the position on X has to be found by wavenumber.
    DEAD_WN = [float(p[1]) for p in META["dead_px"]]
    DEAD = [int(np.argmin(np.abs(X - w))) for w in DEAD_WN]
    print(f"grid {len(X)} points; dead pixels at {[round(w, 1) for w in DEAD_WN]} cm-1 "
          f"-> grid indices {DEAD}")
    REG = [("250-600", 250, 600), ("600-850", 600, 850), ("850-1150", 850, 1150), ("1150-1400", 1150, 1400), ("1400-1800", 1400, 1800)]
    r2, SL, SH, n2 = H.fit_scale(lib, mpk, CAL)

    def synth(peaks, sl, sh, gamma=22.0, do_asls=False, do_dead=False):
        f = np.where(peaks[:, 0] < H.CUT, peaks[:, 0] * sl, peaks[:, 0] * sh)
        a = peaks[:, 1].astype(float) * H.scattering_factor(f)
        if a.max() > 0: a = a / a.max()
        g = gamma / 2.0; I = np.zeros_like(X, dtype=float)
        for ff, aa in zip(f, a): I += aa * g * g / ((X - ff) ** 2 + g * g)
        if do_dead:
            idx = np.arange(len(X)); m = np.zeros(len(X), bool)
            m[DEAD] = True
            if m.any(): I[m] = np.interp(idx[m], idx[~m], I[~m])
        if do_asls:
            I = I - asls(I)
            I = np.clip(I, 0, None)
        return I / (I.max() + 1e-12)

    def evaluate(do_asls, do_dead, label):
        dft = {k: synth(v, SL, SH, do_asls=do_asls, do_dead=do_dead) for k, v in lib.items()}
        mol = tot = cls = 0; selfs = []
        for a in QUERY:
            for rep, spec in acq[a]:
                sims = {k: H.cos(spec, v) for k, v in dft.items()}
                o = sorted(sims.items(), key=lambda t: -t[1]); tot += 1
                mol += o[0][0] == H.JOB[a]; cls += H.cls_of(o[0][0]) == H.cls_of(H.JOB[a])
                selfs.append(sims[H.JOB[a]])
        reg = {}
        for a in QUERY + ["VX", "HD"]:
            if a not in H.JOB: continue
            d = dft[H.JOB[a]]
            reg[a] = [H.cos(mean[a][(X >= lo) & (X < hi)], d[(X >= lo) & (X < hi)]) for _, lo, hi in REG]
        # unmatched computed intensity inside the window (the 35% claim)
        unm = []
        for a in QUERY:
            if a == "HN2": continue
            pk = lib[H.JOB[a]]
            f = np.where(pk[:, 0] < H.CUT, pk[:, 0] * SL, pk[:, 0] * SH)
            amp = pk[:, 1].astype(float) * H.scattering_factor(f)
            w = (f >= 600) & (f < 850) & (amp > 0.02 * amp.max())
            if not w.any(): continue
            mp = np.asarray(mpk[a], float); mp = mp[(mp >= 600) & (mp < 850)]
            tot_i = amp[w].sum()
            un = sum(amp[i] for i in np.where(w)[0] if len(mp) == 0 or np.abs(mp - f[i]).min() > 35)
            unm.append(un / tot_i if tot_i > 0 else 0)
        print(f"\n=== {label} ===")
        print(f"  computed molecule top-1 {mol}/{tot}   class {cls}/{tot}   self-cosine {np.mean(selfs):.3f}")
        print(f"  600-850 window mean cosine {np.mean([v[1] for v in reg.values()]):.3f}")
        print(f"  mean unmatched computed intensity in the window {np.mean(unm) * 100:.1f}%")
        print("  per agent 600-850: " + "  ".join(f"{a} {reg[a][1]:.3f}" for a in ["GD", "GF", "VX", "A232", "A242"] if a in reg))
        return mol, np.mean(selfs), np.mean([v[1] for v in reg.values()])

    evaluate(False, False, "as adopted (no AsLS or dead-pixel treatment on the computed spectra)")
    evaluate(True, False, "AsLS applied symmetrically")
    evaluate(True, True, "AsLS + dead pixels applied symmetrically")


def asls_region_table():
    """Per-region measured-vs-computed cosines with and without AsLS on the computed
    spectra - the body of SI table tab:si-aslssym."""
    X, acq = H.load_measured()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    lib = H.load_library(); CAL = [a for a in H.AGENTS if a in H.JOB]
    QUERY = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]
    r2, SL, SH, _ = H.fit_scale(lib, mpk, CAL)
    REG = [("250-600", 250, 600), ("600-850", 600, 850), ("850-1150", 850, 1150), ("1150-1400", 1150, 1400), ("1400-1800", 1400, 1800)]

    def synth(pk, aslsq):
        f = np.where(pk[:, 0] < H.CUT, pk[:, 0] * SL, pk[:, 0] * SH)
        a = pk[:, 1].astype(float) * H.scattering_factor(f)
        if a.max() > 0: a = a / a.max()
        g = 11.0; I = np.zeros_like(X, float)
        for ff, aa in zip(f, a): I += aa * g * g / ((X - ff) ** 2 + g * g)
        if aslsq: I = np.clip(I - asls(I), 0, None)
        return I / (I.max() + 1e-12)

    rows = {}
    for q in (False, True):
        dft = {k: synth(v, q) for k, v in lib.items()}
        reg = []
        for _, lo, hi in REG:
            c = [H.cos(mean[a][(X >= lo) & (X < hi)], dft[H.JOB[a]][(X >= lo) & (X < hi)]) for a in H.JOB]
            reg.append(np.mean(c))
        mol = tot = 0; selfs = []
        for a in QUERY:
            for rep, spec in acq[a]:
                sims = {k: H.cos(spec, v) for k, v in dft.items()}
                o = sorted(sims.items(), key=lambda t: -t[1]); tot += 1
                mol += o[0][0] == H.JOB[a]; selfs.append(sims[H.JOB[a]])
        rows["AsLS applied" if q else "as adopted"] = (reg, mol, tot, np.mean(selfs))
    print(f"{'condition':12s} " + " ".join(f"{n:>10s}" for n, _, _ in REG) + f" {'molecular':>8s} {'self':>7s}")
    for k, (reg, mol, tot, sf) in rows.items():
        print(f"{k:12s} " + " ".join(f"{v:10.3f}" for v in reg) + f" {mol:>5d}/{tot} {sf:7.3f}")
    a, b = rows["as adopted"][0], rows["AsLS applied"][0]
    print(f"{'change':12s} " + " ".join(f"{y - x:+10.3f}" for x, y in zip(a, b)))


def main():
    print("=" * 68)
    print("tab:si-aslssym  preprocessing applied symmetrically to the computed spectra")
    print("=" * 68)
    asls_symmetry()
    print("\n" + "-" * 68)
    print("per-region table")
    print("-" * 68)
    asls_region_table()


if __name__ == "__main__":
    main()
