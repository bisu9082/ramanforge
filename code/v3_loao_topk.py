"""
v3 leave-one-agent-out calibration and top-k checks for the SI.

Produces SI table \\label{tab:si-loaoident} (the identification rerun under each held-out
calibration, i.e. the two scale factors refitted with one agent's band pairs removed) and
SI table \\label{tab:si-topk} (top-k accuracy and the rank distribution of the correct
entry for the computed and the measured library, plus the errors decided by a margin
below 0.01).
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

X, acq = H.load_measured()
mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
lib = H.load_library()
CAL = [a for a in H.AGENTS if a in H.JOB]
QUERY = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]


def ident(sl, sh):
    dft = {k: H.synth_native(v, X, sl, sh) for k, v in lib.items()}
    mol = cls = tot = 0; ranks = []; margins = []
    for a in QUERY:
        for rep, spec in acq[a]:
            sims = {k: H.cos(spec, v) for k, v in dft.items()}
            o = sorted(sims.items(), key=lambda t: -t[1]); tot += 1
            mol += o[0][0] == H.JOB[a]; cls += H.cls_of(o[0][0]) == H.cls_of(H.JOB[a])
            r = [k for k, _ in o].index(H.JOB[a]) + 1; ranks.append((a, rep, r, o[0][0], o[0][1] - o[1][1], sims[H.JOB[a]]))
    return mol, cls, tot, ranks


def loao_identification():
    """Identification under each held-out calibration (SI table tab:si-loaoident).
    Returns the adopted scale factors and the totals for the top-k section."""
    print("### 1. Identification under the LOAO calibration (checks the 'accuracy unchanged' claim in the main text) ###")
    r0, SL, SH, n0 = H.fit_scale(lib, mpk, CAL)
    m0, c0, t0, _ = ident(SL, SH)
    print(f"{'held out':12s} {'s_low':>7s} {'s_high':>7s} {'RMSE':>6s} {'molecular':>8s} {'class':>8s}")
    print(f"{'(none)':12s} {SL:7.3f} {SH:7.3f} {r0:6.2f} {m0:>5d}/{t0} {c0:>5d}/{t0}")
    res = []
    for held in CAL:
        sub = [a for a in CAL if a != held]
        r, sl, sh, n = H.fit_scale(lib, mpk, sub)
        m, c, t, _ = ident(sl, sh); res.append((held, sl, sh, r, m, c))
        print(f"{held:12s} {sl:7.3f} {sh:7.3f} {r:6.2f} {m:>5d}/{t} {c:>5d}/{t}")
    ms = [x[4] for x in res]
    print(f"\n  molecular top-1 range {min(ms)}-{max(ms)} / {t0}   (full calibration {m0})")
    return SL, SH, t0


def topk_ranks(SL, SH):
    """top-k accuracy and the rank distribution of the correct entry (SI table tab:si-topk)."""
    print("\n### 2. top-k and the rank distribution of the correct entry ###")
    m, c, t, ranks = ident(SL, SH)
    R = np.array([x[2] for x in ranks])
    for k in (1, 3, 5, 10):
        print(f"  computed library top-{k:<2d} : {(R <= k).sum():2d}/{t}")
    print(f"  median rank of the correct entry {np.median(R):.0f}, worst {R.max()}, distribution {dict(zip(*np.unique(R, return_counts=True)))}")
    ties = [x for x in ranks if x[4] < 0.01 and x[3] != H.JOB[x[0]]]
    print(f"\n  errors with margin < 0.01: {len(ties)} cases / {int(t - m)} error cases")
    for a, rep, r, top, mg, sf in sorted(ties, key=lambda z: z[4]):
        print(f"    {a}#{rep}  ->{top:20s} margin {mg:+.4f}  correct entry rank {r}  self {sf:.3f}")
    # measured library
    mr = []
    for a in QUERY:
        for rep, spec in acq[a]:
            ml = {b: np.mean([s for r2_, s in acq[b] if not (b == a and r2_ == rep)], 0) for b in H.AGENTS
                  if [s for r2_, s in acq[b] if not (b == a and r2_ == rep)]}
            o = sorted({b: H.cos(spec, v) for b, v in ml.items()}.items(), key=lambda z: -z[1])
            mr.append([k for k, _ in o].index(a) + 1)
    MR = np.array(mr)
    print(f"\n  measured library top-1 {(MR <= 1).sum()}/{len(MR)}  top-3 {(MR <= 3).sum()}/{len(MR)}  median rank {np.median(MR):.0f}")


def main():
    SL, SH, t0 = loao_identification()
    topk_ranks(SL, SH)


if __name__ == "__main__":
    main()
