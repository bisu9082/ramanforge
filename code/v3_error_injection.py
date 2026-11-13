"""How accurate must a computed Raman reference be? A controlled error-injection test.

The computed library fails, and the substitution analysis (Section S5) says which attribute
of its line list matters, but not how much error a library can carry and still identify.
This script answers that by building reference libraries whose error is KNOWN and graded,
and running the identical identification on the same 30 queries.

Construction. For every agent, the reference the measured library uses (the mean of the
other acquisitions of the same vial, so a query never sits in its own reference) is reduced
to a line list: band maxima under the prominence criterion used throughout (0.05, two
points) and their heights. The line list is rendered exactly as a computed entry is:
Lorentzians of the adopted width (Gamma = 22 cm-1) on the native grid, unit maximum. With
no error injected this is a "perfect" computed library -- right positions, intensities and
band set, computed-library lineshape -- and it identifies 30 of 30.

Errors injected independently per band and per draw:
  sigma  positional error, Gaussian, s.d. in units of Gamma. It is the residual after
         calibration, the same quantity as the computed library's 12.2 cm-1 RMSE.
  m      band-set mismatch. Bands are visited in random order and removed whenever the
         removed share of band intensity stays at or below m (bands the calculation
         misses; the realised share is reported), then spurious bands carrying a share
         0.59*m of the final intensity are added at uniform positions in 250-1800 cm-1
         with heights drawn from the agent's own bands (bands the calculation invents).
  tau    intensity error, multiplicative log-normal with s.d. tau.

The computed library's own error is measured with the same pairing (one-to-one within
35 cm-1, measured agent means, HN2 excluded): sigma 0.553, a measured-only intensity share
of 0.281, a computed-only share of 0.167 (hence the ratio 0.59) and tau 0.579. The model is
validated by asking what it predicts at that operating point against the observed 13 of 30.

Writes data/v3_error_injection.csv (grid), data/v3_error_injection_tau.csv (slices),
data/v3_error_injection_meta.json and latex/sitables/errinject.tex.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_step3_headtohead import (load_measured, load_library, synth_native, cos,
                                 AGENTS, CLASS, JOB, GAMMA)

SIGMAS = np.round(np.arange(0.0, 1.001, 0.05), 3)
MS = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
EXTRA_RATIO = 0.59
R = int(os.environ.get("RF_EI_DRAWS", "100"))
R_OP = int(os.environ.get("RF_EI_DRAWS_OP", "500"))
SEED = 20261112
LO, HI = 250.0, 1800.0


def line_list(X, spec):
    pk, _ = find_peaks(spec, prominence=0.05, distance=2)
    return X[pk].astype(float), spec[pk].astype(float)


def render(X, pos, h, gamma=GAMMA):
    g = gamma / 2.0
    I = (h[:, None] * g * g / ((X[None, :] - pos[:, None]) ** 2 + g * g)).sum(0)
    return I / (I.max() + 1e-12)


def degrade(pos, h, sigma, m, tau, rng):
    pool = h.copy()
    pos = pos + rng.normal(0.0, sigma * GAMMA, size=pos.size) if sigma > 0 else pos.copy()
    h = h * np.exp(rng.normal(0.0, tau, size=h.size)) if tau > 0 else h.copy()
    gone_share = 0.0
    if m > 0 and pos.size > 1:
        total, gone, keep = h.sum(), 0.0, np.ones(pos.size, bool)
        for i in rng.permutation(pos.size):
            if gone + h[i] <= m * total and keep.sum() > 1:
                keep[i] = False; gone += h[i]
        gone_share = gone / total
        pos, h = pos[keep], h[keep]
        share = EXTRA_RATIO * m
        target = share / (1.0 - share) * h.sum()
        add_p, add_h = [], []
        while sum(add_h) < target:
            add_p.append(rng.uniform(LO, HI)); add_h.append(float(rng.choice(pool)))
        if add_h:
            pos = np.concatenate([pos, add_p])
            h = np.concatenate([h, np.array(add_h) * target / sum(add_h)])
    return pos, h, gone_share


def dft_error_stats(X, acq):
    """The computed library's own error, measured with the pairing used throughout."""
    lib = load_library()
    M = json.load(open(os.path.join(ROOT, "data", "v3_headtohead_meta.json")))
    res, lr, mm, me = [], [], [], []
    for a in AGENTS:
        if a not in JOB or a == "HN2":
            continue
        pm, hm = line_list(X, np.mean([s for _, s in acq[a]], 0))
        pc, hc = line_list(X, synth_native(lib[JOB[a]], X, M["s_low"], M["s_high"]))
        C = np.abs(pm[:, None] - pc[None, :]); C = np.where(C <= 35, C, 1e6)
        r, c = linear_sum_assignment(C); ok = C[r, c] < 1e6; r, c = r[ok], c[ok]
        res += list(pc[c] - pm[r]); lr += list(np.log(hc[c] / hm[r]))
        mm.append(1 - hm[r].sum() / hm.sum()); me.append(1 - hc[c].sum() / hc.sum())
    return dict(sigma=float(np.sqrt(np.mean(np.square(res))) / GAMMA),
                rmse=float(np.sqrt(np.mean(np.square(res)))), n_pairs=len(res),
                m_miss=float(np.mean(mm)), m_extra=float(np.mean(me)),
                tau=float(np.std(lr)))


class Runner:
    def __init__(self, X, acq):
        self.X = X
        self.full = {a: line_list(X, np.mean([s for _, s in acq[a]], 0)) for a in AGENTS}
        self.queries = [(a, r, s) for a in AGENTS if len(acq[a]) >= 2 for r, s in acq[a]]
        self.loo = {(a, r): line_list(X, np.mean([s for rr, s in acq[a] if rr != r], 0))
                    for a, r, _ in self.queries}

    def cell(self, sigma, m, tau, draws, rng):
        mol, cls, mol27, real = [], [], [], []
        n = draws if (sigma > 0 or m > 0 or tau > 0) else 1
        for _ in range(n):
            lib_full = {}
            for a in AGENTS:
                p, h, g = degrade(*self.full[a], sigma, m, tau, rng); real.append(g)
                lib_full[a] = render(self.X, p, h)
            ok = ok_c = ok27 = 0
            for a, r, q in self.queries:
                lib = dict(lib_full)
                p, h, g = degrade(*self.loo[(a, r)], sigma, m, tau, rng); real.append(g)
                lib[a] = render(self.X, p, h)
                sims = {k: cos(q, v) for k, v in lib.items()}
                top = max(sims, key=sims.get)
                ok += top == a; ok_c += CLASS[top] == CLASS[a]; ok27 += (top == a) and a != "HN2"
            mol.append(ok); cls.append(ok_c); mol27.append(ok27)
        return dict(sigma=sigma, m=m, tau=tau, draws=n, m_realised=float(np.mean(real)),
                    mol_mean=float(np.mean(mol)), mol_p05=float(np.percentile(mol, 5)),
                    mol_p95=float(np.percentile(mol, 95)), cls_mean=float(np.mean(cls)),
                    mol27_mean=float(np.mean(mol27)))


def main():
    X, acq = load_measured()
    E = dft_error_stats(X, acq)
    print("computed library error:", {k: round(v, 3) for k, v in E.items()})
    run = Runner(X, acq)
    rng = np.random.default_rng(SEED)

    grid = [run.cell(s, m, 0.0, R, rng) for s in SIGMAS for m in MS]
    D = pd.DataFrame(grid)
    D.to_csv(os.path.join(ROOT, "data", "v3_error_injection.csv"), index=False)

    # nominal m whose realised share matches the computed library's measured-only share
    real = D[D.sigma == 0].set_index("m").m_realised
    m_op = float(np.interp(E["m_miss"], real.values, real.index.values))
    slices = [run.cell(s, m, t, R, rng) for t in (0.3, E["tau"]) for m in (0.0, m_op)
              for s in SIGMAS[::2]]
    T = pd.DataFrame(slices)
    T.to_csv(os.path.join(ROOT, "data", "v3_error_injection_tau.csv"), index=False)

    op = run.cell(E["sigma"], m_op, E["tau"], R_OP, rng)
    op_pos = run.cell(E["sigma"], 0.0, 0.0, R_OP, rng)
    print("operating point (all three errors):", op)
    print("operating point (position only):", op_pos)

    def max_sigma(df, m, thr):
        s = df[(np.isclose(df.m, m)) & (df.mol_mean >= thr)].sigma
        return float(s.max()) if len(s) else None
    meta = dict(R=R, R_op=R_OP, seed=SEED, extra_ratio=EXTRA_RATIO, gamma=GAMMA, dft=E,
                m_op_nominal=m_op, ceiling=float(D[(D.sigma == 0) & (D.m == 0)].mol_mean.iloc[0]),
                sigma_for_27={str(m): max_sigma(D, m, 27) for m in MS},
                sigma_for_24={str(m): max_sigma(D, m, 24) for m in MS},
                m_tolerance_at_sigma0_for_27=float(D[(D.sigma == 0) & (D.mol_mean >= 27)].m_realised.max()),
                operating_point=op, operating_point_position_only=op_pos, observed_mol=13)
    json.dump(meta, open(os.path.join(ROOT, "data", "v3_error_injection_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("dft",)}, indent=1)[:2500])


if __name__ == "__main__":
    main()
