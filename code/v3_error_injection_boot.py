"""Agent-level bootstrap intervals on the error-injection thresholds.

The thirty queries come from nine vials, so the unit of resampling is the vial (agent).
For each condition along four one-dimensional paths through the error space, the expected
number of correct molecular identifications is recorded per agent (mean over draws):
  sigma path : positional error sigma/Gamma, exact band set and intensities
  m path     : band-set mismatch (nominal m; realised share reported), exact positions
  tau path   : intensity error tau, exact positions and band set
  sigma|tau  : positional error with tau = 0.30
Agents are then resampled with replacement and the 90 % crossing (27 of 30 on the full set)
is located by linear interpolation on each replicate. Point estimates are the crossings on
the full set. Writes data/v3_error_injection_boot.csv (per-agent curves) and
data/v3_error_injection_boot_meta.json (crossings and 95 % intervals).
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_error_injection import Runner, load_measured, degrade, render, GAMMA
from v3_step3_headtohead import cos

R = int(os.environ.get("RF_EI_DRAWS", "200"))
B = int(os.environ.get("RF_EI_BOOT", "4000"))
SEED = 20261114


def per_agent(run, sigma, m, tau, draws, rng):
    agents = sorted({a for a, _, _ in run.queries})
    hit = {a: 0.0 for a in agents}; nq = {a: 0 for a in agents}; real = []
    for a, _, _ in run.queries:
        nq[a] += 1
    n = draws if (sigma > 0 or m > 0 or tau > 0) else 1
    for _ in range(n):
        lib_full = {}
        for a in run.full:
            p, h, g = degrade(*run.full[a], sigma, m, tau, rng); real.append(g)
            lib_full[a] = render(run.X, p, h)
        for a, r, q in run.queries:
            lib = dict(lib_full)
            p, h, g = degrade(*run.loo[(a, r)], sigma, m, tau, rng); real.append(g)
            lib[a] = render(run.X, p, h)
            sims = {k: cos(q, v) for k, v in lib.items()}
            hit[a] += (max(sims, key=sims.get) == a)
    return {a: hit[a] / n for a in agents}, nq, float(np.mean(real))


def crossing(x, frac, thr=0.9):
    x, frac = np.asarray(x, float), np.asarray(frac, float)
    for i in range(1, len(x)):
        if frac[i - 1] >= thr > frac[i]:
            return x[i - 1] + (frac[i - 1] - thr) / (frac[i - 1] - frac[i]) * (x[i] - x[i - 1])
    return np.nan if frac[0] < thr else np.inf   # never falls below within the grid


def main():
    X, acq = load_measured()
    run = Runner(X, acq)
    rng = np.random.default_rng(SEED)
    paths = {
        "sigma": [(s, 0.0, 0.0) for s in np.round(np.arange(0, 0.801, 0.05), 3)],
        "m": [(0.0, m, 0.0) for m in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)],
        "tau": [(0.0, 0.0, t) for t in (0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0)],
        "sigma_tau0.3": [(s, 0.0, 0.3) for s in np.round(np.arange(0, 0.701, 0.05), 3)],
    }
    axis = {"sigma": 0, "m": 1, "tau": 2, "sigma_tau0.3": 0}
    rows = []
    for name, cells in paths.items():
        for (s, m, t) in cells:
            pa, nq, mr = per_agent(run, s, m, t, R, rng)
            for a in pa:
                rows.append(dict(path=name, sigma=s, m=m, tau=t, m_realised=mr, agent=a,
                                 correct=pa[a], n_queries=nq[a]))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_error_injection_boot.csv"), index=False)
    agents = sorted(D.agent.unique())
    brng = np.random.default_rng(SEED + 1)
    idx = brng.integers(0, len(agents), size=(B, len(agents)))
    meta = dict(R=R, B=B, seed=SEED, agents=agents, unit="vial (agent)", threshold="90% of queries")
    for name, cells in paths.items():
        d = D[D.path == name]
        k = axis[name]
        xs = [c[k] for c in cells]
        if name == "m":   # report on the realised share axis
            xs = [float(d[np.isclose(d.m, c[1])].m_realised.iloc[0]) for c in cells]
        C = np.array([[d[(d.agent == a) & np.isclose(d[["sigma", "m", "tau"][k]], c[k])].correct.iloc[0]
                       for c in cells] for a in agents])          # agents x cells
        N = np.array([d[d.agent == a].n_queries.iloc[0] for a in agents])
        point = crossing(xs, C.sum(0) / N.sum())
        boots = np.array([crossing(xs, C[i].sum(0) / N[i].sum()) for i in idx])
        ok = boots[~np.isnan(boots)]
        n_beyond = int(np.isinf(ok).sum())
        meta[name] = dict(axis=xs, total=(C.sum(0) / N.sum() * 30).round(2).tolist(),
                          crossing=float(point), ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))],
                          ci90=[float(np.percentile(ok, 5)), float(np.percentile(ok, 95))],
                          n_valid=int(ok.size), n_beyond_grid=n_beyond)
        print(name, "crossing %.3f  95%% CI %.3f-%.3f" % (point, *meta[name]["ci95"]),
              "| per-30:", meta[name]["total"])
    json.dump(meta, open(os.path.join(ROOT, "data", "v3_error_injection_boot_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
