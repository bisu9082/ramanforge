"""
Statistics for the external comparison of Section S11.

Reporting "22 of 31 bands reproduced" is not interpretable without a chance level, because
a dense band list matches an arbitrary wavenumber often enough. Two baselines are computed:

  positional   the fraction of the 250-1800 cm-1 axis that lies within the tolerance of one
               of the agent's own detected bands. A wavenumber drawn uniformly at random
               matches at this rate, so it is the chance level for the agent's own spectrum.
  cross-agent  each agent's published band list matched against the ten other measured
               spectra in this set. This is the harder and more relevant baseline: it asks
               whether a Kondo list matches the right agent more often than a wrong one.

Also reported: the tolerance sensitivity of the count and the residual RMS, a bootstrap
interval for the additive component of the residual, and a Fisher exact test of the HN2
vial against the rest of the comparison.

Writes data/v3_kondo_stats.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H
from v3_kondo_compare import KONDO, measured_bands

TOLS = (5.0, 10.0, 15.0, 18.0, 25.0)
LO, HI = 250.0, 1800.0
RNG = np.random.default_rng(20261024)


def covered_fraction(bands, tol):
    """Fraction of [LO, HI] within tol of a band, intervals merged."""
    iv = sorted((max(LO, b - tol), min(HI, b + tol)) for b in bands if LO - tol < b < HI + tol)
    tot, cur_a, cur_b = 0.0, None, None
    for a, b in iv:
        if b <= a:
            continue
        if cur_a is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            tot += cur_b - cur_a; cur_a, cur_b = a, b
    if cur_a is not None:
        tot += cur_b - cur_a
    return tot / (HI - LO)


def main():
    X, acq = H.load_measured()
    mean = {a: (lambda m: m / m.max())(np.mean([s for _, s in acq[a]], 0)) for a in H.AGENTS}
    bands = {a: measured_bands(X, mean[a])[0] for a in H.AGENTS}
    agents = [a for a in KONDO if a in H.AGENTS]

    rows = []
    for tol in TOLS:
        hit = miss = 0
        res = []
        chance_pos, chance_cross, n_cross_hit, n_cross = [], [], 0, 0
        for ag in agents:
            q = [n for n, _ in KONDO[ag] if X[0] <= n <= X[-1]]
            w = bands[ag]
            for nu in q:
                d = w[np.argmin(np.abs(w - nu))] - nu
                if abs(d) <= tol:
                    hit += 1; res.append(d)
                else:
                    miss += 1
            chance_pos += [covered_fraction(w, tol)] * len(q)
            for other in H.AGENTS:
                if other == ag:
                    continue
                wo = bands[other]
                for nu in q:
                    n_cross += 1
                    n_cross_hit += int(np.min(np.abs(wo - nu)) <= tol)
            chance_cross.append(np.nan)
        res = np.asarray(res, float)
        boot = [float(np.mean(RNG.choice(res, res.size, replace=True))) for _ in range(10000)]
        rows.append(dict(
            tol=tol, matched=hit, n=hit + miss, rate=hit / (hit + miss),
            rms=float(np.sqrt((res ** 2).mean())), mean=float(res.mean()),
            mean_lo=float(np.percentile(boot, 2.5)), mean_hi=float(np.percentile(boot, 97.5)),
            chance_positional=float(np.mean(chance_pos)),
            chance_crossagent=n_cross_hit / n_cross))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_kondo_stats.csv"), index=False)

    print(f"{'tol':>5}{'matched':>9}{'rate':>8}{'chance(pos)':>13}{'chance(cross)':>15}"
          f"{'RMS':>7}{'mean':>8}{'  95% CI on mean':<20}")
    for r in D.itertuples():
        print(f"{r.tol:5.0f}{r.matched:>6}/{r.n}{r.rate:>8.2f}{r.chance_positional:>13.2f}"
              f"{r.chance_crossagent:>15.2f}{r.rms:>7.2f}{r.mean:>8.2f}"
              f"   [{r.mean_lo:+.2f}, {r.mean_hi:+.2f}]")

    # HN2 against the rest, at the adopted tolerance
    tol = 18.0
    tab = [[0, 0], [0, 0]]
    for ag in agents:
        q = [n for n, _ in KONDO[ag] if X[0] <= n <= X[-1]]
        w = bands[ag]
        h = sum(int(np.min(np.abs(w - nu)) <= tol) for nu in q)
        i = 0 if ag == "HN2" else 1
        tab[i][0] += h; tab[i][1] += len(q) - h
    odds, p = fisher_exact(tab, alternative="less")
    print(f"\nHN2 {tab[0][0]}/{tab[0][0]+tab[0][1]} reproduced vs "
          f"{tab[1][0]}/{tab[1][0]+tab[1][1]} for the other six: Fisher exact one-sided "
          f"p = {p:.4f} (two-sided {fisher_exact(tab)[1]:.4f})")
    print("\nsaved data/v3_kondo_stats.csv")


if __name__ == "__main__":
    main()
