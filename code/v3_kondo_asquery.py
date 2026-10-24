"""
Cross-provenance control: Kondo's published band lists used as QUERIES.

Motivation. The head-to-head result of Section 3.1 compares a measured library against a
computed one using queries that are acquisitions of the same sealed vials, in the same
session, on the same instrument. The measured arm therefore matches a spectrum against its
own siblings, and its score is an upper bound rather than a transferable figure. The only
independent measured reference for authentic agents at 1064 nm is Kondo et al. 2018, whose
Table 1 overlaps seven of the agents here. Querying both libraries with Kondo's band lists
tests the two arms on material of a different provenance.

Confound. Kondo report assigned bands only (three to seven per agent) and no intensities,
so a Kondo query is also far sparser than an acquisition, which carries the full profile.
Sparsity alone would cost accuracy in either arm. The control separates the two effects:
each agent's own measured spectrum is sparsified to the same band count and queried against
the measured library with the query acquisition removed, exactly as in Section 3.1. The
difference between the external and the self-sparsified result is the cost of provenance;
the difference between the self-sparsified result and the 30/30 of Section 3.1 is the cost
of sparsity.

Two scores are reported because Kondo give no intensities:
  (A) band-overlap: the fraction of query bands having a library band within tol;
  (B) comb cosine: unit-amplitude Lorentzians at the measured Gamma, cosine against the
      library spectrum. (B) must assume equal intensities, which (A) does not.

Writes data/v3_kondo_asquery.csv and data/v3_kondo_asquery_summary.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H
from v3_kondo_compare import KONDO

TOLS = (15.0, 20.0, 25.0)


def comb(nus, grid, gamma=H.GAMMA):
    g = gamma / 2.0
    I = np.zeros_like(grid, dtype=float)
    for f in nus:
        I += g * g / ((grid - f) ** 2 + g * g)
    return I / (I.max() + 1e-12)


def bands_of(X, spec, prominence=0.05):
    """The detector of Section 2.2, with prominence returned so bands can be ranked."""
    idx, props = find_peaks(spec, prominence=prominence, distance=2)
    return X[idx], props["prominences"]


def overlap(qnu, libnu, tol):
    if len(qnu) == 0 or len(libnu) == 0:
        return 0.0
    return float(np.mean([np.min(np.abs(libnu - q)) <= tol for q in qnu]))


def top1(scores, gold):
    o = sorted(scores.items(), key=lambda t: -t[1])
    return o[0][0], (o[0][0] == gold), o[0][1] - (o[1][1] if len(o) > 1 else 0.0)


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    calib = [a for a in H.AGENTS if a in H.JOB]
    _, SL, SH, _ = H.fit_scale(lib, mpk, calib)
    dft = {k: H.synth_native(v, X, SL, SH) for k, v in lib.items()}
    core = [H.JOB[a] for a in H.AGENTS if a in H.JOB]

    mb = {a: bands_of(X, mean[a]) for a in H.AGENTS}            # measured library bands
    db = {k: bands_of(X, v) for k, v in dft.items()}             # computed library bands

    agents = [a for a in KONDO if a in H.AGENTS]
    rows = []
    for ag in agents:
        q = np.array([n for n, _ in KONDO[ag] if X[0] <= n <= X[-1]], float)
        n_b = len(q)
        qc = comb(q, X)
        multi = len(acq[ag]) >= 2

        # --- self-sparsified control query: the n_b most prominent bands of the mean ---
        w, pr = bands_of(X, mean[ag])
        keep = w[np.argsort(-pr)[:n_b]]
        sc = comb(np.sort(keep), X)

        # measured library with this agent's single reference held out where possible
        mlib_hold = {}
        for a in H.AGENTS:
            if a == ag and multi:
                others = [s for r, s in acq[a]][1:]            # drop one acquisition
                mlib_hold[a] = np.mean(others, 0)
            else:
                mlib_hold[a] = mean[a]
        mbh = {a: bands_of(X, v) for a, v in mlib_hold.items()}

        for tol in TOLS:
            sA_m = {a: overlap(q, mb[a][0], tol) for a in H.AGENTS}
            sA_dO = {k: overlap(q, db[k][0], tol) for k in dft}
            sA_dC = {k: sA_dO[k] for k in core}
            sA_ct = {a: overlap(np.sort(keep), mbh[a][0], tol) for a in H.AGENTS}
            gold_d = H.JOB.get(ag, "__none__")
            t_m, ok_m, mg_m = top1(sA_m, ag)
            t_dO, ok_dO, _ = top1(sA_dO, gold_d)
            t_dC, ok_dC, _ = top1(sA_dC, gold_d)
            t_ct, ok_ct, _ = top1(sA_ct, ag)
            rows.append(dict(agent=ag, n_bands=n_b, tol=tol, score="overlap",
                             multi_acq=multi,
                             meas_top1=t_m, meas_ok=ok_m, meas_margin=round(mg_m, 3),
                             dft_open_top1=t_dO, dft_open_ok=ok_dO,
                             dft_closed_top1=t_dC, dft_closed_ok=ok_dC,
                             ctrl_top1=t_ct, ctrl_ok=ok_ct))

        # cosine of the comb, tolerance-free
        sB_m = {a: H.cos(qc, mean[a]) for a in H.AGENTS}
        sB_dO = {k: H.cos(qc, v) for k, v in dft.items()}
        sB_dC = {k: sB_dO[k] for k in core}
        sB_ct = {a: H.cos(sc, v) for a, v in mlib_hold.items()}
        gold_d = H.JOB.get(ag, "__none__")
        t_m, ok_m, mg_m = top1(sB_m, ag)
        t_dO, ok_dO, _ = top1(sB_dO, gold_d)
        t_dC, ok_dC, _ = top1(sB_dC, gold_d)
        t_ct, ok_ct, _ = top1(sB_ct, ag)
        rows.append(dict(agent=ag, n_bands=n_b, tol=np.nan, score="comb cosine",
                         multi_acq=multi,
                         meas_top1=t_m, meas_ok=ok_m, meas_margin=round(mg_m, 3),
                         dft_open_top1=t_dO, dft_open_ok=ok_dO,
                         dft_closed_top1=t_dC, dft_closed_ok=ok_dC,
                         ctrl_top1=t_ct, ctrl_ok=ok_ct))

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_kondo_asquery.csv"), index=False)

    S = []
    for (score, tol), g in D.groupby(["score", "tol"], dropna=False):
        gm = g[g.multi_acq]
        S.append(dict(score=score, tol=tol, n=len(g),
                      meas=int(g.meas_ok.sum()),
                      dft_open=int(g.dft_open_ok.sum()),
                      dft_closed=int(g.dft_closed_ok.sum()),
                      ctrl=int(g.ctrl_ok.sum()),
                      n_multi=len(gm),
                      meas_multi=int(gm.meas_ok.sum()),
                      dft_open_multi=int(gm.dft_open_ok.sum()),
                      ctrl_multi=int(gm.ctrl_ok.sum())))
    SS = pd.DataFrame(S)
    SS.to_csv(os.path.join(ROOT, "data", "v3_kondo_asquery_summary.csv"), index=False)

    print(f"queries: {len(set(D.agent))} agents, {dict(zip(D.agent, D.n_bands))}")
    print("\nHD and VX have a single acquisition, so for those two the control query and the")
    print("library entry come from the same spectrum and the control is circular. The five")
    print("multi-acquisition agents on the right are the non-circular comparison.")
    print(f"\n{'':<13}{'':>5}{'  ---- all seven ----':<34}{'  -- five multi-acq --':<24}")
    print(f"{'score':<13}{'tol':>5}{'meas':>7}{'DFT54':>8}{'DFT11':>8}{'ctrl':>7}"
          f"{'meas':>8}{'DFT54':>8}{'ctrl':>7}")
    print("-" * 78)
    for r in SS.itertuples():
        t = "   --" if pd.isna(r.tol) else f"{r.tol:5.0f}"
        print(f"{r.score:<13}{t}{r.meas:>5}/{r.n}{r.dft_open:>6}/{r.n}{r.dft_closed:>6}/{r.n}"
              f"{r.ctrl:>5}/{r.n}{r.meas_multi:>6}/{r.n_multi}{r.dft_open_multi:>6}/{r.n_multi}"
              f"{r.ctrl_multi:>5}/{r.n_multi}")
    print("\nsaved data/v3_kondo_asquery.csv + data/v3_kondo_asquery_summary.csv")


if __name__ == "__main__":
    main()
