"""
T1-18: metric comparison. Only the 'metric' is changed, with the same query, the same library and the same preprocessing.

Reviewer comment (R2-M11, R3-M10): on the Γ=22 grid the full-profile cosine lies in the
regime of position residuals around 0.6Γ, where it is maximally sensitive to position error.
The reported deficit therefore cannot be separated into 'inaccuracy of the computation'
versus 'unsuitability of the metric'. The authors' own value of 0.833 for the band-match
fraction under the in-library condition suggests the latter.

Fairness: both the measured and the computed side are passed through the **same peak detector** on the same native grid.
"""
import os, sys, json, collections
import numpy as np, pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, measured_peaks, fit_scale,
                                 synth_native, cos, identify, AGENTS, CLASS, JOB, CUT, GAMMA)

DATA = os.path.join(ROOT, "data")

# ------------------------------- definitions of the metrics
def peaklist(spec, X, prom=0.05):
    pk, _ = find_peaks(spec, prominence=prom, distance=2)
    return X[pk], spec[pk]

def hqi(qp, qa, lp, la, tol):
    """Symmetric peak list hit quality. One-to-one assignment + position penalty + intensity weighting.
       S = Σ min(a,b)·(1−|Δ|/tol) / max(Σa, Σb)   ∈ [0,1]
       Both unmatched query peaks and surplus library peaks are penalized."""
    if not len(qp) or not len(lp): return 0.0
    C = np.abs(qp[:, None] - lp[None, :])
    C = np.where(C > tol, 1e6, C)
    ri, ci = linear_sum_assignment(C)
    num = 0.0
    for i, j in zip(ri, ci):
        if C[i, j] < 1e6:
            num += min(qa[i], la[j]) * (1.0 - C[i, j] / tol)
    return float(num / max(qa.sum(), la.sum()))

def dice(qp, lp, tol):
    """Pure band position agreement with intensity excluded (Dice). Shows the position-only contribution."""
    if not len(qp) or not len(lp): return 0.0
    C = np.abs(qp[:, None] - lp[None, :])
    C = np.where(C > tol, 1e6, C)
    ri, ci = linear_sum_assignment(C)
    m = sum(1 for i, j in zip(ri, ci) if C[i, j] < 1e6)
    return float(2 * m / (len(qp) + len(lp)))

def dcos(u, v, w=7):
    """First-derivative cosine: removes gentle differences in background and envelope and compares band structure alone."""
    du = savgol_filter(u, w, 2, deriv=1); dv = savgol_filter(v, w, 2, deriv=1)
    return cos(du, dv)

def shiftcos(u, v, maxpx=3):
    """Cosine allowing ±maxpx pixel alignment (≈±17 cm-1). Absorbs systematic position error."""
    best = -1.0
    for k in range(-maxpx, maxpx + 1):
        vv = np.roll(v, k)
        if k > 0: vv[:k] = v[0]
        elif k < 0: vv[k:] = v[-1]
        best = max(best, cos(u, vv))
    return best

# ─────────────────────────────── main
def main():
    X, acq = load_measured(); lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]
    _, SL, SH, _ = fit_scale(lib, mpk, calib)
    print(f"[calibration] s=({SL:.3f}, {SH:.3f})  Γ_base={GAMMA:.0f}\n")

    # measured linewidth per agent (Table S7); the median where none is available
    LW = pd.read_csv(f"{DATA}/v3_linewidths.csv")
    gcol = [c for c in LW.columns if "gamma" in c.lower() or c.lower() in ("g", "width", "fwhm")][0]
    gmed = float(LW[gcol].median())
    gper = LW.groupby("agent")[gcol].mean().to_dict()
    print(f"measured Γ per agent: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(gper.items())))
    print(f"median {gmed:.1f}\n")

    JOB2AG = {j: a for a, j in JOB.items()}
    def gamma_for(job):
        a = JOB2AG.get(job)
        return float(gper.get(a, gmed)) if a else gmed

    # computed library: synthesised under each Gamma scenario
    scen = {"G22": {k: synth_native(v, X, SL, SH, 22.0) for k, v in lib.items()},
            "G30": {k: synth_native(v, X, SL, SH, 30.0) for k, v in lib.items()},
            "G40": {k: synth_native(v, X, SL, SH, 40.0) for k, v in lib.items()},
            "G50": {k: synth_native(v, X, SL, SH, 50.0) for k, v in lib.items()},
            "Gper": {k: synth_native(v, X, SL, SH, gamma_for(k)) for k, v in lib.items()}}

    core = [JOB[a] for a in AGENTS if a in JOB]
    queryable = [a for a in AGENTS if len(acq[a]) >= 2]
    queries = [(a, r, s) for a in queryable for r, s in acq[a]]
    print(f"queries {len(queries)} / {len(queryable)} agents\n")

    # precomputed peak lists for the library
    PLK = {sk: {k: peaklist(v, X) for k, v in sp.items()} for sk, sp in scen.items()}

    METRICS = []   # (label, kind, param)
    for sk in ["G22", "G30", "G40", "G50", "Gper"]:
        METRICS.append((f"cosine, Γ={sk[1:] if sk!='Gper' else 'per-agent'}", "cos", sk))
    METRICS += [("first-derivative cosine, Γ=22", "dcos", "G22"),
                ("shift-tolerant cosine ±3px, Γ=22", "shift", "G22"),
                ("peak-list HQI, tol=10", "hqi10", "G22"),
                ("peak-list HQI, tol=20", "hqi20", "G22"),
                ("peak-list HQI, tol=35", "hqi35", "G22"),
                ("band-position Dice, tol=20", "dice20", "G22"),
                ("band-position Dice, tol=35", "dice35", "G22")]

    out = []
    for label, kind, sk in METRICS:
        dft = scen[sk]; plk = PLK[sk]
        nmol = ncls = nmol53 = ncls53 = 0
        selfs, margs = [], []
        per = collections.defaultdict(lambda: [0, 0])
        for q_ag, q_rep, q in queries:
            gold = JOB.get(q_ag, "__none__")
            if kind == "cos":
                sc = {k: cos(q, dft[k]) for k in dft}
            elif kind == "dcos":
                sc = {k: dcos(q, dft[k]) for k in dft}
            elif kind == "shift":
                sc = {k: shiftcos(q, dft[k]) for k in dft}
            elif kind.startswith("hqi"):
                tol = float(kind[3:]); qp, qa = peaklist(q, X)
                sc = {k: hqi(qp, qa, plk[k][0], plk[k][1], tol) for k in dft}
            else:
                tol = float(kind[4:]); qp, _ = peaklist(q, X)
                sc = {k: dice(qp, plk[k][0], tol) for k in dft}

            for tag, pool in (("closed", core), ("open", list(dft.keys()))):
                s2 = {k: sc[k] for k in pool}
                o = sorted(s2.items(), key=lambda t: -t[1])
                top = o[0][0]
                ok = (top == gold)
                clsok = (CLASS.get(JOB2AG.get(top, ""), "Nerve") == CLASS[q_ag])
                if tag == "closed":
                    nmol += ok; ncls += clsok
                    selfs.append(s2.get(gold, np.nan))
                    margs.append(o[0][1] - (o[1][1] if len(o) > 1 else 0))
                    per[q_ag][0] += ok; per[q_ag][1] += 1
                else:
                    nmol53 += ok; ncls53 += clsok
        N = len(queries)
        agent_ok = sum(1 for a in per if per[a][0] * 2 > per[a][1])
        out.append(dict(metric=label, kind=kind, gamma=sk,
                        mol_closed=nmol, mol_open=nmol53, cls_closed=ncls, cls_open=ncls53,
                        n=N, agent_closed=agent_ok, n_agents=len(per),
                        self_mean=float(np.nanmean(selfs)), margin_mean=float(np.nanmean(margs))))
        print(f"{label:<34} molecule {nmol:2d}/{N}  (53 agents {nmol53:2d}/{N})  "
              f"class {ncls:2d}/{N}  per agent {agent_ok}/{len(per)}  "
              f"self {np.nanmean(selfs):.3f}  margin {np.nanmean(margs):.3f}")

    D = pd.DataFrame(out)
    D.to_csv(f"{DATA}/v3_t118_metrics.csv", index=False)
    print(f"\nsaved data/v3_t118_metrics.csv")
    base = D[D.metric.str.startswith("cosine, Γ=22")].iloc[0]
    best = D.loc[D.mol_closed.idxmax()]
    print(f"\nbaseline (cosine Γ=22): molecule {base.mol_closed}/{base.n}, per agent {base.agent_closed}/{base.n_agents}")
    print(f"best     ({best.metric}): molecule {best.mol_closed}/{best.n}, per agent {best.agent_closed}/{best.n_agents}")

if __name__ == "__main__":
    main()
