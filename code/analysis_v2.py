"""
ramanforge: v2 master analysis (non-circular calibration).

Changes relative to v1
  1) s is fitted on 'band positions' only (identification accuracy not used)  -> circularity removed
  2) two-region scaling for low and high wavenumbers (mode-dependent, recommended in the literature)
  3) Γ is fixed from the apparent width of the measured bands -> 0 free parameters
  4) band matching changed to one-to-one (Hungarian) -> duplicate assignments removed
  5) leave-one-molecule-out-of-library added (the scenario assumed in the paper)

Output: data/_v2.pkl  (the single source for every figure, table and in-text number)
"""
from __future__ import annotations
import glob, os, sys, pickle, itertools, random
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.spectra import parse_raman
from src.calibrate import scattering_factor, _cos

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRID = np.arange(250.0, 1800.0 + 1e-9, 2.0)
CUT, LASER, TEMP = 800.0, 1064.0, 298.0
FWHM = 30.0
MATCH_TOL = 35.0
PEAK_H, PEAK_D = 0.12, 6

AGENTS = ["GD", "GF", "HD", "HN1", "HN3", "A-230", "A-232", "A-234", "A-242"]
JOB = {"GD": "EXP_GD_soman", "GF": "GF_cyclosarin", "HD": "EXP_HD", "HN1": "EXP_HN1",
       "HN3": "EXP_HN3", "A-230": "EXP_A230", "A-232": "EXP_A232",
       "A-234": "EXP_A234", "A-242": "EXP_A242"}
KEY = {"GD": "gd", "GF": "gf", "HD": "hd", "HN1": "hn1", "HN3": "hn3",
       "A-230": "a230", "A-232": "a232", "A-234": "a234", "A-242": "a242"}
CLASS = {"EXP_GD_soman": "Nerve", "GF_cyclosarin": "Nerve", "EXP_HD": "Blister",
         "EXP_HN1": "Blister", "EXP_HN2": "Blister", "EXP_HN3": "Blister",
         "EXP_A230": "Novichok", "EXP_A232": "Novichok", "EXP_A234": "Novichok",
         "EXP_A242": "Novichok"}
PRETTY = {v: k for k, v in JOB.items()}
PRETTY["EXP_HN2"] = "HN2"


def cls(job): return CLASS.get(job, "Nerve")
def label(job): return PRETTY.get(job, job.replace("_", " "))


# ---------------------------------------------------------------- load
def load_library():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "orca", "outputs", "*.out"))):
        n = os.path.basename(f)[:-4]
        if n == "GB_sarin_fast":
            continue
        out[n] = parse_raman(f)
    return out


def load_measured():
    acc = {}
    for f in glob.glob(os.path.join(ROOT, "data", "exp_raman", "augmented_csv", "*.csv")):
        k = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f)
        acc.setdefault(k, []).append(np.interp(GRID, d.iloc[:, 0].values, d.iloc[:, 1].values))
    return {k: (lambda m: m / m.max())(np.mean(v, 0)) for k, v in acc.items()}


# ----------------------------------------------------------- calibration
def scale_freqs(f, sl, sh): return np.where(f < CUT, f * sl, f * sh)


def synth(pk, sl, sh, fwhm=FWHM, scattering=True):
    f = scale_freqs(pk[:, 0], sl, sh)
    a = pk[:, 1].astype(float).copy()
    if scattering:
        a = a * scattering_factor(f, LASER, TEMP)
    if a.max() > 0:
        a = a / a.max()
    g = fwhm / 2.0
    I = np.zeros_like(GRID)
    for ff, aa in zip(f, a):
        I += aa * g * g / ((GRID - ff) ** 2 + g * g)
    return I / (I.max() + 1e-12)


def band_rmse(lib, mpk, sl, sh, tol=MATCH_TOL):
    """Objective function that does not use identification: measured band positions vs scaled harmonic frequencies."""
    d = []
    for ag in AGENTS:
        pk = lib[JOB[ag]]
        f = scale_freqs(pk[:, 0], sl, sh)
        keep = (f > GRID[0]) & (f < GRID[-1]) & (pk[:, 1] > 0.02 * pk[:, 1].max())
        f = f[keep]
        if not len(f):
            continue
        for w in mpk[ag]:
            j = int(np.argmin(np.abs(f - w)))
            if abs(f[j] - w) <= tol:
                d.append(f[j] - w)
    d = np.asarray(d)
    return (float(np.sqrt((d ** 2).mean())) if len(d) else np.nan), len(d)


def fit_scale(lib, mpk, min_pairs=85):
    best = None
    for sl in np.arange(0.980, 1.0401, 0.002):
        for sh in np.arange(0.965, 1.0101, 0.002):
            r, n = band_rmse(lib, mpk, float(sl), float(sh))
            if n >= min_pairs and (best is None or r < best[0]):
                best = (r, float(sl), float(sh), n)
    return best


# -------------------------------------------- band matching (one-to-one)
def match_bands(exp_spec, dft_spec, tol=MATCH_TOL):
    """Hungarian assignment: each computed band pairs with exactly one measured band."""
    pe, _ = find_peaks(exp_spec / exp_spec.max(), height=PEAK_H, distance=PEAK_D)
    pc, _ = find_peaks(dft_spec / dft_spec.max(), height=PEAK_H, distance=PEAK_D)
    we, wc = GRID[pe], GRID[pc]
    if not len(we) or not len(wc):
        return np.empty((0, 2)), len(we), len(wc)
    C = np.abs(we[:, None] - wc[None, :]).astype(float)
    BIG = 1e6
    C[C > tol] = BIG
    ri, ci = linear_sum_assignment(C)
    pairs = [(we[i], wc[j]) for i, j in zip(ri, ci) if C[i, j] < BIG]
    return np.array(pairs).reshape(-1, 2), len(we), len(wc)


# -------------------------------------------------------- identification
def identify(spec, meas, drop_self=False, pool=None):
    rows = []
    for ag in AGENTS:
        gold = JOB[ag]
        names = pool if pool is not None else list(spec)
        cand = {n: spec[n] for n in names if not (drop_self and n == gold)}
        sims = {n: _cos(meas[KEY[ag]], v) for n, v in cand.items()}
        order = sorted(sims.items(), key=lambda x: -x[1])
        top, ts = order[0]
        others = [(n, s) for n, s in order if n != gold]
        bw, bs = others[0]
        rows.append(dict(agent=ag, gold=gold, top1=top, score=ts,
                         runner_up=order[1][0] if len(order) > 1 else "",
                         margin=ts - order[1][1] if len(order) > 1 else np.nan,
                         self_cos=sims.get(gold, np.nan),
                         best_wrong=bw, best_wrong_score=bs,
                         gold_margin=sims.get(gold, np.nan) - bs,
                         mol_ok=(top == gold), cls_ok=(cls(top) == cls(gold)),
                         bw_same_cls=(cls(bw) == cls(gold))))
    return pd.DataFrame(rows)


# ------------------------------------------------ failure analysis block
REGIONS = [("skeletal\n250-600", 250, 600), ("P-F / P-N\n600-850", 600, 850),
           ("P-O-C / C-N\n850-1150", 850, 1150), ("C=N / P=O\n1150-1400", 1150, 1400),
           ("CH deform\n1400-1800", 1400, 1800)]


def region_breakdown(meas_spec, gold_spec, wrong_spec):
    out = []
    for name, lo, hi in REGIONS:
        m = (GRID >= lo) & (GRID < hi)
        w = float(meas_spec[m].sum() / meas_spec.sum())
        out.append((name, w, _cos(meas_spec[m], gold_spec[m]), _cos(meas_spec[m], wrong_spec[m])))
    return out


def main():
    lib = load_library()
    meas = load_measured()
    mpk = {ag: GRID[find_peaks(meas[KEY[ag]], height=PEAK_H, distance=PEAK_D)[0]] for ag in AGENTS}

    rmse0, SL, SH, npair0 = fit_scale(lib, mpk)
    print(f"[calib] s_low={SL:.3f} s_high={SH:.3f}  positional RMSE={rmse0:.1f} (n={npair0})")

    spec = {n: synth(v, SL, SH) for n, v in lib.items()}

    # --- band matching (one-to-one) ---
    bands, prow = [], []
    for ag in AGENTS:
        P, ne, nc = match_bands(meas[KEY[ag]], spec[JOB[ag]])
        d = P[:, 1] - P[:, 0] if len(P) else np.array([])
        for a, b in P:
            bands.append((ag, float(a), float(b)))
        prow.append(dict(agent=ag, n_meas=ne, n_dft=nc, n_match=len(P),
                         rmse=float(np.sqrt((d ** 2).mean())) if len(d) else np.nan,
                         mae=float(np.abs(d).mean()) if len(d) else np.nan,
                         bias=float(d.mean()) if len(d) else np.nan,
                         cos=_cos(meas[KEY[ag]], spec[JOB[ag]])))
    B = pd.DataFrame(prow)
    allp = np.array([[b[1], b[2]] for b in bands])
    dd = allp[:, 1] - allp[:, 0]
    print(f"[bands] one-to-one n={len(allp)}  RMSE={np.sqrt((dd**2).mean()):.1f}  "
          f"MAE={np.abs(dd).mean():.1f}  bias={dd.mean():+.1f}")
    print(B.to_string(index=False))

    # --- identification ---
    R53 = identify(spec, meas)
    R9 = identify(spec, meas, pool=[JOB[a] for a in AGENTS])
    RLOO = identify(spec, meas, drop_self=True)
    print(f"[ident] 53-lib molecular {R53.mol_ok.sum()}/9  class {R53.cls_ok.sum()}/9")
    print(f"[ident]  9-lib molecular {R9.mol_ok.sum()}/9  class {R9.cls_ok.sum()}/9")
    print(f"[LOO ]  true molecule removed -> class {RLOO.cls_ok.sum()}/9")

    # --- library growth ---
    core = [JOB[a] for a in AGENTS]
    extra = [n for n in spec if n not in core]
    rng = random.Random(0)
    curve = []
    for nlib in [9, 19, 29, 39, 53]:
        ms, cs = [], []
        for _ in range(20):
            add = rng.sample(extra, nlib - 9) if nlib > 9 else []
            r = identify(spec, meas, pool=core + add)
            ms.append(r.mol_ok.sum()); cs.append(r.cls_ok.sum())
        curve.append((nlib, np.mean(ms) / 9 * 100, np.std(ms) / 9 * 100,
                      np.mean(cs) / 9 * 100, np.std(cs) / 9 * 100))
    print("[grow]", [(c[0], round(c[1], 1), round(c[3], 1)) for c in curve])

    # --- abstention ---
    absten = []
    for th in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
        k = R53[R53.margin >= th]
        if len(k):
            absten.append((th, len(k), k.mol_ok.mean() * 100, k.cls_ok.mean() * 100))

    # --- Novichok computed-to-computed similarity ---
    NOV = ["A-230", "A-232", "A-234", "A-242"]
    DD = np.array([[_cos(spec[JOB[a]], spec[JOB[b]]) for b in NOV] for a in NOV])

    # --- region decomposition of the failure cases ---
    fails = {}
    for r in R53[~R53.mol_ok].itertuples():
        fails[r.agent] = dict(wrong=label(r.top1), wrong_job=r.top1,
                              rows=region_breakdown(meas[KEY[r.agent]], spec[r.gold], spec[r.top1]))

    # --- sensitivity to Gamma (pitfall re-check) ---
    pit = []
    for w in [8, 12, 16, 20, 25, 30, 35, 40, 50, 60]:
        sp = {n: synth(v, SL, SH, fwhm=w) for n, v in lib.items()}
        r = identify(sp, meas)
        mc = float(np.mean([_cos(meas[KEY[a]], sp[JOB[a]]) for a in AGENTS]))
        pit.append((w, mc, r.mol_ok.sum() / 9, r.cls_ok.sum() / 9, float(r.gold_margin.mean())))

    # --- contribution of the three correction terms (anchor GF) ---
    gpk = lib[JOB["GF"]]; gexp = meas[KEY["GF"]]
    stages = [("scaling only", _cos(gexp, synth(gpk, SL, SH, fwhm=2.0, scattering=False))),
              ("+ linewidth", _cos(gexp, synth(gpk, SL, SH, fwhm=FWHM, scattering=False))),
              ("+ scattering factor", _cos(gexp, synth(gpk, SL, SH, fwhm=FWHM, scattering=True)))]
    print("[stages]", [(a, round(b, 3)) for a, b in stages])

    # --- number of low-frequency bands (Novichok) ---
    nlow = {}
    for ag in NOV:
        p = mpk[ag]
        nlow[ag] = (int((p < 700).sum()), int(len(p)))

    out = dict(s_low=SL, s_high=SH, fwhm=FWHM, cut=CUT, grid=GRID,
               pos_rmse=rmse0, pos_n=npair0,
               spec={n: spec[n] for n in [JOB[a] for a in AGENTS]},
               meas={a: meas[KEY[a]] for a in AGENTS},
               bands=bands, band_table=B, band_all=allp,
               R53=R53, R9=R9, RLOO=RLOO, curve=curve, absten=absten,
               NOV=NOV, DD=DD, fails=fails, pitfall=pit, stages=stages, nlow=nlow,
               mpk=mpk)
    with open(os.path.join(ROOT, "data", "_v2.pkl"), "wb") as f:
        pickle.dump(out, f)
    R53.to_csv(os.path.join(ROOT, "data", "v2_identification.csv"), index=False)
    RLOO.to_csv(os.path.join(ROOT, "data", "v2_leave_one_out.csv"), index=False)
    B.to_csv(os.path.join(ROOT, "data", "v2_band_table.csv"), index=False)
    print("\nsaved: data/_v2.pkl + v2_*.csv")


if __name__ == "__main__":
    main()
