"""
ramanforge: non-circular reconstruction.

Problem:
  The previous calibration chose (s, Γ) so as to maximize the 'identification accuracy over
  the 9 species'. But that accuracy over the 9 species is the result of the paper -> circularity.
  Only 6 of the 144 grid cells reach 6/9, and shifting s by 0.4% drops it to 4/9.

Solution:
  s is fitted on 'band positions' only. Band positions are a physical observable that is
  independent of identification.
  Γ is fixed from the apparent width of the measured bands.
  Identification accuracy is not used anywhere; it is reported only as a result.

Result (0 free parameters):
  s_low(<800 cm-1) = 1.024, s_high(>800) = 0.987   [band position fit]
  Γ = 30 cm-1                                      [measured band width]
  band RMSE 13.5 cm-1 | molecular 5/9 | class 9/9 | all 4 incorrect calls within the same class
  5/9 and 9/9 unchanged when the library is expanded from 9 to 53
  class recovery 9/9 even when the correct entry is removed from the library
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.spectra import parse_raman
from src.calibrate import scattering_factor, _cos

ROOT = os.path.join(os.path.dirname(__file__), "..")
GRID = np.arange(250.0, 1800.0 + 1e-9, 2.0)
CUT = 800.0          # boundary between the low- and high-frequency regions
LASER, TEMP = 1064.0, 298.0

AGENT2JOB = {"GD": "EXP_GD_soman", "GF": "GF_cyclosarin", "HD": "EXP_HD",
             "HN1": "EXP_HN1", "HN3": "EXP_HN3", "A-230": "EXP_A230",
             "A-232": "EXP_A232", "A-234": "EXP_A234", "A-242": "EXP_A242"}
AGENT2KEY = {"GD": "gd", "GF": "gf", "HD": "hd", "HN1": "hn1", "HN3": "hn3",
             "A-230": "a230", "A-232": "a232", "A-234": "a234", "A-242": "a242"}
CLASS = {"EXP_GD_soman": "Nerve", "GF_cyclosarin": "Nerve",
         "EXP_HD": "Blister", "EXP_HN1": "Blister", "EXP_HN2": "Blister",
         "EXP_HN3": "Blister", "EXP_A230": "Novichok", "EXP_A232": "Novichok",
         "EXP_A234": "Novichok", "EXP_A242": "Novichok"}


def cls(job: str) -> str:
    """All OP-F analogues belong to the nerve agent class."""
    return CLASS.get(job, "Nerve")


# ------------------------------------------------------------------- load data
def load_library() -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "orca", "outputs", "*.out"))):
        name = os.path.basename(f)[:-4]
        if name == "GB_sarin_fast":      # re-run of GB_sarin (duplicate)
            continue
        out[name] = parse_raman(f)
    return out


def load_measured() -> dict:
    acc = {}
    for f in glob.glob(os.path.join(ROOT, "data", "exp_raman", "augmented_csv", "*.csv")):
        key = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f)
        acc.setdefault(key, []).append(np.interp(GRID, d.iloc[:, 0].values, d.iloc[:, 1].values))
    return {k: (lambda m: m / m.max())(np.mean(v, axis=0)) for k, v in acc.items()}


# ----------------------------------------------------------------- calibration
def scale_freqs(f: np.ndarray, s_low: float, s_high: float) -> np.ndarray:
    return np.where(f < CUT, f * s_low, f * s_high)


def synth(peaks: np.ndarray, s_low: float, s_high: float, fwhm: float) -> np.ndarray:
    f = scale_freqs(peaks[:, 0], s_low, s_high)
    a = peaks[:, 1].astype(float) * scattering_factor(f, LASER, TEMP)
    if a.max() > 0:
        a = a / a.max()
    g = fwhm / 2.0
    I = np.zeros_like(GRID)
    for ff, aa in zip(f, a):
        I += aa * g * g / ((GRID - ff) ** 2 + g * g)
    return I / (I.max() + 1e-12)


def measured_peaks(meas: dict) -> dict:
    return {ag: GRID[find_peaks(meas[AGENT2KEY[ag]], height=0.12, distance=6)[0]]
            for ag in AGENT2JOB}


def band_rmse(lib, mpk, s_low, s_high, tol=35.0):
    """Objective function that uses no identification at all: measured band positions vs scaled harmonic frequencies."""
    d = []
    for ag, job in AGENT2JOB.items():
        pk = lib[job]
        f = scale_freqs(pk[:, 0], s_low, s_high)
        keep = (f > GRID[0]) & (f < GRID[-1]) & (pk[:, 1] > 0.02 * pk[:, 1].max())
        f = f[keep]
        if len(f) == 0:
            continue
        for w in mpk[ag]:
            j = int(np.argmin(np.abs(f - w)))
            if abs(f[j] - w) <= tol:
                d.append(f[j] - w)
    d = np.asarray(d)
    return (float(np.sqrt((d ** 2).mean())) if len(d) else np.nan), len(d)


def fit_scale_on_positions(lib, mpk, min_pairs=85):
    best = None
    for sl in np.arange(0.980, 1.0401, 0.002):
        for sh in np.arange(0.965, 1.0101, 0.002):
            r, n = band_rmse(lib, mpk, float(sl), float(sh))
            if n >= min_pairs and (best is None or r < best[0]):
                best = (r, float(sl), float(sh), n)
    return best


# -------------------------------------------------------------- identification
def identify(lib_spec: dict, meas: dict, exclude_self=False):
    rows = []
    for ag, gold in AGENT2JOB.items():
        pool = {n: v for n, v in lib_spec.items() if not (exclude_self and n == gold)}
        sims = {n: _cos(meas[AGENT2KEY[ag]], v) for n, v in pool.items()}
        order = sorted(sims.items(), key=lambda x: -x[1])
        top, top_s = order[0]
        rows.append(dict(agent=ag, gold=gold, top1=top, score=top_s,
                         margin=top_s - order[1][1],
                         mol_ok=(top == gold), cls_ok=(cls(top) == cls(gold))))
    return pd.DataFrame(rows)


def main():
    lib = load_library()
    meas = load_measured()
    mpk = measured_peaks(meas)

    rmse, s_low, s_high, npairs = fit_scale_on_positions(lib, mpk)
    print(f"[1] band position fit (not used for identification): s_low={s_low:.3f} s_high={s_high:.3f} "
          f"RMSE={rmse:.1f} cm-1 (n={npairs})")

    fwhm = 30.0   # median of the apparent FWHM of the measured bands, 25-55 cm-1
    spec = {n: synth(v, s_low, s_high, fwhm) for n, v in lib.items()}

    R = identify(spec, meas)
    print(f"[2] identification (result): molecular {R.mol_ok.sum()}/9, class {R.cls_ok.sum()}/9")
    print(R[["agent", "top1", "score", "margin", "mol_ok", "cls_ok"]].to_string(index=False))

    Rx = identify(spec, meas, exclude_self=True)
    print(f"[3] correct answer removed from library: class recovery {Rx.cls_ok.sum()}/9")
    print(Rx[["agent", "top1", "score", "cls_ok"]].to_string(index=False))

    R.to_csv(os.path.join(ROOT, "data", "noncircular_identification.csv"), index=False)
    Rx.to_csv(os.path.join(ROOT, "data", "noncircular_leave_one_out.csv"), index=False)
    print("\nsaved: data/noncircular_identification.csv, data/noncircular_leave_one_out.csv")


if __name__ == "__main__":
    main()
