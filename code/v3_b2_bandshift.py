"""
Where the dominant 600-850 cm-1 band sits, before and after the larger phosphorus basis.

For each agent that has a V2_tightP trial, this reports the measured position of the
strongest band in the 600-850 cm-1 window, the position of the strongest computed band in
the same window under the baseline library and under the trial, and the signed errors and
shift. It produces data/v3_b2_bandshift.csv, which feeds Figure 5(d) and Table S-bandshift.

Round-4 revision. This script did not previously exist: the deposited CSV had been written
by hand from an earlier state of the analysis, run_all.sh could not regenerate it, and the
positions it carried were on the superseded calibration. Both sides are now detected with
the criterion stated in the Methods, and the scale factors are the ones the trial itself
refits, so the table and the figure follow the analysis instead of trailing it.
"""
import os, sys, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
DATA = os.path.join(ROOT, "data")
from src.spectra import parse_raman
import v3_step3_headtohead as H

LO, HI = 600.0, 850.0
TAG2AGENT = {"A230": "A230", "A232": "A232", "A234": "A234",
             "GF": "GF", "GD": "GD", "HD": "HD", "VX": "VX",
             "HN1": "HN1", "HN2": "HN2", "HN3": "HN3", "A242": "A242"}


def strongest_in_window(f, a):
    k = (f >= LO) & (f <= HI)
    if not k.any():
        return np.nan
    return float(f[k][int(np.argmax(a[k]))])


def main():
    X, acq = H.load_measured()
    base_lib = H.load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    calib = [a for a in H.AGENTS if a in H.JOB]
    _, SLb, SHb, _ = H.fit_scale(base_lib, mpk, calib)
    print(f"[baseline] s=({SLb:.3f}, {SHb:.3f})")

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "orca", "outputs_b2", "*_V2_tightP.out"))):
        tag = os.path.basename(path).split("_")[1]
        ag = TAG2AGENT.get(tag)
        if ag is None or ag not in mean:
            continue
        job = H.JOB[ag]

        # measured: strongest detected band in the window
        w = [v for v in mpk[ag] if LO <= v <= HI]
        if not w:
            continue
        meas = float(max(w, key=lambda v: mean[ag][int(np.argmin(np.abs(X - v)))]))

        # computed, baseline and trial, under the scale factors each refits
        fb, ab = H.peaks_of_lines(
            np.where(base_lib[job][:, 0] < H.CUT, base_lib[job][:, 0] * SLb,
                     base_lib[job][:, 0] * SHb), base_lib[job][:, 1], X, return_height=True)
        base = strongest_in_window(fb, ab)

        trial = parse_raman(path)
        lib2 = dict(base_lib); lib2[job] = trial
        _, SL2, SH2, _ = H.fit_scale(lib2, mpk, calib)
        ft, at = H.peaks_of_lines(
            np.where(trial[:, 0] < H.CUT, trial[:, 0] * SL2, trial[:, 0] * SH2),
            trial[:, 1], X, return_height=True)
        new = strongest_in_window(ft, at)
        if not np.isfinite(base) or not np.isfinite(new):
            continue
        rows.append(dict(agent=ag, measured=meas, base=base, tightp=new,
                         err_base=base - meas, err_new=new - meas, shift=new - base))

    D = pd.DataFrame(rows).sort_values("agent").reset_index(drop=True)
    D.to_csv(os.path.join(DATA, "v3_b2_bandshift.csv"), index=False)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("wrote data/v3_b2_bandshift.csv")


if __name__ == "__main__":
    main()
