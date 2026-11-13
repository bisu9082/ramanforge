"""Are the early acquisitions of A-234 and A-242 a changing sample or a settling measurement?

For each early acquisition, against the mean of the later (stationary) acquisitions of the
same vial: cosine; matched band positions (prominence 0.05, two points, within 12 cm-1);
mean and RMS position difference; share of the difference spectrum in five ranges; and the
height of the broad feature near 454 cm-1 above a local linear baseline (405-505 cm-1).
Writes data/v3_early_acq_check.csv.
"""
import os
import numpy as np, pandas as pd
from scipy.signal import find_peaks
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, "data", "v3_measured.npz")); X = z["shift"]
CASES = [("A234", [3, 4, 5], [1, 2]), ("A242", [2, 3], [1])]
RANGES = [(250, 600), (600, 850), (850, 1150), (1150, 1400), (1400, 1800)]


def peaks(s):
    p, _ = find_peaks(s, prominence=0.05, distance=2); return X[p], s[p]


def h454(s):
    i, i1, i2 = (int(np.argmin(abs(X - v))) for v in (454, 405, 505))
    return float(s[i] - np.interp(X[i], [X[i1], X[i2]], [s[i1], s[i2]]))


rows = []
for ag, late, early in CASES:
    ref = np.mean([z[f"acq_{ag}_{r}"] for r in late], 0)
    pr, _ = peaks(ref)
    lat = [z[f"acq_{ag}_{r}"] for r in late]
    for t in early + late:
        s = z[f"acq_{ag}_{t}"]; ps, _ = peaks(s)
        d = [ps[np.argmin(abs(ps - p))] - p for p in pr if np.min(abs(ps - p)) <= 12]
        dif = s / np.linalg.norm(s) - ref / np.linalg.norm(ref)
        row = dict(agent=ag, acq=t, role="early" if t in early else "late",
                   cos_to_late_mean=float(s @ ref / np.linalg.norm(s) / np.linalg.norm(ref)),
                   bands_matched=len(d), bands_late=len(pr),
                   mean_shift=float(np.mean(d)), rms_shift=float(np.sqrt(np.mean(np.square(d)))),
                   h454=h454(s))
        for lo, hi in RANGES:
            m = (X >= lo) & (X < hi)
            row[f"diff_share_{lo}_{hi}"] = float(np.sum(dif[m] ** 2) / max(np.sum(dif ** 2), 1e-30))
        rows.append(row)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(ROOT, "data", "v3_early_acq_check.csv"), index=False)
print(D.round(3).to_string(index=False))
print("pixel spacing", round(float(np.median(np.diff(X))), 2))
