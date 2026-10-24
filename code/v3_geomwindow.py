"""
ramanforge: window-resolved effect of the geometry correction.

Produces SI table \\label{tab:si-geomwindow}. Two computed entries, GF and A-232, were
found not to be at the global minimum and were recomputed from the global-minimum
geometry. This script compares the published and the global-minimum spectrum of each
against the measured mean spectrum, window by window, with the frequency scale held at
its pre-correction values so that the only thing that changes is the computed spectrum of
the agent named.
"""
import os, sys
os.environ["RF_PRECORR"] = "1"          # must precede the import below
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H
from src.spectra import parse_raman

S_LOW, S_HIGH = 1.018, 0.988            # the pre-correction calibration, held fixed
REGIONS = [(250, 600), (600, 850), (850, 1150), (1150, 1400), (1400, 1800)]
PAIRS = [("GF", "GF_cyclosarin", "GF_cyclosarin_glob"),
         ("A232", "EXP_A232", "EXP_A232_Eglob")]


def window_cosines(query, peaks, X):
    s = H.synth_native(peaks, X, S_LOW, S_HIGH)
    v = [H.cos(query[(X >= lo) & (X <= hi)], s[(X >= lo) & (X <= hi)]) for lo, hi in REGIONS]
    return v + [float(np.mean(v))]


def main():
    import pandas as pd
    X, acq = H.load_measured()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    pre = H.load_library()
    hdr = [f"{lo}-{hi}" for lo, hi in REGIONS] + ["Mean"]
    rows = []
    print(f"calibration held at s_low={S_LOW}, s_high={S_HIGH}\n")
    print(f"{'':24s}" + "".join(f"{h:>12s}" for h in hdr))
    for agent, published, globmin in PAIRS:
        q = mean[agent]
        a = window_cosines(q, pre[published], X)
        b = window_cosines(q, parse_raman(os.path.join(ROOT, "orca", "outputs",
                                                       globmin + ".out")), X)
        lab = {"A232": "A-232"}.get(agent, agent)
        print(f"{lab + ', published':24s}" + "".join(f"{x:12.3f}" for x in a))
        print(f"{lab + ', global minimum':24s}" + "".join(f"{x:12.3f}" for x in b))
        print(f"{'  change':24s}" + "".join(f"{y - x:+12.3f}" for x, y in zip(a, b)))
        print()
        rows.append(dict(agent=lab, state="published", **dict(zip(hdr, a))))
        rows.append(dict(agent=lab, state="global minimum", **dict(zip(hdr, b))))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data", "v3_geomwindow.csv"), index=False)
    print("saved data/v3_geomwindow.csv")


if __name__ == "__main__":
    main()
