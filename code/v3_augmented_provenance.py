"""Provenance of the 23 augmented (non-instrument) files in the delivered data set.

The data set available for this work had been assembled for classifier training and
balanced to five files per vial by augmentation. For every file whose RawData column is not
integer (not an instrument record), find the instrument record of the same vial it
reproduces (highest correlation), and report the fitted factor and offset, the residual
s.d., the same fit on the Dark column, and whether the header matches the parent's.
None of these files enters any analysis. Raw acquisition files are held under the
facility's records (see the paper); this script documents the check.

Writes data/v3_augmented_provenance.csv.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rawio import load_all

RAW = os.environ.get("RF_RAW_DIR", os.path.join(ROOT, "data", "raw"))


def fit(x, y):
    (a, b), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    return a, b, float(np.std(y - (a * x + b)))


def main():
    F = load_all(RAW)
    hdr_keys = ("IntTime", "CCDTemperature", "Gain", "Offset")
    gen = lambda f: np.allclose(f["df"].RawData, np.round(f["df"].RawData))
    rows = []
    for f in F:
        if gen(f):
            continue
        best = None
        for g in F:
            if g["meta"]["agent"] != f["meta"]["agent"] or not gen(g):
                continue
            x, y = g["df"].RawData.values.astype(float), f["df"].RawData.values.astype(float)
            r = float(np.corrcoef(x, y)[0, 1])
            if best is None or r > best[0]:
                best = (r, g)
        r, g = best
        a, b, s = fit(g["df"].RawData.values.astype(float), f["df"].RawData.values.astype(float))
        ad, bd, sd = fit(g["df"].Dark.values.astype(float), f["df"].Dark.values.astype(float))
        rows.append(dict(file=f["meta"]["file"], agent=f["meta"]["agent"], rep=f["meta"]["rep"],
                         parent_rep=g["meta"]["rep"], r=r, factor=a, offset=b, residual_sd=s,
                         dark_factor=ad, dark_residual_sd=sd,
                         header_matches_parent=all(f["meta"][k] == g["meta"][k] for k in hdr_keys)))
    D = pd.DataFrame(rows).sort_values(["agent", "rep"])
    D.to_csv(os.path.join(ROOT, "data", "v3_augmented_provenance.csv"), index=False)
    print(D.round(4).to_string(index=False))
    print(f"\n{len(D)} augmented files; factor {D.factor.min():.3f}-{D.factor.max():.3f}; "
          f"residual s.d. {D.residual_sd.min():.0f}-{D.residual_sd.max():.0f} counts; "
          f"header identical to parent in {int(D.header_matches_parent.sum())}")


if __name__ == "__main__":
    main()
