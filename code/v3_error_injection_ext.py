"""Error-injection test, two extensions requested at review simulation (outputs separate from
v3_error_injection.py, whose files are unchanged).

  (a) intensity error alone: the largest tau that keeps 27 of 30 with exact positions and
      band set (tau in 0.30-0.58 between the two slices of the main run).
  (b) systematic positional error: the positional error of RMS sigma is split into a shift
      shared by every band of an entry (fraction rho of the variance) and an independent
      per-band part; rho = 0 is the main run, rho = 1 a pure per-entry offset. A third
      variant applies a per-entry scale error (shift proportional to wavenumber) with the
      same RMS over the entry's bands.

Writes data/v3_error_injection_ext.csv and data/v3_error_injection_ext_meta.json.
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_error_injection as EI
from v3_error_injection import Runner, load_measured, GAMMA

R = int(os.environ.get("RF_EI_DRAWS", "100"))
SEED = 20261113
MODE = {"kind": "iid", "rho": 0.0}
_orig = EI.degrade


def degrade_sys(pos, h, sigma, m, tau, rng):
    if sigma > 0 and MODE["kind"] != "iid":
        if MODE["kind"] == "shift":
            rho = MODE["rho"]
            d = (rng.normal(0.0, np.sqrt(rho) * sigma * GAMMA)
                 + rng.normal(0.0, np.sqrt(1 - rho) * sigma * GAMMA, size=pos.size))
        else:  # scale: shift proportional to wavenumber, RMS sigma*GAMMA over this entry's bands
            k = rng.normal(0.0, 1.0) * sigma * GAMMA / np.sqrt(np.mean(pos ** 2))
            d = k * pos
        pos = pos + d
        return _orig(pos, h, 0.0, m, tau, rng)
    return _orig(pos, h, sigma, m, tau, rng)


def main():
    X, acq = load_measured()
    run = Runner(X, acq)
    rng = np.random.default_rng(SEED)
    EI.degrade = degrade_sys            # Runner.cell resolves degrade from the module
    rows = []
    MODE.update(kind="iid", rho=0.0)
    for t in (0.35, 0.40, 0.45, 0.50):
        rows.append(dict(variant="tau_alone", **run.cell(0.0, 0.0, t, R, rng)))
    for kind, rho in (("shift", 0.5), ("shift", 1.0), ("scale", 1.0)):
        MODE.update(kind=kind, rho=rho)
        for s in np.round(np.arange(0.1, 1.001, 0.1), 2):
            rows.append(dict(variant=f"{kind}_rho{rho}", **run.cell(float(s), 0.0, 0.0, R, rng)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_error_injection_ext.csv"), index=False)
    main_tau = pd.read_csv(os.path.join(ROOT, "data", "v3_error_injection_tau.csv"))
    def smax(v):
        s = D[(D.variant == v) & (D.mol_mean >= 27)].sigma
        return float(s.max()) if len(s) else None
    t = D[D.variant == "tau_alone"]
    meta = dict(R=R, seed=SEED,
                tau_alone=t[["tau", "mol_mean"]].to_dict("records"),
                tau_alone_max_for_27=float(t[t.mol_mean >= 27].tau.max()) if (t.mol_mean >= 27).any() else 0.30,
                sigma_for_27={v: smax(v) for v in D.variant.unique() if v != "tau_alone"})
    json.dump(meta, open(os.path.join(ROOT, "data", "v3_error_injection_ext_meta.json"), "w"), indent=1)
    print(D.to_string()); print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
