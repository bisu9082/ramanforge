"""
v3 B2 verdict: did the improved P description fix the 600–850 cm-1 deficit.

A1 diagnosis: the measured↔computed gap is concentrated in 600–850 (P–F and P–N) and appears
         only for the phosphorus-containing agents.
         GF 0.562 / A-232 0.205  vs  non-phosphorus agents 0.83–0.97.

This script finds every B2_*.out in orca/outputs_b2/, swaps it in for the baseline
calculation of the corresponding agent, and compares the following.

  1) measured↔computed cosine over the 600–850 region   ← the primary criterion
  2) self-similarity cosine over the full region
  3) decomposition by region (delta relative to the baseline)
  4) whether the molecular top-1 for that agent flips (53-species open set)

The calibration (s_low, s_high) is **refitted separately** under the two conditions;
when the basis set, the functional or the solvent changes, the optimal scale changes too,
so using the baseline scale unchanged would underestimate the improvement. Γ=22 is a
measured value and is common to both.
"""
import os, sys, glob, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.spectra import parse_raman
from v3_step3_headtohead import (load_measured, load_library, synth_native, fit_scale,
                                 measured_peaks, cos, cls_of, JOB, CLASS, AGENTS, GAMMA)

REG = [("250–600", 250, 600), ("600–850", 600, 850), ("850–1150", 850, 1150),
       ("1150–1400", 1150, 1400), ("1400–1800", 1400, 1800)]
TAG2AGENT = {"GF": "GF", "A232": "A232", "A230": "A230", "A234": "A234", "GD": "GD", "HD": "HD"}


def region_cos(X, meas, spec):
    out = {}
    for name, lo, hi in REG:
        m = (X >= lo) & (X < hi)
        out[name] = cos(meas[m], spec[m])
    return out


def main():
    outdir = os.path.join(ROOT, "orca", "outputs_b2")
    files = sorted(glob.glob(os.path.join(outdir, "B2_*.out")))
    done = []
    for f in files:
        txt_ok = False
        with open(f, "r", errors="ignore") as fh:
            txt_ok = "ORCA TERMINATED NORMALLY" in fh.read()
        if txt_ok:
            done.append(f)
        else:
            print(f"  (incomplete, skipped: {os.path.basename(f)})")
    if not done:
        print("No completed B2 calculations. Check orca/outputs_b2/.")
        return
    print(f"completed B2 computations: {len(done)}\n")

    X, acq = load_measured()
    base_lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]

    # ---------- baseline ----------
    rb, SLb, SHb, nb = fit_scale(base_lib, mpk, calib)
    base_spec = {k: synth_native(v, X, SLb, SHb) for k, v in base_lib.items()}
    print(f"[baseline] s=({SLb:.3f},{SHb:.3f})  position RMSE {rb:.1f} cm-1 (n={nb})")

    rows = []
    for f in done:
        name = os.path.basename(f)[:-4]            # B2_GF_V1_cpcm
        _, tag, *rest = name.split("_")
        variant = "_".join(rest)
        agent = TAG2AGENT.get(tag)
        if agent is None:
            print(f"  ! no agent mapping: {name}"); continue
        job = JOB[agent]

        try:
            peaks = parse_raman(f)
        except Exception as e:
            print(f"  ! parsing failed {name}: {e}"); continue

        # swap only that one library entry and refit the scale factors
        lib2 = dict(base_lib); lib2[job] = peaks
        r2, SL2, SH2, n2 = fit_scale(lib2, mpk, calib)
        spec2 = {k: synth_native(v, X, SL2, SH2) for k, v in lib2.items()}

        cb = region_cos(X, mean[agent], base_spec[job])
        cn = region_cos(X, mean[agent], spec2[job])
        self_b = cos(mean[agent], base_spec[job])
        self_n = cos(mean[agent], spec2[job])

        # open-set molecule-level top-1
        def top1(sp):
            sims = {k: cos(mean[agent], v) for k, v in sp.items()}
            return max(sims, key=sims.get)
        t_b, t_n = top1(base_spec), top1(spec2)

        rows.append(dict(job=name, agent=agent, variant=variant,
                         s_low=SL2, s_high=SH2, pos_rmse=r2, n_pairs=n2,
                         self_base=self_b, self_new=self_n,
                         **{f"base_{k}": v for k, v in cb.items()},
                         **{f"new_{k}": v for k, v in cn.items()},
                         top1_base=t_b, top1_new=t_n,
                         correct_base=(t_b == job), correct_new=(t_n == job)))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(ROOT, "data", "v3_b2_evaluation.csv"), index=False)

    print("\n" + "=" * 76)
    print("★ Primary verdict: 600–850 cm-1 measured↔computed cosine")
    print("=" * 76)
    print(f"{'computed':<24}{'baseline':>10}{'improved':>10}{'Δ':>9}   verdict")
    for _, r in R.iterrows():
        b, n = r["base_600–850"], r["new_600–850"]
        d = n - b
        verdict = "★ significant improvement" if d > 0.10 else ("improved" if d > 0.03 else
                  ("no change" if d > -0.03 else "worse"))
        print(f"{r['job']:<24}{b:>10.3f}{n:>10.3f}{d:>+9.3f}   {verdict}")

    print("\nFull-region self-cosine:")
    for _, r in R.iterrows():
        print(f"  {r['job']:<24}{r['self_base']:>8.3f} → {r['self_new']:>6.3f}"
              f"  ({r['self_new']-r['self_base']:+.3f})")

    print("\nPer-region delta (improved − baseline):")
    hdr = "  " + f"{'computed':<24}" + "".join(f"{n:>12}" for n, _, _ in REG)
    print(hdr)
    for _, r in R.iterrows():
        line = f"  {r['job']:<24}"
        for n, _, _ in REG:
            line += f"{r[f'new_{n}'] - r[f'base_{n}']:>+12.3f}"
        print(line)

    print("\nopen-set molecules top-1 (53):")
    for _, r in R.iterrows():
        mark = "✓" if r["correct_new"] else "✗"
        was = "✓" if r["correct_base"] else "✗"
        print(f"  {r['job']:<24} {was} {r['top1_base']:<18} → {mark} {r['top1_new']}")

    print("\nRefitted scale factors:")
    for _, r in R.iterrows():
        print(f"  {r['job']:<24} s=({r['s_low']:.3f},{r['s_high']:.3f})  "
              f"position RMSE {r['pos_rmse']:.1f} (n={int(r['n_pairs'])})")
    print(f"  {'[baseline]':<24} s=({SLb:.3f},{SHb:.3f})  position RMSE {rb:.1f} (n={nb})")

    print("\n" + "=" * 76)
    best = R.assign(delta=R["new_600–850"] - R["base_600–850"]).sort_values("delta",
                                                                            ascending=False)
    b0 = best.iloc[0]
    if b0["delta"] > 0.10:
        print(f"Conclusion: {b0['variant']} improved 600–850 by {b0['delta']:+.3f}.")
        print("      → Recommend confirming A-232 along this line, then recomputing all.")
    elif b0["delta"] > 0.03:
        print(f"Conclusion: {b0['variant']} gave a small improvement ({b0['delta']:+.3f}). Other variants need checking.")
    else:
        print("Conclusion: none of the tested variants meaningfully fixed 600–850.")
        print("      → Remaining candidates: wB97X-D3(V3), P–F anharmonic correction, or diagnostic review.")
    print("=" * 76)
    print("\nsaved data/v3_b2_evaluation.csv")


if __name__ == "__main__":
    main()
