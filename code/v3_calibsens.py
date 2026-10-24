"""
ramanforge: sensitivity of the corrected library to the calibration.

Produces SI table \\label{tab:si-calibsens}. After the GF and A-232 geometries were
replaced by their global minima the frequency scale was refitted, which moved it by
0.2 per cent. This script reports the identification outcome with the scale held at its
pre-correction values and with it refitted, so that the sensitivity of the result to that
change is explicit.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

HELD = (1.018, 0.988)      # pre-correction calibration
COLS = [("Scale held at (%.3f, %.3f)" % HELD, HELD)]


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    calib = [a for a in H.AGENTS if a in H.JOB]
    _, SLr, SHr, _ = H.fit_scale(lib, mpk, calib)
    core = [H.JOB[a] for a in H.AGENTS if a in H.JOB]
    queryable = [a for a in H.AGENTS if len(acq[a]) >= 2]
    queries = [(a, r, s) for a in queryable for r, s in acq[a]]

    rows = []
    for lab, (SL, SH) in (("held", HELD), ("refitted", (SLr, SHr))):
        dft = {k: H.synth_native(v, X, SL, SH) for k, v in lib.items()}
        clib = {j: dft[j] for j in core}
        mol = 0
        per = {}
        top1 = {}
        for qa, qr, q in queries:
            r = H.identify(q, clib, H.JOB.get(qa, "__none__"))
            mol += r["mol_ok"]
            per.setdefault(qa, [0, 0])
            per[qa][0] += r["mol_ok"]
            per[qa][1] += 1
            top1.setdefault(qa, []).append(r["top1"])
        agents_ok = sum(1 for a, (k, n) in per.items() if k * 2 > n)
        rows.append(dict(which=lab, s_low=SL, s_high=SH, mol=mol, n=len(queries),
                         agents_ok=agents_ok, n_agents=len(per),
                         hn1=top1.get("HN1", ["n/a"])[0]))
        print(f"{lab:9s} s=({SL:.3f},{SH:.3f})  molecular {mol}/{len(queries)}  "
              f"agents {agents_ok}/{len(per)}  HN1 -> {top1.get('HN1',['n/a'])[0]}")

    h, r = rows
    esc = lambda t: t.replace("_", "\\_")
    mark = lambda t, a: ("\\checkmark" if t == H.JOB[a] else "$\\times$")
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Sensitivity of the corrected library to the calibration. The two columns differ
only in the frequency scale: the left holds it at the pre-correction values, the right
refits it to the corrected library, a change of 0.2 per cent.}}
\\label{{tab:si-calibsens}}
\\small
\\begin{{tabular}}{{lcc}}
\\toprule
 & Scale held at ({h['s_low']:.3f}, {h['s_high']:.3f}) & Scale refitted to ({r['s_low']:.3f}, {r['s_high']:.3f}) \\\\
\\midrule
Molecular top-1, closed set & {h['mol']}/{h['n']} & {r['mol']}/{r['n']} \\\\
Agents correct (majority) & {h['agents_ok']}/{h['n_agents']} & {r['agents_ok']}/{r['n_agents']} \\\\
HN1 assignment & {esc(h['hn1'])} ({mark(h['hn1'],'HN1')}) & {esc(r['hn1'])} ({mark(r['hn1'],'HN1')}) \\\\
\\bottomrule
\\end{{tabular}}

\\begin{{flushleft}}\\footnotesize A 0.2\\,\\% change in the scale factors moves the
molecular total by {abs(h['mol']-r['mol'])} of {h['n']} and flips all three HN1
acquisitions, whose pre-correction margin was 0.022. The refitted values are the ones
reported throughout, because the calibration procedure is defined before the
identification is computed and is not permitted to depend on it. That the outcome of the
computed library is not determined to the precision at which its calibration is
determined is itself a result about the library.\\end{{flushleft}}
\\end{{table}}
"""
    out = os.path.join(ROOT, "latex", "sitables", "calibsens.tex")
    open(out, "w").write(tex)
    print("\nsaved latex/sitables/calibsens.tex")


if __name__ == "__main__":
    main()
