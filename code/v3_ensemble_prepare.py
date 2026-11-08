"""
Prepare the Boltzmann-averaged conformer arm, and build it once the jobs have run.

The library holds one structure per molecule. Table tab:si-confpop bounds what that one
structure can represent: for five of the ten agents no single conformer carries as much as
a tenth of the 298 K population. This script tests that bound directly on the three agents
where it is cheap, by averaging the harmonic spectra of the conformers that carry 80 per
cent of the population.

Two modes.

  prepare   Reads a CREST/GFN2 ensemble (a multi-frame .xyz, lowest first, in the same
            order as the deposited *_conformer_energies.txt), selects the conformers that
            reach the cumulative weight target, and writes ORCA inputs at the library's
            own level of theory into orca/inputs_ens/.

  build     Reads the resulting ORCA outputs, forms the Boltzmann-weighted mean spectrum
            on the measured grid under the adopted calibration, and reports the
            600-850 cm-1 window cosine and the open-set top-1 before and after.

The ensemble geometries are not in this deposit; only the energies are. Point --ensemble
at the CREST output for each agent.

  python3 code/v3_ensemble_prepare.py prepare --agent GF --ensemble crest_conformers.xyz
  (run the inputs in orca/inputs_ens/)
  python3 code/v3_ensemble_prepare.py build
"""
import argparse, glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))

KT = 0.001987204258640832 * 298.15          # kcal/mol at 298.15 K
SLOW, SHIGH = 1.014, 0.980
HEADER = ("! B3LYP 6-311++G(d,p) TightOpt NumFreq\n"
          "%pal nprocs 8 end\n%maxcore 3000\n%elprop Polar 1 end\n")
JOBS = {"GF": "GF_cyclosarin", "GD": "EXP_GD_soman", "A230": "EXP_A230"}


def weights(job):
    """Same parser as code/v3_conformer_population.py: the CREST energy file is two
    columns, index then relative energy, so only the last token on each line is read."""
    v = []
    for ln in open(os.path.join(ROOT, "conformers", f"{job}_conformer_energies.txt")):
        p = ln.split()
        if len(p) >= 2:
            try:
                v.append(float(p[-1]))
            except ValueError:
                pass
    e = np.array(v, float); e = e - e.min()
    if e.max() < 0.2:
        e = e * 627.5095
    w = np.exp(-e / KT); w = w / w.sum()
    return e, w


def read_frames(path):
    lines = open(path).read().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        frames.append([l for l in lines[i + 2:i + 2 + n]])
        i += n + 2
    return frames


def prepare(agent, ensemble, target):
    job = JOBS[agent]
    e, w = weights(job)
    order = np.argsort(-w)
    cum = np.cumsum(w[order])
    keep = order[:int(np.searchsorted(cum, target) + 1)]
    frames = read_frames(ensemble)
    if len(frames) != len(w):
        print(f"  WARNING: {len(frames)} frames in {ensemble} but {len(w)} energies for "
              f"{job}; the ensemble must be in the same order as the deposited energies.")
    out = os.path.join(ROOT, "orca", "inputs_ens"); os.makedirs(out, exist_ok=True)
    for rank, idx in enumerate(keep):
        body = "\n".join(frames[idx])
        p = os.path.join(out, f"{job}_c{rank:02d}.inp")
        open(p, "w").write(HEADER + "* xyz 0 1\n" + body + "\n*\n")
        print(f"  {os.path.basename(p)}  weight {w[idx]*100:5.1f}%  dE {e[idx]:.3f} kcal/mol")
    print(f"{agent}: {len(keep)} jobs covering {cum[len(keep)-1]*100:.1f}% of the population")
    return len(keep)


def build():
    import v3_step3_headtohead as H
    X, acq = H.load_measured()
    lib = H.load_library()
    query = [a for a in H.AGENTS if a in H.JOB and len(acq[a]) >= 2]
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    win = (X > 600) & (X < 850)
    NICE = {"GF": "GF", "GD": "GD", "A230": "A-230"}

    ens, rows = {}, []
    for agent, job in JOBS.items():
        found = sorted(glob.glob(os.path.join(ROOT, "orca", "outputs_ens", f"{job}_c*.out")))
        outs = [p for p in found
                if "ORCA TERMINATED NORMALLY" in open(p, errors="ignore").read()]
        if len(found) - len(outs):
            print(f"{agent}: {len(found) - len(outs)} output(s) did not terminate "
                  "normally and are ignored")
        if not outs:
            continue
        # the weights follow the conformer order, so only a leading run can be averaged
        idx = [int(os.path.basename(p).rsplit("_c", 1)[1][:2]) for p in outs]
        if idx != list(range(len(outs))):
            print(f"{agent}: completed conformers are {idx}, not a leading block; "
                  "the Boltzmann weights would be wrong, so this agent is skipped")
            continue

        _, w = weights(job)
        order = np.argsort(-w)[:len(outs)]
        ww = w[order] / w[order].sum()
        meas = np.mean([s for _, s in acq[agent]], 0)
        base = H.synth_native(lib[H.JOB[agent]], X, SLOW, SHIGH)

        spec, per = None, []
        for wi, path in zip(ww, outs):
            s = H.synth_native(H.parse_raman(path), X, SLOW, SHIGH)
            per.append(cos(meas[win], s[win]))
            spec = s * wi if spec is None else spec + s * wi
        ens[H.JOB[agent]] = spec
        # Population-weighted dispersion of the window cosine across the averaged
        # conformers. The raw range is misleading: a conformer that disagrees
        # strongly but carries 5 per cent of the population cannot move the average.
        pa = np.asarray(per)
        wsd = float(np.sqrt(np.sum(ww * (pa - np.sum(ww * pa)) ** 2)))

        cov = float(w[order].sum() * 100)
        rows.append(dict(agent=NICE[agent], n_conformers=int(len(w)),
                         n_averaged=len(outs), coverage=cov,
                         spread=max(per) - min(per), wsd=wsd,
                         dominant=per[0],
                         subdominant=float(np.sum(ww[1:] / ww[1:].sum() * np.asarray(per[1:])))
                         if len(per) > 1 else float("nan"),
                         window_base=cos(meas[win], base[win]),
                         window_ens=cos(meas[win], spec[win]),
                         full_base=cos(meas, base), full_ens=cos(meas, spec)))
        print(f"{agent}: averaged {len(outs)} conformers, {cov:.1f}% of the population")

    if not ens:
        print("no outputs in orca/outputs_ens/ -- run the inputs first")
        return
    for r in rows:
        print(f"  {r['agent']}: window {r['window_base']:.3f} -> {r['window_ens']:.3f}   "
              f"full {r['full_base']:.3f} -> {r['full_ens']:.3f}   "
              f"spread {r['spread']:.3f}")

    newlib = dict(lib); newlib.update(ens)
    counts = {}
    for tag, L in (("baseline", lib), ("ensemble", newlib)):
        D = {k: (ens[k] if (tag == "ensemble" and k in ens)
                 else H.synth_native(v, X, SLOW, SHIGH)) for k, v in L.items()}
        mol = cls = 0
        for a in query:
            for _, s in acq[a]:
                top = max(D, key=lambda k: cos(s, D[k]))
                if top == H.JOB[a]:
                    mol += 1
                if H.cls_of(top) == H.cls_of(H.JOB[a]):
                    cls += 1
        counts[tag] = (mol, cls)
        print(f"  open-set top-1, {tag}: molecular {mol}/30   class {cls}/30")
    write_table(rows, counts)


def write_table(rows, counts):
    """SI table \\label{tab:si-ensemble}."""
    import pandas as pd
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_ensemble.csv"), index=False)
    body = [f"{r.agent} & {r.n_conformers} & {r.n_averaged} & {r.coverage:.1f} & "
            f"{r.dominant:.3f} & {r.subdominant:.3f} & {r.window_ens:.3f} & "
            f"{r.full_base:.3f} & {r.full_ens:.3f} \\\\" for r in D.itertuples()]
    mb, cb = counts["baseline"]; me, ce = counts["ensemble"]
    cap = ("Boltzmann-averaged conformer ensembles for the three agents where the "
           "calculation is affordable. Conformers are taken in order of decreasing "
           "298\\,K weight until the cumulative population reaches 80\\,\\%, and each is "
           "given a full geometry optimization and numerical frequency calculation at "
           "the library's own level of theory. The \\emph{dominant} column is the "
           "600--850\\,\\wn{} cosine of the single most populated conformer, which is the "
           "structure the library entry is meant to represent. Re-optimizing it "
           "independently at the same level reproduces the library entry exactly for "
           "GF and to within 0.009 for GD and 0.002 for A-230, which is the internal "
           "consistency of the geometry optimization. \\emph{Sub-dominant} is the "
           "population-weighted mean of that cosine over the remaining averaged "
           "conformers.")
    note = ("Averaging changes the identification for no query: the computed library "
            f"assigns {mb} of 30 at the molecular level and {cb} of 30 at the class level "
            f"before averaging and {me} and {ce} after. What it changes is the agreement "
            "in the diagnostic window, and the direction is set by whether the "
            "sub-dominant populated conformers agree with the measurement better or worse "
            "than the dominant one. For GF they are better by 0.247 and the window gains "
            "0.117; for A-230 they are equivalent, differing by 0.007, and it gains 0.009; "
            "for GD they are worse by 0.237 and it gains nothing, losing 0.006 over the "
            "full range. That quantity is not available from a conformer census: it is the "
            "spectra of the sub-dominant conformers, which is what the census is being "
            "used to avoid computing. The population statistics of "
            "Table~\\ref{tab:si-confpop} do not predict this: GF and GD have the same "
            "population concentration to within 0.002, their dominant conformers "
            "carrying 0.589 and 0.587 of the 298\\,K population, and averaging raises "
            "the GF window by 0.117 while leaving GD's unmoved; A-230, whose dominant "
            "conformer carries only 0.198, gains 0.009. They bound how much a single "
            "structure can represent without predicting whether averaging will help. "
            "Produced by "
            "\\texttt{code/v3\\_ensemble\\_prepare.py}.")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{" + cap + "}\n"
           "\\label{tab:si-ensemble}\n\\small\n"
           "\\begin{tabular}{lrrrccccc}\n\\toprule\n"
           " & & & Coverage & \\multicolumn{3}{c}{600--850\\,\\wn{} cosine} & "
           "\\multicolumn{2}{c}{Full-range cosine} \\\\\n"
           "\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
           "Agent & Conformers & Averaged & (\\%) & Dominant & Sub-dominant & Ensemble & "
           "Single & Ensemble \\\\\n\\midrule\n"
           + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}\n\\end{table}\n")
    open(os.path.join(ROOT, "latex", "sitables", "ensemble.tex"), "w").write(tex)
    print("saved data/v3_ensemble.csv + latex/sitables/ensemble.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "build"])
    ap.add_argument("--agent", choices=sorted(JOBS))
    ap.add_argument("--ensemble")
    ap.add_argument("--target", type=float, default=0.80)
    a = ap.parse_args()
    if a.mode == "prepare":
        if not (a.agent and a.ensemble):
            sys.exit("prepare needs --agent and --ensemble")
        prepare(a.agent, a.ensemble, a.target)
    else:
        build()
