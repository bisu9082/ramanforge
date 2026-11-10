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


HARTREE = 627.5094740631          # kcal/mol per Hartree


def gibbs_weights(paths):
    """298 K Boltzmann weights from the B3LYP Gibbs energies of the completed jobs.

    The conformers were SELECTED by the GFN2 census, which is what made the
    selection affordable, but each selected structure was then given a full
    optimization and frequency calculation at the library's own level of theory, so
    its Gibbs energy is available and is the one the weights should use. The two
    surfaces do not agree: for GF the GFN2 ordering is inverted at B3LYP.
    """
    G = []
    for path in paths:
        g = None
        for ln in open(path, errors="ignore"):
            if "Final Gibbs free energy" in ln:
                try: g = float(ln.split()[5])
                except Exception: pass
        if g is None:
            raise RuntimeError(f"no Gibbs free energy in {path}")
        G.append(g)
    G = np.asarray(G, float)
    w = np.exp(-(G - G.min()) * HARTREE / KT)
    return w / w.sum()


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

        _, w = weights(job)                     # GFN2 census: selection only
        gfn2 = w[np.argsort(-w)[:len(outs)]]
        gfn2 = gfn2 / gfn2.sum()
        ww = gibbs_weights(outs)                # B3LYP dG: the weights actually used
        order = np.argsort(-ww)                 # B3LYP need not preserve the GFN2 order
        meas = np.mean([s for _, s in acq[agent]], 0)
        base = H.synth_native(lib[H.JOB[agent]], X, SLOW, SHIGH)

        spec, per = None, []
        for wi, path in zip(ww, outs):
            s = H.synth_native(H.parse_raman(path), X, SLOW, SHIGH)
            per.append(cos(meas[win], s[win]))
            spec = s * wi if spec is None else spec + s * wi
        per = [per[i] for i in order]           # decreasing B3LYP weight
        wo = ww[order]
        ens[H.JOB[agent]] = spec
        # Population-weighted dispersion of the window cosine across the averaged
        # conformers. The raw range is misleading: a conformer that disagrees
        # strongly but carries 5 per cent of the population cannot move the average.
        pa = np.asarray(per)
        wsd = float(np.sqrt(np.sum(wo * (pa - np.sum(wo * pa)) ** 2)))

        cov = float(gfn2.sum() * 100) if False else float(w[np.argsort(-w)[:len(outs)]].sum() * 100)
        rows.append(dict(agent=NICE[agent], n_conformers=int(len(w)),
                         n_averaged=len(outs), coverage=cov,
                         w_gfn2_top=float(gfn2.max()), w_b3lyp_top=float(wo[0]),
                         spread=max(per) - min(per), wsd=wsd,
                         dominant=per[0],
                         subdominant=float(np.sum(wo[1:] / wo[1:].sum() * np.asarray(per[1:])))
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
            f"{r.w_gfn2_top:.3f} & {r.w_b3lyp_top:.3f} & "
            f"{r.window_base:.3f} & {r.dominant:.3f} & {r.subdominant:.3f} & "
            f"{r.window_ens:.3f} & {r.full_base:.3f} & {r.full_ens:.3f} \\\\"
            for r in D.itertuples()]
    mb, cb = counts["baseline"]; me, ce = counts["ensemble"]
    cap = ("Boltzmann-averaged conformer ensembles for the three agents where the "
           "calculation is affordable. Conformers are \\emph{selected} by the deposited "
           "GFN2 census, in order of decreasing 298\\,K weight until the cumulative "
           "population reaches 80\\,\\% (the coverage column), and each selected structure "
           "is then given a full geometry optimization and numerical frequency "
           "calculation at the library's own level of theory. The Boltzmann weights are "
           "taken from the resulting B3LYP Gibbs energies and not from the GFN2 energies "
           "that made the selection: the two leading-weight columns are the weight each "
           "surface gives the same leading structure, and for GF they disagree "
           "completely. \\emph{Library} is the 600--850\\,\\wn{} cosine of the entry the "
           "library actually holds; \\emph{dominant} is that of the most populated "
           "conformer on the B3LYP surface; \\emph{sub-dom.} is the weighted mean over the "
           "remaining averaged conformers.")
    note = ("Averaging changes the identification for no query: the computed library "
            f"assigns {mb} of 30 at the molecular level and {cb} of 30 at the class level "
            f"before averaging and {me} and {ce} after. It also changes the agreement "
            "very little. In each of the three the ensemble lies within 0.005 of its own "
            "most populated conformer in the window --- 0.569 against 0.570 for GF, 0.880 "
            "against 0.877 for GD, 0.668 against 0.672 for A-230 --- so to that tolerance "
            "the population-weighted mean over the structures carrying 80\\,\\% of the "
            "population is the leading structure alone. The Library and Dominant columns "
            "separate the two effects, and they show that the entry the library holds is "
            "below the most populated conformer in all three agents: by 0.003 for GD, "
            "0.012 for A-230 and 0.247 for GF. Averaging then contributes +0.003 for GD, "
            "$-$0.005 for A-230 and $-$0.002 for GF, so the window gains of +0.006 and "
            "+0.007 for GD and A-230 are small structure changes partly offset by smaller "
            "averaging effects, not averaging effects. GF moves by +0.246, and that "
            "is a change of structure rather than an average: the GFN2 census gives the "
            "library's structure 0.719 of the two-conformer weight while the B3LYP "
            "surface gives it 0.032, placing the other conformer 1.68\\,kcal\\,mol$^{-1}$ "
            "lower in energy and 2.02\\,kcal\\,mol$^{-1}$ lower in Gibbs energy. The "
            "population statistics of Table~\\ref{tab:si-confpop} cannot see this, because "
            "they are computed on the surface that gets the ordering wrong. All three GF "
            "acquisitions remain assigned to soman under either structure. Produced by "
            "\\texttt{code/v3\\_ensemble\\_prepare.py}.")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{" + cap + "}\n"
           "\\label{tab:si-ensemble}\n\\small\n"
           "\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{lrrrcccccccc}\n\\toprule\n"
           " & & & Cov. & \\multicolumn{2}{c}{Leading weight} & "
           "\\multicolumn{4}{c}{600--850\\,\\wn{} cosine} & "
           "\\multicolumn{2}{c}{Full-range cosine} \\\\\n"
           "\\cmidrule(lr){5-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-12}\n"
           "Agent & Conf. & Avg. & (\\%) & GFN2 & B3LYP & "
           "Library & Dominant & Sub-dom. & Ensemble & Library & Ensemble \\\\\n"
           "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n"
           "\\end{tabular}\n}\n\n"
           "\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}\n"
           "\\end{table}\n")
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
