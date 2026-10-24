"""
v3 SI table generator: replaces the v2 tables in latex/sitables/ entirely with v3 data.

The v2 SI is based on the agent-average over 9 species (molecular 5/9, class 9/9), whereas the
v3 main text is based on the 30 genuine acquisitions (measured 30/30, computed 14/30). Having
both in the same document is a direct contradiction, so every table cited by the main text is
rebuilt here in one pass.

The output files keep the existing file names (the \input paths in si.tex are preserved).
"""
import os, sys, json, math
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "latex", "sitables")
os.makedirs(OUT, exist_ok=True)

NICE = {"GD": "GD", "GF": "GF", "VX": "VX", "HD": "HD", "HN1": "HN1", "HN2": "HN2",
        "HN3": "HN3", "A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}
ORDER = ["GD", "GF", "VX", "HD", "HN1", "HN2", "HN3", "A230", "A232", "A234", "A242"]
def nm(a): return NICE.get(str(a), str(a))
def esc(s): return str(s).replace("_", r"\_")


WIDE = {"peracq.tex", "regions.tex", "regbyagent.tex", "absten.tex", "bandlist.tex", "calibgrid.tex",
        "sessions.tex", "ident.tex", "theory.tex"}


# A few tables need a placement or a scale of their own to fit the page as typeset in the
# Supplementary Material. Keeping those here means a clean re-run reproduces the document.
FLOAT = {"bandlist.tex": "p"}
SCALE = {"bandlist.tex": "0.84\\textwidth"}


def table(fn, caption, label, colspec, header, rows, note=None, small=True):
    wide = fn in WIDE
    L = [f"\\begin{{table}}[{FLOAT.get(fn, 'htbp')}]", "\\centering",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small: L.append("\\footnotesize" if wide else "\\small")
    if wide: L.append(f"\\resizebox{{{SCALE.get(fn, chr(92) + 'textwidth')}}}{{!}}{{%")
    L += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
          " & ".join(header) + " \\\\", "\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    if wide: L.append("}")
    if note:
        L.append("\n\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}")
    L.append("\\end{table}")
    p = os.path.join(OUT, fn)
    open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return fn


made = []
H = pd.read_csv(f"{DATA}/v3_headtohead.csv")
_H2HM = json.load(open(f"{DATA}/v3_headtohead_meta.json"))
_LIBN = int(_H2HM["n_library"])        # full computed library (open set)
_CLOSN = int(_H2HM["n_agents_lib"])    # closed set: one computed entry per measured agent
META = json.load(open(f"{DATA}/v3_headtohead_meta.json"))
PRE = json.load(open(f"{DATA}/v3_preproc_meta.json"))

# ------------------------- sessions : register of raw acquisitions
A = pd.read_csv(f"{DATA}/v3_acquisition_audit.csv")
S = pd.read_csv(f"{DATA}/v3_agent_summary.csv").set_index("agent")
rows = []
for a in ORDER:
    if a not in S.index: continue
    r = S.loc[a]
    sub = A[(A.agent == a) & (A.genuine)]
    t = sorted(sub.IntTime_ms.astype(int).tolist())
    ts = f"{t[0]}--{t[-1]}" if len(t) > 1 and t[0] != t[-1] else str(t[0])
    rows.append([nm(a), str(int(r.n_genuine)), str(int(r.n_files) - int(r.n_genuine)),
                 ts, f"{r.peak_counts:.0f}", f"{r.SNR:.0f}",
                 f"{r.counts_per_s:.0f}", str(int(r.n_bands))])
made.append(table(
    "sessions.tex",
    "Inventory of raw acquisitions. Genuine acquisitions are those whose RawData column "
    "contains integer detector counts; the derived files reproduce a genuine parent to a "
    "multiplicative factor near unity with an added offset and noise, and are excluded "
    "throughout. Integration time is recorded per file by the instrument. Bands are those "
    "detected above a normalized intensity of 0.10 in the 250--1800\\,\\wn{} region.",
    "tab:si-sessions", "lrrrrrrr",
    ["Agent", "Genuine", "Derived", "Integration (ms)", "Peak counts",
     "SNR", "Counts\\,s$^{-1}$", "Bands"], rows,
    note=f"Total: {int(S.n_genuine.sum())} genuine acquisitions of "
         f"{len(S)} agents from {int(S.n_files.sum())} files. HD and VX have a single "
         "genuine acquisition each and are never used as queries."))

# ------------------------- quality : data quality and HN2
rows = []
for a in ORDER:
    if a not in S.index: continue
    r = S.loc[a]
    rows.append([nm(a), str(int(r.n_bands)), str(int(r.n10)), str(int(r.n20)),
                 f"{r.SNR:.0f}", f"{r.top_frac:.3f}",
                 "retained" if int(r.n10) >= 4 else "\\textbf{fails}"])
made.append(table(
    "quality.tex",
    "Data-quality criterion, applied before any identification was attempted. A measured "
    "spectrum is retained if at least four bands reach 10\\% of its strongest band within "
    "250--1800\\,\\wn{}.",
    "tab:si-quality", "lrrrrrl",
    ["Agent", "Bands", "$\\geq$10\\%", "$\\geq$20\\%", "SNR",
     "Top-band share", "Decision"], rows,
    note="The vial labeled HN2 is the one failure, and the reason is spectral content "
         "rather than acquisition quality: its signal-to-noise ratio is the highest of the "
         "eleven vials and its count rate the second highest. The previous version of this "
         "work attributed the failure to a poor measurement, which these figures do not "
         "support. Failing the criterion does not remove the vial from the analysis: it is "
         "retained in both libraries and among the thirty queries, so that the two "
         "libraries face an identical query set. It is set aside where its unknown identity "
         "would leave a quantity undefined: from the per-agent window analyses, from the "
         "in-library and class-absent conditions of Section~S7, where it is reported as a "
         "condition of its own, and from the correlation against molecular size in "
         "Figure~\\ref{fig:si-step4}. It contributes no "
         "band pair to the frequency calibration, which is therefore identical with and "
         "without it (Table~\\ref{tab:si-loao}). Section~S6 gives the independent evidence "
         "from the computed library and Section~S11 the external evidence."))

# ------------------------- gamma : the linewidth is measured, not fitted
G = pd.read_csv(f"{DATA}/v3_linewidths.csv")
gg = G.groupby("agent")
rows = []
for a in ORDER:
    if a not in gg.groups: continue
    s = gg.get_group(a)
    rows.append([nm(a), str(len(s)), f"{s.gamma.median():.1f}",
                 f"{s.gamma.min():.1f}--{s.gamma.max():.1f}",
                 f"{s.r2.median():.2f}", str(int(s.n_bands.median()))])
made.append(table(
    "gamma.tex",
    "The Lorentzian linewidth is measured, not fitted to identification. For each "
    "acquisition a single width shared by all detected bands was fitted simultaneously "
    "with the band amplitudes, band centers being free to move by at most one pixel.",
    "tab:si-gamma", "lrrrrr",
    ["Agent", "Acquisitions", "$\\Gamma$ median (\\wn)", "Range (\\wn)",
     "$R^2$ median", "Bands fitted"], rows,
    note=f"Of the {PRE['n_genuine']} genuine acquisitions, {len(G)} yielded a converged "
         f"fit ($R^2 > 0.8$); the remaining {PRE['n_genuine']-len(G)} did not and are "
         f"excluded from this table. Over the {len(G)} the median is "
         f"{G.gamma.median():.1f}\\,\\wn{{}} with an interquartile range of "
         f"{G.gamma.quantile(.25):.1f}--{G.gamma.quantile(.75):.1f}\\,\\wn{{}}. "
         "We adopt $\\Gamma$ = 22\\,\\wn{}. The extremes are not trustworthy: the vial "
         "labeled HN2 has only one band above the 10\\% criterion of "
         "Table~\\ref{tab:si-quality}, so its fitted width rests on weak features, and HD "
         "and VX have a single acquisition each. This is an "
         "apparent width that includes the instrument response function. No "
         "identification result was consulted in obtaining it."))

# ------------------------- preproc : sensitivity grid
P = pd.read_csv(f"{DATA}/v3_preproc_sensitivity.csv")
rows = []
for _, r in P.iterrows():
    v = r["value"]
    vs = v if isinstance(v, str) else (f"{v:g}")
    cos = "n/a" if pd.isna(r.cos_vs_default) else f"{r.cos_vs_default:.4f}"
    star = "\\textbf{" if abs(r.separation - 0.123) < 1e-9 and cos in ("1.0000",) else ""
    end = "}" if star else ""
    rows.append([esc(r.parameter), f"{star}{vs}{end}", cos,
                 f"{r.within_class:.3f}", f"{r.between_class:.3f}",
                 f"{star}{r.separation:.3f}{end}"])
made.append(table(
    "preproc.tex",
    "Sensitivity of the preprocessing specification. Cosine is taken against the adopted "
    "setting (bold); separation is the mean within-class minus the mean between-class "
    "cosine over the eleven measured agents. Entries for the fingerprint limits change "
    "the grid itself, so no cosine is defined.",
    "tab:si-preproc", "llrrrr",
    ["Parameter", "Value", "Cosine vs adopted", "Within", "Between", "Separation"], rows,
    note="The cosine against the adopted spectrum is not defined for the rows that change "
         "the analysis window, because a different window puts the spectrum on a different "
         "grid; those cells read n/a. "
         "The baseline parameters were fixed at values conventional in the literature "
         "before any identification was run. Over three orders of magnitude in $\\lambda$ "
         "the spectra stay within a cosine of 0.967 of the adopted setting and the "
         "separation moves between 0.104 and 0.125; over $p$ from 0.001 to 0.05 the "
         "cosine stays above 0.982 and the separation moves between 0.100 and 0.139. "
         "Standard-normal-variate normalization gives a higher separation (0.188) while "
         "changing the spectra substantially, and both are reported rather than the "
         "better-scoring one being adopted."))

# ------------------------- ident : identification summary by agent
rows = []
for a in ORDER:
    s = H[H.agent == a]
    if not len(s): continue
    rows.append([nm(a), str(len(s)),
                 f"{int(s.meas_mol.sum())}/{len(s)}", f"{s.meas_self.mean():.3f}",
                 f"{int(s.dft_mol.sum())}/{len(s)}", f"{s.dft_self.mean():.3f}",
                 f"{int(s.dft_cls.sum())}/{len(s)}",
                 f"{int(s.dft53_mol.sum())}/{len(s)}"])
tot = [f"\\textbf{{{int(H.meas_mol.sum())}/{len(H)}}}", f"\\textbf{{{H.meas_self.mean():.3f}}}",
       f"\\textbf{{{int(H.dft_mol.sum())}/{len(H)}}}", f"\\textbf{{{H.dft_self.mean():.3f}}}",
       f"\\textbf{{{int(H.dft_cls.sum())}/{len(H)}}}", f"\\textbf{{{int(H.dft53_mol.sum())}/{len(H)}}}"]
rows.append(["\\textbf{All}", f"\\textbf{{{len(H)}}}"] + tot)
made.append(table(
    "ident.tex",
    "Identification of the same thirty genuine acquisitions against a measured and against "
    "a computed library, under identical preprocessing, an identical native grid and an "
    "identical cosine metric. The query acquisition is never present in the library it is "
    "matched against.",
    "tab:si-ident", "lrrrrrrr",
    ["Agent", "$n$", "Measured mol.", "Self cos.", "Computed mol.", "Self cos.",
     "Computed class", f"{_LIBN}-member mol."], rows,
    note=f"HD and VX have a single genuine acquisition each and are retained in both "
         f"libraries but not queried. The three HN2 queries fail "
         f"against the computed library at a self-similarity of "
         f"{H[H.agent=='HN2'].dft_self.mean():.3f}; Section~S6 treats "
         f"that as evidence about the sample."))

# ------------------------- peracq : every acquisition, one row each
rows = []
for _, r in H.sort_values(["agent", "rep"]).iterrows():
    rows.append([nm(r.agent), str(int(r.rep)),
                 esc(r.meas_top1), "\\checkmark" if r.meas_mol else "$\\times$",
                 f"{r.meas_self:.3f}", f"{r.meas_margin:.3f}",
                 esc(r.dft_top1), "\\checkmark" if r.dft_mol else "$\\times$",
                 f"{r.dft_self:.3f}", f"{r.dft_margin:.3f}"])
made.append(table(
    "peracq.tex",
    "Assignment of every individual acquisition. Each row is one genuine acquisition "
    "queried against each library in turn. In the OK columns \\checkmark{} marks a correct "
    "molecular assignment and $\\times$ an incorrect one.",
    "tab:si-peracq", "llllrrllrr",
    ["Agent", "Acq.", "Measured top-1", "OK", "Self", "Margin",
     "Computed top-1", "OK", "Self", "Margin"], rows,
    note="The agent-mean figures quoted elsewhere in the literature average over these "
         "rows. We report the per-acquisition result because it is what a single "
         "interrogation delivers."))

# ------------------------- loao : nested calibration
N = pd.read_csv(f"{DATA}/v3_nested_calibration.csv")
rows = [[nm(r.held_out), f"{r.s_low:.4f}", f"{r.s_high:.4f}",
         f"{r.rmse:.2f}", str(int(r.n_pairs))] for _, r in N.iterrows()]
rows.append(["\\textbf{All agents}", f"\\textbf{{{META['s_low']:.4f}}}",
             f"\\textbf{{{META['s_high']:.4f}}}", f"\\textbf{{{META['pos_rmse']:.2f}}}",
             f"\\textbf{{{META['n_pairs']}}}"])
made.append(table(
    "loao.tex",
    "Nested leave-one-agent-out calibration. The two frequency scale factors were refitted "
    "with each agent's band positions excluded in turn.",
    "tab:si-loao", "lrrrr",
    ["Agent held out", "$s_\\mathrm{low}$", "$s_\\mathrm{high}$",
     "RMSE (\\wn)", "Band pairs"], rows,
    note=f"$s_\\mathrm{{low}}$ varies over "
         f"{N.s_low.min():.4f}--{N.s_low.max():.4f}, a range of "
         f"{(N.s_low.max()-N.s_low.min())/META['s_low']*100:.2f}\\%, and "
         f"$s_\\mathrm{{high}}$ over {N.s_high.min():.4f}--{N.s_high.max():.4f}, a range of "
         f"{(N.s_high.max()-N.s_high.min())/META['s_high']*100:.2f}\\%. No agent drives "
         "the calibration, and the identification results are obtained under factors "
         "effectively independent of the agent being identified."))

print("generated:", len(made), "tables")
for f in made: print("  ", f)


# ------------------------- regions : region decomposition and the phosphorus split
RB = pd.read_csv(f"{DATA}/v3_region_breakdown.csv")
PF = pd.read_csv(f"{DATA}/v3_pf_region_split.csv")
RA = pd.read_csv(f"{DATA}/v3_region_by_agent.csv").rename(columns={"Unnamed: 0": "agent"})
pfm = PF.set_index("region")
rows = []
for _, r in RB.iterrows():
    reg = r.region
    p = pfm.loc[reg]
    hi = (reg.replace("–", "-") == "600-850")
    b = (lambda s: f"\\textbf{{{s}}}") if hi else (lambda s: s)
    rows.append([b(reg.replace("–", "--")), b(f"{r.dft_cos:.3f}"),
                 b(f"{r.repro_cos:.3f}"), b(f"{r.gap:.3f}"),
                 b(f"{r.weight*100:.1f}\\%"),
                 b(f"{p.pf:.3f}"), b(f"{p.nop:.3f}"), b(f"{p.delta:+.3f}"),
                 b(f"{p.p:.3f}")])
_PSET_R = ("GD", "GF", "VX", "A230", "A232", "A234", "A242")
_RAt = pd.read_csv(f"{DATA}/v3_region_by_agent.csv", index_col=0)
_nP_n = sum(1 for a in _RAt.index if a in _PSET_R)
_nN_n = len(_RAt.index) - _nP_n
import math as _m
_pmin_w = 1.0 / _m.comb(_nP_n + _nN_n, _nN_n)
_W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight"}
_nP_w, _nNP_w = _W.get(_nP_n, str(_nP_n)), _W.get(_nN_n, str(_nN_n))
_p68_w = float(PF[PF.region.str.replace("–", "-") == "600-850"].p.iloc[0])
made.append(table(
    "regions.tex",
    f"Where the computed library loses, resolved by spectral region. The measured-to-computed "
    f"cosine is compared with the measured-to-measured cosine in the same region, which is the "
    f"ceiling that measurement repeatability permits. The last four columns split the same "
    f"comparison by whether the agent contains phosphorus ({_nP_w} agents) or not ({_nNP_w}).",
    "tab:si-regions", "lrrrrrrrr",
    ["Region (\\wn)", "Computed", "Repeatability", "Gap", "Intensity",
     "P-containing", "No P", "$\\Delta$", "$p$"], rows,
    note=f"The largest gap coincides with the largest share of measured intensity. Within "
         f"the 600--850\\,\\wn{{}} window the deficit belongs to the phosphorus-containing "
         f"agents; in each of the four other windows the ordering reverses. With "
         f"{_nNP_w} agents on one side and {_nP_w} on the other, the smallest attainable "
         f"one-sided $p$ is {_pmin_w:.3f} and the observed value is ${_p68_w:.3f}$, so the "
         f"separation is not significant at the 0.05 level; the argument rests on the size "
         f"of the effect and on its confinement to one window rather than on that $p$ value. "
         f"$p$ is one-sided Mann--Whitney."))

# ------------------------- regbyagent : regions by agent
cols = [c for c in RA.columns if c != "agent"]
rows = []
for a in ORDER:
    s = RA[RA.agent == a]
    if not len(s): continue
    r = s.iloc[0]
    P_ = a in ("GD", "GF", "VX", "A230", "A232", "A234", "A242")
    vals = []
    for c in cols:
        v = f"{r[c]:.3f}"
        vals.append(f"\\textbf{{{v}}}" if ("600-850" in c and P_) else v)
    rows.append([nm(a), "yes" if P_ else "no"] + vals)
_PSET = ("GD", "GF", "VX", "A230", "A232", "A234", "A242")
_c68 = [c for c in cols if "600-850" in c][0]
_pv = [RA[RA.agent == a].iloc[0][_c68] for a in ORDER if a in _PSET and len(RA[RA.agent == a])]
_nv = [RA[RA.agent == a].iloc[0][_c68] for a in ORDER if a not in _PSET and len(RA[RA.agent == a])]
_RBmin, _RBmax = min(_pv), max(_pv)
_pnames = [a for a in ORDER if a in _PSET and len(RA[RA.agent == a])]
_RBbest = nm(max(_pnames, key=lambda a: RA[RA.agent == a].iloc[0][_c68]))
_RBworst = nm(min(_pnames, key=lambda a: RA[RA.agent == a].iloc[0][_c68]))
_RNmin, _RNmax = min(_nv), max(_nv)
_nNP = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(len(_nv), str(len(_nv)))
made.append(table(
    "regbyagent.tex",
    "Measured-to-computed cosine by region for every queried agent. Bold entries are the "
    "600--850\\,\\wn{} window of the phosphorus-containing agents.",
    "tab:si-regbyagent", "ll" + "r" * len(cols),
    ["Agent", "Contains P"] + [c.replace("–", "--").replace("-", "--") for c in cols], rows,
    note=f"The phosphorus-containing agents span {_RBmin:.3f} to {_RBmax:.3f} in the 600--850\\,\\wn{{}} "
         f"window against {_RNmin:.3f}--{_RNmax:.3f} for the {_nNP} that contain no phosphorus. "
         f"These data do not detect an ordering by conformational flexibility: a GFN2 "
         f"conformer census over all eleven computed molecules gives no association between "
         f"conformational entropy and this cosine (Pearson $r$ = +0.23, $n$ = 10, 95\\% "
         f"interval roughly $-0.45$ to $+0.75$). The interval excludes nothing, and the two "
         f"extremes of the comparison are {_RBbest}, which has a single genuine acquisition "
         f"and is never queried, and the post-correction {_RBworst}. Whether a "
         f"Boltzmann-weighted ensemble spectrum closes the deficit is untested here."))

# ------------------------- absten : abstention score AUC and operating point
AU = pd.read_csv(f"{DATA}/v3_abstention_auc.csv")
OP = pd.read_csv(f"{DATA}/v3_abstention_operating.csv").set_index("score")
CM = pd.read_csv(f"{DATA}/v3_abstention_compare.csv").set_index("score")
SC = {"S1_top1": "top-1 cosine", "S2_margin": "margin",
      "S3_psr": "peak-to-sidelobe ratio", "S4_libz": "library-normalized $z$",
      "S5_resid": "unmixing residual", "S6_bandfrac": "band-match fraction"}
rows = []
for _, r in AU.iterrows():
    k = r.score; o = OP.loc[k]; c = CM.loc[k]
    rows.append([SC[k], f"{r.in_med:.3f}", f"{r.loco_med:.3f}", f"{r.hn2_med:.3f}",
                 f"{r.auc:.3f}", "yes" if c.sep_computed else "no",
                 f"{c.auc_measured:.3f}", "\\textbf{yes}" if c.sep_measured else "no",
                 f"{o.abstain_IN*100:.0f}\\%", f"{o.abstain_LOCO*100:.0f}\\%"])
made.append(table(
    "absten.tex",
    "Six abstention scores evaluated on the same queries under three conditions: the class "
    "present in the library (accept), the query's entire toxicological class removed "
    "(decline), and the HN2 acquisitions. Thresholds are set by nested leave-one-agent-out, "
    "so no threshold is chosen on the queries it is applied to.",
    "tab:si-absten", "lrrrrcrcrr",
    ["Score", "In-library", "Class absent", "HN2", "AUC", "Separable?",
     "AUC (meas.)", "Sep.?", "Decline IN", "Decline LOCO"], rows,
    note="Against the computed library no score separates the two conditions without error. Driving the "
         "class-absent acceptance to zero requires a conjunction of the library-normalized "
         "$z$ and the top-1 cosine, and costs 63\\% of the legitimate queries. Against the "
         "measured library three scores separate completely: in the plain top-1 cosine the "
         "lowest in-library value is 0.842 and the highest class-absent value 0.729, so the "
         "two populations are disjoint and any threshold in between accepts all 27 and "
         "declines all 27. That interval is an upper bound "
         "obtained on one instrument, one session and one vial per agent, and is not a "
         "transferable operating value; the contrast between the libraries is the result."))

# ------------------------- loco : where the answer goes when the class is absent
SCr = pd.read_csv(f"{DATA}/v3_abstention_scores.csv")
LO = SCr[SCr.cond == "LOCO"]
rows = []
for a in ORDER:
    s = LO[LO.agent == a]
    if not len(s): continue
    tops = s.top1_name.value_counts()
    dest = ", ".join(f"{esc(k)}" + (f" ($\\times${v})" if v > 1 else "")
                     for k, v in tops.items())
    rows.append([nm(a), str(len(s)), dest, f"{s.S1_top1.mean():.3f}",
                 f"{s.S2_margin.mean():.3f}", f"{s.S4_libz.mean():+.2f}"])
made.append(table(
    "loco.tex",
    "Leave-one-class-out. Every library member of the query's toxicological class was "
    "deleted before matching, so the correct answer is absent by construction and the "
    "workflow should decline.",
    "tab:si-loco", "llp{4.4cm}rrr",
    ["Agent", "$n$", "Assigned to", "Top-1 cosine", "Margin", "Library $z$"], rows,
    note="Class-level accuracy under this condition is 0 by construction. The workflow "
         "does not signal the absence: the similarities and margins it returns fall inside "
         "the ranges it returns when the answer is present, which is why no threshold on "
         "them separates the two conditions without error (Table~\\ref{tab:si-absten})."))

print("\ntotal", len(made), "tables")
for f in made[7:]: print("  ", f)

# ------------------------- sessionsim : measurement reproducibility
CMx = pd.read_csv(f"{DATA}/v3_measured_cosine_matrix.csv").rename(
    columns={"Unnamed: 0": "agent"}).set_index("agent")
ACQ = PRE["acq_sep"]
_CLS = dict(GD="N", GF="N", VX="N", HD="B", HN1="B", HN2="B", HN3="B",
            A230="V", A232="V", A234="V", A242="V")
import itertools as _it
_n = {a: int(S.loc[a, "n_genuine"]) for a in S.index}
_na = sum(v * (v - 1) // 2 for v in _n.values())
_nc = sum(_n[x] * _n[y] for x, y in _it.combinations(sorted(_n), 2) if _CLS[x] == _CLS[y])
_nd = sum(_n[x] * _n[y] for x, y in _it.combinations(sorted(_n), 2) if _CLS[x] != _CLS[y])
rows = [["Same agent, different acquisition", str(_na), f"{ACQ['same_agent']:.3f}"],
        ["Same class, different agent", str(_nc), f"{ACQ['same_class']:.3f}"],
        ["Different class", str(_nd), f"{ACQ['diff_class']:.3f}"]]
made.append(table(
    "sessionsim.tex",
    "Measurement repeatability. Pairwise cosine similarities between genuine acquisitions, "
    "partitioned by whether the pair is the same agent, two agents of the same "
    "toxicological class, or two classes.",
    "tab:si-sessionsim", "lrr",
    ["Pair type", "$n$", "Mean cosine"], rows,
    note="Computed on the 32 genuine acquisitions only; the derived copies present in the "
         "delivered files were excluded, which the previous version of this analysis did "
         f"not do. The three groups are the {rows[0][1]}, {rows[1][1]} and {rows[2][1]} "
         "pairwise comparisons that 32 acquisitions of eleven vials admit, so they are not "
         "independent observations and we quote no test on them; the separation is larger "
         "than any plausible dependence correction and is reported as an effect size. The "
         "spectra are determined by the material rather than by the session. All "
         "acquisitions were taken within a single hour on "
         "one instrument, so day-to-day reproducibility remains untested. "
         "Table~\\ref{tab:si-repeat} gives the same-vial column per vial, which is the "
         "form in which it should be read."))

# ------------------------- loo : leave-one-molecule-out on the measured library
LOO = pd.read_csv(f"{DATA}/v3_measured_only_loo.csv")
rows = []
for a in ORDER:
    sL = LOO[LOO.agent == a]
    if not len(sL): continue
    tops = sL.top1.value_counts()
    dest = ", ".join(f"{nm(k)}" + (f" ($\\times${v})" if v > 1 else "")
                     for k, v in tops.items())
    rows.append([nm(a), str(len(sL)),
                 f"{int(sL.n_before.iloc[0])} $\\rightarrow$ {int(sL.n_after.iloc[0])}",
                 dest, f"{int(sL.cls_ok.sum())}/{len(sL)}",
                 f"{sL.top_cos.mean():.3f}", f"{sL.margin.mean():.3f}"])
_LX = LOO[LOO.agent != "HN2"]
rows.append(["\\textbf{All}", f"\\textbf{{{len(LOO)}}}", "",
             "", f"\\textbf{{{int(LOO.cls_ok.sum())}/{len(LOO)}}}",
             f"\\textbf{{{LOO.top_cos.mean():.3f}}}",
             f"\\textbf{{{LOO.margin.mean():.3f}}}"])
rows.append(["\\textbf{All, HN2 excluded}", f"\\textbf{{{len(_LX)}}}", "",
             "", f"\\textbf{{{int(_LX.cls_ok.sum())}/{len(_LX)}}}",
             f"\\textbf{{{_LX.top_cos.mean():.3f}}}",
             f"\\textbf{{{_LX.margin.mean():.3f}}}"])
made.append(table(
    "loo.tex",
    "Leave-one-molecule-out against the measured library. Each agent's own reference "
    "spectra were deleted before matching, so the query must fall to a different molecule.",
    "tab:si-loo", "llrp{3.6cm}rrr",
    ["Agent", "$n$", "Library", "Falls to", "Class kept", "Top-1 cosine", "Margin"], rows,
    note="The \\emph{Library} column reports the number of library members before and after "
         "the deletion, so that the ablation is verifiable from the table itself; no query "
         "is assigned to its own molecule. Every query except HN2 falls to a member of its "
         "own toxicological class, and the similarity to that neighbor "
         f"({_LX.top_cos.mean():.2f} on average) is far below the within-molecule value of "
         "0.98. This is the measured-library counterpart of the class-absent test: deleting "
         "one molecule leaves its congeners and the assignment degrades gracefully, whereas "
         "deleting the entire class does not (Table~\\ref{tab:si-loco}). HN2 falls to a "
         "Novichok agent, consistent with the anomaly reported in Section~S5."))

# ------------------------- nov : mutual similarity within the A-series (measured vs computed)
AS = pd.read_csv(f"{DATA}/v3_aseries_margins.csv")
Hs = H.groupby("agent")
rows = []
for _, r in AS.iterrows():
    a = r.agent
    s = Hs.get_group(a) if a in Hs.groups else None
    rows.append([nm(a),
                 f"{s.meas_self.mean():.3f}" if s is not None else "---",
                 f"{r.meas_sibling_max:.3f}",
                 f"{s.dft_self.mean():.3f}" if s is not None else "---",
                 f"{r.dft_sibling_max:.3f}"])
made.append(table(
    "nov.tex",
    "The four Novichok agents compared with one another and with themselves, in each "
    "library. \\emph{Sibling} is the highest similarity to another A-series agent.",
    "tab:si-nov", "lrrrr",
    ["Agent", "Measured self", "Measured sibling", "Computed self", "Computed sibling"],
    rows,
    note="The mutual similarities are comparable in the two libraries "
         f"({AS.meas_sibling_max.min():.2f}--{AS.meas_sibling_max.max():.2f} measured "
         f"against {AS.dft_sibling_max.min():.2f}--{AS.dft_sibling_max.max():.2f} "
         "computed), while the similarity to the correct molecule is not. The four agents "
         "share the P(=O)(F)--N=C amidine core and differ only in the phosphorus "
         "substituent and, for A-242, in guanidine substitution. The computed library "
         "therefore fails on them not because the congeners are too much alike but "
         "because the correct entry is not drawn well enough."))

# ------------------------- bands / bandlist : paired bands
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, measured_peaks,
                                 JOB, CUT, AGENTS)
X, acq = load_measured(); lib = load_library()
mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
SL, SH, TOL = META["s_low"], META["s_high"], 35.0
from scipy.optimize import linear_sum_assignment
pairs = []; census = []
for a in ORDER:
    job = JOB.get(a)
    if job is None or job not in lib: continue
    pk = lib[job]
    f = np.where(pk[:, 0] < CUT, pk[:, 0] * SL, pk[:, 0] * SH)
    keep = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max())
    f = f[keep]
    w_all = np.asarray(mpk[a], float)
    if not len(f) or not len(w_all): continue
    # one-to-one assignment (a linear assignment problem). Nearest-neighbor matching is not used:
    # several measured bands would claim the same computed band and residuals of opposite sign would cancel (same criterion as Analysis in the main text).
    C = np.abs(w_all[:, None] - f[None, :])
    C = np.where(C > TOL, 1e6, C)
    ri, ci = linear_sum_assignment(C)
    mi, cj = set(), set()
    for i, j in zip(ri, ci):
        if C[i, j] < 1e6:
            pairs.append((a, float(w_all[i]), float(f[j]), float(f[j] - w_all[i])))
            mi.add(i); cj.add(j)
    census.append((a, len(w_all), len(f), len(mi), len(w_all) - len(mi), len(f) - len(cj)))
PD = pd.DataFrame(pairs, columns=["agent", "measured", "computed", "resid"])
PD.to_csv(f"{DATA}/v3_band_pairs.csv", index=False)
CEN = pd.DataFrame(census, columns=["agent", "n_meas", "n_calc", "n_pair", "unp_meas", "unp_calc"])
CEN.to_csv(f"{DATA}/v3_band_census.csv", index=False)
rows = []
for a in ORDER:
    s = PD[PD.agent == a]
    if not len(s): continue
    rows.append([nm(a), str(len(s)),
                 f"{np.sqrt((s.resid**2).mean()):.1f}", f"{s.resid.mean():+.1f}",
                 f"{s.resid.min():+.0f}", f"{s.resid.max():+.0f}"])
rows.append(["\\textbf{All}", f"\\textbf{{{len(PD)}}}",
             f"\\textbf{{{np.sqrt((PD.resid**2).mean()):.1f}}}",
             f"\\textbf{{{PD.resid.mean():+.1f}}}",
             f"\\textbf{{{PD.resid.min():+.0f}}}", f"\\textbf{{{PD.resid.max():+.0f}}}"])
made.append(table(
    "bands.tex",
    "Agreement between calibrated computed and measured band positions, the quantity the "
    "frequency scale was fitted to. Measured and computed maxima are paired by one-to-one "
    "assignment, solved as a linear assignment problem on the matrix of absolute wavenumber "
    "differences with pairings beyond 35\\,\\wn{} forbidden; computed bands below 2\\% of the "
    "molecule's strongest activity are not considered. Nearest-neighbor matching is not used, "
    "because it allows one computed band to be claimed by several measured bands with residuals "
    "of opposite sign that cancel.",
    "tab:si-bands", "lrrrrr",
    ["Agent", "Pairs", "RMSE (\\wn)", "Mean residual", "Min", "Max"], rows,
    note=f"The global RMSE of {np.sqrt((PD.resid**2).mean()):.1f}\\,\\wn{{}} is "
         f"approximately {np.sqrt((PD.resid**2).mean())/PRE['grid_step_mean']:.1f} pixels "
         "on the native measurement grid, whose mean spacing in this region is "
         f"{PRE['grid_step_mean']:.2f}\\,\\wn{{}}. The full list of pairs is given in "
         "Table~\\ref{tab:si-bandlist}."))

half = (len(PD) + 1) // 2
rows = []
for i in range(half):
    L = PD.iloc[i]
    left = [nm(L.agent), f"{L.measured:.0f}", f"{L.computed:.0f}", f"{L.resid:+.0f}"]
    if i + half < len(PD):
        R = PD.iloc[i + half]
        right = [nm(R.agent), f"{R.measured:.0f}", f"{R.computed:.0f}", f"{R.resid:+.0f}"]
    else:
        right = ["", "", "", ""]
    rows.append(left + right)
made.append(table(
    "bandlist.tex",
    f"All {len(PD)} matched band pairs used for the frequency calibration. Positions in "
    "\\wn{}; the residual is calibrated computed minus measured.",
    "tab:si-bandlist", "lrrr@{\\hskip 2em}lrrr",
    ["Agent", "Meas.", "Comp.", "$\\Delta$", "Agent", "Meas.", "Comp.", "$\\Delta$"],
    rows, note=None))

# ------------------------- census : pairing register (T1-14)
rows = []
for _, r in CEN.iterrows():
    rows.append([nm(r.agent), str(int(r.n_meas)), str(int(r.n_calc)), str(int(r.n_pair)),
                 str(int(r.unp_meas)), str(int(r.unp_calc))])
rows.append(["\\textbf{All}", f"\\textbf{{{int(CEN.n_meas.sum())}}}",
             f"\\textbf{{{int(CEN.n_calc.sum())}}}", f"\\textbf{{{int(CEN.n_pair.sum())}}}",
             f"\\textbf{{{int(CEN.unp_meas.sum())}}}", f"\\textbf{{{int(CEN.unp_calc.sum())}}}"])
_pen = np.sqrt(np.concatenate([PD.resid.values**2,
                               np.full(int(CEN.unp_meas.sum() + CEN.unp_calc.sum()), TOL**2)]).mean())
made.append(table(
    "census.tex",
    "Complete pairing census for the frequency calibration. Bands that find no partner within "
    "the 35\\,\\wn{} tolerance are counted rather than discarded silently.",
    "tab:si-census", "lrrrrr",
    ["Agent", "Measured bands", "Computed bands", "Paired", "Unpaired meas.", "Unpaired comp."],
    rows,
    note=f"The positional RMSE quoted in the text is computed over the {int(CEN.n_pair.sum())} "
         "paired bands only, which by construction excludes the worst-matched bands and is "
         f"therefore optimistic. Charging every unpaired band the full tolerance as a penalty "
         f"gives {_pen:.1f}\\,\\wn{{}} instead of "
         f"{np.sqrt((PD.resid**2).mean()):.1f}\\,\\wn{{}}. Both are reported so that the "
         "effect of the tolerance gate is visible."))

# ------------------------- theory : the ten B2 pairs and the band shifts
BE = pd.read_csv(f"{DATA}/v3_b2_evaluation.csv")
VN = {"V1_cpcm": "CPCM", "V2_tightP": "tight P/F/S/Cl", "V3_wb97xd3": "$\\omega$B97X-D3"}
BE = BE.assign(delta=BE["new_600–850"] - BE["base_600–850"]).sort_values("delta",
                                                                        ascending=False)
rows = []
for _, r in BE.iterrows():
    d = r.delta
    b = (lambda s: f"\\textbf{{{s}}}") if abs(d) > 0.10 else (lambda s: s)
    rows.append([nm(r.agent), VN.get(r.variant, esc(r.variant)),
                 f"{r['base_600–850']:.3f}", b(f"{r['new_600–850']:.3f}"),
                 b(f"{d:+.3f}"), f"{r.self_base:.3f}", f"{r.self_new:.3f}",
                 ("\\checkmark" if r.correct_base else "$\\times$") + " $\\rightarrow$ " +
                 ("\\checkmark" if r.correct_new else "$\\times$")])
_d = BE["new_600\u2013850"] - BE["base_600\u2013850"]
_theory_note = (
    f"Of the {len(BE)} pairs, {int((_d > 0.10).sum())} improved the window substantially "
    f"($\\Delta > 0.10$), {int(((_d > 0.03) & (_d <= 0.10)).sum())} improved modestly, "
    f"{int((_d.abs() <= 0.03).sum())} were unchanged and {int((_d < -0.03).sum())} degraded. "
    f"Across all {len(BE)} the refitted scale factors stay within "
    f"{BE.s_low.min():.3f}--{BE.s_low.max():.3f} and {BE.s_high.min():.3f}--{BE.s_high.max():.3f}, "
    f"and the global band-position RMSE between {BE.pos_rmse.min():.1f} and "
    f"{BE.pos_rmse.max():.1f}\\,\\wn{{}}, so no global adjustment reaches these errors.")
made.append(table(
    "theory.tex",
    "Ten agent--correction trials. Each improved calculation was swapped into the library "
    "and the frequency scale refitted independently, so that no improvement is understated "
    "by an inherited calibration. All ten trials were computed from the pre-correction "
    "geometries, and the baseline column is therefore the pre-correction library; the "
    "geometry correction reported in Section~S9 is a separate effect and the two are not "
    "additive.",
    "tab:si-theory", "llrrrrrc",
    ["Agent", "Correction", "600--850 base", "Improved", "$\\Delta$",
     "Self base", "Self new", "Top-1"], rows,
    note=_theory_note))

BS = pd.read_csv(f"{DATA}/v3_b2_bandshift.csv")
rows = [[nm(r["agent"]), f'{r["measured"]:.0f}', f'{r["base"]:.0f}',
         f'{r["err_base"]:+.0f}', f'{r["tightp"]:.0f}', f'{r["err_new"]:+.0f}',
         f'{r["shift"]:+.0f}'] for _, r in BS.iterrows()]
made.append(table(
    "bandshift.tex",
    "Position of the dominant 600--850\\,\\wn{} band before and after promoting phosphorus, "
    "fluorine, sulfur and chlorine to a polarized triple-zeta basis.",
    "tab:si-bandshift", "lrrrrrr",
    ["Agent", "Measured", "Baseline", "Error", "tight P", "Error", "Shift"], rows,
    note=f"The shift is upward in {int((BS['shift'] > 0).sum())} "
         f"of the {len(BS)} molecules, by {BS.loc[BS['shift'] > 0, 'shift'].min():.0f} to "
         f"{BS['shift'].max():.0f}\\,\\wn{{}}, and downward by "
         f"{abs(BS['shift'].min()):.0f}\\,\\wn{{}} in the remaining one, while the baseline errors "
         f"scatter from ${'-' if BS.err_base.min() < 0 else '+'}${abs(BS.err_base.min()):.0f} to "
         f"$+${BS.err_base.max():.0f}\\,\\wn{{}}. A correction of fixed sign can only help the subset "
         "whose error is negative and of comparable size. GF satisfies that condition; GD, which "
         f"differs only in the ester substituent and was already within "
         f"{abs(BS.loc[BS.agent == 'GD', 'err_base'].iloc[0]):.0f}\\,\\wn{{}}, does not."))

print("\nfinal", len(made), "tables")

# ------------------------- calibgrid : the circular alternative (kept in print at the editor's request)
try:
    CG = pd.read_csv(f"{DATA}/v3_calibgrid.csv", index_col=0)
    CGM = json.load(open(f"{DATA}/v3_calibgrid_meta.json"))
    scols = [c for c in CG.columns if 0.976 <= float(c) <= 1.036]
    grows = [i for i in CG.index if 12 <= int(i) <= 34]
    rows = []
    for g in grows:
        cells = []
        for c in scols:
            v = int(CG.loc[g, c])
            cells.append(f"\\textbf{{{v}}}" if v == CGM["best_n"] else str(v))
        rows.append([str(g)] + cells)
    _gi = min(CG.index, key=lambda i: abs(int(i) - int(CGM["best_gamma"])))
    _sc = sorted(float(c) for c in CG.columns)
    _s0 = float(CGM["best_s"])
    def _near(t):
        return min(CG.columns, key=lambda c: abs(float(c) - t))
    _CGdn = int(CG.loc[_gi, _near(_s0 * 0.996)])
    _CGup = int(CG.loc[_gi, _near(_s0 * 1.004)])
    made.append(table(
        "calibgrid.tex",
        "The rejected alternative. Each entry is the number of the "
        f"{CGM['n_queries']} queries identified correctly at the molecular level against "
        f"the full {_LIBN}-member library, when a single frequency scale factor $s$ and the "
        "linewidth $\\Gamma$ are chosen by maximizing that number. This is not the "
        "calibration used.",
        "tab:si-calibgrid", "l" + "r" * len(scols),
        ["$\\Gamma$ (\\wn)"] + [c for c in scols], rows,
        note=f"The maximum of {CGM['best_n']} of {CGM['n_queries']} is reached at "
             f"$s$ = {CGM['best_s']:.3f}, $\\Gamma$ = {CGM['best_gamma']}\\,\\wn{{}}, in "
             f"{CGM['n_cells']} of the {CGM['n_total']} cells searched. Moving $s$ by "
             f"0.4\\% in either direction reduces it, to {_CGdn} downward and {_CGup} upward. The "
             f"position-fitted calibration adopted in this work gives {CGM['adopted_n']} "
             f"of {CGM['n_queries']} against the same library, so the circular route buys "
             f"{'nothing' if CGM['adopted_n'] >= CGM['best_n'] else str(CGM['best_n'] - CGM['adopted_n']) + ' queries'}. "
             "Its optimum is in any case a training score of a two-parameter fit, and nothing "
             "in the resulting figures would signal the problem."))
    print("  calibgrid.tex")
except FileNotFoundError:
    print("  (no calibgrid data; run v3_calibgrid.py first)")


# ══════════════════════════════════════════════════════════════════
#  tables added in v3.1 in response to review (T1-6/8/9/10/14b, T1-18, T1-19)
# ══════════════════════════════════════════════════════════════════
WIDE |= {"metrics.tex", "detect.tex", "decomp.tex", "agentstat.tex", "pfmult.tex",
         "aucci.tex", "unmatched.tex"}
new = []

# -- detect : band-detection criterion vs tolerance-window sensitivity
DS = pd.read_csv(f"{DATA}/v3_detect_sensitivity.csv")
rows = [[r.criterion, f"{r.tol:.0f}", f"{r.s_low:.3f}", f"{r.s_high:.3f}",
         f"{r.rmse:.1f}", str(int(r.n_pair)), str(int(r.n_meas)), str(int(r.n_calc)),
         f"{r.rmse_pen:.1f}"] for _, r in DS.iterrows()]
new.append(table(
    "detect.tex",
    "Sensitivity of the frequency calibration to the band-detection criterion and "
    "to the pairing tolerance. The criterion in the first row is the one used throughout this "
    "work. The last criterion scales the threshold to each spectrum's own noise, because the "
    "measured spectra differ in noise by more than an order of magnitude and a single absolute "
    "prominence therefore admits many more peaks from the noisiest of them; it removes 57 of "
    "the 176 detected bands and gives the lowest penalized RMSE of the four, while leaving the "
    "scale factors where they are.",
    "tab:si-detect", "lrrrrrrrr",
    ["Detection criterion", "Tol.\\ (\\wn)", "$s_\\mathrm{low}$", "$s_\\mathrm{high}$",
     "RMSE (\\wn)", "Paired", "Meas.\\ bands", "Comp.\\ bands", "Penalized RMSE"], rows,
    note=f"The scale factors move by at most "
         f"{max((DS.s_low.max()-DS.s_low.min())/DS.s_low.mean(), (DS.s_high.max()-DS.s_high.min())/DS.s_high.mean())*100:.1f}\\% "
         f"and the paired RMSE stays between {DS.rmse.min():.1f} and {DS.rmse.max():.1f}\\,\\wn{{}} "
         "across all six combinations, so neither the detection threshold nor the tolerance "
         "drives the calibration. The penalized RMSE charges every unpaired band a fixed "
         "35\\,\\wn{} so that the six cells remain comparable; it is stable near "
         f"{DS.rmse_pen.mean():.0f}\\,\\wn{{}} and is the honest upper bound on the positional "
         "disagreement, the paired RMSE being the value over bands that found a partner."))

# -- agentstat : per-agent accuracy and McNemar
AA = pd.read_csv(f"{DATA}/v3_agent_accuracy.csv")
QN = {"meas_mol": "Measured library, molecular", "dft_mol": f"Computed library, molecular ({_CLOSN}-member)",
      "dft53_mol": f"Computed library, molecular ({_LIBN}-member)", "meas_cls": "Measured library, class",
      "dft_cls": "Computed library, class"}
AM = json.load(open(f"{DATA}/v3_agent_stats_meta.json"))
SUBN = {"HN2 included (9 agents)": "All nine vials",
        "HN2 excluded (8 agents)": "Without the HN2 vial (eight)"}
rows = []
for sub in AA.subset.unique():
    for _, r in AA[AA.subset == sub].iterrows():
        rows.append([SUBN.get(sub, sub), QN.get(r.quantity, r.quantity),
                     f"{int(r.k)}/{int(r.n)}", f"[{r.lo:.2f}, {r.hi:.2f}]"])
_k9 = AM["HN2 included (9 agents)"]; _k8 = AM["HN2 excluded (8 agents)"]
new.append(table(
    "agentstat.tex",
    "Accuracy with the agent, not the acquisition, as the unit of analysis. Each agent is "
    "scored by the majority of its acquisitions; intervals are Clopper--Pearson.",
    "tab:si-agentstat", "lllc",
    ["Subset", "Quantity", "Agents correct", "95\\% CI"], rows,
    note="The thirty acquisitions are repeated interrogations of nine sealed vials recorded "
         "within a single hour, so the effective sample size is the number of agents. On that "
         f"unit the paired comparison between the two libraries gives, excluding HN2, "
         f"$b = {_k8['mcnemar_mol_b']}$ discordant pairs in favor of the measured library and "
         f"$c = {_k8['mcnemar_mol_c']}$ against, an exact two-sided $p = {_k8['mcnemar_mol_p']:.2f}$ at the "
         f"molecular level; including HN2, $b = {_k9['mcnemar_mol_b']}$, $c = {_k9['mcnemar_mol_c']}$, "
         f"$p = {_k9['mcnemar_mol_p']:.3f}$. At the class level, excluding HN2, the two libraries "
         "do not disagree on any agent. The molecular contrast is therefore a consistent "
         "direction on a sample too small to reach conventional significance, and we report "
         "it as such."))

# -- hn2 : with and without HN2
HNE = pd.read_csv(f"{DATA}/v3_hn2_effect.csv").set_index("subset")
FLD = [("Queries", "n", "{:.0f}"), ("Measured, molecular", "meas_mol", "{:.0f}"),
       (f"Computed, molecular ({_CLOSN}-member)", "dft_mol", "{:.0f}"),
       (f"Computed, molecular ({_LIBN}-member)", "dft53_mol", "{:.0f}"),
       ("Measured, class", "meas_cls", "{:.0f}"), (f"Computed, class ({_CLOSN}-member)", "dft_cls", "{:.0f}"),
       (f"Computed, class ({_LIBN}-member)", "dft53_cls", "{:.0f}"),
       ("Measured self-similarity", "meas_self", "{:.3f}"),
       ("Computed self-similarity", "dft_self", "{:.3f}"),
       ("Measured margin", "meas_margin", "{:.3f}"), ("Computed margin", "dft_margin", "{:.3f}")]
rows = [[lab, f.format(HNE.loc["all", k]), f.format(HNE.loc["noHN2", k])] for lab, k, f in FLD]
new.append(table(
    "hn2.tex",
    "Every reported quantity with and without the HN2 vial, whose contents are not "
    "established (Section~S5).",
    "tab:si-hn2", "lrr",
    ["Quantity", "All acquisitions", "HN2 excluded"], rows,
    note=f"Excluding HN2 raises the computed-library self-similarity from "
         f"{HNE.loc['all','dft_self']:.3f} to {HNE.loc['noHN2','dft_self']:.3f}, so the deficit "
         f"against the measured library falls from {HNE.loc['all','meas_self']-HNE.loc['all','dft_self']:.3f} "
         f"to {HNE.loc['noHN2','meas_self']-HNE.loc['noHN2','dft_self']:.3f}, and it removes the "
         "entire class-level difference between the two libraries. The HN2 acquisitions also "
         "carry an atypically large measured margin, which inflates the measured mean. The "
         "main text quotes the full-set values, with the HN2-excluded values given alongside "
         "at each headline; Section~\\ref{si:hn2sens} gives the comparison in full."))

# -- pfmult : multiple comparison of the phosphorus contrast
PM = pd.read_csv(f"{DATA}/v3_pf_multiplicity.csv")
rows = [[r.region.replace("_", " "), f"{r.mean_p:.3f}", f"{r.mean_np:.3f}", f"{r.delta:+.3f}",
         f"{r.p_two:.3f}", f"{r.p_bonf:.3f}", f"{r.p_holm:.3f}"] for _, r in PM.iterrows()]
_SPELL = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
          8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
_PFLOOR = 2 / math.comb(int(PM.n_p.iloc[0]) + int(PM.n_np.iloc[0]), int(PM.n_np.iloc[0]))
_PFCLAUSE = (" and is precisely the value reached in the 1400--1800\\,\\wn{} window"
             if abs(PM.p_two.min() - _PFLOOR) < 5e-4 else "")
new.append(table(
    "pfmult.tex",
    "The phosphorus contrast in each of the five windows, with the multiplicity of testing "
    "five windows made explicit. Two-sided Mann--Whitney on seven phosphorus-containing "
    "against three phosphorus-free agents.",
    "tab:si-pfmult", "lrrrrrr",
    ["Window (\\wn)", "P-containing", "No P", "$\\Delta$", "$p$", "Bonferroni", "Holm"], rows,
    note=f"No window survives correction. The smallest uncorrected $p$ is not the "
         f"600--850\\,\\wn{{}} window but 1400--1800\\,\\wn{{}}, where the ordering is reversed. "
         f"With {_SPELL.get(int(PM.n_p.iloc[0]), int(PM.n_p.iloc[0]))} against "
         f"{_SPELL.get(int(PM.n_np.iloc[0]), int(PM.n_np.iloc[0]))} molecules the smallest "
         f"two-sided $p$ the design can produce is {_PFLOOR:.3f}{_PFCLAUSE}, "
         f"so this test has essentially no power and we do not quote "
         "a $p$ value in support of the localization. The argument rests on the effect size, "
         f"$\\Delta = {PM.delta.max():+.3f}$ in the 600--850\\,\\wn{{}} window against "
         f"$|\\Delta| \\leq {PM.delta.drop(PM.delta.idxmax()).abs().max():.3f}$ elsewhere, and on "
         "the sign reversal in all four other windows."))

# -- aucci : agent-level confidence interval for the AUC
AC = pd.read_csv(f"{DATA}/v3_auc_ci.csv")
SN = {"S1_top1": "top-1 cosine", "S2_margin": "margin", "S3_psr": "peak-to-sidelobe",
      "S4_libz": "library-normalized $z$", "S5_resid": "unmixing residual",
      "S6_bandfrac": "band-match fraction"}
piv = AC.pivot(index="score", columns="library")
rows = []
for sc in ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]:
    r = []
    for libt in ["measured", "computed"]:
        m = AC[(AC.score == sc) & (AC.library == libt)]
        r += ([f"{m.auc.iloc[0]:.3f}", f"[{m.lo.iloc[0]:.3f}, {m.hi.iloc[0]:.3f}]"]
              if len(m) else ["---", "---"])
    rows.append([SN.get(sc, sc)] + r)
new.append(table(
    "aucci.tex",
    "Separation between the class-present and the class-absent condition, with the agent as "
    "the resampling unit. Intervals are 5000 bootstrap replicates over agents.",
    "tab:si-aucci", "lcccc",
    ["Score", "Measured AUC", "95\\% CI", "Computed AUC", "95\\% CI"], rows,
    note="Resampling agents rather than acquisitions does not change the point estimates, "
         "which are rank statistics over all pairs; it widens the intervals, by removing "
         "the threefold inflation that "
         "treating the acquisitions as independent would introduce. On the measured "
         "library three scores separate the two conditions completely and the bootstrap never "
         "breaks that separation. On the computed library the best score reaches "
         f"{AC[AC.library=='computed'].auc.max():.3f} and no score separates the two conditions "
         "without error on the observed data; the intervals are wide, so this is a statement about what nine "
         "vials can establish and not a proof that no score could separate them. The vial "
         "labeled HN2 is a condition of its own in Section~S7 and is excluded here too, "
         "so that the interval and the point estimate describe the same quantity."))

# -- metrics : T1-18 comparison of similarity metrics
MT = pd.read_csv(f"{DATA}/v3_t118_metrics.csv")
# The measured-library control is computed by v3_t118_metrics.py under the identical code
# and read from its output, so this table cannot drift from the analysis it reports.
MEAS_CTRL = {r.metric: (int(r.meas_mol), int(r.meas_cls), float(r.meas_self))
             for r in pd.read_csv(f"{DATA}/v3_t118_metrics.csv").itertuples()
             if not pd.isna(r.meas_mol)}
ENG = {"cosine, Γ=22": "Full-profile cosine, $\\Gamma$ = 22\\,\\wn{} (adopted)",
       "cosine, Γ=30": "Full-profile cosine, $\\Gamma$ = 30\\,\\wn{}",
       "cosine, Γ=40": "Full-profile cosine, $\\Gamma$ = 40\\,\\wn{}",
       "cosine, Γ=50": "Full-profile cosine, $\\Gamma$ = 50\\,\\wn{}",
       "cosine, Γ=per-agent": "Full-profile cosine, per-agent measured $\\Gamma$",
       "first-derivative cosine, Γ=22": "First-derivative cosine",
       "shift-tolerant cosine ±3px, Γ=22": "Shift-tolerant cosine, $\\pm$3 pixels ($\\pm$17\\,\\wn{})",
       "peak-list HQI, tol=10": "Peak-list hit quality, 10\\,\\wn{} tolerance",
       "peak-list HQI, tol=20": "Peak-list hit quality, 20\\,\\wn{} tolerance",
       "peak-list HQI, tol=35": "Peak-list hit quality, 35\\,\\wn{} tolerance",
       "band-position Dice, tol=20": "Band-position Dice, 20\\,\\wn{} tolerance",
       "band-position Dice, tol=35": "Band-position Dice, 35\\,\\wn{} tolerance"}
rows = []
for _, r in MT.iterrows():
    mc = MEAS_CTRL.get(r.metric)
    rows.append([ENG.get(r.metric, r.metric),
                 f"{mc[0]}/30" if mc else "n/a", f"{mc[2]:.3f}" if mc else "n/a",
                 f"{int(r.mol_closed)}/30", f"{int(r.mol_open)}/30",
                 f"{int(r.agent_closed)}/9", f"{r.self_mean:.3f}"])
_MTc = MT.copy()
_MT_adopt = f"{int(_MTc[_MTc.metric.str.startswith('cosine, Γ=22')].mol_closed.iloc[0])} of 30"
_bi = _MTc.mol_closed.idxmax()
_MT_best = f"{int(_MTc.loc[_bi,'mol_closed'])} of 30"
_MT_bestname = ENG.get(_MTc.loc[_bi,'metric'], _MTc.loc[_bi,'metric']).split(',')[0].rstrip()
_pl = _MTc[_MTc.metric.str.contains('HQI|Dice', na=False)]
_MT_plmin, _MT_plmax = int(_pl.mol_closed.min()), int(_pl.mol_closed.max())
_mv = [v[0] for v in MEAS_CTRL.values()]
_MT_meas_lo, _MT_meas_hi = int(min(_mv)), int(max(_mv))
new.append(table(
    "metrics.tex",
    "The same thirty queries scored with twelve different similarity metrics. The measured "
    "library is carried through the identical code as a control, so that a metric that "
    "performs poorly on the computed library can be distinguished from a metric that is "
    "simply badly implemented.",
    "tab:si-metrics", "lccccc" + "c",
    ["Metric", "Meas.\\ top-1", "Meas.\\ self", f"Comp.\\ top-1 ({_CLOSN})",
     f"Comp.\\ top-1 ({_LIBN})", "Comp.\\ by agent", "Comp.\\ self"], rows,
    note=f"No metric rescues the computed library. The adopted metric gives "
         f"{_MT_adopt}, the best of the twelve is {_MT_best} ({_MT_bestname}), and every "
         f"peak-list metric, which is what commercial library search actually uses, lies "
         f"between {_MT_plmin} and {_MT_plmax} of 30. The measured library reaches "
         f"{_MT_meas_lo} to {_MT_meas_hi} of 30 under the {_SPELL.get(len(MEAS_CTRL), len(MEAS_CTRL))} metrics for which "
         f"it is defined --- the four linewidth variants are a property of the computed "
         f"synthesis only --- so the peak-list "
         f"implementations are sound and the collapse is a property of the computed library "
         f"and not of the comparison. No metric tested accounts for the result."))

# -- decomp : T1-19 component decomposition
DC = pd.read_csv(f"{DATA}/v3_t119_decompose.csv")
rows = [[nm(r.agent), "$\\bullet$" if r.phos else "$\\circ$",
         f"{int(r.n_pair)}/{int(r.n_calc)}/{int(r.n_meas)}",
         f"{r.BASE:.3f}", f"{r.POS:.3f}", f"{r.INT:.3f}", f"{r.WID:.3f}",
         f"{r.PI:.3f}", f"{r.ALL:.3f}"] for _, r in DC.iterrows()]
for lab, sub in (("\\textbf{All}", DC), ("\\textbf{P-containing}", DC[DC.phos]),
                 ("\\textbf{No P}", DC[~DC.phos])):
    rows.append([lab, "", ""] + [f"\\textbf{{{sub[c].mean():.3f}}}"
                                 for c in ["BASE", "POS", "INT", "WID", "PI", "ALL"]])
new.append(table(
    "decomp.tex",
    "Decomposition of the 600--850\\,\\wn{} deficit. One attribute of the computed line list "
    "at a time is replaced by the measured value and the spectrum resynthesized; the "
    "reference is the measured spectrum itself.",
    "tab:si-decomp", "llcrrrrrr",
    ["Agent", "P", "Paired/comp./meas.", "Baseline", "Positions", "Intensities",
     "Linewidth", "Pos.+int.", "All three"], rows,
    note=f"The baseline column is rebuilt from the lines that fall in and around the window "
         "rather than taken from the full-spectrum synthesis, so it differs from "
         "Table~\\ref{tab:si-regbyagent} by up to 0.03; the comparison of interest is the "
         "change within each row. Replacing the linewidth with the width measured for that agent changes the window "
         f"cosine by {(DC.WID-DC.BASE).mean():+.3f} on average, so the single global "
         "$\\Gamma$ is not the origin of the deficit. Position contributes about twice what "
         f"intensity does among the phosphorus-containing agents "
         f"({(DC[DC.phos].POS-DC[DC.phos].BASE).mean():+.3f} against "
         f"{(DC[DC.phos].INT-DC[DC.phos].BASE).mean():+.3f}). Replacing all three attributes "
         f"leaves the cosine at {DC.ALL.mean():.3f} rather than at unity, so the largest "
         "remaining component is neither position nor intensity nor width but the set of bands "
         "itself (Table~\\ref{tab:si-unmatched})."))

# -- unmatched : intensity fraction in unmatched bands
UM = pd.read_csv(f"{DATA}/v3_t119_unmatched.csv")
rows = [[nm(r.agent), str(int(r.n_c)), str(int(r.n_m)), str(int(r.pair)),
         f"{r.unm_calc*100:.0f}\\%", f"{r.unm_meas*100:.0f}\\%"] for _, r in UM.iterrows()]
rows.append(["\\textbf{Mean}", f"\\textbf{{{UM.n_c.mean():.1f}}}", f"\\textbf{{{UM.n_m.mean():.1f}}}",
             f"\\textbf{{{UM.pair.mean():.1f}}}", f"\\textbf{{{UM.unm_calc.mean()*100:.0f}\\%}}",
             f"\\textbf{{{UM.unm_meas.mean()*100:.0f}\\%}}"])
new.append(table(
    "unmatched.tex",
    "Bands inside the 600--850\\,\\wn{} window that find no partner in the other library "
    "within 35\\,\\wn{}, expressed as a share of the intensity carried in that window.",
    "tab:si-unmatched", "lrrrrr",
    ["Agent", "Computed bands", "Measured bands", "Paired",
     "Computed with no measured partner", "Measured with no computed partner"], rows,
    note="Measured bands almost always find a computed partner: the share is zero for GD, "
         "HN1, HN3, A-230 and A-234. The asymmetry runs the other way. The calculation places "
         f"on average {UM[UM.agent!='HN2'].unm_calc.mean()*100:.0f}\\% of the window intensity "
         "in bands the measurement does not show, and "
         f"{float(UM[UM.agent=='A232'].unm_calc.iloc[0])*100:.0f}\\% for A-232 and "
         f"{float(UM[UM.agent=='GF'].unm_calc.iloc[0])*100:.0f}\\% for GF, two of the "
         "misassigned agents. The deficit in this window is therefore not that the calculation "
         "misplaces the measured bands but that it distributes intensity over a band set the "
         "measurement does not have. The vial labeled HN2 pairs no band at all, so both of its "
         f"shares are 100\\% by construction; including it raises the mean to "
         f"{UM.unm_calc.mean()*100:.0f}\\%, and we quote the figure without it. A-232 is "
         "anomalous in both directions and is discussed separately in Section~S8."))

for f in new: print("  +", f)
print(f"\nadded {len(new)} tables (total {len(made) + len(new)})")
