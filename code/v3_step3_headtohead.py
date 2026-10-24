"""
v3 STEP 3 / A1: head-to-head comparison of the measured library vs the computed (DFT) library.

Design principles (fairness)
  F1  the query, the preprocessing, the grid and the metric are kept completely identical under both conditions. The only difference is the source of the library.
  F2  the query acquisition never remains in the library under either condition.
  F3  the frequency scale s is fitted on 'band positions' only (identification performance not used) → no circularity.
      Γ is fixed at the 22 cm-1 measured in STEP 2 (it is not fitted).
  F4  the library composition (the set of agents) is kept the same under both conditions.
  F5  a nested leave-one-agent-out refit is reported alongside, to rebut circularity in s.
"""
import os, sys, json, itertools, collections
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.spectra import parse_raman
from src.preprocess_v3 import LASER_NM, TEMP_K

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 13, "font.size": 15, "font.family": "DejaVu Sans"})

GAMMA = 22.0          # measured value from STEP 2
CUT = 800.0           # boundary between the low- and high-frequency scale factors (as in v2)
HCK = 1.4388

AGENTS = ["GD", "GF", "VX", "HD", "HN1", "HN2", "HN3", "A230", "A232", "A234", "A242"]
CLASS = {"GD": "Nerve", "GF": "Nerve", "VX": "Nerve",
         "HD": "Blister", "HN1": "Blister", "HN2": "Blister", "HN3": "Blister",
         "A230": "Novichok", "A232": "Novichok", "A234": "Novichok", "A242": "Novichok"}
# measured agent -> name of its ORCA output
JOB = {"GD": "EXP_GD_soman", "GF": "GF_cyclosarin_glob", "HD": "EXP_HD", "VX": "EXP_VX",
       "HN1": "EXP_HN1", "HN2": "EXP_HN2", "HN3": "EXP_HN3",
       "A230": "EXP_A230", "A232": "EXP_A232_Eglob", "A234": "EXP_A234", "A242": "EXP_A242"}
# VX computed 2026-10-23. GF and A-232 were replaced by their global-minimum geometries; the
# superseded entries are excluded from the library.
#
# The level-of-theory trials of Section S8 (Tables S-theory and S-bandshift) were run on the
# pre-correction geometries, as their captions state, so reproducing those two tables needs
# the pre-correction library. Set RF_PRECORR=1 for that; every other analysis uses the
# corrected global-minimum entries, which is the default.
PRECORR = os.environ.get("RF_PRECORR") == "1"
if PRECORR:
    JOB["GF"] = "GF_cyclosarin"
    JOB["A232"] = "EXP_A232"
SUPERSEDED = ("GF_cyclosarin_glob", "EXP_A232_Eglob") if PRECORR else ("GF_cyclosarin", "EXP_A232")


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def scattering_factor(f):
    nu0 = 1e7 / LASER_NM
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = ((nu0 - f) ** 4) / f / (1.0 - np.exp(-HCK * f / TEMP_K))
    return np.nan_to_num(fac, nan=0.0, posinf=0.0)


def synth_native(peaks, grid, s_low, s_high, gamma=GAMMA):
    """DFT lines → continuous spectrum on the measured native grid (no upsampling of the measured data)."""
    f = np.where(peaks[:, 0] < CUT, peaks[:, 0] * s_low, peaks[:, 0] * s_high)
    a = peaks[:, 1].astype(float) * scattering_factor(f)
    if a.max() > 0:
        a = a / a.max()
    g = gamma / 2.0
    I = np.zeros_like(grid, dtype=float)
    for ff, aa in zip(f, a):
        I += aa * g * g / ((grid - ff) ** 2 + g * g)
    return I / (I.max() + 1e-12)


# ---------------------------------------------------------------- load
def load_measured():
    d = np.load(os.path.join(ROOT, "data", "v3_measured.npz"), allow_pickle=True)
    X = d["shift"]
    acq = collections.defaultdict(list)
    for k in d.files:
        if k.startswith("acq_"):
            _, ag, rep = k.split("_")
            acq[ag].append((int(rep), d[k]))
    for a in acq:
        acq[a].sort()
    return X, dict(acq)


def load_library():
    lib = {}
    for f in sorted(os.listdir(os.path.join(ROOT, "orca", "outputs"))):
        if not f.endswith(".out") or f.startswith("GB_sarin_fast") or f[:-4] in SUPERSEDED:
            continue
        lib[f[:-4]] = parse_raman(os.path.join(ROOT, "orca", "outputs", f))
    return lib


# ------------------------------------------- fit of s (non-circular)
def measured_peaks(X, spec, prominence=0.05):
    pk, _ = find_peaks(spec, prominence=prominence, distance=2)
    return X[pk]


def position_rmse(lib, mpk, agents, s_low, s_high, tol=35.0):
    """
    Measured↔computed band position residuals. Paired by **one-to-one assignment** (the linear assignment problem).

    With nearest-neighbor matching several measured bands can claim the same computed
    band, so residuals of opposite sign cancel and the number of pairs is inflated.
    This was corrected during the v2 review, and v3 uses the same criterion.
    """
    d = []
    for ag in agents:
        job = JOB.get(ag)
        if job is None or job not in lib:
            continue
        pk = lib[job]
        f = np.where(pk[:, 0] < CUT, pk[:, 0] * s_low, pk[:, 0] * s_high)
        keep = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max())
        f = f[keep]
        w = np.asarray(mpk[ag], float)
        if not len(f) or not len(w):
            continue
        C = np.abs(w[:, None] - f[None, :])
        C = np.where(C > tol, 1e6, C)
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            if C[i, j] < 1e6:
                d.append(float(f[j] - w[i]))
    d = np.asarray(d)
    return (float(np.sqrt((d ** 2).mean())) if len(d) else np.nan), len(d)


def fit_scale(lib, mpk, agents, min_pairs=60):
    best = None
    for sl in np.arange(0.960, 1.0401, 0.002):
        for sh in np.arange(0.950, 1.0201, 0.002):
            r, n = position_rmse(lib, mpk, agents, float(sl), float(sh))
            if n >= min_pairs and (best is None or r < best[0]):
                best = (r, float(sl), float(sh), n)
    return best


# -------------------------------------------------------- identification
JOB2AG = {j: a for a, j in JOB.items()}


def cls_of(name):
    """Returns the class whether the name is an agent code or an ORCA job name.
    The OP-F grid distractors are all of the sarin family = Nerve."""
    if name in CLASS:
        return CLASS[name]
    if name in JOB2AG:
        return CLASS[JOB2AG[name]]
    return "Nerve"


def identify(query, lib_specs, gold):
    sims = {k: cos(query, v) for k, v in lib_specs.items()}
    order = sorted(sims.items(), key=lambda t: -t[1])
    top, ts = order[0]
    runner = order[1] if len(order) > 1 else ("", np.nan)
    return dict(top1=top, top_score=ts, margin=ts - runner[1],
                self_cos=sims.get(gold, np.nan),
                mol_ok=(top == gold),
                cls_ok=(cls_of(top) == cls_of(gold)))


def main():
    X, acq = load_measured()
    lib = load_library()
    print(f"measured acquisitions {sum(len(v) for v in acq.values())} / {len(acq)} agents")
    print(f"DFT library {len(lib)} agents (VX included, GF·A232 global-minimum geometry)\n")

    # ---- mean spectra and band positions ----
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}

    # ---- fit of s: band positions only ----
    calib_agents = [a for a in AGENTS if a in JOB]
    rmse0, SL, SH, npair = fit_scale(lib, mpk, calib_agents)
    print(f"[calibration] s_low={SL:.3f} s_high={SH:.3f}  position RMSE={rmse0:.1f} cm-1 "
          f"({npair} paired bands)  Γ={GAMMA:.0f} fixed")

    # nested leave-one-agent-out (rebuttal to the circularity objection)
    nested = []
    for held in calib_agents:
        sub = [a for a in calib_agents if a != held]
        r, sl, sh, n = fit_scale(lib, mpk, sub)
        nested.append(dict(held_out=held, s_low=sl, s_high=sh, rmse=r, n_pairs=n))
    N = pd.DataFrame(nested)
    print(f"[calibration] nested LOAO: s_low {N.s_low.min():.3f}–{N.s_low.max():.3f} "
          f"(±{(N.s_low.max()-N.s_low.min())/2/SL*100:.2f}%), "
          f"s_high {N.s_high.min():.3f}–{N.s_high.max():.3f} "
          f"(±{(N.s_high.max()-N.s_high.min())/2/SH*100:.2f}%)\n")
    N.to_csv(os.path.join(ROOT, "data", "v3_nested_calibration.csv"), index=False)

    # ---- computed spectra onto the native grid ----
    dft = {k: synth_native(v, X, SL, SH) for k, v in lib.items()}

    # ---- query set: identical under both conditions ----
    queryable = [a for a in AGENTS if len(acq[a]) >= 2]      # a measured self-entry must exist
    queries = [(a, r, s) for a in queryable for r, s in acq[a]]
    print(f"common queries {len(queries)} / {len(queryable)} agents "
          f"(HD·VX have 1 acquisition, excluded from queries, kept in the library)\n")

    core_jobs = [JOB[a] for a in AGENTS if a in JOB]
    rows = []
    for q_ag, q_rep, q in queries:
        # (1) measured library, with the query acquisition removed
        mlib = {}
        for a in AGENTS:
            others = [s for r, s in acq[a] if not (a == q_ag and r == q_rep)]
            if others:
                mlib[a] = np.mean(others, 0)
        rm = identify(q, mlib, q_ag)

        # (2) computed library (closed set: only the computed agents)
        clib = {j: dft[j] for j in core_jobs}
        rc = identify(q, clib, JOB.get(q_ag, "__none__"))

        # (3) computed library (open set: all members)
        ro = identify(q, dft, JOB.get(q_ag, "__none__"))

        rows.append(dict(
            agent=q_ag, rep=q_rep, cls=CLASS[q_ag],
            meas_top1=rm["top1"], meas_mol=rm["mol_ok"], meas_cls=rm["cls_ok"],
            meas_self=rm["self_cos"], meas_margin=rm["margin"],
            dft_top1=rc["top1"], dft_mol=rc["mol_ok"], dft_cls=rc["cls_ok"],
            dft_self=rc["self_cos"], dft_margin=rc["margin"],
            dft53_top1=ro["top1"], dft53_mol=ro["mol_ok"], dft53_cls=ro["cls_ok"]))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(ROOT, "data", "v3_headtohead.csv"), index=False)

    n = len(R)
    print("=" * 68)
    print("A1  measured library vs computed library: same queries, same metrics")
    print("=" * 68)
    print(f"{'condition':<34}{'molecule top-1':>16}{'class':>16}")
    print(f"{'measured library (11 agents)':<32}"
          f"{R.meas_mol.sum():>6}/{n} = {R.meas_mol.mean()*100:5.1f}%"
          f"{R.meas_cls.sum():>8}/{n} = {R.meas_cls.mean()*100:5.1f}%")
    print(f"{f'computed library ({len(JOB)}, closed)':<30}"
          f"{R.dft_mol.sum():>6}/{n} = {R.dft_mol.mean()*100:5.1f}%"
          f"{R.dft_cls.sum():>8}/{n} = {R.dft_cls.mean()*100:5.1f}%")
    print(f"{f'computed library ({len(lib)}, open)':<31}"
          f"{R.dft53_mol.sum():>6}/{n} = {R.dft53_mol.mean()*100:5.1f}%"
          f"{R.dft53_cls.sum():>8}/{n} = {R.dft53_cls.mean()*100:5.1f}%")

    print("\nPer agent (molecule top-1, per acquisition):")
    G = R.groupby("agent").agg(n=("rep", "size"),
                               meas=("meas_mol", "mean"), dft=("dft_mol", "mean"),
                               meas_self=("meas_self", "mean"), dft_self=("dft_self", "mean"),
                               meas_margin=("meas_margin", "mean"),
                               dft_margin=("dft_margin", "mean")).round(3)
    G["meas"] = (G["meas"] * 100).round(0)
    G["dft"] = (G["dft"] * 100).round(0)
    print(G.to_string())

    print("\nMean self-cosine (similarity to the correct entry):")
    print(f"  measured library {R.meas_self.mean():.3f}   computed library {R.dft_self.mean():.3f}"
          f"   difference {R.meas_self.mean()-R.dft_self.mean():+.3f}")
    print("Mean margin (1st − 2nd):")
    print(f"  measured {R.meas_margin.mean():.3f}   computed {R.dft_margin.mean():.3f}")

    bad = R[~R.dft_mol]
    if len(bad):
        print(f"\ncomputed library errors, {len(bad)} cases:")
        for r in bad.itertuples():
            print(f"  {r.agent}#{r.rep} → {r.dft_top1}  "
                  f"(self {r.dft_self:.3f}, top margin {r.dft_margin:+.3f})")

    # ---- A-series detail ----
    NOV = ["A230", "A232", "A234", "A242"]
    print("\nA-series: separation margin from siblings (self − best sibling):")
    mrows = []
    for a in NOV:
        mm = [cos(mean[a], mean[b]) for b in NOV if b != a]
        if JOB[a] in dft:
            dd = [cos(dft[JOB[a]], dft[JOB[b]]) for b in NOV if b != a]
        else:
            dd = [np.nan]
        mrows.append(dict(agent=a,
                          meas_sibling_max=max(mm), dft_sibling_max=max(dd)))
        print(f"  {a}: measured best sibling {max(mm):.3f}   computed best sibling {max(dd):.3f}")
    pd.DataFrame(mrows).to_csv(os.path.join(ROOT, "data", "v3_aseries_margins.csv"), index=False)

    # ---- region decomposition (where the domain gap sits) ----
    REG = [("250–600", 250, 600), ("600–850", 600, 850), ("850–1150", 850, 1150),
           ("1150–1400", 1150, 1400), ("1400–1800", 1400, 1800)]
    ags9 = [a for a in AGENTS if a in JOB and a != "HN2"]
    reg_rows = []
    for name, lo, hi in REG:
        m = (X >= lo) & (X < hi)
        dc = [cos(mean[a][m], dft[JOB[a]][m]) for a in ags9]
        rc = []
        for a in ags9:
            if len(acq[a]) >= 2:
                rc += [cos(u[m], v[m]) for (_, u), (_, v)
                       in itertools.combinations(acq[a], 2)]
        wt = [mean[a][m].sum() / mean[a].sum() for a in ags9]
        reg_rows.append(dict(region=name, dft_cos=float(np.mean(dc)),
                             repro_cos=float(np.mean(rc)), weight=float(np.mean(wt)),
                             gap=float(np.mean(rc) - np.mean(dc))))
    RG = pd.DataFrame(reg_rows)
    RG.to_csv(os.path.join(ROOT, "data", "v3_region_breakdown.csv"), index=False)
    print("\nPer-region domain gap (reproducibility ceiling − measured↔computed):")
    print(RG.round(3).to_string(index=False))

    PF = [a for a in ags9 if a in ("GD", "GF", "VX", "A230", "A232", "A234", "A242")]
    NOP = [a for a in ags9 if a in ("HD", "HN1", "HN3")]
    from scipy.stats import mannwhitneyu
    print("\nP-containing vs P-free, per-region measured↔computed cosine difference (P-free − P-containing):")
    pf_rows = []
    for name, lo, hi in REG:
        m = (X >= lo) & (X < hi)
        av = np.array([cos(mean[a][m], dft[JOB[a]][m]) for a in PF])
        bv = np.array([cos(mean[a][m], dft[JOB[a]][m]) for a in NOP])
        pp = mannwhitneyu(bv, av, alternative="greater").pvalue
        pf_rows.append(dict(region=name, pf=float(av.mean()), nop=float(bv.mean()),
                            delta=float(bv.mean() - av.mean()), p=float(pp)))
        print(f"  {name:<12} P-series {av.mean():.3f}  P-free {bv.mean():.3f}  "
              f"Δ {bv.mean()-av.mean():+.3f}  p={pp:.3f}")
    PFdf = pd.DataFrame(pf_rows)
    PFdf.to_csv(os.path.join(ROOT, "data", "v3_pf_region_split.csv"), index=False)

    meta = dict(s_low=SL, s_high=SH, gamma=GAMMA, pos_rmse=rmse0, n_pairs=npair,
                n_queries=n, n_agents_query=len(queryable),
                n_library=len(lib), n_agents_lib=len(JOB),
                meas_mol=float(R.meas_mol.mean()), meas_cls=float(R.meas_cls.mean()),
                dft_mol=float(R.dft_mol.mean()), dft_cls=float(R.dft_cls.mean()),
                dft53_mol=float(R.dft53_mol.mean()), dft53_cls=float(R.dft53_cls.mean()),
                nested_s_low=[float(N.s_low.min()), float(N.s_low.max())],
                nested_s_high=[float(N.s_high.min()), float(N.s_high.max())],
                meas_self=float(R.meas_self.mean()), dft_self=float(R.dft_self.mean()),
                meas_margin=float(R.meas_margin.mean()),
                dft_margin=float(R.dft_margin.mean()),
                region=RG.to_dict("records"), pf_split=PFdf.to_dict("records"))
    with open(os.path.join(ROOT, "data", "v3_headtohead_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ================================================================ figure
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.40,
                          left=0.06, right=0.965, top=0.93, bottom=0.07)

    # ---------- a ----------
    ax = fig.add_subplot(gs[0, 0])
    mv = [R.meas_mol.mean() * 100, R.meas_cls.mean() * 100]
    dv = [R.dft_mol.mean() * 100, R.dft_cls.mean() * 100]
    xx = np.arange(2); w = 0.36
    ax.bar(xx - w / 2, mv, w, color=P["dark_teal"], label="measured library")
    ax.bar(xx + w / 2, dv, w, color=P["warm_red"], label="computed (DFT) library")
    for i, (u, v) in enumerate(zip(mv, dv)):
        ax.text(i - w / 2, u + 1.5, f"{u:.0f}", ha="center", fontsize=14)
        ax.text(i + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=14)
    ax.set_xticks(xx); ax.set_xticklabels(["molecular", "class"]); ax.set_ylim(0, 112)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title(f"Same {n} queries, same metric", fontsize=16)
    ax.legend(frameon=False, fontsize=11, ncol=1, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.text(-0.17, 1.05, "a", transform=ax.transAxes, fontsize=26, fontweight="bold")

    # ---------- b ----------
    ax = fig.add_subplot(gs[0, 1])
    ags = sorted(G.index, key=lambda a: -G.loc[a, "meas_self"])
    yy = np.arange(len(ags))
    ax.barh(yy - .2, [G.loc[a, "meas_self"] for a in ags], .38,
            color=P["dark_teal"], label="measured")
    ax.barh(yy + .2, [G.loc[a, "dft_self"] for a in ags], .38,
            color=P["warm_red"], label="computed")
    ax.set_yticks(yy); ax.set_yticklabels(ags); ax.invert_yaxis()
    ax.set_xlabel("cosine to the correct entry"); ax.set_xlim(0, 1.0)
    ax.set_title("Similarity to the right answer", fontsize=16)
    ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.text(-0.17, 1.05, "b", transform=ax.transAxes, fontsize=26, fontweight="bold")

    # ---------- c ----------
    ax = fig.add_subplot(gs[0, 2])
    yy = np.arange(len(RG))
    ax.barh(yy + .2, RG.repro_cos, .38, color=P["sage"], label="measurement reproducibility")
    ax.barh(yy - .2, RG.dft_cos, .38, color=P["warm_red"], label="measured vs computed")
    for i, r in RG.iterrows():
        ax.text(1.02, i - .20, f"gap {r.gap:.2f}", va="center", fontsize=10,
                color=P["rose"] if r.gap > .25 else "0.25")
        ax.text(1.02, i + .20, f"{r.weight*100:.0f}% int.", va="center", fontsize=9,
                color="0.45")
    ax.set_yticks(yy); ax.set_yticklabels(RG.region, fontsize=11)
    ax.invert_yaxis(); ax.set_xlim(0, 1.28); ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlabel("cosine"); ax.set_title("Where the computed library loses", fontsize=15)
    ax.legend(frameon=False, fontsize=10, ncol=1, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.text(-0.20, 1.05, "c", transform=ax.transAxes, fontsize=26, fontweight="bold")

    # ---------- d, e : colourbars moved to an inset outside the axes so the axis width matches a and b ----------
    mats = [("measured $\\times$ measured",
             np.array([[cos(mean[i], mean[j]) for j in AGENTS] for i in AGENTS]), "d"),
            ("computed $\\times$ computed",
             np.array([[cos(dft[JOB[i]], dft[JOB[j]]) if (i in JOB and j in JOB) else np.nan
                        for j in AGENTS] for i in AGENTS]), "e")]
    for idx, (title, mat, letter) in enumerate(mats):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(AGENTS)))
        ax.set_xticklabels(AGENTS, rotation=90, fontsize=11)
        ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS, fontsize=11)
        ax.set_title(title, fontsize=15)
        cax = inset_axes(ax, width="3.5%", height="100%", loc="lower left",
                         bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                         borderpad=0)
        plt.colorbar(im, cax=cax)
        cax.tick_params(labelsize=10)
        ax.text(-0.17, 1.05, letter, transform=ax.transAxes, fontsize=26, fontweight="bold")

    # ---------- f : region contrast split by whether the agent contains phosphorus ----------
    ax = fig.add_subplot(gs[1, 2])
    PF = PFdf.reset_index(drop=True)
    yy = np.arange(len(PF))[::-1]
    ax.barh(yy + 0.19, PF["pf"], 0.36, color=P["warm_red"])
    ax.barh(yy - 0.19, PF["nop"], 0.36, color=P["sage"])
    ax.text(0.025, yy[0] + 0.19, "P-containing (n = 6)", va="center", ha="left",
            fontsize=11, color="white", fontweight="bold")
    ax.text(0.025, yy[0] - 0.19, "no phosphorus (n = 3)", va="center", ha="left",
            fontsize=11, color="white", fontweight="bold")
    for i, r in enumerate(PF.itertuples()):
        y0 = yy[i]
        hit = r.region.replace("\u2013", "-") == "600-850"
        ax.text(max(r.pf, r.nop) + 0.02, y0,
                f"{r.delta:+.2f}", va="center", fontsize=11.5,
                fontweight="bold" if hit else "normal",
                color=P["rose"] if hit else "0.35")
    ax.set_yticks(yy)
    ax.set_yticklabels([r.replace("\u2013", "–") for r in PF["region"]], fontsize=11.5)
    ax.set_xlim(0, 1.24); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("measured vs computed cosine")
    ax.set_title("The deficit is specific to phosphorus", fontsize=15)
    ax.text(-0.26, 1.05, "f", transform=ax.transAxes, fontsize=26, fontweight="bold")

    fig.savefig(os.path.join(ROOT, "figure", "v3_fig_headtohead.png"), dpi=200)
    print("\nsaved figure/v3_fig_headtohead.png + data/v3_headtohead.csv, "
          "v3_nested_calibration.csv, v3_aseries_margins.csv, v3_headtohead_meta.json")


if __name__ == "__main__":
    main()
