"""
v3 A2: abstention rule.

Problem: the computed library does not abstain even when the correct class is absent; it
      returns an incorrect answer with confidence. An absolute similarity threshold cannot
      separate the two (REMEDIATION §0(2): LOCO 0.74–0.85 vs correct 0.70–0.91).

Evaluation design
  IN    the full computed library (the correct class is present)   → must not abstain
  LOCO  the class of the query removed from the library entirely   → must abstain
  LOMO  only the molecule of the query removed (the class remains) → the class call stays valid, for reference
  HN2   the sample is not HN2 (self cosine 0.061) = a naturally occurring out-of-library positive control

Candidate scores (all single features; the threshold is selected by nested leave-one-agent-out)
  S1 top1        the highest cosine (baseline, expected to fail)
  S2 margin      first − second
  S3 psr         (first − mean of the rest) / standard deviation of the rest   [peak-to-sidelobe]
  S4 libz        z normalized by the within-library similarity distribution    [local normalization]
  S5 resid       the fraction not explained by a non-negative combination of the top m entries [unmixing residual]
  S6 bandfrac    the fraction of measured bands with a corresponding computed band [physical, non-cosine]
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import nnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, synth_native, fit_scale,
                                 measured_peaks, cos, cls_of, JOB, CLASS, AGENTS, GAMMA)

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 13, "font.size": 15, "font.family": "DejaVu Sans"})

TOL = 35.0        # tolerance for band correspondence (as in A1)
TOPM = 5          # number of top candidates used for unmixing


# ----------------------------------------------------------------------- scores
def library_stats(dft):
    """How much each library entry 'typically resembles the other entries'; for local normalization."""
    keys = list(dft)
    M = np.array([[cos(dft[i], dft[j]) for j in keys] for i in keys])
    np.fill_diagonal(M, np.nan)
    mu = np.nanmean(M, 1)
    sd = np.nanstd(M, 1)
    return {k: (float(mu[i]), float(max(sd[i], 1e-6))) for i, k in enumerate(keys)}


def unmix_residual(q, dft, order, m=TOPM):
    """Fraction of q left unexplained by a non-negative combination of the top m library spectra."""
    A = np.stack([dft[k] for k, _ in order[:m]], 1)
    qn = q / (np.linalg.norm(q) + 1e-12)
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    w, _ = nnls(An, qn)
    return float(np.linalg.norm(qn - An @ w))


def band_fraction(qpk, dpk, tol=TOL):
    """Fraction of measured bands with a computed band within tol."""
    if not len(qpk) or not len(dpk):
        return 0.0
    return float(np.mean([np.min(np.abs(dpk - w)) <= tol for w in qpk]))


def score_query(q, qpk, dft, dftpk, libstat, pool):
    sims = {k: cos(q, dft[k]) for k in pool}
    order = sorted(sims.items(), key=lambda t: -t[1])
    top, t1 = order[0]
    vals = np.array([v for _, v in order])
    others = vals[1:]
    mu_o, sd_o = float(others.mean()), float(max(others.std(), 1e-9))
    mu_k, sd_k = libstat[top]
    return dict(
        top1_name=top,
        S1_top1=t1,
        S2_margin=t1 - (vals[1] if len(vals) > 1 else 0.0),
        S3_psr=(t1 - mu_o) / sd_o,
        S4_libz=(t1 - mu_k) / sd_k,
        S5_resid=unmix_residual(q, dft, order),
        S6_bandfrac=band_fraction(qpk, dftpk[top]),
    )


# ------------------------------------------------------------------- evaluation
SCORES = ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]
# a larger residual means 'should abstain', so the sign is inverted
HIGHER_IS_IN = {"S1_top1": True, "S2_margin": True, "S3_psr": True,
                "S4_libz": True, "S5_resid": False, "S6_bandfrac": True}


def auc(pos, neg):
    """pos = IN (must not abstain), neg = LOCO (must abstain). Higher score means more IN."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    X, acq = load_measured()
    lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    _, SL, SH, _ = fit_scale(lib, mpk, [a for a in AGENTS if a in JOB])
    dft = {k: synth_native(v, X, SL, SH) for k, v in lib.items()}
    dftpk = {k: X[find_peaks(v, prominence=0.05, distance=2)[0]] for k, v in dft.items()}
    libstat = library_stats(dft)
    # Default: the full 54-member computed library, which is the adopted analysis and the
    # basis of the main abstention table. Set RF_LIB11=1 for the candidate-count control
    # added in response to review, which restricts the computed library to the same eleven
    # members as the measured one and so removes the confound of candidate-set size.
    if os.environ.get("RF_LIB11") == "1":
        full = [JOB[a] for a in AGENTS if a in JOB]
    else:
        full = list(dft)
    print(f"computed library {len(full)} agents, calibration s=({SL:.3f},{SH:.3f}), Γ={GAMMA:.0f}\n")

    queryable = [a for a in AGENTS if len(acq[a]) >= 2 and a in JOB]
    rows = []
    for a in queryable:
        for rep, q in acq[a]:
            qpk = X[find_peaks(q, prominence=0.05, distance=2)[0]]
            gold = JOB[a]
            gold_cls = CLASS[a]

            pools = {
                "IN": full,
                "LOMO": [k for k in full if k != gold],
                "LOCO": [k for k in full if cls_of(k) != gold_cls],
            }
            for cond, pool in pools.items():
                if not pool:
                    continue
                sc = score_query(q, qpk, dft, dftpk, libstat, pool)
                rows.append(dict(agent=a, rep=rep, cls=gold_cls, cond=cond,
                                 correct=(sc["top1_name"] == gold),
                                 cls_correct=(cls_of(sc["top1_name"]) == gold_cls),
                                 **sc))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_abstention_scores.csv"), index=False)

    # the HN2 sample is itself out of library, so it belongs on the 'should abstain' side even under the IN condition
    IN = D[(D.cond == "IN") & (D.agent != "HN2")]
    HN2 = D[(D.cond == "IN") & (D.agent == "HN2")]
    LOCO = D[(D.cond == "LOCO") & (D.agent != "HN2")]
    LOMO = D[(D.cond == "LOMO") & (D.agent != "HN2")]

    print(f"IN {len(IN)} / LOCO {len(LOCO)} / LOMO {len(LOMO)} / HN2 {len(HN2)} cases\n")
    print("Fraction of correct top-1 class under LOCO (= silent failure check): "
          f"{LOCO.cls_correct.mean()*100:.0f}%  ← 0% is expected (the whole class was removed)")

    print("\n" + "=" * 74)
    print("Abstention score candidates: separation of IN (must not abstain) vs LOCO (must abstain)")
    print("=" * 74)
    print(f"{'score':<14}{'IN median':>11}{'LOCO median':>13}{'HN2 median':>12}{'AUC':>8}{'separated':>9}")
    res = []
    for s in SCORES:
        sign = 1.0 if HIGHER_IS_IN[s] else -1.0
        a_ = auc(sign * IN[s], sign * LOCO[s])
        sep = (sign * IN[s]).min() > (sign * LOCO[s]).max()
        res.append(dict(score=s, auc=a_, separable=bool(sep),
                        in_med=float(IN[s].median()), loco_med=float(LOCO[s].median()),
                        hn2_med=float(HN2[s].median())))
        print(f"{s:<14}{IN[s].median():>11.3f}{LOCO[s].median():>13.3f}"
              f"{HN2[s].median():>12.3f}{a_:>8.3f}{'  yes' if sep else '  no':>9}")
    RES = pd.DataFrame(res)
    RES.to_csv(os.path.join(ROOT, "data", "v3_abstention_auc.csv"), index=False)

    # -------- operating threshold: nested leave-one-agent-out --------
    print("\n" + "=" * 74)
    print("Operating rule: nested LOAO, threshold set with one agent left out at a time")
    print("  (rule: abstain if score < τ. τ is the smallest value that catches every LOCO case except the held-out one)")
    print("=" * 74)
    oper = []
    for s in SCORES:
        sign = 1.0 if HIGHER_IS_IN[s] else -1.0
        abst_in, abst_loco, abst_hn2, taus = [], [], [], []
        for held in queryable:
            if held == "HN2":
                continue
            tr_loco = LOCO[LOCO.agent != held]
            if not len(tr_loco):
                continue
            tau = (sign * tr_loco[s]).max()          # the boundary that makes every training LOCO case abstain
            taus.append(tau)
            te_in = IN[IN.agent == held]
            te_loco = LOCO[LOCO.agent == held]
            abst_in += list((sign * te_in[s] <= tau).astype(int))
            abst_loco += list((sign * te_loco[s] <= tau).astype(int))
        tau_all = (sign * LOCO[s]).max()
        abst_hn2 = list((sign * HN2[s] <= tau_all).astype(int))
        oper.append(dict(score=s, tau_med=float(np.median(taus)) * sign,
                         abstain_IN=float(np.mean(abst_in)),
                         abstain_LOCO=float(np.mean(abst_loco)),
                         abstain_HN2=float(np.mean(abst_hn2))))
        print(f"  {s:<12} τ≈{np.median(taus)*sign:+8.3f}   "
              f"IN abstain {np.mean(abst_in)*100:5.1f}% (lower is better)   "
              f"LOCO abstain {np.mean(abst_loco)*100:5.1f}% (100% is the target)   "
              f"HN2 abstain {np.mean(abst_hn2)*100:5.0f}%")
    OP = pd.DataFrame(oper)
    OP.to_csv(os.path.join(ROOT, "data", "v3_abstention_operating.csv"), index=False)

    # -------- two-feature AND rule (the two most promising features) --------
    best2 = OP.sort_values(["abstain_LOCO", "abstain_IN"], ascending=[False, True]).score.tolist()[:2]
    print(f"\n2-feature AND rule: accept only if {best2[0]} and {best2[1]} both pass")
    s1, s2 = best2
    g1 = 1.0 if HIGHER_IS_IN[s1] else -1.0
    g2 = 1.0 if HIGHER_IS_IN[s2] else -1.0
    t1 = (g1 * LOCO[s1]).max(); t2 = (g2 * LOCO[s2]).max()
    acc_in = float(np.mean((g1 * IN[s1] > t1) & (g2 * IN[s2] > t2)))
    acc_loco = float(np.mean((g1 * LOCO[s1] > t1) & (g2 * LOCO[s2] > t2)))
    acc_hn2 = float(np.mean((g1 * HN2[s1] > t1) & (g2 * HN2[s2] > t2)))
    print(f"  IN accept {acc_in*100:.1f}%   LOCO accept {acc_loco*100:.1f}%   HN2 accept {acc_hn2*100:.0f}%")

    # ===================================== measured-library control (the core comparison)
    print("\n" + "=" * 74)
    print("The same abstention problem on the 'measured library': what happens when the forward model is accurate")
    print("=" * 74)
    mrows = []
    for a in queryable:
        for rep, q in acq[a]:
            mlib = {}
            for b in AGENTS:
                oth = [s_ for r, s_ in acq[b] if not (b == a and r == rep)]
                if oth:
                    mlib[b] = np.mean(oth, 0)
            if a not in mlib:
                continue
            mpk_ = {k: X[find_peaks(v, prominence=0.05, distance=2)[0]] for k, v in mlib.items()}
            st = library_stats(mlib)
            qpk = X[find_peaks(q, prominence=0.05, distance=2)[0]]
            for cond, pool in [("IN", list(mlib)),
                               ("LOCO", [k for k in mlib if CLASS[k] != CLASS[a]])]:
                if not pool:
                    continue
                sc = score_query(q, qpk, mlib, mpk_, st, pool)
                mrows.append(dict(agent=a, rep=rep, cond=cond,
                                  cls_correct=CLASS[sc["top1_name"]] == CLASS[a], **sc))
    M = pd.DataFrame(mrows)
    M.to_csv(os.path.join(ROOT, "data", "v3_abstention_measured.csv"), index=False)
    MIN_ = M[(M.cond == "IN") & (M.agent != "HN2")]
    MLOCO = M[(M.cond == "LOCO") & (M.agent != "HN2")]
    MHN2 = M[(M.cond == "IN") & (M.agent == "HN2")]

    print(f"{'score':<14}{'comp AUC':>11}{'meas AUC':>11}{'comp separated':>15}{'meas separated':>15}")
    cmp_rows = []
    for s_ in SCORES:
        g = 1.0 if HIGHER_IS_IN[s_] else -1.0
        a_c = auc(g * IN[s_], g * LOCO[s_])
        a_m = auc(g * MIN_[s_], g * MLOCO[s_])
        sc_ = (g * IN[s_]).min() > (g * LOCO[s_]).max()
        sm_ = (g * MIN_[s_]).min() > (g * MLOCO[s_]).max()
        cmp_rows.append(dict(score=s_, auc_computed=a_c, auc_measured=a_m,
                             sep_computed=bool(sc_), sep_measured=bool(sm_)))
        print(f"{s_:<14}{a_c:>11.3f}{a_m:>11.3f}"
              f"{('yes' if sc_ else 'no'):>15}{('yes' if sm_ else 'no'):>15}")
    CMP = pd.DataFrame(cmp_rows)
    CMP.to_csv(os.path.join(ROOT, "data", "v3_abstention_compare.csv"), index=False)

    s_best = "S1_top1"
    tau_m = MLOCO[s_best].max()
    print(f"\nmeasured library + single threshold τ={tau_m:.3f} (S1 top-1 cosine):")
    print(f"  IN accept {(MIN_[s_best] > tau_m).mean()*100:.0f}%   "
          f"LOCO accept {(MLOCO[s_best] > tau_m).mean()*100:.0f}%   "
          f"HN2 accept {(MHN2[s_best] > tau_m).mean()*100:.0f}%")
    print("  → Abstention is not a separately hard problem; if the forward model is accurate, one threshold solves it.")

    meta = dict(n_in=len(IN), n_loco=len(LOCO), n_lomo=len(LOMO), n_hn2=len(HN2),
                loco_cls_correct=float(LOCO.cls_correct.mean()),
                auc_computed={r["score"]: r["auc"] for r in res},
                auc_measured={r["score"]: r["auc_measured"] for r in cmp_rows},
                separable_computed=[r["score"] for r in res if r["separable"]],
                separable_measured=[r["score"] for r in cmp_rows if r["sep_measured"]],
                tau_measured=float(tau_m),
                operating=OP.to_dict("records"))
    with open(os.path.join(ROOT, "data", "v3_abstention_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ================================================================ figure
    SHOW = ["S1_top1", "S4_libz", "S5_resid"]
    NICE = {"S1_top1": "top-1 cosine", "S4_libz": "library-normalised z",
            "S5_resid": "unmixing residual"}
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.32,
                          left=0.135, right=0.975, top=0.885, bottom=0.08)
    letters = "abcdef"
    rng = np.random.default_rng(0)
    for row, (tag, dIN, dLO, dH) in enumerate([
            ("computed (DFT) library", IN, LOCO, HN2),
            ("measured library", MIN_, MLOCO, MHN2)]):
        for col, s_ in enumerate(SHOW):
            ax = fig.add_subplot(gs[row, col])
            g = 1.0 if HIGHER_IS_IN[s_] else -1.0
            groups = [dIN[s_].values, dLO[s_].values, dH[s_].values]
            cols = [P["dark_teal"], P["warm_red"], P["amber"]]
            for j, (v, c) in enumerate(zip(groups, cols)):
                ax.scatter(rng.normal(j, .07, len(v)), v, s=60, color=c,
                           alpha=.85, edgecolor="w", lw=.7, zorder=3)
                ax.hlines(np.median(v), j - .3, j + .3, color="k", lw=2.2, zorder=4)
            tau = (g * dLO[s_]).max() * g
            ax.axhline(tau, color=P["rose"], ls="--", lw=1.7)
            a_ = auc(g * dIN[s_], g * dLO[s_])
            sep = (g * dIN[s_]).min() > (g * dLO[s_]).max()
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["in-library\n(accept)", "class absent\n(abstain)", "HN2"])
            ax.set_ylabel(NICE[s_])
            ttl = f"AUC {a_:.3f}" + ("   separable" if sep else "   overlapping")
            ax.set_title(ttl, fontsize=15,
                         color=P["dark_teal"] if sep else P["warm_red"])
            ax.text(-0.22, 1.06, letters[row * 3 + col], transform=ax.transAxes,
                    fontsize=26, fontweight="bold")
            if col == 0:
                ax.text(-0.33, 0.5, tag, transform=ax.transAxes, fontsize=17,
                        rotation=90, va="center", ha="center", fontweight="bold",
                        color=P["brown"])
    fig.suptitle("Abstention fails for the same reason identification fails",
                 fontsize=19, y=0.965)
    fig.savefig(os.path.join(ROOT, "figure", "v3_fig_abstention.png"), dpi=200)
    print("\nsaved figure/v3_fig_abstention.png + data/v3_abstention_*.csv")


if __name__ == "__main__":
    main()
