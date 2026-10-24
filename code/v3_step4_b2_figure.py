"""
v3 B2: final verdict figure (fully revised 2026-09-24).

The previous version plotted the GF-only result ("the computed library can be fixed").
The GD, A-230 and A-234 results refuted that conclusion, so the figure is replaced entirely as well.

New claim: the standard calibration ladder was tested on 5 molecules × 3 variants = 10 cases,
         but no correction generalizes. The single success (GF) is not a mechanism; it is a
         shift in one direction that happened to coincide with the error of one molecule.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.spectra import parse_raman
from v3_step3_headtohead import (load_measured, load_library, synth_native, fit_scale,
                                 measured_peaks, cos, JOB, AGENTS)

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "font.size": 14, "font.family": "DejaVu Sans"})

REG = [("250–600", 250, 600), ("600–850", 600, 850), ("850–1150", 850, 1150),
       ("1150–1400", 1150, 1400), ("1400–1800", 1400, 1800)]
TIGHTP = ["GF", "GD", "A230", "A232", "A234", "HD"]
NICE = {"GF": "GF", "GD": "GD", "A230": "A-230", "A232": "A-232",
        "A234": "A-234", "HD": "HD"}
VNICE = {"V1_cpcm": "V1 CPCM", "V2_tightP": "V2 tight-P", "V3_wb97xd3": "V3 $\\omega$B97X-D3"}


def letter(ax, s, x=-0.16):
    ax.text(x, 1.06, s, transform=ax.transAxes, fontsize=25, fontweight="bold")


def main():
    X, acq = load_measured(); lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    calib = [a for a in AGENTS if a in JOB]
    _, SLb, SHb, _ = fit_scale(lib, mpk, calib)

    R = pd.read_csv(os.path.join(ROOT, "data", "v3_b2_evaluation.csv"))
    B = pd.read_csv(os.path.join(ROOT, "data", "v3_b2_bandshift.csv"))

    # cache of the tight-P spectra
    spec = {}
    for ag in TIGHTP:
        e = JOB[ag]
        new = parse_raman(os.path.join(ROOT, "orca", "outputs_b2", f"B2_{ag}_V2_tightP.out"))
        lib2 = dict(lib); lib2[e] = new
        _, SL2, SH2, _ = fit_scale(lib2, mpk, calib)
        spec[ag] = (synth_native(lib[e], X, SLb, SHb), synth_native(new, X, SL2, SH2))

    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.46,
                          left=0.105, right=0.930, top=0.875, bottom=0.085)

    # ---------- a : verdict on all ten pairs ----------
    ax = fig.add_subplot(gs[0, 0])
    T = R.assign(delta=R["new_600–850"] - R["base_600–850"]).sort_values("delta")
    lbl = [f"{NICE[r.agent]}  {VNICE[r.variant]}" for r in T.itertuples()]
    col = [P["dark_teal"] if d > 0.10 else (P["sage"] if d > 0.03 else
           (P["tan"] if d > -0.03 else P["warm_red"])) for d in T.delta]
    y = np.arange(len(T))
    ax.barh(y, T.delta, 0.68, color=col)
    for i, d in enumerate(T.delta):
        ax.text(d + (0.012 if d > 0 else -0.012), i, f"{d:+.3f}", va="center",
                ha="left" if d > 0 else "right", fontsize=11.5)
    ax.axvline(0, color="k", lw=1.6)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=11.5)
    ax.set_xlim(-0.42, 0.42)
    ax.set_xlabel("$\\Delta$ cosine, 600–850 cm$^{-1}$")
    ax.set_title("Ten agent–correction trials", fontsize=15.5)
    ax.text(0.97, 0.06, "1 improved\n3 degraded", transform=ax.transAxes, ha="right",
            fontsize=12.5, style="italic", color=P["brown"])
    letter(ax, "a", -0.34)

    # ---------- b : GF 600–850 ----------
    def spectra_panel(ax, ag, tag):
        m = (X >= 600) & (X <= 860)
        base, new = spec[ag]
        ax.plot(X[m], mean[ag][m] / mean[ag][m].max(), color="k", lw=2.8,
                label="measured", zorder=5)
        ax.plot(X[m], base[m] / max(base[m].max(), 1e-9), color=P["brown"], lw=1.9,
                label="baseline")
        ax.plot(X[m], new[m] / max(new[m].max(), 1e-9), color=P["dark_teal"], lw=1.9,
                label="V2 tight-P")
        r = B[B.agent == ag].iloc[0]
        ax.axvline(r.measured, color="0.7", lw=1.0, ls=":")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("normalised intensity")
        cb = R[(R.agent == ag) & (R.variant == "V2_tightP")].iloc[0]
        d = cb["new_600–850"] - cb["base_600–850"]
        ax.set_title(f"{NICE[ag]}   {cb['base_600–850']:.3f} $\\rightarrow$ "
                     f"{cb['new_600–850']:.3f}  ({d:+.3f})", fontsize=15.5)
        ax.legend(frameon=False, fontsize=11.5, loc="upper left",
                  handlelength=1.6, borderaxespad=0.6)
        letter(ax, tag)

    spectra_panel(fig.add_subplot(gs[0, 1]), "GF", "b")
    spectra_panel(fig.add_subplot(gs[1, 1]), "GD", "e")

    # ---------- c : heat map of the per-region delta (the six tight-P cases) ----------
    ax = fig.add_subplot(gs[0, 2])
    M = np.zeros((len(TIGHTP), len(REG)))
    for i, ag in enumerate(TIGHTP):
        r = R[(R.agent == ag) & (R.variant == "V2_tightP")].iloc[0]
        for j, (n, _, _) in enumerate(REG):
            M[i, j] = r[f"new_{n}"] - r[f"base_{n}"]
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(REG)))
    ax.set_xticklabels([n for n, _, _ in REG], rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(TIGHTP)))
    ax.set_yticklabels([NICE[a] for a in TIGHTP], fontsize=12)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=10.5,
                    color="white" if abs(M[i, j]) > 0.6 * v else "0.15")
    ax.set_title("V2 tight-P, by region", fontsize=15.5)
    cax = inset_axes(ax, width="4%", height="100%", loc="lower left",
                     bbox_to_anchor=(1.04, 0., 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\Delta$ cosine", fontsize=11.5)
    cb.ax.tick_params(labelsize=10.5)
    letter(ax, "c", -0.22)

    # ---------- d : the mechanism, scattered error vs a one-directional shift ----------
    ax = fig.add_subplot(gs[1, 0])
    Bs = B.sort_values("err_base")
    y = np.arange(len(Bs))
    for i, r in enumerate(Bs.itertuples()):
        better = abs(r.err_new) < abs(r.err_base) - 1.0
        worse = abs(r.err_new) > abs(r.err_base) + 1.0
        c = P["dark_teal"] if better else (P["warm_red"] if worse else P["tan"])
        if abs(r.shift) > 1.0:
            ax.annotate("", xy=(r.err_new, i), xytext=(r.err_base, i),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=c,
                                        shrinkA=0, shrinkB=0, mutation_scale=17))
        ax.plot(r.err_base, i, "o", ms=8, mfc="white", mec=P["brown"], mew=2, zorder=4)
        ax.plot(r.err_new, i, "o", ms=8, color=c, zorder=5)
        lab = "0" if abs(r.shift) < 1.0 else f"{r.shift:+.0f}"
        ax.text((r.err_base + r.err_new) / 2, i + 0.26, lab,
                fontsize=11.5, color=c, ha="center", va="bottom")
    ax.axvline(0, color="k", lw=2)
    ax.set_yticks(y); ax.set_yticklabels([NICE[a] for a in Bs.agent], fontsize=12.5)
    ax.set_ylim(-0.85, len(Bs) + 1.7)
    ax.set_xlim(-72, 104)
    ax.set_xlabel("computed $-$ measured band position (cm$^{-1}$)")
    ax.set_title("A one-directional shift meets scattered errors",
                 fontsize=15.5, pad=12)
    ax.text(0, -0.82, "measured", fontsize=11, color="0.35", ha="center", va="bottom")
    ax.text(0.985, 0.98,
            "baseline error   $-$50 … $+$55\nshift applied      0 … $+$23",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
            family="DejaVu Sans Mono", color=P["brown"], linespacing=1.5,
            bbox=dict(fc="white", ec="0.82", boxstyle="round,pad=0.4"))
    ax.text(0.03, 0.80, "the shift never points downward", transform=ax.transAxes,
            ha="left", va="top", fontsize=11, style="italic", color=P["brown"])
    hb, = ax.plot([], [], "o", ms=8, mfc="white", mec=P["brown"], mew=2)
    ht, = ax.plot([], [], "o", ms=8, color=P["dark_teal"])
    ax.legend([hb, ht], ["baseline", "V2 tight-P"], frameon=False, fontsize=11,
              loc="lower right", handletextpad=0.4, borderaxespad=0.8)
    letter(ax, "d", -0.22)

    # ---------- f : top-1 outcome ----------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0.5, 1.06, "Open-set top-1 (53-member library)", ha="center",
            fontsize=15.5, transform=ax.transAxes)
    ax.text(0.27, 0.95, "baseline", fontsize=12, color="0.35", transform=ax.transAxes)
    ax.text(0.70, 0.95, "V2 tight-P", fontsize=12, color="0.35", transform=ax.transAxes)
    ax.plot([0.02, 0.99], [0.925, 0.925], color="0.8", lw=1,
            transform=ax.transAxes, clip_on=False)

    def short(s_):
        return (s_.replace("EXP_", "").replace("_cyclosarin", "").replace("_soman", "")
                  .replace("A2", "A-2"))
    yy = 0.82
    for ag in TIGHTP:
        r = R[(R.agent == ag) & (R.variant == "V2_tightP")].iloc[0]
        ok_b, ok_n = r["correct_base"], r["correct_new"]
        ax.text(0.03, yy, NICE[ag], fontsize=13.5, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.27, yy, ("\u2713 " if ok_b else "\u2717 ") + short(r["top1_base"]),
                fontsize=12.5, color=P["dark_teal"] if ok_b else P["warm_red"],
                transform=ax.transAxes)
        ax.annotate("", xy=(0.67, yy + 0.012), xytext=(0.61, yy + 0.012),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=1.4))
        ax.text(0.70, yy, ("\u2713 " if ok_n else "\u2717 ") + short(r["top1_new"]),
                fontsize=12.5, fontweight="bold" if ok_b != ok_n else "normal",
                color=P["dark_teal"] if ok_n else P["warm_red"], transform=ax.transAxes)
        if ok_b and not ok_n:
            ax.text(0.70, yy - 0.058, "correct $\\rightarrow$ wrong", fontsize=10.5,
                    style="italic", color=P["warm_red"], transform=ax.transAxes)
            yy -= 0.045
        elif (not ok_b) and ok_n:
            ax.text(0.70, yy - 0.058, "wrong $\\rightarrow$ correct", fontsize=10.5,
                    style="italic", color=P["dark_teal"], transform=ax.transAxes)
            yy -= 0.045
        yy -= 0.115
    ax.plot([0.02, 0.99], [0.145, 0.145], color="0.8", lw=1,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.03, 0.115,
            "Global scale factors 1.018–1.024 / 0.988 and global position\n"
            "RMSE 12.3–13.3 cm$^{-1}$ across all ten runs: no global\n"
            "correction reaches a molecule-specific error.",
            fontsize=11.5, color=P["brown"], style="italic", va="top",
            transform=ax.transAxes, linespacing=1.5)
    letter(ax, "f", -0.08)

    fig.suptitle("No generalisable fix: the one apparent success does not transfer to its "
                 "nearest neighbour", fontsize=19, y=0.955)
    out = os.path.join(ROOT, "figure", "v3_fig_b2.png")
    fig.savefig(out, dpi=200)
    print("saved figure/v3_fig_b2.png")


if __name__ == "__main__":
    main()
