"""
v3 Fig 2: calibration. Shows graphically that no parameter was tuned on identification.

a  how the three calibrations move the computed spectrum into a measurable form (GF)
b  cumulative cosine contribution of each calibration step
c  band position RMSE surface, the objective function that was actually minimized
d  optimal scale factor per region (why two factors are used)
e  nested leave-one-agent-out stability (the evidence of non-circularity)
f  the measured linewidth distribution and the adopted Gamma
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, synth_native, cos,
                                 measured_peaks, position_rmse, scattering_factor,
                                 JOB, AGENTS, CUT)

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "font.size": 14, "font.family": "DejaVu Sans"})
NICE = {"GD": "GD", "GF": "GF", "VX": "VX", "HD": "HD", "HN1": "HN1", "HN2": "HN2",
        "HN3": "HN3", "A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}
ORDER = ["GD", "GF", "VX", "HD", "HN1", "HN2", "HN3", "A230", "A232", "A234", "A242"]


def letter(ax, s, x=-0.17):
    ax.text(x, 1.06, s, transform=ax.transAxes, fontsize=25, fontweight="bold")


def lorentz(peaks_f, peaks_a, grid, gamma):
    g = gamma / 2.0
    I = np.zeros_like(grid, float)
    for f, a in zip(peaks_f, peaks_a):
        I += a * g * g / ((grid - f) ** 2 + g * g)
    return I / (I.max() + 1e-12)


def main():
    X, acq = load_measured(); lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    M = json.load(open(os.path.join(ROOT, "data", "v3_headtohead_meta.json")))
    SL, SH, GAM = M["s_low"], M["s_high"], 22.0
    calib = [a for a in AGENTS if a in JOB]

    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.40,
                          left=0.075, right=0.945, top=0.885, bottom=0.095)

    # ---------- a : effect of the three corrections (GF) ----------
    ax = fig.add_subplot(gs[0, 0])
    AG = "GF"; pk = lib[JOB[AG]]
    m = (X >= 250) & (X <= 1800)
    ax.plot(X[m], mean[AG][m] / mean[AG][m].max() + 2.2, color="k", lw=2.2,
            label="measured")
    raw = lorentz(pk[:, 0], pk[:, 1] / pk[:, 1].max(), X, GAM)
    ax.plot(X[m], raw[m] + 1.1, color=P["tan"], lw=1.8, label="harmonic, unscaled")
    cal = synth_native(pk, X, SL, SH, gamma=GAM)
    ax.plot(X[m], cal[m], color=P["dark_teal"], lw=1.8, label="calibrated")
    ax.set_yticks([0.5, 1.6, 2.7]); ax.set_yticklabels(["calibrated", "harmonic", "measured"],
                                                       fontsize=12)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_title("What the three corrections do (GF)", fontsize=15)
    ax.set_xlim(250, 1800)
    letter(ax, "a", -0.24)

    # ---------- b : cumulative cosine by correction stage ----------
    ax = fig.add_subplot(gs[0, 1])
    stages = ["harmonic\nlines", "+ linewidth", "+ scattering\nfactor", "+ frequency\nscale"]
    vals = []
    for a in calib:
        p = lib[JOB[a]]; y = mean[a]
        s1 = lorentz(p[:, 0], p[:, 1] / p[:, 1].max(), X, 4.0)
        s2 = lorentz(p[:, 0], p[:, 1] / p[:, 1].max(), X, GAM)
        aa = p[:, 1] * scattering_factor(p[:, 0]); aa = aa / aa.max()
        s3 = lorentz(p[:, 0], aa, X, GAM)
        s4 = synth_native(p, X, SL, SH, gamma=GAM)
        vals.append([cos(y, s) for s in (s1, s2, s3, s4)])
    V = np.array(vals)
    xx = np.arange(len(stages))
    for i, a in enumerate(calib):
        ax.plot(xx, V[i], "-o", ms=4, lw=1.1, color=P["tan"], alpha=.75, zorder=2)
    ax.plot(xx, V.mean(0), "-o", ms=10, lw=3, color=P["dark_teal"], zorder=4, label="mean")
    for k in range(len(stages)):
        ax.text(k, V.mean(0)[k] + 0.035, f"{V.mean(0)[k]:.3f}", ha="center",
                fontsize=12, fontweight="bold", color=P["dark_teal"])
    j = calib.index("HN2") if "HN2" in calib else None
    if j is not None:
        ax.plot(xx, V[j], "-o", ms=4, lw=1.6, color=P["warm_red"], zorder=3)
        ax.text(len(stages) - 1.05, V[j][-1] + 0.03, "HN2", fontsize=11.5,
                color=P["warm_red"], ha="right")
    ax.set_xticks(xx); ax.set_xticklabels(stages, fontsize=11.5)
    ax.set_ylabel("cosine to measured"); ax.set_ylim(0, 0.95)
    ax.set_title("Cumulative effect, all agents", fontsize=15)
    ax.legend(frameon=False, loc="upper left", fontsize=11.5)
    letter(ax, "b")

    # ---------- c : band-position RMSE surface (the actual objective) ----------
    ax = fig.add_subplot(gs[0, 2])
    sl = np.arange(0.995, 1.0451, 0.0025)
    sh = np.arange(0.960, 1.0101, 0.0025)
    R = np.full((len(sh), len(sl)), np.nan)
    for i, b in enumerate(sh):
        for j, a_ in enumerate(sl):
            r, n = position_rmse(lib, mpk, calib, float(a_), float(b))
            if n >= 60: R[i, j] = r
    im = ax.imshow(R, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[sl[0], sl[-1], sh[0], sh[-1]])
    ax.plot(SL, SH, "*", ms=22, mfc=P["gold"], mec="k", mew=1.4, zorder=5)
    ax.annotate(f"({SL:.3f}, {SH:.3f})\n{M['pos_rmse']:.1f} cm$^{{-1}}$",
                xy=(SL, SH), xytext=(SL - 0.021, SH - 0.026), fontsize=11.5,
                color="white", fontweight="bold")
    ax.set_xlabel("$s_\\mathrm{low}$ ($<$800 cm$^{-1}$)")
    ax.set_ylabel("$s_\\mathrm{high}$ ($\\geq$800 cm$^{-1}$)")
    ax.set_title("The objective actually minimized", fontsize=15)
    cax = inset_axes(ax, width="4%", height="100%", loc="lower left",
                     bbox_to_anchor=(1.04, 0., 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = fig.colorbar(im, cax=cax); cb.set_label("band-position RMSE (cm$^{-1}$)",
                                                 fontsize=11.5)
    cb.ax.tick_params(labelsize=10.5)
    letter(ax, "c", -0.22)

    # ---------- d : optimal scale factor per region ----------
    ax = fig.add_subplot(gs[1, 0])
    REG = [(250, 500), (500, 800), (800, 1100), (1100, 1400), (1400, 1800)]
    grid = np.arange(0.955, 1.0501, 0.001)
    opt, lab = [], []
    for lo, hi in REG:
        best = (None, np.inf)
        for s in grid:
            d = []
            for a in calib:
                p = lib[JOB[a]]
                f = p[:, 0] * s
                keep = (f > lo) & (f < hi) & (p[:, 1] > 0.02 * p[:, 1].max())
                for w in [w for w in mpk[a] if lo <= w < hi]:
                    if keep.sum():
                        j = int(np.argmin(np.abs(f[keep] - w)))
                        if abs(f[keep][j] - w) <= 35: d.append(f[keep][j] - w)
            if len(d) >= 4:
                r = float(np.sqrt(np.mean(np.array(d) ** 2)))
                if r < best[1]: best = (float(s), r)
        if best[0] is None:
            print(f"  ! {lo}-{hi}: too few paired bands, excluded")
            continue
        opt.append(best[0]); lab.append(f"{lo}--{hi}")
    xx = np.arange(len(opt))
    cols = [P["warm_red"] if o > 1.0 else P["dusty"] for o in opt]
    ax.bar(xx, [o - 1 for o in opt], 0.55, bottom=1, color=cols)
    for i, o in enumerate(opt):
        ax.text(i, o + (0.0012 if o > 1 else -0.0012), f"{o:.3f}", ha="center",
                va="bottom" if o > 1 else "top", fontsize=12)
    ax.axhline(1.0, color="k", lw=1.4)
    ax.axhline(SL, color=P["warm_red"], ls="--", lw=1.5)
    ax.axhline(SH, color=P["dusty"], ls="--", lw=1.5)
    ax.text(len(opt) - 0.3, SL, "$s_\\mathrm{low}$", va="bottom", ha="right",
            fontsize=12, color=P["warm_red"])
    ax.text(len(opt) - 0.3, SH, "$s_\\mathrm{high}$", va="top", ha="right",
            fontsize=12, color=P["dusty"])
    ax.axvline(0.5, color="0.6", lw=1.2, ls=":")
    ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=25, fontsize=11)
    ax.set_xlim(-0.6, len(opt) - 0.15)
    ax.set_ylim(0.9855, 1.0235)
    ax.set_ylabel("best single scale factor")
    ax.set_title("The optimum is not uniform", fontsize=15, pad=14)
    letter(ax, "d", -0.22)

    # ---------- e : nested LOAO ----------
    ax = fig.add_subplot(gs[1, 1])
    N = pd.read_csv(os.path.join(ROOT, "data", "v3_nested_calibration.csv"))
    xx = np.arange(len(N))
    ax.plot(xx, N.s_low, "-o", color=P["warm_red"], ms=8, label="$s_\\mathrm{low}$")
    ax.plot(xx, N.s_high, "-s", color=P["dusty"], ms=8, label="$s_\\mathrm{high}$")
    ax.axhline(SL, color=P["warm_red"], ls=":", lw=1.4)
    ax.axhline(SH, color=P["dusty"], ls=":", lw=1.4)
    ax.set_xticks(xx); ax.set_xticklabels([NICE.get(h, h) for h in N.held_out],
                                          rotation=90, fontsize=11)
    ax.set_ylim(0.980, 1.028)
    ax.set_xlabel("agent held out of the calibration")
    ax.set_ylabel("refitted scale factor")
    ax.set_title("No agent drives the calibration", fontsize=15)
    ax.text(0.97, 0.52, f"$s_\\mathrm{{low}}$ range {N.s_low.max()-N.s_low.min():.3f}\n"
                        f"$s_\\mathrm{{high}}$ invariant",
            transform=ax.transAxes, ha="right", fontsize=11.5, color=P["brown"],
            bbox=dict(fc="white", ec="0.82", boxstyle="round,pad=0.4"))
    ax.legend(frameon=False, fontsize=11.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))
    letter(ax, "e")

    # ---------- f : measured linewidths ----------
    ax = fig.add_subplot(gs[1, 2])
    G = pd.read_csv(os.path.join(ROOT, "data", "v3_linewidths.csv"))
    ags = [a for a in ORDER if a in set(G.agent)]
    for i, a in enumerate(ags):
        s = G[G.agent == a]
        ax.scatter(np.full(len(s), i) + np.random.RandomState(i).uniform(-.13, .13, len(s)),
                   s.gamma, s=55, color=P["dusty"], edgecolor="w", linewidth=.8, zorder=3)
        ax.plot([i - .3, i + .3], [s.gamma.median()] * 2, color="k", lw=2.4, zorder=4)
    ax.axhline(22, color=P["warm_red"], lw=2, ls="--", zorder=2,
               label="adopted $\\Gamma$ = 22 cm$^{-1}$")
    ax.axhspan(G.gamma.quantile(.25), G.gamma.quantile(.75), color=P["pale_teal"],
               alpha=.25, zorder=1, label="interquartile range")
    ax.set_xticks(range(len(ags))); ax.set_xticklabels([NICE[a] for a in ags],
                                                       rotation=90, fontsize=11)
    ax.set_ylabel("fitted linewidth (cm$^{-1}$)")
    ax.set_title("The linewidth is measured", fontsize=15)
    ax.set_ylim(5, 58)
    ax.legend(frameon=False, fontsize=11, loc="upper right", borderaxespad=0.6)
    letter(ax, "f", -0.20)

    fig.suptitle("A calibration with no parameter fitted to identification",
                 fontsize=19, y=0.955)
    out = os.path.join(ROOT, "figure", "v3_fig_calibration.png")
    fig.savefig(out, dpi=200)
    print("saved figure/v3_fig_calibration.png")
    print("  per-region optimum:", dict(zip(lab, [round(o, 3) for o in opt])))


if __name__ == "__main__":
    main()
