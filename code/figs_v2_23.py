"""ramanforge v2 — Figure 2 (calibration) and Figure 3 (measured vs computed)."""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.spectra import parse_raman
from src.calibrate import _cos
from scripts.analysis_v2 import (load_library, load_measured, synth, band_rmse,
                                 scale_freqs, GRID, AGENTS, JOB, KEY, CUT, FWHM)

LAB, AX, TK, LG = 28, 22, 18, 18
RED, ORG, TEAL, BLU, GREY = "#C94F4A", "#E8943A", "#4AACB0", "#5B8DB8", "#7A7269"
D = pickle.load(open(os.path.join(ROOT, "data", "_v2.pkl"), "rb"))
SL, SH = D["s_low"], D["s_high"]


def panel(ax, txt, x=-0.10, y=1.16):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=LAB,
            fontweight="bold", ha="left", va="top")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TK)


# =====================================================================  FIG 2
def fig2():
    lib, meas = load_library(), load_measured()
    gpk, gexp = lib[JOB["GF"]], meas[KEY["GF"]]
    mpk = D["mpk"]

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                          left=0.07, right=0.975, top=0.90, bottom=0.09)

    # (a) raw vs calibrated against measurement
    ax = fig.add_subplot(gs[0, 0])
    raw = synth(gpk, 1.0, 1.0, fwhm=2.0, scattering=False)
    cal = synth(gpk, SL, SH, fwhm=FWHM, scattering=True)
    ax.plot(GRID, gexp, color="#2B2B2B", lw=2.6, label="measured GF", zorder=3)
    ax.plot(GRID, -raw, color=GREY, lw=1.6, label="raw harmonic (unscaled)", zorder=2)
    ax.plot(GRID, -cal, color=RED, lw=2.6, label="calibrated", zorder=3)
    ax.axhline(0, color="#BBBBBB", lw=1)
    ax.set_xlim(250, 1800); ax.set_ylim(-1.25, 1.25)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["1", "0.5", "0", "0.5", "1"])
    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=AX)
    ax.set_ylabel("Normalized intensity", fontsize=AX)
    ax.legend(fontsize=LG, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCC")
    clean(ax); panel(ax, "(a)")

    # (b) three-term contribution
    ax = fig.add_subplot(gs[0, 1])
    names = ["frequency\nscaling", "+ condensed-phase\nlinewidth", "+ Raman scattering\nfactor"]
    vals = [v for _, v in D["stages"]]
    bars = ax.bar(range(3), vals, color=[GREY, TEAL, RED], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.022, f"{v:.3f}", ha="center", va="bottom", fontsize=LG, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=LG - 1)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Cosine similarity to\nmeasured GF", fontsize=AX)
    clean(ax); panel(ax, "(b)")

    # (c) positional-RMSE map over the two scale factors
    ax = fig.add_subplot(gs[1, 0])
    sl = np.arange(0.995, 1.0451, 0.005)
    sh = np.arange(0.970, 1.0051, 0.005)
    M = np.full((len(sh), len(sl)), np.nan)
    for i, b in enumerate(sh):
        for j, a in enumerate(sl):
            r, n = band_rmse(lib, mpk, float(a), float(b))
            if n >= 85:
                M[i, j] = r
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="YlOrRd_r",
                   extent=[sl[0], sl[-1], sh[0], sh[-1]])
    ax.plot(SL, SH, "*", ms=26, color="#1F1F1F", zorder=5)
    ax.annotate(f"adopted ({SL:.3f}, {SH:.3f})", xy=(SL, SH), xytext=(0.035, 0.90),
                textcoords="axes fraction", fontsize=LG, fontweight="bold", color="#1F1F1F",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#1F1F1F", lw=1.6))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("band-position RMSE (cm$^{-1}$)", fontsize=AX - 2); cb.ax.tick_params(labelsize=TK)
    ax.set_xlabel("$s_{\\mathrm{low}}$  (below 800 cm$^{-1}$)", fontsize=AX)
    ax.set_ylabel("$s_{\\mathrm{high}}$  (above 800 cm$^{-1}$)", fontsize=AX)
    ax.tick_params(labelsize=TK); panel(ax, "(c)")

    # (d) region-resolved optimum scale factor
    ax = fig.add_subplot(gs[1, 1])
    regions = [("250-800", 250, 800), ("800-1300", 800, 1300), ("1300-1800", 1300, 1800)]
    xs = np.arange(0.950, 1.0401, 0.002)
    opt = []
    for nm, lo, hi in regions:
        best = None
        for s in xs:
            d = []
            for ag in AGENTS:
                pk = lib[JOB[ag]]
                f = pk[:, 0] * s
                keep = (f > 250) & (f < 1800) & (pk[:, 1] > 0.02 * pk[:, 1].max())
                f = f[keep]
                for w in mpk[ag]:
                    if not (lo <= w < hi) or not len(f):
                        continue
                    j = int(np.argmin(abs(f - w)))
                    if abs(f[j] - w) <= 35:
                        d.append(f[j] - w)
            if len(d) >= 8:
                r = float(np.sqrt((np.array(d) ** 2).mean()))
                if best is None or r < best[0]:
                    best = (r, float(s), len(d))
        opt.append((nm, best[1], best[0], best[2]))
    cols = [RED, ORG, TEAL]
    ax.bar(range(3), [o[1] for o in opt], color=cols, width=0.6)
    for i, o in enumerate(opt):
        ax.text(i, o[1] + 0.0015, f"{o[1]:.3f}", ha="center", va="bottom",
                fontsize=LG, fontweight="bold")
        ax.text(i, 0.9585, f"n = {o[3]}", ha="center", va="bottom", fontsize=LG - 2, color="#555")
    ax.axhline(1.0, ls="--", lw=2, color="#555")
    ax.text(2.42, 1.0009, "no scaling", ha="right", va="bottom", fontsize=LG - 1, color="#555")
    ax.set_xticks(range(3)); ax.set_xticklabels([o[0] for o in opt], fontsize=LG)
    ax.set_xlabel("Wavenumber region (cm$^{-1}$)", fontsize=AX)
    ax.set_ylabel("Optimum scale factor", fontsize=AX)
    ax.set_ylim(0.956, 1.032)
    clean(ax); panel(ax, "(d)")

    out = os.path.join(ROOT, "figure", "fig2_calibration.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig2 ->", out, "| region optima:", [(o[0], o[1]) for o in opt])


# =====================================================================  FIG 3
def fig3():
    spec, meas = D["spec"], D["meas"]
    B = D["band_table"].set_index("agent")
    allp = D["band_all"]

    fig = plt.figure(figsize=(20, 15.5))
    gs = fig.add_gridspec(4, 3, hspace=0.62, wspace=0.26,
                          left=0.065, right=0.98, top=0.93, bottom=0.055,
                          height_ratios=[1, 1, 1, 1.25])
    tags = "abcdefghi"
    for k, ag in enumerate(AGENTS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        e, c = meas[ag], spec[JOB[ag]]
        ax.plot(GRID, e, color="#2B2B2B", lw=2.0, zorder=3)
        ax.plot(GRID, -c, color=RED, lw=2.0, zorder=3)
        ax.axhline(0, color="#BBBBBB", lw=1)
        ax.set_xlim(250, 1800); ax.set_ylim(-1.35, 1.35)
        ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["1", "0", "1"])
        ax.set_title(f"{ag}", fontsize=AX, pad=10)
        ax.text(0.975, 0.93, f"cos {B.loc[ag,'cos']:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=LG - 1, color="#444",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#DDD"))
        if k // 3 == 2:
            ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=AX - 2)
        if k % 3 == 0:
            ax.set_ylabel("Norm. intensity", fontsize=AX - 2)
        clean(ax); panel(ax, f"({tags[k]})", x=-0.13, y=1.30)

    # (j) parity
    ax = fig.add_subplot(gs[3, 0])
    x, y = allp[:, 0], allp[:, 1]
    ax.plot([250, 1800], [250, 1800], "--", lw=2, color="#888", zorder=1)
    ax.scatter(x, y, s=52, color=TEAL, edgecolor="white", lw=0.7, alpha=0.9, zorder=3)
    d = y - x
    ax.set_xlim(230, 1820); ax.set_ylim(230, 1820)
    ax.set_xlabel("Measured (cm$^{-1}$)", fontsize=AX - 2)
    ax.set_ylabel("Computed (cm$^{-1}$)", fontsize=AX - 2)
    ax.text(0.04, 0.955, f"$n$ = {len(x)}\nRMSE {np.sqrt((d**2).mean()):.1f} cm$^{{-1}}$\n"
                         f"MAE {np.abs(d).mean():.1f} cm$^{{-1}}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=LG,
            bbox=dict(boxstyle="round,pad=0.36", fc="white", ec="#DDD"))
    clean(ax); panel(ax, "(j)", x=-0.16, y=1.14)

    # (k) residual vs wavenumber
    ax = fig.add_subplot(gs[3, 1])
    ax.axhline(0, color="#888", lw=2, ls="--", zorder=1)
    ax.scatter(x, d, s=52, color=ORG, edgecolor="white", lw=0.7, alpha=0.9, zorder=3)
    sl_, ic_ = np.polyfit(x, d, 1)
    xx = np.linspace(250, 1800, 50)
    ax.plot(xx, sl_ * xx + ic_, color=RED, lw=2.4, zorder=4)
    ax.set_xlim(230, 1820)
    ax.set_xlabel("Measured (cm$^{-1}$)", fontsize=AX - 2)
    ax.set_ylabel("Residual (cm$^{-1}$)", fontsize=AX - 2)
    ax.text(0.04, 0.955, f"mean {d.mean():+.1f} cm$^{{-1}}$\nslope {sl_*1000:+.1f} / 1000 cm$^{{-1}}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=LG,
            bbox=dict(boxstyle="round,pad=0.36", fc="white", ec="#DDD"))
    clean(ax); panel(ax, "(k)", x=-0.235, y=1.14)

    # (l) per-agent RMSE
    ax = fig.add_subplot(gs[3, 2])
    cmap = {"Nerve": RED, "Blister": ORG, "Novichok": TEAL}
    kls = {"GD": "Nerve", "GF": "Nerve", "HD": "Blister", "HN1": "Blister", "HN3": "Blister",
           "A-230": "Novichok", "A-232": "Novichok", "A-234": "Novichok", "A-242": "Novichok"}
    vals = [B.loc[a, "rmse"] for a in AGENTS]
    ax.bar(range(9), vals, color=[cmap[kls[a]] for a in AGENTS], width=0.66)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=LG - 3)
    ax.set_xticks(range(9)); ax.set_xticklabels(AGENTS, rotation=45, ha="right", fontsize=LG - 2)
    ax.set_ylabel("RMSE (cm$^{-1}$)", fontsize=AX - 2)
    ax.set_ylim(0, max(vals) * 1.52)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=cmap[c]) for c in ["Nerve", "Blister", "Novichok"]]
    ax.legend(hs, ["Nerve", "Blister", "Novichok"], fontsize=LG - 3, loc="upper center",
              frameon=True, framealpha=0.95, edgecolor="#CCC", ncol=3,
              columnspacing=0.8, handlelength=1.1, bbox_to_anchor=(0.5, 1.02))
    clean(ax); panel(ax, "(l)", x=-0.16, y=1.14)

    out = os.path.join(ROOT, "figure", "fig3_measured_vs_computed.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig3 ->", out)


if __name__ == "__main__":
    fig2()
    fig3()
