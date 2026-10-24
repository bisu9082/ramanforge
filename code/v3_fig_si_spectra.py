"""
Figure S-spectra: measured against calibrated computed spectrum for each of the eleven
vials, with the 600-850 cm-1 window shaded.

Writes figures/v3_fig_si_spectra.png. The cosines printed on each panel are recomputed
here from the same library and calibration the identification uses, so the figure cannot
drift from Table S-regbyagent.
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

NICE = {"GD": "GD", "GF": "GF", "VX": "VX", "HD": "HD", "HN1": "HN1", "HN2": "HN2",
        "HN3": "HN3", "A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}
ORDER = ["GD", "GF", "VX", "A230", "A232", "A234", "A242", "HD", "HN1", "HN3", "HN2"]
W0, W1 = 600.0, 850.0
BLACK, RED, SHADE = "#1a1a19", "#C94F4A", "#F2D9BE"


def main():
    X, acq = H.load_measured()
    lib = H.load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in H.AGENTS}
    mpk = {a: H.measured_peaks(X, mean[a]) for a in H.AGENTS}
    calib = [a for a in H.AGENTS if a in H.JOB]
    _, SL, SH, _ = H.fit_scale(lib, mpk, calib)
    dft = {k: H.synth_native(v, X, SL, SH) for k, v in lib.items()}
    win = (X >= W0) & (X <= W1)

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 4, figsize=(20, 11.5), dpi=200)
    for k, a in enumerate(ORDER):
        ax = axes[k // 4][k % 4]
        m = mean[a] / mean[a].max()
        c = dft[H.JOB[a]] / dft[H.JOB[a]].max()
        ax.axvspan(W0, W1, color=SHADE, alpha=0.75, lw=0)
        ax.plot(X, m, color=BLACK, lw=1.3)
        ax.plot(X, -c, color=RED, lw=1.3)
        ax.axhline(0, color="0.55", lw=0.8)
        full = H.cos(m, c)
        wcos = H.cos(m[win], c[win])
        ax.text(0.03, 1.015, f"({chr(97 + k)})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")
        ax.text(0.17, 1.02, f"{NICE[a]}    {full:.3f}", transform=ax.transAxes,
                fontsize=13.5, va="bottom")
        ax.text(0.975, 0.045, f"600--850: {wcos:.3f}".replace("--", "–"),
                transform=ax.transAxes, ha="right", fontsize=11, color="0.3",
                bbox=dict(fc="white", ec="0.75", boxstyle="round,pad=0.3"))
        ax.set_xlim(250, 1800); ax.set_ylim(-1.18, 1.18)
        ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["1", "0", "1"])
        if k // 4 == 2 or k >= len(ORDER) - 4:
            ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=12)
        if k % 4 == 0:
            ax.set_ylabel("normalized intensity", fontsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].plot([], [], color=BLACK, lw=1.3, label="measured")
    axes[0][0].plot([], [], color=RED, lw=1.3, label="computed")
    axes[0][0].legend(frameon=False, fontsize=11, loc="upper right")
    axes[2][3].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.6, w_pad=2.0)
    out = os.path.join(ROOT, "figures", "v3_fig_si_spectra.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved figures/v3_fig_si_spectra.png")


if __name__ == "__main__":
    main()
