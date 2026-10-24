"""
Figure S-kondo: measured mean spectra with Kondo et al.'s published band positions marked.

Reads data/v3_kondo_comparison.csv (written by v3_kondo_compare.py) and writes
figures/kondo_comparison.png.
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H

BLUE, ORANGE, INK, INK2, BG = "#2a78d6", "#eb6834", "#1a1a19", "#5c5b55", "#fcfcfb"
ORDER = ["GD", "GF", "VX", "HD", "HN1", "HN3", "HN2"]


def main():
    X, acq = H.load_measured()
    D = pd.read_csv(os.path.join(ROOT, "data", "v3_kondo_comparison.csv"))
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(9.5, 12.0), dpi=200,
                             facecolor=BG, sharex=True)
    for ax, a in zip(axes, ORDER):
        m = np.mean([s for _, s in acq[a]], 0); m = m / m.max()
        bad = a == "HN2"
        col = ORANGE if bad else INK
        ax.plot(X, m, color=col, lw=1.1)
        ax.fill_between(X, 0, m, color=col, alpha=0.07)
        ax.set_facecolor(BG)
        s = D[D.agent == a]
        for r in s.itertuples():
            c = BLUE if r.matched == "yes" else ORANGE
            ax.plot([r.kondo_cm1, r.kondo_cm1], [-0.14, -0.03], color=c, lw=2.0,
                    solid_capstyle="butt", clip_on=False)
        nmt = int((s.matched == "yes").sum())
        nin = int((s.matched != "out of range").sum())
        lab = "the vial labeled HN2" if bad else a
        ax.text(0.006, 1.02, lab, transform=ax.transAxes, fontsize=11.5,
                fontweight="bold", color=col, va="bottom")
        ax.text(0.26 if bad else 0.055, 1.03,
                f"{nmt} of {nin} Kondo bands reproduced", transform=ax.transAxes,
                fontsize=9, color=(ORANGE if bad else INK2), va="bottom",
                fontweight=("bold" if bad else "normal"))
        ax.set_xlim(250, 1800); ax.set_ylim(-0.16, 1.12); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#d9d7cf")
        ax.tick_params(colors=INK2, labelsize=9, length=2)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10.5, color=INK2)

    inr = D[D.matched != "out of range"]
    mt = inr[inr.matched == "yes"]
    rms = float(np.sqrt((mt.residual ** 2).mean()))
    fig.suptitle("Measured band positions against Kondo et al. 2018, authentic agents at 1064 nm",
                 fontsize=12.5, fontweight="bold", color=INK, y=0.992, x=0.012, ha="left")
    fig.text(0.012, 0.940,
             "Ticks below each trace are Kondo's assigned bands: blue, reproduced here within "
             f"18 cm$^{{-1}}$; orange, not reproduced.\nOverall {len(mt)} of {len(inr)} reproduced, "
             f"residual RMS {rms:.1f} cm$^{{-1}}$, which is below the 5.73 cm$^{{-1}}$ sampling "
             "of this measurement.", fontsize=9.3, color=INK2)
    fig.text(0.012, 0.012,
             "The vial labeled HN2 is the only one that reproduces none of them at this floor.\n"
             "Its three bands (994, 1176, 1585 cm$^{-1}$) are not in Kondo's HN2, and a dominant "
             "band near 1000 cm$^{-1}$ with a partner\nnear 1590 cm$^{-1}$ is consistent with a "
             "monosubstituted aromatic ring, which no nitrogen mustard contains.",
             fontsize=9.3, color=ORANGE)
    fig.subplots_adjust(left=0.012, right=0.985, top=0.900, bottom=0.095, hspace=0.52)
    out = os.path.join(ROOT, "figures", "kondo_comparison.png")
    fig.savefig(out, dpi=200, facecolor=BG)
    print("saved figures/kondo_comparison.png")


if __name__ == "__main__":
    main()
