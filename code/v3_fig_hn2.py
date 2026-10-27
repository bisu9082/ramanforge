"""
Figure S-hn2identity: what the vial labeled HN2 contains.

Top: the measured mean spectrum with the liquid-benzene reference positions marked.
Bottom: the same spectrum magnified, with the four bands Kondo et al. report for HN2
scored against the local noise, and the positions a monosubstituted ring would require.

Writes figures/v3_fig_hn2.png. The numbers printed are recomputed here from the deposited
acquisitions; data/v3_hn2_identity.csv carries the same values in tabular form.
"""
import os
import sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import v3_step3_headtohead as H
from scipy.signal import find_peaks

X, acq = H.load_measured()
m = np.mean([s for _, s in acq["HN2"]], 0); m = m / m.max()
sd = float(np.std(np.diff(m[X > 1600]))) / np.sqrt(2.0)
BZ = [(606, "$\\nu_6$"), (992, "$\\nu_1$ ring breathing"), (1178, "$\\nu_9$"), (1586, "$\\nu_8$")]
KON = [(388, "skeletal"), (664, "tert-amine"), (729, "C--Cl"), (1448, "N--C")]
KOFF = {388: 0, 664: -62, 729: 42, 1448: 0}
ABS = [(702, "mono-sub."), (787, "mono-sub."), (1030, "mono-sub.")]
BLUE, ORANGE, GREEN, INK, INK2, BG = "#2a78d6", "#eb6834", "#2e8b57", "#1a1a19", "#5c5b55", "#fcfcfb"

fig, ax = plt.subplots(2, 1, figsize=(13, 7.6), dpi=200, facecolor=BG,
                       gridspec_kw=dict(height_ratios=[2, 1.35], hspace=0.30))
a = ax[0]; a.set_facecolor(BG)
a.plot(X, m, color=INK, lw=1.2); a.fill_between(X, 0, m, color=INK, alpha=0.06)
for nu, lab in BZ:
    a.axvline(nu, color=BLUE, lw=1.0, ls=(0, (4, 3)), alpha=0.85)
    a.text(nu, 1.09, f"{nu}", color=BLUE, fontsize=9.5, ha="center", fontweight="bold")
    a.text(nu, 1.03, lab, color=BLUE, fontsize=8.2, ha="center")
a.set_xlim(250, 1800); a.set_ylim(-0.03, 1.20); a.set_yticks([0, 0.5, 1])
a.set_ylabel("normalized intensity", fontsize=11, color=INK2)
a.set_title("The vial labeled HN2: the four strong bands are liquid benzene",
            fontsize=13.5, fontweight="bold", color=INK, loc="left")
a.text(0.60, 0.70, "blue: benzene reference positions\n(606, 992, 1178, 1586 cm$^{-1}$)",
       transform=a.transAxes, ha="left", fontsize=9.5, color=BLUE)
for sp in ("top", "right"): a.spines[sp].set_visible(False)
a.tick_params(colors=INK2, labelsize=9)

b = ax[1]; b.set_facecolor(BG)
b.plot(X, m, color=INK, lw=1.2)
b.axhline(5 * sd, color=ORANGE, lw=1.0, ls=":")
b.axhline(0.05, color=GREEN, lw=1.0, ls="--")
b.text(1790, 5 * sd, f"  5$\\sigma$ = {5*sd:.3f}", color=ORANGE, fontsize=9, va="bottom", ha="right")
b.text(1790, 0.05, "  absolute floor 0.05 used in the paper", color=GREEN, fontsize=9, va="bottom", ha="right")
idx, pr = find_peaks(m, prominence=0.0, distance=2)
w, p = X[idx], pr["prominences"]
for nu, lab in KON:
    j = int(np.argmin(np.abs(w - nu)))
    hit = abs(w[j] - nu) <= 18
    col = ORANGE if hit and p[j] > 3 * sd else "0.6"
    b.plot([nu, nu], [-0.004, -0.010], color=col, lw=2.2, clip_on=False)
    txt = (f"{w[j]:.0f}\n{p[j]/sd:.1f}$\\sigma$" if hit else "absent")
    xo = nu + KOFF.get(nu, 0)
    b.text(xo, 0.056, f"Kondo {nu}", color=col, fontsize=8.6, ha="center", fontweight="bold")
    b.text(xo, 0.040, txt, color=col, fontsize=8.2, ha="center", linespacing=1.3)
for nu, _ in ABS:
    jj = int(np.argmin(np.abs(X - nu)))
    b.text(nu, 0.012, f"{nu}\n{m[jj]/sd:.1f}$\\sigma$", color="0.55", fontsize=8.0, ha="center", linespacing=1.3)
b.set_xlim(250, 1800); b.set_ylim(-0.012, 0.075)
b.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=11, color=INK2)
b.set_ylabel("normalized intensity", fontsize=11, color=INK2)
b.set_title("Same spectrum, 13$\\times$ vertical zoom: three of Kondo's four HN2 bands clear 3$\\sigma$ in this mean spectrum, two of them in every acquisition",
            fontsize=11.5, color=INK, loc="left")
b.text(0.995, 0.97, "grey, lower row: positions a monosubstituted ring would require (702, 787, 1030) — all below 1.5$\\sigma$",
       transform=b.transAxes, ha="right", fontsize=9, color="0.45")
for sp in ("top", "right"): b.spines[sp].set_visible(False)
b.tick_params(colors=INK2, labelsize=9)
fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.085)
out = os.path.join(ROOT, "figures", "v3_fig_hn2.png")
fig.savefig(out, dpi=200, facecolor=BG)
print("saved figures/v3_fig_hn2.png")
