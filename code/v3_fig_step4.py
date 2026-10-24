"""
Figure S-step4: the geometry correction resolved by window, and the size explanation.

Panel (a) reads data/v3_geomwindow.csv (written by v3_geomwindow.py), which holds the
window cosines of the published and global-minimum geometries with the calibration held at
its pre-correction values so that only the structure changes.

Panel (b) reads data/v3_region_by_agent.csv, the 600-850 cm-1 window cosine of the adopted
corrected library, against heavy-atom count taken from each agent's optimized geometry.

Writes figures/v3_fig_step4.png.
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H
from v3_export_library import xyz_from_out

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5c5b55", "#dedcd5"
REG = ["250-600", "600-850", "850-1150", "1150-1400", "1400-1800"]
DE = {"GF": 4.57, "A-232": 9.47}          # B3LYP/6-311++G(d,p), Table S-a232 and Section 3.7
NICE = {"A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}


def heavy_counts():
    out = {}
    for a, j in H.JOB.items():
        el, _ = xyz_from_out(os.path.join(ROOT, "orca", "outputs", j + ".out"))
        if len(el):
            out[a] = int(sum(1 for e in el if e.upper() != "H"))
    return out


def main():
    G = pd.read_csv(os.path.join(ROOT, "data", "v3_geomwindow.csv"))
    R = pd.read_csv(os.path.join(ROOT, "data", "v3_region_by_agent.csv"), index_col=0)
    HV = heavy_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), facecolor="#fcfcfb")

    # ---- (a) window-resolved effect of the geometry correction ----
    ax = axes[0]
    x = np.arange(len(REG))
    for agent, col in (("GF", ORANGE), ("A-232", BLUE)):
        b = G[(G.agent == agent) & (G.state == "published")][REG].values[0]
        a_ = G[(G.agent == agent) & (G.state == "global minimum")][REG].values[0]
        for i in range(len(REG)):
            ax.annotate("", xy=(x[i] + 0.16, a_[i]), xytext=(x[i] - 0.16, b[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, alpha=0.9,
                                        shrinkA=3, shrinkB=3, mutation_scale=11))
        ax.scatter(x - 0.16, b, s=46, facecolor="white", edgecolor=col, linewidth=1.8, zorder=3)
        ax.scatter(x + 0.16, a_, s=52, color=col, edgecolor="white", linewidth=1.3, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([r.replace("-", "–") for r in REG],
                                         fontsize=8.6, color=INK2)
    ax.set_xlabel("Raman window  (cm$^{-1}$)", fontsize=9.5, color=INK)
    ax.set_ylabel("cosine vs measured spectrum", fontsize=9.5, color=INK)
    ax.set_title("(a)  correcting a verified geometry error", fontsize=11, color=INK,
                 loc="left", pad=10)
    ax.set_ylim(0.12, 1.02); ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(handles=[
        Line2D([], [], color=ORANGE, lw=2, label=f"GF  ($\\Delta E$ {DE['GF']} kcal/mol)"),
        Line2D([], [], color=BLUE, lw=2, label=f"A-232  ($\\Delta E$ {DE['A-232']} kcal/mol)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
               markersize=7, label="paper geometry"),
        Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor="white",
               markersize=7, label="global minimum")],
        frameon=False, fontsize=8.3, loc="lower left", labelcolor=INK2, ncol=2,
        columnspacing=1.0, handlelength=1.6)

    # ---- (b) window cosine against molecular size ----
    ax = axes[1]
    ags = [a for a in R.index if a in HV and a != "HN2"]
    hh = np.array([HV[a] for a in ags], float)
    cc = np.array([float(R.loc[a, "600-850"]) for a in ags])
    isp = np.array([H.cls_of(a) != "Blister" for a in ags])
    for a, h, c, p in zip(ags, hh, cc, isp):
        ax.scatter(h, c, s=120 if a == "VX" else 70, color=ORANGE if p else BLUE,
                   edgecolor="white", linewidth=1.4, zorder=3,
                   marker="D" if a == "VX" else "o")
    off = {"HD": (7, -3), "HN1": (7, -3), "GD": (-12, 9), "HN3": (7, -3), "A234": (7, -3),
           "A230": (-34, -4), "GF": (7, -10), "A242": (-40, -16), "A232": (7, -3),
           "VX": (-10, 11)}
    for a, h, c in zip(ags, hh, cc):
        ax.annotate(NICE.get(a, a), (h, c), textcoords="offset points",
                    xytext=off.get(a, (7, -3)), fontsize=8.5, color=INK2)
    m9 = np.array([a != "VX" for a in ags])
    r9 = np.corrcoef(hh[m9], cc[m9])[0, 1]
    r10 = np.corrcoef(hh, cc)[0, 1]
    for msk, col, ls, lab in [
            (m9, INK2, "--", f"without VX  ($n$={int(m9.sum())}):  $r$ = {r9:+.2f}"),
            (np.ones(len(ags), bool), ORANGE, "-",
             f"with VX  ($n$={len(ags)}):  $r$ = {r10:+.2f}")]:
        b = np.polyfit(hh[msk], cc[msk], 1)
        xs = np.array([hh.min() - 0.4, hh.max() + 0.6])
        ax.plot(xs, np.polyval(b, xs), ls, color=col, lw=1.5, alpha=0.75, zorder=1,
                label=lab.replace("+", "") if False else lab)
    ax.set_xlabel("heavy-atom count", fontsize=9.5, color=INK)
    ax.set_ylabel("600–850 cm$^{-1}$ window cosine", fontsize=9.5, color=INK)
    ax.set_title("(b)  VX breaks the size explanation", fontsize=11, color=INK,
                 loc="left", pad=10)
    ax.set_ylim(0.12, 1.08); ax.set_xlim(hh.min() - 0.8, hh.max() + 1.2)
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)
    cvx = float(R.loc["VX", "600-850"]); c42 = float(R.loc["A242", "600-850"])
    ax.annotate("", xy=(HV["VX"], cvx), xytext=(HV["A242"], c42),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2, alpha=0.55))
    ax.text(HV["VX"] - 0.4, (cvx + c42) / 2, f"same size,\n$\\Delta$cos {cvx - c42:.2f}",
            fontsize=8, color=ORANGE, ha="right", style="italic")
    fig.tight_layout(pad=1.4)
    out = os.path.join(ROOT, "figures", "v3_fig_step4.png")
    fig.savefig(out, dpi=200, facecolor="#fcfcfb")
    print(f"saved figures/v3_fig_step4.png   r(without VX) = {r9:+.3f}, r(all) = {r10:+.3f}, "
          f"VX - A-242 = {cvx - c42:+.3f}")


if __name__ == "__main__":
    main()
