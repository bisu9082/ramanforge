"""v3 Fig 1: study design. The two paths proceed independently and meet at the same query."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
NICE = {"GD": "GD", "GF": "GF", "VX": "VX", "HD": "HD", "HN1": "HN1", "HN2": "HN2",
        "HN3": "HN3", "A230": "A-230", "A232": "A-232", "A234": "A-234", "A242": "A-242"}
ORDER = ["GD", "GF", "VX", "HD", "HN1", "HN2", "HN3", "A230", "A232", "A234", "A242"]
CLS = dict(GD="Nerve", GF="Nerve", VX="Nerve", HD="Vesicant", HN1="Vesicant",
           HN2="Vesicant", HN3="Vesicant", A230="Novichok", A232="Novichok",
           A234="Novichok", A242="Novichok")
CCOL = {"Nerve": P["warm_red"], "Vesicant": P["amber"], "Novichok": P["dusty"]}


def box(ax, x, y, w, h, text, fc, ec=None, fs=12, tc="k", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec or fc, lw=1.4, transform=ax.transAxes,
                                clip_on=False, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            transform=ax.transAxes, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, col="0.45", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                                 arrowstyle=style, mutation_scale=17, lw=lw,
                                 color=col, clip_on=False, zorder=1))


def main():
    S = pd.read_csv(os.path.join(ROOT, "data", "v3_agent_summary.csv")).set_index("agent")
    M = json.load(open(os.path.join(ROOT, "data", "v3_headtohead_meta.json")))
    PRE = json.load(open(os.path.join(ROOT, "data", "v3_preproc_meta.json")))

    fig = plt.figure(figsize=(19, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1], hspace=0.46, wspace=0.30,
                          left=0.045, right=0.975, top=0.91, bottom=0.075)

    # ========== a : workflow ==========
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(-0.012, 1.02, "a", transform=ax.transAxes, fontsize=25, fontweight="bold")

    W, H = 0.145, 0.24
    yM, yC = 0.62, 0.16
    xs = [0.035, 0.225, 0.415, 0.605]
    ax.text(0.012, yM + H + 0.05, "MEASUREMENT", fontsize=13, fontweight="bold",
            color=P["dark_teal"], transform=ax.transAxes)
    ax.text(0.012, yC - 0.075, "CALCULATION", fontsize=13, fontweight="bold",
            color=P["brown"], transform=ax.transAxes)

    meas = [("11 authentic agents\nneat, sealed vials", P["pale_teal"]),
            ("Raman, 1064 nm\nraw detector counts", P["pale_teal"]),
            (f"{PRE['n_genuine']} genuine of 55 files\n23 derived, excluded", P["pale_teal"]),
            ("preprocessing\nspecified in full", P["teal"])]
    comp = [("SMILES\n53 molecules", "#EBDFD2"),
            ("B3LYP/6-311++G(d,p)\nORCA, NumFreq", "#EBDFD2"),
            ("harmonic lines\n+ Raman activities", "#EBDFD2"),
            ("calibration\n3 corrections", P["tan"])]
    for i, ((tm, cm), (tc_, cc)) in enumerate(zip(meas, comp)):
        box(ax, xs[i], yM, W, H, tm, cm, fs=11.5)
        box(ax, xs[i], yC, W, H, tc_, cc, fs=11.5)
        if i < 3:
            arrow(ax, xs[i] + W + 0.004, yM + H / 2, xs[i + 1] - 0.004, yM + H / 2)
            arrow(ax, xs[i] + W + 0.004, yC + H / 2, xs[i + 1] - 0.004, yC + H / 2)

    xj = 0.795
    box(ax, xj, 0.30, 0.175, 0.42,
        "SAME native grid\n270 points, 250\u20131800 cm$^{-1}$\n\nSAME cosine metric\n\n"
        "30 genuine acquisitions\nof 9 agents",
        "white", ec=P["dark_teal"], fs=11.5, bold=False)
    arrow(ax, xs[3] + W + 0.004, yM + H / 2, xj - 0.004, 0.62, col=P["dark_teal"])
    arrow(ax, xs[3] + W + 0.004, yC + H / 2, xj - 0.004, 0.40, col=P["brown"])

    ax.text(0.5, -0.045,
            "No calibration parameter is fitted to identification: the frequency scale comes "
            "from band positions, the linewidth from measured band widths,\nand the "
            "activity-to-intensity conversion has no free parameter.",
            ha="center", va="top", fontsize=12, style="italic", color=P["brown"],
            transform=ax.transAxes, linespacing=1.5)

    # ========== b : the agent set ==========
    ax = fig.add_subplot(gs[1, :2])
    ax.text(-0.055, 1.05, "b", transform=ax.transAxes, fontsize=25, fontweight="bold")
    n = [int(S.loc[a, "n_genuine"]) for a in ORDER]
    xx = np.arange(len(ORDER))
    ax.bar(xx, n, 0.6, color=[CCOL[CLS[a]] for a in ORDER])
    for i, a in enumerate(ORDER):
        ax.text(i, n[i] + 0.12, str(n[i]), ha="center", fontsize=12)
        if n[i] < 3:
            ax.text(i, 0.45, "not\nqueried", ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold")
        if a == "VX":
            ax.text(i, n[i] + 0.75, "no DFT\nentry", ha="center", va="bottom",
                    fontsize=10, color=P["warm_red"], fontweight="bold", linespacing=1.3)
        if a == "HN2":
            ax.text(i, n[i] + 0.75, "not HN2\n(see text)", ha="center", va="bottom",
                    fontsize=10, color=P["warm_red"], fontweight="bold", linespacing=1.3)
    ax.set_xticks(xx); ax.set_xticklabels([NICE[a] for a in ORDER], fontsize=12)
    ax.set_ylabel("genuine acquisitions", fontsize=14)
    ax.set_ylim(0, 7.4); ax.set_yticks([0, 2, 4, 6])
    ax.set_title("Eleven authentic agents, three toxicological classes", fontsize=15)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=CCOL[c]) for c in ["Nerve", "Vesicant", "Novichok"]]
    ax.legend(hs, ["nerve", "vesicant", "Novichok"], frameon=False, fontsize=11.5,
              ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    # ========== c : the two libraries ==========
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(-0.09, 1.05, "c", transform=ax.transAxes, fontsize=25, fontweight="bold")
    ax.text(0.5, 1.02, "The two libraries", ha="center", fontsize=15,
            transform=ax.transAxes)
    box(ax, 0.02, 0.50, 0.44, 0.34,
        "MEASURED\n\n11 entries\none per agent",
        P["pale_teal"], fs=12)
    box(ax, 0.54, 0.50, 0.44, 0.34,
        "COMPUTED\n\n53 entries\n10 agents + 43 analogues",
        P["tan"], fs=12)
    ax.text(0.5, 0.40,
            "45 nerve  /  4 vesicant  /  4 Novichok\n"
            "85% of the computed library carries the nerve-agent label",
            ha="center", va="top", fontsize=11.5, color=P["brown"],
            transform=ax.transAxes, linespacing=1.6)
    ax.text(0.5, 0.10, "A query is never present in the library\nit is matched against.",
            ha="center", va="bottom", fontsize=11.5, style="italic",
            color=P["dark_teal"], transform=ax.transAxes, linespacing=1.6)

    fig.suptitle("Two libraries, one comparison", fontsize=19, y=0.975)
    fig.savefig(os.path.join(ROOT, "figures", "v3_fig_design.png"), dpi=200)
    print("saved figures/v3_fig_design.png")


if __name__ == "__main__":
    main()
