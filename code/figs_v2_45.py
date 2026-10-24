"""ramanforge v2 — Figure 4 (identification) and Figure 5 (failure analysis)."""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
sys.path.insert(0, ROOT)
from src.calibrate import _cos
from scripts.analysis_v2 import GRID, AGENTS, JOB, cls, label

LAB, AX, TK, LG = 28, 22, 18, 18
RED, ORG, TEAL, BLU, GREY = "#C94F4A", "#E8943A", "#4AACB0", "#5B8DB8", "#7A7269"
D = pickle.load(open(os.path.join(ROOT, "data", "_v2.pkl"), "rb"))
WARM = LinearSegmentedColormap.from_list("warm", ["#FFFFFF", "#F6D9A8", "#E8943A", "#C94F4A", "#7A2E2A"])
KLS = {a: cls(JOB[a]) for a in AGENTS}
CCOL = {"Nerve": RED, "Blister": ORG, "Novichok": TEAL}


def panel(ax, t, x=-0.12, y=1.16):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=LAB, fontweight="bold", ha="left", va="top")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TK)


# =====================================================================  FIG 4
def fig4():
    spec, meas = D["spec"], D["meas"]
    R = D["R53"].set_index("agent")
    RL = D["RLOO"].set_index("agent")

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.34,
                          left=0.085, right=0.965, top=0.93, bottom=0.055,
                          height_ratios=[1.35, 1, 1])

    # (a) measured x computed similarity matrix
    ax = fig.add_subplot(gs[0, 0])
    M = np.array([[_cos(meas[a], spec[JOB[b]]) for b in AGENTS] for a in AGENTS])
    im = ax.imshow(M, cmap=WARM, vmin=0.55, vmax=1.0)
    for i in range(9):
        j = int(np.argmax(M[i]))
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#1F1F1F", lw=3.2))
        for k in range(9):
            ax.text(k, i, f"{M[i,k]:.2f}", ha="center", va="center", fontsize=LG - 5,
                    color="white" if M[i, k] > 0.86 else "#333")
    ax.set_xticks(range(9)); ax.set_xticklabels(AGENTS, rotation=45, ha="right", fontsize=LG - 3)
    ax.set_yticks(range(9)); ax.set_yticklabels(AGENTS, fontsize=LG - 3)
    ax.set_xlabel("Computed library entry", fontsize=AX - 1)
    ax.set_ylabel("Measured agent", fontsize=AX - 1)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("cosine similarity", fontsize=AX - 3); cb.ax.tick_params(labelsize=TK - 2)
    ax.tick_params(labelsize=TK); panel(ax, "(a)", x=-0.16, y=1.14)

    # (b) class confusion
    ax = fig.add_subplot(gs[0, 1])
    KL = ["Nerve", "Blister", "Novichok"]
    C = np.zeros((3, 3))
    for a in AGENTS:
        C[KL.index(KLS[a]), KL.index(cls(R.loc[a, "top1"]))] += 1
    im = ax.imshow(C, cmap=WARM, vmin=0, vmax=4)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(C[i,j])}", ha="center", va="center", fontsize=AX + 6,
                    fontweight="bold", color="white" if C[i, j] > 2.4 else "#333")
    ax.set_xticks(range(3)); ax.set_xticklabels(KL, fontsize=LG)
    ax.set_yticks(range(3)); ax.set_yticklabels(KL, fontsize=LG)
    ax.set_xlabel("Assigned class", fontsize=AX - 1)
    ax.set_ylabel("True class", fontsize=AX - 1)
    ax.set_title("9 / 9 correct", fontsize=AX, pad=14)
    ax.tick_params(labelsize=TK); panel(ax, "(b)", x=-0.18, y=1.14)

    # (c) library growth
    ax = fig.add_subplot(gs[1, 0])
    cur = D["curve"]
    n = [c[0] for c in cur]
    ax.errorbar(n, [c[3] for c in cur], yerr=[c[4] for c in cur], marker="o", ms=13, lw=3.2,
                capsize=6, color=TEAL, label="class")
    ax.errorbar(n, [c[1] for c in cur], yerr=[c[2] for c in cur], marker="s", ms=12, lw=3.2,
                capsize=6, color=RED, label="molecule")
    ax.set_ylim(0, 112); ax.set_xlim(4, 58)
    ax.set_xlabel("Library size (molecules)", fontsize=AX - 1)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=AX - 1)
    ax.legend(fontsize=LG, loc="center right", frameon=True, framealpha=0.95, edgecolor="#CCC")
    ax.text(0.03, 0.055, "both flat from 9 to 53 members", transform=ax.transAxes,
            fontsize=LG - 1, color="#555")
    clean(ax); panel(ax, "(c)", x=-0.14)

    # (d) leave-one-molecule-out-of-library
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(9)[::-1]
    for i, a in enumerate(AGENTS):
        yy = y[i]
        ax.barh(yy, RL.loc[a, "score"], color=CCOL[KLS[a]], height=0.62, alpha=0.92)
        ax.text(RL.loc[a, "score"] + 0.006, yy, label(RL.loc[a, "top1"]),
                va="center", ha="left", fontsize=LG - 3, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(AGENTS, fontsize=LG - 2)
    ax.set_xlim(0.55, 1.06)
    ax.set_xlabel("Similarity to best remaining entry", fontsize=AX - 1)
    ax.set_title("True molecule removed from library\nclass recovered 9 / 9", fontsize=AX - 2, pad=12)
    clean(ax); panel(ax, "(d)", x=-0.16, y=1.20)

    # (e) margins
    ax = fig.add_subplot(gs[2, 0])
    mg = [R.loc[a, "gold_margin"] for a in AGENTS]
    cols = [CCOL[KLS[a]] if m > 0 else "#BFB4A8" for a, m in zip(AGENTS, mg)]
    ax.bar(range(9), mg, color=cols, width=0.66)
    ax.axhline(0, color="#555", lw=2)
    for i, m in enumerate(mg):
        ax.text(i, m + (0.004 if m > 0 else -0.004), f"{m:+.3f}", ha="center",
                va="bottom" if m > 0 else "top", fontsize=LG - 4)
    ax.set_xticks(range(9)); ax.set_xticklabels(AGENTS, rotation=45, ha="right", fontsize=LG - 3)
    ax.set_ylabel("Margin over best\nincorrect match", fontsize=AX - 2)
    ax.set_ylim(-0.118, 0.125)
    ax.text(0.985, 0.97, "grey = molecule missed\n(every miss stays within class)", transform=ax.transAxes,
            ha="right", va="top", fontsize=LG - 3, color="#555")
    clean(ax); panel(ax, "(e)", x=-0.14)

    # (f) abstention operating characteristic
    ax = fig.add_subplot(gs[2, 1])
    ab = D["absten"]
    th = [a[0] for a in ab]; kept = [a[1] for a in ab]
    ax.plot(th, [a[3] for a in ab], "-o", ms=13, lw=3.2, color=TEAL, label="class precision")
    ax.plot(th, [a[2] for a in ab], "-s", ms=12, lw=3.2, color=RED, label="molecule precision")
    for t, k in zip(th, kept):
        ax.text(t, 108.5, f"{k}/9", ha="center", va="bottom", fontsize=LG - 4, color="#555")
    ax.text(0.5, 1.075, "queries judged (remainder abstained)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=LG - 3, color="#555")
    ax.set_ylim(28, 118)
    ax.set_xlabel("Abstention threshold on margin", fontsize=AX - 1)
    ax.set_ylabel("Precision (%)", fontsize=AX - 1)
    ax.legend(fontsize=LG - 1, loc="center right", frameon=True, framealpha=0.95, edgecolor="#CCC")
    clean(ax); panel(ax, "(f)", x=-0.16)

    out = os.path.join(ROOT, "figures", "fig4_identification.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig4 ->", out)


# =====================================================================  FIG 5
def fig5():
    spec, meas = D["spec"], D["meas"]
    fails = D["fails"]
    order = [a for a in AGENTS if a in fails]

    fig = plt.figure(figsize=(20, 13.5))
    gs = fig.add_gridspec(3, 2, hspace=0.66, wspace=0.30,
                          left=0.085, right=0.965, top=0.92, bottom=0.06,
                          height_ratios=[1.15, 1.15, 1])

    # (a-d) region-resolved decision for each failure
    tags = "abcd"
    for k, ag in enumerate(order):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        rows = fails[ag]["rows"]
        names = [r[0].replace("\n", "\n") for r in rows]
        wts = np.array([r[1] for r in rows]) * 100
        gold = [r[2] for r in rows]; wrong = [r[3] for r in rows]
        x = np.arange(5); w = 0.36
        ax.bar(x - w / 2, gold, w, color=TEAL, label=f"correct ({ag})")
        ax.bar(x + w / 2, wrong, w, color=RED, label=f"competitor ({fails[ag]['wrong']})")
        for i in range(5):
            if wrong[i] > gold[i]:
                ax.plot(x[i], max(gold[i], wrong[i]) + 0.055, "v", ms=13, color="#1F1F1F")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{wt:.0f}% wt" for n, wt in zip(names, wts)], fontsize=LG - 6)
        ax.set_ylim(0, 1.20); ax.set_ylabel("Regional cosine", fontsize=AX - 3)
        ax.set_title(f"{ag}  →  {fails[ag]['wrong']}", fontsize=AX - 1, pad=12)
        ax.legend(fontsize=LG - 5, loc="upper left", ncol=2, frameon=True,
                  framealpha=0.95, edgecolor="#CCC", columnspacing=0.9, handlelength=1.2)
        clean(ax); panel(ax, f"({tags[k]})", x=-0.13, y=1.22)

    # (e) the unified mechanism: margin lost in the P-F/P-N window
    ax = fig.add_subplot(gs[2, 0])
    regs = [r[0].replace("\n", " ") for r in fails[order[0]]["rows"]]
    Wn = np.array([[r[3] - r[2] for r in fails[a]["rows"]] for a in order])
    xx = np.arange(5)
    for i, a in enumerate(order):
        ax.plot(xx, Wn[i], "-o", ms=11, lw=2.8, color=[RED, ORG, TEAL, BLU][i], label=a)
    ax.axhline(0, color="#555", lw=2)
    ax.axvspan(0.62, 1.38, color="#E8943A", alpha=0.16, zorder=0)
    ax.text(1.0, 0.335, "P–F / P–N", ha="center", va="top", fontsize=LG - 1,
            color="#8A5A1E", fontweight="bold")
    ax.set_xticks(xx); ax.set_xticklabels([r.split()[0] for r in regs], fontsize=LG - 3)
    ax.set_ylabel("Competitor advantage\n(wrong − correct)", fontsize=AX - 3)
    ax.set_xlabel("Spectral region", fontsize=AX - 2)
    ax.set_ylim(-0.30, 0.42)
    ax.legend(fontsize=LG - 4, loc="lower left", ncol=2, frameon=True,
              framealpha=0.95, edgecolor="#CCC", columnspacing=0.9)
    clean(ax); panel(ax, "(e)", x=-0.15, y=1.18)

    # (f) computed-computed similarity among the Novichok agents
    ax = fig.add_subplot(gs[2, 1])
    NOV, DD = D["NOV"], np.array(D["DD"])
    Md = DD.copy(); np.fill_diagonal(Md, np.nan)
    im = ax.imshow(Md, cmap=WARM, vmin=0.70, vmax=0.95, aspect="auto")
    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f"{DD[i,j]:.3f}", ha="center", va="center", fontsize=LG - 5,
                        color="white" if DD[i, j] > 0.875 else "#333")
        ax.text(i, i, "—", ha="center", va="center", fontsize=LG, color="#999")
    ax.set_xticks(range(4)); ax.set_xticklabels(NOV, fontsize=LG - 4)
    ax.set_yticks(range(4)); ax.set_yticklabels(NOV, fontsize=LG - 4)
    ax.set_title("Computed spectra only", fontsize=AX - 3, pad=14)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("cosine similarity", fontsize=AX - 4); cb.ax.tick_params(labelsize=TK - 2)
    ax.tick_params(labelsize=TK - 2); panel(ax, "(f)", x=-0.19, y=1.30)

    out = os.path.join(ROOT, "figures", "fig5_failure_analysis.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig5 ->", out, "| failures:", {a: fails[a]["wrong"] for a in order})


if __name__ == "__main__":
    fig4()
    fig5()
