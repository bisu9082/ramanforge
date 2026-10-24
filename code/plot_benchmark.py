"""
ramanforge Step 4-c
AL benchmark learning curve plot (data/al_benchmark.csv -> figure/al_learning_curve.png).

Left: # DFT-labeled molecules vs held-out spectral RMSE
Right: # DFT-labeled molecules vs identification accuracy
Seed mean per strategy + standard deviation band, full = horizontal ceiling (dashed).

Ku figure rules applied: earth-tone palette, no overlap, panel labels on the same line.
Run:  python scripts/plot_benchmark.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- approved earth-tone palette ----
COL = {
    "ucb_diversity": "#C94F4A",   # AL (proposed)
    "ucb_only":      "#E8943A",
    "diversity_only":"#4AACB0",
    "random":        "#5B8DB8",
    "full":          "#7E6BA8",
}
LAB = {
    "ucb_diversity": "AL (unc + div)",
    "ucb_only":      "uncertainty only",
    "diversity_only":"diversity only",
    "random":        "random",
    "full":          "full (LOO ceiling)",
}
ORDER = ["ucb_diversity", "ucb_only", "diversity_only", "random"]

# font scale (approved family; reduced slightly for a two-panel layout)
FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 20, 14, 13, 24


def _agg(df, strat, metric):
    sub = df[df.strategy == strat]
    g = sub.groupby("n_labeled")[metric].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def main(root: Path):
    df = pd.read_csv(root / "data" / "al_benchmark.csv")
    df = df.dropna(subset=["strategy"])
    n_mol = int(df.n_labeled.max())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    plt.subplots_adjust(wspace=0.28)

    panels = [("rmse", "Held-out spectral RMSE", "(a)"),
              ("acc", "Identification accuracy", "(b)")]
    for ax, (metric, ylab, plab) in zip(axes, panels):
        for s in ORDER:
            g = _agg(df, s, metric)
            if g.empty:
                continue
            ax.plot(g.n_labeled, g["mean"], "-o", color=COL[s], lw=2.0,
                    ms=6, label=LAB[s], zorder=3)
            ax.fill_between(g.n_labeled, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=COL[s], alpha=0.15, zorder=1)
        # full = the horizontal ceiling
        fu = df[df.strategy == "full"][metric]
        if len(fu):
            ax.axhline(fu.mean(), color=COL["full"], ls="--", lw=1.8,
                       label=LAB["full"], zorder=2)
        ax.set_xlabel("# DFT-labeled molecules", fontsize=FS_LABEL)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.text(-0.02, 1.14, plab, transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight="bold", ha="left", va="top")
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=FS_LEG, frameon=False, loc="upper right")

    status = "" if n_mol >= 40 else f"  (preliminary: {n_mol}/44 molecules)"
    fig.suptitle(f"Active-learning replay on OP-F Raman pool{status}",
                 fontsize=FS_LABEL, y=1.02)
    out = root / "figures" / "al_learning_curve.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[saved] {out}")

    # console summary
    print(f"\nSummary over {n_mol} molecules:")
    for s in ORDER + ["full"]:
        sub = df[df.strategy == s]
        if len(sub) == 0:
            continue
        last = sub[sub.n_labeled == sub.n_labeled.max()]
        print(f"  {LAB[s]:22s} final RMSE {last.rmse.mean():.4f}  acc {last.acc.mean():.3f}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
