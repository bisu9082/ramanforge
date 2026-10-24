"""T1-18/T1-19 result figure: metric comparison and deficit component decomposition."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
P = dict(warm_red="#C94F4A", amber="#E8943A", teal="#4AACB0", dusty="#5B8DB8",
         brown="#8B6347", dark_teal="#2E7D7A", sage="#7FAF6A", rose="#B84040")
plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "font.size": 13, "font.family": "DejaVu Sans"})

# measured-library control under the identical metric; final values
MEAS = {"cosine": (30, 0.983), "derivative": (30, 0.981), "shift-tol": (30, 0.983),
        "HQI 10": (28, 0.801), "HQI 20": (29, 0.835), "HQI 35": (29, 0.856),
        "Dice 20": (28, 0.891), "Dice 35": (25, 0.898)}
COMP = {"cosine": (17, 0.687), "derivative": (11, 0.356), "shift-tol": (19, 0.729),
        "HQI 10": (5, 0.107), "HQI 20": (11, 0.216), "HQI 35": (11, 0.320),
        "Dice 20": (7, 0.453), "Dice 35": (10, 0.613)}
LBL = list(MEAS)

fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.5))

# (a) molecule-level top-1 by metric
a = ax[0, 0]; x = np.arange(len(LBL)); w = 0.38
a.bar(x - w/2, [MEAS[k][0] for k in LBL], w, color=P["dark_teal"], label="measured library")
a.bar(x + w/2, [COMP[k][0] for k in LBL], w, color=P["warm_red"], label="computed library")
for i, k in enumerate(LBL):
    a.text(i - w/2, MEAS[k][0] + 0.6, str(MEAS[k][0]), ha="center", fontsize=11, color=P["dark_teal"])
    a.text(i + w/2, COMP[k][0] + 0.6, str(COMP[k][0]), ha="center", fontsize=11, color=P["warm_red"])
a.set_xticks(x); a.set_xticklabels(LBL, rotation=30, ha="right")
a.set_ylabel("molecular top-1 (of 30)"); a.set_ylim(0, 34)
a.axhline(30, ls=":", lw=1, color="0.5")
a.legend(loc="lower right", frameon=False)
a.set_title("(a) No metric rescues the computed library", loc="left", fontsize=14)

# (b) self-similarity
b = ax[0, 1]
b.bar(x - w/2, [MEAS[k][1] for k in LBL], w, color=P["dark_teal"])
b.bar(x + w/2, [COMP[k][1] for k in LBL], w, color=P["warm_red"])
b.set_xticks(x); b.set_xticklabels(LBL, rotation=30, ha="right")
b.set_ylabel("similarity to the correct entry"); b.set_ylim(0, 1.08)
for i, k in enumerate(LBL):
    b.text(i, 1.02, f"{MEAS[k][1]-COMP[k][1]:+.2f}".replace("+", "−" if False else "Δ"),
           ha="center", fontsize=9, color="0.35")
b.set_title("(b) The deficit persists under every metric", loc="left", fontsize=14)

# (c) recovery under component decomposition
D = pd.read_csv(f"{DATA}/v3_t119_decompose.csv")
c = ax[1, 0]
comp = ["d_POS", "d_INT", "d_WID"]
names = ["position", "intensity", "linewidth"]
grp = [("phosphorus-containing", D[D.phos], P["warm_red"]), ("no phosphorus", D[~D.phos], P["teal"])]
xx = np.arange(len(names) + 1); w2 = 0.36
for gi, (lab, sub, col) in enumerate(grp):
    vals = [sub[k].mean() for k in comp] + [(sub.ALL - sub.BASE).mean()]
    c.bar(xx + (gi - 0.5) * w2, vals, w2, color=col, label=lab)
    for i, v in enumerate(vals):
        c.text(xx[i] + (gi - 0.5) * w2, v + (0.006 if v >= 0 else -0.016),
               f"{v:+.3f}", ha="center", fontsize=10, color=col)
c.axhline(0, color="0.3", lw=0.9)
c.set_xticks(xx); c.set_xticklabels(names + ["all three"])
c.set_ylabel("recovery in 600–850 cm$^{-1}$ cosine")
c.set_ylim(-0.04, 0.23); c.legend(loc="upper left", frameon=False)
c.set_title("(c) Linewidth contributes nothing; position dominates", loc="left", fontsize=14)

# (d) unmatched intensity
U = pd.read_csv(f"{DATA}/v3_t119_unmatched.csv")
d = ax[1, 1]; xu = np.arange(len(U))
d.bar(xu - w/2, U.unm_calc * 100, w, color=P["warm_red"], label="computed intensity with\nno measured partner")
d.bar(xu + w/2, U.unm_meas * 100, w, color=P["dark_teal"], label="measured intensity with\nno computed partner")
d.set_xticks(xu); d.set_xticklabels([s.replace("A2", "A-2") for s in U.agent], rotation=30, ha="right")
d.set_ylabel("share of window intensity (%)"); d.set_ylim(0, 118)
d.legend(loc="upper left", frameon=False, fontsize=11)
d.set_title("(d) The calculation adds bands the measurement does not show", loc="left", fontsize=14)

for A in ax.ravel():
    A.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = os.path.join(ROOT, "figure", "v3_fig_metrics_decomp.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
