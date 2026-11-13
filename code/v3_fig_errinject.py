"""Figure: how accurate a computed Raman reference must be (error-injection test).
Reads data/v3_error_injection*.csv/json written by v3_error_injection.py."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figfonts import scale_fonts

P = dict(red="#C94F4A", teal="#2E7D7A", tan="#C9A882", dusty="#5B8DB8", brown="#8B6347")
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 12, "font.family": "DejaVu Sans"})
D = pd.read_csv(os.path.join(ROOT, "data", "v3_error_injection.csv"))
T = pd.read_csv(os.path.join(ROOT, "data", "v3_error_injection_tau.csv"))
M = json.load(open(os.path.join(ROOT, "data", "v3_error_injection_meta.json")))
BP = os.path.join(ROOT, "data", "v3_error_injection_boot_meta.json")
BM = json.load(open(BP)) if os.path.exists(BP) else None
E, OP, OPP = M["dft"], M["operating_point"], M["operating_point_position_only"]

fig = plt.figure(figsize=(14, 6.0))
gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.07, right=0.97, top=0.90, bottom=0.14)

# ---- a : map over position error and band-set mismatch (intensities exact)
ax = fig.add_subplot(gs[0, 0])
piv = D.pivot(index="sigma", columns="m", values="mol_mean")
mr = D[D.sigma == 0].set_index("m").m_realised.reindex(piv.columns).values
S, Mg = np.meshgrid(piv.index.values, mr, indexing="ij")
cf = ax.contourf(Mg * 100, S, piv.values, levels=np.arange(8, 31, 2), cmap="viridis")
cs = ax.contour(Mg * 100, S, piv.values, levels=[24, 27], colors="white", linewidths=2.0)
ax.clabel(cs, fmt={24: "24/30", 27: "27/30"}, fontsize=12)
ax.plot(E["m_miss"] * 100, E["sigma"], "*", ms=20, mfc=P["red"], mec="white", mew=1.5)
ax.annotate("computed library", (E["m_miss"] * 100, E["sigma"]),
            xytext=(E["m_miss"] * 100 - 1.0, E["sigma"] + 0.08), color="white", fontsize=12.5,
            fontweight="bold", ha="right")
ax.set_xlabel("band-set mismatch (% of band intensity missing)")
ax.set_ylabel("band-position error (RMS, units of $\\Gamma$)")
ax.set_title("Molecular top-1 of 30, exact intensities", fontsize=14)
cb = fig.colorbar(cf, ax=ax, pad=0.02); cb.set_label("queries identified (of 30)")
ax.text(-0.13, 1.04, "a", transform=ax.transAxes, fontsize=22, fontweight="bold")

# ---- b : the three errors separately and together
ax = fig.add_subplot(gs[0, 1])
d0 = D[np.isclose(D.m, 0.0)]
ax.plot(d0.sigma, d0.mol_mean, "-", color=P["teal"], lw=2.6, label="position error only")
mop = M["m_op_nominal"]
for (tau, m), col, lab in [((round(E["tau"], 3), 0.0), P["dusty"], "+ intensity error ($\\tau$ = %.2f)" % E["tau"]),
                           ((0.0, None), P["tan"], None),
                           ((round(E["tau"], 3), round(mop, 3)), P["red"], "+ intensity and band-set error")]:
    if tau == 0.0:
        continue
    t = T[np.isclose(T.tau, tau, atol=1e-3) & np.isclose(T.m, m, atol=1e-3)]
    ax.plot(t.sigma, t.mol_mean, "-", color=col, lw=2.6, label=lab)
ax.axhline(13, color="k", ls=":", lw=1.6)
ax.text(0.02, 12.6, "observed: 13", ha="left", va="top", fontsize=12)
ax.errorbar([E["sigma"]], [OP["mol_mean"]], yerr=[[OP["mol_mean"] - OP["mol_p05"]], [OP["mol_p95"] - OP["mol_mean"]]],
            fmt="o", color=P["red"], ms=9, capsize=5, lw=2, zorder=5)
ax.axvline(E["sigma"], color="0.6", lw=1.2, ls="--")
ax.text(E["sigma"] - 0.015, 9.0, "computed library", fontsize=12, color="0.35", va="bottom", ha="right")
ax.axhline(27, color="0.6", lw=1.0)
if BM is not None:
    lo, hi = BM["sigma"]["ci95"]; xc = BM["sigma"]["crossing"]
    ax.errorbar([xc], [27], xerr=[[xc - lo], [hi - xc]], fmt="D", color=P["teal"], ms=7,
                capsize=6, lw=2.2, zorder=6)
    ax.text(0.015, 23.2, "27/30 crossing,\n95% CI (vials)", fontsize=11.5,
            color=P["teal"], va="top", ha="left")
ax.set_xlim(0, 1.0); ax.set_ylim(8, 31)
ax.set_xlabel("band-position error (RMS, units of $\\Gamma$)")
ax.set_ylabel("molecular top-1 (of 30)")
ax.set_title("The three errors compound", fontsize=14)
ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=12)
ax.text(-0.13, 1.04, "b", transform=ax.transAxes, fontsize=22, fontweight="bold")

print(f"smallest label at 7 in: {scale_fonts(fig, 1.0):.2f} pt")
out = os.path.join(ROOT, "figures", "v3_fig_errinject.png")
fig.savefig(out, dpi=250); print("saved", out)
