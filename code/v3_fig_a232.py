"""
Figure S-a232: the A-232 improvement decomposed into conformer and configuration.

The three structures are those of Table S-a232: the geometry published with the original
report (Z, an arbitrary conformer), the lowest-energy Z conformer, and the global minimum,
which is the E configuration. Energies are B3LYP/6-311++G(d,p) relative to the global
minimum and the cosines are over the 600-850 cm-1 window with the calibration held at its
pre-correction values, so that only the structure changes between points.

The lowest-Z-conformer single point is not among the deposited ORCA outputs (only the two
geometries that enter the library are: orca/outputs/EXP_A232.out and EXP_A232_Eglob.out),
so the three values are carried here as constants and are the same numbers as Table
S-a232. The conformer search itself is deposited under conformers/.

Writes figures/v3_fig_a232.png.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
BLUE="#2a78d6"; ORANGE="#eb6834"; INK="#1a1a19"; INK2="#5c5b55"; GRID="#dedcd5"
E=[9.47,5.32,0.00]; C=[0.199,0.146,0.446]
L=["paper geometry\n(Z, arbitrary conformer)","Z global-minimum\nconformer","E global minimum\n(true minimum)"]
fig,ax=plt.subplots(figsize=(7.4,4.8))
ax.plot(E,C,"-",color=INK2,lw=1.2,alpha=0.45,zorder=1)
for i,(e,c) in enumerate(zip(E,C)):
    ax.scatter(e,c,s=150,color=[ORANGE,ORANGE,BLUE][i],edgecolor="white",linewidth=1.8,zorder=4)
ax.annotate("",xy=(E[1],C[1]),xytext=(E[0],C[0]),
            arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=2.2,mutation_scale=15,shrinkA=9,shrinkB=9),zorder=3)
ax.annotate("",xy=(E[2],C[2]),xytext=(E[1],C[1]),
            arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=2.2,mutation_scale=15,shrinkA=9,shrinkB=9),zorder=3)
ax.text(7.35,0.213,"fix the conformer\n$-$4.15 kcal/mol\n$\\Delta$cos $-$0.053",fontsize=8.8,color=ORANGE,
        ha="center",va="bottom",linespacing=1.5)
ax.text(2.45,0.335,"fix the E/Z configuration\n$-$5.32 kcal/mol\n$\\Delta$cos $+$0.300",fontsize=8.8,color=BLUE,
        ha="center",va="center",linespacing=1.5)
off=[(0,16),(0,16),(0,16)]
for i,(e,c) in enumerate(zip(E,C)):
    ax.annotate(L[i],(e,c),textcoords="offset points",xytext=off[i],fontsize=8.5,
                color=INK2,ha="center",va="bottom",linespacing=1.4)
ax.set_xlim(10.6,-1.3); ax.set_ylim(0.09,0.60)
ax.set_xlabel("B3LYP energy above the true minimum  (kcal mol$^{-1}$)  —  better geometry →",
              fontsize=9.5,color=INK)
ax.set_ylabel("A-232  600–850 cm$^{-1}$ window cosine",fontsize=9.5,color=INK)
ax.set_title("A-232: the gain is configuration, not conformer",fontsize=11.5,color=INK,loc="left",pad=12)
ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2,labelsize=9)
ax.text(0.015,0.045,"a better geometry is not monotonically a better spectrum",
        transform=ax.transAxes,fontsize=8.6,color=INK2,style="italic")
fig.tight_layout(pad=1.3)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig.savefig(os.path.join(ROOT, "figures", "v3_fig_a232.png"), dpi=200, facecolor="#fcfcfb")
print("saved figures/v3_fig_a232.png")
