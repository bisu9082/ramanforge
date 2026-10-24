"""
Graphical abstract. Writes figures/graphical_abstract.png and .pdf.

Six conditions, all on the same thirty acquisitions under the same metric: the result as
reported, the best of the twelve similarity metrics, the affine wavenumber calibration, the
imposed axis offset, symmetric preprocessing, and the calibration refitted with each agent
held out. The counts are those of Sections 3.1 and 3.3-3.5 and of Tables S-metrics,
S-affine, S-offsetscan, S-aslssym and S-loaoident; they are written out here rather than
recomputed, since each comes from a different script.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
BLUE="#2a78d6"; ORANGE="#eb6834"; INK="#1a1a19"; INK2="#5c5b55"; GRID="#dedcd5"; BG="#fcfcfb"
# Elsevier recommends roughly 1328 x 531 px for the graphical abstract (2.5:1).
fig=plt.figure(figsize=(12.5,5.0), dpi=200, facecolor=BG)
ax=fig.add_axes([0.085,0.20,0.885,0.58]); ax.set_facecolor(BG)

COND=["as reported","best of 12\nsimilarity metrics","affine wavenumber\ncalibration",
      "axis offset imposed\n$-$20 to $+$30 cm$^{-1}$","preprocessing applied\nsymmetrically",
      "calibration refitted\nleaving each agent out"]
COMP=[13,16,13,13,13,13]
x=np.arange(len(COND))
ax.fill_between([-0.55,len(COND)-0.45],[13,13],[30,30],color=ORANGE,alpha=0.055,zorder=0)
ax.plot([-0.55,len(COND)-0.45],[30,30],color=BLUE,lw=2.6,zorder=3)
ax.plot(x,COMP,"-",color=ORANGE,lw=2.0,alpha=0.5,zorder=2)
ax.scatter(x,COMP,s=190,color=ORANGE,edgecolor="white",linewidth=2.0,zorder=4)
for xi,v in zip(x,COMP):
    ax.annotate(f"{v}",(xi,v),textcoords="offset points",xytext=(0,-22),
                ha="center",fontsize=11.5,color=ORANGE,fontweight="bold")
ax.text(-0.42,30.75,"measured reference library   30 / 30",fontsize=12,color=BLUE,
        va="bottom",fontweight="bold")
ax.text(len(COND)-0.52,21.3,"the gap no repair closes",fontsize=11.5,color=ORANGE,
        ha="right",style="italic")
ax.set_xticks(x); ax.set_xticklabels(COND,fontsize=9.3,color=INK2,linespacing=1.45)
ax.set_xlim(-0.55,len(COND)-0.45); ax.set_ylim(8,33.5)
ax.set_yticks([10,15,20,25,30]); ax.tick_params(colors=INK2,labelsize=9.5,length=0)
ax.set_ylabel("acquisitions identified\ncorrectly  (of 30)",fontsize=10.5,color=INK,linespacing=1.4)
ax.grid(True,axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
for s_ in ("top","right","bottom","left"): ax.spines[s_].set_visible(False)

fig.text(0.085,0.905,"A computed Raman library does not replace measured reference spectra",
         fontsize=15.5,color=INK,fontweight="bold",va="bottom")
fig.text(0.085,0.845,"Eleven vials of authentic chemical warfare agent  ·  1064 nm  ·  identical pipeline, grid and metric; only the library differs",
         fontsize=10.3,color=INK2,va="bottom")
fig.text(0.085,0.055,"Every entry is the same thirty acquisitions under the same metric, after two wrong computed structures were corrected.",
         fontsize=9.6,color=INK2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig.savefig(os.path.join(ROOT, "figures", "graphical_abstract.png"), dpi=200, facecolor=BG)
fig.savefig(os.path.join(ROOT, "figures", "graphical_abstract.pdf"), facecolor=BG)
print("saved figures/graphical_abstract.png and .pdf")
