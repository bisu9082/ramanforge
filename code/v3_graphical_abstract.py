import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
BLUE="#2a78d6"; ORANGE="#eb6834"; INK="#1a1a19"; INK2="#5c5b55"; GRID="#e6e4dd"; BG="#fcfcfb"
X=np.load("/tmp/ga_X.npy"); A="A232"
meas=np.load(f"/tmp/ga_meas_{A}.npy"); rep=np.load(f"/tmp/ga_rep2_{A}.npy"); dft=np.load(f"/tmp/ga_dft_{A}.npy")
meas/=meas.max(); rep/=rep.max(); dft/=dft.max()

fig=plt.figure(figsize=(12.5,5.0),dpi=200,facecolor=BG)
fig.text(0.030,0.930,"A computed Raman library does not replace measured reference spectra",
         fontsize=16,color=INK,fontweight="bold",va="bottom")
fig.text(0.030,0.875,"Eleven authentic chemical warfare agents at 1064 nm  ·  same queries, same pipeline, same metric  ·  only the library differs",
         fontsize=9.8,color=INK2,va="bottom")

# -- left: the query, aligned to the mid-height of the two panels
axq=fig.add_axes([0.030,0.300,0.200,0.420]); axq.set_facecolor(BG)
axq.plot(X,meas,color=INK,lw=1.6); axq.fill_between(X,0,meas,color=INK,alpha=0.07)
axq.set_xlim(250,1800); axq.set_ylim(-0.03,1.12); axq.set_yticks([])
axq.set_xticks([400,800,1200,1600]); axq.tick_params(colors=INK2,labelsize=8.2,length=2)
for s in ("top","right","left"): axq.spines[s].set_visible(False)
axq.spines["bottom"].set_color(GRID)
fig.text(0.030,0.760,"QUERY",fontsize=8.6,color=INK2,fontweight="bold")
fig.text(0.030,0.728,"one acquisition of an authentic agent",fontsize=9.4,color=INK)
fig.text(0.030,0.218,"Raman shift (cm$^{-1}$)",fontsize=8.4,color=INK2)
fig.text(0.030,0.190,"the thin black trace is this query, repeated on both panels;\nthe thick coloured trace is the library entry it is compared with",fontsize=8.6,color=INK2,va="top",linespacing=1.5)

def panel(y, ref, col, title, cosv):
    ax=fig.add_axes([0.320,y,0.360,0.200]); ax.set_facecolor(BG)
    ax.fill_between(X,meas,ref,color=col,alpha=0.20,zorder=1)
    ax.plot(X,ref,color=col,lw=2.6,alpha=0.95,zorder=2)
    ax.plot(X,meas,color=INK,lw=1.2,zorder=3)
    ax.set_xlim(250,1800); ax.set_ylim(-0.05,1.15)
    ax.set_yticks([]); ax.set_xticks([400,800,1200,1600])
    ax.tick_params(colors=INK2,labelsize=8.2,length=2)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.text(0.0,1.08,title,transform=ax.transAxes,fontsize=10.4,color=col,
            fontweight="bold",va="bottom")
    ax.text(1.0,1.08,f"cosine {cosv}",transform=ax.transAxes,fontsize=10.4,color=col,
            ha="right",va="bottom",fontweight="bold")
    return ax
pt=panel(0.520, rep,  BLUE,   "measured reference library", "1.00")
pb=panel(0.195, dft,  ORANGE, "computed library",           "0.65")
pb.text(0.0,-0.22,"shaded area is the disagreement  ·  A-232 shown; the set mean is 0.67 against 0.98 for the measured library",
        transform=pb.transAxes,fontsize=8.4,color=INK2,va="top")


def score(y,val,col,lab,sub):
    ax=fig.add_axes([0.745,y,0.233,0.200]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02,0.08),0.96,0.84,
        boxstyle="round,pad=0.02,rounding_size=0.07",transform=ax.transAxes,
        facecolor=col,alpha=0.10,edgecolor=col,linewidth=1.5))
    ax.text(0.095,0.50,val,transform=ax.transAxes,fontsize=26,color=col,fontweight="bold",va="center")
    ax.text(0.545,0.62,lab,transform=ax.transAxes,fontsize=9.3,color=INK,va="center")
    ax.text(0.545,0.37,sub,transform=ax.transAxes,fontsize=8.3,color=INK2,va="center")
score(0.520,"30/30",BLUE,  "acquisitions","identified correctly")
score(0.195,"13/30",ORANGE,"acquisitions","and no repair closes it")

for y in (0.620,0.295):
    fig.add_artist(FancyArrowPatch((0.243,y),(0.305,y),transform=fig.transFigure,
        arrowstyle="-|>",mutation_scale=13,color=INK2,lw=1.3,alpha=0.55))
    fig.add_artist(FancyArrowPatch((0.692,y),(0.738,y),transform=fig.transFigure,
        arrowstyle="-|>",mutation_scale=13,color=INK2,lw=1.3,alpha=0.55))

fig.text(0.030,0.052,"Computed library: B3LYP/6-311++G(d,p), 54 members, calibrated on band positions alone with no parameter fitted to identification.",
         fontsize=9.0,color=INK2)
fig.text(0.030,0.018,"The best of twelve similarity metrics reaches 16 of 30; two corrected structures, an affine wavenumber axis, a $\pm$30 cm$^{-1}$ offset scan and symmetric preprocessing all leave it at 13.",
         fontsize=9.0,color=INK2)
fig.savefig("/mnt/user-data/outputs/graphical_abstract.png",dpi=240,facecolor=BG)
fig.savefig("/mnt/user-data/outputs/graphical_abstract.pdf",facecolor=BG)
print("saved")
