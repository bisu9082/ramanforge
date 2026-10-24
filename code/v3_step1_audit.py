"""v3 STEP 1: raw data audit (grid, exposure, replicate provenance)."""
import os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rawio import load_all

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize":16,"xtick.labelsize":15,"ytick.labelsize":15,
                     "legend.fontsize":13,"font.size":15,"font.family":"DejaVu Sans"})

def asls(y, lam=1e5, p=0.01, niter=20):
    y = np.asarray(y, float); L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2), dtype=float)
    D = lam*D.dot(D.T); w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w,0,L,L); z = spsolve((W+D).tocsc(), w*y)
        w = p*(y>z)+(1-p)*(y<z)
    return z

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = load_all(os.path.join(ROOT,"data","raw"))
rs = R[0]["df"]["RamanShift"].values
FP = (rs>=250)&(rs<=1800); NOISE=(rs>=2200)&(rs<=2560)

recs=[]
for r in R:
    m,d = r["meta"], r["df"]
    raw = d["RawData"].values.astype(float); sub = d["DarkSubed"].values.astype(float)
    genuine = np.abs(raw-np.round(raw)).max() < 1e-6
    x,y = rs[NOISE], sub[NOISE]; res = y-np.polyval(np.polyfit(x,y,1),x)
    noise = float(res.std())
    bl = asls(sub[FP]); corr = sub[FP]-bl
    pk,_ = find_peaks(corr, prominence=5*noise, distance=2)
    recs.append(dict(file=m["file"], agent=m["agent"], rep=m["rep"],
        genuine=genuine, IntTime_ms=m["IntTime"], Gain=m["Gain"], Offset=m["Offset"],
        CCDT=round(m["CCDTemperature"],4), raw_max=raw.max(), peak_counts=sub[FP].max(),
        noise=round(noise,1), SNR=round(sub[FP].max()/max(noise,1e-9),1),
        counts_per_s=round(sub[FP].max()/m["IntTime"]*1000,1),
        n_bands_5sigma=len(pk), top_band=round(float(rs[FP][int(np.argmax(corr))]),1),
        n_above_10pct=int((corr[pk]>=0.1*corr[pk].max()).sum()) if len(pk) else 0,
        n_above_20pct=int((corr[pk]>=0.2*corr[pk].max()).sum()) if len(pk) else 0,
        top_frac=round(float(corr[pk].max()/corr[pk].sum()),3) if len(pk) else np.nan))
Q = pd.DataFrame(recs).sort_values(["agent","rep"])
Q.to_csv(os.path.join(ROOT,"data","v3_acquisition_audit.csv"), index=False)

G = Q[Q.genuine]
agg = G.groupby("agent").agg(n_genuine=("rep","size"), n_files=("rep","size"),
    IntTime_ms=("IntTime_ms","mean"), peak_counts=("peak_counts","mean"),
    noise=("noise","mean"), SNR=("SNR","mean"), counts_per_s=("counts_per_s","mean"),
    n_bands=("n_bands_5sigma","mean"), n10=("n_above_10pct","mean"),
    n20=("n_above_20pct","mean"), top_frac=("top_frac","mean"),
    top_band=("top_band","median")).round(2)
agg["n_files"] = Q.groupby("agent").size()
agg["n_augmented"] = agg["n_files"]-agg["n_genuine"]
print(agg.to_string())
agg.to_csv(os.path.join(ROOT,"data","v3_agent_summary.csv"))

# ---------------- figure ----------------
fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(2,3, hspace=0.34, wspace=0.26)

ax = fig.add_subplot(gs[0,0])
ax.plot(rs[:-1], np.diff(rs), color=P["brown"], lw=2)
ax.axvspan(600,850, color=P["gold"], alpha=.30, label="P–F diagnostic 600–850")
ax.axvspan(250,1800, color=P["pale_teal"], alpha=.18, label="fingerprint 250–1800")
ax.set_xlim(0,2560); ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("native pixel spacing (cm$^{-1}$)")
ax.set_title("512-pixel detector: non-uniform sampling", fontsize=16)
ax.legend(loc="upper right", frameon=False)
ax.text(.03,.06,f"{int(FP.sum())} pixels in 250–1800\nmean {np.diff(rs)[FP[:-1]].mean():.1f} cm$^{{-1}}$",
        transform=ax.transAxes, fontsize=13, style="italic")
ax.text(-.13,1.04,"a",transform=ax.transAxes,fontsize=26,fontweight="bold")

ax = fig.add_subplot(gs[0,1])
o = agg.sort_values("IntTime_ms")
ax.barh(o.index, o.IntTime_ms, color=[P["warm_red"] if v>=20000 else P["teal"] for v in o.IntTime_ms])
ax.set_xscale("log"); ax.set_xlabel("integration time (ms, log)")
ax.set_title("Auto-exposure spans 252$\\times$", fontsize=16)
for i,(a,v) in enumerate(zip(o.index,o.IntTime_ms)):
    ax.text(v*1.15, i, f"{v:.0f}", va="center", fontsize=11)
ax.text(-.13,1.04,"b",transform=ax.transAxes,fontsize=26,fontweight="bold")

ax = fig.add_subplot(gs[0,2])
c = [P["warm_red"] if s<60 else P["sage"] for s in agg.SNR]
ax.scatter(agg.IntTime_ms, agg.SNR, s=190, c=c, edgecolor="k", lw=.8, zorder=3)
for a,row in agg.iterrows():
    ax.annotate(a,(row.IntTime_ms,row.SNR), xytext=(7,-4), textcoords="offset points", fontsize=12)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("integration time (ms)"); ax.set_ylabel("SNR (peak / high-shift noise)")
ax.set_title("A-230 and HD are exposure-limited", fontsize=16)
ax.text(-.15,1.04,"c",transform=ax.transAxes,fontsize=26,fontweight="bold")

ax = fig.add_subplot(gs[1,0])
cnt = Q.groupby(["agent","genuine"]).size().unstack(fill_value=0)
cnt = cnt.reindex(columns=[True,False], fill_value=0)
ax.bar(cnt.index, cnt[True], color=P["dark_teal"], label="genuine (integer counts)")
ax.bar(cnt.index, cnt[False], bottom=cnt[True], color=P["coral"], label="augmented copy")
ax.set_ylabel("files in data/raw"); ax.set_ylim(0,7.4); ax.set_yticks([0,1,2,3,4,5])
ax.tick_params(axis="x", rotation=45)
ax.set_title("Only 32/55 files carry integer counts", fontsize=16, pad=12)
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=12)
for i,a in enumerate(cnt.index):
    if cnt[True][a]==1: ax.text(i, 5.15, "n=1", ha="center", color=P["rose"], fontweight="bold", fontsize=13)
ax.text(-.11,1.04,"d",transform=ax.transAxes,fontsize=26,fontweight="bold")

ax = fig.add_subplot(gs[1,1])
for ag,col in [("HN3",P["dark_teal"]),("HN2",P["warm_red"])]:
    r = [x for x in R if x["meta"]["agent"]==ag and np.abs(x["df"]["RawData"].values-np.round(x["df"]["RawData"].values)).max()<1e-6][0]
    s = r["df"]["DarkSubed"].values[FP].astype(float); s = s-asls(s); s = s/s.max()
    ax.plot(rs[FP], s+(0 if ag=="HN2" else 1.15), color=col, lw=1.6, label=f"{ag} ({r['meta']['IntTime']:.0f} ms)")
ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("normalised intensity (offset)")
ax.set_yticks([]); ax.legend(frameon=False, loc="upper right")
ax.annotate("993.6", xy=(993.6,1.0), xytext=(1150,0.75), fontsize=13, color=P["rose"],
            arrowprops=dict(arrowstyle="->", color=P["rose"]))
ax.set_title("HN2: one band, at the best SNR of the set", fontsize=16)
ax.text(-.11,1.04,"e",transform=ax.transAxes,fontsize=26,fontweight="bold")

ax = fig.add_subplot(gs[1,2])
ax.scatter(agg.SNR, agg.n10, s=190, c=[P["warm_red"] if b<4 else P["teal"] for b in agg.n10],
           edgecolor="k", lw=.8, zorder=3)
for a,row in agg.iterrows():
    dx,dy = (7,-4) if a not in ("A234","GD") else (7,6)
    ax.annotate(a,(row.SNR,row.n10), xytext=(dx,dy), textcoords="offset points", fontsize=12)
ax.axhline(4, color=P["rose"], ls="--", lw=1.6)
ax.text(agg.SNR.min()*1.3, 4.6, "quality cut: $\\geq$4 bands at $\\geq$10% of max",
        color=P["rose"], fontsize=12)
ax.set_xscale("log"); ax.set_ylim(0,22)
ax.set_xlabel("SNR"); ax.set_ylabel("bands $\\geq$10% of strongest band")
ax.set_title("HN2 fails on spectral content, not on SNR", fontsize=16)
ax.text(-.15,1.04,"f",transform=ax.transAxes,fontsize=26,fontweight="bold")

fig.savefig(os.path.join(ROOT,"figures","v3_fig_raw_audit.png"), dpi=200, bbox_inches="tight")
print("\nsaved figures/v3_fig_raw_audit.png")
