"""v3 STEP 2 — 전처리 사양 확정 + 민감도 + 실측 선폭(Γ) 측정."""
import os, sys, itertools, json, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.rawio import load_all
from src import preprocess_v3 as PP

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 13, "font.size": 15, "font.family": "DejaVu Sans"})

AGENTS = ["GD", "GF", "VX", "HD", "HN1", "HN2", "HN3", "A230", "A232", "A234", "A242"]
CLASS = {"GD": "Nerve", "GF": "Nerve", "VX": "Nerve",
         "HD": "Blister", "HN1": "Blister", "HN2": "Blister", "HN3": "Blister",
         "A230": "Novichok", "A232": "Novichok", "A234": "Novichok", "A242": "Novichok"}


def load_genuine():
    R = load_all(os.path.join(ROOT, "data", "raw_files"))
    acq = []
    for r in R:
        m, d = r["meta"], r["df"]
        if not PP.is_genuine(d["RawData"].values.astype(float)):
            continue
        acq.append(dict(agent=m["agent"], rep=m["rep"], int_time=m["IntTime"],
                        gain=m["Gain"], offset=m["Offset"],
                        shift=d["RamanShift"].values.astype(float),
                        sub=d["DarkSubed"].values.astype(float)))
    return sorted(acq, key=lambda a: (a["agent"], a["rep"]))


def cosmic_sets(acq, w=1, k=6.0):
    """작용제별 복제본 합의로 우주선 픽셀 집합을 만든다. 부산물로 재현율도 반환."""
    by = collections.defaultdict(list)
    for i, a in enumerate(acq):
        by[a["agent"]].append(i)
    flags = [None] * len(acq)
    for ag, ids in by.items():
        fs = []
        for i in ids:
            a = acq[i]
            _, idx = PP.despike(a["sub"], PP.noise_sigma(a["shift"], a["sub"]), w, k)
            fs.append(set(idx.tolist()))
        cm = PP.cosmic_mask(fs)
        for i, c, f in zip(ids, cm, fs):
            flags[i] = dict(raw_flag=f, cosmic=c, n_rep=len(ids))
    return flags


# ------------------------------------------------------------------ Γ
def multi_lorentz(x, centers, amps, gamma, offset):
    g = gamma / 2.0
    y = np.full_like(x, offset, dtype=float)
    for c, a in zip(centers, amps):
        y = y + a * g * g / ((x - c) ** 2 + g * g)
    return y


def fit_shared_gamma(x, y, sigma_rel, max_peaks=24):
    """한 획득의 모든 검출 밴드에 공통 Γ 하나를 적합한다."""
    pk, props = find_peaks(y, prominence=5 * sigma_rel, distance=2)
    if len(pk) < 3:
        return None
    if len(pk) > max_peaks:
        pk = pk[np.argsort(-props["prominences"])[:max_peaks]]
        pk = np.sort(pk)
    c0 = x[pk].astype(float)
    a0 = y[pk].astype(float)
    dx = float(np.median(np.diff(x)))

    def resid(th):
        n = len(pk)
        c = c0 + th[:n] * dx            # 중심은 +-1 픽셀만 허용
        a = th[n:2 * n]
        return multi_lorentz(x, c, a, abs(th[-2]), th[-1]) - y

    th0 = np.concatenate([np.zeros(len(pk)), a0, [20.0, 0.0]])
    lo = np.concatenate([np.full(len(pk), -1.0), np.zeros(len(pk)), [4.0, -0.5]])
    hi = np.concatenate([np.full(len(pk), 1.0), a0 * 5 + 1e-6, [120.0, 0.5]])
    try:
        r = least_squares(resid, th0, bounds=(lo, hi), max_nfev=6000)
    except Exception:
        return None
    pred = y + r.fun
    r2 = 1 - (r.fun ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return dict(gamma=float(abs(r.x[-2])), r2=float(r2), n_bands=int(len(pk)))


def main():
    acq = load_genuine()
    flags = cosmic_sets(acq)
    X0 = acq[0]["shift"]
    DEAD = PP.detect_dead_pixels(X0, [a["sub"] for a in acq])
    print(f"진짜 획득 {len(acq)}건 / 11종\n")
    print("S2b 불량 픽셀 (사전 규칙: 편차가 90% 이상 획득에서 음 & 50% 이상에서 |z|>6)")
    Z = np.stack([PP.pixel_z(X0, a["sub"]) for a in acq])
    for i in DEAD:
        print(f"    픽셀 {i:>3}  {X0[i]:7.1f} cm-1   z 중앙값 {np.median(Z[:, i]):+6.1f}  "
              f"|z|>6 인 획득 {int((np.abs(Z[:, i])>6).sum())}/{len(acq)}")
    print("    → 813.7은 P–F 진단창(600–850) 상단, 1220.6은 1150–1400 창 내부에 있다.\n")

    # --- 우주선 진단 -------------------------------------------------
    nraw = sum(len(f["raw_flag"]) for f in flags)
    ncos = sum(len(f["cosmic"]) for f in flags)
    rep = []
    for f in flags:
        if f["n_rep"] > 1 and f["raw_flag"]:
            rep.append(1 - len(f["cosmic"]) / len(f["raw_flag"]))
    print("S2 우주선 제거 — 복제본 합의")
    print(f"  단일 스펙트럼 flag {nraw}건 중 독립 획득에서 재현되는 비율 "
          f"{np.mean(rep)*100:.1f}%  → 실제 신호")
    print(f"  비재현 = 우주선으로 판정 {ncos}건 "
          f"(획득당 {ncos/len(acq):.1f}px, {ncos/(len(acq)*512)*100:.2f}%)")
    solo = [a["agent"] for a, f in zip(acq, flags) if f["n_rep"] == 1]
    print(f"  ⚠️ 획득 1건이라 이 단계를 적용할 수 없는 작용제: {sorted(set(solo))}\n")

    # 여러 작용제에서 반복되는 픽셀 = 검출기 고정패턴 후보
    ap = collections.defaultdict(set)
    for a, f in zip(acq, flags):
        for px in f["raw_flag"]:
            ap[px].add(a["agent"])
    fixed = sorted([(px, len(v)) for px, v in ap.items() if len(v) >= 3], key=lambda t: -t[1])
    if fixed:
        print("  3종 이상에서 같은 픽셀이 flag됨 (검출기 고정패턴 후보):")
        for px, n in fixed:
            print(f"    픽셀 {px:>3}  {X0[px]:7.1f} cm-1  {n}종")
    print()

    # --- 전처리 ------------------------------------------------------
    proc = []
    for a, f in zip(acq, flags):
        r = PP.preprocess(a["shift"], a["sub"], a["int_time"], cosmic_px=f["cosmic"],
                          dead_px=DEAD)
        proc.append({**a, **r})
    X = proc[0]["shift"]
    print(f"분석 격자: {len(X)} 픽셀, {X[0]:.1f}–{X[-1]:.1f} cm-1, "
          f"간격 {np.diff(X).min():.2f}–{np.diff(X).max():.2f} (평균 {np.diff(X).mean():.2f})\n")

    # --- Γ -----------------------------------------------------------
    grows = []
    for p in proc:
        sig_rel = p["noise"] / max(p["corrected"].max(), 1e-9)
        f = fit_shared_gamma(p["shift"], p["spec"], sig_rel)
        if f and f["r2"] > 0.8:
            grows.append(dict(agent=p["agent"], rep=p["rep"], **f))
    W = pd.DataFrame(grows)
    W.to_csv(os.path.join(ROOT, "data", "v3_linewidths.csv"), index=False)
    GAMMA = float(np.round(W.gamma.median(), 0))
    print(f"Γ — 획득별 공통 선폭 적합 ({len(W)}/{len(proc)}건, R2>0.8)")
    print(f"  median {W.gamma.median():.1f}  IQR {W.gamma.quantile(.25):.1f}–"
          f"{W.gamma.quantile(.75):.1f}  range {W.gamma.min():.1f}–{W.gamma.max():.1f} cm-1")
    print("  작용제별:")
    print("   " + W.groupby("agent").agg(n=("rep", "size"), gamma=("gamma", "median"),
                                          bands=("n_bands", "median"), r2=("r2", "median")
                                          ).round(2).to_string().replace("\n", "\n   "))
    print(f"  => Γ = {GAMMA:.0f} cm-1 로 고정 (실측 스펙트럼에서만 적합, 식별 성능 미사용)\n")

    # --- 민감도 ------------------------------------------------------
    def spec_for(**kw):
        norm = kw.pop("norm", "max")
        out = {}
        for a, f in zip(acq, flags):
            r = PP.preprocess(a["shift"], a["sub"], a["int_time"], cosmic_px=f["cosmic"],
                              dead_px=DEAD, norm=norm, **kw)
            out.setdefault(a["agent"], []).append(r["spec"])
        return {k: np.mean(v, 0) for k, v in out.items()}

    base = spec_for()

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    def sep(s):
        ks = list(s)
        wi = [cos(s[i], s[j]) for i, j in itertools.combinations(ks, 2) if CLASS[i] == CLASS[j]]
        bw = [cos(s[i], s[j]) for i, j in itertools.combinations(ks, 2) if CLASS[i] != CLASS[j]]
        return float(np.mean(wi)), float(np.mean(bw))

    sens = []
    grid = ([("AsLS lambda", "lam", v) for v in [1e3, 3e3, 1e4, 3e4, 1e5, 1e6]] +
            [("AsLS p", "p", v) for v in [0.001, 0.005, 0.01, 0.02, 0.05]] +
            [("fingerprint low", "lo", v) for v in [200.0, 250.0, 300.0, 400.0]] +
            [("fingerprint high", "hi", v) for v in [1600.0, 1700.0, 1800.0]] +
            [("normalisation", "norm", v) for v in ["max", "snv"]])
    for label, key, val in grid:
        kw = {}
        if key == "lam": kw["lam"] = val
        elif key == "p": kw["p"] = val
        elif key == "lo": kw["fp"] = (val, PP.FP_HI)
        elif key == "hi": kw["fp"] = (PP.FP_LO, val)
        elif key == "norm": kw["norm"] = val
        s = spec_for(**kw)
        same_grid = key not in ("lo", "hi")
        c = float(np.mean([cos(base[a], s[a]) for a in base])) if same_grid else np.nan
        wi, bw = sep(s)
        sens.append(dict(parameter=label, value=val,
                         cos_vs_default=round(c, 4) if c == c else None,
                         within_class=round(wi, 3), between_class=round(bw, 3),
                         separation=round(wi - bw, 3)))
    S = pd.DataFrame(sens)
    S.to_csv(os.path.join(ROOT, "data", "v3_preproc_sensitivity.csv"), index=False)
    print("전처리 민감도 (작용제-평균 스펙트럼):")
    print(S.to_string(index=False))

    wi0, bw0 = sep(base)
    print(f"\n기본 사양 클래스 분리 (11종 평균): within {wi0:.3f}  between {bw0:.3f}  Δ {wi0-bw0:.3f}")
    comp = {"11종 전부": AGENTS,
            "HN2 제외 (10종)": [a for a in AGENTS if a != "HN2"],
            "HN2·VX 제외 (9종 = v2 구성)": [a for a in AGENTS if a not in ("HN2", "VX")],
            "HN2·HD·VX 제외 (8종, n>=3)": [a for a in AGENTS if a not in ("HN2", "HD", "VX")]}
    for name, ks in comp.items():
        w_, b_ = sep({k: base[k] for k in ks})
        print(f"   {name:<30} within {w_:.3f}  between {b_:.3f}  Δ {w_-b_:.3f}")

    from scipy.stats import mannwhitneyu
    pairs = [(p["agent"], p["spec"]) for p in proc]
    same = [cos(u, v) for (a1, u), (a2, v) in itertools.combinations(pairs, 2) if a1 == a2]
    wcl = [cos(u, v) for (a1, u), (a2, v) in itertools.combinations(pairs, 2)
           if a1 != a2 and CLASS[a1] == CLASS[a2]]
    bcl = [cos(u, v) for (a1, u), (a2, v) in itertools.combinations(pairs, 2)
           if CLASS[a1] != CLASS[a2]]
    print(f"\n획득 단위(32건, 증강본 0):")
    print(f"   같은 작용제 다른 획득 {np.mean(same):.3f} (n={len(same)})")
    print(f"   같은 클래스 다른 작용제 {np.mean(wcl):.3f} (n={len(wcl)})")
    print(f"   다른 클래스           {np.mean(bcl):.3f} (n={len(bcl)})")
    print(f"   MWU same-agent vs same-class p={mannwhitneyu(same, wcl).pvalue:.1e}; "
          f"same-class vs diff-class p={mannwhitneyu(wcl, bcl).pvalue:.1e}")
    acqsep = dict(same_agent=float(np.mean(same)), same_class=float(np.mean(wcl)),
                  diff_class=float(np.mean(bcl)),
                  p_same_vs_class=float(mannwhitneyu(same, wcl).pvalue),
                  p_class_vs_diff=float(mannwhitneyu(wcl, bcl).pvalue))

    # --- 저장 --------------------------------------------------------
    np.savez(os.path.join(ROOT, "data", "v3_measured.npz"),
             shift=X, agents=np.array(AGENTS),
             **{f"mean_{k}": v for k, v in base.items()},
             **{f"acq_{p['agent']}_{p['rep']}": p["spec"] for p in proc})
    meta = dict(n_genuine=len(acq), n_pixels=int(len(X)),
                grid_lo=float(X[0]), grid_hi=float(X[-1]),
                grid_step_mean=float(np.diff(X).mean()),
                gamma_measured=GAMMA,
                gamma_iqr=[float(W.gamma.quantile(.25)), float(W.gamma.quantile(.75))],
                n_gamma_fits=int(len(W)),
                cosmic_reproducibility=float(np.mean(rep)),
                cosmic_removed=int(ncos), raw_flagged=int(nraw),
                no_despike_agents=sorted(set(solo)),
                fixed_pattern_px=[[int(px), float(X0[px]), int(n)] for px, n in fixed],
                dead_px=[[int(i), float(X0[i])] for i in DEAD],
                class_sep=dict(within=wi0, between=bw0), acq_sep=acqsep,
                spec=dict(FP_LO=PP.FP_LO, FP_HI=PP.FP_HI, CR_W=1, CR_K=6.0,
                          ASLS_LAM=PP.ASLS_LAM, ASLS_P=PP.ASLS_P, ASLS_IT=PP.ASLS_IT,
                          NOISE_WIN=[PP.NOISE_LO, PP.NOISE_HI]))
    with open(os.path.join(ROOT, "data", "v3_preproc_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- figure ------------------------------------------------------
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    q = [p for p in proc if p["agent"] == "GF"][0]
    ax.plot(q["shift"], q["raw_fp"] / q["raw_fp"].max(), color=P["tan"], lw=1.5,
            label="S1 dark-subtracted")
    ax.plot(q["shift"], q["baseline"] / q["raw_fp"].max(), color=P["warm_red"], lw=2.2,
            ls="--", label="S4 AsLS background")
    ax.plot(q["shift"], q["corrected"] / q["corrected"].max() * .55 - .70,
            color=P["dark_teal"], lw=1.6, label="S4 output (offset)")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("normalised counts")
    for i in DEAD:
        ax.axvline(X0[i], color=P["gold"], lw=1.2, alpha=.9)
    ax.set_title("GF: pipeline stages\n(gold = masked dead pixels)", fontsize=15, pad=8)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.text(-.16, 1.10, "a", transform=ax.transAxes, fontsize=26, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar([0, 1], [nraw / len(acq), ncos / len(acq)],
           color=[P["warm_red"], P["dark_teal"]], width=.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["single-spectrum\nflag", "replicate-consensus\ncosmic"])
    ax.set_ylabel("pixels per acquisition")
    ax.set_title(f"{np.mean(rep)*100:.0f}% of naive flags are reproducible signal", fontsize=15)
    for i, v in enumerate([nraw / len(acq), ncos / len(acq)]):
        ax.text(i, v + .15, f"{v:.1f}", ha="center", fontsize=14)
    ax.text(-.17, 1.06, "b", transform=ax.transAxes, fontsize=26, fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(W.gamma, bins=np.arange(5, 65, 4), color=P["teal"], edgecolor="w")
    ax.axvline(GAMMA, color=P["rose"], lw=2.4)
    ax.text(GAMMA + 1.5, ax.get_ylim()[1] * .82, f"median {GAMMA:.0f}", color=P["rose"], fontsize=13)
    ax.axvline(np.diff(X).mean(), color=P["brown"], ls=":", lw=2)
    ax.text(np.diff(X).mean() - 1, ax.get_ylim()[1] * .45, "pixel\nspacing", color=P["brown"],
            fontsize=11, ha="right")
    ax.set_xlabel("$\\Gamma$ per acquisition (cm$^{-1}$)"); ax.set_ylabel("acquisitions")
    ax.set_title(f"Measured linewidth, n={len(W)} acquisitions", fontsize=16)
    ax.text(-.14, 1.06, "c", transform=ax.transAxes, fontsize=26, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    L = S[S.parameter == "AsLS lambda"]
    ax.semilogx(L.value, L.separation, "-o", color=P["dark_teal"], lw=2.2, ms=9)
    Pp = S[S.parameter == "AsLS p"]
    ax2 = ax.twiny()
    ax2.semilogx(Pp.value, Pp.separation, "-s", color=P["amber"], lw=2.2, ms=9, label="p")
    ax2.set_xlabel("AsLS $p$", color=P["amber"]); ax2.tick_params(colors=P["amber"])
    ax.axvline(PP.ASLS_LAM, color=P["rose"], ls="--", lw=1.8)
    ax.set_xlabel("AsLS $\\lambda$"); ax.set_ylabel("within − between class cosine")
    ax.set_title("Class separation vs background parameters", fontsize=14, pad=34)
    ax.plot([], [], "-o", color=P["dark_teal"], label="$\\lambda$ (bottom axis)")
    ax.plot([], [], "-s", color=P["amber"], label="$p$ (top axis)")
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.text(-.17, 1.16, "d", transform=ax.transAxes, fontsize=26, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    D = S.dropna(subset=["cos_vs_default"])
    cmap = {"AsLS lambda": P["dark_teal"], "AsLS p": P["amber"], "normalisation": P["sage"]}
    labs = [f"{r.parameter.split()[-1]}={r.value}" for _, r in D.iterrows()]
    ax.barh(range(len(D)), D.cos_vs_default, color=[cmap.get(p, P["tan"]) for p in D.parameter])
    ax.set_yticks(range(len(D))); ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlim(0.78, 1.004); ax.set_xlabel("cosine vs default specification")
    ax.axvline(0.99, color=P["rose"], ls="--", lw=1.5)
    ax.set_title("Analyst degrees of freedom, disclosed", fontsize=15)
    ax.invert_yaxis()
    ax.text(-.34, 1.06, "e", transform=ax.transAxes, fontsize=26, fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    CC = {"Nerve": P["warm_red"], "Blister": P["dark_teal"], "Novichok": P["amber"]}
    for i, a in enumerate(AGENTS):
        ax.plot(X, base[a] / base[a].max() * .92 + i, color=CC[CLASS[a]], lw=1.2)
        ax.text(X[-1] + 25, i + .30, a, fontsize=11, color=CC[CLASS[a]])
    ax.set_xlim(X[0], X[-1] + 200); ax.set_yticks([])
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_title("v3 measured set, 270-pixel native grid", fontsize=16)
    for k, c in CC.items():
        ax.plot([], [], color=c, lw=3, label=k)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.text(-.11, 1.06, "f", transform=ax.transAxes, fontsize=26, fontweight="bold")

    fig.savefig(os.path.join(ROOT, "figure", "v3_fig_preprocessing.png"),
                dpi=200, bbox_inches="tight")
    print("\nsaved figure/v3_fig_preprocessing.png + data/v3_measured.npz, "
          "v3_preproc_meta.json, v3_preproc_sensitivity.csv, v3_linewidths.csv")


if __name__ == "__main__":
    main()
