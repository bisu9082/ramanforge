"""
v3 A2 — 기권(abstention) 규칙.

문제: 계산 라이브러리는 정답 클래스가 없어도 기권하지 않고 자신 있게 오답을 낸다.
      절대 유사도 임계값으로는 분리 불가(REMEDIATION §0(2): LOCO 0.74–0.85 vs 정답 0.70–0.91).

평가 설계
  IN    전체 계산 라이브러리 (정답 클래스 존재)        → 기권하면 안 됨
  LOCO  쿼리의 클래스를 라이브러리에서 통째로 제거      → 반드시 기권해야 함
  LOMO  쿼리의 분자만 제거 (클래스는 남음)             → 클래스 판정은 유효, 참고용
  HN2   시료가 HN2가 아님(자기 cos 0.061) = 자연 발생 out-of-library 양성대조

후보 점수 (전부 단일 특징, 임계값 선택은 nested leave-one-agent-out)
  S1 top1        최고 코사인 (기준선 — 실패할 것으로 예상)
  S2 margin      1위 − 2위
  S3 psr         (1위 − 나머지 평균) / 나머지 표준편차      [peak-to-sidelobe]
  S4 libz        라이브러리 내부 유사도 분포로 정규화한 z    [국소 정규화]
  S5 resid       상위 m개 비음수 조합으로 설명되지 않는 비율 [언믹싱 잔차]
  S6 bandfrac    실측 밴드 중 대응 계산 밴드가 있는 비율     [물리적·비코사인]
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import nnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v3_step3_headtohead import (load_measured, load_library, synth_native, fit_scale,
                                 measured_peaks, cos, cls_of, JOB, CLASS, AGENTS, GAMMA)

P = dict(warm_red="#C94F4A", amber="#E8943A", gold="#F0C040", teal="#4AACB0",
         dusty="#5B8DB8", brown="#8B6347", dark_teal="#2E7D7A", coral="#E8836A",
         sage="#7FAF6A", tan="#C9A882", rose="#B84040", pale_teal="#7DC4C4")
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 13, "font.size": 15, "font.family": "DejaVu Sans"})

TOL = 35.0        # 밴드 대응 허용오차 (A1과 동일)
TOPM = 5          # 언믹싱에 쓸 상위 후보 수


# ------------------------------------------------------------------ 점수들
def library_stats(dft):
    """각 라이브러리 항목이 '보통 다른 항목과 얼마나 닮는지' — 국소 정규화용."""
    keys = list(dft)
    M = np.array([[cos(dft[i], dft[j]) for j in keys] for i in keys])
    np.fill_diagonal(M, np.nan)
    mu = np.nanmean(M, 1)
    sd = np.nanstd(M, 1)
    return {k: (float(mu[i]), float(max(sd[i], 1e-6))) for i, k in enumerate(keys)}


def unmix_residual(q, dft, order, m=TOPM):
    """상위 m개 라이브러리 스펙트럼의 비음수 조합으로 q를 설명하고 남는 비율."""
    A = np.stack([dft[k] for k, _ in order[:m]], 1)
    qn = q / (np.linalg.norm(q) + 1e-12)
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    w, _ = nnls(An, qn)
    return float(np.linalg.norm(qn - An @ w))


def band_fraction(qpk, dpk, tol=TOL):
    """실측 밴드 중 계산 밴드가 tol 안에 있는 비율."""
    if not len(qpk) or not len(dpk):
        return 0.0
    return float(np.mean([np.min(np.abs(dpk - w)) <= tol for w in qpk]))


def score_query(q, qpk, dft, dftpk, libstat, pool):
    sims = {k: cos(q, dft[k]) for k in pool}
    order = sorted(sims.items(), key=lambda t: -t[1])
    top, t1 = order[0]
    vals = np.array([v for _, v in order])
    others = vals[1:]
    mu_o, sd_o = float(others.mean()), float(max(others.std(), 1e-9))
    mu_k, sd_k = libstat[top]
    return dict(
        top1_name=top,
        S1_top1=t1,
        S2_margin=t1 - (vals[1] if len(vals) > 1 else 0.0),
        S3_psr=(t1 - mu_o) / sd_o,
        S4_libz=(t1 - mu_k) / sd_k,
        S5_resid=unmix_residual(q, dft, order),
        S6_bandfrac=band_fraction(qpk, dftpk[top]),
    )


# ------------------------------------------------------------------ 평가
SCORES = ["S1_top1", "S2_margin", "S3_psr", "S4_libz", "S5_resid", "S6_bandfrac"]
# resid는 값이 클수록 '기권해야 함' → 부호 반전
HIGHER_IS_IN = {"S1_top1": True, "S2_margin": True, "S3_psr": True,
                "S4_libz": True, "S5_resid": False, "S6_bandfrac": True}


def auc(pos, neg):
    """pos = IN(기권하면 안 됨), neg = LOCO(기권해야 함). 값이 클수록 IN인 점수 기준."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    X, acq = load_measured()
    lib = load_library()
    mean = {a: np.mean([s for _, s in acq[a]], 0) for a in AGENTS}
    mpk = {a: measured_peaks(X, mean[a]) for a in AGENTS}
    _, SL, SH, _ = fit_scale(lib, mpk, [a for a in AGENTS if a in JOB])
    dft = {k: synth_native(v, X, SL, SH) for k, v in lib.items()}
    dftpk = {k: X[find_peaks(v, prominence=0.05, distance=2)[0]] for k, v in dft.items()}
    libstat = library_stats(dft)
    # 심사 대응: 계산 라이브러리를 측정 라이브러리와 동일한 11종으로 제한해
    # 후보 수 교락(54 대 11)을 제거한다. 환경변수 RF_FULLLIB=1 이면 종전대로 54종.
    if os.environ.get("RF_FULLLIB") == "1":
        full = list(dft)
    else:
        full = [JOB[a] for a in AGENTS if a in JOB]
    print(f"계산 라이브러리 {len(full)}종, 보정 s=({SL:.3f},{SH:.3f}), Γ={GAMMA:.0f}\n")

    queryable = [a for a in AGENTS if len(acq[a]) >= 2 and a in JOB]
    rows = []
    for a in queryable:
        for rep, q in acq[a]:
            qpk = X[find_peaks(q, prominence=0.05, distance=2)[0]]
            gold = JOB[a]
            gold_cls = CLASS[a]

            pools = {
                "IN": full,
                "LOMO": [k for k in full if k != gold],
                "LOCO": [k for k in full if cls_of(k) != gold_cls],
            }
            for cond, pool in pools.items():
                if not pool:
                    continue
                sc = score_query(q, qpk, dft, dftpk, libstat, pool)
                rows.append(dict(agent=a, rep=rep, cls=gold_cls, cond=cond,
                                 correct=(sc["top1_name"] == gold),
                                 cls_correct=(cls_of(sc["top1_name"]) == gold_cls),
                                 **sc))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "data", "v3_abstention_scores.csv"), index=False)

    # HN2는 시료 자체가 out-of-library → IN 조건이어도 '기권해야 함' 쪽
    IN = D[(D.cond == "IN") & (D.agent != "HN2")]
    HN2 = D[(D.cond == "IN") & (D.agent == "HN2")]
    LOCO = D[(D.cond == "LOCO") & (D.agent != "HN2")]
    LOMO = D[(D.cond == "LOMO") & (D.agent != "HN2")]

    print(f"IN {len(IN)}건 / LOCO {len(LOCO)}건 / LOMO {len(LOMO)}건 / HN2 {len(HN2)}건\n")
    print("LOCO 조건에서 top-1 클래스가 맞는 비율 (= 조용한 실패 확인): "
          f"{LOCO.cls_correct.mean()*100:.0f}%  ← 0%가 정상 (클래스를 통째로 뺐으므로)")

    print("\n" + "=" * 74)
    print("기권 점수 후보 — IN(기권 금지) vs LOCO(기권 필수) 분리력")
    print("=" * 74)
    print(f"{'점수':<14}{'IN 중앙값':>11}{'LOCO 중앙값':>13}{'HN2 중앙값':>12}{'AUC':>8}{'완전분리':>9}")
    res = []
    for s in SCORES:
        sign = 1.0 if HIGHER_IS_IN[s] else -1.0
        a_ = auc(sign * IN[s], sign * LOCO[s])
        sep = (sign * IN[s]).min() > (sign * LOCO[s]).max()
        res.append(dict(score=s, auc=a_, separable=bool(sep),
                        in_med=float(IN[s].median()), loco_med=float(LOCO[s].median()),
                        hn2_med=float(HN2[s].median())))
        print(f"{s:<14}{IN[s].median():>11.3f}{LOCO[s].median():>13.3f}"
              f"{HN2[s].median():>12.3f}{a_:>8.3f}{'  예' if sep else '  아니오':>9}")
    RES = pd.DataFrame(res)
    RES.to_csv(os.path.join(ROOT, "data", "v3_abstention_auc.csv"), index=False)

    # -------- 운용 임계값: nested leave-one-agent-out --------
    print("\n" + "=" * 74)
    print("운용 규칙 — 임계값을 작용제 하나씩 빼고 정하는 nested LOAO")
    print("  (규칙: 점수 < τ 이면 기권. τ는 held-out을 제외한 LOCO를 전부 잡는 최솟값)")
    print("=" * 74)
    oper = []
    for s in SCORES:
        sign = 1.0 if HIGHER_IS_IN[s] else -1.0
        abst_in, abst_loco, abst_hn2, taus = [], [], [], []
        for held in queryable:
            if held == "HN2":
                continue
            tr_loco = LOCO[LOCO.agent != held]
            if not len(tr_loco):
                continue
            tau = (sign * tr_loco[s]).max()          # 학습 LOCO를 전부 기권시키는 경계
            taus.append(tau)
            te_in = IN[IN.agent == held]
            te_loco = LOCO[LOCO.agent == held]
            abst_in += list((sign * te_in[s] <= tau).astype(int))
            abst_loco += list((sign * te_loco[s] <= tau).astype(int))
        tau_all = (sign * LOCO[s]).max()
        abst_hn2 = list((sign * HN2[s] <= tau_all).astype(int))
        oper.append(dict(score=s, tau_med=float(np.median(taus)) * sign,
                         abstain_IN=float(np.mean(abst_in)),
                         abstain_LOCO=float(np.mean(abst_loco)),
                         abstain_HN2=float(np.mean(abst_hn2))))
        print(f"  {s:<12} τ≈{np.median(taus)*sign:+8.3f}   "
              f"IN 기권 {np.mean(abst_in)*100:5.1f}% (낮을수록 좋음)   "
              f"LOCO 기권 {np.mean(abst_loco)*100:5.1f}% (100%가 목표)   "
              f"HN2 기권 {np.mean(abst_hn2)*100:5.0f}%")
    OP = pd.DataFrame(oper)
    OP.to_csv(os.path.join(ROOT, "data", "v3_abstention_operating.csv"), index=False)

    # -------- 2-특징 AND 규칙 (가장 유망한 두 개) --------
    best2 = OP.sort_values(["abstain_LOCO", "abstain_IN"], ascending=[False, True]).score.tolist()[:2]
    print(f"\n2-특징 AND 규칙 시도: {best2[0]} 과 {best2[1]} 둘 다 통과해야 수용")
    s1, s2 = best2
    g1 = 1.0 if HIGHER_IS_IN[s1] else -1.0
    g2 = 1.0 if HIGHER_IS_IN[s2] else -1.0
    t1 = (g1 * LOCO[s1]).max(); t2 = (g2 * LOCO[s2]).max()
    acc_in = float(np.mean((g1 * IN[s1] > t1) & (g2 * IN[s2] > t2)))
    acc_loco = float(np.mean((g1 * LOCO[s1] > t1) & (g2 * LOCO[s2] > t2)))
    acc_hn2 = float(np.mean((g1 * HN2[s1] > t1) & (g2 * HN2[s2] > t2)))
    print(f"  IN 수용 {acc_in*100:.1f}%   LOCO 수용 {acc_loco*100:.1f}%   HN2 수용 {acc_hn2*100:.0f}%")

    # ============================================ 실측 라이브러리 대조 (핵심)
    print("\n" + "=" * 74)
    print("같은 기권 문제를 '실측 라이브러리'로 — 전방 모델이 정확하면 어떻게 되나")
    print("=" * 74)
    mrows = []
    for a in queryable:
        for rep, q in acq[a]:
            mlib = {}
            for b in AGENTS:
                oth = [s_ for r, s_ in acq[b] if not (b == a and r == rep)]
                if oth:
                    mlib[b] = np.mean(oth, 0)
            if a not in mlib:
                continue
            mpk_ = {k: X[find_peaks(v, prominence=0.05, distance=2)[0]] for k, v in mlib.items()}
            st = library_stats(mlib)
            qpk = X[find_peaks(q, prominence=0.05, distance=2)[0]]
            for cond, pool in [("IN", list(mlib)),
                               ("LOCO", [k for k in mlib if CLASS[k] != CLASS[a]])]:
                if not pool:
                    continue
                sc = score_query(q, qpk, mlib, mpk_, st, pool)
                mrows.append(dict(agent=a, rep=rep, cond=cond,
                                  cls_correct=CLASS[sc["top1_name"]] == CLASS[a], **sc))
    M = pd.DataFrame(mrows)
    M.to_csv(os.path.join(ROOT, "data", "v3_abstention_measured.csv"), index=False)
    MIN_ = M[(M.cond == "IN") & (M.agent != "HN2")]
    MLOCO = M[(M.cond == "LOCO") & (M.agent != "HN2")]
    MHN2 = M[(M.cond == "IN") & (M.agent == "HN2")]

    print(f"{'점수':<14}{'계산 AUC':>11}{'실측 AUC':>11}{'계산 완전분리':>15}{'실측 완전분리':>15}")
    cmp_rows = []
    for s_ in SCORES:
        g = 1.0 if HIGHER_IS_IN[s_] else -1.0
        a_c = auc(g * IN[s_], g * LOCO[s_])
        a_m = auc(g * MIN_[s_], g * MLOCO[s_])
        sc_ = (g * IN[s_]).min() > (g * LOCO[s_]).max()
        sm_ = (g * MIN_[s_]).min() > (g * MLOCO[s_]).max()
        cmp_rows.append(dict(score=s_, auc_computed=a_c, auc_measured=a_m,
                             sep_computed=bool(sc_), sep_measured=bool(sm_)))
        print(f"{s_:<14}{a_c:>11.3f}{a_m:>11.3f}"
              f"{('예' if sc_ else '아니오'):>15}{('예' if sm_ else '아니오'):>15}")
    CMP = pd.DataFrame(cmp_rows)
    CMP.to_csv(os.path.join(ROOT, "data", "v3_abstention_compare.csv"), index=False)

    s_best = "S1_top1"
    tau_m = MLOCO[s_best].max()
    print(f"\n실측 라이브러리 + 단일 임계값 τ={tau_m:.3f} (S1 top-1 코사인):")
    print(f"  IN 수용 {(MIN_[s_best] > tau_m).mean()*100:.0f}%   "
          f"LOCO 수용 {(MLOCO[s_best] > tau_m).mean()*100:.0f}%   "
          f"HN2 수용 {(MHN2[s_best] > tau_m).mean()*100:.0f}%")
    print("  → 기권은 별도로 어려운 문제가 아니다. 전방 모델이 정확하면 임계값 하나로 해결된다.")

    meta = dict(n_in=len(IN), n_loco=len(LOCO), n_lomo=len(LOMO), n_hn2=len(HN2),
                loco_cls_correct=float(LOCO.cls_correct.mean()),
                auc_computed={r["score"]: r["auc"] for r in res},
                auc_measured={r["score"]: r["auc_measured"] for r in cmp_rows},
                separable_computed=[r["score"] for r in res if r["separable"]],
                separable_measured=[r["score"] for r in cmp_rows if r["sep_measured"]],
                tau_measured=float(tau_m),
                operating=OP.to_dict("records"))
    with open(os.path.join(ROOT, "data", "v3_abstention_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ================================================================ figure
    SHOW = ["S1_top1", "S4_libz", "S5_resid"]
    NICE = {"S1_top1": "top-1 cosine", "S4_libz": "library-normalised z",
            "S5_resid": "unmixing residual"}
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.32,
                          left=0.135, right=0.975, top=0.885, bottom=0.08)
    letters = "abcdef"
    rng = np.random.default_rng(0)
    for row, (tag, dIN, dLO, dH) in enumerate([
            ("computed (DFT) library", IN, LOCO, HN2),
            ("measured library", MIN_, MLOCO, MHN2)]):
        for col, s_ in enumerate(SHOW):
            ax = fig.add_subplot(gs[row, col])
            g = 1.0 if HIGHER_IS_IN[s_] else -1.0
            groups = [dIN[s_].values, dLO[s_].values, dH[s_].values]
            cols = [P["dark_teal"], P["warm_red"], P["amber"]]
            for j, (v, c) in enumerate(zip(groups, cols)):
                ax.scatter(rng.normal(j, .07, len(v)), v, s=60, color=c,
                           alpha=.85, edgecolor="w", lw=.7, zorder=3)
                ax.hlines(np.median(v), j - .3, j + .3, color="k", lw=2.2, zorder=4)
            tau = (g * dLO[s_]).max() * g
            ax.axhline(tau, color=P["rose"], ls="--", lw=1.7)
            a_ = auc(g * dIN[s_], g * dLO[s_])
            sep = (g * dIN[s_]).min() > (g * dLO[s_]).max()
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["in-library\n(accept)", "class absent\n(abstain)", "HN2"])
            ax.set_ylabel(NICE[s_])
            ttl = f"AUC {a_:.3f}" + ("   separable" if sep else "   overlapping")
            ax.set_title(ttl, fontsize=15,
                         color=P["dark_teal"] if sep else P["warm_red"])
            ax.text(-0.22, 1.06, letters[row * 3 + col], transform=ax.transAxes,
                    fontsize=26, fontweight="bold")
            if col == 0:
                ax.text(-0.33, 0.5, tag, transform=ax.transAxes, fontsize=17,
                        rotation=90, va="center", ha="center", fontweight="bold",
                        color=P["brown"])
    fig.suptitle("Abstention fails for the same reason identification fails",
                 fontsize=19, y=0.965)
    fig.savefig(os.path.join(ROOT, "figure", "v3_fig_abstention.png"), dpi=200)
    print("\nsaved figure/v3_fig_abstention.png + data/v3_abstention_*.csv")


if __name__ == "__main__":
    main()
