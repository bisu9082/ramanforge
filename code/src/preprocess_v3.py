"""
ramanforge v3 — 전처리 사양 (명시적·재현가능).

설계 원칙
  P1  실측은 절대 업샘플하지 않는다. 검출기 native 512픽셀 축이 분석 격자다.
      (55파일 전부 동일 축임을 검증했다 — max dev = 0.0)
      DFT 합성 스펙트럼을 이 축으로 보간해 내린다.
  P2  모든 단계가 파라미터를 하나씩만 갖고, 전부 민감도 표로 공개된다.
  P3  증강 복제본은 쓰지 않는다. 정수 CCD 카운트를 갖는 획득만 쓴다.

단계
  S0  진짜 획득 선별          : RawData가 정수인 파일만
  S1  암전류 제거             : DarkSubed = RawData - Dark  (파일에 이미 있고 검증됨)
  S2  우주선 제거 (복제본 합의): 단일 스펙트럼 flag는 쓰지 않는다. 같은 작용제의
                                다른 획득에서 flag되지 않은 픽셀만 우주선으로 보고 보간한다.
                                근거: 단일 스펙트럼 기준으로 flag된 픽셀의 81%가
                                독립 획득에서 재현된다 = 실제 신호이지 우주선이 아니다.
                                획득이 1건뿐인 HD·VX는 이 단계를 적용할 수 없다(명시).
  S2b 불량픽셀 마스킹         : 전 획득에서 계통적으로 낮게 읽히는 픽셀(검출기 결함)을
                                자동 검출해 보간. 판정: 5점 중앙값 대비 편차가 획득의
                                DEAD_NEG_FRAC 이상에서 음이고, DEAD_HIT_FRAC 이상에서
                                |z| > DEAD_K.
  S3  지문영역 절단           : FP_LO ~ FP_HI cm-1  (native 격자 유지)
  S4  배경 제거               : AsLS (lam, p)
  S5  적산시간 정규화         : counts/s = I / IntTime * 1000   (보고용, 코사인 불변)
  S6  세기 정규화             : max=1  (기본) 또는 SNV
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ---- 사양 (논문 Experimental에 그대로 들어갈 값) -------------------------
FP_LO, FP_HI = 250.0, 1800.0
CR_W, CR_K = 2, 8.0          # 우주선: 폭 <=2 픽셀, 국소 잡음의 8배
ASLS_LAM, ASLS_P, ASLS_IT = 1.0e4, 0.01, 30
NOISE_LO, NOISE_HI = 2200.0, 2560.0   # 밴드 없는 고주파 창 (노이즈 추정)
DEAD_K, DEAD_NEG_FRAC, DEAD_HIT_FRAC = 6.0, 0.90, 0.50
LASER_NM, TEMP_K = 1064.0, 298.0


def is_genuine(raw: np.ndarray) -> bool:
    """원시 CCD 카운트는 정수. 비정수면 증강 복제본."""
    return bool(np.abs(raw - np.round(raw)).max() < 1e-6)


def _mad(x):
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def noise_sigma(shift: np.ndarray, y: np.ndarray) -> float:
    """고주파 무신호 창에서 1차 배경 제거 후 잔차 표준편차."""
    m = (shift >= NOISE_LO) & (shift <= NOISE_HI)
    x, v = shift[m], y[m]
    return float((v - np.polyval(np.polyfit(x, v, 1), x)).std())


def local_sigma(resid: np.ndarray, win: int = 51) -> np.ndarray:
    """국소 MAD — 산탄잡음이 세기에 따라 커지는 것을 흡수한다."""
    n = len(resid); half = win // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = resid[lo:hi]
        out[i] = np.median(np.abs(seg - np.median(seg))) * 1.4826
    return out


def despike(y: np.ndarray, sigma0: float, width: int = CR_W, k: float = CR_K):
    """
    폭 <= width 픽셀의 양/음 스파이크 제거.
    기준: 5점 중앙값과의 편차가 국소 잡음의 k배 초과이고,
          국소 잡음 = max(51점 이동 MAD, 고주파창 sigma0)  — 평탄구간에서
          MAD가 0으로 붕괴해 정상 픽셀이 대량 flag되는 것을 막는다.
          연속 이상픽셀 길이가 width 이하일 것.
    실제 라만 밴드는 FWHM이 픽셀 간격의 3배 안팎이라 width로 분리된다.
    반환: (정정된 y, 제거된 픽셀 인덱스)
    """
    y = np.asarray(y, float).copy()
    n = len(y)
    med = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        med[i] = np.median(y[lo:hi])
    resid = y - med
    s = np.maximum(local_sigma(resid), float(sigma0))
    flag = np.abs(resid) > k * s
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if (j - i) <= width:
                out[i:j] = True
            i = j
        else:
            i += 1
    if out.any():
        idx = np.arange(n)
        y[out] = np.interp(idx[out], idx[~out], y[~out])
    return y, np.where(out)[0]


def asls(y: np.ndarray, lam: float = ASLS_LAM, p: float = ASLS_P,
         niter: int = ASLS_IT) -> np.ndarray:
    """Eilers & Boelens 비대칭 최소제곱 배경."""
    y = np.asarray(y, float)
    L = len(y)
    D = sparse.diags([1., -2., 1.], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def snv(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    return (y - y.mean()) / (y.std() + 1e-12)


def cosmic_mask(flag_sets: list[set]) -> list[set]:
    """복제본 합의: 한 획득에서만 flag된 픽셀 = 우주선 후보."""
    out = []
    for i, s in enumerate(flag_sets):
        others = set().union(*[flag_sets[j] for j in range(len(flag_sets)) if j != i]) \
            if len(flag_sets) > 1 else set()
        out.append(s - others)
    return out


def apply_mask(y: np.ndarray, px: set) -> np.ndarray:
    y = np.asarray(y, float).copy()
    if not px:
        return y
    idx = np.arange(len(y))
    m = np.zeros(len(y), bool); m[sorted(px)] = True
    y[m] = np.interp(idx[m], idx[~m], y[~m])
    return y


def pixel_z(shift: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5점 중앙값 대비 편차 / 국소잡음 (하한 = 고주파창 sigma)."""
    y = np.asarray(y, float); n = len(y)
    med = np.array([np.median(y[max(0, i - 2):i + 3]) for i in range(n)])
    resid = y - med
    s = np.maximum(local_sigma(resid), noise_sigma(shift, y))
    return resid / s


def detect_dead_pixels(shift: np.ndarray, subs: list[np.ndarray],
                       k: float = DEAD_K, neg_frac: float = DEAD_NEG_FRAC,
                       hit_frac: float = DEAD_HIT_FRAC) -> list[int]:
    """전 획득에서 계통적으로 낮게 읽히는 검출기 불량 픽셀."""
    Z = np.stack([pixel_z(shift, y) for y in subs])
    neg = (Z < 0).mean(0)
    hit = (np.abs(Z) > k).mean(0)
    return sorted(np.where((neg >= neg_frac) & (hit >= hit_frac))[0].tolist())


def preprocess(shift: np.ndarray, darksubed: np.ndarray, int_time_ms: float,
               lam: float = ASLS_LAM, p: float = ASLS_P,
               cr_w: int = CR_W, cr_k: float = CR_K,
               fp=(FP_LO, FP_HI), norm: str = "max",
               cosmic_px: set | None = None,
               dead_px: list[int] | None = None) -> dict:
    """원시 한 획득 -> 분석용 스펙트럼. 반환 dict에 중간 산물 포함."""
    shift = np.asarray(shift, float)
    y0 = np.asarray(darksubed, float)
    sig = noise_sigma(shift, y0)

    if cosmic_px is None:
        y1, spikes = despike(y0, sig, cr_w, cr_k)
    else:
        y1, spikes = apply_mask(y0, cosmic_px), np.array(sorted(cosmic_px), int)
    if dead_px:
        y1 = apply_mask(y1, set(dead_px))

    m = (shift >= fp[0]) & (shift <= fp[1])
    x, y2 = shift[m], y1[m]

    bl = asls(y2, lam, p)
    y3 = y2 - bl

    cps = y3 / int_time_ms * 1000.0

    if norm == "snv":
        out = snv(y3)
    else:
        out = y3 / (y3.max() + 1e-12)

    return dict(shift=x, spec=out, corrected=y3, baseline=bl, raw_fp=y2,
                cps=cps, noise=sig, n_spikes=int(len(spikes)), spikes=spikes,
                snr=float(y3.max() / max(sig, 1e-9)))
