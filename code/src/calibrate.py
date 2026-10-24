"""
ramanforge — DFT→실측 스펙트럼 보정(calibration).

DFT 조화·기체상·단일 conformer 계산과 상온 응축상 실측 사이의
도메인 갭을 물리 기반으로 보정한다. 앵커(GF)에서 실측으로 검증된
기여도: 코사인 0.587 → 0.707(FWHM) → 0.723(산란인자).

보정 요소
  1) 주파수 스케일링 s : B3LYP 조화 진동수의 계통 과대추정 보정
  2) Lorentzian FWHM   : 응축상 선폭(기체상 계산엔 없음)
  3) Raman 산란인자    : ORCA 'activity'(∂α/∂Q)² → 실측 강도로 변환
        I ∝ (ν0-ν)^4 / ν · 1/(1-exp(-hcν/kT))   (Bose-Einstein)
  4) (옵션) 선형 baseline : 실측 형광 배경 근사

파라미터는 실측 앵커에 대해 코사인 최대화로 적합(fit_calibration)하고,
그 값을 라이브러리 전체에 적용(apply)한다. — 즉 분자마다 따로 맞추지
않으므로 과적합/순환 위험이 없다.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

HCK = 1.4388      # hc/k  [cm·K]
# 실측 장비: Metrohm TacticID-1064 ST (1064 nm ± 0.2 nm, Class IIIB, max 495 mW)
DEFAULT_LASER = 1064.0  # nm


@dataclass
class CalibParams:
    scale: float = 0.992      # 주파수 스케일
    fwhm: float = 30.0        # cm-1
    laser_nm: float = DEFAULT_LASER
    temp_K: float = 298.0
    scattering: bool = True   # 산란인자 적용 여부

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path):
        return CalibParams(**json.loads(Path(path).read_text()))


def scattering_factor(freqs: np.ndarray, laser_nm: float, T: float) -> np.ndarray:
    """ORCA activity → 실측 Raman 강도 변환 인자."""
    nu0 = 1e7 / laser_nm                       # 여기광 [cm-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        bose = 1.0 / (1.0 - np.exp(-HCK * freqs / T))
        fac = ((nu0 - freqs) ** 4) / freqs * bose
    return np.nan_to_num(fac, nan=0.0, posinf=0.0)


def synth_spectrum(peaks: np.ndarray, grid: np.ndarray,
                   p: CalibParams) -> np.ndarray:
    """
    DFT 라인 스펙트럼(peaks[:,0]=freq, [:,1]=activity) → 보정된 연속 스펙트럼.
    반환: grid 위 정규화(max=1) 강도.
    """
    f = peaks[:, 0] * p.scale
    a = peaks[:, 1].astype(float).copy()
    if p.scattering:
        a = a * scattering_factor(f, p.laser_nm, p.temp_K)
        if a.max() > 0:
            a = a / a.max()
    g = p.fwhm / 2.0
    I = np.zeros_like(grid, dtype=float)
    for ff, aa in zip(f, a):
        I += aa * g ** 2 / ((grid - ff) ** 2 + g ** 2)
    return I / (I.max() + 1e-12)


def _cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def fit_calibration(peaks: np.ndarray, exp_spec: np.ndarray, grid: np.ndarray,
                    laser_nm: float = DEFAULT_LASER,
                    scales=np.arange(0.94, 1.011, 0.002),
                    fwhms=(8, 12, 16, 20, 25, 30, 35, 40),
                    scattering: bool = True) -> tuple[CalibParams, float]:
    """
    [주의] 유사도(코사인) 최대화 기준 적합.
    FWHM은 넓힐수록 스펙트럼이 뭉개져 '아무 쌍이나' 코사인이 올라가므로,
    이 기준으로 FWHM을 고르면 식별 변별력이 오히려 나빠진다(실측 확인:
    fwhm40에서 황겨자가 신경작용제 DFT를 1위로 오답). 주파수 스케일 적합에만
    쓰고, FWHM/식별용 보정은 fit_calibration_discriminative()를 사용할 것.
    """
    best, best_cs = None, -1.0
    for s in scales:
        for w in fwhms:
            p = CalibParams(scale=float(s), fwhm=float(w),
                            laser_nm=laser_nm, scattering=scattering)
            cs = _cos(synth_spectrum(peaks, grid, p), exp_spec)
            if cs > best_cs:
                best, best_cs = p, cs
    return best, best_cs


def fit_calibration_discriminative(
        lib_peaks: dict, exp_specs: dict, truth: dict, grid: np.ndarray,
        laser_nm: float = DEFAULT_LASER,
        scales=np.arange(0.94, 1.011, 0.004),
        fwhms=(8, 10, 12, 15, 20, 25, 30),
        scattering: bool = True):
    """
    **식별 성능** 기준 보정 적합(권장).

    lib_peaks : {라이브러리명: DFT peaks 배열}
    exp_specs : {실측명: grid 위 스펙트럼}
    truth     : {실측명: 정답 라이브러리명}
    목적함수  : 정답이 1위인 비율(top-1) 우선, 동률이면 정답과 최고오답의
                코사인 마진 평균이 큰 쪽.

    반환: (최적 CalibParams, top1, 평균마진)
    """
    best = None
    for s in scales:
        for w in fwhms:
            p = CalibParams(scale=float(s), fwhm=float(w),
                            laser_nm=laser_nm, scattering=scattering)
            lib = {k: synth_spectrum(v, grid, p) for k, v in lib_peaks.items()}
            hits, margins = 0, []
            for name, e in exp_specs.items():
                if name not in truth:
                    continue
                sims = {k: _cos(e, v) for k, v in lib.items()}
                gold = truth[name]
                if gold not in sims:
                    continue
                top = max(sims, key=sims.get)
                others = [v for k, v in sims.items() if k != gold]
                margins.append(sims[gold] - (max(others) if others else 0.0))
                hits += int(top == gold)
            n = max(len(margins), 1)
            score = (hits / n, float(np.mean(margins)) if margins else -1.0)
            if best is None or score > best[0]:
                best = (score, p)
    (top1, margin), p = best
    return p, top1, margin


def snv(x: np.ndarray) -> np.ndarray:
    """Standard Normal Variate — 실측 강도 스케일/오프셋 정규화."""
    return (x - x.mean()) / (x.std() + 1e-12)


def baseline_correct(y: np.ndarray, order: int = 3, n_iter: int = 30) -> np.ndarray:
    """
    형광 배경 제거(반복 다항 피팅). 실측 스펙트럼 전처리용.
    """
    x = np.linspace(-1, 1, len(y))
    z = y.copy()
    for _ in range(n_iter):
        c = np.polyfit(x, z, order)
        b = np.polyval(c, x)
        z = np.minimum(z, b)
    return y - np.polyval(np.polyfit(x, z, order), x)
