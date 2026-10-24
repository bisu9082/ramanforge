"""
spectrum.py — peak 리스트(freq, activity) ↔ 연속 스펙트럼 벡터 변환 + 진단 밴드 추출.

- to_vector: Lorentzian broadening으로 (x_min..x_max) 그리드 벡터 생성 → ML 입력용.
- diagnostic_bands: 지정 밴드 창에서 최대활성 peak 위치 추출 → AL surrogate 타깃.
"""
from __future__ import annotations
import numpy as np

# 진단 밴드 창 (cm-1) — Kim 2026 Table 5 근거 (Novichok 판별 밴드)
BAND_WINDOWS = {
    "P=O":       (1220, 1265),
    "C=N":       (1595, 1690),
    "P-F":       (700, 755),
    "P-O-C_asym":(1050, 1075),
    "P-O-C_sym": (790, 810),
    "N=C-N":     (1355, 1380),
}


def grid(cfg: dict) -> np.ndarray:
    s = cfg["spectrum"]
    return np.arange(s["x_min"], s["x_max"] + s["x_step"], s["x_step"], dtype=float)


def to_vector(peaks: list[dict], cfg: dict) -> np.ndarray:
    """peaks → Lorentzian 합성 스펙트럼 (정규화 max=1)."""
    s = cfg["spectrum"]
    x = grid(cfg)
    y = np.zeros_like(x)
    gamma = s["broadening_fwhm"] / 2.0
    scale = s.get("freq_scale", 1.0)
    for p in peaks:
        f = p["freq"] * scale
        a = max(p["activity"], 0.0)
        y += a * (gamma**2) / ((x - f) ** 2 + gamma**2)
    if y.max() > 0:
        y = y / y.max()
    return y


def diagnostic_bands(peaks: list[dict], targets: list[str], cfg: dict) -> dict:
    """각 타깃 밴드 창에서 최대활성 peak의 freq 반환 (없으면 nan)."""
    scale = cfg["spectrum"].get("freq_scale", 1.0)
    res = {}
    for t in targets:
        lo, hi = BAND_WINDOWS[t]
        cand = [(p["activity"], p["freq"] * scale) for p in peaks if lo <= p["freq"] * scale <= hi]
        res[t] = max(cand)[1] if cand else float("nan")
    return res
