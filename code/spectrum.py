"""
spectrum.py: conversion between a peak list (freq, activity) ↔ a continuous spectrum vector, plus diagnostic band extraction.

- to_vector: builds an (x_min..x_max) grid vector by Lorentzian broadening → for the ML input.
- diagnostic_bands: extracts the position of the most active peak in a given band window → the AL surrogate target.
"""
from __future__ import annotations
import numpy as np

# diagnostic band windows (cm-1), after Kim 2026 Table 5 (Novichok discriminating bands)
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
    """peaks → synthesized Lorentzian spectrum (normalized max=1)."""
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
    """Return the freq of the maximum-activity peak in each target band window (nan if none)."""
    scale = cfg["spectrum"].get("freq_scale", 1.0)
    res = {}
    for t in targets:
        lo, hi = BAND_WINDOWS[t]
        cand = [(p["activity"], p["freq"] * scale) for p in peaks if lo <= p["freq"] * scale <= hi]
        res[t] = max(cand)[1] if cand else float("nan")
    return res
