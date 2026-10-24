"""
ramanforge: DFT→measured spectrum calibration.

The domain gap between harmonic, gas phase, single conformer DFT calculations and
room temperature condensed phase measurements is calibrated on physical grounds. The
contributions verified against the measurement at the anchor (GF):
cosine 0.587 → 0.707 (FWHM) → 0.723 (scattering factor).

Calibration terms
  1) frequency scaling s : corrects the systematic overestimation of the B3LYP harmonic wavenumbers
  2) Lorentzian FWHM     : condensed phase linewidth (absent from a gas phase calculation)
  3) Raman scattering factor : converts the ORCA 'activity' (∂α/∂Q)² into measured intensity
        I ∝ (ν0-ν)^4 / ν · 1/(1-exp(-hcν/kT))   (Bose-Einstein)
  4) (optional) linear baseline : approximates the measured fluorescence background

The parameters are fitted by maximizing the cosine against the measured anchor
(fit_calibration) and those values are applied to the whole library (apply); that is, they
are not fitted molecule by molecule, so there is no risk of overfitting or circularity.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

HCK = 1.4388      # hc/k  [cm·K]
# instrument: Metrohm TacticID-1064 ST (1064 nm +- 0.2 nm, Class IIIB, max 495 mW)
DEFAULT_LASER = 1064.0  # nm


@dataclass
class CalibParams:
    scale: float = 0.992      # frequency scale factor
    fwhm: float = 30.0        # cm-1
    laser_nm: float = DEFAULT_LASER
    temp_K: float = 298.0
    scattering: bool = True   # whether the scattering factor is applied

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path):
        return CalibParams(**json.loads(Path(path).read_text()))


def scattering_factor(freqs: np.ndarray, laser_nm: float, T: float) -> np.ndarray:
    """ORCA activity → measured Raman intensity conversion factor."""
    nu0 = 1e7 / laser_nm                       # excitation [cm-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        bose = 1.0 / (1.0 - np.exp(-HCK * freqs / T))
        fac = ((nu0 - freqs) ** 4) / freqs * bose
    return np.nan_to_num(fac, nan=0.0, posinf=0.0)


def synth_spectrum(peaks: np.ndarray, grid: np.ndarray,
                   p: CalibParams) -> np.ndarray:
    """
    DFT line spectrum (peaks[:,0]=freq, [:,1]=activity) → calibrated continuous spectrum.
    Returns: intensity normalized (max=1) on the grid.
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
    [Caution] Fit based on maximizing the similarity (cosine).
    The broader the FWHM, the more the spectrum is smeared, so the cosine rises for
    'any pair'; choosing the FWHM by this criterion therefore degrades identification
    discrimination (confirmed on measured data: at fwhm40, sulfur mustard ranks a nerve
    agent DFT spectrum first, an incorrect call). Use it for the frequency scale fit only;
    for the FWHM and for identification calibration use fit_calibration_discriminative().
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
    Calibration fit based on **identification performance** (recommended).

    lib_peaks : {library name: DFT peaks array}
    exp_specs : {measured name: spectrum on the grid}
    truth     : {measured name: correct library name}
    Objective : the fraction for which the correct entry ranks first (top-1) takes
                precedence; on a tie, the larger mean cosine margin between the
                correct entry and the best incorrect one.

    Returns: (best CalibParams, top1, mean margin)
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
    """Standard Normal Variate; normalizes measured intensity scale/offset."""
    return (x - x.mean()) / (x.std() + 1e-12)


def baseline_correct(y: np.ndarray, order: int = 3, n_iter: int = 30) -> np.ndarray:
    """
    Fluorescence background removal (iterative polynomial fitting). For preprocessing of measured spectra.
    """
    x = np.linspace(-1, 1, len(y))
    z = y.copy()
    for _ in range(n_iter):
        c = np.polyfit(x, z, order)
        b = np.polyval(c, x)
        z = np.minimum(z, b)
    return y - np.polyval(np.polyfit(x, z, order), x)
