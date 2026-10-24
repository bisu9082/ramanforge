"""
ramanforge v3: preprocessing specification (explicit and reproducible).

Design principles
  P1  Measured data are never upsampled. The detector's native 512-pixel axis is the analysis grid.
      (all 55 files were verified to share the same axis: max dev = 0.0)
      The synthetic DFT spectra are interpolated down onto this axis.
  P2  Every step has exactly one parameter, and all of them are disclosed in a sensitivity table.
  P3  Augmented replicates are not used. Only acquisitions with integer CCD counts are used.

Steps
  S0  genuine acquisition selection : only files whose RawData are integers
  S1  dark current removal          : DarkSubed = RawData - Dark  (already in the file and verified)
  S2  cosmic ray removal (replicate consensus) : single-spectrum flags are not used. Only a
                                pixel that is not flagged in another acquisition of the same
                                agent is taken as a cosmic ray and interpolated.
                                Rationale: 81% of the pixels flagged on a single-spectrum
                                basis are reproduced in independent acquisitions, that is,
                                they are real signal and not cosmic rays.
                                HD and VX, which have only one acquisition each, cannot
                                receive this step (stated explicitly).
  S2b bad pixel masking             : pixels that read systematically low across all
                                acquisitions (detector defects) are detected automatically
                                and interpolated. Criterion: the deviation from the 5-point
                                median is negative in at least DEAD_NEG_FRAC of the
                                acquisitions and |z| > DEAD_K in at least DEAD_HIT_FRAC of them.
  S3  fingerprint region truncation : FP_LO ~ FP_HI cm-1  (the native grid is retained)
  S4  background removal            : AsLS (lam, p)
  S5  integration time normalization: counts/s = I / IntTime * 1000   (for reporting, cosine invariant)
  S6  intensity normalization       : max=1  (default) or SNV
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ---- specification (the values that go into Experimental verbatim) ------
FP_LO, FP_HI = 250.0, 1800.0
CR_W, CR_K = 2, 8.0          # cosmic ray: at most 2 pixels wide and 8 times the local noise
ASLS_LAM, ASLS_P, ASLS_IT = 1.0e4, 0.01, 30
NOISE_LO, NOISE_HI = 2200.0, 2560.0   # band-free high-frequency window, used to estimate the noise
DEAD_K, DEAD_NEG_FRAC, DEAD_HIT_FRAC = 6.0, 0.90, 0.50
LASER_NM, TEMP_K = 1064.0, 298.0


def is_genuine(raw: np.ndarray) -> bool:
    """Raw CCD counts are integers; non-integer means an augmented replicate."""
    return bool(np.abs(raw - np.round(raw)).max() < 1e-6)


def _mad(x):
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def noise_sigma(shift: np.ndarray, y: np.ndarray) -> float:
    """Residual standard deviation after linear background removal in the signal-free high-frequency window."""
    m = (shift >= NOISE_LO) & (shift <= NOISE_HI)
    x, v = shift[m], y[m]
    return float((v - np.polyval(np.polyfit(x, v, 1), x)).std())


def local_sigma(resid: np.ndarray, win: int = 51) -> np.ndarray:
    """Local MAD; absorbs the growth of shot noise with intensity."""
    n = len(resid); half = win // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = resid[lo:hi]
        out[i] = np.median(np.abs(seg - np.median(seg))) * 1.4826
    return out


def despike(y: np.ndarray, sigma0: float, width: int = CR_W, k: float = CR_K):
    """
    Remove positive and negative spikes of width <= width pixels.
    Criterion: the deviation from the 5-point median exceeds k times the local noise, with
          local noise = max(51-point moving MAD, high-frequency-window sigma0); in flat
          regions the MAD collapses to 0, which would flag large numbers of normal pixels,
          and this guards against that.
          The run of consecutive outlier pixels must be no longer than width.
    A real Raman band has a FWHM of about 3 times the pixel spacing, so it is separated by width.
    Returns: (corrected y, indices of the removed pixels)
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
    """Eilers & Boelens asymmetric least squares background."""
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
    """Replicate consensus: a pixel flagged in only one acquisition = cosmic ray candidate."""
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
    """Deviation from the 5-point median / local noise (floor = high-frequency window sigma)."""
    y = np.asarray(y, float); n = len(y)
    med = np.array([np.median(y[max(0, i - 2):i + 3]) for i in range(n)])
    resid = y - med
    s = np.maximum(local_sigma(resid), noise_sigma(shift, y))
    return resid / s


def detect_dead_pixels(shift: np.ndarray, subs: list[np.ndarray],
                       k: float = DEAD_K, neg_frac: float = DEAD_NEG_FRAC,
                       hit_frac: float = DEAD_HIT_FRAC) -> list[int]:
    """Detector bad pixels that read systematically low across all acquisitions."""
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
    """One raw acquisition -> analysis spectrum. The returned dict includes intermediate products."""
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
