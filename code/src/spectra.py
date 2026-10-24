"""
ramanforge Step 4-a
ORCA RAMAN SPECTRUM parser + full-spectrum (Lorentzian broadening) representation.

Parses (mode, freq[cm-1], activity[A^4/amu], depolarization) from the 'RAMAN SPECTRUM'
block of an ORCA 6.x freq calculation output and builds the continuous spectrum
vector I(nu).

Why the full-spectrum representation is used:
  the chemical space of this project is the OP-F (sarin type) family, so the
  diagnostic band targets in config (C=N, P-O-C and the like, intended for Novichok
  agents) do not exist. The whole spectrum, rather than a specific band, is therefore
  used as the ML input.
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np

# ---- spectral grid and broadening defaults ----
GRID_MIN = 200.0     # cm-1
GRID_MAX = 1800.0    # cm-1  (fingerprint region: low-frequency skeleton from 200 + P-F/P-O-C/P=O + CH bending to 1800)
GRID_STEP = 2.0      # cm-1  -> 801 point
# note: the C-H stretching region (2800-3200) is excluded; it carries little molecular specificity and acts as a proxy for alkyl size.
#       its effect is checked only in a separate ablation.
FWHM = 10.0          # cm-1  Lorentzian half-width, approximating the experimental Raman resolution

_RAMAN_HEADER = re.compile(r"RAMAN SPECTRUM")
# example:  "   6:        25.49      0.227087      0.743578"
_ROW = re.compile(
    r"^\s*\d+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


def parse_raman(out_path: str | Path) -> np.ndarray:
    """
    Parse the RAMAN SPECTRUM block from an ORCA .out file.
    Returns: shape (M, 3) array = [freq, activity, depolarization]
    Rows with imaginary or negative frequencies (freq<=0) are removed (a sign of an unstable structure).
    """
    out_path = Path(out_path)
    text = out_path.read_text(errors="ignore").splitlines()

    # use the last RAMAN SPECTRUM block, in case of a restart or multiple outputs
    starts = [i for i, ln in enumerate(text) if _RAMAN_HEADER.search(ln)]
    if not starts:
        raise ValueError(f"RAMAN SPECTRUM block not found: {out_path.name}")
    i = starts[-1]

    rows = []
    for ln in text[i + 1:]:
        m = _ROW.match(ln)
        if m:
            freq, act, depol = map(float, m.groups())
            rows.append((freq, act, depol))
        elif rows:
            # once data has started, a failed match marks the end of the block
            break
    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        raise ValueError(f"RAMAN row parsing failed: {out_path.name}")
    arr = arr[arr[:, 0] > 0.0]  # drop imaginary frequencies
    return arr


def _grid() -> np.ndarray:
    n = int(round((GRID_MAX - GRID_MIN) / GRID_STEP)) + 1
    return np.linspace(GRID_MIN, GRID_MAX, n)


def broaden(peaks: np.ndarray, fwhm: float = FWHM,
            normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Line spectrum (peaks[:,0]=freq, peaks[:,1]=activity) ->
    continuous spectrum by Lorentzian broadening.
    Returns: (grid, intensity)  intensity is normalized to max=1 (normalize=True).
    """
    grid = _grid()
    gamma = fwhm / 2.0
    inten = np.zeros_like(grid)
    freqs = peaks[:, 0]
    acts = peaks[:, 1]
    for f, a in zip(freqs, acts):
        inten += a * (gamma ** 2) / ((grid - f) ** 2 + gamma ** 2)
    if normalize and inten.max() > 0:
        inten = inten / inten.max()
    return grid, inten


def spectrum_vector(out_path: str | Path, fwhm: float = FWHM) -> np.ndarray:
    """out file -> normalized intensity vector (grid is fixed, so only the vector is returned)."""
    peaks = parse_raman(out_path)
    _, inten = broaden(peaks, fwhm=fwhm)
    return inten


def build_matrix(out_files: dict[str, str | Path],
                 fwhm: float = FWHM) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Several molecules -> spectrum matrix.
    Returns: (names, grid, X)  X shape=(N, len(grid)).
    Molecules that fail to parse are skipped and a warning is printed.
    """
    grid = _grid()
    names, rows = [], []
    for name, p in out_files.items():
        try:
            v = spectrum_vector(p, fwhm=fwhm)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue
        names.append(name)
        rows.append(v)
    X = np.vstack(rows) if rows else np.empty((0, len(grid)))
    return names, grid, X


if __name__ == "__main__":
    import sys
    # check: parse one completed anchor or molecule output
    args = sys.argv[1:]
    if not args:
        base = Path(__file__).resolve().parents[1] / "orca" / "outputs"
        cands = sorted(base.glob("*.out"))
        args = [str(c) for c in cands[:1]]
    for a in args:
        pk = parse_raman(a)
        g, y = broaden(pk)
        top = pk[np.argsort(pk[:, 1])[::-1][:5]]
        print(f"\n=== {Path(a).name} ===")
        print(f"  number of modes (freq>0): {len(pk)}   grid: {g[0]:.0f}-{g[-1]:.0f} cm-1 x{len(g)}")
        print(f"  5 strongest bands (freq, activity):")
        for f, ac, _dp in top:
            print(f"    {f:8.1f} cm-1   act={ac:8.3f}")
