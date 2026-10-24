"""
ramanforge Step 4-a
ORCA RAMAN SPECTRUM 파서 + full-spectrum(Lorentzian broadening) 표현.

ORCA 6.x freq 계산 출력의 'RAMAN SPECTRUM' 블록에서
(mode, freq[cm-1], activity[A^4/amu], depolarization) 를 파싱하고,
연속 스펙트럼 벡터 I(nu) 를 만든다.

full-spectrum 표현을 쓰는 이유:
  이 프로젝트 화학공간은 OP-F(사린류) 계열이라 config의
  진단밴드 타깃(C=N, P-O-C 등 Novichok용)이 존재하지 않는다.
  따라서 특정 밴드가 아니라 스펙트럼 전체를 ML 입력으로 쓴다.
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np

# ---- 스펙트럼 그리드 / broadening 기본값 ----
GRID_MIN = 200.0     # cm-1
GRID_MAX = 1800.0    # cm-1  (지문영역: 저주파 골격 200- + P-F/P-O-C/P=O + CH 변각 -1800)
GRID_STEP = 2.0      # cm-1  -> 801 point
# 주: C-H 신축역(2800-3200)은 분자 특이성 낮고 알킬 크기 대리변수라 제외.
#     포함 효과는 별도 ablation으로만 검증.
FWHM = 10.0          # cm-1  Lorentzian 반치폭 (실험 Raman 해상도 근사)

_RAMAN_HEADER = re.compile(r"RAMAN SPECTRUM")
# 예:  "   6:        25.49      0.227087      0.743578"
_ROW = re.compile(
    r"^\s*\d+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


def parse_raman(out_path: str | Path) -> np.ndarray:
    """
    ORCA .out에서 RAMAN SPECTRUM 블록을 파싱.
    반환: shape (M, 3) array = [freq, activity, depolarization]
    허수/음의 진동수(freq<=0) 행은 제거(비정상 구조 신호).
    """
    out_path = Path(out_path)
    text = out_path.read_text(errors="ignore").splitlines()

    # 마지막 RAMAN SPECTRUM 블록 사용 (재시작/다중출력 대비)
    starts = [i for i, ln in enumerate(text) if _RAMAN_HEADER.search(ln)]
    if not starts:
        raise ValueError(f"RAMAN SPECTRUM 블록 없음: {out_path.name}")
    i = starts[-1]

    rows = []
    for ln in text[i + 1:]:
        m = _ROW.match(ln)
        if m:
            freq, act, depol = map(float, m.groups())
            rows.append((freq, act, depol))
        elif rows:
            # 데이터가 시작된 뒤 매칭 실패 = 블록 끝
            break
    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        raise ValueError(f"RAMAN 행 파싱 실패: {out_path.name}")
    arr = arr[arr[:, 0] > 0.0]  # 허수 진동수 제거
    return arr


def _grid() -> np.ndarray:
    n = int(round((GRID_MAX - GRID_MIN) / GRID_STEP)) + 1
    return np.linspace(GRID_MIN, GRID_MAX, n)


def broaden(peaks: np.ndarray, fwhm: float = FWHM,
            normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    라인 스펙트럼(peaks[:,0]=freq, peaks[:,1]=activity) ->
    Lorentzian broadening으로 연속 스펙트럼.
    반환: (grid, intensity)  intensity는 max=1 정규화(normalize=True).
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
    """out 파일 -> 정규화된 intensity 벡터 (그리드 고정이므로 벡터만 반환)."""
    peaks = parse_raman(out_path)
    _, inten = broaden(peaks, fwhm=fwhm)
    return inten


def build_matrix(out_files: dict[str, str | Path],
                 fwhm: float = FWHM) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    여러 분자 -> 스펙트럼 행렬.
    반환: (names, grid, X)  X shape=(N, len(grid)).
    파싱 실패 분자는 건너뛰고 경고 출력.
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
    # 검증: 완료된 앵커/분자 out 하나 파싱
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
        print(f"  모드수(freq>0): {len(pk)}   그리드: {g[0]:.0f}-{g[-1]:.0f} cm-1 x{len(g)}")
        print(f"  최강 5개 밴드 (freq, activity):")
        for f, ac, _dp in top:
            print(f"    {f:8.1f} cm-1   act={ac:8.3f}")
