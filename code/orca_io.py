"""
orca_io.py — ORCA 인풋 생성 + 아웃풋(Raman) 파싱.

인풋: opt+freq+Raman (Kim 2026 프로토콜: B3LYP/6-311++G(d,p), scaling 미적용).
파싱: ORCA 'RAMAN SPECTRUM' 블록 → (frequency cm-1, activity) 리스트.

※ ORCA 버전별로 Raman 트리거/출력 포맷이 다를 수 있음. 6.x 기준으로 작성.
   Ku 환경에서 1개 샘플로 keyword/파싱을 먼저 검증할 것 (README 참고).
"""
from __future__ import annotations
import os, re


def write_input(name: str, xyz_block: str, cfg: dict, charge: int = 0, mult: int = 1) -> str:
    o = cfg["orca"]
    kw = [o["functional"], o["basis"]]
    if o.get("opt", True):
        kw.append("TightOpt")
    # Raman 활성도는 수치미분(NumFreq)으로 안정적으로 얻는다. analytic Freq는
    # 하이브리드(B3LYP) + 극성률 도함수 조합에서 ORCA가 "use LibXC" 에러를 냄.
    kw.append("NumFreq" if o.get("raman", True) else "Freq")
    if o.get("extra_keywords"):
        kw.append(o["extra_keywords"])
    header = f"! {' '.join(kw)}\n"
    blocks = f"%pal nprocs {o['nprocs']} end\n%maxcore {o['maxcore_mb']}\n"
    if o.get("raman", True):
        # Raman 활성도: 극성률 도함수 요청
        blocks += "%elprop Polar 1 end\n"
    geom = f"* xyz {charge} {mult}\n{xyz_block}\n*\n"
    return header + blocks + geom


def save_input(name: str, xyz_block: str, cfg: dict, out_dir: str,
               charge: int = 0, mult: int = 1) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.inp")
    with open(path, "w") as f:
        f.write(write_input(name, xyz_block, cfg, charge, mult))
    return path


# ORCA 'RAMAN SPECTRUM' 표: "  Mode   freq(cm**-1)   Activity   Depolarization"
_RAMAN_HDR = re.compile(r"RAMAN SPECTRUM", re.I)
_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def parse_raman(out_path: str) -> list[dict]:
    """ORCA .out → [{mode, freq, activity, depol}] (freq>0만)."""
    with open(out_path, "r", errors="ignore") as f:
        lines = f.readlines()
    peaks, in_block = [], False
    for ln in lines:
        if _RAMAN_HDR.search(ln):
            in_block, peaks = True, []      # 마지막 블록 채택
            continue
        if in_block:
            m = _ROW.match(ln)
            if m:
                mode, freq, act, dep = int(m[1]), float(m[2]), float(m[3]), float(m[4])
                if freq > 0:
                    peaks.append({"mode": mode, "freq": freq, "activity": act, "depol": dep})
            elif peaks and ln.strip() == "":
                # 블록 종료 후 빈 줄 → 계속 스캔 (다음 블록 있으면 갱신)
                in_block = False
    return peaks


def parse_converged(out_path: str) -> bool:
    """정상 종료 + 허수진동 없음 확인."""
    try:
        txt = open(out_path, "r", errors="ignore").read()
    except FileNotFoundError:
        return False
    if "ORCA TERMINATED NORMALLY" not in txt:
        return False
    # 허수(음수) 진동수 존재 시 실패로 간주
    imags = re.findall(r"\*\*\*imaginary mode\*\*\*", txt, re.I)
    return len(imags) == 0


if __name__ == "__main__":
    import sys
    print(parse_raman(sys.argv[1])[:10])
