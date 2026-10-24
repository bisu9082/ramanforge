"""
run_orca.py — ORCA 실행 래퍼 (subprocess 스트리밍 + 주기적 상태 출력).

로컬 ORCA를 호출해 .inp → .out 실행. 계산 모니터링 원칙에 맞춰
30초마다 진행 상황(경과시간, out 파일 크기, 마지막 SCF 라인)을 출력한다.
샌드박스처럼 ORCA가 없으면 명확히 에러 반환.
"""
from __future__ import annotations
import os, subprocess, time, shutil


def run_one(inp_path: str, out_path: str, orca_exe: str, poll: int = 30) -> dict:
    exe = shutil.which(orca_exe) or (orca_exe if os.path.isfile(orca_exe) else None)
    if exe is None:
        return {"ok": False, "reason": f"ORCA 실행파일 없음: {orca_exe}", "out": out_path}

    t0 = time.time()
    with open(out_path, "w") as fout:
        # ORCA는 병렬 실행 시 절대경로 인풋 권장
        proc = subprocess.Popen([exe, os.path.abspath(inp_path)],
                                 stdout=fout, stderr=subprocess.STDOUT)
        last = 0.0
        while proc.poll() is None:
            time.sleep(1)
            if time.time() - last >= poll:
                last = time.time()
                sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                tail = _tail_scf(out_path)
                print(f"  [orca] {os.path.basename(inp_path)} | "
                      f"{int(time.time()-t0)}s | {sz//1024}KB | {tail}", flush=True)
    rc = proc.returncode
    ok = (rc == 0)
    return {"ok": ok, "reason": "" if ok else f"returncode={rc}",
            "out": out_path, "seconds": round(time.time() - t0, 1)}


def _tail_scf(path: str) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()[-40:]
        for ln in reversed(lines):
            if "SCF ITERATIONS" in ln or "Total Energy" in ln or "OPTIMIZATION" in ln:
                return ln.strip()[:70]
    except Exception:
        pass
    return "..."
