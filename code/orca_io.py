"""
orca_io.py: ORCA input generation plus output (Raman) parsing.

Input: opt+freq+Raman (Kim 2026 protocol: B3LYP/6-311++G(d,p), no scaling applied).
Parsing: ORCA 'RAMAN SPECTRUM' block → list of (frequency cm-1, activity).

Note: the Raman trigger and the output format may differ between ORCA versions; written for 6.x.
   Verify the keyword and the parsing on a single sample in Ku's environment first (see README).
"""
from __future__ import annotations
import os, re


def write_input(name: str, xyz_block: str, cfg: dict, charge: int = 0, mult: int = 1) -> str:
    o = cfg["orca"]
    kw = [o["functional"], o["basis"]]
    if o.get("opt", True):
        kw.append("TightOpt")
    # Raman activities come out reliably from numerical differentiation (NumFreq); analytic Freq
    # raises a "use LibXC" error in ORCA for a hybrid (B3LYP) combined with polarisability derivatives.
    kw.append("NumFreq" if o.get("raman", True) else "Freq")
    if o.get("extra_keywords"):
        kw.append(o["extra_keywords"])
    header = f"! {' '.join(kw)}\n"
    blocks = f"%pal nprocs {o['nprocs']} end\n%maxcore {o['maxcore_mb']}\n"
    if o.get("raman", True):
        # Raman activities: request polarisability derivatives
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


# ORCA 'RAMAN SPECTRUM' table: "  Mode   freq(cm**-1)   Activity   Depolarization"
_RAMAN_HDR = re.compile(r"RAMAN SPECTRUM", re.I)
_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def parse_raman(out_path: str) -> list[dict]:
    """ORCA .out → [{mode, freq, activity, depol}] (freq>0 only)."""
    with open(out_path, "r", errors="ignore") as f:
        lines = f.readlines()
    peaks, in_block = [], False
    for ln in lines:
        if _RAMAN_HDR.search(ln):
            in_block, peaks = True, []      # take the last block
            continue
        if in_block:
            m = _ROW.match(ln)
            if m:
                mode, freq, act, dep = int(m[1]), float(m[2]), float(m[3]), float(m[4])
                if freq > 0:
                    peaks.append({"mode": mode, "freq": freq, "activity": act, "depol": dep})
            elif peaks and ln.strip() == "":
                # a blank line after the block ends -> keep scanning (a later block supersedes this one)
                in_block = False
    return peaks


def parse_converged(out_path: str) -> bool:
    """Check normal termination + no imaginary frequencies."""
    try:
        txt = open(out_path, "r", errors="ignore").read()
    except FileNotFoundError:
        return False
    if "ORCA TERMINATED NORMALLY" not in txt:
        return False
    # treat the presence of an imaginary (negative) frequency as a failure
    imags = re.findall(r"\*\*\*imaginary mode\*\*\*", txt, re.I)
    return len(imags) == 0


if __name__ == "__main__":
    import sys
    print(parse_raman(sys.argv[1])[:10])
