"""
ramanforge v3: raw spectrum input and output.

Structure of the TacticID-1064 ST raw CSV
  line1 : SN,<serial>,PixelNum,512,IntTime,<ms>,CCDTemperature,<C>,
          Gain,<g>,Offset,<o>,LEW,<laser excitation wavelength nm>
  line2-5 : Coef_0..Coef_3  (pixel -> wavelength polynomial)
  line6 : header  PixelNum,WaveLength,RamanShift,RawData,Dark,DarkSubed,Ratio3
  line7+: 512 pixel rows
"""
from __future__ import annotations
import os, re, glob
import numpy as np
import pandas as pd


def parse_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        l1 = fh.readline().strip()
        coefs = {}
        pos = fh.tell()
        while True:
            pos = fh.tell()
            ln = fh.readline()
            if not ln:
                break
            m = re.match(r"^Coef_(\d+),(-?[\d.eE+-]+)", ln.strip())
            if not m:
                fh.seek(pos)
                break
            coefs[int(m.group(1))] = float(m.group(2))
        df = pd.read_csv(fh)

    toks = l1.split(",")
    meta = {}
    for i in range(0, len(toks) - 1, 2):
        k, v = toks[i], toks[i + 1]
        try:
            meta[k] = float(v)
        except ValueError:
            meta[k] = v
    meta["coefs"] = [coefs[i] for i in sorted(coefs)]

    base = os.path.basename(path)
    meta["file"] = base
    meta["agent"] = base.split("_")[0]
    m = re.search(r"_(\d{2})_raw\.csv$", base)
    meta["rep"] = int(m.group(1)) if m else -1
    df.columns = [c.strip() for c in df.columns]
    return {"meta": meta, "df": df}


def load_all(raw_dir: str) -> list[dict]:
    fs = sorted(f for f in glob.glob(os.path.join(raw_dir, "*_raw.csv"))
                if "MACOSX" not in f)
    return [parse_raw(f) for f in fs]
