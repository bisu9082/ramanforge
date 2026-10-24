"""
ramanforge: export the computed library and its manifest.

Writes computed_library/<job>.csv -- one line list per library member, with the columns
the header names -- and computed_library/MANIFEST.csv, which maps every member to its
ORCA job, its role, and an isomeric SMILES and InChIKey derived from the Cartesian
geometry that was actually computed rather than from the 2D string it was built from.
That distinction matters: the stereochemistry of the computed structure is fixed by the
coordinates in orca/inputs/, and Section S9 shows that for A-232 this choice is worth
0.300 in the 600-850 cm-1 cosine.

Members whose entries were superseded during the work are exported as well and marked
as such in the manifest, so the file count exceeds the library size.
"""
import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v3_step3_headtohead as H
from src.spectra import parse_raman

OUT = os.path.join(ROOT, "computed_library")
SUPERSEDED = {"GF_cyclosarin": "superseded by GF_cyclosarin_glob (global minimum)",
              "EXP_A232": "superseded by EXP_A232_Eglob (E configuration)",
              "GB_sarin_fast": "duplicate re-run of GB_sarin"}

# The three entries above are not distractors: each is an earlier calculation of a measured
# agent, kept in the deposit so the superseded geometries can be inspected. The agent and
# the role are recorded explicitly, since the current job-to-agent map no longer names them.
RETIRED = {"GF_cyclosarin": ("GF", "measured agent (superseded entry)"),
           "EXP_A232": ("A232", "measured agent (superseded entry)"),
           "GB_sarin_fast": ("GB", "measured agent (duplicate run)")}


def xyz_from_out(path):
    """Element symbols and coordinates from the INPUT FILE echo of an ORCA output."""
    import re
    lines = open(path, errors="replace").read().splitlines()
    try:
        i = lines.index("                                       INPUT FILE")
    except ValueError:
        return [], np.zeros((0, 3))
    echo = []
    for ln in lines[i + 1:]:
        m = re.match(r"\|\s*\d+>\s?(.*)$", ln)
        if m:
            echo.append(m.group(1))
        elif echo:
            break
    return _xyz_from_lines(echo)


def _xyz_from_lines(lines):
    el, xyz = [], []
    started = False
    for ln in lines:
        t = ln.strip()
        if t.startswith("*") and "xyz" in t:
            started = True
            continue
        if started:
            if t.startswith("*"):
                break
            p = t.split()
            if len(p) == 4:
                el.append(p[0])
                xyz.append([float(v) for v in p[1:]])
    return el, np.asarray(xyz)


def smiles_from_geometry(el, xyz):
    """Isomeric SMILES and InChIKey perceived from the 3D structure."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdDetermineBonds
        RDLogger.DisableLog("rdApp.*")
        block = f"{len(el)}\n\n" + "\n".join(
            f"{e} {x:.8f} {y:.8f} {z:.8f}" for e, (x, y, z) in zip(el, xyz))
        m = Chem.MolFromXYZBlock(block)
        if m is None:
            return "", "", "xyz not parsed"
        rdDetermineBonds.DetermineBonds(m, charge=0)
        Chem.SanitizeMol(m)
        Chem.AssignStereochemistryFrom3D(m)
        key = Chem.MolToInchiKey(m)
        # Hydrogens are explicit after perception from coordinates; collapse them so the
        # string is readable. The stereo descriptors are unaffected.
        return (Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True), key, "")
    except Exception as e:                       # perception is not always possible
        return "", "", type(e).__name__


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "orca", "outputs"))
                  if f.endswith(".out"))
    inlib = set(H.load_library())
    job2agent = {j: a for a, j in H.JOB.items()}
    rows, nfail = [], 0
    for j in jobs:
        peaks = parse_raman(os.path.join(ROOT, "orca", "outputs", j + ".out"))
        dep = np.full(len(peaks), np.nan) if peaks.shape[1] < 3 else peaks[:, 2]
        with open(os.path.join(OUT, j + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["harmonic_wavenumber_cm-1", "raman_activity_A4_amu",
                        "depolarization_ratio"])
            for k in range(len(peaks)):
                w.writerow([f"{peaks[k,0]:.4f}", f"{peaks[k,1]:.6f}",
                            "" if np.isnan(dep[k]) else f"{dep[k]:.6f}"])
        smi = ik = note = ""
        el, xyz = xyz_from_out(os.path.join(ROOT, "orca", "outputs", j + ".out"))
        if len(el):
            smi, ik, note = smiles_from_geometry(el, xyz)
        if not smi:
            nfail += 1
        rows.append(dict(orca_job=j,
                         agent=job2agent.get(j, RETIRED.get(j, ("", ""))[0]),
                         role=("measured agent" if j in job2agent
                               else RETIRED[j][1] if j in RETIRED
                               else "analog distractor"),
                         in_library=("yes" if j in inlib else "no"),
                         superseded_note=SUPERSEDED.get(j, ""),
                         n_modes=len(peaks),
                         isomeric_smiles_from_geometry=smi,
                         inchikey=ik,
                         perception_note=note))
    with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_in = sum(r["in_library"] == "yes" for r in rows)
    print(f"exported {len(rows)} line lists, {n_in} of them in the library")
    print(f"structure perception failed for {nfail}")
    print("wrote computed_library/MANIFEST.csv")


if __name__ == "__main__":
    main()
