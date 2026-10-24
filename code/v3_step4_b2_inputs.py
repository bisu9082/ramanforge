"""
v3 B2: generation of the ORCA inputs for testing an improved P–F/P–N description.

A1 diagnosis: the measured↔computed gap is concentrated in 600–850 cm-1 (the P–F and P–N
stretches) and appears only for the phosphorus-containing agents (0.569 vs 0.913). The
conformer hypothesis is rejected (GD, the most flexible agent, is the best in this region).
The 'phosphorus description' of the forward model is therefore what is fixed.

Ladder (in order of expected effect; checked on the anchor first, then on all agents)
  V0  baseline            B3LYP/6-311++G(d,p)                      = current (no recomputation needed)
  V1  condensed phase     V0 + CPCM (a self-consistent permittivity instead of that of water is excessive → plain CPCM)
  V2  phosphorus tight-d  V0 + def2-TZVPP on P, F, S and Cl        (the classical bottleneck for phosphorus wavenumbers)
  V3  dispersion and range correction  wB97X-D3/6-311++G(d,p)      = Kim 2026 future work
  V4  V1+V2 combined      the best combination (only if V1 and V2 each have an effect)

Anchor: GF (cyclosarin, 25 atoms, 600–850 cos 0.562) → A-232 (29 atoms, 0.205)
The geometry reuses the converged structure of the existing baseline .out (reoptimization is performed again for each variant).
"""
import os, re, sys, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHORS = {
    "GF": "GF_cyclosarin",
    "A232": "EXP_A232",
}

VARIANTS = {
    "V1_cpcm": dict(
        keywords="! B3LYP 6-311++G(d,p) TightOpt NumFreq CPCM(Acetone)",
        note="Condensed-phase dielectric effect; approximated with acetone, whose permittivity (~20) is close to that of the neat liquid agents.",
        blocks=""),
    "V2_tightP": dict(
        keywords="! B3LYP 6-311++G(d,p) TightOpt NumFreq",
        note="Promote only P, F, S, Cl to def2-TZVPP. Resolves the classic bottleneck of phosphorus frequencies (lack of tight-d).",
        blocks='%basis\n  NewGTO P  "def2-TZVPP" end\n  NewGTO F  "def2-TZVPP" end\n'
               '  NewGTO S  "def2-TZVPP" end\n  NewGTO Cl "def2-TZVPP" end\nend\n'),
    "V3_wb97xd3": dict(
        keywords="! wB97X-D3 6-311++G(d,p) TightOpt NumFreq",
        note="Dispersion and long-range corrected functional announced as future work by Kim 2026.",
        blocks=""),
    "V4_cpcm_tightP": dict(
        keywords="! B3LYP 6-311++G(d,p) TightOpt NumFreq CPCM(Acetone)",
        note="V1+V2 combined. Run only when V1 and V2 each show an effect.",
        blocks='%basis\n  NewGTO P  "def2-TZVPP" end\n  NewGTO F  "def2-TZVPP" end\n'
               '  NewGTO S  "def2-TZVPP" end\n  NewGTO Cl "def2-TZVPP" end\nend\n'),
}

NPROCS, MAXCORE = 8, 3000


def final_geometry(out_path: str) -> str:
    """Extract the final (converged) Cartesian coordinate block from an ORCA .out."""
    with open(out_path, "r", errors="ignore") as f:
        lines = f.readlines()
    idx = [i for i, ln in enumerate(lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in ln]
    if not idx:
        raise ValueError(f"no coordinate block: {out_path}")
    i = idx[-1] + 2                      # header and rule
    rows = []
    pat = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
    for ln in lines[i:]:
        m = pat.match(ln)
        if m:
            rows.append(f"  {m[1]:<2} {float(m[2]):14.8f} {float(m[3]):14.8f} {float(m[4]):14.8f}")
        elif rows:
            break
    if not rows:
        raise ValueError(f"coordinate parsing failed: {out_path}")
    return "\n".join(rows)


def build(tag: str, job: str, variant: str, spec: dict, xyz: str) -> str:
    name = f"B2_{tag}_{variant}"
    head = spec["keywords"] + "\n"
    blk = f"%pal nprocs {NPROCS} end\n%maxcore {MAXCORE}\n%elprop Polar 1 end\n"
    blk += spec["blocks"]
    body = f"* xyz 0 1\n{xyz}\n*\n"
    return name, f"# {spec['note']}\n# starting geometry: {job}.out converged structure\n" + head + blk + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "orca", "inputs_b2"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    made = []
    for tag, job in ANCHORS.items():
        out = os.path.join(ROOT, "orca", "outputs", f"{job}.out")
        xyz = final_geometry(out)
        natoms = len(xyz.strip().splitlines())
        print(f"{tag:<6} ({job}) {natoms} atoms — geometry extraction done")
        for variant, spec in VARIANTS.items():
            name, text = build(tag, job, variant, spec, xyz)
            p = os.path.join(a.outdir, f"{name}.inp")
            with open(p, "w") as f:
                f.write(text)
            made.append((name, natoms, spec["note"]))

    # driver script, in priority order and idempotent
    order = ["GF_V1_cpcm", "GF_V2_tightP", "GF_V3_wb97xd3",
             "A232_V1_cpcm", "A232_V2_tightP", "A232_V3_wb97xd3",
             "GF_V4_cpcm_tightP", "A232_V4_cpcm_tightP"]
    sh = ["#!/usr/bin/env bash",
          "# B2: tests for improved description of P. In priority order; jobs already finished are skipped.",
          "# Run V4 only after V1 and V2 each show an effect (placed last).",
          "set -u",
          'ORCA=/home/k9/orca_6_1_1/orca',
          'cd "$(dirname "$0")"',
          'mkdir -p ../outputs_b2',
          ""]
    for nm in order:
        f = f"B2_{nm}"
        sh += [f'if [ ! -s ../outputs_b2/{f}.out ]; then',
               f'  echo "[$(date +%F\\ %T)] {f} start"',
               f'  "$ORCA" {f}.inp > ../outputs_b2/{f}.out 2>&1',
               f'  echo "[$(date +%F\\ %T)] {f} done"',
               'else',
               f'  echo "skip {f} (already present)"',
               'fi', ""]
    sp = os.path.join(a.outdir, "run_b2.sh")
    with open(sp, "w") as f:
        f.write("\n".join(sh))
    os.chmod(sp, 0o755)

    print(f"\ngenerated inputs: {len(made)} → {a.outdir}")
    for nm, na, note in made:
        print(f"  {nm:<26} ({na} atoms)  {note.split('.')[0]}")
    print(f"\nrun: bash {os.path.relpath(sp, ROOT)}")


if __name__ == "__main__":
    main()
