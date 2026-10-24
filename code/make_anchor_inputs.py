"""
make_anchor_inputs.py: generation of the ORCA inputs for the Step 1-A anchor validation.

3D geometries (RDKit) of GB (sarin) and VX → ORCA opt+freq+Raman .inp files.
Protocol: the functional/basis from config.yaml (default B3LYP/6-311++G(d,p), the same as Kim 2026).
Output location: config.paths.orca_inputs (default ../orca/inputs).
"""
import os, yaml
import structures as ST
import orca_io as IO

ANCHORS = ["GB_sarin", "VX"]   # keys of structures.VALIDATION; extend when reproducing all eleven agents


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    out_dir = cfg["paths"]["orca_inputs"]
    os.makedirs(out_dir, exist_ok=True)
    for name in ANCHORS:
        smi = ST.VALIDATION[name]
        xyz = ST.gen_xyz(smi, seed=cfg["active_learning"]["seed"])
        if xyz is None:
            print(f"[fail] geometry generation failed: {name}")
            continue
        path = IO.save_input(name, xyz, cfg, out_dir)
        n_atoms = len(xyz.strip().splitlines())
        print(f"[ok] {name} ({n_atoms} atoms) -> {path}")


if __name__ == "__main__":
    main()
