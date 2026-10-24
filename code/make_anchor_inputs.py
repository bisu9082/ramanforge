"""
make_anchor_inputs.py — Step 1-A 앵커 검증용 ORCA 인풋 생성.

GB(사린)·VX의 3D 지오메트리(RDKit) → ORCA opt+freq+Raman .inp 파일.
프로토콜: config.yaml의 functional/basis (기본 B3LYP/6-311++G(d,p), Kim 2026 동일).
생성 위치: config.paths.orca_inputs (기본 ../orca/inputs).
"""
import os, yaml
import structures as ST
import orca_io as IO

ANCHORS = ["GB_sarin", "VX"]   # structures.VALIDATION 키. 11종 전체 재현 시 확장.


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    out_dir = cfg["paths"]["orca_inputs"]
    os.makedirs(out_dir, exist_ok=True)
    for name in ANCHORS:
        smi = ST.VALIDATION[name]
        xyz = ST.gen_xyz(smi, seed=cfg["active_learning"]["seed"])
        if xyz is None:
            print(f"[fail] geometry 생성 실패: {name}")
            continue
        path = IO.save_input(name, xyz, cfg, out_dir)
        n_atoms = len(xyz.strip().splitlines())
        print(f"[ok] {name} ({n_atoms} atoms) -> {path}")


if __name__ == "__main__":
    main()
