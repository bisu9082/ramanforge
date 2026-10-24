"""
v3 B2 — P–F/P–N 기술 개선 테스트용 ORCA 인풋 생성.

A1 진단: 측정↔계산 갭이 600–850 cm-1(P–F·P–N 신축)에 집중되고, 인 함유 작용제에만
나타난다(0.569 vs 0.913). conformer 가설은 기각(가장 유연한 GD가 이 구간 최고).
따라서 전방 모델의 '인 기술'을 고친다.

사다리 (효과 기대 순서 · 앵커부터 확인 후 전수)
  V0  baseline        B3LYP/6-311++G(d,p)                      = 현행 (재계산 불필요)
  V1  응축상          V0 + CPCM(물 유전율 대신 자기일관 유전율은 과함 → 기본 CPCM)
  V2  인 tight-d      V0 + P·F·S·Cl에 def2-TZVPP               (인 진동수의 고전적 병목)
  V3  분산·범위보정   wB97X-D3/6-311++G(d,p)                    = Kim 2026 future work
  V4  V1+V2 결합      최선 조합 (V1·V2가 각각 효과 있을 때만)

앵커: GF(사이클로사린, 25원자, 600–850 cos 0.562) → A-232(29원자, 0.205)
기하는 기존 baseline .out의 수렴 구조를 재사용한다(재최적화는 각 변형에서 다시 수행).
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
        note="응축상 유전 효과. 순수액상 작용제의 유전율(~20)에 가까운 아세톤으로 근사.",
        blocks=""),
    "V2_tightP": dict(
        keywords="! B3LYP 6-311++G(d,p) TightOpt NumFreq",
        note="P·F·S·Cl만 def2-TZVPP로 승격. 인 진동수의 고전적 병목(tight-d 부족) 해소.",
        blocks='%basis\n  NewGTO P  "def2-TZVPP" end\n  NewGTO F  "def2-TZVPP" end\n'
               '  NewGTO S  "def2-TZVPP" end\n  NewGTO Cl "def2-TZVPP" end\nend\n'),
    "V3_wb97xd3": dict(
        keywords="! wB97X-D3 6-311++G(d,p) TightOpt NumFreq",
        note="Kim 2026이 future work로 예고한 분산·장거리 보정 범함수.",
        blocks=""),
    "V4_cpcm_tightP": dict(
        keywords="! B3LYP 6-311++G(d,p) TightOpt NumFreq CPCM(Acetone)",
        note="V1+V2 결합. V1·V2가 각각 효과를 보일 때만 실행할 것.",
        blocks='%basis\n  NewGTO P  "def2-TZVPP" end\n  NewGTO F  "def2-TZVPP" end\n'
               '  NewGTO S  "def2-TZVPP" end\n  NewGTO Cl "def2-TZVPP" end\nend\n'),
}

NPROCS, MAXCORE = 8, 3000


def final_geometry(out_path: str) -> str:
    """ORCA .out에서 최종(수렴) Cartesian 좌표 블록을 뽑는다."""
    with open(out_path, "r", errors="ignore") as f:
        lines = f.readlines()
    idx = [i for i, ln in enumerate(lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in ln]
    if not idx:
        raise ValueError(f"좌표 블록 없음: {out_path}")
    i = idx[-1] + 2                      # 헤더 + 구분선
    rows = []
    pat = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
    for ln in lines[i:]:
        m = pat.match(ln)
        if m:
            rows.append(f"  {m[1]:<2} {float(m[2]):14.8f} {float(m[3]):14.8f} {float(m[4]):14.8f}")
        elif rows:
            break
    if not rows:
        raise ValueError(f"좌표 파싱 실패: {out_path}")
    return "\n".join(rows)


def build(tag: str, job: str, variant: str, spec: dict, xyz: str) -> str:
    name = f"B2_{tag}_{variant}"
    head = spec["keywords"] + "\n"
    blk = f"%pal nprocs {NPROCS} end\n%maxcore {MAXCORE}\n%elprop Polar 1 end\n"
    blk += spec["blocks"]
    body = f"* xyz 0 1\n{xyz}\n*\n"
    return name, f"# {spec['note']}\n# 출발 기하: {job}.out 수렴 구조\n" + head + blk + body


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
        print(f"{tag:<6} ({job}) 원자 {natoms}개 — 기하 추출 완료")
        for variant, spec in VARIANTS.items():
            name, text = build(tag, job, variant, spec, xyz)
            p = os.path.join(a.outdir, f"{name}.inp")
            with open(p, "w") as f:
                f.write(text)
            made.append((name, natoms, spec["note"]))

    # 실행 스크립트 — 우선순위 순, idempotent
    order = ["GF_V1_cpcm", "GF_V2_tightP", "GF_V3_wb97xd3",
             "A232_V1_cpcm", "A232_V2_tightP", "A232_V3_wb97xd3",
             "GF_V4_cpcm_tightP", "A232_V4_cpcm_tightP"]
    sh = ["#!/usr/bin/env bash",
          "# B2 — P 기술 개선 테스트. 우선순위 순, 이미 끝난 잡은 건너뛴다.",
          "# V4는 V1·V2가 각각 효과를 보인 뒤에만 돌릴 것 (맨 뒤에 배치).",
          "set -u",
          'ORCA=/home/k9/orca_6_1_1/orca',
          'cd "$(dirname "$0")"',
          'mkdir -p ../outputs_b2',
          ""]
    for nm in order:
        f = f"B2_{nm}"
        sh += [f'if [ ! -s ../outputs_b2/{f}.out ]; then',
               f'  echo "[$(date +%F\\ %T)] {f} 시작"',
               f'  "$ORCA" {f}.inp > ../outputs_b2/{f}.out 2>&1',
               f'  echo "[$(date +%F\\ %T)] {f} 완료"',
               'else',
               f'  echo "skip {f} (이미 있음)"',
               'fi', ""]
    sp = os.path.join(a.outdir, "run_b2.sh")
    with open(sp, "w") as f:
        f.write("\n".join(sh))
    os.chmod(sp, 0o755)

    print(f"\n생성된 인풋 {len(made)}개 → {a.outdir}")
    for nm, na, note in made:
        print(f"  {nm:<26} ({na}원자)  {note.split('.')[0]}")
    print(f"\n실행: bash {os.path.relpath(sp, ROOT)}")


if __name__ == "__main__":
    main()
