"""
structures.py — 후보 구조 풀 생성 + 3D 지오메트리.

두 가지를 만든다:
  1) 검증 세트(validation): Kim 2026이 실험 대조한 신경작용제 (공개 구조). 프로토콜 재현용.
  2) 유도체 풀(pool): 백본 SMILES 템플릿의 치환 위치([*:1] 등)에 R-그룹 조합을 대입해
     조합 공간을 열거. Kim 2026의 4위치×{Me,Et,Pr}=3^4=81 을 일반화 (>1000 확장 가능).

※ 백본 SMILES는 config가 아니라 여기 TEMPLATES에 둔다. 민감 구조는 Ku가 직접 채운다
   (플레이스홀더 제공). 실제 계산은 로컬에서만 수행.
"""
from __future__ import annotations
import argparse, itertools, os, csv
from rdkit import Chem
from rdkit.Chem import AllChem

# ---- 검증 세트: 공개된 신경작용제 SMILES (스펙트럼 예측 대상; 합성정보 아님) ----
# 프로토콜 재현(1-A)용. 필요한 것만 사용.
VALIDATION = {
    "GB_sarin":   "CC(C)OP(C)(=O)F",
    "GD_soman":   "CC(C)(C)C(C)OP(C)(=O)F",
    "GA_tabun":   "CCOP(=O)(C#N)N(C)C",
    "VX":         "CCOP(=O)(C)SCCN(C(C)C)C(C)C",
}

# ---- 유도체 풀 템플릿 ----
# 백본에 치환점을 [*:n] 으로 표기. R_GROUPS에서 조합 대입.
# 아래는 '구조식 골격'만 담은 플레이스홀더 — Ku가 A-series 백본으로 교체.
TEMPLATES = {
    # OP 신경작용제-계열 2D 그리드: R1-O-P(R2)(=O)F ([*:1]=O치환, [*:2]=P치환).
    # R2=Me 행이 Kim 2026 실험검증 5종(GB/EtGB/nPrGB/iBuGB/GF)을 포함 → 검증 앵커.
    # AL replay 벤치마크용 화학공간 (스펙트럼 DB·탐지 연구, 계산 전용).
    "OP_grid": "[*:1]OP([*:2])(=O)F",
    # A-series 백본은 Ku가 여기에 추가 (플레이스홀더):
    # "A_series": "<Ku 입력>",
}
R_GROUPS = {              # 라벨: 치환 알킬 조각 (O·P 두 위치 공통)
    "Me":  "C",           # methyl
    "Et":  "CC",          # ethyl
    "nPr": "CCC",         # n-propyl
    "iPr": "C(C)C",       # isopropyl
    "iBu": "CC(C)C",      # isobutyl
    "sBu": "C(C)CC",      # sec-butyl
    "cPn": "C1CCCC1",     # cyclopentyl
    "cHx": "C1CCCCC1",    # cyclohexyl
}                          # 8 × 8 = 64종 (R2=Me 행에 GB/EtGB/nPrGB/iBuGB/GF)


def _attach(backbone_smiles: str, subs: dict[int, str]) -> str | None:
    """[*:n] 위치에 subs[n] R-그룹을 부착한 SMILES 반환."""
    mol = Chem.MolFromSmiles(backbone_smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    # 각 더미원자([*:n])를 R그룹 첫 원자로 연결
    dummies = [(a.GetIdx(), a.GetAtomMapNum()) for a in rw.GetAtoms() if a.GetSymbol() == "*"]
    for idx, mapnum in sorted(dummies, key=lambda t: -t[0]):
        frag = Chem.MolFromSmiles(subs[mapnum])
        if frag is None:
            return None
        combo = Chem.CombineMols(rw.GetMol(), frag)
        rw = Chem.RWMol(combo)
        # 더미와 이웃 찾기
        d_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*" and a.GetAtomMapNum() == mapnum]
        if not d_idx:
            return None
        d = d_idx[0]
        nbr = [n.GetIdx() for n in rw.GetAtomWithIdx(d).GetNeighbors()][0]
        # frag 첫 원자 = combine 후 원래 원자수 offset
        frag_start = rw.GetNumAtoms() - frag.GetNumAtoms()
        rw.AddBond(nbr, frag_start, Chem.BondType.SINGLE)
        rw.RemoveAtom(d)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def enumerate_pool(templates=TEMPLATES, rgroups=R_GROUPS) -> list[dict]:
    """모든 백본×치환조합 열거 → [{name, smiles}]."""
    out, seen = [], set()
    labels = list(rgroups.keys())
    for bb_name, bb in templates.items():
        mol = Chem.MolFromSmiles(bb)
        if mol is None:
            print(f"[warn] backbone 파싱 실패: {bb_name}")
            continue
        n_sites = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "*")
        if n_sites == 0:
            continue
        for combo in itertools.product(labels, repeat=n_sites):
            mapnums = sorted({a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetSymbol() == "*"})
            subs = {mn: rgroups[lab] for mn, lab in zip(mapnums, combo)}
            smi = _attach(bb, subs)
            if smi and smi not in seen:
                seen.add(smi)
                name = f"{bb_name}_" + "".join(str(labels.index(c) + 1) for c in combo)
                out.append({"name": name, "smiles": smi})
    return out


def gen_xyz(smiles: str, seed: int = 42) -> str | None:
    """ETKDG 3D 임베딩 + MMFF 사전최적화 → xyz 블록(원소 x y z)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/pool.csv")
    ap.add_argument("--include-validation", action="store_true")
    args = ap.parse_args()

    rows = []
    if args.include_validation:
        for name, smi in VALIDATION.items():
            rows.append({"name": name, "smiles": Chem.CanonSmiles(smi), "set": "validation"})
    for r in enumerate_pool():
        r["set"] = "pool"
        rows.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "smiles", "set"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} 구조 → {args.out}  (validation={sum(r['set']=='validation' for r in rows)}, pool={sum(r['set']=='pool' for r in rows)})")


if __name__ == "__main__":
    main()
