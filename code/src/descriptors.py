"""
ramanforge Step 4-b 보조
SMILES -> 구조 특징벡터 (AL surrogate 입력).

full 스펙트럼(라벨)을 예측하기 위한 '값싼' 구조 기술자.
DFT 없이 얻을 수 있는 위상/조성/물리화학 특징만 사용
(AL의 요점: 비싼 라벨을 적게 쓰는 것이므로 입력은 값싸야 함).
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# 고정 특징 순서 (재현성)
_FUNC = [
    ("MolWt", Descriptors.MolWt),
    ("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms()),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("TPSA", Descriptors.TPSA),
    ("LabuteASA", Descriptors.LabuteASA),
    ("NumCH3", lambda m: _count_smarts(m, "[CH3]")),
    ("NumCH2", lambda m: _count_smarts(m, "[CH2]")),
    ("NumCH", lambda m: _count_smarts(m, "[CH1]")),
    ("NumC", lambda m: sum(a.GetSymbol() == "C" for a in m.GetAtoms())),
    ("NumO", lambda m: sum(a.GetSymbol() == "O" for a in m.GetAtoms())),
    ("NumP", lambda m: sum(a.GetSymbol() == "P" for a in m.GetAtoms())),
    ("NumF", lambda m: sum(a.GetSymbol() == "F" for a in m.GetAtoms())),
    ("NumRingC", lambda m: sum(a.GetSymbol() == "C" and a.IsInRing() for a in m.GetAtoms())),
]


def _count_smarts(mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(patt)) if patt else 0


def feature_names() -> list[str]:
    base = [n for n, _ in _FUNC]
    return base + [f"MorganBit_{i}" for i in range(_N_MORGAN)]


_N_MORGAN = 32  # 짧은 Morgan fingerprint (구조 위상 보강)


def smiles_to_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 파싱 실패: {smiles}")
    mol = Chem.AddHs(mol)
    scalar = np.array([f(mol) for _, f in _FUNC], dtype=float)

    gen = AllChem.GetMorganGenerator(radius=2, fpSize=_N_MORGAN)
    fp = gen.GetFingerprint(mol)
    bits = np.array(fp, dtype=float)

    return np.concatenate([scalar, bits])


if __name__ == "__main__":
    for smi in ["COP(C)(=O)F", "CC(C)OP(C)(=O)F", "CP(=O)(F)OC1CCCCC1"]:
        v = smiles_to_features(smi)
        print(f"{smi:28s} -> dim={len(v)}  head={v[:6]}")
