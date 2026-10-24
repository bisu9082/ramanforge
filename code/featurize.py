"""
featurize.py — 값싼 분자 피처 (AL surrogate 입력).

DFT 없이 계산 가능한 피처로 surrogate를 학습해, 아직 계산 안 한 구조의
스펙트럼 특성을 예측 → acquisition이 '다음 계산할 구조'를 고른다.

피처: RDKit 2D descriptor(정규화) + Morgan fingerprint(반경2, 1024bit).
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# 대표 물리화학 descriptor (분자량, 극성, 회전결합, P/N/F 관련 대리 지표 등)
_DESC = [
    "MolWt", "TPSA", "NumRotatableBonds", "NumHAcceptors", "NumHDonors",
    "FractionCSP3", "NumHeavyAtoms", "MolLogP", "NumValenceElectrons",
]
_calc = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC)
_mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def featurize_smiles(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = np.array(_calc.CalcDescriptors(mol), dtype=float)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    fp = _mfgen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.concatenate([desc, arr])


def featurize_frame(names, smiles_list, **kw):
    """(names, X, valid_mask) 반환. 파싱 실패 구조는 제외."""
    X, keep = [], []
    for nm, smi in zip(names, smiles_list):
        v = featurize_smiles(smi, **kw)
        if v is not None:
            X.append(v)
            keep.append(nm)
    return keep, (np.vstack(X) if X else np.empty((0, 0)))
