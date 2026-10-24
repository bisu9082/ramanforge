"""
ramanforge Step 4-b helper
SMILES -> structural feature vector (input to the AL surrogate).

A 'cheap' structural descriptor for predicting the full spectrum (the label).
Only topological, compositional and physicochemical features obtainable without DFT are used
(the point of AL is to use few expensive labels, so the input must be cheap).
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# fixed feature order, for reproducibility
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


_N_MORGAN = 32  # short Morgan fingerprint, to reinforce structural topology


def smiles_to_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES parsing failed: {smiles}")
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
