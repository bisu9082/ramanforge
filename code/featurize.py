"""
featurize.py: cheap molecular features (input to the AL surrogate).

The surrogate is trained on features computable without DFT and predicts the spectral
properties of structures not yet computed → acquisition then picks the 'next structure to compute'.

Features: RDKit 2D descriptors (normalized) + Morgan fingerprint (radius 2, 1024 bit).
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# representative physicochemical descriptors (molecular weight, polarity, rotatable bonds, P/N/F proxies)
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
    """Return (names, X, valid_mask). Structures that fail parsing are excluded."""
    X, keep = [], []
    for nm, smi in zip(names, smiles_list):
        v = featurize_smiles(smi, **kw)
        if v is not None:
            X.append(v)
            keep.append(nm)
    return keep, (np.vstack(X) if X else np.empty((0, 0)))
