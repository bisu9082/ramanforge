"""
structures.py: candidate structure pool generation plus 3D geometries.

Two things are built:
  1) validation set: the nerve agents that Kim 2026 compared against experiment (public structures). For reproducing the protocol.
  2) derivative pool: the combinatorial space is enumerated by substituting R-group combinations
     into the substitution positions ([*:1] and the like) of the backbone SMILES templates. This
     generalizes the 4 positions×{Me,Et,Pr}=3^4=81 of Kim 2026 (extensible to >1000).

Note: the backbone SMILES are kept here in TEMPLATES rather than in config. Sensitive structures
   are filled in by Ku directly (placeholders are provided). The actual calculations are run locally only.
"""
from __future__ import annotations
import argparse, itertools, os, csv
from rdkit import Chem
from rdkit.Chem import AllChem

# ---- validation set: published nerve-agent SMILES (targets for spectrum prediction; no synthesis information) ----
# for protocol reproduction (1-A); use only what is needed
VALIDATION = {
    "GB_sarin":   "CC(C)OP(C)(=O)F",
    "GD_soman":   "CC(C)(C)C(C)OP(C)(=O)F",
    "GA_tabun":   "CCOP(=O)(C#N)N(C)C",
    "VX":         "CCOP(=O)(C)SCCN(C(C)C)C(C)C",
}

# ---- derivative pool templates ----
# substitution points on the backbone are marked [*:n]; combinations are substituted from R_GROUPS
# placeholder holding the skeleton only; replaced with the A-series backbone
TEMPLATES = {
    # two-dimensional grid of organophosphorus nerve-agent analogs: R1-O-P(R2)(=O)F ([*:1] on O, [*:2] on P)
    # the R2=Me row contains the five experimentally validated species of Kim 2026 (GB/EtGB/nPrGB/iBuGB/GF) -> validation anchor
    # chemical space for the AL replay benchmark (spectral database and detection research; computational only)
    "OP_grid": "[*:1]OP([*:2])(=O)F",
    # the A-series backbone is added here (placeholder):
    # "A_series": "<to be supplied>",
}
R_GROUPS = {              # labels: substituent alkyl fragments (shared by the O and P positions)
    "Me":  "C",           # methyl
    "Et":  "CC",          # ethyl
    "nPr": "CCC",         # n-propyl
    "iPr": "C(C)C",       # isopropyl
    "iBu": "CC(C)C",      # isobutyl
    "sBu": "C(C)CC",      # sec-butyl
    "cPn": "C1CCCC1",     # cyclopentyl
    "cHx": "C1CCCCC1",    # cyclohexyl
}                          # 8 x 8 = 64 species (the R2=Me row holds GB/EtGB/nPrGB/iBuGB/GF)


def _attach(backbone_smiles: str, subs: dict[int, str]) -> str | None:
    """Return SMILES with the subs[n] R-group attached at each [*:n] site."""
    mol = Chem.MolFromSmiles(backbone_smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    # bond each dummy atom ([*:n]) to the first atom of its R group
    dummies = [(a.GetIdx(), a.GetAtomMapNum()) for a in rw.GetAtoms() if a.GetSymbol() == "*"]
    for idx, mapnum in sorted(dummies, key=lambda t: -t[0]):
        frag = Chem.MolFromSmiles(subs[mapnum])
        if frag is None:
            return None
        combo = Chem.CombineMols(rw.GetMol(), frag)
        rw = Chem.RWMol(combo)
        # locate the dummy atom and its neighbour
        d_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*" and a.GetAtomMapNum() == mapnum]
        if not d_idx:
            return None
        d = d_idx[0]
        nbr = [n.GetIdx() for n in rw.GetAtomWithIdx(d).GetNeighbors()][0]
        # the first atom of the fragment sits at the original atom count as offset after combining
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
    """Enumerate all backbone × substituent combinations → [{name, smiles}]."""
    out, seen = [], set()
    labels = list(rgroups.keys())
    for bb_name, bb in templates.items():
        mol = Chem.MolFromSmiles(bb)
        if mol is None:
            print(f"[warn] backbone parsing failed: {bb_name}")
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
    """ETKDG 3D embedding + MMFF pre-optimization → xyz block (element x y z)."""
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
    print(f"[ok] {len(rows)} structures → {args.out}  (validation={sum(r['set']=='validation' for r in rows)}, pool={sum(r['set']=='pool' for r in rows)})")


if __name__ == "__main__":
    main()
