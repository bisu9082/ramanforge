# Conformer census

GFN2 metadynamic conformer search (CREST, 6 kcal/mol window) run from each agent's
published B3LYP geometry.

- `<AGENT>_paper.xyz` — the published geometry relaxed at GFN2, i.e. the nearest local
  minimum to the structure used in the manuscript's computed library.
- `<AGENT>_conformer_energies.txt` — relative energies (kcal/mol) of every conformer the
  search returned, ascending. Conformational entropy in the manuscript is
  −R Σ pᵢ ln pᵢ over the 298 K Boltzmann weights of this list.
- `EXP_A232_E_best.xyz`, `EXP_A232_orig_best.xyz`, `EXP_VX_best.xyz` — lowest-energy
  conformer found, used as the ORCA starting geometry for the corrected calculations
  reported in Section S9.

Nine of the eleven agents lie within 1.74 kcal/mol of their search minimum. GF and A-232
do not; both were reoptimised at B3LYP/6-311++G(d,p) and the errors confirmed at 4.57 and
9.47 kcal/mol respectively.
