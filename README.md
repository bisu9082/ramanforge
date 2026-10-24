# ramanforge

Data and analysis code for:

> **The computed library, not the spectroscopy, limits Raman identification of authentic
> chemical warfare agents**
> Doo-Hee Lee†, Seongchan Lee†, Sungbong Kim, Jinkwang Jeong, Rosy Oh, Hyein Kim, Ku Kang*
> *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy* (submitted)
> † equal contribution · \* corresponding author

---

## What this is

Eleven authentic chemical warfare agents were measured by Raman spectroscopy at 1064 nm
from raw detector counts. Thirty genuine acquisitions of nine of them were matched against
a **measured** reference library and against a **calibrated 54-member DFT library** under an
identical pipeline, grid and similarity metric — the only difference between the two
conditions is where the library came from.

The measured library assigns all thirty correctly. The computed library assigns thirteen.
The gap survives every repair we could apply: twelve similarity metrics, three standard
escalations of the level of theory, correction of two computed structures that were not at
the global minimum, an affine wavenumber calibration, an imposed axis offset of −20 to
+30 cm⁻¹, symmetric preprocessing, and leave-one-agent-out recalibration.

## What is here

| Path | Contents |
|---|---|
| `code/` | Analysis pipeline. `src/preprocess_v3.py` is the preprocessing specification; `v3_step3_headtohead.py` is the head-to-head comparison that every other script imports. |
| `data/` | Preprocessed spectra (`v3_measured.npz`), every derived CSV, and the metadata JSON files that the manuscript tables are generated from. |
| `computed_library/` | The calibrated computed library as line lists: harmonic wavenumber and Raman activity for each of the 57 computed structures, one CSV per molecule. This is the reusable product. |
| `conformers/` | GFN2 metadynamic conformer census (CREST) for all eleven agents. |
| `latex/` | Manuscript and supporting information source. |
| `figures/` | Figures as submitted. |

## What is not here

**Raw acquisition files of the authentic agents are not in this repository.** They are
analytical data on Schedule 1 chemicals generated in a laboratory designated by the
Organisation for the Prohibition of Chemical Weapons. The appropriate route for them is
submission to the OPCW Central Analytical Database, through which they are available to
States Parties and designated laboratories. Everything needed to reproduce every number in
the manuscript from the deposited preprocessed spectra is in this repository.

`computed_library/MANIFEST.csv` maps every entry to its ORCA input and output, its line
list, an **isomeric** SMILES and an InChIKey perceived from the optimised geometry, and
whether it is one of the 54 library members or a superseded structure. The isomeric SMILES
is the authoritative structural record: the plain SMILES in Table S-library carries no
E/Z descriptor, and Section S9 shows that the amidine configuration of A-232 is worth
0.300 in the 600--850 cm-1 window, so a structure rebuilt from the plain SMILES will not
reproduce this work.

ORCA inputs and outputs for all 57 computed structures are under `orca/` (62 MB); see
`orca/README.md` for what is there and why some of the working comments in the inputs are
in Korean.

## Reproducing the results

```bash
pip install numpy scipy pandas matplotlib
python code/v3_step3_headtohead.py      # head-to-head; writes data/v3_headtohead*.csv
python code/v3_region_by_agent.py       # window-resolved cosines
python code/v3_t118_metrics.py          # twelve-metric control
python code/v3_t119_decompose.py        # position / intensity / width decomposition
python code/v3_step5_a2_abstention.py   # abstention and the class-absent condition
python code/v3_agent_stats.py           # per-agent accuracy, McNemar, bootstrap CIs
python code/v3_detect_sensitivity.py    # band-detection criterion sensitivity
python code/v3_axis_checks.py           # affine calibration and the axis-offset scan
python code/v3_preproc_symmetry.py      # preprocessing applied symmetrically
python code/v3_loao_topk.py             # leave-one-agent-out identification, top-k ranks
python code/v3_pairrule.py              # one-to-one vs nearest-neighbour band pairing
python code/v3_geomwindow.py            # window-resolved effect of the geometry correction
python code/v3_si_tables.py             # regenerates the SI tables into latex/sitables/
```

Scripts resolve paths relative to the repository root and write into `data/`. Run
`v3_step3_headtohead.py` first; the others import from it. The order above is also the
dependency order: `v3_si_tables.py` reads what the earlier scripts write, and
`v3_agent_stats.py` reads the abstention scores, so it must follow
`v3_step5_a2_abstention.py`.

`v3_step1_audit.py` and `v3_step2_preprocess.py` rebuild `data/v3_measured.npz` from the
raw acquisition files, which are not in this repository (see **What is not here**). They
expect those files in `data/raw/`.

Two environment variables select a non-default condition:

| Variable | Effect |
|---|---|
| `RF_LIB11=1` | restricts the computed library to the same eleven members as the measured one, the candidate-count control of Table S-abstsize |
| `RF_PRECORR=1` | rebuilds the pre-correction library, needed for the level-of-theory trials of Section S8 and for Table S-geomwindow, whose captions state that they are computed on the pre-correction geometries |

Every number in the manuscript is produced by one of these scripts. No table is
hand-entered. All 345 data rows in the Supplementary Information tables were checked
against the output of a clean re-run.

## Compliance

All scheduled chemicals were held, handled and destroyed under the Chemical Weapons
Convention and its implementing legislation in the Republic of Korea, in a Schedule 1
facility declared to the National Authority. No agent was synthesised for this work. This
repository contains no synthetic route, precursor, purification procedure, quantity,
scale-up, formulation or dissemination information.

## License

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0.

## Citation

Citation metadata will be added here when the DOI is issued.
