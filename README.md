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

ORCA output files (54 MB) are archived at the Zenodo DOI rather than here.

## Reproducing the results

```bash
pip install numpy scipy pandas matplotlib
cd code
python v3_step3_headtohead.py      # head-to-head; writes data/v3_headtohead*.csv
python v3_region_by_agent.py       # window-resolved cosines
python v3_t118_metrics.py          # twelve-metric control
python v3_t119_decompose.py        # position / intensity / width decomposition
python v3_step5_a2_abstention.py   # abstention and the class-absent condition
python v3_si_tables.py             # regenerates every table in the SI
```

Scripts resolve paths relative to the repository root and write into `data/`. Run
`v3_step3_headtohead.py` first; the others import from it.

Every number in the manuscript is produced by one of these scripts. No table is
hand-entered.

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
