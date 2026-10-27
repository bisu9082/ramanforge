# ramanforge

Data and analysis code for:

> **A calibrated density functional theory Raman library does not reach molecular identification of authentic
> chemical warfare agents**
> Doo-Hee Lee†, Seongchan Lee†, Sungbong Kim, Jinkwang Jeong, Rosy Oh, Hyein Kim, Ku Kang*
> *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy* (submitted)
> † equal contribution · \* corresponding author

---

## What this is

Eleven vials of authentic chemical warfare agent were measured by Raman spectroscopy at
1064 nm from raw detector counts. Ten are consistent with their labels; the eleventh, labeled
HN2, was measured as benzene rather than as neat agent (see `data/v3_hn2_identity.csv`).
No material was independently assayed for identity or purity. Thirty genuine acquisitions of nine of them were matched against
a **measured** reference library and against a **calibrated 54-member DFT library** under an
identical pipeline, grid and similarity metric — the only difference between the two
conditions is where the library came from.

The measured library assigns all thirty correctly. The computed library assigns thirteen.
Setting aside the three acquisitions of the mislabeled vial leaves 27 of 27 against 13 of 27;
every headline figure is reported both ways, and the exclusion raises the computed figure.
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
python code/v3_abstention_bycls.py      # abstention AUC stratified by the deleted class
python code/v3_agent_stats.py           # per-agent accuracy, McNemar, bootstrap CIs
python code/v3_detect_sensitivity.py    # band-detection criterion sensitivity
python code/v3_axis_checks.py           # affine calibration and the axis-offset scan
python code/v3_preproc_symmetry.py      # preprocessing applied symmetrically
python code/v3_loao_topk.py             # leave-one-agent-out identification, top-k ranks
python code/v3_pairrule.py              # one-to-one vs nearest-neighbour band pairing
python code/v3_geomwindow.py            # window-resolved effect of the geometry correction
python code/v3_kondo_compare.py         # external comparison with Kondo et al. 2018
python code/v3_kondo_stats.py           # chance levels and tolerance sensitivity for it
python code/v3_kondo_asquery.py         # published band lists as queries (cross-provenance)
python code/v3_fig_kondo.py             # figures/kondo_comparison.png
python code/v3_hn2_sensitivity.py       # headline result with and without the HN2 vial
python code/v3_si_tables.py             # regenerates the SI tables into latex/sitables/
```

`run_all.sh` runs all of these in dependency order, with the environment variables each one
needs, and skips the two raw-data steps when `data/raw/` is absent.

### Which script makes which figure

The manuscript's figure files are the outputs below, renamed. Both names are in `figures/`.

| Manuscript figure | File | Script |
|---|---|---|
| Figure 1 | `design.png` | `v3_fig_design.py` |
| Figure 2 | `calibration.png` | `v3_fig_calibration.py` |
| Figure 3 | `headtohead.png` | `v3_step3_headtohead.py` |
| Figure 4 | `metrics_decomp.png` | `v3_fig_t118_t119.py` |
| Figure 5 | `abstention.png` | `v3_step5_a2_abstention.py` |
| Figure 6 | `theory.png` | `v3_step4_b2_figure.py` (needs `RF_PRECORR=1`) |
| Figure 7 | `a232_decomposition.png` | `v3_fig_a232.py` |
| Figure S-spectra | `si_spectra.png` | `v3_fig_si_spectra.py` |
| Figure S-step4 | `step4_results.png` | `v3_fig_step4.py` |
| Figure S-kondo | `kondo_comparison.png` | `v3_fig_kondo.py` |
| Figure S-hn2 | `hn2_identity.png` | `v3_fig_hn2.py` |
| Graphical abstract | `graphical_abstract.png`, `.pdf` | `v3_graphical_abstract.py` |

Every number printed inside a figure is computed by its script from the deposited data,
with two stated exceptions: the three A-232 structure energies of `v3_fig_a232.py`, whose
intermediate single point is not among the deposited ORCA outputs, and the six condition
counts of the graphical abstract, which come from six different scripts. Both are
documented in the scripts' own docstrings. No other figure carries a hard-coded value;
`v3_fig_t118_t119.py` and `v3_fig_calibration.py` did until the third revision, and the
measured-library control they printed is now computed by `v3_t118_metrics.py` and
deposited.

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
| `RF_PRECORR=1` | rebuilds the pre-correction library, needed for the level-of-theory trials of Section S8 and for Table S-geomwindow, whose captions state that they are computed on the pre-correction geometries. `run_all.sh` sets it for `v3_step4_b2_evaluate.py` and for no other step |
| `RF_EXCL_QUERY=HN2` | drops the acquisitions of the named agent from the query set, leaving the libraries intact. Used by `v3_hn2_sensitivity.py`, which suppresses every write so that a sensitivity run cannot overwrite a deposited result |

Every number in the manuscript is produced by one of these scripts. No table is
hand-entered. All 345 data rows in the Supplementary Information tables were checked
against the output of a clean re-run.

## Do not use the computed library for identification

`computed_library/` is the object this paper evaluates, and the result is that it does not
work: 13 of 30 genuine acquisitions assigned correctly, self-cosines of 0.65 to 0.80, and no
operating threshold that separates a query whose answer is present from one whose answer is
absent. See `computed_library/README.md`.

## Compliance

All scheduled chemicals were held, handled and destroyed under the Chemical Weapons
Convention and its implementing legislation in the Republic of Korea, in a Schedule 1
facility declared to the National Authority. The agents were synthesized at the designated
laboratory rather than acquired from an outside source: ten were drawn from existing
declared stocks and A-232 was prepared within the declared facility under its existing
authorization, in the minimum quantity needed for a single set of measurements, because no
stock of it was held. All handling and measurement followed the institution's safety
procedures. This repository contains no synthetic route, precursor, purification procedure, quantity,
scale-up, formulation or dissemination information.

## License

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0.

## Citation

Release `v1.2.1` is the state from which every number in the manuscript was produced.
`CITATION.cff` carries the version and the repository; the article DOI will be added here
when it is issued.
