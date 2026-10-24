#!/usr/bin/env bash
# Regenerates every number and every table in the manuscript, and the analysis figures,
# in dependency order. The manuscript's figure files are these outputs renamed: the mapping
# is given in README.md.
# Run from the repository root. Exits non-zero on the first failure.
#
# The raw per-acquisition spectra are not deposited (they are the primary records of a
# Schedule 1 facility), so the two steps that read them are skipped unless data/raw/ is
# present; data/v3_measured.npz, which is what every later step reads, is deposited.
# Set RAMANFORGE_RAW=1 to force them to run.
set -euo pipefail
cd "$(dirname "$0")"

run () {   # run <script> [VAR=value ...]
  local s="$1"; shift
  printf '%-24s' "$s"
  env "$@" python3 "code/$s.py" > "/tmp/ramanforge_$s.log" 2>&1
  echo 'ok'
}

if [[ -d data/raw || "${RAMANFORGE_RAW:-0}" == 1 ]]; then
  run v3_step1_audit                  # raw acquisition audit
  run v3_step2_preprocess             # preprocessing; writes data/v3_measured.npz
else
  echo 'v3_step1_audit          skipped (data/raw not present)'
  echo 'v3_step2_preprocess     skipped (data/raw not present)'
fi

run v3_step3_headtohead               # the head-to-head comparison
run v3_region_by_agent                # window-resolved cosines
run v3_t118_metrics                   # twelve-metric control
run v3_t119_decompose                 # position / intensity / width decomposition
# The level-of-theory trials of Section S8 were run on the pre-correction library, as
# their captions state, so this one step needs RF_PRECORR=1 to reproduce those tables.
run v3_step4_b2_evaluate RF_PRECORR=1 # level-of-theory trials
run v3_step5_a2_abstention            # abstention and the class-absent condition
run v3_abstention_bycls               # abstention AUC by deleted class
run v3_agent_stats                    # per-agent accuracy, McNemar, bootstrap intervals
run v3_detect_sensitivity             # band-detection criterion sensitivity
run v3_axis_checks                    # affine calibration and the axis-offset scan
run v3_preproc_symmetry               # preprocessing applied symmetrically
run v3_loao_topk                      # leave-one-agent-out identification, top-k ranks
run v3_pairrule                       # one-to-one vs nearest-neighbour band pairing
run v3_geomwindow                     # window-resolved effect of the geometry correction
run v3_calibsens                      # sensitivity of the corrected library to the calibration
run v3_freqcheck                      # vibrational check of every library member
run v3_repeatability                  # within-vial repeatability, per vial
run v3_hn2_identity                   # what the vial labeled HN2 contains
run v3_fig_hn2                        # figures/v3_fig_hn2.png          (Figure S-hn2)
run v3_export_library                 # computed_library/ line lists and MANIFEST.csv
run v3_kondo_compare                  # external comparison with Kondo et al. 2018
run v3_kondo_stats                    # chance levels and tolerance sensitivity for it
run v3_kondo_asquery                  # published band lists as queries (cross-provenance)
run v3_affine_window                  # window diagnosis under the monotone affine map
run v3_conformer_population           # what one conformer represents at 298 K
run v3_fig_kondo                      # figures/kondo_comparison.png
run v3_hn2_sensitivity                # headline result with and without the HN2 vial
run v3_fig_calibration                # figures/v3_fig_calibration.png  (Figure 2)
run v3_step4_b2_figure RF_PRECORR=1   # figures/v3_fig_b2.png           (Figure 5)
run v3_fig_t118_t119                  # figures/v3_fig_metrics_decomp.png (Figure 4)
run v3_fig_design                     # figures/v3_fig_design.png       (Figure 1)
run v3_fig_si_spectra                 # figures/v3_fig_si_spectra.png   (Figure S-spectra)
run v3_fig_step4                      # figures/v3_fig_step4.png        (Figure S-step4)
run v3_fig_a232                       # figures/v3_fig_a232.png         (Figure 6)
run v3_graphical_abstract             # figures/graphical_abstract.png and .pdf
run v3_si_tables                      # every SI table, into latex/sitables/

echo
echo 'All steps completed. Tables are in latex/sitables/, figures in figures/.'
