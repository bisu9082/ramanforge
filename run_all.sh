#!/usr/bin/env bash
# Regenerates every number, table and figure in the manuscript, in dependency order.
# Run from the repository root. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")"

steps=(
  v3_step1_audit          # raw acquisition audit
  v3_step2_preprocess     # preprocessing; writes data/v3_measured.npz
  v3_step3_headtohead     # the head-to-head comparison
  v3_region_by_agent      # window-resolved cosines
  v3_t118_metrics         # twelve-metric control
  v3_t119_decompose       # position / intensity / width decomposition
  v3_step4_b2_evaluate    # level-of-theory trials
  v3_step5_a2_abstention  # abstention and the class-absent condition
  v3_agent_stats          # per-agent accuracy, McNemar, bootstrap intervals
  v3_detect_sensitivity   # band-detection criterion sensitivity
  v3_axis_checks          # affine calibration and the axis-offset scan
  v3_preproc_symmetry     # preprocessing applied symmetrically
  v3_loao_topk            # leave-one-agent-out identification, top-k ranks
  v3_pairrule             # one-to-one vs nearest-neighbour band pairing
  v3_geomwindow           # window-resolved effect of the geometry correction
  v3_calibsens            # sensitivity of the corrected library to the calibration
  v3_export_library       # computed_library/ line lists and MANIFEST.csv
  v3_si_tables            # every SI table, into latex/sitables/
)

for s in "${steps[@]}"; do
  printf '%-24s' "$s"
  python3 "code/$s.py" > "/tmp/ramanforge_$s.log" 2>&1
  echo 'ok'
done
echo
echo 'All steps completed. Tables are in latex/sitables/, figures in figures/.'
