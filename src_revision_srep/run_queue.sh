#!/bin/bash
# Serial GPU queue for the remaining revision experiments.
# The accelerator is a single device, so these must not overlap. Each step is
# idempotent (existing outputs are skipped), so the queue can be restarted safely.
set -u
cd "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
PY=python3
LOG=experiments/revision_srep
MT="$PWD/experiments/phase3_multitissue/sae_models"

stamp () { echo "=== $(date '+%H:%M:%S') $* ==="; }

# Wait for the in-flight K562 extraction to finish before touching the GPU.
while pgrep -f "e2_extract_cell_features.py --cell-line k562" > /dev/null; do sleep 60; done
stamp "K562 extraction finished"

# --- E3: cell-level analysis of the K562 cohort (CPU) ------------------------
stamp "E3 k562 / atlas dictionary"
$PY -u src_revision_srep/e3_cell_level_analysis.py --cell-line k562 --tag main/k562sae \
    > "$LOG/e3_k562_atlas.log" 2>&1
stamp "E3 k562 / multi-tissue dictionary"
$PY -u src_revision_srep/e3_cell_level_analysis.py --cell-line k562 --tag main/multitissue \
    > "$LOG/e3_k562_multitissue.log" 2>&1

# --- E4: the other three cell lines (GPU) -----------------------------------
for LINE in rpe1 jurkat hepg2; do
  stamp "E4 extraction $LINE"
  $PY -u src_revision_srep/e2_extract_cell_features.py --cell-line "$LINE" --layer 11 \
      --cells-per-target 100 --control-cells 400 --max-targets 16 --n-nontf-controls 6 \
      --tag main --sae-dirs default "$MT" --sae-names k562sae multitissue \
      > "$LOG/e2_$LINE.log" 2>&1
  stamp "E3 analysis $LINE"
  $PY -u src_revision_srep/e3_cell_level_analysis.py --cell-line "$LINE" --tag main/k562sae \
      --n-sweep 10 20 50 100 > "$LOG/e3_$LINE.log" 2>&1
done

# --- E5: causal patching with independent evaluation genes (GPU) ------------
stamp "E5 causal patching"
$PY -u src_revision_srep/e5_causal_patching_v2.py --layer 11 --cell-line k562 \
    --n-features 40 --n-cells 60 --seed 42 > "$LOG/e5.log" 2>&1

# --- E9: matched cross-model design (GPU, SAE training only) ----------------
stamp "E9 matched cross-model"
$PY -u src_revision_srep/e9_matched_crossmodel.py --layers 0 5 11 17 \
    > "$LOG/e9.log" 2>&1

# --- E8: seed stability (GPU, SAE training only) ----------------------------
stamp "E8 seed stability"
$PY -u src_revision_srep/e8_seed_stability.py --layers 0 11 --seeds 1 2 3 4 5 \
    > "$LOG/e8.log" 2>&1

stamp "queue complete"
