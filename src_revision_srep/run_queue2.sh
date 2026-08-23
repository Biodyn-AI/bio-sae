#!/bin/bash
# Second queue: the regulatory measurement on independent external perturbation datasets.
# Waits for the primary queue so the accelerator is never double-booked.
set -u
cd "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
LOG=experiments/revision_srep
stamp () { echo "=== $(date '+%H:%M:%S') $* ==="; }

while pgrep -f "run_queue.sh" > /dev/null; do sleep 60; done
stamp "primary queue finished, starting external datasets"

stamp "E10 papalexi (THP-1, ECCITE-seq, 18,649 genes)"
python3 -u src_revision_srep/e10_external_datasets.py --dataset papalexi \
    --max-targets 16 --cells-per-target 100 --control-cells 400 \
    > "$LOG/e10_papalexi.log" 2>&1

stamp "E10 norman (K562, Perturb-seq, 33,694 genes)"
python3 -u src_revision_srep/e10_external_datasets.py --dataset norman \
    --max-targets 16 --cells-per-target 100 --control-cells 400 \
    > "$LOG/e10_norman.log" 2>&1

stamp "ALL QUEUES COMPLETE"
