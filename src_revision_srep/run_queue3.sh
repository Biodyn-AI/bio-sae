#!/bin/bash
# External perturbation datasets. Launched directly: the primary queue is finished and the
# accelerator is free, so no wait loop is needed. (The previous version waited on a pgrep
# pattern that its own watcher process matched, which is why it never started.)
set -u
cd "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
LOG=experiments/revision_srep
stamp () { echo "=== $(date '+%H:%M:%S') $* ===" | tee -a "$LOG/queue2.log"; }

stamp "E10 papalexi (THP-1, ECCITE-seq, 18,649 genes)"
python3 -u src_revision_srep/e10_external_datasets.py --dataset papalexi \
    --max-targets 16 --cells-per-target 100 --control-cells 400 > "$LOG/e10_papalexi.log" 2>&1
stamp "E10 papalexi done (exit $?)"

stamp "E10 norman (K562, Perturb-seq, 33,694 genes)"
python3 -u src_revision_srep/e10_external_datasets.py --dataset norman \
    --max-targets 16 --cells-per-target 100 --control-cells 400 > "$LOG/e10_norman.log" 2>&1
stamp "E10 norman done (exit $?)"

stamp "ALL QUEUES COMPLETE"
