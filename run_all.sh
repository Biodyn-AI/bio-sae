#!/usr/bin/env bash
# run_all.sh — end-to-end SAE feature-atlas pipeline for Geneformer + scGPT.
#
# Reviewer 2 (minor #4) asked for a single-command reproducibility entry point.
# This script chains the Phase 1 -> Phase 2 -> Phase 3 scripts for Geneformer
# (src/) and the parallel scGPT pipeline (scgpt_src/). It also runs every
# revision-phase analysis under src_revision/ for reviewer-requested ablations
# and controls.
#
# Before running, edit the BASE and PYTHON variables below to match your
# environment. The scripts assume the canonical folder layout documented in
# README.md:
#
#   experiments/phase1_k562/                -- Geneformer K562 activations + SAEs
#   experiments/phase3_multitissue/          -- Tabula Sapiens activations
#   experiments/scgpt_atlas/                 -- scGPT activations + SAEs
#   experiments/revision/                    -- Revision-phase outputs
#
# Usage:
#   ./run_all.sh              # runs everything sequentially
#   ./run_all.sh phase1        # just the Geneformer K562 pipeline
#   ./run_all.sh phase2
#   ./run_all.sh phase3
#   ./run_all.sh scgpt
#   ./run_all.sh revision      # the new ablations/controls
#   ./run_all.sh quick          # skip re-extraction and re-train steps

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration -- edit these paths for your environment
# ----------------------------------------------------------------------------
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/bio_mech_interp/bin/python}"
SRC="$BASE/src"
SCGPT_SRC="$BASE/scgpt_src"
REV="$BASE/src_revision"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python at $PYTHON not found."
  echo "Edit PYTHON at the top of run_all.sh or export PYTHON before running."
  exit 1
fi

echo "BASE    = $BASE"
echo "PYTHON  = $PYTHON"
echo "Invoked with args: ${*:-<none, running all phases>}"
echo

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
step() {
  echo
  echo "=============================================================="
  echo "  $1"
  echo "=============================================================="
}

run() {
  local label="$1"; shift
  echo
  echo "--- $label ---"
  "$PYTHON" "$@"
}

# ----------------------------------------------------------------------------
# Phase 1 — Geneformer K562 extraction + SAE training + annotation
# ----------------------------------------------------------------------------
phase1() {
  step "Phase 1: Geneformer K562 feature atlas"
  run "Extract residual-stream activations (18 layers, K562)" "$SRC/01_extract_activations.py"
  run "Extract remaining layers (if needed)"                   "$SRC/01b_extract_remaining_layers.py"
  run "Train SAE at all 18 layers (4x, k=32)"                  "$SRC/02b_train_all_layers.py"
  run "Feature analysis (top genes, SVD alignment)"            "$SRC/03_analyze_features.py"
  run "Ontology annotation (GO/KEGG/Reactome/STRING/TRRUST)"   "$SRC/04_annotate_features.py"
  run "SVD comparison across all 18 layers"                    "$SRC/05_compare_svd.py"
  run "Cross-layer persistence tracking"                       "$SRC/06_cross_layer.py"
}

# ----------------------------------------------------------------------------
# Phase 2 — Co-activation modules, causal patching, perturbation mapping
# ----------------------------------------------------------------------------
phase2() {
  step "Phase 2: co-activation, causal patching, perturbation"
  run "Co-activation graph + Leiden modules per layer" "$SRC/07_feature_coactivation.py" --all
  run "Causal feature patching at L11 (50 features)"   "$SRC/08_causal_patching.py" --layer 11
  run "Perturbation response mapping (100 targets)"    "$SRC/09_perturbation_response.py" --layer 11 --n-targets 100
  run "Unannotated feature characterization"           "$SRC/10_novel_features.py"
  run "Cross-layer computational-graph highways"       "$SRC/11_computational_graph.py"
}

# ----------------------------------------------------------------------------
# Phase 3 — Multi-tissue control (K562 + Tabula Sapiens pool)
# ----------------------------------------------------------------------------
phase3() {
  step "Phase 3: multi-tissue control"
  run "Extract Tabula Sapiens activations through Geneformer"  "$SRC/12a_extract_tabula_sapiens.py"
  run "Pool + train multi-tissue SAEs"                         "$SRC/12b_pool_and_train.py"
  run "Analyze and annotate multi-tissue features"             "$SRC/12c_analyze_and_annotate.py"
  run "Multi-tissue perturbation specificity test"             "$SRC/12d_perturbation_test.py"
  run "K562-only vs multi-tissue comparison"                   "$SRC/12e_compare_results.py"
}

# ----------------------------------------------------------------------------
# scGPT — parallel pipeline
# ----------------------------------------------------------------------------
scgpt() {
  step "scGPT: parallel pipeline"
  run "Extract scGPT activations on Tabula Sapiens" "$SCGPT_SRC/01_extract_activations.py"
  run "Train scGPT SAEs (12 layers, 4x, k=32)"      "$SCGPT_SRC/02_train_sae.py"
  run "scGPT feature analysis"                       "$SCGPT_SRC/03_analyze_features.py"
  run "scGPT ontology annotation"                    "$SCGPT_SRC/04_annotate_features.py"
  run "scGPT co-activation modules"                  "$SCGPT_SRC/07_coactivation.py"
  run "scGPT causal patching (L7)"                   "$SCGPT_SRC/08_causal_patching.py"
  run "scGPT cross-layer highways"                   "$SCGPT_SRC/11_computational_graph.py"
  run "scGPT cell-type enrichment"                   "$SCGPT_SRC/compute_celltype_enrichments.py"
}

# ----------------------------------------------------------------------------
# Revision — new ablations and controls for the BMC Genomics revision
# ----------------------------------------------------------------------------
revision() {
  step "Revision: new reviewer-requested analyses"
  run "Phase B1 -- SVD threshold sweep"                  "$REV/b1_svd_threshold_sweep.py"
  run "Phase B2 -- PMI memory + Leiden sweep"            "$REV/b2_pmi_and_leiden_sweep.py"
  run "Phase B3 -- Cross-layer null + consecutive sweep" "$REV/b3_crosslayer_null_and_consecutive.py"
  run "Phase B4 -- Unannotated feature characterization" "$REV/b4_unannotated_characterize.py"
  run "Phase C1 -- Hyperparameter ablation at L11"       "$REV/c1_hyperparam_ablation.py"
  run "Phase C2 -- Multi-database perturbation test"     "$REV/c2_multidb_perturbation.py"
  run "Phase C5 -- Matched-input cross-model comparison" "$REV/c5_matched_data_crossmodel.py"
  run "Phase C6 -- Batch/assay confound analysis"        "$REV/c6_batch_assay_analysis.py"
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
case "${1:-all}" in
  phase1)   phase1 ;;
  phase2)   phase2 ;;
  phase3)   phase3 ;;
  scgpt)    scgpt ;;
  revision) revision ;;
  quick)
    step "Quick pipeline: revision analyses only (no re-extraction, no retraining)"
    revision
    ;;
  all)
    phase1
    phase2
    phase3
    scgpt
    revision
    ;;
  *)
    echo "Usage: $0 [phase1|phase2|phase3|scgpt|revision|quick|all]"
    exit 2
    ;;
esac

echo
echo "=============================================================="
echo "  run_all.sh done"
echo "=============================================================="
