#!/usr/bin/env python3
"""
Train SAE + analyze features + annotate for all 18 layers sequentially.
Skips layers that already have a sae_final.pt file.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 02b_train_all_layers.py [--expansion 4] [--k 32] [--epochs 5] [--subsample 1000000]
"""

import os
import sys
import json
import time
import subprocess
import argparse
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
SRC_DIR = os.path.join(PROJ_DIR, "src")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")
PYTHON = os.path.expanduser("~/anaconda3/envs/bio_mech_interp/bin/python")

N_LAYERS = 18


def run_script(script, args_list, description):
    """Run a Python script and return success/failure."""
    cmd = [PYTHON, os.path.join(SRC_DIR, script)] + args_list
    print(f"\n  Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SRC_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        # Print last 10 lines of stderr
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-10:]:
            print(f"    {line}")
        return False
    else:
        # Print last 5 lines of stdout (summary)
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines[-5:]:
            print(f"    {line}")
        print(f"  OK ({elapsed:.1f}s)")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--subsample', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    print("=" * 70)
    print("BATCH SAE TRAINING: ALL 18 LAYERS")
    print(f"  Expansion: {args.expansion}x, k={args.k}, epochs={args.epochs}")
    print(f"  Subsample: {args.subsample:,}")
    print("=" * 70)

    total_t0 = time.time()
    results = {}

    for layer in range(N_LAYERS):
        run_name = f"layer{layer:02d}_x{args.expansion}_k{args.k}"
        run_dir = os.path.join(SAE_BASE, run_name)

        print(f"\n{'=' * 70}")
        print(f"LAYER {layer}/17 ({run_name})")
        print(f"{'=' * 70}")

        # Check if activations exist
        act_path = os.path.join(DATA_DIR, f"layer_{layer:02d}_activations.npy")
        if not os.path.exists(act_path):
            print(f"  SKIP: activations not found ({act_path})")
            results[layer] = 'skipped_no_data'
            continue

        # Step 1: Train SAE (skip if already done)
        model_path = os.path.join(run_dir, "sae_final.pt")
        if os.path.exists(model_path):
            print(f"  SAE already trained: {model_path}")
        else:
            ok = run_script("02_train_sae.py", [
                '--layer', str(layer),
                '--expansion', str(args.expansion),
                '--k', str(args.k),
                '--epochs', str(args.epochs),
                '--subsample', str(args.subsample),
                '--lr', str(args.lr),
            ], f"Train SAE layer {layer}")
            if not ok:
                results[layer] = 'train_failed'
                continue

        # Step 2: Feature analysis (skip if already done)
        catalog_path = os.path.join(run_dir, "feature_catalog.json")
        if os.path.exists(catalog_path):
            print(f"  Feature catalog exists: {catalog_path}")
        else:
            ok = run_script("03_analyze_features.py", [
                '--layer', str(layer),
                '--expansion', str(args.expansion),
                '--k', str(args.k),
            ], f"Analyze features layer {layer}")
            if not ok:
                results[layer] = 'analyze_failed'
                continue

        # Step 3: Ontology annotation (skip if already done)
        ann_path = os.path.join(run_dir, "feature_annotations.json")
        if os.path.exists(ann_path):
            print(f"  Annotations exist: {ann_path}")
        else:
            ok = run_script("04_annotate_features.py", [
                '--layer', str(layer),
                '--expansion', str(args.expansion),
                '--k', str(args.k),
            ], f"Annotate features layer {layer}")
            if not ok:
                results[layer] = 'annotate_failed'
                continue

        results[layer] = 'complete'

    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - total_t0

    print(f"\n\n{'=' * 70}")
    print(f"BATCH COMPLETE: {total_time/60:.1f} min total")
    print(f"{'=' * 70}")

    for layer in range(N_LAYERS):
        status = results.get(layer, 'unknown')
        marker = "OK" if status == 'complete' else status.upper()
        print(f"  Layer {layer:2d}: {marker}")

    n_complete = sum(1 for v in results.values() if v == 'complete')
    print(f"\n  {n_complete}/{N_LAYERS} layers complete")

    # Save summary
    summary_path = os.path.join(SAE_BASE, "batch_training_summary.json")
    summary = {
        'config': {
            'expansion': args.expansion,
            'k': args.k,
            'epochs': args.epochs,
            'subsample': args.subsample,
        },
        'results': {str(k): v for k, v in results.items()},
        'n_complete': n_complete,
        'total_time_s': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == '__main__':
    main()
