#!/usr/bin/env python3
"""
Phase 3, Script 2: Pool K562 + Tabula Sapiens activations and train SAEs.

For each of 4 layers (0, 5, 11, 17):
  1. Load K562 memmap (~4.05M positions) — subsample 500K
  2. Load Tabula Sapiens memmap (~5.6M positions) — subsample 500K
  3. Concatenate → 1M positions
  4. Compute activation mean on pooled data
  5. Train TopK SAE (d=1152, 4x=4608, k=32, 5 epochs, lr=3e-4, batch=4096)

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 12b_pool_and_train.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_model import TopKSAE, SAETrainer

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")

K562_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
TS_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/ts_activations")
OUT_BASE = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/sae_models")

HIDDEN_DIM = 1152
EXPANSION = 4
K_VAL = 32
EPOCHS = 5
LR = 3e-4
BATCH_SIZE = 4096
LOG_EVERY = 500
CHECKPOINT_EVERY = 10000

TARGET_LAYERS = [0, 5, 11, 17]
K562_SUBSAMPLE = 500000
TS_SUBSAMPLE = 500000


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def pool_activations(layer, k562_dir, ts_dir, k562_n, ts_n, seed=42):
    """Load and pool K562 + Tabula Sapiens activations for a layer."""
    rng = np.random.RandomState(seed)

    # Load K562
    k562_path = os.path.join(k562_dir, f"layer_{layer:02d}_activations.npy")
    k562_mmap = np.load(k562_path, mmap_mode='r')
    n_k562 = len(k562_mmap)
    print(f"    K562: {n_k562:,} positions")

    k562_n_actual = min(k562_n, n_k562)
    k562_idx = rng.choice(n_k562, k562_n_actual, replace=False)
    k562_idx.sort()

    # Load Tabula Sapiens
    ts_path = os.path.join(ts_dir, f"layer_{layer:02d}_activations.npy")
    ts_mmap = np.load(ts_path, mmap_mode='r')
    n_ts = len(ts_mmap)
    print(f"    Tabula Sapiens: {n_ts:,} positions")

    ts_n_actual = min(ts_n, n_ts)
    ts_idx = rng.choice(n_ts, ts_n_actual, replace=False)
    ts_idx.sort()

    # Load subsamples into RAM
    print(f"    Loading K562 subsample ({k562_n_actual:,})...")
    k562_data = k562_mmap[k562_idx].copy()
    print(f"    Loading TS subsample ({ts_n_actual:,})...")
    ts_data = ts_mmap[ts_idx].copy()

    # Concatenate
    pooled = np.concatenate([k562_data, ts_data], axis=0)
    print(f"    Pooled: {len(pooled):,} positions ({len(pooled) * HIDDEN_DIM * 4 / 1e9:.2f} GB)")

    del k562_data, ts_data
    return pooled


def center_activations(activations):
    """Compute mean for centering (streaming)."""
    n = len(activations)
    chunk_size = 50000
    running_sum = np.zeros(activations.shape[1], dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        running_sum += activations[start:end].astype(np.float64).sum(axis=0)
    mean = (running_sum / n).astype(np.float32)
    return mean


def main():
    n_features = EXPANSION * HIDDEN_DIM

    print("=" * 70)
    print("PHASE 3: POOL ACTIVATIONS AND TRAIN SAEs")
    print(f"  Layers: {TARGET_LAYERS}")
    print(f"  K562 subsample: {K562_SUBSAMPLE:,}")
    print(f"  TS subsample: {TS_SUBSAMPLE:,}")
    print(f"  Features: {n_features} ({EXPANSION}x)")
    print(f"  TopK: {K_VAL}, Epochs: {EPOCHS}")
    print("=" * 70)

    total_t0 = time.time()

    # Check TS activations exist
    ts_check = os.path.join(TS_DIR, f"layer_{TARGET_LAYERS[0]:02d}_activations.npy")
    if not os.path.exists(ts_check):
        print(f"ERROR: TS activations not found: {ts_check}")
        print("  Run 12a_extract_tabula_sapiens.py first.")
        sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")

    all_results = {}

    for layer in TARGET_LAYERS:
        run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
        run_dir = os.path.join(OUT_BASE, run_name)
        final_path = os.path.join(run_dir, "sae_final.pt")

        if os.path.exists(final_path):
            print(f"\n  Layer {layer}: already trained, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"  LAYER {layer}")
        print(f"{'=' * 60}")

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        layer_t0 = time.time()

        # Pool activations
        print(f"  Pooling activations...")
        pooled = pool_activations(layer, K562_DIR, TS_DIR,
                                  K562_SUBSAMPLE, TS_SUBSAMPLE)
        n_samples = len(pooled)

        # Compute and save mean
        print(f"  Computing activation mean...")
        act_mean = center_activations(pooled)
        np.save(os.path.join(run_dir, "activation_mean.npy"), act_mean)
        print(f"    Mean norm: {np.linalg.norm(act_mean):.4f}")

        # Center the pooled data
        pooled -= act_mean[np.newaxis, :]

        # Train SAE
        print(f"  Training SAE ({EPOCHS} epochs, {n_samples:,} samples)...")
        sae = TopKSAE(d_model=HIDDEN_DIM, n_features=n_features, k=K_VAL)
        trainer = SAETrainer(sae, lr=LR, device=device)

        epoch_losses = []
        for epoch in range(EPOCHS):
            print(f"\n    --- Epoch {epoch + 1}/{EPOCHS} ---")
            t0 = time.time()
            avg_loss = trainer.train_epoch(
                pooled, batch_size=BATCH_SIZE, log_every=LOG_EVERY,
                checkpoint_dir=os.path.join(run_dir, "checkpoints"),
                checkpoint_every=CHECKPOINT_EVERY,
            )
            epoch_losses.append(avg_loss)
            print(f"    Epoch {epoch+1}: avg_loss={avg_loss:.6f}, time={time.time()-t0:.1f}s")

        # Evaluate
        print(f"  Final evaluation...")
        sae.eval()
        eval_size = min(100000, n_samples)
        eval_idx = np.random.choice(n_samples, eval_size, replace=False)
        eval_batch = torch.tensor(pooled[eval_idx], dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, h_sparse, topk_indices = sae(eval_batch)
            mse = torch.nn.functional.mse_loss(eval_batch, x_hat).item()
            total_var = eval_batch.var(dim=0).sum().item()
            resid_var = (eval_batch - x_hat).var(dim=0).sum().item()
            var_explained = 1.0 - resid_var / max(total_var, 1e-10)
            stats = sae.get_feature_stats(h_sparse)
            dead_features = int((stats['activation_freq'] == 0).sum())
            alive_features = n_features - dead_features

        print(f"    MSE: {mse:.6f}")
        print(f"    Variance explained: {var_explained:.4f}")
        print(f"    L0 norm: {stats['l0_norm']:.1f}")
        print(f"    Alive: {alive_features}/{n_features}")

        # Save
        sae.save(final_path)
        trainer.save_log(os.path.join(run_dir, "training_log.json"))

        layer_time = time.time() - layer_t0
        results = {
            'layer': layer,
            'n_features': n_features,
            'k': K_VAL,
            'n_samples': n_samples,
            'k562_subsample': K562_SUBSAMPLE,
            'ts_subsample': TS_SUBSAMPLE,
            'final_mse': mse,
            'variance_explained': var_explained,
            'l0_norm': stats['l0_norm'],
            'alive_features': alive_features,
            'dead_features': dead_features,
            'epoch_losses': epoch_losses,
            'training_time_s': layer_time,
        }
        all_results[f"layer{layer:02d}"] = results

        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=_json_default)

        print(f"  Layer {layer} done: {layer_time/60:.1f} min")
        del pooled

    # Save summary
    summary_path = os.path.join(OUT_BASE, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    for layer_key, res in all_results.items():
        print(f"  {layer_key}: VarExpl={res['variance_explained']:.4f}, "
              f"Alive={res['alive_features']}/{res['n_features']}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
