#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 2: Train TopK SAE on extracted scGPT activations.

Trains SAE for all 12 layers sequentially.
Uses same architecture and hyperparameters as Geneformer SAE (adapted for d_model=512).

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 02_train_sae.py [--layer 0] [--all]
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import torch

# Add parent src to path for TopKSAE import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from sae_model import TopKSAE, SAETrainer

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")
OUT_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")

D_MODEL = 512
EXPANSION = 4  # n_features = 4 * 512 = 2048
K = 32
EPOCHS = 5
LR = 3e-4
BATCH_SIZE = 4096
N_LAYERS = 12

LOG_EVERY = 500
CHECKPOINT_EVERY = 10000


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def center_activations(activations):
    """Compute activation mean (streaming for memmap arrays)."""
    n = len(activations)
    chunk_size = 50000
    running_sum = np.zeros(activations.shape[1], dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        running_sum += activations[start:end].astype(np.float64).sum(axis=0)
    mean = (running_sum / n).astype(np.float32)
    return mean


def train_layer(layer, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train SAE for a single layer."""
    n_features = EXPANSION * D_MODEL
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K}"
    run_dir = os.path.join(OUT_BASE, run_name)

    final_model_path = os.path.join(run_dir, "sae_final.pt")
    if os.path.exists(final_model_path):
        print(f"\n  Layer {layer}: Model already exists, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Training Layer {layer} SAE")
    print(f"  Features: {n_features} ({EXPANSION}x), TopK: {K}")
    print(f"  Output: {run_dir}")
    print(f"{'=' * 60}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    t0 = time.time()

    # Load activations
    act_path = os.path.join(DATA_DIR, f"layer_{layer:02d}_activations.npy")
    if not os.path.exists(act_path):
        print(f"  Activations not found: {act_path}, skipping.")
        return
    activations = np.load(act_path, mmap_mode='r')
    n_samples = len(activations)
    print(f"  Loaded: {n_samples:,} × {activations.shape[1]}")

    # Compute and save activation mean
    print("  Computing activation mean...")
    act_mean = center_activations(activations)
    np.save(os.path.join(run_dir, "activation_mean.npy"), act_mean)
    print(f"  Mean norm: {np.linalg.norm(act_mean):.4f}")

    # Load into RAM and center (should be small enough: ~1.7 GB per layer)
    est_gb = n_samples * D_MODEL * 4 / 1e9
    print(f"  Loading into RAM ({est_gb:.1f} GB)...")
    act_array = np.array(activations, dtype=np.float32)
    act_array -= act_mean[np.newaxis, :]

    # Set up device and model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sae = TopKSAE(d_model=D_MODEL, n_features=n_features, k=K)
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    trainer = SAETrainer(sae, lr=lr, device=device)

    # Train
    epoch_losses = []
    for epoch in range(epochs):
        print(f"\n  --- Epoch {epoch + 1}/{epochs} ---")
        et0 = time.time()
        avg_loss = trainer.train_epoch(
            act_array,
            batch_size=batch_size,
            log_every=LOG_EVERY,
            checkpoint_dir=os.path.join(run_dir, "checkpoints"),
            checkpoint_every=CHECKPOINT_EVERY,
        )
        epoch_time = time.time() - et0
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.6f}, time={epoch_time:.1f}s")

    # Final evaluation
    sae.eval()
    eval_size = min(100000, n_samples)
    eval_idx = np.random.choice(n_samples, eval_size, replace=False)
    eval_batch = torch.tensor(act_array[eval_idx], dtype=torch.float32).to(device)

    with torch.no_grad():
        x_hat, h_sparse, topk_indices = sae(eval_batch)
        mse = torch.nn.functional.mse_loss(eval_batch, x_hat).item()
        total_var = eval_batch.var(dim=0).sum().item()
        resid_var = (eval_batch - x_hat).var(dim=0).sum().item()
        var_explained = 1.0 - resid_var / max(total_var, 1e-10)
        stats = sae.get_feature_stats(h_sparse)
        dead_features = int((stats['activation_freq'] == 0).sum())
        alive_features = n_features - dead_features

    print(f"\n  Results for Layer {layer}:")
    print(f"    Variance explained: {var_explained:.4f}")
    print(f"    Alive features: {alive_features}/{n_features} ({100*alive_features/n_features:.1f}%)")
    print(f"    MSE: {mse:.6f}")

    # Save
    sae.save(final_model_path)
    trainer.save_log(os.path.join(run_dir, "training_log.json"))

    results = {
        'config': {
            'layer': layer,
            'd_model': D_MODEL,
            'n_features': n_features,
            'expansion': EXPANSION,
            'k': K,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'n_samples': n_samples,
        },
        'results': {
            'final_mse': mse,
            'variance_explained': var_explained,
            'l0_norm': stats['l0_norm'],
            'alive_features': alive_features,
            'dead_features': dead_features,
            'dead_feature_pct': 100 * dead_features / n_features,
            'epoch_losses': epoch_losses,
        },
        'timing': {
            'total_time_s': time.time() - t0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    }

    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

    train_time = time.time() - t0
    print(f"  Layer {layer} done in {train_time/60:.1f} min")

    # Free memory
    del act_array, eval_batch, sae, trainer
    if device.type == 'mps':
        torch.mps.empty_cache()
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Train TopK SAE on scGPT activations")
    parser.add_argument('--layer', type=int, default=-1,
                        help='Specific layer to train (-1 = all layers)')
    parser.add_argument('--all', action='store_true',
                        help='Train all 12 layers sequentially')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    total_t0 = time.time()

    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 2: TRAIN SAE")
    print(f"  d_model: {D_MODEL}, expansion: {EXPANSION}x, k: {K}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print("=" * 70)

    if args.all or args.layer == -1:
        layers = list(range(N_LAYERS))
        print(f"  Training all {N_LAYERS} layers")
    else:
        layers = [args.layer]
        print(f"  Training layer {args.layer}")

    for layer in layers:
        train_layer(layer, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
