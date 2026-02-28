#!/usr/bin/env python3
"""
Step 2: Train TopK Sparse Autoencoder on extracted Geneformer activations.

Trains SAE at specified layer(s). Supports multiple expansion factors and
sparsity levels. Includes checkpointing, logging, and dead feature tracking.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 02_train_sae.py [--layer 0] [--expansion 4] [--k 32] [--epochs 5]
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

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_model import TopKSAE, SAETrainer

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
DATA_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map/experiments/phase1_k562")
OUT_BASE = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map/experiments/phase1_k562/sae_models")

HIDDEN_DIM = 1152
DEFAULT_LAYER = 0
DEFAULT_EXPANSION = 4  # n_features = expansion * d_model
DEFAULT_K = 32
DEFAULT_EPOCHS = 5
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 4096

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


def load_activations(layer, data_dir=DATA_DIR):
    """Load memory-mapped activations for a layer."""
    act_path = os.path.join(data_dir, f"layer_{layer:02d}_activations.npy")
    if not os.path.exists(act_path):
        raise FileNotFoundError(f"Activations not found: {act_path}")

    print(f"  Loading activations from {act_path}")
    activations = np.load(act_path, mmap_mode='r')
    print(f"  Shape: {activations.shape}, dtype: {activations.dtype}")
    return activations


def center_activations(activations):
    """Compute and subtract mean (streaming for large arrays)."""
    print("  Computing activation mean...")
    n = len(activations)
    # Compute mean in chunks to handle memmap
    chunk_size = 50000
    running_sum = np.zeros(activations.shape[1], dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        running_sum += activations[start:end].astype(np.float64).sum(axis=0)
    mean = (running_sum / n).astype(np.float32)
    print(f"  Mean norm: {np.linalg.norm(mean):.4f}")
    return mean


def main():
    parser = argparse.ArgumentParser(description="Train TopK SAE on Geneformer activations")
    parser.add_argument('--layer', type=int, default=DEFAULT_LAYER,
                        help=f'Layer to train on (default: {DEFAULT_LAYER})')
    parser.add_argument('--expansion', type=int, default=DEFAULT_EXPANSION,
                        help=f'Dictionary expansion factor (default: {DEFAULT_EXPANSION})')
    parser.add_argument('--k', type=int, default=DEFAULT_K,
                        help=f'TopK sparsity (default: {DEFAULT_K})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'Learning rate (default: {DEFAULT_LR})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--subsample', type=int, default=0,
                        help='Subsample N vectors (0=use all; e.g. 1000000 for 1M)')
    args = parser.parse_args()

    n_features = args.expansion * HIDDEN_DIM
    run_name = f"layer{args.layer:02d}_x{args.expansion}_k{args.k}"
    run_dir = os.path.join(OUT_BASE, run_name)

    print("=" * 70)
    print(f"SUBPROJECT 42: TRAIN TopK SAE")
    print(f"  Layer: {args.layer}")
    print(f"  Features: {n_features} ({args.expansion}x overcomplete)")
    print(f"  TopK: {args.k}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {run_dir}")
    print("=" * 70)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    total_t0 = time.time()

    # ============================================================
    # STEP 1: Load activations
    # ============================================================
    print("\nSTEP 1: Loading activations...")
    activations_mmap = load_activations(args.layer)
    n_total = len(activations_mmap)
    print(f"  Total samples: {n_total:,}")

    # Subsample if requested
    if args.subsample > 0 and args.subsample < n_total:
        print(f"  Subsampling {args.subsample:,} vectors...")
        np.random.seed(42)
        sub_idx = np.random.choice(n_total, args.subsample, replace=False)
        sub_idx.sort()
        activations = activations_mmap[sub_idx].copy()  # Load into RAM
        n_samples = len(activations)
        print(f"  Subsampled: {n_samples:,} vectors ({n_samples * HIDDEN_DIM * 4 / 1e9:.1f} GB)")
    else:
        activations = activations_mmap
        n_samples = n_total

    # Compute centering (on full data for accuracy, or subsample for speed)
    act_mean = center_activations(activations)
    np.save(os.path.join(run_dir, "activation_mean.npy"), act_mean)

    # ============================================================
    # STEP 2: Set up device and model
    # ============================================================
    print("\nSTEP 2: Setting up model...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("  Using CPU")

    # Check for existing checkpoint
    final_model_path = os.path.join(run_dir, "sae_final.pt")
    if os.path.exists(final_model_path):
        print(f"  Final model already exists: {final_model_path}")
        print("  Delete it to retrain. Exiting.")
        return

    sae = TopKSAE(d_model=HIDDEN_DIM, n_features=n_features, k=args.k)
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"  Encoder: ({HIDDEN_DIM}, {n_features})")
    print(f"  Decoder: ({n_features}, {HIDDEN_DIM})")

    trainer = SAETrainer(sae, lr=args.lr, device=device)

    # ============================================================
    # STEP 3: Train
    # ============================================================
    print(f"\nSTEP 3: Training ({args.epochs} epochs)...")

    # Pre-load into RAM if small enough (< 8 GB)
    est_size_gb = n_samples * HIDDEN_DIM * 4 / 1e9
    if est_size_gb < 8.0:
        print(f"  Pre-loading activations into RAM ({est_size_gb:.1f} GB)...")
        t0 = time.time()
        act_array = np.array(activations, dtype=np.float32)
        # Center
        act_array -= act_mean[np.newaxis, :]
        print(f"  Loaded and centered in {time.time()-t0:.1f}s")
    else:
        print(f"  Activations too large for RAM ({est_size_gb:.1f} GB), using streaming...")
        # For streaming, we'll center on-the-fly in train_epoch
        act_array = activations  # memmap
        # Note: centering needs to happen in the training loop
        # We'll modify the approach for large data

    epoch_losses = []
    for epoch in range(args.epochs):
        print(f"\n  --- Epoch {epoch + 1}/{args.epochs} ---")
        t0 = time.time()

        avg_loss = trainer.train_epoch(
            act_array,
            batch_size=args.batch_size,
            log_every=LOG_EVERY,
            checkpoint_dir=os.path.join(run_dir, "checkpoints"),
            checkpoint_every=CHECKPOINT_EVERY,
        )

        epoch_time = time.time() - t0
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1} complete: avg_loss={avg_loss:.6f}, time={epoch_time:.1f}s")

    # ============================================================
    # STEP 4: Final evaluation
    # ============================================================
    print("\nSTEP 4: Final evaluation...")

    sae.eval()
    # Evaluate on a large sample
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

    print(f"  MSE: {mse:.6f}")
    print(f"  Variance explained: {var_explained:.4f}")
    print(f"  L0 norm: {stats['l0_norm']:.1f}")
    print(f"  Alive features: {alive_features}/{n_features} ({100*alive_features/n_features:.1f}%)")
    print(f"  Dead features: {dead_features}/{n_features} ({100*dead_features/n_features:.1f}%)")

    # ============================================================
    # STEP 5: Save final model and results
    # ============================================================
    print("\nSTEP 5: Saving results...")

    sae.save(final_model_path)
    print(f"  Model saved: {final_model_path}")

    trainer.save_log(os.path.join(run_dir, "training_log.json"))

    # Save comprehensive results
    results = {
        'config': {
            'layer': args.layer,
            'd_model': HIDDEN_DIM,
            'n_features': n_features,
            'expansion': args.expansion,
            'k': args.k,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
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
        'feature_stats': {
            'mean_activation': stats['mean_activation'].tolist(),
            'activation_freq': stats['activation_freq'].tolist(),
        },
        'timing': {
            'total_time_s': time.time() - total_t0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    }

    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Variance explained: {var_explained:.4f}")
    print(f"  Alive features: {alive_features}/{n_features}")
    print(f"  Model: {final_model_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
