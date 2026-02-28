#!/usr/bin/env python3
"""
Step 3: Analyze SAE features — max-activating genes, activation patterns,
decoder directions, and comparison to SVD axes.

For each SAE feature produces:
  - Top-20 max-activating genes (by mean activation across all positions of that gene)
  - Activation frequency and distribution
  - Decoder weight vector analysis
  - Cosine alignment with top SVD axes

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 03_analyze_features.py --layer 0 --expansion 4 --k 32
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_model import TopKSAE

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")

HIDDEN_DIM = 1152
TOP_GENES_PER_FEATURE = 20
TOP_EXAMPLES_PER_FEATURE = 100
N_SVD_AXES = 50  # Compare to top-50 SVD axes


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--k', type=int, default=32)
    args = parser.parse_args()

    run_name = f"layer{args.layer:02d}_x{args.expansion}_k{args.k}"
    run_dir = os.path.join(SAE_BASE, run_name)
    model_path = os.path.join(run_dir, "sae_final.pt")

    print("=" * 70)
    print(f"SUBPROJECT 42: ANALYZE SAE FEATURES")
    print(f"  Run: {run_name}")
    print("=" * 70)

    total_t0 = time.time()

    # ============================================================
    # STEP 1: Load model and data
    # ============================================================
    print("\nSTEP 1: Loading model and data...")

    sae = TopKSAE.load(model_path, device='cpu')
    sae.eval()
    n_features = sae.n_features
    print(f"  SAE: {sae.d_model}d -> {n_features} features, k={sae.k}")

    # Load activations
    act_path = os.path.join(DATA_DIR, f"layer_{args.layer:02d}_activations.npy")
    gene_id_path = os.path.join(DATA_DIR, f"layer_{args.layer:02d}_gene_ids.npy")
    cell_id_path = os.path.join(DATA_DIR, f"layer_{args.layer:02d}_cell_ids.npy")

    activations = np.load(act_path, mmap_mode='r')
    gene_ids = np.load(gene_id_path, mmap_mode='r')
    cell_ids = np.load(cell_id_path, mmap_mode='r')
    n_samples = len(activations)
    print(f"  Activations: {activations.shape}")

    # Load centering mean
    mean_path = os.path.join(run_dir, "activation_mean.npy")
    act_mean = np.load(mean_path)

    # Load gene name mapping
    with open(os.path.join(DATA_DIR, "token_id_to_gene_name.json")) as f:
        token_to_gene = json.load(f)
    # Keys are strings in JSON
    token_to_gene = {int(k): v for k, v in token_to_gene.items()}

    print(f"  Gene name mapping: {len(token_to_gene)} tokens")

    # ============================================================
    # STEP 2: Compute per-feature activations (streaming)
    # ============================================================
    print("\nSTEP 2: Computing feature activations...")
    t0 = time.time()

    # Accumulate per-feature statistics in streaming fashion
    chunk_size = 10000
    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_sq_sum = np.zeros(n_features, dtype=np.float64)
    feature_count = np.zeros(n_features, dtype=np.int64)  # times feature fired
    feature_n = 0

    # Per-gene-per-feature accumulator: for each feature, track total activation per gene
    # Use dict of dicts to handle sparse data
    gene_feature_sum = defaultdict(lambda: defaultdict(float))
    gene_feature_count = defaultdict(lambda: defaultdict(int))

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch_act = activations[start:end].astype(np.float32) - act_mean[np.newaxis, :]
        batch_gene_ids = gene_ids[start:end]

        batch_tensor = torch.tensor(batch_act, dtype=torch.float32)
        with torch.no_grad():
            h_sparse, topk_indices = sae.encode(batch_tensor)
        h_np = h_sparse.numpy()

        # Global feature stats
        active_mask = h_np > 0
        feature_sum += h_np.sum(axis=0)
        feature_sq_sum += (h_np ** 2).sum(axis=0)
        feature_count += active_mask.sum(axis=0)
        feature_n += len(batch_act)

        # Per-gene-per-feature stats (only for active features to stay sparse)
        for i in range(len(batch_act)):
            gid = int(batch_gene_ids[i])
            active_features = np.where(h_np[i] > 0)[0]
            for fi in active_features:
                gene_feature_sum[fi][gid] += h_np[i, fi]
                gene_feature_count[fi][gid] += 1

        if (end) % 100000 < chunk_size:
            print(f"    Processed {end:,}/{n_samples:,}")

    print(f"  Feature activation computation: {time.time()-t0:.1f}s")

    # ============================================================
    # STEP 3: Build feature catalog
    # ============================================================
    print("\nSTEP 3: Building feature catalog...")

    feature_mean_act = feature_sum / max(feature_n, 1)
    feature_freq = feature_count / max(feature_n, 1)
    dead_mask = feature_count == 0

    print(f"  Alive features: {(~dead_mask).sum()}/{n_features}")
    print(f"  Dead features: {dead_mask.sum()}/{n_features}")
    print(f"  Mean activation freq: {feature_freq[~dead_mask].mean():.4f}")

    catalog = []
    for fi in range(n_features):
        entry = {
            'feature_idx': fi,
            'is_dead': bool(dead_mask[fi]),
            'activation_freq': float(feature_freq[fi]),
            'mean_activation': float(feature_mean_act[fi]),
            'fire_count': int(feature_count[fi]),
        }

        if not dead_mask[fi]:
            # Top genes for this feature
            gene_acts = gene_feature_sum[fi]
            gene_counts = gene_feature_count[fi]

            # Mean activation per gene (for genes where this feature fires)
            gene_mean_acts = {}
            for gid in gene_acts:
                if gene_counts[gid] > 0:
                    gene_mean_acts[gid] = gene_acts[gid] / gene_counts[gid]

            # Sort by mean activation
            sorted_genes = sorted(gene_mean_acts.items(), key=lambda x: -x[1])
            top_genes = []
            for gid, act in sorted_genes[:TOP_GENES_PER_FEATURE]:
                gene_name = token_to_gene.get(gid, f"token_{gid}")
                top_genes.append({
                    'gene_id': gid,
                    'gene_name': gene_name,
                    'mean_activation': float(act),
                    'fire_count': int(gene_counts[gid]),
                })
            entry['top_genes'] = top_genes
            entry['n_unique_genes'] = len(gene_acts)
        else:
            entry['top_genes'] = []
            entry['n_unique_genes'] = 0

        catalog.append(entry)

    # ============================================================
    # STEP 4: Decoder direction analysis
    # ============================================================
    print("\nSTEP 4: Analyzing decoder directions...")

    decoder_weights = sae.W_dec.weight.data.numpy().T  # (n_features, d_model) — transposed from (d_model, n_features)

    # Compute pairwise cosine similarity between top alive features
    alive_indices = np.where(~dead_mask)[0]
    n_alive = len(alive_indices)

    if n_alive > 0:
        alive_weights = decoder_weights[alive_indices]
        norms = np.linalg.norm(alive_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1
        alive_normed = alive_weights / norms

        # Mean pairwise cosine (sample 1000 pairs if too many)
        if n_alive > 100:
            sample_size = min(1000, n_alive * (n_alive - 1) // 2)
            idx_a = np.random.randint(0, n_alive, sample_size)
            idx_b = np.random.randint(0, n_alive, sample_size)
            mask = idx_a != idx_b
            cosines = np.sum(alive_normed[idx_a[mask]] * alive_normed[idx_b[mask]], axis=1)
            mean_cos = float(np.mean(np.abs(cosines)))
        else:
            cos_matrix = alive_normed @ alive_normed.T
            np.fill_diagonal(cos_matrix, 0)
            mean_cos = float(np.mean(np.abs(cos_matrix)))

        print(f"  Mean |cosine| between features: {mean_cos:.4f}")
    else:
        mean_cos = 0.0

    # ============================================================
    # STEP 5: SVD comparison
    # ============================================================
    print("\nSTEP 5: Comparing to SVD axes...")

    # Compute SVD of the same activations
    print("  Computing SVD (sampling 50K positions)...")
    svd_sample_size = min(50000, n_samples)
    svd_idx = np.random.choice(n_samples, svd_sample_size, replace=False)
    svd_data = activations[svd_idx].astype(np.float32) - act_mean[np.newaxis, :]

    U, S, Vt = np.linalg.svd(svd_data, full_matrices=False)
    svd_axes = Vt[:N_SVD_AXES]  # (N_SVD_AXES, d_model)
    svd_variance = S ** 2 / (S ** 2).sum()
    print(f"  Top-{N_SVD_AXES} SVD axes explain {svd_variance[:N_SVD_AXES].sum()*100:.1f}% variance")

    # For each alive feature, find max alignment with SVD axes
    if n_alive > 0:
        # (n_alive, d_model) @ (d_model, N_SVD_AXES) -> (n_alive, N_SVD_AXES)
        alignments = alive_normed @ svd_axes.T
        max_alignment = np.max(np.abs(alignments), axis=1)
        best_svd_axis = np.argmax(np.abs(alignments), axis=1)

        # Features well-aligned with SVD (|cos| > 0.5)
        svd_aligned = max_alignment > 0.5
        n_svd_aligned = svd_aligned.sum()

        # Features NOT aligned — potential superposition discoveries
        n_novel = n_alive - n_svd_aligned

        print(f"  Features aligned with SVD (|cos|>0.5): {n_svd_aligned}/{n_alive} ({100*n_svd_aligned/n_alive:.1f}%)")
        print(f"  Novel (non-SVD-aligned) features: {n_novel}/{n_alive} ({100*n_novel/n_alive:.1f}%)")

        # Add SVD alignment to catalog
        for i, fi in enumerate(alive_indices):
            catalog[fi]['max_svd_alignment'] = float(max_alignment[i])
            catalog[fi]['best_svd_axis'] = int(best_svd_axis[i])
            catalog[fi]['is_svd_aligned'] = bool(svd_aligned[i])
    else:
        n_svd_aligned = 0
        n_novel = 0

    # Save SVD info
    svd_info = {
        'n_svd_axes': N_SVD_AXES,
        'variance_explained_top50': float(svd_variance[:N_SVD_AXES].sum()),
        'singular_values': S[:N_SVD_AXES].tolist(),
        'variance_ratios': svd_variance[:N_SVD_AXES].tolist(),
    }
    np.save(os.path.join(run_dir, "svd_axes.npy"), svd_axes)

    # ============================================================
    # STEP 6: Save catalog
    # ============================================================
    print("\nSTEP 6: Saving feature catalog...")

    summary = {
        'run_name': run_name,
        'layer': args.layer,
        'n_features': n_features,
        'k': sae.k,
        'n_alive': int(n_alive),
        'n_dead': int(dead_mask.sum()),
        'mean_feature_cosine': mean_cos,
        'n_svd_aligned': int(n_svd_aligned),
        'n_novel': int(n_novel),
        'svd_info': svd_info,
    }

    output = {
        'summary': summary,
        'features': catalog,
    }

    out_path = os.path.join(run_dir, "feature_catalog.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Alive features: {n_alive}/{n_features}")
    print(f"  SVD-aligned: {n_svd_aligned}")
    print(f"  Novel (superposition): {n_novel}")
    print(f"  Catalog: {out_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
