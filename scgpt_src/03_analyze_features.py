#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 3: Analyze SAE features.

For each SAE feature: top-20 max-activating genes, activation frequency,
decoder direction analysis, SVD comparison.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 03_analyze_features.py --layer 0
    ~/anaconda3/envs/bio_mech_interp/bin/python 03_analyze_features.py --all
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from sae_model import TopKSAE

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")

D_MODEL = 512
N_LAYERS = 12
TOP_GENES_PER_FEATURE = 20
N_SVD_AXES = 50


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def analyze_layer(layer, expansion=4, k=32):
    """Analyze features for a single layer."""
    run_name = f"layer{layer:02d}_x{expansion}_k{k}"
    run_dir = os.path.join(SAE_BASE, run_name)
    model_path = os.path.join(run_dir, "sae_final.pt")

    catalog_path = os.path.join(run_dir, "feature_catalog.json")
    if os.path.exists(catalog_path):
        print(f"\n  Layer {layer}: Catalog already exists, skipping.")
        return

    if not os.path.exists(model_path):
        print(f"\n  Layer {layer}: SAE model not found, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Analyzing Layer {layer}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Load SAE
    sae = TopKSAE.load(model_path, device='cpu')
    sae.eval()
    n_features = sae.n_features
    print(f"  SAE: {sae.d_model}d -> {n_features} features, k={sae.k}")

    # Load activations
    activations = np.load(os.path.join(DATA_DIR, f"layer_{layer:02d}_activations.npy"), mmap_mode='r')
    gene_ids = np.load(os.path.join(DATA_DIR, f"layer_{layer:02d}_gene_ids.npy"), mmap_mode='r')
    n_samples = len(activations)
    print(f"  Activations: {activations.shape}")

    # Load centering mean
    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))

    # Load gene name mapping (scGPT vocab token_id -> gene_name)
    gene_map_path = os.path.join(DATA_DIR, "token_id_to_gene_name.json")
    with open(gene_map_path) as f:
        token_to_gene = {int(k): v for k, v in json.load(f).items()}
    print(f"  Gene name mapping: {len(token_to_gene)} tokens")

    # ============================================================
    # Compute per-feature activations (streaming)
    # ============================================================
    print("  Computing feature activations...")

    chunk_size = 10000
    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_count = np.zeros(n_features, dtype=np.int64)
    feature_n = 0
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

        active_mask = h_np > 0
        feature_sum += h_np.sum(axis=0)
        feature_count += active_mask.sum(axis=0)
        feature_n += len(batch_act)

        for i in range(len(batch_act)):
            gid = int(batch_gene_ids[i])
            active_features = np.where(h_np[i] > 0)[0]
            for fi in active_features:
                gene_feature_sum[fi][gid] += h_np[i, fi]
                gene_feature_count[fi][gid] += 1

        if (end) % 200000 < chunk_size:
            print(f"    Processed {end:,}/{n_samples:,}")

    # ============================================================
    # Build feature catalog
    # ============================================================
    print("  Building feature catalog...")

    feature_mean_act = feature_sum / max(feature_n, 1)
    feature_freq = feature_count / max(feature_n, 1)
    dead_mask = feature_count == 0

    n_alive = int((~dead_mask).sum())
    print(f"  Alive features: {n_alive}/{n_features}")

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
            gene_acts = gene_feature_sum[fi]
            gene_counts = gene_feature_count[fi]
            gene_mean_acts = {gid: gene_acts[gid] / gene_counts[gid]
                              for gid in gene_acts if gene_counts[gid] > 0}
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
    # Decoder direction + SVD comparison
    # ============================================================
    print("  Decoder direction analysis + SVD comparison...")

    decoder_weights = sae.W_dec.weight.data.numpy().T  # (n_features, d_model)
    alive_indices = np.where(~dead_mask)[0]

    mean_cos = 0.0
    n_svd_aligned = 0
    n_novel = 0

    if n_alive > 0:
        alive_weights = decoder_weights[alive_indices]
        norms = np.linalg.norm(alive_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1
        alive_normed = alive_weights / norms

        # Mean pairwise cosine
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

        # SVD comparison
        svd_sample_size = min(50000, n_samples)
        svd_idx = np.random.choice(n_samples, svd_sample_size, replace=False)
        svd_data = activations[svd_idx].astype(np.float32) - act_mean[np.newaxis, :]
        U, S, Vt = np.linalg.svd(svd_data, full_matrices=False)
        svd_axes = Vt[:N_SVD_AXES]

        alignments = alive_normed @ svd_axes.T
        max_alignment = np.max(np.abs(alignments), axis=1)
        best_svd_axis = np.argmax(np.abs(alignments), axis=1)
        svd_aligned = max_alignment > 0.5
        n_svd_aligned = int(svd_aligned.sum())
        n_novel = n_alive - n_svd_aligned

        for i, fi in enumerate(alive_indices):
            catalog[fi]['max_svd_alignment'] = float(max_alignment[i])
            catalog[fi]['best_svd_axis'] = int(best_svd_axis[i])
            catalog[fi]['is_svd_aligned'] = bool(svd_aligned[i])

        np.save(os.path.join(run_dir, "svd_axes.npy"), svd_axes)

    print(f"  Mean |cosine|: {mean_cos:.4f}")
    print(f"  SVD-aligned: {n_svd_aligned}, Novel: {n_novel}")

    # Save catalog
    summary = {
        'run_name': run_name,
        'layer': layer,
        'n_features': n_features,
        'k': sae.k,
        'n_alive': n_alive,
        'n_dead': int(dead_mask.sum()),
        'mean_feature_cosine': mean_cos,
        'n_svd_aligned': n_svd_aligned,
        'n_novel': n_novel,
    }

    output = {'summary': summary, 'features': catalog}
    with open(catalog_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    elapsed = time.time() - t0
    print(f"  Layer {layer} done in {elapsed/60:.1f} min")

    del activations, gene_ids, sae
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--k', type=int, default=32)
    args = parser.parse_args()

    total_t0 = time.time()
    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 3: ANALYZE FEATURES")
    print("=" * 70)

    if args.all or args.layer == -1:
        layers = list(range(N_LAYERS))
    else:
        layers = [args.layer]

    for layer in layers:
        analyze_layer(layer, args.expansion, args.k)

    print(f"\nAll done in {(time.time()-total_t0)/60:.1f} min")


if __name__ == '__main__':
    main()
