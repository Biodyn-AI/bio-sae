#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 11: Cross-layer computational graph.

For selected layer pairs, compute which upstream features are "information
highways" to downstream features using PMI on co-activation patterns.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 11_computational_graph.py
"""

import os
import sys
import json
import time
import numpy as np
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/computational_graph")

EXPANSION = 4
K_VAL = 32
CHUNK_SIZE = 10000
PMI_THRESHOLD = 3.0
MIN_COACTIVATION = 20

# Layer pairs: early→mid, mid→late, early→late
LAYER_PAIRS = [(0, 4), (4, 8), (8, 11)]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def process_layer_pair(layer_a, layer_b):
    """Compute cross-layer information highways between two layers."""
    import torch
    from sae_model import TopKSAE

    out_path = os.path.join(OUT_DIR, f"graph_L{layer_a:02d}_L{layer_b:02d}.json")
    if os.path.exists(out_path):
        print(f"\n  L{layer_a}→L{layer_b}: Already done.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Cross-layer Graph: L{layer_a} → L{layer_b}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Load SAEs
    run_a = f"layer{layer_a:02d}_x{EXPANSION}_k{K_VAL}"
    run_b = f"layer{layer_b:02d}_x{EXPANSION}_k{K_VAL}"

    sae_a = TopKSAE.load(os.path.join(SAE_BASE, run_a, "sae_final.pt"), device='cpu')
    sae_b = TopKSAE.load(os.path.join(SAE_BASE, run_b, "sae_final.pt"), device='cpu')
    sae_a.eval()
    sae_b.eval()

    mean_a = np.load(os.path.join(SAE_BASE, run_a, "activation_mean.npy"))
    mean_b = np.load(os.path.join(SAE_BASE, run_b, "activation_mean.npy"))
    mean_a_t = torch.tensor(mean_a, dtype=torch.float32)
    mean_b_t = torch.tensor(mean_b, dtype=torch.float32)

    n_features_a = sae_a.n_features
    n_features_b = sae_b.n_features

    # Load activations
    act_a = np.load(os.path.join(DATA_DIR, f"layer_{layer_a:02d}_activations.npy"), mmap_mode='r')
    act_b = np.load(os.path.join(DATA_DIR, f"layer_{layer_b:02d}_activations.npy"), mmap_mode='r')
    cell_ids_a = np.load(os.path.join(DATA_DIR, f"layer_{layer_a:02d}_cell_ids.npy"), mmap_mode='r')
    cell_ids_b = np.load(os.path.join(DATA_DIR, f"layer_{layer_b:02d}_cell_ids.npy"), mmap_mode='r')

    n_pos = min(len(act_a), len(act_b))
    print(f"  Positions: {n_pos:,}")

    # Accumulate cross-layer co-activation counts
    # Since features are 2048 each, cross matrix is 2048 × 2048 × 4 = 16 MB
    cross_count = np.zeros((n_features_a, n_features_b), dtype=np.int32)
    count_a = np.zeros(n_features_a, dtype=np.int64)
    count_b = np.zeros(n_features_b, dtype=np.int64)
    n_total = 0

    print("  Computing cross-layer co-activations...")
    for start in range(0, n_pos, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_pos)

        # Only process matching positions (same cell_id)
        cids_a = cell_ids_a[start:end]
        cids_b = cell_ids_b[start:end]
        match = cids_a == cids_b
        if not match.all():
            # Use only matching positions
            match_idx = np.where(match)[0]
            if len(match_idx) == 0:
                continue
            batch_a = torch.tensor(act_a[start:end][match_idx].astype(np.float32)) - mean_a_t
            batch_b = torch.tensor(act_b[start:end][match_idx].astype(np.float32)) - mean_b_t
        else:
            batch_a = torch.tensor(act_a[start:end].astype(np.float32)) - mean_a_t
            batch_b = torch.tensor(act_b[start:end].astype(np.float32)) - mean_b_t

        with torch.no_grad():
            _, topk_a = sae_a.encode(batch_a)
            _, topk_b = sae_b.encode(batch_b)

        idx_a = topk_a.numpy()  # (batch, k)
        idx_b = topk_b.numpy()

        np.add.at(count_a, idx_a.ravel(), 1)
        np.add.at(count_b, idx_b.ravel(), 1)

        # Cross co-activation: for each position, pair each active upstream
        # feature with each active downstream feature (vectorized)
        k = idx_a.shape[1]
        fa_all = np.repeat(idx_a, k, axis=1).ravel()  # each upstream repeated k times
        fb_all = np.tile(idx_b, (1, k)).ravel()        # downstream tiled k times
        np.add.at(cross_count, (fa_all, fb_all), 1)

        n_total += len(batch_a)
        if (end) % 500000 < CHUNK_SIZE:
            print(f"    {end:>10,}/{n_pos:,}")

    print(f"  Cross co-activation done ({n_total:,} matching positions)")

    # Compute PMI for cross-layer pairs
    print("  Computing PMI...")
    rows, cols = np.where(cross_count >= MIN_COACTIVATION)

    pmi_edges = []
    for idx in range(len(rows)):
        i, j = int(rows[idx]), int(cols[idx])
        count = int(cross_count[i, j])
        p_ij = count / n_total
        p_i = count_a[i] / n_total
        p_j = count_b[j] / n_total
        if p_i == 0 or p_j == 0:
            continue
        pmi = np.log2(p_ij / (p_i * p_j))
        if pmi >= PMI_THRESHOLD:
            pmi_edges.append({
                'upstream': int(i),
                'downstream': int(j),
                'pmi': float(pmi),
                'count': count,
            })

    pmi_edges.sort(key=lambda e: -e['pmi'])
    print(f"  PMI edges (>{PMI_THRESHOLD}): {len(pmi_edges)}")

    # Identify information highways (features that appear in many cross-layer edges)
    upstream_degree = {}
    downstream_degree = {}
    for e in pmi_edges:
        upstream_degree[e['upstream']] = upstream_degree.get(e['upstream'], 0) + 1
        downstream_degree[e['downstream']] = downstream_degree.get(e['downstream'], 0) + 1

    n_upstream_connected = len(upstream_degree)
    n_downstream_connected = len(downstream_degree)
    alive_a = int(np.sum(count_a > 0))
    alive_b = int(np.sum(count_b > 0))

    print(f"  Connected features: {n_upstream_connected}/{alive_a} upstream, "
          f"{n_downstream_connected}/{alive_b} downstream")

    # Save
    output = {
        'layer_upstream': layer_a,
        'layer_downstream': layer_b,
        'summary': {
            'n_positions': n_total,
            'n_pmi_edges': len(pmi_edges),
            'n_upstream_connected': n_upstream_connected,
            'n_downstream_connected': n_downstream_connected,
            'alive_upstream': alive_a,
            'alive_downstream': alive_b,
            'upstream_coverage': n_upstream_connected / max(alive_a, 1),
            'downstream_coverage': n_downstream_connected / max(alive_b, 1),
        },
        'top_edges': pmi_edges[:200],
        'upstream_hubs': sorted(upstream_degree.items(), key=lambda x: -x[1])[:50],
        'downstream_hubs': sorted(downstream_degree.items(), key=lambda x: -x[1])[:50],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    elapsed = time.time() - t0
    print(f"  L{layer_a}→L{layer_b} done in {elapsed/60:.1f} min")

    del act_a, act_b, sae_a, sae_b, cross_count
    import gc
    gc.collect()


def main():
    total_t0 = time.time()
    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 11: COMPUTATIONAL GRAPH")
    print(f"  Layer pairs: {LAYER_PAIRS}")
    print("=" * 70)

    for layer_a, layer_b in LAYER_PAIRS:
        process_layer_pair(layer_a, layer_b)

    print(f"\nAll done in {(time.time()-total_t0)/60:.1f} min")


if __name__ == '__main__':
    main()
