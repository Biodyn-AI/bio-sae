#!/usr/bin/env python3
"""
Step 11: Cross-Layer Computational Graph.

Map how features at layer L predict features at layer L+1. Even though
features don't persist in decoder space, they may be functionally linked —
activating feature A at L may predict which features activate at L+1.

Method:
  For each adjacent layer pair, encode the same inputs through both SAEs,
  compute mutual information or correlation between feature activations.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 11_computational_graph.py [--layers 0,5,11,17]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")

EXPANSION = 4
K_VAL = 32
BATCH_SIZE = 8192
N_SAMPLE = 500000  # Subsample for efficiency
TOP_K_DEPS = 50     # Keep top-K dependencies per feature


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def process_layer_pair(layer_a, layer_b):
    """Compute feature dependencies from layer_a → layer_b."""
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from sae_model import TopKSAE

    out_dir = os.path.join(DATA_DIR, "computational_graph")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"deps_L{layer_a:02d}_to_L{layer_b:02d}.json")
    if os.path.exists(out_path):
        print(f"  Already done: {out_path}")
        return out_path

    print(f"\n{'=' * 60}")
    print(f"LAYER {layer_a} → LAYER {layer_b}: COMPUTATIONAL DEPENDENCIES")
    print(f"{'=' * 60}")

    # Load both SAEs
    t0 = time.time()
    run_a = f"layer{layer_a:02d}_x{EXPANSION}_k{K_VAL}"
    run_b = f"layer{layer_b:02d}_x{EXPANSION}_k{K_VAL}"
    dir_a = os.path.join(SAE_BASE, run_a)
    dir_b = os.path.join(SAE_BASE, run_b)

    sae_a = TopKSAE.load(os.path.join(dir_a, "sae_final.pt"), device='cpu')
    sae_b = TopKSAE.load(os.path.join(dir_b, "sae_final.pt"), device='cpu')
    sae_a.eval()
    sae_b.eval()

    mean_a = torch.tensor(np.load(os.path.join(dir_a, "activation_mean.npy")), dtype=torch.float32)
    mean_b = torch.tensor(np.load(os.path.join(dir_b, "activation_mean.npy")), dtype=torch.float32)

    n_features = sae_a.n_features
    print(f"  SAEs loaded ({time.time()-t0:.1f}s)")

    # Load activations for both layers
    act_path_a = os.path.join(DATA_DIR, f"layer_{layer_a:02d}_activations.npy")
    act_path_b = os.path.join(DATA_DIR, f"layer_{layer_b:02d}_activations.npy")
    activations_a = np.lib.format.open_memmap(act_path_a, mode='r')
    activations_b = np.lib.format.open_memmap(act_path_b, mode='r')
    n_total = activations_a.shape[0]
    assert activations_b.shape[0] == n_total

    # Subsample for efficiency
    rng = np.random.RandomState(42)
    n_use = min(N_SAMPLE, n_total)
    sample_idx = rng.choice(n_total, n_use, replace=False)
    sample_idx.sort()

    print(f"  Using {n_use:,} / {n_total:,} positions")

    # ============================================================
    # Encode both layers and compute co-occurrence statistics
    # ============================================================
    print("  Encoding and computing cross-layer dependencies...")
    t0 = time.time()

    # We compute: for each feature_a that fires, which features_b also fire?
    # Use conditional probability: P(b fires | a fires) vs P(b fires)
    # PMI: log2(P(a,b) / (P(a) * P(b)))

    # Accumulators
    count_a = np.zeros(n_features, dtype=np.int64)  # How often feature a fires
    count_b = np.zeros(n_features, dtype=np.int64)  # How often feature b fires

    # For cross-layer co-activation, use sparse accumulation
    # Since n_features=4608 and k=32, each input has at most 32*32=1024 cross pairs
    # With 500K inputs, dense 4608^2 matrix is 81 MB (int32) — fits in memory
    coact_ab = np.zeros((n_features, n_features), dtype=np.int32)

    n_processed = 0
    for start in range(0, n_use, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_use)
        idx = sample_idx[start:end]

        batch_a = torch.tensor(activations_a[idx], dtype=torch.float32) - mean_a
        batch_b = torch.tensor(activations_b[idx], dtype=torch.float32) - mean_b

        with torch.no_grad():
            _, topk_a = sae_a.encode(batch_a)  # (batch, k)
            _, topk_b = sae_b.encode(batch_b)  # (batch, k)

        topk_a_np = topk_a.numpy()
        topk_b_np = topk_b.numpy()

        # Count individual features
        for fi in topk_a_np.ravel():
            count_a[fi] += 1
        for fi in topk_b_np.ravel():
            count_b[fi] += 1

        # Count cross-layer co-activations
        for row_idx in range(topk_a_np.shape[0]):
            a_active = topk_a_np[row_idx]  # k features
            b_active = topk_b_np[row_idx]  # k features
            # All pairs (a_i, b_j)
            for ai in a_active:
                for bi in b_active:
                    coact_ab[ai, bi] += 1

        n_processed += (end - start)
        if n_processed % (BATCH_SIZE * 10) == 0:
            print(f"    {n_processed:>10,}/{n_use:,}")

    elapsed = time.time() - t0
    print(f"  Encoding done: {elapsed:.1f}s")

    # ============================================================
    # Compute cross-layer PMI
    # ============================================================
    print("  Computing cross-layer PMI...")
    t0 = time.time()

    # P(a) = count_a[a] / n_use
    # P(b) = count_b[b] / n_use
    # P(a,b) = coact_ab[a,b] / n_use
    # PMI(a,b) = log2(P(a,b) / (P(a) * P(b)))

    # For each feature at layer_a, find top dependencies at layer_b
    dependencies = []
    n_alive_a = int((count_a > 0).sum())
    n_alive_b = int((count_b > 0).sum())

    # Compute PMI for all pairs where coact > 0
    # Only consider features that fire at least 100 times
    MIN_COUNT = 100
    active_a = np.where(count_a >= MIN_COUNT)[0]
    active_b = np.where(count_b >= MIN_COUNT)[0]

    print(f"    Active features (count>{MIN_COUNT}): L{layer_a}={len(active_a)}, L{layer_b}={len(active_b)}")

    # For each active feature at layer A, find its top dependencies at layer B
    for a_idx in active_a:
        p_a = count_a[a_idx] / n_use

        # Get co-activation counts for this feature
        coact_row = coact_ab[a_idx, active_b]
        nonzero = coact_row > 0
        if not nonzero.any():
            continue

        b_indices = active_b[nonzero]
        coact_vals = coact_row[nonzero]

        # Compute PMI for each
        pmis = []
        for bi, count in zip(b_indices, coact_vals):
            p_b = count_b[bi] / n_use
            p_ab = count / n_use
            if p_a > 0 and p_b > 0:
                pmi = np.log2(p_ab / (p_a * p_b))
                pmis.append((int(bi), float(pmi), int(count)))

        # Keep top dependencies by PMI
        pmis.sort(key=lambda x: -x[1])
        top_deps = pmis[:TOP_K_DEPS]

        if top_deps:
            dependencies.append({
                'feature_a': int(a_idx),
                'count_a': int(count_a[a_idx]),
                'n_dependencies': len(pmis),
                'top_dependencies': [
                    {'feature_b': bi, 'pmi': pmi, 'coact_count': cnt}
                    for bi, pmi, cnt in top_deps
                ],
                'max_pmi': top_deps[0][1],
                'mean_pmi_top5': float(np.mean([p for _, p, _ in top_deps[:5]])),
            })

    dependencies.sort(key=lambda d: -d['max_pmi'])
    print(f"    Features with dependencies: {len(dependencies)} ({time.time()-t0:.1f}s)")

    del coact_ab

    # ============================================================
    # Load annotations to characterize information flow
    # ============================================================
    print("  Annotating information flow...")

    # Load annotations for both layers
    for ldir, layer_num in [(dir_a, layer_a), (dir_b, layer_b)]:
        ann_path = os.path.join(ldir, "feature_annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                ann = json.load(f).get('feature_annotations', {})
            # Add labels to dependencies
            for dep in dependencies:
                fi_a = dep['feature_a']
                if layer_num == layer_a:
                    anns_a = ann.get(str(fi_a), [])
                    label_a = "unlabeled"
                    for a in anns_a:
                        if a['ontology'] in ('GO_BP', 'Reactome', 'KEGG'):
                            label_a = a['term']
                            break
                    dep['label_a'] = label_a

                for td in dep['top_dependencies']:
                    fi_b = td['feature_b']
                    if layer_num == layer_b:
                        anns_b = ann.get(str(fi_b), [])
                        label_b = "unlabeled"
                        for a in anns_b:
                            if a['ontology'] in ('GO_BP', 'Reactome', 'KEGG'):
                                label_b = a['term']
                                break
                        td['label_b'] = label_b

    # ============================================================
    # Compute summary: how much information flows across layers?
    # ============================================================
    all_max_pmis = [d['max_pmi'] for d in dependencies]
    all_mean_pmis = [d['mean_pmi_top5'] for d in dependencies]

    # Find "information highways" — features with very strong cross-layer dependencies
    highways = [d for d in dependencies if d['max_pmi'] > 3.0]

    print(f"\n  Summary:")
    print(f"    Features with cross-layer dependencies: {len(dependencies)}")
    print(f"    Max PMI distribution: mean={np.mean(all_max_pmis):.2f}, "
          f"median={np.median(all_max_pmis):.2f}, max={np.max(all_max_pmis):.2f}")
    print(f"    Information highways (PMI>3): {len(highways)}")

    if dependencies:
        print(f"\n  Top cross-layer dependencies:")
        for d in dependencies[:10]:
            top_dep = d['top_dependencies'][0]
            label_a = d.get('label_a', '?')[:30]
            label_b = top_dep.get('label_b', '?')[:30]
            print(f"    L{layer_a}:F{d['feature_a']} ({label_a}) → "
                  f"L{layer_b}:F{top_dep['feature_b']} ({label_b}) "
                  f"PMI={top_dep['pmi']:.2f}")

    # ============================================================
    # Save
    # ============================================================
    output = {
        'layer_a': layer_a,
        'layer_b': layer_b,
        'config': {
            'n_sample': n_use,
            'min_count': MIN_COUNT,
            'top_k_deps': TOP_K_DEPS,
        },
        'summary': {
            'n_alive_a': n_alive_a,
            'n_alive_b': n_alive_b,
            'n_features_with_deps': len(dependencies),
            'mean_max_pmi': float(np.mean(all_max_pmis)) if all_max_pmis else 0,
            'median_max_pmi': float(np.median(all_max_pmis)) if all_max_pmis else 0,
            'n_highways': len(highways),
        },
        'dependencies': dependencies[:500],  # Top 500 to save space
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='0,5,11,17',
                        help='Comma-separated layers for pairwise analysis')
    args = parser.parse_args()

    total_t0 = time.time()
    layers = [int(x) for x in args.layers.split(',')]

    print("=" * 70)
    print("STEP 11: CROSS-LAYER COMPUTATIONAL GRAPH")
    print(f"  Layers: {layers}")
    print("=" * 70)

    # Process adjacent pairs within the selected layers
    # Also process the full adjacent chain if selected layers are adjacent
    pairs = []
    for i in range(len(layers) - 1):
        pairs.append((layers[i], layers[i + 1]))

    print(f"  Layer pairs: {pairs}")

    for layer_a, layer_b in pairs:
        process_layer_pair(layer_a, layer_b)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
