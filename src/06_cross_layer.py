#!/usr/bin/env python3
"""
Step 6: Cross-layer feature tracking across all 18 Geneformer layers.

Tracks how SAE features evolve across layers by comparing decoder weight vectors.
A feature at layer L "matches" a feature at layer L' if their decoder directions
have high cosine similarity — this means they represent the same direction in
activation space (and presumably the same biological concept).

Analyses:
1. Adjacent-layer feature persistence: what fraction of L features persist at L+1?
2. Feature lifecycle: when does each feature first appear and last appear?
3. Persistent vs transient features: features present in many layers vs few
4. Biological annotation stability: do persistent features have richer biology?
5. Feature trajectory clustering: group features by their layer-span patterns

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 06_cross_layer.py
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")

N_LAYERS = 18
EXPANSION = 4
K = 32
MATCH_THRESHOLD = 0.7  # Cosine similarity threshold for "same feature"
HIGH_MATCH_THRESHOLD = 0.9  # Strong match threshold


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_layer_data(layer):
    """Load decoder weights, alive mask, and annotations for a layer."""
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from sae_model import TopKSAE

    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K}"
    run_dir = os.path.join(SAE_BASE, run_name)

    if not os.path.exists(os.path.join(run_dir, "sae_final.pt")):
        return None

    sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
    decoder_weights = sae.W_dec.weight.data.numpy().T  # (n_features, d_model)

    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)

    is_alive = np.array([not f['is_dead'] for f in catalog['features']])

    # Annotation info
    ann_path = os.path.join(run_dir, "feature_annotations.json")
    annotated_set = set()
    ann_counts = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            annotations = json.load(f)
        for fid_str, anns in annotations['feature_annotations'].items():
            ann_counts[int(fid_str)] = len(anns)
            if len(anns) > 0:
                annotated_set.add(int(fid_str))

    # Top genes per feature
    top_genes = {}
    for f in catalog['features']:
        if not f['is_dead'] and f['top_genes']:
            top_genes[f['feature_idx']] = [g['gene_name'] for g in f['top_genes'][:5]]

    return {
        'decoder': decoder_weights,
        'is_alive': is_alive,
        'annotated_set': annotated_set,
        'ann_counts': ann_counts,
        'top_genes': top_genes,
        'n_alive': int(is_alive.sum()),
    }


def compute_matches(dec_a, alive_a, dec_b, alive_b, threshold=MATCH_THRESHOLD):
    """Find matching features between two layers.

    Returns: list of (feat_a, feat_b, cosine_sim) tuples for matches above threshold.
    """
    idx_a = np.where(alive_a)[0]
    idx_b = np.where(alive_b)[0]

    if len(idx_a) == 0 or len(idx_b) == 0:
        return []

    # Normalize decoder directions
    da = dec_a[idx_a]
    db = dec_b[idx_b]
    da_norm = da / (np.linalg.norm(da, axis=1, keepdims=True) + 1e-10)
    db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-10)

    # Compute absolute cosine similarity
    cos_matrix = np.abs(da_norm @ db_norm.T)  # (n_a, n_b)

    # For each feature in A, find best match in B
    matches = []
    best_b_for_a = np.argmax(cos_matrix, axis=1)
    best_cos_for_a = cos_matrix[np.arange(len(idx_a)), best_b_for_a]

    for i, (bi, cos) in enumerate(zip(best_b_for_a, best_cos_for_a)):
        if cos >= threshold:
            matches.append((int(idx_a[i]), int(idx_b[bi]), float(cos)))

    return matches


def main():
    total_t0 = time.time()

    print("=" * 70)
    print("STEP 6: CROSS-LAYER FEATURE TRACKING")
    print(f"  Match threshold: {MATCH_THRESHOLD}")
    print(f"  High match threshold: {HIGH_MATCH_THRESHOLD}")
    print("=" * 70)

    # ============================================================
    # Load all layer data
    # ============================================================
    print("\nLoading all 18 layers...")
    layers_data = {}
    for layer in range(N_LAYERS):
        data = load_layer_data(layer)
        if data is None:
            print(f"  Layer {layer}: MISSING")
            continue
        layers_data[layer] = data
        print(f"  Layer {layer}: {data['n_alive']} alive, "
              f"{len(data['annotated_set'])} annotated")

    available_layers = sorted(layers_data.keys())
    print(f"\n  {len(available_layers)} layers loaded")

    # ============================================================
    # Analysis 1: Adjacent-layer persistence
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS 1: ADJACENT-LAYER FEATURE PERSISTENCE")
    print(f"{'=' * 70}")

    adjacent_results = []
    for i in range(len(available_layers) - 1):
        la = available_layers[i]
        lb = available_layers[i + 1]
        da = layers_data[la]
        db = layers_data[lb]

        matches = compute_matches(
            da['decoder'], da['is_alive'],
            db['decoder'], db['is_alive'],
            threshold=MATCH_THRESHOLD
        )
        high_matches = [m for m in matches if m[2] >= HIGH_MATCH_THRESHOLD]

        persistence_rate = len(matches) / max(da['n_alive'], 1)
        high_rate = len(high_matches) / max(da['n_alive'], 1)
        mean_cos = np.mean([m[2] for m in matches]) if matches else 0

        res = {
            'from_layer': la,
            'to_layer': lb,
            'n_matches': len(matches),
            'n_high_matches': len(high_matches),
            'persistence_rate': float(persistence_rate),
            'high_persistence_rate': float(high_rate),
            'mean_match_cosine': float(mean_cos),
            'n_alive_from': da['n_alive'],
        }
        adjacent_results.append(res)

        print(f"  L{la:>2d}→L{lb:>2d}: {len(matches):>4d}/{da['n_alive']} "
              f"persist ({persistence_rate:.1%}), "
              f"{len(high_matches)} strong (>{HIGH_MATCH_THRESHOLD}), "
              f"mean cos={mean_cos:.3f}")

    # ============================================================
    # Analysis 2: Feature lifecycle (first/last appearance)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS 2: FEATURE LIFECYCLE TRACKING")
    print(f"{'=' * 70}")

    # Track features from layer 0 forward through all layers
    # For each alive feature at L0, find its "chain" through subsequent layers
    print("\n  Tracking features from L0 through all layers...")

    ref_layer = 0
    ref_data = layers_data[ref_layer]
    ref_alive = np.where(ref_data['is_alive'])[0]
    n_ref = len(ref_alive)

    # Track each L0 feature's persistence
    feature_chains = {}  # feat_idx -> {layer: (matched_feat, cosine)}
    for fi in ref_alive:
        feature_chains[int(fi)] = {0: (int(fi), 1.0)}

    # Chain through layers
    ref_dec = ref_data['decoder']
    for layer in available_layers[1:]:
        ld = layers_data[layer]
        matches = compute_matches(
            ref_dec, ref_data['is_alive'],
            ld['decoder'], ld['is_alive'],
            threshold=MATCH_THRESHOLD
        )
        for feat_a, feat_b, cos in matches:
            if feat_a in feature_chains:
                feature_chains[feat_a][layer] = (feat_b, cos)

    # Compute lifecycle stats
    lifespans = []
    for fi, chain in feature_chains.items():
        last_layer = max(chain.keys())
        span = last_layer - ref_layer + 1
        n_layers_present = len(chain)
        lifespans.append({
            'feature_idx': fi,
            'first_layer': ref_layer,
            'last_layer': last_layer,
            'span': span,
            'n_layers_present': n_layers_present,
            'layers': sorted(chain.keys()),
        })

    # Classify features
    transient = [f for f in lifespans if f['n_layers_present'] <= 3]
    moderate = [f for f in lifespans if 4 <= f['n_layers_present'] <= 10]
    persistent = [f for f in lifespans if f['n_layers_present'] > 10]

    print(f"  L0 alive features: {n_ref}")
    print(f"  Transient (1-3 layers): {len(transient)} ({len(transient)/n_ref*100:.1f}%)")
    print(f"  Moderate (4-10 layers): {len(moderate)} ({len(moderate)/n_ref*100:.1f}%)")
    print(f"  Persistent (11+ layers): {len(persistent)} ({len(persistent)/n_ref*100:.1f}%)")

    # Distribution histogram
    span_hist = defaultdict(int)
    for f in lifespans:
        span_hist[f['n_layers_present']] += 1
    print(f"\n  Span distribution:")
    for n_layers in sorted(span_hist.keys()):
        bar = '#' * (span_hist[n_layers] // 20 + 1)
        print(f"    {n_layers:>2d} layers: {span_hist[n_layers]:>5d} {bar}")

    # ============================================================
    # Analysis 3: Persistent features have richer biology?
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS 3: PERSISTENCE vs BIOLOGICAL ANNOTATION")
    print(f"{'=' * 70}")

    ann_set_l0 = ref_data['annotated_set']
    ann_counts_l0 = ref_data['ann_counts']

    for label, group in [("Transient", transient), ("Moderate", moderate), ("Persistent", persistent)]:
        if not group:
            continue
        indices = [f['feature_idx'] for f in group]
        n_ann = sum(1 for i in indices if i in ann_set_l0)
        rate = n_ann / len(indices)
        enrichments = [ann_counts_l0.get(i, 0) for i in indices]
        mean_enrich = np.mean(enrichments)
        print(f"  {label:>12s}: {len(group):>5d} features, "
              f"{n_ann}/{len(group)} annotated ({rate:.1%}), "
              f"mean enrichments={mean_enrich:.1f}")

    # ============================================================
    # Analysis 4: Feature emergence and disappearance patterns
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS 4: FEATURE EMERGENCE/DISAPPEARANCE BY LAYER")
    print(f"{'=' * 70}")

    # For each non-L0 layer, how many NEW features appear that don't match anything at L0?
    print("\n  Features NOT matching any L0 feature (novel at deeper layers):")

    for layer in available_layers[1:]:
        ld = layers_data[layer]
        # Check L0→L matches from L's perspective (reverse: which L features have no L0 match?)
        matches = compute_matches(
            ld['decoder'], ld['is_alive'],
            ref_data['decoder'], ref_data['is_alive'],
            threshold=MATCH_THRESHOLD
        )
        matched_l_feats = set(m[0] for m in matches)
        alive_l = set(np.where(ld['is_alive'])[0])
        unmatched = alive_l - matched_l_feats
        unmatch_rate = len(unmatched) / max(len(alive_l), 1)

        # Are unmatched features annotated?
        ann_unmatched = sum(1 for i in unmatched if i in ld['annotated_set'])
        ann_rate_unmatched = ann_unmatched / max(len(unmatched), 1)

        print(f"    L{layer:>2d}: {len(unmatched):>5d}/{len(alive_l)} novel ({unmatch_rate:.1%}), "
              f"{ann_unmatched} annotated ({ann_rate_unmatched:.1%})")

    # ============================================================
    # Analysis 5: Long-range tracking (L0→L17 direct)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS 5: LONG-RANGE FEATURE TRACKING (L0 → every layer)")
    print(f"{'=' * 70}")

    long_range = []
    for layer in available_layers:
        if layer == 0:
            long_range.append({
                'target_layer': 0,
                'n_matches': ref_data['n_alive'],
                'persistence_rate': 1.0,
                'mean_cosine': 1.0,
            })
            continue

        ld = layers_data[layer]
        matches = compute_matches(
            ref_data['decoder'], ref_data['is_alive'],
            ld['decoder'], ld['is_alive'],
            threshold=MATCH_THRESHOLD
        )
        rate = len(matches) / max(ref_data['n_alive'], 1)
        mean_cos = np.mean([m[2] for m in matches]) if matches else 0

        long_range.append({
            'target_layer': layer,
            'n_matches': len(matches),
            'persistence_rate': float(rate),
            'mean_cosine': float(mean_cos),
        })

        print(f"  L0→L{layer:>2d}: {len(matches):>5d}/{ref_data['n_alive']} "
              f"persist ({rate:.1%}), mean cos={mean_cos:.3f}")

    # ============================================================
    # Analysis 6: Top genes stability for persistent features
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS 6: TOP-GENE STABILITY FOR PERSISTENT FEATURES")
    print(f"{'=' * 70}")

    # For the most persistent features (11+ layers), check if top genes are stable
    if persistent:
        sample_persistent = persistent[:20]  # Top 20
        print(f"\n  Checking top-gene overlap for {len(sample_persistent)} persistent features:")

        jaccard_scores = []
        for feat_info in sample_persistent:
            fi = feat_info['feature_idx']
            genes_l0 = set(ref_data['top_genes'].get(fi, []))
            if not genes_l0:
                continue

            # Check genes at last matching layer
            last_layer = feat_info['last_layer']
            if last_layer in feature_chains[fi]:
                matched_feat, cos = feature_chains[fi][last_layer]
                genes_last = set(layers_data[last_layer]['top_genes'].get(matched_feat, []))
                if genes_last:
                    jaccard = len(genes_l0 & genes_last) / max(len(genes_l0 | genes_last), 1)
                    jaccard_scores.append(jaccard)

                    if len(jaccard_scores) <= 5:
                        print(f"    Feature {fi} (L0→L{last_layer}, cos={cos:.3f}): "
                              f"Jaccard={jaccard:.2f}")
                        print(f"      L0 genes: {sorted(genes_l0)[:5]}")
                        print(f"      L{last_layer} genes: {sorted(genes_last)[:5]}")

        if jaccard_scores:
            print(f"\n  Mean top-gene Jaccard (persistent features): {np.mean(jaccard_scores):.3f}")
            print(f"  Median: {np.median(jaccard_scores):.3f}")

    # ============================================================
    # Save results
    # ============================================================
    out_path = os.path.join(DATA_DIR, "cross_layer_tracking.json")
    output = {
        'config': {
            'match_threshold': MATCH_THRESHOLD,
            'high_match_threshold': HIGH_MATCH_THRESHOLD,
            'n_layers': len(available_layers),
            'reference_layer': ref_layer,
        },
        'adjacent_persistence': adjacent_results,
        'long_range_persistence': long_range,
        'lifecycle_summary': {
            'n_ref_features': n_ref,
            'n_transient': len(transient),
            'n_moderate': len(moderate),
            'n_persistent': len(persistent),
            'span_histogram': {str(k): v for k, v in sorted(span_hist.items())},
        },
        'persistence_vs_biology': {
            cat: {
                'n_features': len(group),
                'annotation_rate': sum(1 for f in group if f['feature_idx'] in ann_set_l0) / max(len(group), 1),
                'mean_enrichments': float(np.mean([ann_counts_l0.get(f['feature_idx'], 0) for f in group])) if group else 0,
            }
            for cat, group in [("transient", transient), ("moderate", moderate), ("persistent", persistent)]
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Saved: {out_path}")

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
