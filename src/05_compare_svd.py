#!/usr/bin/env python3
"""
Step 5: Systematic SAE vs SVD comparison across all 18 layers.

For each layer:
1. Load SVD axes (top 50) and SAE decoder directions
2. Quantify alignment between SVD axes and best-matching SAE features
3. Compute biological annotation rates for SVD-aligned vs non-SVD-aligned features
4. Test whether SAE features capture more biology than SVD axes alone

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 05_compare_svd.py
"""

import os
import sys
import json
import time
import numpy as np
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")

N_LAYERS = 18
EXPANSION = 4
K = 32
SVD_ALIGN_THRESHOLD = 0.5


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def analyze_layer(layer):
    """Analyze SAE vs SVD for a single layer."""
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K}"
    run_dir = os.path.join(SAE_BASE, run_name)

    # Load SVD axes (50, 1152)
    svd_path = os.path.join(run_dir, "svd_axes.npy")
    if not os.path.exists(svd_path):
        return None
    svd_axes = np.load(svd_path)  # (50, 1152)
    n_svd = svd_axes.shape[0]

    # Load SAE decoder weights
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from sae_model import TopKSAE
    sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
    decoder_weights = sae.W_dec.weight.data.numpy().T  # (n_features, d_model)

    # Load feature catalog
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)

    # Load annotations
    with open(os.path.join(run_dir, "feature_annotations.json")) as f:
        annotations = json.load(f)

    features = catalog['features']
    n_features = len(features)

    # Build feature info arrays
    is_alive = np.array([not f['is_dead'] for f in features])
    is_svd_aligned = np.array([f.get('is_svd_aligned', False) for f in features])
    max_svd_align = np.array([f.get('max_svd_alignment', 0.0) for f in features])
    best_svd_axis = np.array([f.get('best_svd_axis', -1) for f in features])

    # Which features are annotated?
    annotated_set = set()
    for fid_str, anns in annotations['feature_annotations'].items():
        if len(anns) > 0:
            annotated_set.add(int(fid_str))
    is_annotated = np.array([i in annotated_set for i in range(n_features)])

    # ---- Analysis 1: SVD axis coverage ----
    # For each SVD axis, find best-matching SAE feature and its cosine sim
    alive_decoder = decoder_weights[is_alive]
    alive_indices = np.where(is_alive)[0]

    # Normalize
    svd_norm = svd_axes / (np.linalg.norm(svd_axes, axis=1, keepdims=True) + 1e-10)
    dec_norm = alive_decoder / (np.linalg.norm(alive_decoder, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity matrix: (n_svd, n_alive)
    cos_matrix = np.abs(svd_norm @ dec_norm.T)

    svd_coverage = []
    for si in range(n_svd):
        best_feat_local = np.argmax(cos_matrix[si])
        best_feat_global = int(alive_indices[best_feat_local])
        best_cos = float(cos_matrix[si, best_feat_local])

        # Top-3 matching features
        top3_local = np.argsort(cos_matrix[si])[-3:][::-1]
        top3 = [(int(alive_indices[j]), float(cos_matrix[si, j])) for j in top3_local]

        svd_coverage.append({
            'svd_axis': si,
            'best_sae_feature': best_feat_global,
            'best_cosine': best_cos,
            'is_covered': best_cos >= SVD_ALIGN_THRESHOLD,
            'top3_features': top3,
        })

    n_covered = sum(1 for s in svd_coverage if s['is_covered'])
    mean_best_cos = np.mean([s['best_cosine'] for s in svd_coverage])

    # ---- Analysis 2: Annotation rates ----
    alive_mask = is_alive
    n_alive = alive_mask.sum()

    svd_aligned_mask = is_alive & is_svd_aligned
    novel_mask = is_alive & ~is_svd_aligned

    n_svd_aligned = svd_aligned_mask.sum()
    n_novel = novel_mask.sum()

    # Annotation rates by group
    ann_svd = (is_annotated & svd_aligned_mask).sum()
    ann_novel = (is_annotated & novel_mask).sum()
    ann_all = (is_annotated & alive_mask).sum()

    rate_svd = ann_svd / max(n_svd_aligned, 1)
    rate_novel = ann_novel / max(n_novel, 1)
    rate_all = ann_all / max(n_alive, 1)

    # ---- Analysis 3: Annotation richness (number of enrichments per feature) ----
    ann_counts = {}
    for fid_str, anns in annotations['feature_annotations'].items():
        ann_counts[int(fid_str)] = len(anns)

    enrichments_svd = [ann_counts.get(i, 0) for i in range(n_features) if svd_aligned_mask[i]]
    enrichments_novel = [ann_counts.get(i, 0) for i in range(n_features) if novel_mask[i]]

    mean_enrich_svd = np.mean(enrichments_svd) if enrichments_svd else 0
    mean_enrich_novel = np.mean(enrichments_novel) if enrichments_novel else 0

    # ---- Analysis 4: Unique ontology terms captured ----
    terms_svd = set()
    terms_novel = set()
    for fid_str, anns in annotations['feature_annotations'].items():
        fid = int(fid_str)
        for a in anns:
            term_key = (a['ontology'], a['term'])
            if svd_aligned_mask[fid]:
                terms_svd.add(term_key)
            elif novel_mask[fid]:
                terms_novel.add(term_key)

    terms_only_novel = terms_novel - terms_svd  # terms found ONLY in non-SVD features

    # ---- Analysis 5: SVD variance explained vs SAE variance explained ----
    svd_varexpl = catalog['summary']['svd_info']['variance_explained_top50']

    # Load training results for SAE variance explained
    with open(os.path.join(run_dir, "training_log.json")) as f:
        train_log = json.load(f)
    sae_varexpl = train_log[-1]['var_explained'] if train_log else 0

    return {
        'layer': layer,
        'svd_coverage': {
            'n_svd_axes': n_svd,
            'n_covered_by_sae': n_covered,
            'coverage_rate': n_covered / n_svd,
            'mean_best_cosine': mean_best_cos,
            'per_axis': svd_coverage[:10],  # Top 10 only to save space
        },
        'feature_counts': {
            'n_alive': int(n_alive),
            'n_svd_aligned': int(n_svd_aligned),
            'n_novel': int(n_novel),
            'pct_novel': float(n_novel / max(n_alive, 1) * 100),
        },
        'annotation_rates': {
            'all_alive': float(rate_all),
            'svd_aligned': float(rate_svd),
            'novel_only': float(rate_novel),
            'n_annotated_svd': int(ann_svd),
            'n_annotated_novel': int(ann_novel),
        },
        'annotation_richness': {
            'mean_enrichments_svd': float(mean_enrich_svd),
            'mean_enrichments_novel': float(mean_enrich_novel),
        },
        'ontology_coverage': {
            'n_unique_terms_svd': len(terms_svd),
            'n_unique_terms_novel': len(terms_novel),
            'n_terms_only_novel': len(terms_only_novel),
            'pct_terms_novel_exclusive': float(len(terms_only_novel) / max(len(terms_novel | terms_svd), 1) * 100),
        },
        'variance_explained': {
            'svd_top50': float(svd_varexpl),
            'sae_4x_k32': float(sae_varexpl),
        },
    }


def main():
    total_t0 = time.time()

    print("=" * 70)
    print("STEP 5: SAE vs SVD COMPARISON — ALL 18 LAYERS")
    print("=" * 70)

    results = {}
    for layer in range(N_LAYERS):
        print(f"\n  Layer {layer}...", end=" ")
        t0 = time.time()
        res = analyze_layer(layer)
        if res is None:
            print("SKIPPED (missing data)")
            continue
        results[layer] = res
        fc = res['feature_counts']
        ar = res['annotation_rates']
        oc = res['ontology_coverage']
        print(f"SVD-aligned={fc['n_svd_aligned']}, "
              f"Novel={fc['n_novel']} ({fc['pct_novel']:.1f}%), "
              f"AnnRate novel={ar['novel_only']:.1%}, "
              f"Novel-only terms={oc['n_terms_only_novel']} "
              f"({time.time()-t0:.1f}s)")

    # ============================================================
    # Cross-layer summary table
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print("CROSS-LAYER SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Layer':>5} | {'SVD VarExpl':>11} | {'SAE VarExpl':>11} | "
          f"{'SVD-align':>9} | {'Novel':>5} | "
          f"{'AnnRate SVD':>11} | {'AnnRate Nov':>11} | "
          f"{'Unique Terms':>12} | {'Novel-Only':>10}")
    print("-" * 105)

    total_novel_annotated = 0
    total_novel = 0
    total_novel_terms = set()
    total_svd_terms = set()

    for layer in range(N_LAYERS):
        if layer not in results:
            continue
        r = results[layer]
        ve = r['variance_explained']
        fc = r['feature_counts']
        ar = r['annotation_rates']
        oc = r['ontology_coverage']

        print(f"{layer:>5} | {ve['svd_top50']:>10.1%} | {ve['sae_4x_k32']:>10.1%} | "
              f"{fc['n_svd_aligned']:>9} | {fc['n_novel']:>5} | "
              f"{ar['svd_aligned']:>10.1%} | {ar['novel_only']:>10.1%} | "
              f"{oc['n_unique_terms_novel']:>12} | {oc['n_terms_only_novel']:>10}")

        total_novel_annotated += ar['n_annotated_novel']
        total_novel += fc['n_novel']

    # Global statistics
    print(f"\n{'=' * 70}")
    print("AGGREGATE STATISTICS")
    print(f"{'=' * 70}")

    all_novel = sum(results[l]['feature_counts']['n_novel'] for l in results)
    all_svd = sum(results[l]['feature_counts']['n_svd_aligned'] for l in results)
    all_ann_novel = sum(results[l]['annotation_rates']['n_annotated_novel'] for l in results)
    all_ann_svd = sum(results[l]['annotation_rates']['n_annotated_svd'] for l in results)

    print(f"  Total SAE features across {len(results)} layers: {all_novel + all_svd}")
    print(f"  SVD-aligned: {all_svd} ({all_svd/(all_novel+all_svd)*100:.1f}%)")
    print(f"  Novel (superposition): {all_novel} ({all_novel/(all_novel+all_svd)*100:.1f}%)")
    print()
    print(f"  Annotated SVD-aligned: {all_ann_svd}/{all_svd} ({all_ann_svd/max(all_svd,1)*100:.1f}%)")
    print(f"  Annotated novel: {all_ann_novel}/{all_novel} ({all_ann_novel/max(all_novel,1)*100:.1f}%)")
    print()

    # Mean annotation richness
    mean_enrich_svd = np.mean([results[l]['annotation_richness']['mean_enrichments_svd'] for l in results])
    mean_enrich_novel = np.mean([results[l]['annotation_richness']['mean_enrichments_novel'] for l in results])
    print(f"  Mean enrichments per SVD-aligned feature: {mean_enrich_svd:.1f}")
    print(f"  Mean enrichments per novel feature: {mean_enrich_novel:.1f}")
    print()

    # Novel-only terms
    total_novel_only = sum(results[l]['ontology_coverage']['n_terms_only_novel'] for l in results)
    mean_novel_pct = np.mean([results[l]['ontology_coverage']['pct_terms_novel_exclusive'] for l in results])
    print(f"  Mean ontology terms found ONLY in novel features: {mean_novel_pct:.1f}% of all terms")

    # ============================================================
    # Save
    # ============================================================
    out_path = os.path.join(DATA_DIR, "svd_vs_sae_comparison.json")
    output = {
        'per_layer': {str(k): v for k, v in results.items()},
        'aggregate': {
            'total_features': all_novel + all_svd,
            'total_svd_aligned': all_svd,
            'total_novel': all_novel,
            'pct_novel': all_novel / (all_novel + all_svd) * 100,
            'annotated_svd': all_ann_svd,
            'annotated_novel': all_ann_novel,
            'annotation_rate_svd': all_ann_svd / max(all_svd, 1),
            'annotation_rate_novel': all_ann_novel / max(all_novel, 1),
            'mean_enrichments_svd': mean_enrich_svd,
            'mean_enrichments_novel': mean_enrich_novel,
            'mean_novel_exclusive_terms_pct': mean_novel_pct,
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
