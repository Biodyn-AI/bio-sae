#!/usr/bin/env python3
"""
Phase 3, Script 5: Compare K562-only vs multi-tissue SAE perturbation results.

Head-to-head comparison:
  1. Load Phase 2 Step 9 result (K562-only SAE, layer 11): 6.2% TF specificity
  2. Load Phase 3 results (multi-tissue SAE, layers 0/5/11/17)
  3. Compare per-TF: specificity K562-SAE vs multi-tissue-SAE
  4. Diagnostic: how many multi-tissue SAE features have TFs in their top genes?
  5. Quantify: does multi-tissue SAE have more TF-related features?

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 12e_compare_results.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")

# Phase 2 results (K562-only SAE)
K562_PERT_PATH = os.path.join(PROJ_DIR, "experiments/phase1_k562/perturbation_response/perturbation_response_layer11.json")

# Phase 3 results (multi-tissue SAE)
MT_PERT_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/perturbation_response")

# SAE model dirs for feature analysis
K562_SAE_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562/sae_models")
MT_SAE_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/sae_models")

OUT_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/comparison")
TARGET_LAYERS = [0, 5, 11, 17]
TRRUST_PATH = os.path.join(BASE, "biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_perturbation_results(path):
    """Load perturbation response results."""
    with open(path) as f:
        data = json.load(f)
    return data


def count_tf_features(catalog_path, trrust_tfs):
    """Count features whose top genes include known TFs."""
    if not os.path.exists(catalog_path):
        return 0, 0, 0

    with open(catalog_path) as f:
        catalog = json.load(f)

    n_alive = 0
    n_with_tf = 0
    n_tf_dominant = 0

    for feat in catalog['features']:
        if feat['is_dead']:
            continue
        n_alive += 1
        top_genes = [g['gene_name'].upper() for g in feat.get('top_genes', [])[:20]]
        tf_in_top = sum(1 for g in top_genes if g in trrust_tfs)
        if tf_in_top > 0:
            n_with_tf += 1
        if tf_in_top >= 3:
            n_tf_dominant += 1

    return n_alive, n_with_tf, n_tf_dominant


def compare_per_tf(k562_results, mt_results):
    """Compare specificity per TF between K562-only and multi-tissue SAE."""
    # Build per-TF lookup for K562
    k562_by_tf = {}
    for r in k562_results.get('target_results', []):
        if r['is_trrust_tf']:
            k562_by_tf[r['target_gene']] = r

    # Build per-TF lookup for multi-tissue
    mt_by_tf = {}
    for r in mt_results.get('target_results', []):
        if r['is_trrust_tf']:
            mt_by_tf[r['target_gene']] = r

    comparisons = []
    common_tfs = sorted(set(k562_by_tf.keys()) & set(mt_by_tf.keys()))

    for tf in common_tfs:
        k562_r = k562_by_tf[tf]
        mt_r = mt_by_tf[tf]
        comparisons.append({
            'tf': tf,
            'n_known_targets': k562_r['n_known_targets'],
            'k562_responding': k562_r['n_responding_features'],
            'k562_specific': k562_r['n_specific_responding'],
            'k562_max_effect': k562_r['max_abs_effect'],
            'mt_responding': mt_r['n_responding_features'],
            'mt_specific': mt_r['n_specific_responding'],
            'mt_max_effect': mt_r['max_abs_effect'],
            'improved': mt_r['n_specific_responding'] > k562_r['n_specific_responding'],
            'worsened': mt_r['n_specific_responding'] < k562_r['n_specific_responding'],
        })

    return comparisons, common_tfs


def main():
    total_t0 = time.time()

    print("=" * 70)
    print("PHASE 3: COMPARISON — K562-ONLY vs MULTI-TISSUE SAE")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load TRRUST TFs
    trrust_tfs = set()
    if os.path.exists(TRRUST_PATH):
        with open(TRRUST_PATH) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    trrust_tfs.add(parts[0].upper())

    # ============================================================
    # 1. Load Phase 2 (K562-only) results
    # ============================================================
    print("\n1. Loading K562-only SAE results (Phase 2, Layer 11)...")

    if not os.path.exists(K562_PERT_PATH):
        print(f"  ERROR: K562 results not found: {K562_PERT_PATH}")
        sys.exit(1)

    k562_results = load_perturbation_results(K562_PERT_PATH)
    k562_summary = k562_results['summary']
    print(f"  TFs tested: {k562_summary['n_trrust_tfs']}")
    print(f"  TFs with specific response: {k562_summary['n_tfs_with_specific_response']}")
    print(f"  TF specificity rate: {k562_summary['frac_tfs_specific']*100:.1f}%")

    # ============================================================
    # 2. Load Phase 3 (multi-tissue) results
    # ============================================================
    print("\n2. Loading multi-tissue SAE results (Phase 3)...")

    mt_results_by_layer = {}
    for layer in TARGET_LAYERS:
        mt_path = os.path.join(MT_PERT_DIR, f"perturbation_response_layer{layer:02d}.json")
        if not os.path.exists(mt_path):
            print(f"  Layer {layer}: not found, skipping")
            continue
        mt_results_by_layer[layer] = load_perturbation_results(mt_path)
        s = mt_results_by_layer[layer]['summary']
        print(f"  Layer {layer}: {s['n_tfs_with_specific_response']}/{s['n_trrust_tfs']} TFs specific "
              f"({s['frac_tfs_specific']*100:.1f}%)")

    # ============================================================
    # 3. Per-TF comparison (multi-tissue layer 11 vs K562 layer 11)
    # ============================================================
    print("\n3. Per-TF head-to-head comparison (Layer 11)...")

    if 11 in mt_results_by_layer:
        comparisons, common_tfs = compare_per_tf(k562_results, mt_results_by_layer[11])
        n_improved = sum(1 for c in comparisons if c['improved'])
        n_worsened = sum(1 for c in comparisons if c['worsened'])
        n_same = len(comparisons) - n_improved - n_worsened

        print(f"  Common TFs: {len(common_tfs)}")
        print(f"  Improved (more specific): {n_improved}")
        print(f"  Worsened (less specific): {n_worsened}")
        print(f"  Unchanged: {n_same}")

        # Show changed TFs
        for c in comparisons:
            if c['improved'] or c['worsened']:
                direction = "+" if c['improved'] else "-"
                print(f"    {direction} {c['tf']}: K562={c['k562_specific']} → MT={c['mt_specific']} "
                      f"(known targets: {c['n_known_targets']})")
    else:
        comparisons = []
        common_tfs = []

    # ============================================================
    # 4. TF feature diagnostics
    # ============================================================
    print("\n4. TF feature diagnostics...")

    # K562-only SAE (layer 11)
    k562_catalog_path = os.path.join(K562_SAE_DIR, "layer11_x4_k32", "feature_catalog.json")
    k562_alive, k562_tf_feats, k562_tf_dom = count_tf_features(k562_catalog_path, trrust_tfs)
    print(f"  K562 SAE (L11): {k562_tf_feats}/{k562_alive} features have TFs in top genes "
          f"({100*k562_tf_feats/max(k562_alive,1):.1f}%), "
          f"{k562_tf_dom} TF-dominant (≥3)")

    for layer in TARGET_LAYERS:
        mt_catalog_path = os.path.join(MT_SAE_DIR, f"layer{layer:02d}_x4_k32", "feature_catalog.json")
        mt_alive, mt_tf_feats, mt_tf_dom = count_tf_features(mt_catalog_path, trrust_tfs)
        if mt_alive > 0:
            print(f"  MT SAE (L{layer:02d}): {mt_tf_feats}/{mt_alive} features have TFs in top genes "
                  f"({100*mt_tf_feats/max(mt_alive,1):.1f}%), "
                  f"{mt_tf_dom} TF-dominant (≥3)")

    # ============================================================
    # 5. Best layer analysis
    # ============================================================
    print("\n5. Best layer for perturbation specificity...")

    best_layer = None
    best_rate = 0
    for layer, results in mt_results_by_layer.items():
        rate = results['summary']['frac_tfs_specific']
        if rate > best_rate:
            best_rate = rate
            best_layer = layer
        print(f"  Layer {layer}: {results['summary']['frac_tfs_specific']*100:.1f}% TF specificity")

    if best_layer is not None:
        print(f"  Best layer: {best_layer} ({best_rate*100:.1f}%)")

    # ============================================================
    # 6. Interpretation
    # ============================================================
    print("\n6. Interpretation...")

    k562_rate = k562_summary['frac_tfs_specific']
    if best_layer is not None:
        improvement = best_rate - k562_rate

        if best_rate > 0.20:
            verdict = "STRONG POSITIVE"
            interpretation = "Multi-tissue SAE unlocks regulatory features. Geneformer has regulatory knowledge."
        elif best_rate > 0.12:
            verdict = "MODERATE POSITIVE"
            interpretation = "Partial improvement. Some regulatory knowledge exists but is limited."
        elif best_rate > k562_rate + 0.01:
            verdict = "WEAK POSITIVE"
            interpretation = "Marginal improvement. Geneformer's regulatory knowledge is minimal."
        elif best_rate >= k562_rate - 0.01:
            verdict = "NULL"
            interpretation = "No change. Geneformer lacks regulatory knowledge regardless of SAE training data."
        else:
            verdict = "NEGATIVE"
            interpretation = "Multi-tissue noise degrades specificity."

        print(f"  K562-only SAE: {k562_rate*100:.1f}% ({k562_summary['n_tfs_with_specific_response']}/{k562_summary['n_trrust_tfs']} TFs)")
        print(f"  Multi-tissue SAE (best L{best_layer}): {best_rate*100:.1f}%")
        print(f"  Change: {improvement*100:+.1f}pp")
        print(f"  Verdict: {verdict}")
        print(f"  {interpretation}")

    # ============================================================
    # 7. Save comparison
    # ============================================================
    print("\n7. Saving comparison results...")

    output = {
        'k562_only': {
            'layer': 11,
            'n_tfs_tested': k562_summary['n_trrust_tfs'],
            'n_tfs_specific': k562_summary['n_tfs_with_specific_response'],
            'specificity_rate': k562_summary['frac_tfs_specific'],
        },
        'multi_tissue': {},
        'per_tf_comparison_layer11': comparisons if comparisons else [],
        'tf_feature_diagnostics': {
            'k562_layer11': {
                'alive': k562_alive,
                'with_tf_in_top_genes': k562_tf_feats,
                'tf_dominant': k562_tf_dom,
            },
        },
        'verdict': verdict if best_layer is not None else 'INCOMPLETE',
        'interpretation': interpretation if best_layer is not None else 'Results incomplete',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    for layer, results in mt_results_by_layer.items():
        output['multi_tissue'][f'layer{layer:02d}'] = {
            'n_tfs_tested': results['summary']['n_trrust_tfs'],
            'n_tfs_specific': results['summary']['n_tfs_with_specific_response'],
            'specificity_rate': results['summary']['frac_tfs_specific'],
        }

    out_path = os.path.join(OUT_DIR, "comparison_results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"  Saved: {out_path}")

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time:.1f}s")
    if best_layer is not None:
        print(f"  ANSWER: {verdict}")
        print(f"  {interpretation}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
