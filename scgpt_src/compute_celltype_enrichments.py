#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 6: Cell type & tissue enrichments.

For each layer, encode TS activations with SAE, compute per-cell feature
activation means, then Fisher's exact test for cell type/tissue enrichment.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python compute_celltype_enrichments.py --all
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
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/celltype_enrichments")

N_LAYERS = 12
EXPANSION = 4
K_VAL = 32
D_MODEL = 512
TOP_PCT = 0.10  # Top 10% of cells
MIN_ACTIVE_CELLS = 10
BH_ALPHA = 0.05
MIN_CELLTYPE_N = 3


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def compute_enrichments_for_layer(layer, cell_data, n_cells):
    """Compute cell type/tissue enrichments for one layer."""
    import torch
    from sae_model import TopKSAE

    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    model_path = os.path.join(run_dir, "sae_final.pt")

    out_path = os.path.join(OUT_DIR, f"celltype_enrichment_layer{layer:02d}.json")
    if os.path.exists(out_path):
        print(f"\n  Layer {layer}: Already computed, skipping.")
        return

    if not os.path.exists(model_path):
        print(f"\n  Layer {layer}: SAE not found, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Layer {layer}: Cell Type Enrichments")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Load SAE and activation mean
    sae = TopKSAE.load(model_path, device='cpu')
    sae.eval()
    n_features = sae.n_features

    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))

    # Load activations
    activations = np.load(os.path.join(DATA_DIR, f"layer_{layer:02d}_activations.npy"), mmap_mode='r')
    cell_ids = np.load(os.path.join(DATA_DIR, f"layer_{layer:02d}_cell_ids.npy"), mmap_mode='r')
    n_positions = len(activations)
    print(f"  Positions: {n_positions:,}, Features: {n_features}")

    # Accumulate per-cell feature activations
    cell_feature_sum = np.zeros((n_cells, n_features), dtype=np.float64)
    cell_feature_count = np.zeros((n_cells, n_features), dtype=np.int32)

    chunk_size = 10000
    for start in range(0, n_positions, chunk_size):
        end = min(start + chunk_size, n_positions)
        batch_act = activations[start:end].astype(np.float32) - act_mean[np.newaxis, :]
        batch_cids = cell_ids[start:end]

        batch_tensor = torch.tensor(batch_act, dtype=torch.float32)
        with torch.no_grad():
            h_sparse, _ = sae.encode(batch_tensor)
        h_np = h_sparse.numpy()

        active_mask = h_np > 0
        for i in range(len(batch_act)):
            ci = int(batch_cids[i])
            if ci < 0 or ci >= n_cells:
                continue
            active = active_mask[i]
            cell_feature_sum[ci, active] += h_np[i, active]
            cell_feature_count[ci, active] += 1

        if (end) % 500000 < chunk_size:
            print(f"    Encoded {end:,}/{n_positions:,}")

    # Compute mean-when-active
    cell_feature_mean = np.zeros_like(cell_feature_sum, dtype=np.float32)
    nonzero = cell_feature_count > 0
    cell_feature_mean[nonzero] = (cell_feature_sum[nonzero] / cell_feature_count[nonzero]).astype(np.float32)

    print(f"  Per-cell features computed in {time.time()-t0:.1f}s")

    # Get cell types and tissues
    cell_types = [cd['cell_type'] for cd in cell_data]
    tissues = [cd['tissue'] for cd in cell_data]
    unique_cell_types = sorted(set(cell_types))
    unique_tissues = sorted(set(tissues))

    # For each feature, get top-activating cells and test enrichments
    top_n = max(int(n_cells * TOP_PCT), 10)
    feature_results = {}

    alive_features = np.where(cell_feature_mean.max(axis=0) > 0)[0]
    print(f"  Testing {len(alive_features)} alive features...")

    for feat_i, fi in enumerate(alive_features):
        if feat_i % 500 == 0 and feat_i > 0:
            print(f"    Feature {feat_i}/{len(alive_features)}...")

        feature_acts = cell_feature_mean[:, fi]
        active_cells = np.where(feature_acts > 0)[0]
        if len(active_cells) < MIN_ACTIVE_CELLS:
            continue

        # Top-activating cells
        sorted_cells = np.argsort(-feature_acts)
        top_cells = sorted_cells[:top_n]
        is_top = np.zeros(n_cells, dtype=bool)
        is_top[top_cells] = True

        # Cell type enrichment (Fisher's exact)
        ct_results = []
        ct_pvals = []
        for ct in unique_cell_types:
            ct_mask = np.array([c == ct for c in cell_types])
            n_ct = ct_mask.sum()
            if n_ct < MIN_CELLTYPE_N:
                continue

            a = int((is_top & ct_mask).sum())
            b = int((is_top & ~ct_mask).sum())
            c = int((~is_top & ct_mask).sum())
            d = int((~is_top & ~ct_mask).sum())

            if a == 0:
                continue

            odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            ct_results.append({
                'cell_type': ct,
                'p_value': pval,
                'odds_ratio': float(odds_ratio),
                'n_in_top': a,
                'n_total': int(n_ct),
            })
            ct_pvals.append(pval)

        # Tissue enrichment (Fisher's exact)
        ti_results = []
        ti_pvals = []
        for ti in unique_tissues:
            ti_mask = np.array([t == ti for t in tissues])
            n_ti = ti_mask.sum()

            a = int((is_top & ti_mask).sum())
            b = int((is_top & ~ti_mask).sum())
            c = int((~is_top & ti_mask).sum())
            d = int((~is_top & ~ti_mask).sum())

            if a == 0:
                continue

            odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            ti_results.append({
                'tissue': ti,
                'p_value': pval,
                'odds_ratio': float(odds_ratio),
                'n_in_top': a,
                'n_total': int(n_ti),
            })
            ti_pvals.append(pval)

        # BH correction per feature
        if ct_pvals:
            reject, corrected, _, _ = multipletests(ct_pvals, alpha=BH_ALPHA, method='fdr_bh')
            for i, r in enumerate(ct_results):
                r['p_adjusted'] = float(corrected[i])
                r['significant'] = bool(reject[i])

        if ti_pvals:
            reject, corrected, _, _ = multipletests(ti_pvals, alpha=BH_ALPHA, method='fdr_bh')
            for i, r in enumerate(ti_results):
                r['p_adjusted'] = float(corrected[i])
                r['significant'] = bool(reject[i])

        # Top 10 cells
        top10_cells = []
        for ci in top_cells[:10]:
            top10_cells.append({
                'cell_idx': int(ci),
                'cell_type': cell_types[ci],
                'tissue': tissues[ci],
                'activation': float(feature_acts[ci]),
            })

        sig_ct = [r for r in ct_results if r.get('significant', False)]
        sig_ti = [r for r in ti_results if r.get('significant', False)]

        if sig_ct or sig_ti or top10_cells:
            feature_results[int(fi)] = {
                'cell_type_enrichments': sorted(sig_ct, key=lambda x: x['p_adjusted']),
                'tissue_enrichments': sorted(sig_ti, key=lambda x: x['p_adjusted']),
                'top_cells': top10_cells,
                'n_active_cells': int(len(active_cells)),
            }

    print(f"  Features with enrichments: {len(feature_results)}")

    # Save
    output = {
        'layer': layer,
        'n_cells': n_cells,
        'n_features': n_features,
        'n_features_with_enrichments': len(feature_results),
        'tissues': unique_tissues,
        'cell_types': unique_cell_types,
        'features': feature_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    elapsed = time.time() - t0
    print(f"  Layer {layer} done in {elapsed/60:.1f} min")

    del activations, cell_ids, sae, cell_feature_sum, cell_feature_count, cell_feature_mean
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    total_t0 = time.time()
    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 6: CELL TYPE ENRICHMENTS")
    print("=" * 70)

    # Load cell metadata
    meta_path = os.path.join(DATA_DIR, "extraction_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    cell_data = meta['cell_data']
    n_cells = len(cell_data)
    print(f"  Cells: {n_cells}")

    if args.all:
        layers = list(range(N_LAYERS))
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(N_LAYERS))

    for layer in layers:
        compute_enrichments_for_layer(layer, cell_data, n_cells)

    print(f"\nAll done in {(time.time()-total_t0)/60:.1f} min")


if __name__ == '__main__':
    main()
