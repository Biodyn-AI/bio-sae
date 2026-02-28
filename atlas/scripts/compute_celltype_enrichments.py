#!/usr/bin/env python3
"""
Compute cell type and tissue enrichments per SAE feature using Tabula Sapiens activations.

Runs Phase 1 K562-trained SAE encoders on multi-tissue TS activations (3000 cells,
immune/kidney/lung) to compute per-feature cell type and tissue enrichments.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python atlas/scripts/compute_celltype_enrichments.py
"""

import sys
import os
import json
import time
import numpy as np
import torch
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Line-buffered stdout for live progress
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'src'))

from sae_model import TopKSAE

TS_DIR = os.path.join(BASE, 'experiments', 'phase3_multitissue', 'ts_activations')
SAE_DIR = os.path.join(BASE, 'experiments', 'phase1_k562', 'sae_models')
OUT_DIR = os.path.join(BASE, 'experiments', 'phase3_multitissue', 'celltype_enrichments')

LAYERS = [0, 5, 11, 17]
CHUNK_SIZE = 10_000
TOP_FRAC = 0.10
MIN_ACTIVE_CELLS = 10
MIN_CELLS_PER_TYPE = 3
MAX_ENRICHMENTS = 10
MAX_TOP_CELLS = 10


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_cell_metadata():
    """Load cell metadata from extraction_metadata.json."""
    meta_path = os.path.join(TS_DIR, 'extraction_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    cell_data = meta['cell_data']
    n_cells = meta['n_cells']

    cell_tissue = np.array([cd['tissue'] for cd in cell_data])
    cell_type = np.array([cd['cell_type'] for cd in cell_data])

    print(f"Loaded metadata: {n_cells} cells, {len(set(cell_tissue))} tissues, "
          f"{len(set(cell_type))} cell types")

    # Count cells per type and tissue
    unique_types, type_counts = np.unique(cell_type, return_counts=True)
    unique_tissues, tissue_counts = np.unique(cell_tissue, return_counts=True)

    print(f"  Tissues: {dict(zip(unique_tissues, tissue_counts))}")
    print(f"  Cell types with >= {MIN_CELLS_PER_TYPE} cells: "
          f"{(type_counts >= MIN_CELLS_PER_TYPE).sum()}/{len(unique_types)}")

    return cell_tissue, cell_type, n_cells, meta


def compute_cell_feature_matrix(layer, sae, act_mean):
    """Compute per-cell mean-when-active feature activation matrix.

    Returns:
        cell_feature_mean: (n_cells, n_features) mean activation when feature fires
        cell_feature_frac: (n_cells, n_features) fraction of positions where feature fires
    """
    layer_str = f'{layer:02d}'
    act_path = os.path.join(TS_DIR, f'layer_{layer_str}_activations.npy')
    cell_ids_path = os.path.join(TS_DIR, f'layer_{layer_str}_cell_ids.npy')

    # Memory-map activations (24 GB, zero RAM)
    activations = np.load(act_path, mmap_mode='r')
    cell_ids = np.load(cell_ids_path)
    n_positions = len(cell_ids)
    n_features = sae.n_features

    print(f"  Activations shape: {activations.shape}, cell_ids shape: {cell_ids.shape}")

    n_cells = int(cell_ids.max()) + 1

    # Accumulators
    cell_feat_sum = np.zeros((n_cells, n_features), dtype=np.float64)
    cell_feat_count = np.zeros((n_cells, n_features), dtype=np.int32)
    cell_n_positions = np.zeros(n_cells, dtype=np.int64)

    n_chunks = (n_positions + CHUNK_SIZE - 1) // CHUNK_SIZE
    t0 = time.time()

    for chunk_i in range(n_chunks):
        start = chunk_i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_positions)

        # Read chunk from mmap and center
        batch_act = activations[start:end].astype(np.float32)
        batch_act -= act_mean[np.newaxis, :]

        batch_cids = cell_ids[start:end]

        # SAE encode
        with torch.no_grad():
            x = torch.tensor(batch_act, dtype=torch.float32)
            h_sparse, topk_indices = sae.encode(x)
            h_np = h_sparse.numpy()  # (batch, n_features)

        # Accumulate per-cell
        for pos in range(end - start):
            cid = batch_cids[pos]
            cell_n_positions[cid] += 1
            active_mask = h_np[pos] > 0
            if active_mask.any():
                active_idx = np.where(active_mask)[0]
                cell_feat_sum[cid, active_idx] += h_np[pos, active_idx]
                cell_feat_count[cid, active_idx] += 1

        if (chunk_i + 1) % 50 == 0 or chunk_i == n_chunks - 1:
            elapsed = time.time() - t0
            pct = (chunk_i + 1) / n_chunks * 100
            speed = (end) / elapsed
            print(f"    Chunk {chunk_i+1}/{n_chunks} ({pct:.1f}%) | "
                  f"{speed:.0f} pos/s | {elapsed:.1f}s elapsed")

    # Compute mean-when-active
    cell_feature_mean = np.divide(
        cell_feat_sum, cell_feat_count,
        out=np.zeros_like(cell_feat_sum),
        where=cell_feat_count > 0
    )

    # Compute fraction of positions where feature fires
    cell_feature_frac = np.divide(
        cell_feat_count.astype(np.float64),
        cell_n_positions[:, np.newaxis],
        out=np.zeros((n_cells, n_features), dtype=np.float64),
        where=cell_n_positions[:, np.newaxis] > 0
    )

    total_time = time.time() - t0
    print(f"  Cell-feature matrix computed in {total_time:.1f}s")
    print(f"  Mean active features per cell-position: "
          f"{cell_feat_count.sum() / max(cell_n_positions.sum(), 1):.1f}")

    return cell_feature_mean, cell_feature_frac, cell_n_positions


def run_enrichment_tests(cell_feature_mean, cell_tissue, cell_type, n_features):
    """Run Fisher's exact tests for cell type and tissue enrichment per feature.

    Returns dict mapping feature_idx -> enrichment results.
    """
    n_cells = len(cell_tissue)
    unique_types = np.unique(cell_type)
    unique_tissues = np.unique(cell_tissue)

    # Filter cell types with enough cells
    type_counts = {t: (cell_type == t).sum() for t in unique_types}
    testable_types = [t for t, c in type_counts.items() if c >= MIN_CELLS_PER_TYPE]
    print(f"  Testing {len(testable_types)} cell types (>= {MIN_CELLS_PER_TYPE} cells)")

    n_top = max(1, int(n_cells * TOP_FRAC))
    results = {}
    n_tested = 0
    n_with_ct = 0
    n_with_ti = 0

    t0 = time.time()

    for fi in range(n_features):
        # Feature activation profile across cells
        feat_acts = cell_feature_mean[:, fi]
        n_active = (feat_acts > 0).sum()

        if n_active < MIN_ACTIVE_CELLS:
            continue

        n_tested += 1

        # Determine top-activating cells
        if n_active <= n_top:
            top_mask = feat_acts > 0
            actual_n_top = int(n_active)
        else:
            threshold = np.partition(feat_acts, -n_top)[-n_top]
            top_mask = feat_acts >= threshold
            actual_n_top = int(top_mask.sum())

        actual_n_other = n_cells - actual_n_top

        # --- Cell type enrichment ---
        ct_results = []
        ct_pvals = []

        for ct in testable_types:
            ct_mask = cell_type == ct
            a = int((top_mask & ct_mask).sum())      # top AND this type
            b = int((top_mask & ~ct_mask).sum())      # top AND NOT this type
            c = int((~top_mask & ct_mask).sum())      # not top AND this type
            d = int((~top_mask & ~ct_mask).sum())     # not top AND NOT this type

            if a == 0:
                ct_pvals.append(1.0)
                ct_results.append(None)
                continue

            _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            odds = (a * d) / max(b * c, 1e-10)

            ct_pvals.append(pval)
            ct_results.append({
                'ct': ct,
                'p_raw': pval,
                'or': round(odds, 2),
                'n_top': a,
                'n_total': int(ct_mask.sum()),
                'frac_top': round(a / max(actual_n_top, 1), 4),
            })

        # BH correction for cell types
        if ct_pvals:
            reject, padj, _, _ = multipletests(ct_pvals, alpha=0.05, method='fdr_bh')
            sig_ct = []
            for j, (res, p_adj, rej) in enumerate(zip(ct_results, padj, reject)):
                if rej and res is not None:
                    res['p_adj'] = float(p_adj)
                    sig_ct.append(res)
            sig_ct.sort(key=lambda x: x['p_adj'])
            sig_ct = sig_ct[:MAX_ENRICHMENTS]
        else:
            sig_ct = []

        # --- Tissue enrichment ---
        ti_results = []
        ti_pvals = []

        for ti in unique_tissues:
            ti_mask = cell_tissue == ti
            a = int((top_mask & ti_mask).sum())
            b = int((top_mask & ~ti_mask).sum())
            c = int((~top_mask & ti_mask).sum())
            d = int((~top_mask & ~ti_mask).sum())

            _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            odds = (a * d) / max(b * c, 1e-10)

            ti_pvals.append(pval)
            ti_results.append({
                't': ti,
                'p_raw': pval,
                'or': round(odds, 2),
                'n_top': a,
                'n_total': int(ti_mask.sum()),
            })

        # BH correction for tissues
        if ti_pvals:
            reject_ti, padj_ti, _, _ = multipletests(ti_pvals, alpha=0.05, method='fdr_bh')
            sig_ti = []
            for res, p_adj, rej in zip(ti_results, padj_ti, reject_ti):
                if rej:
                    res['p_adj'] = float(p_adj)
                    sig_ti.append(res)
            sig_ti.sort(key=lambda x: x['p_adj'])
        else:
            sig_ti = []

        # --- Top activating cells ---
        top_indices = np.argsort(feat_acts)[-MAX_TOP_CELLS:][::-1]
        top_cells = []
        for ci in top_indices:
            if feat_acts[ci] > 0:
                top_cells.append({
                    'idx': int(ci),
                    'ct': cell_type[ci],
                    't': cell_tissue[ci],
                    'a': round(float(feat_acts[ci]), 4),
                })

        if sig_ct:
            n_with_ct += 1
        if sig_ti:
            n_with_ti += 1

        results[fi] = {
            'n_active_cells': int(n_active),
            'cell_types': sig_ct,
            'tissues': sig_ti,
            'top_cells': top_cells,
        }

        if (n_tested % 500 == 0):
            elapsed = time.time() - t0
            print(f"    Tested {n_tested} features | {n_with_ct} with CT enrichment | "
                  f"{n_with_ti} with tissue enrichment | {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"  Enrichment tests done in {total_time:.1f}s")
    print(f"  {n_tested} features tested, {n_with_ct} with cell type enrichment, "
          f"{n_with_ti} with tissue enrichment")

    return results, n_tested


def process_layer(layer, cell_tissue, cell_type, n_cells, meta):
    """Process one layer: encode TS activations, run enrichment tests, save."""
    print(f"\n{'='*60}")
    print(f"Processing Layer {layer}")
    print(f"{'='*60}")

    # Load SAE
    layer_str = f'{layer:02d}'
    sae_path = os.path.join(SAE_DIR, f'layer{layer_str}_x4_k32', 'sae_final.pt')
    mean_path = os.path.join(SAE_DIR, f'layer{layer_str}_x4_k32', 'activation_mean.npy')

    print(f"  Loading SAE from {sae_path}")
    sae = TopKSAE.load(sae_path, device='cpu')
    sae.eval()

    act_mean = np.load(mean_path).astype(np.float32)
    print(f"  SAE: d_model={sae.d_model}, n_features={sae.n_features}, k={sae.k}")
    print(f"  Activation mean shape: {act_mean.shape}")

    # Compute cell-feature matrix
    cell_feature_mean, cell_feature_frac, cell_n_positions = \
        compute_cell_feature_matrix(layer, sae, act_mean)

    # Run enrichment tests
    enrichments, n_tested = run_enrichment_tests(
        cell_feature_mean, cell_tissue, cell_type, sae.n_features
    )

    # Build output
    unique_types, type_counts = np.unique(cell_type, return_counts=True)
    unique_tissues, tissue_counts = np.unique(cell_tissue, return_counts=True)

    output = {
        'layer': layer,
        'n_cells': n_cells,
        'n_features_tested': n_tested,
        'top_frac': TOP_FRAC,
        'min_cells_per_type': MIN_CELLS_PER_TYPE,
        'cell_type_counts': dict(zip(unique_types.tolist(), type_counts.tolist())),
        'tissue_counts': dict(zip(unique_tissues.tolist(), tissue_counts.tolist())),
        'features': enrichments,
    }

    # Save
    out_path = os.path.join(OUT_DIR, f'celltype_enrichment_layer{layer_str}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=1, default=_json_default)

    file_size = os.path.getsize(out_path) / 1024
    print(f"  Saved {out_path} ({file_size:.1f} KB)")

    return output


def main():
    print("Cell Type & Tissue Enrichment Computation")
    print("=" * 60)
    t_start = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata
    cell_tissue, cell_type, n_cells, meta = load_cell_metadata()

    for layer in LAYERS:
        # Check for existing output (resume capability)
        layer_str = f'{layer:02d}'
        out_path = os.path.join(OUT_DIR, f'celltype_enrichment_layer{layer_str}.json')
        if os.path.exists(out_path):
            print(f"\nLayer {layer}: output already exists at {out_path}, skipping.")
            print(f"  Delete the file to recompute.")
            continue

        process_layer(layer, cell_tissue, cell_type, n_cells, meta)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All done in {total:.1f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main()
