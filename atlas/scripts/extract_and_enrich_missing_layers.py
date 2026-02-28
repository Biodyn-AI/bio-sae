#!/usr/bin/env python3
"""
Extract Tabula Sapiens activations for the 14 missing layers (1-4, 6-10, 12-16)
and compute cell type / tissue enrichments on-the-fly.

Reuses the exact same 3000 cells from Phase 3 extraction (extraction_metadata.json).
Processes cells through Geneformer, captures hidden states at all 14 missing layers,
runs SAE encoders immediately, and accumulates per-cell feature activations in memory
(no 24 GB per-layer disk writes needed).

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python atlas/scripts/extract_and_enrich_missing_layers.py
"""

import sys
import os
import json
import time
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import h5py
import torch
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ============================================================
# Configuration
# ============================================================
BASE_BIO = "/Volumes/Crucial X6/MacBook/biomechinterp"
PAPER_DIR = os.path.join(BASE_BIO, "biodyn-nmi-paper")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
PROJ_DIR = os.path.join(BASE_BIO, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")

sys.path.insert(0, os.path.join(PROJ_DIR, 'src'))
from sae_model import TopKSAE

TS_DIR = os.path.join(PROJ_DIR, 'experiments', 'phase3_multitissue', 'ts_activations')
SAE_DIR = os.path.join(PROJ_DIR, 'experiments', 'phase1_k562', 'sae_models')
OUT_DIR = os.path.join(PROJ_DIR, 'experiments', 'phase3_multitissue', 'celltype_enrichments')

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

HIDDEN_DIM = 1152
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

# Layers already processed (have activation files)
EXISTING_LAYERS = {0, 5, 11, 17}
# Missing layers to extract and process
MISSING_LAYERS = sorted(set(range(18)) - EXISTING_LAYERS)

# Enrichment params (same as compute_celltype_enrichments.py)
TOP_FRAC = 0.10
MIN_ACTIVE_CELLS = 10
MIN_CELLS_PER_TYPE = 3
MAX_ENRICHMENTS = 10
MAX_TOP_CELLS = 10

TISSUES = {
    'immune': {
        'path': os.path.join(BASE_BIO, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"),
    },
    'kidney': {
        'path': os.path.join(BASE_BIO, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"),
    },
    'lung': {
        'path': os.path.join(BASE_BIO, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"),
    },
}

CHECKPOINT_FILE = os.path.join(OUT_DIR, "missing_layers_checkpoint.json")


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


def load_categorical_column(h5group, col_name):
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col['categories'][:]
        codes = col['codes'][:]
        if categories.dtype.kind in ('O', 'S'):
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    else:
        data = col[:]
        if data.dtype.kind in ('O', 'S'):
            return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
        return data


def load_sparse_row(f_group, row_idx, n_cols):
    indptr = f_group['indptr']
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    indices = f_group['indices'][start:end]
    data = f_group['data'][start:end]
    row = np.zeros(n_cols, dtype=np.float32)
    row[indices] = data
    return row


def tokenize_cell(expression_vector, var_indices, token_ids, medians, max_len=2048):
    expr = expression_vector[var_indices]
    nonzero = expr > 0
    if nonzero.sum() == 0:
        return None
    expr_nz = expr[nonzero]
    tokens_nz = token_ids[nonzero]
    medians_nz = medians[nonzero]
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = expr_nz / medians_nz
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0)
    rank_order = np.argsort(-normalized)
    ranked_tokens = tokens_nz[rank_order][:max_len - 2]
    return np.concatenate([[2], ranked_tokens, [3]]).astype(np.int64)


def run_enrichment_tests(cell_feature_mean, cell_tissue, cell_type, n_features):
    """Run Fisher's exact tests for cell type and tissue enrichment per feature."""
    n_cells = len(cell_tissue)
    unique_types = np.unique(cell_type)
    unique_tissues = np.unique(cell_tissue)

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
        feat_acts = cell_feature_mean[:, fi]
        n_active = (feat_acts > 0).sum()

        if n_active < MIN_ACTIVE_CELLS:
            continue

        n_tested += 1

        if n_active <= n_top:
            top_mask = feat_acts > 0
            actual_n_top = int(n_active)
        else:
            threshold = np.partition(feat_acts, -n_top)[-n_top]
            top_mask = feat_acts >= threshold
            actual_n_top = int(top_mask.sum())

        # --- Cell type enrichment ---
        ct_results = []
        ct_pvals = []

        for ct in testable_types:
            ct_mask = cell_type == ct
            a = int((top_mask & ct_mask).sum())
            b = int((top_mask & ~ct_mask).sum())
            c = int((~top_mask & ct_mask).sum())
            d = int((~top_mask & ~ct_mask).sum())

            if a == 0:
                ct_pvals.append(1.0)
                ct_results.append(None)
                continue

            _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            odds = (a * d) / max(b * c, 1e-10)

            ct_pvals.append(pval)
            ct_results.append({
                'ct': ct, 'p_raw': pval, 'or': round(odds, 2),
                'n_top': a, 'n_total': int(ct_mask.sum()),
                'frac_top': round(a / max(actual_n_top, 1), 4),
            })

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
                't': ti, 'p_raw': pval, 'or': round(odds, 2),
                'n_top': a, 'n_total': int(ti_mask.sum()),
            })

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
                    'idx': int(ci), 'ct': cell_type[ci],
                    't': cell_tissue[ci], 'a': round(float(feat_acts[ci]), 4),
                })

        if sig_ct:
            n_with_ct += 1
        if sig_ti:
            n_with_ti += 1

        results[fi] = {
            'n_active_cells': int(n_active),
            'cell_types': sig_ct, 'tissues': sig_ti, 'top_cells': top_cells,
        }

        if n_tested % 500 == 0:
            elapsed = time.time() - t0
            print(f"    Tested {n_tested} features | {n_with_ct} with CT enrichment | "
                  f"{n_with_ti} with tissue enrichment | {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"  Enrichment tests done in {total_time:.1f}s")
    print(f"  {n_tested} features tested, {n_with_ct} with cell type enrichment, "
          f"{n_with_ti} with tissue enrichment")

    return results, n_tested


def save_enrichment(layer, enrichments, n_tested, cell_tissue, cell_type, n_cells):
    """Save enrichment results for one layer."""
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

    layer_str = f'{layer:02d}'
    out_path = os.path.join(OUT_DIR, f'celltype_enrichment_layer{layer_str}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=1, default=_json_default)

    file_size = os.path.getsize(out_path) / 1024
    print(f"  Saved {out_path} ({file_size:.1f} KB)")


def main():
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    # Determine which layers still need processing
    layers_to_process = []
    for layer in MISSING_LAYERS:
        out_path = os.path.join(OUT_DIR, f'celltype_enrichment_layer{layer:02d}.json')
        if os.path.exists(out_path):
            print(f"Layer {layer}: enrichment already exists, skipping.")
        else:
            layers_to_process.append(layer)

    if not layers_to_process:
        print("All layers already processed!")
        return

    print("=" * 70)
    print("EXTRACT & ENRICH MISSING LAYERS")
    print(f"Layers to process: {layers_to_process}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load checkpoint
    start_cell = 0
    completed_layers = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        start_cell = ckpt.get('cells_completed', 0)
        completed_layers = set(ckpt.get('completed_layers', []))
        print(f"Resuming from checkpoint: {start_cell} cells, layers done: {sorted(completed_layers)}")

    # Remove already-completed layers
    layers_to_process = [l for l in layers_to_process if l not in completed_layers]
    if not layers_to_process:
        print("All target layers completed per checkpoint!")
        return

    print(f"Remaining layers: {layers_to_process}")

    # ============================================================
    # Load cell metadata (same cells as Phase 3 extraction)
    # ============================================================
    print("\nLoading cell metadata...")
    meta_path = os.path.join(TS_DIR, 'extraction_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    cell_data = meta['cell_data']
    n_cells = meta['n_cells']
    cell_tissue = np.array([cd['tissue'] for cd in cell_data])
    cell_type = np.array([cd['cell_type'] for cd in cell_data])
    print(f"  {n_cells} cells, {len(set(cell_tissue))} tissues, {len(set(cell_type))} cell types")

    # ============================================================
    # Load tokenization dicts
    # ============================================================
    print("\nLoading Geneformer tokenization dicts...")
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)

    # ============================================================
    # Build gene mappings per tissue
    # ============================================================
    print("\nBuilding gene mappings per tissue...")
    tissue_gene_maps = {}
    for tissue_name, tissue_info in TISSUES.items():
        h5_path = tissue_info['path']
        with h5py.File(h5_path, 'r') as f:
            var_index = f['var']['_index'][:]
            n_genes = len(var_index)

        mapped_var_indices = []
        mapped_token_ids_list = []
        mapped_medians_list = []

        for i in range(n_genes):
            ensembl_id = var_index[i].decode() if isinstance(var_index[i], bytes) else var_index[i]
            if ensembl_id in token_dict:
                mapped_var_indices.append(i)
                mapped_token_ids_list.append(token_dict[ensembl_id])
                mapped_medians_list.append(gene_median_dict.get(ensembl_id, 1.0))

        tissue_gene_maps[tissue_name] = {
            'mapped_var_indices': np.array(mapped_var_indices),
            'mapped_token_ids': np.array(mapped_token_ids_list),
            'mapped_medians': np.array(mapped_medians_list),
            'n_genes_total': n_genes,
        }
        print(f"  {tissue_name}: {len(mapped_var_indices)}/{n_genes} genes mapped")

    # ============================================================
    # Tokenize all cells (must reproduce exact same tokenization)
    # ============================================================
    print("\nTokenizing cells...")
    t0 = time.time()

    # Build mapping: cell_data index -> (tissue, h5_path, cell_idx)
    # The original script builds all_cell_data in tissue order: immune, kidney, lung
    all_cell_info = []
    for tissue_name in ['immune', 'kidney', 'lung']:
        tissue_cells = [cd for cd in cell_data if cd['tissue'] == tissue_name]
        for cd in tissue_cells:
            all_cell_info.append({
                'tissue': cd['tissue'],
                'cell_idx': cd['cell_idx'],
                'h5_path': TISSUES[cd['tissue']]['path'],
            })

    all_tokens = []
    for ci, cell_info in enumerate(all_cell_info):
        tissue = cell_info['tissue']
        h5_path = cell_info['h5_path']
        cell_idx = cell_info['cell_idx']
        gmap = tissue_gene_maps[tissue]

        with h5py.File(h5_path, 'r') as f:
            expr = load_sparse_row(f['X'], cell_idx, gmap['n_genes_total'])

        row_sum = expr.sum()
        if row_sum > 0:
            expr = np.log1p(expr / row_sum * 1e4)

        tokens = tokenize_cell(
            expr, gmap['mapped_var_indices'],
            gmap['mapped_token_ids'], gmap['mapped_medians'],
            MAX_SEQ_LEN
        )

        if tokens is not None:
            all_tokens.append(tokens)
        else:
            all_tokens.append(None)

        if (ci + 1) % 500 == 0:
            print(f"    Tokenized {ci+1}/{len(all_cell_info)} cells...")

    valid_cells = sum(1 for t in all_tokens if t is not None)
    print(f"  Valid cells: {valid_cells}/{len(all_cell_info)}")
    print(f"  Tokenization time: {time.time()-t0:.1f}s")

    # ============================================================
    # Load SAE models for all target layers
    # ============================================================
    print(f"\nLoading {len(layers_to_process)} SAE models...")
    saes = {}
    act_means = {}
    for layer in layers_to_process:
        layer_str = f'{layer:02d}'
        sae_path = os.path.join(SAE_DIR, f'layer{layer_str}_x4_k32', 'sae_final.pt')
        mean_path = os.path.join(SAE_DIR, f'layer{layer_str}_x4_k32', 'activation_mean.npy')
        saes[layer] = TopKSAE.load(sae_path, device='cpu')
        saes[layer].eval()
        act_means[layer] = np.load(mean_path).astype(np.float32)
        print(f"  L{layer}: d={saes[layer].d_model}, n_features={saes[layer].n_features}, k={saes[layer].k}")

    n_features = saes[layers_to_process[0]].n_features  # 4608

    # ============================================================
    # Initialize accumulators for all layers
    # ============================================================
    print("\nInitializing per-cell feature accumulators...")
    accumulators = {}
    for layer in layers_to_process:
        accumulators[layer] = {
            'feat_sum': np.zeros((n_cells, n_features), dtype=np.float64),
            'feat_count': np.zeros((n_cells, n_features), dtype=np.int32),
            'n_positions': np.zeros(n_cells, dtype=np.int64),
        }

    est_mb = n_cells * n_features * (8 + 4) * len(layers_to_process) / 1e6
    print(f"  Accumulator memory: ~{est_mb:.0f} MB for {len(layers_to_process)} layers")

    # ============================================================
    # Load Geneformer model
    # ============================================================
    print("\nLoading Geneformer V2-316M...")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("  Using CPU")

    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_hidden_states=True,
        output_attentions=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ============================================================
    # Extract activations and encode with SAEs
    # ============================================================
    print(f"\nExtracting and encoding (cells {start_cell}..{n_cells-1})...")
    t0 = time.time()
    cells_per_sec_history = []

    for ci in range(start_cell, n_cells):
        cell_t0 = time.time()

        if all_tokens[ci] is None:
            continue

        tokens = all_tokens[ci]
        seq_len = len(tokens)

        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        n_genes = len(gene_positions)

        if n_genes == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, 1152)

        # Process each target layer: extract gene positions, center, encode with SAE
        for layer in layers_to_process:
            layer_hidden = hidden_states[layer + 1][0]  # (seq_len, 1152)
            gene_hidden = layer_hidden[gene_positions].cpu().numpy()  # (n_genes, 1152)

            # Center with training mean
            gene_hidden -= act_means[layer][np.newaxis, :]

            # SAE encode
            with torch.no_grad():
                x = torch.tensor(gene_hidden, dtype=torch.float32)
                h_sparse, _ = saes[layer].encode(x)
                h_np = h_sparse.numpy()  # (n_genes, n_features)

            # Accumulate per-cell
            acc = accumulators[layer]
            acc['n_positions'][ci] += n_genes
            for pos in range(n_genes):
                active_mask = h_np[pos] > 0
                if active_mask.any():
                    active_idx = np.where(active_mask)[0]
                    acc['feat_sum'][ci, active_idx] += h_np[pos, active_idx]
                    acc['feat_count'][ci, active_idx] += 1

        del outputs, hidden_states, input_ids, attention_mask
        if device.type == 'mps':
            torch.mps.empty_cache()

        cell_time = time.time() - cell_t0
        cells_per_sec_history.append(1.0 / max(cell_time, 0.001))

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg_cps = np.mean(cells_per_sec_history[-50:])
            remaining_cells = n_cells - ci - 1
            eta = remaining_cells / max(avg_cps, 0.01)
            print(f"  Cell {ci+1:>5d}/{n_cells} | "
                  f"{avg_cps:.2f} cells/s | "
                  f"{elapsed/60:.1f} min elapsed | "
                  f"ETA: {eta/60:.1f} min")

        # Checkpoint every 200 cells
        if (ci + 1) % 200 == 0:
            ckpt = {
                'cells_completed': ci + 1,
                'completed_layers': sorted(completed_layers),
                'layers_in_progress': layers_to_process,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(ckpt, f, indent=2)

    extract_time = time.time() - t0
    print(f"\n  Extraction complete in {extract_time:.1f}s ({extract_time/60:.1f} min)")

    # Free Geneformer model
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    # ============================================================
    # Compute enrichments for each layer
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPUTING ENRICHMENTS")
    print("=" * 70)

    for layer in layers_to_process:
        print(f"\n--- Layer {layer} ---")
        t_layer = time.time()

        acc = accumulators[layer]

        # Compute mean-when-active
        cell_feature_mean = np.divide(
            acc['feat_sum'], acc['feat_count'],
            out=np.zeros_like(acc['feat_sum']),
            where=acc['feat_count'] > 0
        )

        mean_active = acc['feat_count'].sum() / max(acc['n_positions'].sum(), 1)
        print(f"  Mean active features per cell-position: {mean_active:.1f}")

        # Run enrichment tests
        enrichments, n_tested = run_enrichment_tests(
            cell_feature_mean, cell_tissue, cell_type, n_features
        )

        # Save
        save_enrichment(layer, enrichments, n_tested, cell_tissue, cell_type, n_cells)

        completed_layers.add(layer)

        # Free accumulator memory
        del accumulators[layer]
        gc.collect()

        print(f"  Layer {layer} done in {time.time()-t_layer:.1f}s")

        # Update checkpoint
        ckpt = {
            'cells_completed': n_cells,
            'completed_layers': sorted(completed_layers),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(ckpt, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"ALL DONE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Processed layers: {sorted(layers_to_process)}")
    print(f"{'='*70}")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == '__main__':
    main()
