#!/usr/bin/env python3
"""
Phase 3, Script 1: Extract per-position activations from Geneformer for
Tabula Sapiens multi-tissue cells.

Extracts layers 0, 5, 11, 17 from 3000 cells (1000 immune + 1000 kidney
+ 1000 lung) with stratified sampling by cell type for maximum TF diversity.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 12a_extract_tabula_sapiens.py
"""

import os
import sys
import gc
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import h5py

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PAPER_DIR = os.path.join(BASE, "biodyn-nmi-paper")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/ts_activations")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

HIDDEN_DIM = 1152
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
CHECKPOINT_EVERY = 100

TARGET_LAYERS = [0, 5, 11, 17]

TISSUES = {
    'immune': {
        'path': os.path.join(BASE, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"),
        'n_cells': 1000,
    },
    'kidney': {
        'path': os.path.join(BASE, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"),
        'n_cells': 1000,
    },
    'lung': {
        'path': os.path.join(BASE, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"),
        'n_cells': 1000,
    },
}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_categorical_column(h5group, col_name):
    """Load a categorical or plain column from h5py group."""
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
    """Load a single row from a sparse CSR matrix in h5py."""
    indptr = f_group['indptr']
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    indices = f_group['indices'][start:end]
    data = f_group['data'][start:end]
    row = np.zeros(n_cols, dtype=np.float32)
    row[indices] = data
    return row


def tokenize_cell(expression_vector, var_indices, token_ids, medians, max_len=2048):
    """Tokenize a single cell for Geneformer (rank-value encoding)."""
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


def select_cells_stratified(h5_path, n_cells, seed=42):
    """Select cells with stratified sampling by cell type."""
    with h5py.File(h5_path, 'r') as f:
        cell_types = load_categorical_column(f['obs'], 'cell_type')
        n_total = len(cell_types)

    unique_types, type_counts = np.unique(cell_types, return_counts=True)
    n_types = len(unique_types)

    rng = np.random.RandomState(seed)

    # Proportional allocation with minimum 1 per type (if enough cells)
    n_cells = min(n_cells, n_total)
    allocations = {}
    remaining = n_cells
    for ct, count in sorted(zip(unique_types, type_counts), key=lambda x: x[1]):
        alloc = max(1, int(round(count / n_total * n_cells)))
        alloc = min(alloc, count, remaining)
        allocations[ct] = alloc
        remaining -= alloc
        if remaining <= 0:
            break

    # Distribute any remaining slots to largest types
    if remaining > 0:
        for ct, count in sorted(zip(unique_types, type_counts), key=lambda x: -x[1]):
            can_add = min(remaining, count - allocations.get(ct, 0))
            if can_add > 0:
                allocations[ct] = allocations.get(ct, 0) + can_add
                remaining -= can_add
            if remaining <= 0:
                break

    # Sample from each type
    selected = []
    for ct, n_select in allocations.items():
        ct_indices = np.where(cell_types == ct)[0]
        chosen = rng.choice(ct_indices, min(n_select, len(ct_indices)), replace=False)
        selected.extend(chosen.tolist())

    selected.sort()
    return np.array(selected), cell_types


def main():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("PHASE 3: EXTRACT TABULA SAPIENS ACTIVATIONS")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

    # Check for checkpoint
    ckpt_path = os.path.join(OUT_DIR, "checkpoints", "extraction_checkpoint.json")
    start_cell = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        start_cell = ckpt['cells_completed']
        print(f"  Resuming from cell {start_cell}")

    # ============================================================
    # STEP 1: Load tokenization dicts
    # ============================================================
    print("\nSTEP 1: Loading Geneformer tokenization dicts...")

    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)

    print(f"  Token dict: {len(token_dict)} entries")
    print(f"  Median dict: {len(gene_median_dict)} entries")

    # Reverse mapping for gene names
    token_to_ensembl = {v: k for k, v in token_dict.items()}

    # ============================================================
    # STEP 2: Select cells and build gene mappings per tissue
    # ============================================================
    print("\nSTEP 2: Selecting cells from each tissue...")

    all_cell_data = []  # List of (tissue, h5_path, cell_idx, cell_type)
    tissue_stats = {}

    for tissue_name, tissue_info in TISSUES.items():
        h5_path = tissue_info['path']
        n_cells = tissue_info['n_cells']

        print(f"\n  {tissue_name}: {h5_path}")
        selected_indices, cell_types = select_cells_stratified(h5_path, n_cells)

        selected_types = cell_types[selected_indices]
        unique_sel, counts_sel = np.unique(selected_types, return_counts=True)
        n_types_sel = len(unique_sel)

        for idx in selected_indices:
            all_cell_data.append({
                'tissue': tissue_name,
                'h5_path': h5_path,
                'cell_idx': int(idx),
                'cell_type': cell_types[idx],
            })

        tissue_stats[tissue_name] = {
            'n_selected': len(selected_indices),
            'n_cell_types': n_types_sel,
            'cell_types': {ct: int(c) for ct, c in zip(unique_sel, counts_sel)},
        }
        print(f"    Selected {len(selected_indices)} cells, {n_types_sel} cell types")
        for ct, c in sorted(zip(unique_sel, counts_sel), key=lambda x: -x[1])[:5]:
            print(f"      {ct}: {c}")
        if n_types_sel > 5:
            print(f"      ... and {n_types_sel - 5} more types")

    n_total_cells = len(all_cell_data)
    print(f"\n  Total cells selected: {n_total_cells}")

    # ============================================================
    # STEP 3: Build gene mapping for each tissue file
    # ============================================================
    print("\nSTEP 3: Building gene mappings per tissue...")

    tissue_gene_maps = {}
    token_id_to_gene_name = {}

    for tissue_name, tissue_info in TISSUES.items():
        h5_path = tissue_info['path']
        with h5py.File(h5_path, 'r') as f:
            # var/_index contains Ensembl IDs (no version)
            var_index = f['var']['_index'][:]
            n_genes = len(var_index)

            # Also get feature names for gene name mapping
            feature_name_cats = f['var']['feature_name']['categories'][:]
            feature_name_codes = f['var']['feature_name']['codes'][:]

        # Map Ensembl IDs directly to token dict
        mapped_var_indices = []
        mapped_token_ids_list = []
        mapped_medians_list = []

        for i in range(n_genes):
            ensembl_id = var_index[i].decode() if isinstance(var_index[i], bytes) else var_index[i]
            if ensembl_id in token_dict:
                mapped_var_indices.append(i)
                mapped_token_ids_list.append(token_dict[ensembl_id])
                mapped_medians_list.append(gene_median_dict.get(ensembl_id, 1.0))

                # Build gene name mapping
                tid = token_dict[ensembl_id]
                if tid not in token_id_to_gene_name:
                    gname = feature_name_cats[feature_name_codes[i]]
                    if isinstance(gname, bytes):
                        gname = gname.decode()
                    token_id_to_gene_name[int(tid)] = gname

        tissue_gene_maps[tissue_name] = {
            'mapped_var_indices': np.array(mapped_var_indices),
            'mapped_token_ids': np.array(mapped_token_ids_list),
            'mapped_medians': np.array(mapped_medians_list),
            'n_genes_total': n_genes,
        }
        print(f"  {tissue_name}: {len(mapped_var_indices)}/{n_genes} genes mapped to token dict")

    # ============================================================
    # STEP 4: Tokenize all cells
    # ============================================================
    print("\nSTEP 4: Tokenizing cells...")
    t0 = time.time()

    all_tokens = []
    valid_cell_indices = []
    total_positions = 0
    tissue_positions = {}

    for ci, cell_info in enumerate(all_cell_data):
        tissue = cell_info['tissue']
        h5_path = cell_info['h5_path']
        cell_idx = cell_info['cell_idx']
        gmap = tissue_gene_maps[tissue]

        # Load expression for this cell (sparse CSR)
        with h5py.File(h5_path, 'r') as f:
            expr = load_sparse_row(f['X'], cell_idx, gmap['n_genes_total'])

        # Normalize same way as K562 pipeline
        row_sum = expr.sum()
        if row_sum > 0:
            expr = np.log1p(expr / row_sum * 1e4)

        # Tokenize
        tokens = tokenize_cell(
            expr, gmap['mapped_var_indices'],
            gmap['mapped_token_ids'], gmap['mapped_medians'],
            MAX_SEQ_LEN
        )

        if tokens is not None:
            all_tokens.append(tokens)
            valid_cell_indices.append(ci)
            gene_positions = int(np.sum((tokens != 2) & (tokens != 3)))
            total_positions += gene_positions
            tissue_positions[tissue] = tissue_positions.get(tissue, 0) + gene_positions

        if (ci + 1) % 500 == 0:
            print(f"    Tokenized {ci+1}/{n_total_cells} cells...")

    print(f"  Valid cells: {len(all_tokens)}/{n_total_cells}")
    print(f"  Total gene positions: {total_positions:,}")
    print(f"  Mean genes/cell: {total_positions / max(len(all_tokens), 1):.0f}")
    for tissue, pos in tissue_positions.items():
        n_tissue = sum(1 for c in all_cell_data if c['tissue'] == tissue)
        print(f"    {tissue}: {pos:,} positions ({pos/max(n_tissue,1):.0f} genes/cell)")
    print(f"  Tokenization time: {time.time()-t0:.1f}s")

    # Save gene name mapping
    with open(os.path.join(OUT_DIR, "token_id_to_gene_name.json"), 'w') as f:
        json.dump(token_id_to_gene_name, f, indent=2, default=_json_default)

    # ============================================================
    # STEP 5: Pre-allocate output arrays
    # ============================================================
    print(f"\nSTEP 5: Pre-allocating output arrays ({total_positions:,} positions)...")

    act_memmaps = {}
    gene_id_memmaps = {}
    cell_id_memmaps = {}

    for layer in TARGET_LAYERS:
        act_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_activations.npy")
        gid_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_gene_ids.npy")
        cid_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_cell_ids.npy")

        if start_cell == 0:
            act_memmaps[layer] = np.lib.format.open_memmap(
                act_path, mode='w+', dtype=np.float32,
                shape=(total_positions, HIDDEN_DIM))
            gene_id_memmaps[layer] = np.lib.format.open_memmap(
                gid_path, mode='w+', dtype=np.int32,
                shape=(total_positions,))
            cell_id_memmaps[layer] = np.lib.format.open_memmap(
                cid_path, mode='w+', dtype=np.int32,
                shape=(total_positions,))
        else:
            act_memmaps[layer] = np.lib.format.open_memmap(act_path, mode='r+')
            gene_id_memmaps[layer] = np.lib.format.open_memmap(gid_path, mode='r+')
            cell_id_memmaps[layer] = np.lib.format.open_memmap(cid_path, mode='r+')

    est_gb = total_positions * HIDDEN_DIM * 4 * len(TARGET_LAYERS) / 1e9
    print(f"  Estimated storage: {est_gb:.1f} GB across {len(TARGET_LAYERS)} layers")

    # ============================================================
    # STEP 6: Load model and extract activations
    # ============================================================
    print("\nSTEP 6: Loading Geneformer V2-316M...")
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
    # STEP 7: Extract per-position hidden states
    # ============================================================
    print(f"\nSTEP 7: Extracting activations (cells {start_cell}..{len(all_tokens)-1})...")
    t0 = time.time()

    # Compute write offset for resuming
    write_offset = 0
    if start_cell > 0:
        for ci in range(start_cell):
            tokens = all_tokens[ci]
            gene_mask = (tokens != 2) & (tokens != 3)
            write_offset += gene_mask.sum()
        print(f"  Resume write offset: {write_offset}")

    pos_written = write_offset
    cells_per_sec_history = []

    for ci in range(start_cell, len(all_tokens)):
        cell_t0 = time.time()
        tokens = all_tokens[ci]
        seq_len = len(tokens)

        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        gene_token_ids = tokens[gene_positions]
        n_genes = len(gene_positions)

        if n_genes == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states

        for layer in TARGET_LAYERS:
            layer_hidden = hidden_states[layer + 1][0]  # (seq_len, 1152)
            gene_hidden = layer_hidden[gene_positions]  # (n_genes, 1152)
            act_memmaps[layer][pos_written:pos_written + n_genes] = gene_hidden.cpu().numpy()
            gene_id_memmaps[layer][pos_written:pos_written + n_genes] = gene_token_ids.astype(np.int32)
            cell_id_memmaps[layer][pos_written:pos_written + n_genes] = ci

        pos_written += n_genes

        del outputs, hidden_states, input_ids, attention_mask
        if device.type == 'mps':
            torch.mps.empty_cache()

        cell_time = time.time() - cell_t0
        cells_per_sec_history.append(1.0 / max(cell_time, 0.001))

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg_cps = np.mean(cells_per_sec_history[-50:])
            remaining = (len(all_tokens) - ci - 1) / max(avg_cps, 0.01)
            print(f"  Cell {ci+1:>5d}/{len(all_tokens)} | "
                  f"Positions: {pos_written:>8,} | "
                  f"{avg_cps:.1f} cells/s | "
                  f"ETA: {remaining/60:.1f} min")

        if (ci + 1) % CHECKPOINT_EVERY == 0:
            for layer in TARGET_LAYERS:
                act_memmaps[layer].flush()
                gene_id_memmaps[layer].flush()
                cell_id_memmaps[layer].flush()
            ckpt = {
                'cells_completed': ci + 1,
                'positions_written': int(pos_written),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(ckpt_path, 'w') as f:
                json.dump(ckpt, f, indent=2)

    # Final flush
    for layer in TARGET_LAYERS:
        act_memmaps[layer].flush()
        gene_id_memmaps[layer].flush()
        cell_id_memmaps[layer].flush()

    extract_time = time.time() - t0
    print(f"\n  Extraction complete: {pos_written:,} positions, {extract_time:.1f}s")

    # ============================================================
    # STEP 8: Save metadata
    # ============================================================
    print("\nSTEP 8: Saving metadata...")

    metadata = {
        'model': MODEL_NAME,
        'model_subfolder': MODEL_SUBFOLDER,
        'target_layers': TARGET_LAYERS,
        'n_cells': len(all_tokens),
        'total_positions': int(pos_written),
        'mean_genes_per_cell': pos_written / max(len(all_tokens), 1),
        'extraction_time_s': extract_time,
        'cells_per_sec': len(all_tokens) / max(extract_time, 1),
        'storage_per_layer_gb': pos_written * HIDDEN_DIM * 4 / 1e9,
        'tissue_stats': tissue_stats,
        'tissue_positions': {k: int(v) for k, v in tissue_positions.items()},
        'cell_data': [
            {'tissue': c['tissue'], 'cell_type': c['cell_type'], 'cell_idx': c['cell_idx']}
            for c in [all_cell_data[vi] for vi in valid_cell_indices]
        ],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(OUT_DIR, "extraction_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Cells: {len(all_tokens)}")
    print(f"  Positions: {pos_written:,}")
    print(f"  Layers: {TARGET_LAYERS}")
    print(f"  Storage: {pos_written * HIDDEN_DIM * 4 * len(TARGET_LAYERS) / 1e9:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
