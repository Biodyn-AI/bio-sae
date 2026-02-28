#!/usr/bin/env python3
"""
Extract per-position activations for the 13 remaining layers (1-4, 6-10, 12-14, 16).
Layers 0, 5, 11, 15, 17 already extracted by 01_extract_activations.py.

Reuses the same tokenization and cell ordering for consistency.
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

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PAPER_DIR = os.path.join(BASE, "biodyn-nmi-paper")
DATA_PATH = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
OUT_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map/experiments/phase1_k562")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_LAYERS = 18
HIDDEN_DIM = 1152
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

# Only the layers we haven't extracted yet
ALREADY_DONE = {0, 5, 11, 15, 17}
TARGET_LAYERS = [i for i in range(N_LAYERS) if i not in ALREADY_DONE]

CHECKPOINT_EVERY = 100


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


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


def main():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("EXTRACT REMAINING 13 LAYERS")
    print(f"Layers: {TARGET_LAYERS}")
    print(f"Already done: {sorted(ALREADY_DONE)}")
    print("=" * 70)

    # Check for checkpoint
    ckpt_path = os.path.join(OUT_DIR, "checkpoints", "extraction_remaining_checkpoint.json")
    start_cell = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        start_cell = ckpt['cells_completed']
        print(f"  Resuming from cell {start_cell}")

    # ============================================================
    # Load data (same as original script)
    # ============================================================
    print("\nLoading data...")
    t0 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()

    print(f"  Loading {len(ctrl_indices)} control cells...")
    with h5py.File(DATA_PATH, 'r') as f:
        X_ctrl_full = np.empty((len(ctrl_indices), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(ctrl_indices):
            X_ctrl_full[ci, :] = f['X'][int(idx), :]

    row_sums = X_ctrl_full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_ctrl_norm = np.log1p(X_ctrl_full / row_sums * 1e4)
    del X_ctrl_full
    gc.collect()
    print(f"  Data loaded: {time.time()-t0:.1f}s")

    # ============================================================
    # Tokenization
    # ============================================================
    print("Loading tokenization...")

    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []
    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(gene_median_dict.get(ensembl, 1.0))

    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)

    print("Tokenizing cells...")
    all_tokens = []
    total_positions = 0
    for ci in range(N_CTRL):
        tokens = tokenize_cell(X_ctrl_norm[ci], mapped_var_indices,
                               mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
        if tokens is not None:
            all_tokens.append(tokens)
            gene_positions = np.sum((tokens != 2) & (tokens != 3))
            total_positions += gene_positions

    print(f"  {len(all_tokens)} cells, {total_positions:,} positions")

    # ============================================================
    # Pre-allocate memmap files for remaining layers
    # ============================================================
    print(f"\nPre-allocating arrays for {len(TARGET_LAYERS)} layers...")
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
    print(f"  New storage: {est_gb:.1f} GB across {len(TARGET_LAYERS)} layers")

    # ============================================================
    # Load model
    # ============================================================
    print("\nLoading Geneformer V2-316M...")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_hidden_states=True,
        output_attentions=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {time.time()-t0:.1f}s")

    # ============================================================
    # Extract
    # ============================================================
    # Compute write offset for resuming
    write_offset = 0
    if start_cell > 0:
        for ci in range(start_cell):
            tokens = all_tokens[ci]
            gene_mask = (tokens != 2) & (tokens != 3)
            write_offset += gene_mask.sum()
        print(f"  Resume offset: {write_offset}")

    pos_written = write_offset
    t0 = time.time()

    print(f"\nExtracting layers {TARGET_LAYERS} (cells {start_cell}..{len(all_tokens)-1})...")

    for ci in range(start_cell, len(all_tokens)):
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
            layer_hidden = hidden_states[layer + 1][0]
            gene_hidden = layer_hidden[gene_positions]
            act_memmaps[layer][pos_written:pos_written + n_genes] = gene_hidden.cpu().numpy()
            gene_id_memmaps[layer][pos_written:pos_written + n_genes] = gene_token_ids.astype(np.int32)
            cell_id_memmaps[layer][pos_written:pos_written + n_genes] = ci

        pos_written += n_genes

        del outputs, hidden_states, input_ids, attention_mask
        if device.type == 'mps':
            torch.mps.empty_cache()

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t0
            cps = (ci + 1 - start_cell) / max(elapsed, 0.01)
            remaining = (len(all_tokens) - ci - 1) / max(cps, 0.01)
            print(f"  Cell {ci+1:>5d}/{len(all_tokens)} | "
                  f"Pos: {pos_written:>8,} | "
                  f"{cps:.1f} cells/s | "
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
    print(f"\n  Done: {pos_written:,} positions, {extract_time:.1f}s")

    # Update metadata
    meta_path = os.path.join(OUT_DIR, "extraction_metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata['target_layers'] = list(range(N_LAYERS))
    metadata['total_storage_gb'] = total_positions * HIDDEN_DIM * 4 * N_LAYERS / 1e9
    metadata['remaining_layers_extraction_time_s'] = extract_time
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  New layers: {TARGET_LAYERS}")
    print(f"  All 18 layers now extracted")
    print(f"  Total storage: {metadata['total_storage_gb']:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
