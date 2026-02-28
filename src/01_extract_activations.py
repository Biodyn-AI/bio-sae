#!/usr/bin/env python3
"""
Step 1: Extract per-position residual stream activations from Geneformer V2-316M.

For each cell, at target layers, saves the hidden state vector for every
gene position in the sequence. This produces the raw training data for
SAE training.

Output per layer:
  - layer_{L}_activations.npy  (N_total, 1152) float32 memmap
  - layer_{L}_gene_ids.npy     (N_total,) int32 — token ID at each position
  - layer_{L}_cell_ids.npy     (N_total,) int32 — which cell each position belongs to
  - layer_{L}_gene_names.json  — mapping from token ID to gene name
  - extraction_metadata.json   — run metadata

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 01_extract_activations.py
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
DATA_PATH = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
OUT_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map/experiments/phase1_k562")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

N_CTRL = 2000
N_HVG = 2000
N_LAYERS = 18
HIDDEN_DIM = 1152
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

# Layers to extract — all 18 transformer layers
TARGET_LAYERS = list(range(18))

CHECKPOINT_EVERY = 100  # Save progress every N cells


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


def main():
    import torch
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("SUBPROJECT 42: EXTRACT PER-POSITION ACTIVATIONS")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

    # Check for existing checkpoint
    ckpt_path = os.path.join(OUT_DIR, "checkpoints", "extraction_checkpoint.json")
    start_cell = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        start_cell = ckpt['cells_completed']
        print(f"  Resuming from cell {start_cell}")

    # ============================================================
    # STEP 1: Load data
    # ============================================================
    print("\nSTEP 1: Loading Replogle CRISPRi data...")
    t0 = time.time()

    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    print(f"  Total: {n_cells_total:,} cells x {n_genes_total:,} genes")

    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]
    print(f"  Control cells: {len(ctrl_indices)}")

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()
    print(f"  Using {len(ctrl_indices)} control cells")

    print("  Loading control expression matrix...")
    with h5py.File(DATA_PATH, 'r') as f:
        X_ctrl_full = np.empty((len(ctrl_indices), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(ctrl_indices):
            X_ctrl_full[ci, :] = f['X'][int(idx), :]
            if (ci + 1) % 500 == 0:
                print(f"    ... {ci+1}/{len(ctrl_indices)} cells loaded")

    print("  Normalizing...")
    row_sums = X_ctrl_full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_ctrl_norm = np.log1p(X_ctrl_full / row_sums * 1e4)
    del X_ctrl_full
    gc.collect()

    print(f"  Step 1 time: {time.time()-t0:.1f}s")

    # ============================================================
    # STEP 2: Load Geneformer tokenization
    # ============================================================
    print("\nSTEP 2: Loading Geneformer gene mappings...")

    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    print(f"  Token dict: {len(token_dict)}, Median dict: {len(gene_median_dict)}, Name->ID: {len(gene_name_id_dict)}")

    # Build mapping arrays for tokenization
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

    # Build reverse mapping: token_id -> gene_name
    id_to_gene_name = {v: k for k, v in gene_name_id_dict.items()}
    token_to_ensembl = {v: k for k, v in token_dict.items()}
    token_id_to_gene_name = {}
    for tid in set(mapped_token_ids_list):
        ens = token_to_ensembl.get(tid)
        if ens:
            gname = id_to_gene_name.get(ens, ens)
            token_id_to_gene_name[int(tid)] = gname

    print(f"  Mapped var genes: {len(mapped_var_indices)}/{n_genes_total}")
    print(f"  Unique token->gene mappings: {len(token_id_to_gene_name)}")

    # ============================================================
    # STEP 3: Tokenize all cells
    # ============================================================
    print("\nSTEP 3: Tokenizing cells...")
    t0 = time.time()

    all_tokens = []
    valid_cell_indices = []
    total_positions = 0

    for ci in range(N_CTRL):
        tokens = tokenize_cell(X_ctrl_norm[ci], mapped_var_indices,
                               mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
        if tokens is not None:
            all_tokens.append(tokens)
            valid_cell_indices.append(ci)
            # Count gene positions (exclude CLS=token 2 and EOS=token 3)
            gene_positions = np.sum((tokens != 2) & (tokens != 3))
            total_positions += gene_positions

    print(f"  Valid cells: {len(all_tokens)}/{N_CTRL}")
    print(f"  Total gene positions: {total_positions:,}")
    print(f"  Mean genes/cell: {total_positions / len(all_tokens):.0f}")
    print(f"  Tokenization time: {time.time()-t0:.1f}s")

    # Save gene name mapping
    with open(os.path.join(OUT_DIR, "token_id_to_gene_name.json"), 'w') as f:
        json.dump(token_id_to_gene_name, f, indent=2, default=_json_default)

    # ============================================================
    # STEP 4: Pre-allocate output arrays
    # ============================================================
    print(f"\nSTEP 4: Pre-allocating output arrays ({total_positions:,} positions)...")

    # Memory-mapped arrays for each target layer
    act_memmaps = {}
    gene_id_memmaps = {}
    cell_id_memmaps = {}

    for layer in TARGET_LAYERS:
        act_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_activations.npy")
        gid_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_gene_ids.npy")
        cid_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_cell_ids.npy")

        if start_cell == 0:
            # Create new memmap files
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
            # Open existing memmap files
            act_memmaps[layer] = np.lib.format.open_memmap(act_path, mode='r+')
            gene_id_memmaps[layer] = np.lib.format.open_memmap(gid_path, mode='r+')
            cell_id_memmaps[layer] = np.lib.format.open_memmap(cid_path, mode='r+')

    est_gb = total_positions * HIDDEN_DIM * 4 * len(TARGET_LAYERS) / 1e9
    print(f"  Estimated total storage: {est_gb:.1f} GB across {len(TARGET_LAYERS)} layers")

    # ============================================================
    # STEP 5: Load model and extract activations
    # ============================================================
    print("\nSTEP 5: Loading Geneformer V2-316M...")
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
    # STEP 6: Extract per-position hidden states
    # ============================================================
    print(f"\nSTEP 6: Extracting activations (cells {start_cell}..{len(all_tokens)-1})...")
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

        # Identify gene positions (not CLS/EOS)
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        gene_token_ids = tokens[gene_positions]
        n_genes = len(gene_positions)

        if n_genes == 0:
            continue

        # Forward pass
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, 1152)
        # Index 0 = embedding layer, index 1..18 = transformer layers 0..17
        hidden_states = outputs.hidden_states

        # Extract gene-position activations at target layers
        for layer in TARGET_LAYERS:
            # hidden_states[layer+1] is the output of transformer layer `layer`
            layer_hidden = hidden_states[layer + 1][0]  # (seq_len, 1152)
            gene_hidden = layer_hidden[gene_positions]  # (n_genes, 1152)

            # Write to memmap
            act_memmaps[layer][pos_written:pos_written + n_genes] = gene_hidden.cpu().numpy()
            gene_id_memmaps[layer][pos_written:pos_written + n_genes] = gene_token_ids.astype(np.int32)
            cell_id_memmaps[layer][pos_written:pos_written + n_genes] = ci

        pos_written += n_genes

        # Clean up GPU memory
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

        # Checkpoint
        if (ci + 1) % CHECKPOINT_EVERY == 0:
            # Flush memmaps
            for layer in TARGET_LAYERS:
                act_memmaps[layer].flush()
                gene_id_memmaps[layer].flush()
                cell_id_memmaps[layer].flush()

            # Save checkpoint
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
    print(f"  Average speed: {len(all_tokens) / max(extract_time, 1):.1f} cells/s")

    # ============================================================
    # STEP 7: Save metadata
    # ============================================================
    print("\nSTEP 7: Saving metadata...")

    # Verify actual positions match expected
    assert pos_written == total_positions, \
        f"Position mismatch: wrote {pos_written}, expected {total_positions}"

    metadata = {
        'model': MODEL_NAME,
        'model_subfolder': MODEL_SUBFOLDER,
        'n_layers': N_LAYERS,
        'hidden_dim': HIDDEN_DIM,
        'target_layers': TARGET_LAYERS,
        'n_cells': len(all_tokens),
        'n_ctrl_source': N_CTRL,
        'n_hvg': N_HVG,
        'total_positions': int(total_positions),
        'mean_genes_per_cell': total_positions / len(all_tokens),
        'extraction_time_s': extract_time,
        'cells_per_sec': len(all_tokens) / max(extract_time, 1),
        'storage_per_layer_gb': total_positions * HIDDEN_DIM * 4 / 1e9,
        'total_storage_gb': total_positions * HIDDEN_DIM * 4 * len(TARGET_LAYERS) / 1e9,
        'data_source': DATA_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(OUT_DIR, "extraction_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    # Save cell info
    cell_info = {
        'n_cells': len(all_tokens),
        'genes_per_cell': [int(np.sum((t != 2) & (t != 3))) for t in all_tokens],
    }
    with open(os.path.join(OUT_DIR, "cell_info.json"), 'w') as f:
        json.dump(cell_info, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Cells: {len(all_tokens)}")
    print(f"  Positions: {total_positions:,}")
    print(f"  Layers: {TARGET_LAYERS}")
    print(f"  Storage: {metadata['total_storage_gb']:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
