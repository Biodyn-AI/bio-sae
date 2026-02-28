#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 1: Extract per-position hidden-state activations
from scGPT whole-human model for 3000 Tabula Sapiens cells.

Extracts all 12 layers using forward hooks on TransformerEncoderLayer.
Uses the same 3000 cells (1000 immune + 1000 kidney + 1000 lung)
selected for the Geneformer Phase 3 pipeline.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 01_extract_activations.py
"""

import os
import sys
import gc
import json
import time
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# MPS fallback for unsupported ops (nested tensor in TransformerEncoder)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import h5py

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
MECHINTERP_DIR = os.path.join(BASE, "biodyn-work/single_cell_mechinterp")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")

# scGPT model
SCGPT_REPO = os.path.join(MECHINTERP_DIR, "external/scGPT")
SCGPT_CHECKPOINT = os.path.join(MECHINTERP_DIR, "external/scGPT_checkpoints/whole-human/best_model.pt")
SCGPT_VOCAB = os.path.join(MECHINTERP_DIR, "external/scGPT_checkpoints/whole-human/vocab.json")

# Model architecture (from args.json)
N_LAYERS = 12
N_HEADS = 8
D_MODEL = 512
D_HID = 512
DROPOUT = 0.2
MAX_SEQ_LEN = 1200

# Extraction settings
BATCH_SIZE = 1
CHECKPOINT_EVERY = 100

# Cell data (reuse Geneformer Phase 3 selection)
GF_METADATA = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/ts_activations/extraction_metadata.json")

TISSUES = {
    'immune': os.path.join(MECHINTERP_DIR, "data/raw/tabula_sapiens_immune_subset_20000.h5ad"),
    'kidney': os.path.join(MECHINTERP_DIR, "data/raw/tabula_sapiens_kidney.h5ad"),
    'lung': os.path.join(MECHINTERP_DIR, "data/raw/tabula_sapiens_lung.h5ad"),
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


def build_gene_name_map(h5_path):
    """Build gene index -> gene symbol mapping from h5ad file."""
    with h5py.File(h5_path, 'r') as f:
        # feature_name contains gene symbols
        if 'feature_name' in f['var']:
            fn = f['var']['feature_name']
            if isinstance(fn, h5py.Group):
                categories = fn['categories'][:]
                codes = fn['codes'][:]
                if categories.dtype.kind in ('O', 'S'):
                    categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
                gene_names = categories[codes]
            else:
                gene_names = fn[:]
                if gene_names.dtype.kind in ('O', 'S'):
                    gene_names = np.array([x.decode() if isinstance(x, bytes) else x for x in gene_names])
        else:
            # Fall back to var index
            var_index = f['var']['_index'][:]
            gene_names = np.array([x.decode() if isinstance(x, bytes) else x for x in var_index])

        n_genes = len(gene_names)
    return gene_names, n_genes


def tokenize_cell_scgpt(expression_vector, gene_names, vocab, pad_token_id,
                        max_seq_len=1200, pad_value=-2):
    """
    Tokenize a single cell for scGPT.

    Returns:
        gene_ids: (max_seq_len,) int64 — token IDs, padded
        gene_values: (max_seq_len,) float32 — expression values, padded
        src_key_padding_mask: (max_seq_len,) bool — True where padded
        n_genes: int — number of non-padded positions
        mapped_gene_names: list of str — gene names in order (unpadded)
    """
    # Filter to nonzero expressed genes that exist in vocab
    nonzero_mask = expression_vector > 0
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(nonzero_indices) == 0:
        return None

    # Map to vocab
    valid_indices = []
    valid_token_ids = []
    valid_expr = []
    valid_names = []

    for idx in nonzero_indices:
        gname = gene_names[idx]
        if gname in vocab:
            valid_indices.append(idx)
            valid_token_ids.append(vocab[gname])
            valid_expr.append(expression_vector[idx])
            valid_names.append(gname)

    if len(valid_token_ids) == 0:
        return None

    valid_token_ids = np.array(valid_token_ids, dtype=np.int64)
    valid_expr = np.array(valid_expr, dtype=np.float32)

    # Sort by expression descending (scGPT convention)
    order = np.argsort(-valid_expr)
    valid_token_ids = valid_token_ids[order]
    valid_expr = valid_expr[order]
    valid_names = [valid_names[i] for i in order]

    # Truncate to max_seq_len
    if len(valid_token_ids) > max_seq_len:
        valid_token_ids = valid_token_ids[:max_seq_len]
        valid_expr = valid_expr[:max_seq_len]
        valid_names = valid_names[:max_seq_len]

    n_genes = len(valid_token_ids)

    # Pad to max_seq_len
    pad_len = max_seq_len - n_genes
    gene_ids = np.pad(valid_token_ids, (0, pad_len), mode='constant',
                      constant_values=pad_token_id)
    gene_values = np.pad(valid_expr, (0, pad_len), mode='constant',
                         constant_values=pad_value)
    src_key_padding_mask = np.zeros(max_seq_len, dtype=bool)
    src_key_padding_mask[n_genes:] = True

    return {
        'gene_ids': gene_ids,
        'gene_values': gene_values,
        'src_key_padding_mask': src_key_padding_mask,
        'n_genes': n_genes,
        'gene_names': valid_names,
    }


def main():
    import torch
    total_t0 = time.time()

    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 1: EXTRACT ACTIVATIONS")
    print(f"Model: whole-human scGPT (12L x 8H x 512D)")
    print(f"Layers: all {N_LAYERS}")
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
    # STEP 1: Load vocab
    # ============================================================
    print("\nSTEP 1: Loading scGPT vocab...")
    with open(SCGPT_VOCAB, 'r') as f:
        vocab = json.load(f)
    pad_token_id = vocab['<pad>']
    id_to_gene = {v: k for k, v in vocab.items()}
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Pad token ID: {pad_token_id}")

    # ============================================================
    # STEP 2: Load cell metadata from Geneformer Phase 3
    # ============================================================
    print("\nSTEP 2: Loading cell metadata from Geneformer Phase 3...")
    with open(GF_METADATA, 'r') as f:
        gf_meta = json.load(f)

    cell_data = gf_meta['cell_data']  # List of {tissue, cell_type, cell_idx}
    n_total_cells = len(cell_data)
    print(f"  Total cells: {n_total_cells}")

    tissue_counts = {}
    for cd in cell_data:
        tissue_counts[cd['tissue']] = tissue_counts.get(cd['tissue'], 0) + 1
    for tissue, count in tissue_counts.items():
        print(f"    {tissue}: {count}")

    # ============================================================
    # STEP 3: Build gene name mappings for each tissue h5ad
    # ============================================================
    print("\nSTEP 3: Building gene name mappings per tissue...")

    tissue_gene_data = {}
    for tissue_name, h5_path in TISSUES.items():
        gene_names, n_genes = build_gene_name_map(h5_path)

        # Count how many map to scGPT vocab
        n_mapped = sum(1 for g in gene_names if g in vocab)
        tissue_gene_data[tissue_name] = {
            'gene_names': gene_names,
            'n_genes': n_genes,
        }
        print(f"  {tissue_name}: {n_mapped}/{n_genes} genes in scGPT vocab ({100*n_mapped/n_genes:.1f}%)")

    # ============================================================
    # STEP 4: Tokenize all cells
    # ============================================================
    print("\nSTEP 4: Tokenizing cells for scGPT...")
    t0 = time.time()

    all_tokenized = []
    valid_cell_indices = []
    total_positions = 0

    for ci, cell_info in enumerate(cell_data):
        tissue = cell_info['tissue']
        h5_path = TISSUES[tissue]
        cell_idx = cell_info['cell_idx']
        gdata = tissue_gene_data[tissue]

        # Load expression
        with h5py.File(h5_path, 'r') as f:
            expr = load_sparse_row(f['X'], cell_idx, gdata['n_genes'])

        # Tokenize for scGPT
        result = tokenize_cell_scgpt(
            expr, gdata['gene_names'], vocab, pad_token_id,
            max_seq_len=MAX_SEQ_LEN, pad_value=-2
        )

        if result is not None:
            all_tokenized.append(result)
            valid_cell_indices.append(ci)
            total_positions += result['n_genes']

        if (ci + 1) % 500 == 0:
            print(f"    Tokenized {ci+1}/{n_total_cells} cells...")

    n_valid = len(all_tokenized)
    print(f"  Valid cells: {n_valid}/{n_total_cells}")
    print(f"  Total gene positions: {total_positions:,}")
    print(f"  Mean genes/cell: {total_positions / max(n_valid, 1):.1f}")
    print(f"  Tokenization time: {time.time()-t0:.1f}s")

    # Save gene name mapping (token_id -> gene_name for later analysis)
    gene_name_map_path = os.path.join(OUT_DIR, "token_id_to_gene_name.json")
    if not os.path.exists(gene_name_map_path):
        # Build from all tokenized cells
        token_id_to_gene = {}
        for tok_data in all_tokenized:
            for i, gname in enumerate(tok_data['gene_names']):
                tid = int(tok_data['gene_ids'][i])
                if tid not in token_id_to_gene:
                    token_id_to_gene[tid] = gname
        with open(gene_name_map_path, 'w') as f:
            json.dump(token_id_to_gene, f, indent=2, default=_json_default)
        print(f"  Saved token->gene mapping ({len(token_id_to_gene)} entries)")

    # ============================================================
    # STEP 5: Pre-allocate output arrays
    # ============================================================
    print(f"\nSTEP 5: Pre-allocating output arrays ({total_positions:,} positions × {D_MODEL}D × {N_LAYERS} layers)...")

    target_layers = list(range(N_LAYERS))
    act_memmaps = {}
    gene_id_memmaps = {}
    cell_id_memmaps = {}

    for layer in target_layers:
        act_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_activations.npy")
        gid_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_gene_ids.npy")
        cid_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_cell_ids.npy")

        if start_cell == 0:
            act_memmaps[layer] = np.lib.format.open_memmap(
                act_path, mode='w+', dtype=np.float32,
                shape=(total_positions, D_MODEL))
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

    est_gb = total_positions * D_MODEL * 4 * N_LAYERS / 1e9
    print(f"  Estimated storage: {est_gb:.1f} GB across {N_LAYERS} layers")

    # ============================================================
    # STEP 6: Load scGPT model
    # ============================================================
    print("\nSTEP 6: Loading scGPT whole-human model...")
    t0 = time.time()

    # Add scGPT repo to path
    sys.path.insert(0, SCGPT_REPO)
    import scgpt  # noqa — needed for module resolution
    from scgpt.model.model import TransformerModel

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("  Using CPU")

    # Construct model (use_fast_transformer=False since no FlashMHA on macOS)
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=D_MODEL,
        nhead=N_HEADS,
        d_hid=D_HID,
        nlayers=N_LAYERS,
        vocab=vocab,
        dropout=DROPOUT,
        pad_token="<pad>",
        pad_value=-2,
        input_emb_style="continuous",
        use_fast_transformer=False,  # FlashMHA not available on macOS
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        cell_emb_style="avg-pool",
        n_cls=1,
    )

    # Load checkpoint with FlashMHA -> MultiheadAttention weight conversion
    checkpoint = torch.load(SCGPT_CHECKPOINT, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Convert FlashMHA weight names to standard MultiheadAttention names
    # FlashMHA uses "Wqkv" while nn.MultiheadAttention uses "in_proj"
    converted_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("Wqkv.", "in_proj_")
        converted_state_dict[new_k] = v

    # Load with strict=False to handle any extra/missing keys
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Verify transformer layers
    n_encoder_layers = len(model.transformer_encoder.layers)
    print(f"  Transformer encoder layers: {n_encoder_layers}")
    assert n_encoder_layers == N_LAYERS, f"Expected {N_LAYERS} layers, got {n_encoder_layers}"

    # ============================================================
    # STEP 7: Register hooks for hidden state extraction
    # ============================================================
    print("\nSTEP 7: Registering forward hooks on encoder layers...")

    layer_outputs = {}  # layer_idx -> tensor

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            layer_outputs[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for i, layer_module in enumerate(model.transformer_encoder.layers):
        h = layer_module.register_forward_hook(make_hook(i))
        hooks.append(h)
    print(f"  Registered {len(hooks)} hooks")

    # ============================================================
    # STEP 8: Extract per-position hidden states
    # ============================================================
    print(f"\nSTEP 8: Extracting activations (cells {start_cell}..{n_valid-1})...")
    t0 = time.time()

    # Compute write offset for resuming
    write_offset = 0
    if start_cell > 0:
        for ci in range(start_cell):
            write_offset += all_tokenized[ci]['n_genes']
        print(f"  Resume write offset: {write_offset}")

    pos_written = write_offset
    cells_per_sec_history = []

    for ci in range(start_cell, n_valid):
        cell_t0 = time.time()
        tok_data = all_tokenized[ci]
        n_genes = tok_data['n_genes']

        if n_genes == 0:
            continue

        # Prepare inputs
        gene_ids_t = torch.tensor(tok_data['gene_ids'], dtype=torch.long).unsqueeze(0).to(device)
        gene_values_t = torch.tensor(tok_data['gene_values'], dtype=torch.float32).unsqueeze(0).to(device)
        padding_mask_t = torch.tensor(tok_data['src_key_padding_mask'], dtype=torch.bool).unsqueeze(0).to(device)

        # Forward pass
        layer_outputs.clear()
        with torch.no_grad():
            model._encode(
                src=gene_ids_t,
                values=gene_values_t,
                src_key_padding_mask=padding_mask_t,
            )

        # Extract gene positions (non-padded)
        gene_token_ids = tok_data['gene_ids'][:n_genes].astype(np.int32)

        for layer in target_layers:
            if layer not in layer_outputs:
                print(f"  WARNING: Layer {layer} not captured for cell {ci}")
                continue
            layer_hidden = layer_outputs[layer][0]  # (seq_len, d_model)
            gene_hidden = layer_hidden[:n_genes]  # (n_genes, d_model)
            act_memmaps[layer][pos_written:pos_written + n_genes] = gene_hidden.cpu().numpy()
            gene_id_memmaps[layer][pos_written:pos_written + n_genes] = gene_token_ids
            cell_id_memmaps[layer][pos_written:pos_written + n_genes] = ci

        pos_written += n_genes

        # Cleanup
        del gene_ids_t, gene_values_t, padding_mask_t
        layer_outputs.clear()
        if device.type == 'mps':
            torch.mps.empty_cache()

        cell_time = time.time() - cell_t0
        cells_per_sec_history.append(1.0 / max(cell_time, 0.001))

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg_cps = np.mean(cells_per_sec_history[-50:])
            remaining = (n_valid - ci - 1) / max(avg_cps, 0.01)
            print(f"  Cell {ci+1:>5d}/{n_valid} | "
                  f"Positions: {pos_written:>8,} | "
                  f"{avg_cps:.1f} cells/s | "
                  f"ETA: {remaining/60:.1f} min")

        if (ci + 1) % CHECKPOINT_EVERY == 0:
            for layer in target_layers:
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
    for layer in target_layers:
        act_memmaps[layer].flush()
        gene_id_memmaps[layer].flush()
        cell_id_memmaps[layer].flush()

    # Remove hooks
    for h in hooks:
        h.remove()

    extract_time = time.time() - t0
    print(f"\n  Extraction complete: {pos_written:,} positions, {extract_time:.1f}s")

    # ============================================================
    # STEP 9: Save metadata
    # ============================================================
    print("\nSTEP 9: Saving metadata...")

    # Collect tissue stats
    tissue_stats = {}
    for cd in cell_data:
        tissue = cd['tissue']
        if tissue not in tissue_stats:
            tissue_stats[tissue] = {'n_selected': 0, 'cell_types': {}}
        tissue_stats[tissue]['n_selected'] += 1
        ct = cd['cell_type']
        tissue_stats[tissue]['cell_types'][ct] = tissue_stats[tissue]['cell_types'].get(ct, 0) + 1
    for ts in tissue_stats.values():
        ts['n_cell_types'] = len(ts['cell_types'])

    metadata = {
        'model': 'scGPT-whole-human',
        'model_checkpoint': SCGPT_CHECKPOINT,
        'architecture': {
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'd_model': D_MODEL,
            'd_hid': D_HID,
            'max_seq_len': MAX_SEQ_LEN,
        },
        'target_layers': target_layers,
        'n_cells': n_valid,
        'total_positions': int(pos_written),
        'mean_genes_per_cell': pos_written / max(n_valid, 1),
        'extraction_time_s': extract_time,
        'cells_per_sec': n_valid / max(extract_time, 1),
        'storage_per_layer_gb': pos_written * D_MODEL * 4 / 1e9,
        'tissue_stats': tissue_stats,
        'cell_data': [
            {'tissue': cell_data[vi]['tissue'],
             'cell_type': cell_data[vi]['cell_type'],
             'cell_idx': cell_data[vi]['cell_idx']}
            for vi in valid_cell_indices
        ],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(OUT_DIR, "extraction_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Cells: {n_valid}")
    print(f"  Positions: {pos_written:,}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Storage: {pos_written * D_MODEL * 4 * N_LAYERS / 1e9:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
