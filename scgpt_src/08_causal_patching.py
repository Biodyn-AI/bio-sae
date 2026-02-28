#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 8: Causal Feature Patching.

For selected features at a target layer, zero the feature activation in SAE
space during forward pass, then measure downstream output change. Tests
whether features are causally necessary for the model's predictions.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 08_causal_patching.py [--layer 7] [--n-features 50] [--n-cells 200]
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
MECHINTERP_DIR = os.path.join(BASE, "biodyn-work/single_cell_mechinterp")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/causal_patching")

SCGPT_REPO = os.path.join(MECHINTERP_DIR, "external/scGPT")
SCGPT_CHECKPOINT = os.path.join(MECHINTERP_DIR, "external/scGPT_checkpoints/whole-human/best_model.pt")
SCGPT_VOCAB = os.path.join(MECHINTERP_DIR, "external/scGPT_checkpoints/whole-human/vocab.json")

D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
D_HID = 512
DROPOUT = 0.2
MAX_SEQ_LEN = 1200
EXPANSION = 4
K_VAL = 32


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def select_features(layer, n_features=50):
    """Select well-annotated features for causal patching."""
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)

    with open(os.path.join(run_dir, "feature_annotations.json")) as f:
        ann_data = json.load(f)
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)

    feature_annotations = ann_data.get('feature_annotations', {})

    feature_genes = {}
    feature_freq = {}
    for feat in catalog['features']:
        fi = feat['feature_idx']
        if feat.get('top_genes'):
            feature_genes[fi] = [g['gene_name'] for g in feat['top_genes'][:20]]
        feature_freq[fi] = feat.get('activation_freq', 0)

    scored = []
    for fid_str, anns in feature_annotations.items():
        fi = int(fid_str)
        if fi not in feature_genes or len(feature_genes[fi]) < 5:
            continue
        # Score: number of significant ontology annotations
        n_anns = len([a for a in anns if 'p_adjusted' in a])
        n_edge = len([a for a in anns if 'n_edges' in a])
        score = n_anns * 2 + n_edge
        if score > 0:
            scored.append((fi, score, feature_freq.get(fi, 0)))

    scored.sort(key=lambda x: (-x[1], -x[2]))
    selected = [s[0] for s in scored[:n_features]]

    print(f"  Selected {len(selected)} features (top annotation scores)")
    return selected, feature_genes


def main():
    import torch
    import h5py

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=7,
                        help='Layer to patch (default: 7, middle of 12)')
    parser.add_argument('--n-features', type=int, default=50)
    parser.add_argument('--n-cells', type=int, default=200)
    args = parser.parse_args()

    layer = args.layer
    n_features_to_test = args.n_features
    n_cells = args.n_cells

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"layer_{layer:02d}_patching.json")

    if os.path.exists(out_path):
        print(f"Already done: {out_path}")
        return

    print("=" * 70)
    print(f"scGPT SAE PIPELINE — STEP 8: CAUSAL PATCHING")
    print(f"  Layer: {layer}, Features: {n_features_to_test}, Cells: {n_cells}")
    print("=" * 70)

    total_t0 = time.time()

    # Load vocab
    with open(SCGPT_VOCAB) as f:
        vocab = json.load(f)
    pad_token_id = vocab['<pad>']
    id_to_gene = {v: k for k, v in vocab.items()}

    # Select features
    print("\nSelecting features...")
    selected_features, feature_genes = select_features(layer, n_features_to_test)
    if not selected_features:
        print("No annotated features found. Exiting.")
        return

    # Load SAE
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from sae_model import TopKSAE

    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
    sae.eval()
    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)

    # Load scGPT model
    print("\nLoading scGPT model...")
    sys.path.insert(0, SCGPT_REPO)
    from scgpt.model.model import TransformerModel

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TransformerModel(
        ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS,
        d_hid=D_HID, nlayers=N_LAYERS, vocab=vocab,
        dropout=DROPOUT, pad_token="<pad>", pad_value=-2,
        input_emb_style="continuous", use_fast_transformer=False,
        do_mvc=False, do_dab=False, use_batch_labels=False,
        cell_emb_style="avg-pool", n_cls=1)

    checkpoint = torch.load(SCGPT_CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict',
                    checkpoint.get('model', checkpoint)))
    converted = {k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict.items()}
    model.load_state_dict(converted, strict=False)
    model = model.to(device)
    model.eval()

    # Load tokenized TS cells (reuse from extraction)
    print(f"\nLoading cells...")
    meta_path = os.path.join(DATA_DIR, "extraction_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load from the extraction data directly
    # Use the first n_cells cells' gene_ids from layer data
    gene_ids_all = np.load(os.path.join(DATA_DIR, f"layer_{layer:02d}_gene_ids.npy"), mmap_mode='r')
    cell_ids_all = np.load(os.path.join(DATA_DIR, f"layer_{layer:02d}_cell_ids.npy"), mmap_mode='r')

    # Find unique cells and their position ranges
    unique_cells = sorted(set(cell_ids_all[:1000000].tolist()))[:n_cells]

    print(f"  Using {len(unique_cells)} cells for patching")

    # For each feature, run patching experiment
    results = {}

    for feat_idx_i, feat_idx in enumerate(selected_features):
        feat_t0 = time.time()

        # Gene set for this feature
        feat_gene_names = set(g.upper() for g in feature_genes.get(feat_idx, []))
        feat_gene_ids = set()
        for gname, gid in vocab.items():
            if gname.upper() in feat_gene_names:
                feat_gene_ids.add(gid)

        if not feat_gene_ids:
            continue

        # For each cell: normal forward, patched forward
        logit_changes = []  # Change in logits for feature-related genes
        logit_changes_control = []  # Change in logits for random genes

        for ci_idx, ci in enumerate(unique_cells[:n_cells]):
            # Find positions for this cell
            cell_mask = cell_ids_all == ci
            # This is expensive on memmap, so use a range search
            # Assume cell positions are contiguous
            first_pos = np.searchsorted(cell_ids_all, ci, side='left')
            last_pos = np.searchsorted(cell_ids_all, ci, side='right')
            if first_pos >= last_pos:
                continue

            cell_gene_ids = gene_ids_all[first_pos:last_pos].astype(np.int64)
            n_genes = len(cell_gene_ids)
            if n_genes < 10:
                continue

            # Build scGPT input for this cell
            # Gene IDs + values (we don't have exact values, use ones as proxy)
            gene_ids_t = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long, device=device)
            gene_values_t = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.float32, device=device)
            padding_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.bool, device=device)

            seq_len = min(n_genes, MAX_SEQ_LEN)
            gene_ids_t[0, :seq_len] = torch.tensor(cell_gene_ids[:seq_len], dtype=torch.long)
            gene_values_t[0, :seq_len] = 1.0  # Uniform values as proxy
            padding_mask[0, :seq_len] = False

            # Normal forward pass — capture hidden state at target layer
            hidden_states = {}

            def capture_hook(layer_idx):
                def hook_fn(module, input, output):
                    hidden_states[layer_idx] = output.detach().clone()
                return hook_fn

            hooks = []
            for i, layer_mod in enumerate(model.transformer_encoder.layers):
                hooks.append(layer_mod.register_forward_hook(capture_hook(i)))

            with torch.no_grad():
                normal_output = model._encode(
                    src=gene_ids_t, values=gene_values_t,
                    src_key_padding_mask=padding_mask)

            normal_logits = normal_output[0, :seq_len].cpu()  # (seq_len, d_model)

            # Get hidden state at target layer
            target_hidden = hidden_states[layer][0, :seq_len]  # (seq_len, d_model)

            # Encode through SAE
            centered = target_hidden.cpu() - act_mean_t
            with torch.no_grad():
                h_sparse, topk_idx = sae.encode(centered)

            # Zero the target feature
            h_patched = h_sparse.clone()
            h_patched[:, feat_idx] = 0.0

            # Decode back
            with torch.no_grad():
                reconstructed = sae.decode(h_patched) + act_mean_t

            # Replace hidden state and re-run remaining layers
            patched_hidden = reconstructed.to(device).unsqueeze(0)

            # Pad back to full seq
            full_patched = hidden_states[layer].clone()
            full_patched[0, :seq_len] = patched_hidden[0]

            # Run through remaining layers manually
            x = full_patched
            for remaining_layer in range(layer + 1, N_LAYERS):
                with torch.no_grad():
                    x = model.transformer_encoder.layers[remaining_layer](
                        x, src_key_padding_mask=padding_mask)

            patched_logits = x[0, :seq_len].cpu()

            # Remove hooks
            for h in hooks:
                h.remove()

            # Measure logit change for feature-related genes vs. random genes
            logit_diff = (normal_logits - patched_logits).norm(dim=-1)  # per-position norm

            # Feature-related positions
            feat_positions = [p for p in range(seq_len)
                              if int(cell_gene_ids[p]) in feat_gene_ids]
            # Control positions (random sample)
            control_positions = [p for p in range(seq_len)
                                 if int(cell_gene_ids[p]) not in feat_gene_ids]

            if feat_positions:
                logit_changes.append(float(logit_diff[feat_positions].mean()))
            if control_positions:
                n_ctrl = min(len(control_positions), max(len(feat_positions), 5))
                ctrl_sample = np.random.choice(control_positions, n_ctrl, replace=False)
                logit_changes_control.append(float(logit_diff[ctrl_sample].mean()))

            del hidden_states, normal_output
            if device.type == 'mps':
                torch.mps.empty_cache()

        # Compute specificity
        if logit_changes and logit_changes_control:
            mean_feat = np.mean(logit_changes)
            mean_ctrl = np.mean(logit_changes_control)
            specificity = mean_feat / max(mean_ctrl, 1e-10)

            results[int(feat_idx)] = {
                'feature_idx': int(feat_idx),
                'n_cells_tested': len(logit_changes),
                'mean_feat_logit_change': float(mean_feat),
                'mean_ctrl_logit_change': float(mean_ctrl),
                'specificity': float(specificity),
                'top_genes': feature_genes.get(feat_idx, [])[:10],
            }

            elapsed = time.time() - feat_t0
            print(f"  Feature {feat_idx:>5d} ({feat_idx_i+1}/{len(selected_features)}) | "
                  f"Spec: {specificity:.2f}x | "
                  f"Feat: {mean_feat:.4f} | Ctrl: {mean_ctrl:.4f} | "
                  f"{elapsed:.1f}s")

    # Save results
    specificity_values = [r['specificity'] for r in results.values()]
    summary = {
        'layer': layer,
        'n_features_tested': len(results),
        'n_cells': n_cells,
        'median_specificity': float(np.median(specificity_values)) if specificity_values else 0,
        'mean_specificity': float(np.mean(specificity_values)) if specificity_values else 0,
        'n_above_2x': int(sum(1 for s in specificity_values if s > 2.0)),
        'n_above_5x': int(sum(1 for s in specificity_values if s > 5.0)),
    }

    output = {
        'summary': summary,
        'features': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Features tested: {len(results)}")
    print(f"  Median specificity: {summary['median_specificity']:.2f}x")
    print(f"  >2x specificity: {summary['n_above_2x']}/{len(results)}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
