#!/usr/bin/env python3
"""
Step 8: Causal Feature Patching.

Test whether individual SAE features are causally necessary for the model's
predictions. For each feature:
  1. Run normal forward pass, record hidden state at target layer
  2. Encode through SAE, zero the target feature, decode back
  3. Replace hidden state with modified version, continue forward pass
  4. Measure change in output logits — specifically for genes related to
     the feature's biology vs. unrelated genes (specificity)

This distinguishes "the model has learned biology" from "features merely
reflect co-expression statistics."

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 08_causal_patching.py [--layer 11] [--n-features 50] [--n-cells 200]
"""

import os
import sys
import gc
import json
import time
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PAPER_DIR = os.path.join(BASE, "biodyn-nmi-paper")
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
DATA_PATH = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

EXPANSION = 4
K_VAL = 32
N_CTRL = 2000
MAX_SEQ_LEN = 2048
HIDDEN_DIM = 1152


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_categorical_column(h5group, col_name):
    import h5py
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


def select_features(layer, n_features=50):
    """Select well-annotated features for causal patching.

    Picks features with the most ontology annotations and clear biological
    identity (strong enrichment p-values across multiple ontologies).
    """
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)

    with open(os.path.join(run_dir, "feature_annotations.json")) as f:
        ann_data = json.load(f)
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)

    feature_annotations = ann_data.get('feature_annotations', {})

    # Build gene sets for each feature
    feature_genes = {}
    feature_freq = {}
    for feat in catalog['features']:
        fi = feat['feature_idx']
        if feat.get('top_genes'):
            feature_genes[fi] = [g['gene_name'] for g in feat['top_genes'][:20]]
        feature_freq[fi] = feat.get('activation_freq', 0)

    # Score features by annotation quality
    scored = []
    for fid_str, anns in feature_annotations.items():
        fi = int(fid_str)
        if fi not in feature_genes or len(feature_genes[fi]) < 10:
            continue
        if feature_freq.get(fi, 0) < 0.01:
            continue  # Too rare to test reliably

        ontologies = set(a['ontology'] for a in anns)
        n_ont = len(ontologies)
        n_ann = len(anns)
        min_p = min(a.get('p_adjusted', 1.0) for a in anns) if anns else 1.0

        # Get top GO/KEGG/Reactome term as label
        best_label = "unknown"
        for a in anns:
            if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                best_label = a['term']
                break

        # Collect annotated gene set for this feature
        annotated_gene_sets = set()
        for a in anns:
            if 'genes' in a:
                annotated_gene_sets.update(a['genes'])

        scored.append({
            'feature_idx': fi,
            'n_ontologies': n_ont,
            'n_annotations': n_ann,
            'min_p': min_p,
            'label': best_label,
            'top_genes': feature_genes[fi],
            'activation_freq': feature_freq.get(fi, 0),
            'score': n_ont * 10 + n_ann - np.log10(max(min_p, 1e-30)),
        })

    # Sort by score, take top N
    scored.sort(key=lambda x: -x['score'])
    selected = scored[:n_features]
    print(f"  Selected {len(selected)} features for patching")
    for i, s in enumerate(selected[:10]):
        print(f"    [{i}] Feature {s['feature_idx']}: {s['n_ontologies']} ont, "
              f"{s['n_annotations']} ann, freq={s['activation_freq']:.3f} | {s['label'][:50]}")

    return selected


def run_causal_patching(layer, selected_features, n_cells=200):
    """Run causal patching experiment."""
    import torch
    import h5py
    from transformers import BertForMaskedLM
    sys.path.insert(0, os.path.dirname(__file__))
    from sae_model import TopKSAE

    out_dir = os.path.join(DATA_DIR, "causal_patching")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"causal_patching_layer{layer:02d}.json")
    if os.path.exists(out_path):
        print(f"  Already done: {out_path}")
        return out_path

    # ============================================================
    # Load SAE
    # ============================================================
    print("  Loading SAE...")
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
    sae.eval()

    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)

    # ============================================================
    # Load Geneformer
    # ============================================================
    print("  Loading Geneformer V2-316M...")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("    Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("    Using CPU")

    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_hidden_states=True,
        output_attentions=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"    Model loaded in {time.time()-t0:.1f}s")

    # ============================================================
    # Tokenize cells
    # ============================================================
    print("  Loading and tokenizing cells...")

    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

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

    # Subsample for causal patching
    cell_sample = ctrl_indices[:n_cells]
    print(f"    Using {len(cell_sample)} cells for patching")

    # Load expression
    with h5py.File(DATA_PATH, 'r') as f:
        X_sample = np.empty((len(cell_sample), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(cell_sample):
            X_sample[ci, :] = f['X'][int(idx), :]

    row_sums = X_sample.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_sample = np.log1p(X_sample / row_sums * 1e4)

    # Build tokenization arrays
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

    # Reverse mapping
    id_to_gene_name = {v: k for k, v in gene_name_id_dict.items()}
    token_to_ensembl = {v: k for k, v in token_dict.items()}
    token_id_to_gene_name = {}
    for tid in set(mapped_token_ids_list):
        ens = token_to_ensembl.get(tid)
        if ens:
            gname = id_to_gene_name.get(ens, ens)
            token_id_to_gene_name[int(tid)] = gname

    # Tokenize
    all_tokens = []
    for ci in range(len(cell_sample)):
        tokens = tokenize_cell(X_sample[ci], mapped_var_indices,
                               mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
        if tokens is not None:
            all_tokens.append(tokens)
    print(f"    Tokenized {len(all_tokens)} cells")
    del X_sample
    gc.collect()

    # ============================================================
    # Build gene-to-feature mapping for specificity test
    # ============================================================
    # For each feature, determine which token IDs correspond to its top genes
    gene_name_to_token_ids = {}
    for tid, gname in token_id_to_gene_name.items():
        if gname not in gene_name_to_token_ids:
            gene_name_to_token_ids[gname] = []
        gene_name_to_token_ids[gname].append(tid)

    feature_target_tokens = {}
    for sf in selected_features:
        fi = sf['feature_idx']
        target_tids = set()
        for gname in sf['top_genes']:
            tids = gene_name_to_token_ids.get(gname, [])
            target_tids.update(tids)
        feature_target_tokens[fi] = target_tids

    # ============================================================
    # Run patching
    # ============================================================
    print(f"\n  Running causal patching on {len(selected_features)} features × {len(all_tokens)} cells...")
    print(f"  Target layer: {layer}")

    feature_results = []
    total_t0 = time.time()

    # Process features one at a time
    for fi_idx, sf in enumerate(selected_features):
        fi = sf['feature_idx']
        target_tids = feature_target_tokens[fi]

        # Accumulators
        logit_diffs_target = []   # Logit changes for target genes
        logit_diffs_other = []    # Logit changes for other genes
        cos_dists = []            # Cosine distance of hidden states
        feature_activations = []  # Original feature activation values

        t0 = time.time()

        for ci, tokens in enumerate(all_tokens):
            seq_len = len(tokens)
            gene_mask = (tokens != 2) & (tokens != 3)
            gene_positions = np.where(gene_mask)[0]
            gene_token_ids = tokens[gene_positions]

            if len(gene_positions) == 0:
                continue

            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

            # --- Normal forward pass ---
            with torch.no_grad():
                outputs_normal = model(input_ids=input_ids, attention_mask=attention_mask)

            normal_logits = outputs_normal.logits[0].cpu()  # (seq_len, vocab_size)
            normal_hidden = outputs_normal.hidden_states[layer + 1][0].cpu()  # (seq_len, 1152)

            # --- Encode through SAE at gene positions ---
            gene_hidden = normal_hidden[gene_positions]  # (n_genes, 1152)
            gene_hidden_centered = gene_hidden - act_mean_t

            with torch.no_grad():
                h_sparse, topk_indices = sae.encode(gene_hidden_centered)

            # Check if target feature is active at any position
            active_mask = (topk_indices == fi).any(dim=1)  # (n_genes,)
            if not active_mask.any():
                continue  # Feature not active in this cell — skip

            # Record activation values where active
            for pos_idx in range(len(gene_positions)):
                if active_mask[pos_idx]:
                    act_val = h_sparse[pos_idx, fi].item()
                    feature_activations.append(act_val)

            # --- Ablation: zero the target feature, decode, replace ---
            h_ablated = h_sparse.clone()
            h_ablated[:, fi] = 0.0  # Zero the target feature

            # Decode both
            with torch.no_grad():
                recon_normal = sae.decode(h_sparse) + act_mean_t
                recon_ablated = sae.decode(h_ablated) + act_mean_t

            # Compute the delta to apply to original hidden state
            delta = recon_ablated - recon_normal  # (n_genes, 1152)

            # Build modified hidden state for full sequence
            modified_hidden = normal_hidden.clone()
            modified_hidden[gene_positions] = normal_hidden[gene_positions] + delta

            # Cosine distance between original and modified
            cos = torch.nn.functional.cosine_similarity(
                normal_hidden[gene_positions][active_mask],
                modified_hidden[gene_positions][active_mask],
                dim=1,
            )
            cos_dists.extend((1.0 - cos).tolist())

            # --- Forward pass from modified hidden state ---
            # Use a hook to replace the hidden state at layer `layer`
            hook_fired = [False]

            def make_hook(new_hidden):
                def hook_fn(module, input, output):
                    if hook_fired[0]:
                        return output
                    hook_fired[0] = True
                    # output is a tuple: (hidden_states, ...)
                    # Replace hidden_states with our modified version
                    modified = list(output)
                    modified[0] = new_hidden.unsqueeze(0).to(device)
                    return tuple(modified)
                return hook_fn

            # Register hook on the target layer's output
            # model.bert.encoder.layer[layer] is the transformer block
            hook_handle = model.bert.encoder.layer[layer].register_forward_hook(
                make_hook(modified_hidden)
            )

            with torch.no_grad():
                outputs_ablated = model(input_ids=input_ids, attention_mask=attention_mask)

            hook_handle.remove()
            ablated_logits = outputs_ablated.logits[0].cpu()  # (seq_len, vocab_size)

            # --- Measure logit changes at gene positions ---
            for pos_idx, gpos in enumerate(gene_positions):
                if not active_mask[pos_idx]:
                    continue  # Only measure where feature was active

                tid = int(gene_token_ids[pos_idx])
                # Logit difference at this position for the correct token
                normal_logit = normal_logits[gpos, tid].item()
                ablated_logit = ablated_logits[gpos, tid].item()
                diff = ablated_logit - normal_logit

                if tid in target_tids:
                    logit_diffs_target.append(diff)
                else:
                    logit_diffs_other.append(diff)

            # Cleanup
            del outputs_normal, outputs_ablated, normal_logits, ablated_logits
            del normal_hidden, modified_hidden, gene_hidden
            if device.type == 'mps':
                torch.mps.empty_cache()

        elapsed = time.time() - t0

        # Compile results for this feature
        result = {
            'feature_idx': fi,
            'label': sf['label'],
            'n_ontologies': sf['n_ontologies'],
            'n_annotations': sf['n_annotations'],
            'activation_freq': sf['activation_freq'],
            'top_genes': sf['top_genes'][:10],
            'n_target_tokens': len(target_tids),
            'n_cells_active': len(feature_activations),
            'mean_activation': float(np.mean(feature_activations)) if feature_activations else 0,
            'n_target_measurements': len(logit_diffs_target),
            'n_other_measurements': len(logit_diffs_other),
        }

        if logit_diffs_target:
            result['target_logit_diff_mean'] = float(np.mean(logit_diffs_target))
            result['target_logit_diff_std'] = float(np.std(logit_diffs_target))
        else:
            result['target_logit_diff_mean'] = 0
            result['target_logit_diff_std'] = 0

        if logit_diffs_other:
            result['other_logit_diff_mean'] = float(np.mean(logit_diffs_other))
            result['other_logit_diff_std'] = float(np.std(logit_diffs_other))
        else:
            result['other_logit_diff_mean'] = 0
            result['other_logit_diff_std'] = 0

        if cos_dists:
            result['mean_cos_distance'] = float(np.mean(cos_dists))
        else:
            result['mean_cos_distance'] = 0

        # Specificity: ratio of target impact to other impact
        if result['other_logit_diff_mean'] != 0:
            result['specificity_ratio'] = abs(result['target_logit_diff_mean']) / abs(result['other_logit_diff_mean'])
        else:
            result['specificity_ratio'] = float('inf') if result['target_logit_diff_mean'] != 0 else 0

        feature_results.append(result)

        print(f"  [{fi_idx+1}/{len(selected_features)}] Feature {fi} ({sf['label'][:40]}): "
              f"active in {result['n_cells_active']} cells, "
              f"target Δlogit={result['target_logit_diff_mean']:.4f}, "
              f"other Δlogit={result['other_logit_diff_mean']:.4f}, "
              f"specificity={result['specificity_ratio']:.2f} "
              f"({elapsed:.1f}s)")

        # Incremental save
        if (fi_idx + 1) % 10 == 0:
            _save_results(out_path, layer, feature_results, selected_features, all_tokens)

    total_elapsed = time.time() - total_t0
    _save_results(out_path, layer, feature_results, selected_features, all_tokens)
    print(f"\n  Total patching time: {total_elapsed/60:.1f} min")
    print(f"  Saved: {out_path}")

    return out_path


def _save_results(out_path, layer, feature_results, selected_features, all_tokens):
    """Save intermediate results."""
    # Compute summary statistics
    specs = [r['specificity_ratio'] for r in feature_results
             if r['specificity_ratio'] != float('inf') and r['specificity_ratio'] > 0]
    target_diffs = [r['target_logit_diff_mean'] for r in feature_results
                    if r['n_target_measurements'] > 0]
    other_diffs = [r['other_logit_diff_mean'] for r in feature_results
                   if r['n_other_measurements'] > 0]

    output = {
        'layer': layer,
        'config': {
            'n_features_tested': len(feature_results),
            'n_features_selected': len(selected_features),
            'n_cells': len(all_tokens),
        },
        'summary': {
            'n_features_with_activity': sum(1 for r in feature_results if r['n_cells_active'] > 0),
            'mean_specificity_ratio': float(np.mean(specs)) if specs else 0,
            'median_specificity_ratio': float(np.median(specs)) if specs else 0,
            'mean_target_logit_diff': float(np.mean(target_diffs)) if target_diffs else 0,
            'mean_other_logit_diff': float(np.mean(other_diffs)) if other_diffs else 0,
            'n_specific_features': sum(1 for s in specs if s > 2.0),
            'frac_specific': sum(1 for s in specs if s > 2.0) / max(len(specs), 1),
        },
        'feature_results': feature_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=11,
                        help='Layer to patch (default: 11, mid-late with good annotations)')
    parser.add_argument('--n-features', type=int, default=50,
                        help='Number of features to test')
    parser.add_argument('--n-cells', type=int, default=200,
                        help='Number of cells to patch through')
    args = parser.parse_args()

    total_t0 = time.time()

    print("=" * 70)
    print("STEP 8: CAUSAL FEATURE PATCHING")
    print(f"  Layer: {args.layer}")
    print(f"  Features: {args.n_features}")
    print(f"  Cells: {args.n_cells}")
    print("=" * 70)

    # Select features
    selected = select_features(args.layer, args.n_features)

    # Run patching
    run_causal_patching(args.layer, selected, args.n_cells)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
