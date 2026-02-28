#!/usr/bin/env python3
"""
Phase 3, Script 4: Perturbation response test with multi-tissue SAE.

Re-runs the Phase 2 Step 9 perturbation response test using multi-tissue SAEs
on layers 0, 5, 11, 17. Same CRISPRi perturbation data (Replogle K562),
same 100 targets (48 TRRUST TFs + 52 others), same 20 cells/target.

The key question: does training on diverse cell types unlock TF-specific
feature responses that the K562-only SAE couldn't detect?

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 12d_perturbation_test.py
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
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PAPER_DIR = os.path.join(BASE, "biodyn-nmi-paper")
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")

# Multi-tissue SAE location
SAE_BASE = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/sae_models")
# K562 activations for control baseline
K562_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")

TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
DATA_PATH = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")
TRRUST_PATH = os.path.join(BASE, "biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv")

OUT_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/perturbation_response")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

EXPANSION = 4
K_VAL = 32
HIDDEN_DIM = 1152
MAX_SEQ_LEN = 2048
TARGET_LAYERS = [0, 5, 11, 17]
N_TARGETS = 100
CELLS_PER_TARGET = 20


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


def select_perturbation_targets():
    """Select perturbation targets — identical to Phase 2 Step 9."""
    import h5py

    print("  Selecting perturbation targets...")

    trrust_tfs = set()
    trrust_targets = {}
    if os.path.exists(TRRUST_PATH):
        with open(TRRUST_PATH) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    tf = parts[0]
                    target = parts[1]
                    trrust_tfs.add(tf)
                    if tf not in trrust_targets:
                        trrust_targets[tf] = set()
                    trrust_targets[tf].add(target)
        print(f"    TRRUST: {len(trrust_tfs)} TFs, {sum(len(v) for v in trrust_targets.values())} edges")

    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        cell_lines = load_categorical_column(f['obs'], 'cell_line')

    k562_mask = cell_lines == 'k562'
    k562_genes = cell_genes[k562_mask]
    k562_indices = np.where(k562_mask)[0]

    unique_genes, counts = np.unique(k562_genes, return_counts=True)
    gene_counts = dict(zip(unique_genes, counts))
    del gene_counts['non-targeting']

    scored = []
    for gene, count in gene_counts.items():
        if count < 30:
            continue
        is_tf = gene in trrust_tfs
        n_known_targets = len(trrust_targets.get(gene, set()))
        score = n_known_targets * 100 + count
        scored.append({
            'gene': gene,
            'n_cells': int(count),
            'is_trrust_tf': is_tf,
            'n_known_targets': n_known_targets,
            'score': score,
        })

    scored.sort(key=lambda x: -x['score'])
    selected = scored[:N_TARGETS]

    n_tfs = sum(1 for s in selected if s['is_trrust_tf'])
    print(f"    Selected {len(selected)} targets: {n_tfs} TRRUST TFs")

    return selected, trrust_targets, k562_indices, cell_genes


def run_perturbation_response_layer(layer, selected_targets, trrust_targets,
                                     k562_indices, cell_genes,
                                     model, device, mapped_var_indices,
                                     mapped_token_ids, mapped_medians,
                                     token_id_to_gene_name, n_genes_total):
    """Run perturbation test for one layer using multi-tissue SAE."""
    import torch
    from sae_model import TopKSAE

    out_path = os.path.join(OUT_DIR, f"perturbation_response_layer{layer:02d}.json")
    partial_path = out_path.replace('.json', '_partial.json')

    # Check for existing results
    completed_targets = set()
    partial_results = []
    if os.path.exists(out_path):
        print(f"  Layer {layer}: already done, skipping")
        return
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        partial_results = partial.get('target_results', [])
        completed_targets = {r['target_gene'] for r in partial_results}
        print(f"  Resuming from partial: {len(completed_targets)} targets done")

    # Load multi-tissue SAE
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
    sae.eval()
    n_features = sae.n_features

    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)

    # Compute control baseline from K562 pre-extracted activations
    print(f"  Computing control baseline (K562 activations, multi-tissue SAE)...")
    t0 = time.time()

    k562_act_path = os.path.join(K562_DIR, f"layer_{layer:02d}_activations.npy")
    k562_activations = np.lib.format.open_memmap(k562_act_path, mode='r')
    n_total = k562_activations.shape[0]

    n_sample = min(100000, n_total)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, n_sample, replace=False)
    sample_idx.sort()

    ctrl_feature_means = np.zeros(n_features, dtype=np.float64)
    ctrl_feature_vars = np.zeros(n_features, dtype=np.float64)
    ctrl_feature_freq = np.zeros(n_features, dtype=np.float64)

    batch_size = 4096
    for start in range(0, n_sample, batch_size):
        end = min(start + batch_size, n_sample)
        idx = sample_idx[start:end]
        batch = torch.tensor(k562_activations[idx], dtype=torch.float32) - act_mean_t
        with torch.no_grad():
            h_sparse, _ = sae.encode(batch)
        h_np = h_sparse.numpy()
        ctrl_feature_means += h_np.sum(axis=0)
        ctrl_feature_vars += (h_np ** 2).sum(axis=0)
        ctrl_feature_freq += (h_np > 0).sum(axis=0)

    ctrl_feature_means /= n_sample
    ctrl_feature_vars = ctrl_feature_vars / n_sample - ctrl_feature_means ** 2
    ctrl_feature_freq /= n_sample

    print(f"    Control baseline: {n_sample:,} samples ({time.time()-t0:.1f}s)")
    print(f"    Active features: {(ctrl_feature_freq > 0).sum()}")

    del k562_activations
    gc.collect()

    # Load feature annotations and catalog
    ann_path = os.path.join(run_dir, "feature_annotations.json")
    catalog_path = os.path.join(run_dir, "feature_catalog.json")

    feature_annotations = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            ann_data = json.load(f)
        feature_annotations = ann_data.get('feature_annotations', {})

    feature_top_genes = {}
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog = json.load(f)
        for feat in catalog['features']:
            fi = feat['feature_idx']
            if feat.get('top_genes'):
                feature_top_genes[fi] = set(g['gene_name'] for g in feat['top_genes'][:20])

    # Process perturbation targets
    print(f"\n  Processing {len(selected_targets)} targets for layer {layer}...")
    target_results = list(partial_results)

    for ti, target_info in enumerate(selected_targets):
        target_gene = target_info['gene']
        if target_gene in completed_targets:
            continue

        t0 = time.time()

        target_cell_mask = (cell_genes == target_gene)
        target_cell_indices = np.intersect1d(
            np.where(target_cell_mask)[0], k562_indices)

        if len(target_cell_indices) == 0:
            continue

        rng_t = np.random.RandomState(42 + ti)
        if len(target_cell_indices) > CELLS_PER_TARGET:
            target_cell_indices = rng_t.choice(
                target_cell_indices, CELLS_PER_TARGET, replace=False)
        target_cell_indices.sort()

        import h5py
        with h5py.File(DATA_PATH, 'r') as f:
            X_target = np.empty((len(target_cell_indices), n_genes_total), dtype=np.float32)
            for ci, idx in enumerate(target_cell_indices):
                X_target[ci, :] = f['X'][int(idx), :]

        row_sums = X_target.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_target = np.log1p(X_target / row_sums * 1e4)

        perturbed_tokens = []
        for ci in range(len(target_cell_indices)):
            tokens = tokenize_cell(X_target[ci], mapped_var_indices,
                                   mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
            if tokens is not None:
                perturbed_tokens.append(tokens)
        del X_target

        if len(perturbed_tokens) == 0:
            continue

        perturbed_feature_sums = np.zeros(n_features, dtype=np.float64)
        perturbed_feature_active = np.zeros(n_features, dtype=np.float64)
        n_positions = 0

        for ci, tokens in enumerate(perturbed_tokens):
            seq_len = len(tokens)
            gene_mask = (tokens != 2) & (tokens != 3)
            gene_positions = np.where(gene_mask)[0]

            if len(gene_positions) == 0:
                continue

            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            hidden = outputs.hidden_states[layer + 1][0].cpu()
            gene_hidden = hidden[gene_positions] - act_mean_t

            with torch.no_grad():
                h_sparse, _ = sae.encode(gene_hidden)

            h_np = h_sparse.numpy()
            perturbed_feature_sums += h_np.sum(axis=0)
            perturbed_feature_active += (h_np > 0).sum(axis=0)
            n_positions += len(gene_positions)

            del outputs, hidden
            if device.type == 'mps':
                torch.mps.empty_cache()

        if n_positions == 0:
            continue

        pert_means = perturbed_feature_sums / n_positions
        pert_freq = perturbed_feature_active / n_positions

        ctrl_std = np.sqrt(np.maximum(ctrl_feature_vars, 1e-10))
        effect_sizes = (pert_means - ctrl_feature_means) / ctrl_std

        abs_effects = np.abs(effect_sizes)
        top_changed_idx = np.argsort(-abs_effects)[:20]

        top_changed = []
        for fi in top_changed_idx:
            fi = int(fi)
            if abs_effects[fi] < 0.1:
                break

            feat_genes = feature_top_genes.get(fi, set())
            known_targets = trrust_targets.get(target_gene, set())
            overlaps_target = target_gene in feat_genes
            overlaps_known_targets = len(feat_genes & known_targets) if known_targets else 0

            anns = feature_annotations.get(str(fi), [])
            label = "unlabeled"
            for a in anns:
                if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                    label = a['term']
                    break

            top_changed.append({
                'feature_idx': fi,
                'effect_size': float(effect_sizes[fi]),
                'ctrl_mean': float(ctrl_feature_means[fi]),
                'pert_mean': float(pert_means[fi]),
                'ctrl_freq': float(ctrl_feature_freq[fi]),
                'pert_freq': float(pert_freq[fi]),
                'label': label,
                'overlaps_target_gene': overlaps_target,
                'n_overlaps_known_targets': overlaps_known_targets,
            })

        known_targets = trrust_targets.get(target_gene, set())
        n_specific_responding = 0
        n_responding = 0
        for tc in top_changed:
            if abs(tc['effect_size']) > 0.5:
                n_responding += 1
                if tc['overlaps_target_gene'] or tc['n_overlaps_known_targets'] > 0:
                    n_specific_responding += 1

        elapsed = time.time() - t0

        result = {
            'target_gene': target_gene,
            'is_trrust_tf': target_info['is_trrust_tf'],
            'n_known_targets': target_info['n_known_targets'],
            'n_cells_used': len(perturbed_tokens),
            'n_positions': n_positions,
            'n_responding_features': n_responding,
            'n_specific_responding': n_specific_responding,
            'top_changed_features': top_changed,
            'mean_abs_effect': float(np.mean(abs_effects)),
            'max_abs_effect': float(np.max(abs_effects)),
        }
        target_results.append(result)

        tf_tag = " [TF]" if target_info['is_trrust_tf'] else ""
        print(f"    [{ti+1}/{len(selected_targets)}] {target_gene}{tf_tag}: "
              f"{n_responding} responding, {n_specific_responding} specific, "
              f"max_effect={result['max_abs_effect']:.2f} ({elapsed:.1f}s)")

        # Incremental save
        if (len(target_results) - len(partial_results)) % 10 == 0:
            _save_results(partial_path, layer, target_results, ctrl_feature_freq)

    # Final save
    _save_results(out_path, layer, target_results, ctrl_feature_freq)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    del sae
    gc.collect()


def _save_results(out_path, layer, target_results, ctrl_feature_freq):
    n_tfs_tested = sum(1 for r in target_results if r['is_trrust_tf'])
    n_tfs_with_specific = sum(1 for r in target_results
                               if r['is_trrust_tf'] and r['n_specific_responding'] > 0)
    responding_counts = [r['n_responding_features'] for r in target_results]
    specific_counts = [r['n_specific_responding'] for r in target_results]

    output = {
        'layer': layer,
        'sae_type': 'multi_tissue',
        'config': {
            'n_targets': len(target_results),
            'cells_per_target': CELLS_PER_TARGET,
        },
        'summary': {
            'n_targets_processed': len(target_results),
            'n_trrust_tfs': n_tfs_tested,
            'n_tfs_with_specific_response': n_tfs_with_specific,
            'frac_tfs_specific': n_tfs_with_specific / max(n_tfs_tested, 1),
            'mean_responding_features': float(np.mean(responding_counts)) if responding_counts else 0,
            'mean_specific_features': float(np.mean(specific_counts)) if specific_counts else 0,
            'n_alive_features': int((ctrl_feature_freq > 0).sum()),
        },
        'target_results': target_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)


def main():
    import torch
    import h5py
    from transformers import BertForMaskedLM

    total_t0 = time.time()

    print("=" * 70)
    print("PHASE 3: PERTURBATION RESPONSE TEST (MULTI-TISSUE SAE)")
    print(f"  Layers: {TARGET_LAYERS}")
    print(f"  Targets: {N_TARGETS}")
    print(f"  Cells per target: {CELLS_PER_TARGET}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Select perturbation targets (same as Phase 2)
    selected, trrust_targets, k562_indices, cell_genes = \
        select_perturbation_targets()

    # Load Geneformer
    print("\n  Loading Geneformer V2-316M...")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
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
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Load tokenization resources
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    with h5py.File(DATA_PATH, 'r') as f:
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_genes_total = f['X'].shape[1]

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

    # Gene name lookups
    id_to_gene_name = {v: k for k, v in gene_name_id_dict.items()}
    token_to_ensembl = {v: k for k, v in token_dict.items()}
    token_id_to_gene_name = {}
    for tid in set(mapped_token_ids_list):
        ens = token_to_ensembl.get(tid)
        if ens:
            gname = id_to_gene_name.get(ens, ens)
            token_id_to_gene_name[int(tid)] = gname

    # Run perturbation test for each layer
    for layer in TARGET_LAYERS:
        print(f"\n{'=' * 60}")
        print(f"  LAYER {layer}: PERTURBATION RESPONSE")
        print(f"{'=' * 60}")

        layer_t0 = time.time()
        run_perturbation_response_layer(
            layer, selected, trrust_targets,
            k562_indices, cell_genes,
            model, device,
            mapped_var_indices, mapped_token_ids, mapped_medians,
            token_id_to_gene_name, n_genes_total,
        )
        print(f"  Layer {layer} done: {(time.time()-layer_t0)/60:.1f} min")

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
