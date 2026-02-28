#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 7: Feature co-activation graph.

For each layer, compute feature co-activation, PMI, and Leiden clustering.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 07_coactivation.py --all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/coactivation")

EXPANSION = 4
K_VAL = 32
BATCH_SIZE = 8192
PMI_THRESHOLD = 2.0
MIN_COACTIVATION = 50
N_LAYERS = 12


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def process_layer(layer):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from sae_model import TopKSAE

    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(OUT_DIR, f"coactivation_layer{layer:02d}.json")
    if os.path.exists(out_path):
        print(f"\n  Layer {layer}: Already done.")
        return

    model_path = os.path.join(run_dir, "sae_final.pt")
    if not os.path.exists(model_path):
        print(f"\n  Layer {layer}: SAE model not found, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Layer {layer}: Co-activation Analysis")
    print(f"{'=' * 60}")

    t0 = time.time()

    sae = TopKSAE.load(model_path, device='cpu')
    sae.eval()
    n_features = sae.n_features

    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)

    activations = np.lib.format.open_memmap(
        os.path.join(DATA_DIR, f"layer_{layer:02d}_activations.npy"), mode='r')
    n_total = activations.shape[0]
    print(f"  Activations: {n_total:,}, Features: {n_features}")

    # Co-activation matrix (2048^2 * 4 bytes = 16 MB — much smaller than Geneformer's 81 MB)
    feature_count = np.zeros(n_features, dtype=np.int64)
    coact_matrix = np.zeros((n_features, n_features), dtype=np.int32)

    print("  Computing co-activations...")
    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch = torch.tensor(activations[start:end], dtype=torch.float32) - act_mean_t

        with torch.no_grad():
            h_sparse, topk_indices = sae.encode(batch)

        indices_np = topk_indices.numpy()
        for fi in indices_np.ravel():
            feature_count[fi] += 1

        for row in indices_np:
            row_sorted = np.sort(row)
            ii, jj = np.triu_indices(len(row_sorted), k=1)
            coact_matrix[row_sorted[ii], row_sorted[jj]] += 1

        if (end) % (BATCH_SIZE * 50) < BATCH_SIZE:
            print(f"    {end:>10,}/{n_total:,}")

    # PMI
    print("  Computing PMI...")
    rows, cols = np.where(coact_matrix >= MIN_COACTIVATION)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]

    pmi_edges = []
    for idx in range(len(rows)):
        i, j = int(rows[idx]), int(cols[idx])
        count = int(coact_matrix[i, j])
        p_ij = count / n_total
        p_i = feature_count[i] / n_total
        p_j = feature_count[j] / n_total
        if p_i == 0 or p_j == 0:
            continue
        pmi = np.log2(p_ij / (p_i * p_j))
        if pmi >= PMI_THRESHOLD:
            pmi_edges.append((i, j, float(pmi), count))

    print(f"  PMI edges (>{PMI_THRESHOLD}): {len(pmi_edges)}")
    del coact_matrix

    # Leiden clustering
    print("  Leiden clustering...")
    modules = []
    if pmi_edges:
        import igraph as ig
        import leidenalg

        nodes_in_graph = set()
        for i, j, p, c in pmi_edges:
            nodes_in_graph.add(i)
            nodes_in_graph.add(j)
        node_list = sorted(nodes_in_graph)
        node_to_idx = {n: idx for idx, n in enumerate(node_list)}

        g = ig.Graph(n=len(node_list), directed=False)
        g.add_edges([(node_to_idx[i], node_to_idx[j]) for i, j, p, c in pmi_edges])
        g.es['weight'] = [p for i, j, p, c in pmi_edges]

        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=1.0,
            n_iterations=-1, seed=42)

        labels = np.array(partition.membership)
        for comm_id in range(len(partition)):
            member_local = np.where(labels == comm_id)[0]
            if len(member_local) < 3:
                continue
            members_global = [node_list[i] for i in member_local]
            member_set = set(members_global)
            internal_pmi = [p for i, j, p, c in pmi_edges if i in member_set and j in member_set]
            modules.append({
                'module_id': len(modules),
                'n_features': len(members_global),
                'features': sorted(members_global),
                'n_internal_edges': len(internal_pmi),
                'mean_pmi': float(np.mean(internal_pmi)) if internal_pmi else 0,
            })
        modules.sort(key=lambda m: -m['n_features'])

    # Annotate modules
    ann_path = os.path.join(run_dir, "feature_annotations.json")
    feature_annotations = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            feature_annotations = json.load(f).get('feature_annotations', {})

    catalog_path = os.path.join(run_dir, "feature_catalog.json")
    feature_genes = {}
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog = json.load(f)
        for feat in catalog['features']:
            if feat.get('top_genes'):
                feature_genes[feat['feature_idx']] = [g['gene_name'] for g in feat['top_genes'][:10]]

    for module in modules:
        ont_counts = defaultdict(lambda: defaultdict(int))
        all_genes = set()
        for fi in module['features']:
            for ann in feature_annotations.get(str(fi), []):
                ont_counts[ann['ontology']][ann['term']] += 1
            all_genes.update(feature_genes.get(fi, []))

        module_annotations = {}
        for ont, terms in ont_counts.items():
            top_terms = sorted(terms.items(), key=lambda x: -x[1])[:5]
            module_annotations[ont] = [{'term': t, 'n_features': c} for t, c in top_terms]

        module['annotations'] = module_annotations
        module['n_annotated_features'] = sum(
            1 for fi in module['features'] if feature_annotations.get(str(fi), []))
        module['n_unique_genes'] = len(all_genes)
        module['top_genes'] = sorted(all_genes)[:20]

    # Summary
    n_features_in_modules = sum(m['n_features'] for m in modules)
    feature_count_alive = int(np.sum(feature_count > 0))

    print(f"  Modules: {len(modules)}, Features in modules: {n_features_in_modules}/{feature_count_alive}")

    output = {
        'layer': layer,
        'summary': {
            'n_positions': n_total,
            'n_alive_features': feature_count_alive,
            'n_pmi_edges': len(pmi_edges),
            'n_modules': len(modules),
            'n_features_in_modules': n_features_in_modules,
            'module_coverage': n_features_in_modules / max(feature_count_alive, 1),
        },
        'modules': modules,
        'top_pmi_edges': sorted(pmi_edges, key=lambda e: -e[2])[:100],
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"  Layer {layer} done in {(time.time()-t0)/60:.1f} min")

    del activations, sae
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    total_t0 = time.time()
    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 7: CO-ACTIVATION")
    print("=" * 70)

    if args.all:
        layers = list(range(N_LAYERS))
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(N_LAYERS))

    for layer in layers:
        process_layer(layer)

    print(f"\nAll done in {(time.time()-total_t0)/60:.1f} min")


if __name__ == '__main__':
    main()
