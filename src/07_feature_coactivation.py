#!/usr/bin/env python3
"""
Step 7: Feature co-activation graph.

For each layer, compute which SAE features co-activate on the same gene positions.
Build a sparse co-activation graph, compute PMI (pointwise mutual information),
cluster into modules, and annotate modules with aggregated biology.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 07_feature_coactivation.py [--layer 0] [--all]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict
from scipy import sparse
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")

EXPANSION = 4
K_VAL = 32
BATCH_SIZE = 8192
PMI_THRESHOLD = 2.0      # Only keep edges with PMI > threshold
MIN_COACTIVATION = 50     # Minimum co-activation count
N_LAYERS = 18


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def process_layer(layer):
    """Build co-activation graph for one layer."""
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from sae_model import TopKSAE

    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    out_dir = os.path.join(DATA_DIR, "coactivation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"LAYER {layer}: FEATURE CO-ACTIVATION")
    print(f"{'=' * 60}")

    # Check if already done
    out_path = os.path.join(out_dir, f"coactivation_layer{layer:02d}.json")
    if os.path.exists(out_path):
        print(f"  Already done: {out_path}")
        return out_path

    # Load SAE
    t0 = time.time()
    sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
    sae.eval()
    n_features = sae.n_features
    print(f"  SAE loaded: {n_features} features ({time.time()-t0:.1f}s)")

    # Load activation mean for centering
    act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)

    # Load activations
    act_path = os.path.join(DATA_DIR, f"layer_{layer:02d}_activations.npy")
    activations = np.lib.format.open_memmap(act_path, mode='r')
    n_total = activations.shape[0]
    print(f"  Activations: {n_total:,} positions")

    # ============================================================
    # Stream through activations, accumulate co-activation counts
    # ============================================================
    print("  Computing co-activations...")
    t0 = time.time()

    # For memory efficiency, use sparse accumulation
    # feature_count[i] = number of times feature i fires
    feature_count = np.zeros(n_features, dtype=np.int64)

    # Co-activation: accumulate into a sparse matrix using coordinate lists
    # We'll use a dense (n_features, n_features) int32 matrix since k=32
    # means at most k*(k-1)/2 = 496 pairs per input, and with 4M inputs
    # the counts can get large but the matrix is only 4608^2 * 4 bytes = 81 MB
    coact_matrix = np.zeros((n_features, n_features), dtype=np.int32)

    n_processed = 0
    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch = torch.tensor(activations[start:end], dtype=torch.float32)
        batch = batch - act_mean_t

        with torch.no_grad():
            h_sparse, topk_indices = sae.encode(batch)
            # topk_indices: (batch_size, k) — which features are active

        indices_np = topk_indices.numpy()  # (batch_size, k)

        # Vectorized feature counting
        for fi in indices_np.ravel():
            feature_count[fi] += 1

        # Vectorized co-activation: for each row, accumulate all k*(k-1)/2 pairs
        # Use numpy outer indexing
        for row in indices_np:
            row_sorted = np.sort(row)
            # Upper triangle pairs
            ii, jj = np.triu_indices(len(row_sorted), k=1)
            coact_matrix[row_sorted[ii], row_sorted[jj]] += 1

        n_processed += (end - start)
        if n_processed % (BATCH_SIZE * 50) == 0:
            n_nonzero = np.count_nonzero(coact_matrix)
            print(f"    {n_processed:>10,}/{n_total:,} | "
                  f"{n_nonzero:,} nonzero pairs")

    elapsed = time.time() - t0
    n_nonzero = np.count_nonzero(coact_matrix)
    print(f"  Encoding done: {elapsed:.1f}s, {n_nonzero:,} nonzero pairs")

    # ============================================================
    # Compute PMI for each pair
    # ============================================================
    print("  Computing PMI...")
    t0 = time.time()

    # P(i) = feature_count[i] / n_total
    # P(i,j) = coact_matrix[i,j] / n_total
    # PMI(i,j) = log2(P(i,j) / (P(i) * P(j)))

    # Find all pairs above min coactivation threshold
    rows, cols = np.where(coact_matrix >= MIN_COACTIVATION)
    # Only upper triangle
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

    print(f"  PMI edges (>{PMI_THRESHOLD}): {len(pmi_edges)} ({time.time()-t0:.1f}s)")

    del coact_matrix  # Free 81 MB

    # ============================================================
    # Build graph and cluster with Leiden algorithm
    # ============================================================
    print("  Clustering with Leiden algorithm...")
    t0 = time.time()

    if len(pmi_edges) == 0:
        print("  WARNING: No PMI edges found. Skipping clustering.")
        modules = []
    else:
        import igraph as ig
        import leidenalg

        # Build igraph graph from PMI edges
        # Map feature indices to contiguous node IDs
        nodes_in_graph = set()
        for i, j, p, c in pmi_edges:
            nodes_in_graph.add(i)
            nodes_in_graph.add(j)
        node_list = sorted(nodes_in_graph)
        node_to_idx = {n: idx for idx, n in enumerate(node_list)}

        g = ig.Graph(n=len(node_list), directed=False)
        edge_list = [(node_to_idx[i], node_to_idx[j]) for i, j, p, c in pmi_edges]
        weights = [p for i, j, p, c in pmi_edges]
        g.add_edges(edge_list)
        g.es['weight'] = weights

        print(f"    Graph: {g.vcount()} nodes, {g.ecount()} edges")

        # Leiden clustering (modularity optimization with resolution parameter)
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=1.0,
            n_iterations=-1,  # iterate until stable
            seed=42,
        )
        n_communities = len(partition)
        print(f"    Leiden communities: {n_communities}")

        # Build modules from communities (only ≥3 features)
        labels = np.array(partition.membership)
        modules = []
        for comm_id in range(n_communities):
            member_local = np.where(labels == comm_id)[0]
            if len(member_local) < 3:
                continue

            members_global = [node_list[i] for i in member_local]

            # Internal edges and mean PMI
            member_set = set(members_global)
            internal_pmi = [p for i, j, p, c in pmi_edges
                            if i in member_set and j in member_set]
            mean_pmi = np.mean(internal_pmi) if internal_pmi else 0

            modules.append({
                'module_id': len(modules),
                'n_features': len(members_global),
                'features': sorted(members_global),
                'n_internal_edges': len(internal_pmi),
                'mean_pmi': float(mean_pmi),
            })

        modules.sort(key=lambda m: -m['n_features'])
        print(f"    Modules (≥3 features): {len(modules)} ({time.time()-t0:.1f}s)")

    # ============================================================
    # Annotate modules using feature annotations
    # ============================================================
    print("  Annotating modules...")

    ann_path = os.path.join(run_dir, "feature_annotations.json")
    feature_annotations = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            ann_data = json.load(f)
        feature_annotations = ann_data.get('feature_annotations', {})

    # Also load top genes from catalog
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)
    feature_genes = {}
    for feat in catalog['features']:
        if feat.get('top_genes'):
            feature_genes[feat['feature_idx']] = [g['gene_name'] for g in feat['top_genes'][:10]]

    for module in modules:
        # Aggregate annotations
        ont_counts = defaultdict(lambda: defaultdict(int))
        all_genes = set()
        for fi in module['features']:
            anns = feature_annotations.get(str(fi), [])
            for ann in anns:
                ont_counts[ann['ontology']][ann['term']] += 1
            genes = feature_genes.get(fi, [])
            all_genes.update(genes)

        # Top terms per ontology
        module_annotations = {}
        for ont, terms in ont_counts.items():
            top_terms = sorted(terms.items(), key=lambda x: -x[1])[:5]
            module_annotations[ont] = [{'term': t, 'n_features': c} for t, c in top_terms]

        module['annotations'] = module_annotations
        module['n_annotated_features'] = sum(
            1 for fi in module['features']
            if len(feature_annotations.get(str(fi), [])) > 0
        )
        module['n_unique_genes'] = len(all_genes)
        module['top_genes'] = sorted(all_genes)[:20]

    # ============================================================
    # Summary statistics
    # ============================================================
    n_features_in_modules = sum(m['n_features'] for m in modules)
    feature_count_alive = np.sum(feature_count > 0)

    print(f"\n  Summary:")
    print(f"    Alive features: {feature_count_alive}")
    print(f"    Features in modules: {n_features_in_modules} "
          f"({n_features_in_modules/max(feature_count_alive,1)*100:.1f}%)")
    print(f"    Modules: {len(modules)}")
    if modules:
        sizes = [m['n_features'] for m in modules]
        print(f"    Module sizes: min={min(sizes)}, median={int(np.median(sizes))}, "
              f"max={max(sizes)}, mean={np.mean(sizes):.1f}")

    # Top 10 modules
    if modules:
        print(f"\n  Top modules:")
        for m in modules[:10]:
            anns = m.get('annotations', {})
            # Best label
            best_label = "unlabeled"
            for ont in ['GO_BP', 'Reactome', 'KEGG']:
                if ont in anns and anns[ont]:
                    best_label = anns[ont][0]['term']
                    break
            print(f"    Module {m['module_id']:>3d}: {m['n_features']:>4d} features, "
                  f"{m['n_annotated_features']} annotated, "
                  f"PMI={m['mean_pmi']:.2f} | {best_label[:60]}")

    # ============================================================
    # Save
    # ============================================================
    output = {
        'layer': layer,
        'config': {
            'pmi_threshold': PMI_THRESHOLD,
            'min_coactivation': MIN_COACTIVATION,
            'k': K_VAL,
            'batch_size': BATCH_SIZE,
        },
        'summary': {
            'n_positions': n_total,
            'n_alive_features': int(feature_count_alive),
            'n_pmi_edges': len(pmi_edges),
            'n_modules': len(modules),
            'n_features_in_modules': n_features_in_modules,
            'module_coverage': n_features_in_modules / max(feature_count_alive, 1),
        },
        'modules': modules,
        'top_pmi_edges': sorted(pmi_edges, key=lambda e: -e[2])[:100],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Saved: {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    total_t0 = time.time()

    print("=" * 70)
    print("STEP 7: FEATURE CO-ACTIVATION GRAPH")
    print("=" * 70)

    if args.all:
        layers = list(range(N_LAYERS))
    elif args.layer is not None:
        layers = [args.layer]
    else:
        # Default: representative subset
        layers = [0, 5, 11, 17]

    for layer in layers:
        process_layer(layer)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
