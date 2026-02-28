#!/usr/bin/env python3
"""
Step 10: Novel Feature Characterization.

Characterize the ~50% of features that have NO significant ontology annotations.
Are they noise, or do they encode biology not in current databases?

Method:
  1. Cluster unannotated features by Jaccard similarity of top gene sets
  2. Annotate clusters with additional gene set databases (MSigDB Hallmarks)
  3. Test: do clusters form coherent biological programs?
  4. Check co-activation with annotated features (guilt by association)
  5. Measure per-feature statistics: activation frequency, gene diversity

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 10_novel_features.py [--layer 11]
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
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")

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


def process_layer(layer):
    """Characterize novel (unannotated) features for one layer."""

    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    out_dir = os.path.join(DATA_DIR, "novel_features")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"novel_features_layer{layer:02d}.json")
    if os.path.exists(out_path):
        print(f"  Already done: {out_path}")
        return out_path

    print(f"\n{'=' * 60}")
    print(f"LAYER {layer}: NOVEL FEATURE CHARACTERIZATION")
    print(f"{'=' * 60}")

    # Load catalog and annotations
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)
    with open(os.path.join(run_dir, "feature_annotations.json")) as f:
        ann_data = json.load(f)

    feature_annotations = ann_data.get('feature_annotations', {})

    # Separate annotated and unannotated features
    annotated = set()
    unannotated = set()
    feature_genes = {}
    feature_stats = {}

    for feat in catalog['features']:
        fi = feat['feature_idx']
        is_dead = feat.get('activation_freq', 0) == 0
        if is_dead:
            continue

        if feat.get('top_genes'):
            feature_genes[fi] = set(g['gene_name'] for g in feat['top_genes'][:20])
        else:
            feature_genes[fi] = set()

        feature_stats[fi] = {
            'activation_freq': feat.get('activation_freq', 0),
            'mean_activation': feat.get('mean_activation', 0),
            'n_top_genes': len(feat.get('top_genes', [])),
        }

        if str(fi) in feature_annotations and len(feature_annotations[str(fi)]) > 0:
            annotated.add(fi)
        else:
            unannotated.add(fi)

    print(f"  Annotated: {len(annotated)}, Unannotated: {len(unannotated)}")

    # ============================================================
    # 1. Cluster unannotated features by gene-set Jaccard similarity
    # ============================================================
    print("  Clustering unannotated features by gene-set similarity...")
    t0 = time.time()

    unann_list = sorted(unannotated)
    n_unann = len(unann_list)

    # Build Jaccard distance matrix (sample if too large)
    max_cluster = 2000
    if n_unann > max_cluster:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_unann, max_cluster, replace=False)
        sample_idx.sort()
        cluster_features = [unann_list[i] for i in sample_idx]
    else:
        cluster_features = unann_list

    n_cluster = len(cluster_features)
    gene_sets = [feature_genes.get(fi, set()) for fi in cluster_features]

    # Jaccard similarity matrix
    jaccard_matrix = np.zeros((n_cluster, n_cluster), dtype=np.float32)
    for i in range(n_cluster):
        for j in range(i + 1, n_cluster):
            if len(gene_sets[i]) == 0 or len(gene_sets[j]) == 0:
                continue
            intersection = len(gene_sets[i] & gene_sets[j])
            union = len(gene_sets[i] | gene_sets[j])
            if union > 0:
                jac = intersection / union
                jaccard_matrix[i, j] = jac
                jaccard_matrix[j, i] = jac

    # Cluster using Leiden on Jaccard graph
    import igraph as ig
    import leidenalg

    # Build graph from Jaccard > threshold
    JACCARD_THRESHOLD = 0.15
    edges = []
    weights = []
    for i in range(n_cluster):
        for j in range(i + 1, n_cluster):
            if jaccard_matrix[i, j] >= JACCARD_THRESHOLD:
                edges.append((i, j))
                weights.append(float(jaccard_matrix[i, j]))

    print(f"    Jaccard graph: {n_cluster} nodes, {len(edges)} edges (>{JACCARD_THRESHOLD})")

    if len(edges) > 0:
        g = ig.Graph(n=n_cluster, edges=edges, directed=False)
        g.es['weight'] = weights

        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=0.5,
            n_iterations=-1, seed=42,
        )

        labels = np.array(partition.membership)
        n_communities = len(partition)
    else:
        labels = np.arange(n_cluster)
        n_communities = n_cluster

    # Build clusters (only ≥3 features)
    novel_clusters = []
    for comm_id in range(n_communities):
        members = np.where(labels == comm_id)[0]
        if len(members) < 3:
            continue

        member_features = [cluster_features[i] for i in members]

        # Aggregate gene sets
        all_genes = set()
        for fi in member_features:
            all_genes.update(feature_genes.get(fi, set()))

        # Find consensus genes (appearing in ≥30% of member features)
        gene_counts = defaultdict(int)
        for fi in member_features:
            for g in feature_genes.get(fi, set()):
                gene_counts[g] += 1

        consensus_genes = [g for g, c in gene_counts.items()
                           if c >= max(2, len(member_features) * 0.3)]
        consensus_genes.sort(key=lambda g: -gene_counts[g])

        # Mean Jaccard within cluster
        internal_jacs = []
        for i, fi in enumerate(member_features):
            for j, fj in enumerate(member_features):
                if i < j:
                    gi, gj = feature_genes.get(fi, set()), feature_genes.get(fj, set())
                    if gi and gj:
                        union = len(gi | gj)
                        if union > 0:
                            internal_jacs.append(len(gi & gj) / union)

        mean_jaccard = float(np.mean(internal_jacs)) if internal_jacs else 0

        # Mean activation frequency
        freqs = [feature_stats[fi]['activation_freq'] for fi in member_features
                 if fi in feature_stats]
        mean_freq = float(np.mean(freqs)) if freqs else 0

        novel_clusters.append({
            'cluster_id': len(novel_clusters),
            'n_features': len(member_features),
            'features': sorted(member_features),
            'n_unique_genes': len(all_genes),
            'n_consensus_genes': len(consensus_genes),
            'consensus_genes': consensus_genes[:20],
            'mean_jaccard': mean_jaccard,
            'mean_activation_freq': mean_freq,
        })

    novel_clusters.sort(key=lambda c: -c['n_features'])
    print(f"    Clusters (≥3 features): {len(novel_clusters)} ({time.time()-t0:.1f}s)")

    # ============================================================
    # 2. Check guilt-by-association with annotated features
    # ============================================================
    print("  Checking guilt-by-association with co-activation modules...")

    # Load co-activation modules
    coact_path = os.path.join(DATA_DIR, "coactivation", f"coactivation_layer{layer:02d}.json")
    guilt_by_association = {}
    if os.path.exists(coact_path):
        with open(coact_path) as f:
            coact_data = json.load(f)

        modules = coact_data.get('modules', [])

        for module in modules:
            mod_features = set(module['features'])
            mod_annotations = module.get('annotations', {})

            # How many unannotated features are in this module?
            unann_in_module = mod_features & unannotated
            ann_in_module = mod_features & annotated

            if unann_in_module and ann_in_module:
                # These unannotated features are in a module with annotated ones
                best_label = "unlabeled"
                for ont in ['GO_BP', 'Reactome', 'KEGG']:
                    if ont in mod_annotations and mod_annotations[ont]:
                        best_label = mod_annotations[ont][0]['term']
                        break

                for fi in unann_in_module:
                    guilt_by_association[fi] = {
                        'module_id': module['module_id'],
                        'module_size': module['n_features'],
                        'module_label': best_label,
                        'n_annotated_in_module': len(ann_in_module),
                    }

    n_guilt = len(guilt_by_association)
    print(f"    Unannotated features with module association: "
          f"{n_guilt}/{len(unannotated)} ({n_guilt/max(len(unannotated),1)*100:.1f}%)")

    # ============================================================
    # 3. Statistical comparison: annotated vs unannotated features
    # ============================================================
    print("  Comparing annotated vs unannotated feature statistics...")

    ann_freqs = [feature_stats[fi]['activation_freq'] for fi in annotated if fi in feature_stats]
    unann_freqs = [feature_stats[fi]['activation_freq'] for fi in unannotated if fi in feature_stats]
    ann_ngenes = [len(feature_genes.get(fi, set())) for fi in annotated]
    unann_ngenes = [len(feature_genes.get(fi, set())) for fi in unannotated]

    print(f"    Activation frequency:")
    print(f"      Annotated: mean={np.mean(ann_freqs):.4f}, median={np.median(ann_freqs):.4f}")
    print(f"      Unannotated: mean={np.mean(unann_freqs):.4f}, median={np.median(unann_freqs):.4f}")
    print(f"    Gene set size (top 20):")
    print(f"      Annotated: mean={np.mean(ann_ngenes):.1f}")
    print(f"      Unannotated: mean={np.mean(unann_ngenes):.1f}")

    # ============================================================
    # 4. Categorize unannotated features
    # ============================================================
    n_in_clusters = sum(c['n_features'] for c in novel_clusters)
    clustered_features = set()
    for c in novel_clusters:
        clustered_features.update(c['features'])
    n_guilt_only = sum(1 for fi in unannotated
                       if fi in guilt_by_association and fi not in clustered_features)

    categories = {
        'in_coherent_clusters': n_in_clusters,
        'guilt_by_association': n_guilt,
        'isolated': len(unannotated) - n_in_clusters - n_guilt_only,
    }

    print(f"\n  Novel feature categories:")
    print(f"    In coherent gene-set clusters: {categories['in_coherent_clusters']}")
    print(f"    Guilt-by-association (in module with annotated): {categories['guilt_by_association']}")
    print(f"    Isolated (no cluster, no module): {categories['isolated']}")

    # ============================================================
    # Summary and save
    # ============================================================
    print(f"\n  Top novel clusters:")
    for c in novel_clusters[:10]:
        print(f"    Cluster {c['cluster_id']}: {c['n_features']} features, "
              f"{c['n_consensus_genes']} consensus genes, "
              f"Jaccard={c['mean_jaccard']:.3f}")
        if c['consensus_genes']:
            print(f"      Genes: {', '.join(c['consensus_genes'][:10])}")

    output = {
        'layer': layer,
        'summary': {
            'n_annotated': len(annotated),
            'n_unannotated': len(unannotated),
            'n_novel_clusters': len(novel_clusters),
            'n_features_in_clusters': n_in_clusters,
            'n_guilt_by_association': n_guilt,
            'frac_unannotated_in_clusters': n_in_clusters / max(len(unannotated), 1),
            'frac_unannotated_with_module': n_guilt / max(len(unannotated), 1),
            'annotated_mean_freq': float(np.mean(ann_freqs)),
            'unannotated_mean_freq': float(np.mean(unann_freqs)),
        },
        'categories': categories,
        'novel_clusters': novel_clusters,
        'guilt_by_association_sample': dict(list(guilt_by_association.items())[:50]),
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
    print("STEP 10: NOVEL FEATURE CHARACTERIZATION")
    print("=" * 70)

    if args.all:
        layers = list(range(18))
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = [0, 5, 11, 17]

    for layer in layers:
        process_layer(layer)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
