#!/usr/bin/env python3
"""
Phase 3, Script 3: Analyze and annotate multi-tissue SAE features.

For each of 4 layers (0, 5, 11, 17):
  1. Encode FULL pooled activations (K562 + TS) through SAE
  2. Build feature catalog: top genes, activation freq, mean activation
  3. Run enrichment tests against GO BP, KEGG, Reactome, STRING, TRRUST
  4. Save feature_catalog.json and feature_annotations.json

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 12c_analyze_and_annotate.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_model import TopKSAE

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")

K562_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
TS_DIR = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/ts_activations")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/phase3_multitissue/sae_models")

ONTOLOGY_DIR = os.path.join(BASE, "biodyn-nmi-paper/results/biological_impact/reference_edge_sets")
TRRUST_PATH = os.path.join(BASE, "biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv")

HIDDEN_DIM = 1152
EXPANSION = 4
K_VAL = 32
TARGET_LAYERS = [0, 5, 11, 17]
TOP_N_GENES = 20
MIN_OVERLAP = 2
MIN_TERM_SIZE = 5
MAX_TERM_SIZE = 500
BH_ALPHA = 0.05


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_ontologies():
    """Load all biological ontologies."""
    ontologies = {}

    go_bp_path = os.path.join(ONTOLOGY_DIR, "go_bp_gene_sets.json")
    if os.path.exists(go_bp_path):
        with open(go_bp_path) as f:
            raw = json.load(f)
        ontologies['GO_BP'] = {k: set(v) for k, v in raw.items()
                               if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
        print(f"  GO BP: {len(ontologies['GO_BP'])} terms")

    kegg_path = os.path.join(ONTOLOGY_DIR, "kegg_gene_sets.json")
    if os.path.exists(kegg_path):
        with open(kegg_path) as f:
            raw = json.load(f)
        ontologies['KEGG'] = {k: set(v) for k, v in raw.items()
                              if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
        print(f"  KEGG: {len(ontologies['KEGG'])} terms")

    reactome_path = os.path.join(ONTOLOGY_DIR, "reactome_gene_sets.json")
    if os.path.exists(reactome_path):
        with open(reactome_path) as f:
            raw = json.load(f)
        ontologies['Reactome'] = {k: set(v) for k, v in raw.items()
                                  if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
        print(f"  Reactome: {len(ontologies['Reactome'])} terms")

    string_path = os.path.join(ONTOLOGY_DIR, "string_ppi_edges.json")
    if os.path.exists(string_path):
        with open(string_path) as f:
            raw = json.load(f)
        string_edges = set()
        for tier in ['pairs_700', 'pairs_900']:
            if tier in raw:
                for pair in raw[tier]:
                    if len(pair) == 2:
                        a, b = pair[0].upper(), pair[1].upper()
                        string_edges.add((min(a, b), max(a, b)))
        ontologies['STRING_edges'] = string_edges
        print(f"  STRING PPI: {len(string_edges)} edges")

    if os.path.exists(TRRUST_PATH):
        df = pd.read_csv(TRRUST_PATH, sep='\t', header=None,
                         names=['tf', 'target', 'mode', 'pmid'])
        trrust_edges = set()
        trrust_tfs = set()
        for _, row in df.iterrows():
            tf = row['tf'].upper()
            tgt = row['target'].upper()
            trrust_edges.add((tf, tgt))
            trrust_tfs.add(tf)
        ontologies['TRRUST_edges'] = trrust_edges
        ontologies['TRRUST_tfs'] = trrust_tfs
        print(f"  TRRUST: {len(trrust_edges)} edges, {len(trrust_tfs)} TFs")

    return ontologies


def test_gene_set_enrichment(gene_set, term_genes, background_size):
    overlap = gene_set & term_genes
    if len(overlap) < MIN_OVERLAP:
        return None, 1.0, set()
    a = len(overlap)
    b = len(gene_set) - a
    c = len(term_genes) - a
    d = background_size - a - b - c
    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative='greater')
    return odds_ratio, p_value, overlap


def test_edge_enrichment(gene_set, edge_set):
    gene_list = list(gene_set)
    n = len(gene_list)
    if n < 2:
        return 0, 0, 0.0
    n_possible = n * (n - 1) // 2
    n_within = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gene_list[i].upper(), gene_list[j].upper()
            edge = (min(a, b), max(a, b))
            if edge in edge_set:
                n_within += 1
    density = n_within / max(n_possible, 1)
    return n_within, n_possible, density


def build_gene_name_mapping():
    """Combine gene name mappings from K562 and Tabula Sapiens."""
    token_to_gene = {}

    # K562 mapping
    k562_map_path = os.path.join(K562_DIR, "token_id_to_gene_name.json")
    if os.path.exists(k562_map_path):
        with open(k562_map_path) as f:
            k562_map = json.load(f)
        for k, v in k562_map.items():
            token_to_gene[int(k)] = v

    # TS mapping
    ts_map_path = os.path.join(TS_DIR, "token_id_to_gene_name.json")
    if os.path.exists(ts_map_path):
        with open(ts_map_path) as f:
            ts_map = json.load(f)
        for k, v in ts_map.items():
            if int(k) not in token_to_gene:
                token_to_gene[int(k)] = v

    return token_to_gene


def analyze_layer(layer, sae, act_mean, token_to_gene, n_features):
    """Build feature catalog by encoding K562 + TS activations."""
    print(f"\n  Analyzing layer {layer}...")
    t0 = time.time()

    act_mean_np = act_mean

    # Set up accumulators
    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_sq_sum = np.zeros(n_features, dtype=np.float64)
    feature_count = np.zeros(n_features, dtype=np.int64)
    feature_n = 0
    gene_feature_sum = defaultdict(lambda: defaultdict(float))
    gene_feature_count = defaultdict(lambda: defaultdict(int))

    chunk_size = 10000

    # Process K562 activations
    k562_act_path = os.path.join(K562_DIR, f"layer_{layer:02d}_activations.npy")
    k562_gid_path = os.path.join(K562_DIR, f"layer_{layer:02d}_gene_ids.npy")
    k562_act = np.load(k562_act_path, mmap_mode='r')
    k562_gids = np.load(k562_gid_path, mmap_mode='r')
    n_k562 = len(k562_act)
    print(f"    Processing K562: {n_k562:,} positions...")

    for start in range(0, n_k562, chunk_size):
        end = min(start + chunk_size, n_k562)
        batch_act = k562_act[start:end].astype(np.float32) - act_mean_np[np.newaxis, :]
        batch_gids = k562_gids[start:end]

        batch_tensor = torch.tensor(batch_act, dtype=torch.float32)
        with torch.no_grad():
            h_sparse, _ = sae.encode(batch_tensor)
        h_np = h_sparse.numpy()

        active_mask = h_np > 0
        feature_sum += h_np.sum(axis=0)
        feature_sq_sum += (h_np ** 2).sum(axis=0)
        feature_count += active_mask.sum(axis=0)
        feature_n += len(batch_act)

        for i in range(len(batch_act)):
            gid = int(batch_gids[i])
            active_features = np.where(h_np[i] > 0)[0]
            for fi in active_features:
                gene_feature_sum[fi][gid] += h_np[i, fi]
                gene_feature_count[fi][gid] += 1

        if end % 500000 < chunk_size:
            print(f"      K562: {end:,}/{n_k562:,}")

    del k562_act, k562_gids

    # Process TS activations
    ts_act_path = os.path.join(TS_DIR, f"layer_{layer:02d}_activations.npy")
    ts_gid_path = os.path.join(TS_DIR, f"layer_{layer:02d}_gene_ids.npy")
    ts_act = np.load(ts_act_path, mmap_mode='r')
    ts_gids = np.load(ts_gid_path, mmap_mode='r')
    n_ts = len(ts_act)
    print(f"    Processing TS: {n_ts:,} positions...")

    for start in range(0, n_ts, chunk_size):
        end = min(start + chunk_size, n_ts)
        batch_act = ts_act[start:end].astype(np.float32) - act_mean_np[np.newaxis, :]
        batch_gids = ts_gids[start:end]

        batch_tensor = torch.tensor(batch_act, dtype=torch.float32)
        with torch.no_grad():
            h_sparse, _ = sae.encode(batch_tensor)
        h_np = h_sparse.numpy()

        active_mask = h_np > 0
        feature_sum += h_np.sum(axis=0)
        feature_sq_sum += (h_np ** 2).sum(axis=0)
        feature_count += active_mask.sum(axis=0)
        feature_n += len(batch_act)

        for i in range(len(batch_act)):
            gid = int(batch_gids[i])
            active_features = np.where(h_np[i] > 0)[0]
            for fi in active_features:
                gene_feature_sum[fi][gid] += h_np[i, fi]
                gene_feature_count[fi][gid] += 1

        if end % 500000 < chunk_size:
            print(f"      TS: {end:,}/{n_ts:,}")

    del ts_act, ts_gids

    # Build catalog
    feature_mean_act = feature_sum / max(feature_n, 1)
    feature_freq = feature_count / max(feature_n, 1)
    dead_mask = feature_count == 0

    n_alive = int((~dead_mask).sum())
    print(f"    Alive: {n_alive}/{n_features}, Dead: {dead_mask.sum()}")
    print(f"    Total positions encoded: {feature_n:,}")
    print(f"    Analysis time: {time.time()-t0:.1f}s")

    catalog = []
    for fi in range(n_features):
        entry = {
            'feature_idx': fi,
            'is_dead': bool(dead_mask[fi]),
            'activation_freq': float(feature_freq[fi]),
            'mean_activation': float(feature_mean_act[fi]),
            'fire_count': int(feature_count[fi]),
        }

        if not dead_mask[fi]:
            gene_acts = gene_feature_sum[fi]
            gene_counts = gene_feature_count[fi]
            gene_mean_acts = {}
            for gid in gene_acts:
                if gene_counts[gid] > 0:
                    gene_mean_acts[gid] = gene_acts[gid] / gene_counts[gid]

            sorted_genes = sorted(gene_mean_acts.items(), key=lambda x: -x[1])
            top_genes = []
            for gid, act in sorted_genes[:TOP_N_GENES]:
                gene_name = token_to_gene.get(gid, f"token_{gid}")
                top_genes.append({
                    'gene_id': gid,
                    'gene_name': gene_name,
                    'mean_activation': float(act),
                    'fire_count': int(gene_counts[gid]),
                })
            entry['top_genes'] = top_genes
            entry['n_unique_genes'] = len(gene_acts)
        else:
            entry['top_genes'] = []
            entry['n_unique_genes'] = 0

        catalog.append(entry)

    return catalog, n_alive, dead_mask


def annotate_features(catalog, n_alive, ontologies, run_dir):
    """Run enrichment tests on feature catalog."""
    print(f"    Running enrichment tests...")
    t0 = time.time()

    all_genes = set()
    alive_features = [f for f in catalog if not f['is_dead']]
    for feat in alive_features:
        for g in feat.get('top_genes', []):
            all_genes.add(g['gene_name'].upper())
    background_size = len(all_genes)

    all_pvals = []
    for fi, feat in enumerate(alive_features):
        top_genes = set(g['gene_name'].upper() for g in feat.get('top_genes', []))
        if len(top_genes) < 3:
            continue
        feature_idx = feat['feature_idx']

        for ont_name in ['GO_BP', 'KEGG', 'Reactome']:
            if ont_name not in ontologies:
                continue
            for term, term_genes in ontologies[ont_name].items():
                term_genes_upper = set(g.upper() for g in term_genes)
                odds_ratio, pval, overlap = test_gene_set_enrichment(
                    top_genes, term_genes_upper, background_size)
                if odds_ratio is not None:
                    all_pvals.append({
                        'feature_idx': feature_idx,
                        'ontology': ont_name,
                        'term': term,
                        'p_value': pval,
                        'odds_ratio': odds_ratio,
                        'overlap': list(overlap),
                        'n_overlap': len(overlap),
                        'term_size': len(term_genes),
                    })

        for edge_ont in ['STRING_edges', 'TRRUST_edges']:
            if edge_ont not in ontologies:
                continue
            n_within, n_possible, density = test_edge_enrichment(
                top_genes, ontologies[edge_ont])
            if n_within > 0:
                all_pvals.append({
                    'feature_idx': feature_idx,
                    'ontology': edge_ont,
                    'term': 'pairwise_enrichment',
                    'n_edges_within': n_within,
                    'n_possible': n_possible,
                    'density': density,
                })

        if 'TRRUST_tfs' in ontologies:
            tf_overlap = top_genes & ontologies['TRRUST_tfs']
            if len(tf_overlap) >= MIN_OVERLAP:
                n_tf_bg = len(ontologies['TRRUST_tfs'] & all_genes)
                odds_ratio, pval, _ = test_gene_set_enrichment(
                    top_genes, ontologies['TRRUST_tfs'] & all_genes, background_size)
                if odds_ratio is not None:
                    all_pvals.append({
                        'feature_idx': feature_idx,
                        'ontology': 'TRRUST_TF_enrichment',
                        'term': 'transcription_factors',
                        'p_value': pval,
                        'odds_ratio': odds_ratio,
                        'overlap': list(tf_overlap),
                        'n_overlap': len(tf_overlap),
                        'term_size': n_tf_bg,
                    })

        if (fi + 1) % 500 == 0:
            print(f"      Feature {fi+1}/{len(alive_features)}...")

    print(f"    Total tests: {len(all_pvals)}")

    # BH correction
    pval_tests = [t for t in all_pvals if 'p_value' in t]
    edge_tests = [t for t in all_pvals if 'p_value' not in t]

    n_significant = 0
    if pval_tests:
        raw_pvals = np.array([t['p_value'] for t in pval_tests])
        reject, corrected, _, _ = multipletests(raw_pvals, alpha=BH_ALPHA, method='fdr_bh')
        for i, test in enumerate(pval_tests):
            test['p_adjusted'] = float(corrected[i])
            test['significant'] = bool(reject[i])
        n_significant = int(reject.sum())

    # Build per-feature annotations
    feature_annotations = defaultdict(list)
    for test in pval_tests:
        if test.get('significant', False):
            feature_annotations[test['feature_idx']].append({
                'ontology': test['ontology'],
                'term': test['term'],
                'p_adjusted': test['p_adjusted'],
                'odds_ratio': test['odds_ratio'],
                'n_overlap': test['n_overlap'],
                'overlap_genes': test.get('overlap', []),
            })

    for test in edge_tests:
        if test.get('n_edges_within', 0) > 0:
            feature_annotations[test['feature_idx']].append({
                'ontology': test['ontology'],
                'term': test['term'],
                'n_edges': test['n_edges_within'],
                'density': test['density'],
            })

    n_annotated = len(feature_annotations)
    print(f"    Annotated: {n_annotated}/{n_alive} ({100*n_annotated/max(n_alive,1):.1f}%)")
    print(f"    Significant tests: {n_significant}")
    print(f"    Annotation time: {time.time()-t0:.1f}s")

    # Count by ontology
    ont_counts = defaultdict(int)
    for fi, anns in feature_annotations.items():
        for ann in anns:
            ont_counts[ann['ontology']] += 1
    for ont, count in sorted(ont_counts.items(), key=lambda x: -x[1]):
        print(f"      {ont}: {count}")

    return feature_annotations, n_annotated, n_significant, pval_tests, edge_tests, ont_counts


def main():
    n_features = EXPANSION * HIDDEN_DIM

    print("=" * 70)
    print("PHASE 3: ANALYZE AND ANNOTATE MULTI-TISSUE SAE FEATURES")
    print(f"  Layers: {TARGET_LAYERS}")
    print("=" * 70)

    total_t0 = time.time()

    # Load gene name mapping
    print("\nLoading gene name mappings...")
    token_to_gene = build_gene_name_mapping()
    print(f"  Combined mapping: {len(token_to_gene)} tokens")

    # Load ontologies
    print("\nLoading ontologies...")
    ontologies = load_ontologies()

    for layer in TARGET_LAYERS:
        run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
        run_dir = os.path.join(SAE_BASE, run_name)

        catalog_path = os.path.join(run_dir, "feature_catalog.json")
        ann_path = os.path.join(run_dir, "feature_annotations.json")

        if os.path.exists(ann_path):
            print(f"\n  Layer {layer}: already annotated, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"  LAYER {layer}")
        print(f"{'=' * 60}")

        # Load SAE
        model_path = os.path.join(run_dir, "sae_final.pt")
        if not os.path.exists(model_path):
            print(f"  SAE not found: {model_path}")
            print(f"  Run 12b_pool_and_train.py first.")
            continue

        sae = TopKSAE.load(model_path, device='cpu')
        sae.eval()

        act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))

        # Analyze
        catalog, n_alive, dead_mask = analyze_layer(
            layer, sae, act_mean, token_to_gene, n_features)

        # Save catalog
        catalog_output = {
            'summary': {
                'run_name': run_name,
                'layer': layer,
                'n_features': n_features,
                'k': K_VAL,
                'n_alive': n_alive,
                'n_dead': int(dead_mask.sum()),
            },
            'features': catalog,
        }
        with open(catalog_path, 'w') as f:
            json.dump(catalog_output, f, indent=2, default=_json_default)
        print(f"    Catalog saved: {catalog_path}")

        # Annotate
        feature_annotations, n_annotated, n_significant, pval_tests, edge_tests, ont_counts = \
            annotate_features(catalog, n_alive, ontologies, run_dir)

        ann_output = {
            'summary': {
                'run_name': run_name,
                'layer': layer,
                'n_features': n_features,
                'n_alive': n_alive,
                'n_annotated': n_annotated,
                'annotation_rate': n_annotated / max(n_alive, 1),
                'n_significant_tests': n_significant,
                'total_tests': len(pval_tests),
                'bh_alpha': BH_ALPHA,
                'ontology_counts': dict(ont_counts),
            },
            'feature_annotations': {str(k): v for k, v in feature_annotations.items()},
            'all_significant_tests': [t for t in pval_tests if t.get('significant', False)],
            'edge_enrichments': edge_tests,
        }
        with open(ann_path, 'w') as f:
            json.dump(ann_output, f, indent=2, default=_json_default)
        print(f"    Annotations saved: {ann_path}")

        del sae

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
