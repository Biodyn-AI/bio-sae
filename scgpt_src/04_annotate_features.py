#!/usr/bin/env python3
"""
scGPT SAE Pipeline — Step 4: Annotate SAE features against biological ontologies.

Tests each alive feature's top-gene set against GO BP, KEGG, Reactome,
STRING PPI, TRRUST TF-targets. Fisher's exact test + BH correction.

Usage:
    ~/anaconda3/envs/bio_mech_interp/bin/python 04_annotate_features.py --layer 0
    ~/anaconda3/envs/bio_mech_interp/bin/python 04_annotate_features.py --all
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")

# Ontology paths (same as Geneformer pipeline)
ONTOLOGY_DIR = os.path.join(BASE, "biodyn-nmi-paper/results/biological_impact/reference_edge_sets")
TRRUST_PATH = os.path.join(BASE, "biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv")

TOP_N_GENES = 20
MIN_OVERLAP = 2
MIN_TERM_SIZE = 5
MAX_TERM_SIZE = 500
BH_ALPHA = 0.05
N_LAYERS = 12


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

    for name, fname in [('GO_BP', 'go_bp_gene_sets.json'),
                         ('KEGG', 'kegg_gene_sets.json'),
                         ('Reactome', 'reactome_gene_sets.json')]:
        path = os.path.join(ONTOLOGY_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            ontologies[name] = {k: set(v) for k, v in raw.items()
                                if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
            print(f"    {name}: {len(ontologies[name])} terms")

    # STRING PPI
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
        print(f"    STRING PPI: {len(string_edges)} edges")

    # TRRUST
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
        print(f"    TRRUST: {len(trrust_edges)} edges, {len(trrust_tfs)} TFs")

    return ontologies


def test_gene_set_enrichment(gene_set, term_genes, background_size):
    overlap = gene_set & term_genes
    if len(overlap) < MIN_OVERLAP:
        return None, 1.0, set()
    a = len(overlap)
    b = len(gene_set) - a
    c = len(term_genes) - a
    d = background_size - a - b - c
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
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
            if (min(a, b), max(a, b)) in edge_set:
                n_within += 1
    return n_within, n_possible, n_within / max(n_possible, 1)


def annotate_layer(layer, ontologies, expansion=4, k=32):
    """Annotate features for a single layer."""
    run_name = f"layer{layer:02d}_x{expansion}_k{k}"
    run_dir = os.path.join(SAE_BASE, run_name)

    ann_path = os.path.join(run_dir, "feature_annotations.json")
    if os.path.exists(ann_path):
        print(f"\n  Layer {layer}: Annotations already exist, skipping.")
        return

    catalog_path = os.path.join(run_dir, "feature_catalog.json")
    if not os.path.exists(catalog_path):
        print(f"\n  Layer {layer}: Feature catalog not found, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Annotating Layer {layer}")
    print(f"{'=' * 60}")

    t0 = time.time()

    with open(catalog_path) as f:
        catalog_data = json.load(f)
    features = catalog_data['features']
    summary = catalog_data['summary']
    n_features = summary['n_features']
    n_alive = summary['n_alive']
    print(f"  Features: {n_features} total, {n_alive} alive")

    # Background genes
    all_genes = set()
    for feat in features:
        for g in feat.get('top_genes', []):
            all_genes.add(g['gene_name'].upper())
    background_size = len(all_genes)
    print(f"  Background: {background_size} genes")

    # Enrichment tests
    all_pvals = []
    alive_features = [f for f in features if not f['is_dead']]

    for fi, feat in enumerate(alive_features):
        if fi % 500 == 0 and fi > 0:
            print(f"    Feature {fi}/{len(alive_features)}...")

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
            n_within, n_possible, density = test_edge_enrichment(top_genes, ontologies[edge_ont])
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

    # Build per-feature annotation summary
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
    ont_counts = defaultdict(int)
    for fi, anns in feature_annotations.items():
        for ann in anns:
            ont_counts[ann['ontology']] += 1

    print(f"  Annotated: {n_annotated}/{n_alive} ({100*n_annotated/max(n_alive,1):.1f}%)")
    print(f"  Significant tests: {n_significant}/{len(pval_tests)}")

    # Save
    output = {
        'summary': {
            'run_name': run_name,
            'layer': layer,
            'n_features': n_features,
            'n_alive': n_alive,
            'n_annotated': n_annotated,
            'annotation_rate': n_annotated / max(n_alive, 1),
            'n_significant_tests': n_significant,
            'total_tests': len(pval_tests),
            'ontology_counts': dict(ont_counts),
        },
        'feature_annotations': {str(k): v for k, v in feature_annotations.items()},
        'all_significant_tests': [t for t in pval_tests if t.get('significant', False)],
        'edge_enrichments': edge_tests,
    }

    with open(ann_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    elapsed = time.time() - t0
    print(f"  Layer {layer} done in {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--k', type=int, default=32)
    args = parser.parse_args()

    total_t0 = time.time()
    print("=" * 70)
    print("scGPT SAE PIPELINE — STEP 4: ANNOTATE FEATURES")
    print("=" * 70)

    print("\n  Loading ontologies...")
    ontologies = load_ontologies()

    if args.all or args.layer == -1:
        layers = list(range(N_LAYERS))
    else:
        layers = [args.layer]

    for layer in layers:
        annotate_layer(layer, ontologies, args.expansion, args.k)

    print(f"\nAll done in {(time.time()-total_t0)/60:.1f} min")


if __name__ == '__main__':
    main()
