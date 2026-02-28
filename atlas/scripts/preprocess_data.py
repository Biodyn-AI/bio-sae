#!/usr/bin/env python3
"""Preprocess SAE analysis data into compact JSON for the interactive atlas.

Reads all raw analysis JSON from experiments/phase1_k562/ and produces
web-optimized JSON files in atlas/public/data/.
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import sys, os

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = Path(__file__).resolve().parent.parent.parent
K562 = BASE / "experiments" / "phase1_k562"
OUT = Path(__file__).resolve().parent.parent / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

N_FEATURES = 4608
N_LAYERS = 18


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'), default=_json_default)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Wrote {path.name} ({size_kb:.0f} KB)")


# ── Step 1: Load raw data ────────────────────────────────────────────────────

def load_catalog(layer):
    path = K562 / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_catalog.json"
    with open(path) as f:
        return json.load(f)


def load_annotations(layer):
    path = K562 / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_annotations.json"
    with open(path) as f:
        return json.load(f)


def load_coactivation(layer):
    path = K562 / "coactivation" / f"coactivation_layer{layer:02d}.json"
    with open(path) as f:
        return json.load(f)


# ── Step 2: Compute 2D positions ──────────────────────────────────────────────

def compute_positions(layer, coact):
    """Force-directed layout from co-activation modules."""
    modules = coact['modules']
    G = nx.Graph()
    np.random.seed(42)
    for mod in modules:
        feats = [f for f in mod['features'] if f < N_FEATURES]
        for fi in feats:
            G.add_node(fi, module=mod['module_id'])
            neighbors = np.random.choice(feats, size=min(10, len(feats) - 1), replace=False)
            for fj in neighbors:
                if fi != fj:
                    G.add_edge(fi, fj)

    pos = nx.spring_layout(G, k=0.3, iterations=150, seed=42)

    emb = np.zeros((N_FEATURES, 2), dtype=np.float64)
    if pos:
        all_xy = np.array(list(pos.values()))
        cx, cy = all_xy.mean(axis=0)
        spread = all_xy.std() * 0.3
    else:
        cx, cy, spread = 0.0, 0.0, 0.1

    for i in range(N_FEATURES):
        if i in pos:
            emb[i] = pos[i]
        else:
            emb[i] = [cx + np.random.randn() * spread, cy + np.random.randn() * spread]
    return emb


# ── Step 3: Process each layer ────────────────────────────────────────────────

def process_layer(layer):
    print(f"\nProcessing layer {layer}...")
    catalog = load_catalog(layer)
    annot_data = load_annotations(layer)
    coact = load_coactivation(layer)

    features_raw = catalog['features']
    fa = annot_data['feature_annotations']

    # Build module lookup
    module_map = {}
    for mod in coact['modules']:
        for fi in mod['features']:
            if fi < N_FEATURES:
                module_map[fi] = mod['module_id']

    # Compact feature records
    compact_features = []
    for feat in features_raw:
        idx = feat['feature_idx']
        anns = fa.get(str(idx), [])

        # Best annotation label
        label = ""
        if anns:
            best = min(anns, key=lambda a: a.get('p_adjusted', 1.0))
            label = best.get('term', '')

        # Top ontology
        top_ont = "none"
        if anns:
            best = min(anns, key=lambda a: a.get('p_adjusted', 1.0))
            top_ont = best.get('ontology', 'none')

        top_genes = []
        for g in feat.get('top_genes', [])[:5]:
            top_genes.append({
                'n': g['gene_name'],
                'a': round(g['mean_activation'], 4)
            })

        compact_features.append({
            'i': idx,
            'd': feat.get('is_dead', False),
            'f': round(feat.get('activation_freq', 0.0), 6),
            'ma': round(feat.get('mean_activation', 0.0), 4),
            'fc': feat.get('fire_count', 0),
            'na': len(anns),
            'm': module_map.get(idx, -1),
            'sv': feat.get('is_svd_aligned', False),
            'lb': label,
            'to': top_ont,
            'tg': top_genes,
        })

    # Full annotations (lazy-loaded)
    annotations_out = {}
    for feat in features_raw:
        idx = feat['feature_idx']
        anns = fa.get(str(idx), [])
        if not anns:
            continue

        all_genes = []
        for g in feat.get('top_genes', [])[:20]:
            all_genes.append({
                'n': g['gene_name'],
                'a': round(g['mean_activation'], 4),
                'fc': g.get('fire_count', 0)
            })

        compact_anns = []
        for a in anns:
            p_val = a.get('p_adjusted', a.get('density', 1.0))
            compact_anns.append({
                'o': a['ontology'],
                't': a['term'],
                'p': round(p_val, 6) if p_val > 1e-10 else p_val,
                'or': round(a.get('odds_ratio', 0), 2),
                'n': a.get('n_overlap', 0),
                'g': a.get('overlap_genes', []),
            })

        annotations_out[str(idx)] = {
            'genes': all_genes,
            'anns': compact_anns,
        }

    # Positions
    print(f"  Computing positions for layer {layer}...")
    positions = compute_positions(layer, coact)

    # Save
    save_json(compact_features, OUT / f"layer_{layer:02d}_features.json")
    save_json(annotations_out, OUT / f"layer_{layer:02d}_annotations.json")
    save_json(positions.round(5).tolist(), OUT / f"layer_{layer:02d}_positions.json")

    return catalog['summary'], annot_data['summary'], coact['summary'], compact_features


# ── Step 4: Global summary ────────────────────────────────────────────────────

def build_global_summary(all_catalog_summaries, all_annot_summaries, all_coact_summaries):
    print("\nBuilding global summary...")
    layers = []
    total_alive = 0
    total_annotated = 0
    total_modules = 0

    for layer in range(N_LAYERS):
        cs = all_catalog_summaries[layer]
        ans = all_annot_summaries[layer]
        cos = all_coact_summaries[layer]

        alive = cs.get('n_alive', N_FEATURES)
        dead = cs.get('n_dead', 0)
        annotated = ans.get('n_annotated', 0)
        annotation_rate = ans.get('annotation_rate', 0)
        n_modules = cos.get('n_modules', 0)
        n_svd = cs.get('n_svd_aligned', 0)
        n_novel = cs.get('n_novel', 0)

        total_alive += alive
        total_annotated += annotated
        total_modules += n_modules

        ont_counts = ans.get('ontology_counts', {})

        layers.append({
            'layer': layer,
            'alive': alive,
            'dead': dead,
            'annotated': annotated,
            'annotation_rate': round(annotation_rate, 4),
            'n_modules': n_modules,
            'n_svd_aligned': n_svd,
            'n_novel': n_novel,
            'ontology_counts': {
                'GO_BP': ont_counts.get('GO_BP', 0),
                'KEGG': ont_counts.get('KEGG', 0),
                'Reactome': ont_counts.get('Reactome', 0),
                'STRING': ont_counts.get('STRING_edges', 0),
                'TRRUST': ont_counts.get('TRRUST_TF_enrichment', 0) + ont_counts.get('TRRUST_edges', 0),
            },
            'mean_feature_cosine': round(cs.get('mean_feature_cosine', 0), 4),
        })

    # Load results.json for variance explained
    for layer_info in layers:
        layer = layer_info['layer']
        results_path = K562 / "sae_models" / f"layer{layer:02d}_x4_k32" / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            layer_info['variance_explained'] = round(results['results'].get('variance_explained', 0), 4)

    summary = {
        'total_features': N_FEATURES * N_LAYERS,
        'total_alive': total_alive,
        'total_annotated': total_annotated,
        'total_modules': total_modules,
        'total_novel': sum(l['n_novel'] for l in layers),
        'n_layers': N_LAYERS,
        'n_features_per_layer': N_FEATURES,
        'layers': layers,
    }

    save_json(summary, OUT / "global_summary.json")
    return summary


# ── Step 5: Modules ──────────────────────────────────────────────────────────

def build_modules_file(all_features):
    print("\nBuilding modules file...")
    modules = []
    for layer in range(N_LAYERS):
        coact = load_coactivation(layer)
        feats = {f['i']: f for f in all_features[layer]}

        for mod in coact['modules']:
            mod_feats = [fi for fi in mod['features'] if fi < N_FEATURES]

            # Find top shared annotations for the module
            ann_counts = defaultdict(int)
            for fi in mod_feats[:100]:  # Sample for speed
                f = feats.get(fi)
                if f and f['lb']:
                    ann_counts[f['lb']] += 1

            top_annotations = sorted(ann_counts.items(), key=lambda x: -x[1])[:5]

            modules.append({
                'layer': layer,
                'id': mod['module_id'],
                'n': len(mod_feats),
                'features': mod_feats,
                'top_anns': [{'t': t, 'c': c} for t, c in top_annotations],
            })

    save_json(modules, OUT / "modules.json")


# ── Step 6: Cross-layer graph ────────────────────────────────────────────────

def build_cross_layer_graph():
    print("\nBuilding cross-layer graph...")
    pairs = [("00", "05"), ("05", "11"), ("11", "17")]
    graph = {}

    for la, lb in pairs:
        path = K562 / "computational_graph" / f"deps_L{la}_to_L{lb}.json"
        if not path.exists():
            print(f"  Skipping {path.name} (not found)")
            continue

        with open(path) as f:
            data = json.load(f)

        key = f"L{la}_L{lb}"
        deps = []
        for d in data['dependencies']:
            top5 = d['top_dependencies'][:5]
            deps.append({
                'a': d['feature_a'],
                'deps': [{
                    'b': t['feature_b'],
                    'pmi': round(t['pmi'], 2),
                    'lb': t.get('label_b', ''),
                } for t in top5]
            })

        graph[key] = {
            'summary': data['summary'],
            'deps': deps,
        }

    save_json(graph, OUT / "cross_layer_graph.json")


# ── Step 7: Causal patching ──────────────────────────────────────────────────

def build_causal_patching():
    print("\nBuilding causal patching data...")
    path = K562 / "causal_patching" / "causal_patching_layer11.json"
    if not path.exists():
        print("  Not found, skipping")
        return

    with open(path) as f:
        data = json.load(f)

    compact = {
        'summary': data['summary'],
        'features': [{
            'i': f['feature_idx'],
            'lb': f.get('label', ''),
            'na': f.get('n_annotations', 0),
            'af': round(f.get('activation_freq', 0), 4),
            'tg': f.get('top_genes', [])[:10],
            'td': round(f.get('target_logit_diff_mean', 0), 4),
            'od': round(f.get('other_logit_diff_mean', 0), 4),
            'sr': round(f.get('specificity_ratio', 0), 2),
        } for f in data['feature_results']]
    }

    save_json(compact, OUT / "causal_patching.json")


# ── Step 8: Perturbation response ─────────────────────────────────────────────

def build_perturbation_response():
    print("\nBuilding perturbation response data...")
    path = K562 / "perturbation_response" / "perturbation_response_layer11.json"
    if not path.exists():
        print("  Not found, skipping")
        return

    with open(path) as f:
        data = json.load(f)

    compact = {
        'summary': data['summary'],
        'targets': [{
            'gene': t['target_gene'],
            'tf': t.get('is_trrust_tf', False),
            'nk': t.get('n_known_targets', 0),
            'nr': t.get('n_responding_features', 0),
            'ns': t.get('n_specific_responding', 0),
            'top': [{
                'i': f['feature_idx'],
                'es': round(f['effect_size'], 3),
                'lb': f.get('label', ''),
            } for f in t.get('top_changed_features', [])[:5]]
        } for t in data['target_results']]
    }

    save_json(compact, OUT / "perturbation_response.json")


# ── Step 9: SVD comparison ────────────────────────────────────────────────────

def build_svd_comparison():
    print("\nBuilding SVD comparison data...")
    path = K562 / "svd_vs_sae_comparison.json"
    if not path.exists():
        print("  Not found, skipping")
        return

    with open(path) as f:
        data = json.load(f)

    compact = {
        'aggregate': data.get('aggregate', {}),
        'per_layer': {}
    }
    for layer_key, layer_data in data['per_layer'].items():
        ve = layer_data.get('variance_explained', {})
        fc = layer_data.get('feature_counts', {})
        sc = layer_data.get('svd_coverage', {})
        svd_var = ve.get('svd_top50', 0)
        sae_var = ve.get('sae_4x_k32', 0)
        compact['per_layer'][layer_key] = {
            'svd_variance': round(svd_var, 4),
            'sae_variance': round(sae_var, 4),
            'gain': round(sae_var / svd_var, 2) if svd_var > 0 else 0,
            'n_aligned': fc.get('n_svd_aligned', sc.get('n_covered_by_sae', 0)),
            'n_novel': fc.get('n_novel', 0),
        }

    save_json(compact, OUT / "svd_comparison.json")


# ── Step 10: Cross-layer tracking ─────────────────────────────────────────────

def build_cross_layer_tracking():
    print("\nBuilding cross-layer tracking data...")
    path = K562 / "cross_layer_tracking.json"
    if not path.exists():
        print("  Not found, skipping")
        return

    with open(path) as f:
        data = json.load(f)

    save_json(data, OUT / "cross_layer_tracking.json")


# ── Step 11: Novel clusters ──────────────────────────────────────────────────

def build_novel_clusters():
    print("\nBuilding novel clusters data...")
    all_clusters = {}
    for layer in [0, 5, 11, 17]:
        path = K562 / "novel_features" / f"novel_features_layer{layer:02d}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        all_clusters[str(layer)] = {
            'summary': data['summary'],
            'clusters': data.get('novel_clusters', []),
        }

    save_json(all_clusters, OUT / "novel_clusters.json")


# ── Step 12: Gene index ──────────────────────────────────────────────────────

def build_gene_index(all_features):
    print("\nBuilding gene index...")
    gene_idx = defaultdict(list)

    for layer in range(N_LAYERS):
        for feat in all_features[layer]:
            for rank, g in enumerate(feat['tg']):
                gene_idx[g['n']].append({
                    'l': layer,
                    'i': feat['i'],
                    'r': rank,
                    'lb': feat['lb'][:60] if feat['lb'] else '',
                    'm': feat['m'],
                })

    save_json(dict(gene_idx), OUT / "gene_index.json")


# ── Step 13: Ontology index ──────────────────────────────────────────────────

def build_ontology_index():
    print("\nBuilding ontology index...")
    ont_idx = defaultdict(list)

    for layer in range(N_LAYERS):
        annot_data = load_annotations(layer)
        fa = annot_data['feature_annotations']

        for idx_str, anns in fa.items():
            for a in anns:
                term = a['term']
                p_val = a.get('p_adjusted', a.get('density', 1.0))
                ont_idx[term].append({
                    'l': layer,
                    'i': int(idx_str),
                    'p': round(p_val, 6) if p_val > 1e-10 else p_val,
                    'o': a['ontology'],
                })

    # Only keep terms with at least 2 features to reduce size
    filtered = {k: v for k, v in ont_idx.items() if len(v) >= 2}
    save_json(filtered, OUT / "ontology_index.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_celltype_enrichments():
    """Convert raw celltype enrichment JSONs to compact web format."""
    CT_DIR = BASE / "experiments" / "phase3_multitissue" / "celltype_enrichments"
    TS_LAYERS = list(range(18))
    MAX_CT = 10  # max cell type enrichments per feature
    MAX_TC = 5   # max top cells per feature

    found = 0
    for layer in TS_LAYERS:
        raw_path = CT_DIR / f"celltype_enrichment_layer{layer:02d}.json"
        if not raw_path.exists():
            print(f"  [celltype] Layer {layer}: no raw file, skipping")
            continue

        print(f"  [celltype] Processing layer {layer}...")
        with open(raw_path) as f:
            raw = json.load(f)

        # Build cell type metadata
        ct_meta = {}
        ct_counts = raw.get('cell_type_counts', {})
        # Determine which tissues each cell type appears in (from top_cells across features)
        ct_tissues = defaultdict(set)
        for fi, fdata in raw.get('features', {}).items():
            for tc in fdata.get('top_cells', []):
                ct_tissues[tc['ct']].add(tc['t'])
        for ct_name, count in ct_counts.items():
            ct_meta[ct_name] = {
                'n': count,
                't': ','.join(sorted(ct_tissues.get(ct_name, set())))
            }

        # Convert features to compact format
        compact_features = {}
        n_with_enrichment = 0
        for fi, fdata in raw.get('features', {}).items():
            ct_list = []
            for ct in fdata.get('cell_types', [])[:MAX_CT]:
                ct_list.append({
                    'c': ct['ct'],
                    'p': ct.get('p_adj', ct.get('p_raw', 1.0)),
                    'or': ct['or'],
                    'n': ct['n_top'],
                })

            ti_list = []
            for ti in fdata.get('tissues', []):
                ti_list.append({
                    't': ti['t'],
                    'p': ti.get('p_adj', ti.get('p_raw', 1.0)),
                    'or': ti['or'],
                })

            tc_list = []
            for tc in fdata.get('top_cells', [])[:MAX_TC]:
                tc_list.append({
                    'ct': tc['ct'],
                    't': tc['t'],
                    'a': tc['a'],
                })

            if ct_list or ti_list:
                n_with_enrichment += 1
                compact_features[fi] = {
                    'ct': ct_list,
                    'ti': ti_list,
                    'tc': tc_list,
                }

        output = {
            'summary': {
                'n_cells': raw.get('n_cells', 3000),
                'tissues': sorted(raw.get('tissue_counts', {}).keys()),
                'n_cell_types': len(ct_counts),
                'n_features_with_enrichment': n_with_enrichment,
            },
            'cell_type_meta': ct_meta,
            'features': compact_features,
        }

        out_path = OUT / f"layer_{layer:02d}_celltypes.json"
        save_json(output, out_path)
        size = os.path.getsize(out_path) / 1024
        print(f"    -> {out_path.name}: {n_with_enrichment} features with enrichment, {size:.1f} KB")
        found += 1

    if found == 0:
        print("  [celltype] No enrichment files found. Run compute_celltype_enrichments.py first.")
    else:
        print(f"  [celltype] {found} layers processed")


def main():
    print("=" * 60)
    print("Preprocessing SAE data for interactive atlas")
    print("=" * 60)

    all_catalog_summaries = {}
    all_annot_summaries = {}
    all_coact_summaries = {}
    all_features = {}

    # Process each layer
    for layer in range(N_LAYERS):
        cs, ans, cos, feats = process_layer(layer)
        all_catalog_summaries[layer] = cs
        all_annot_summaries[layer] = ans
        all_coact_summaries[layer] = cos
        all_features[layer] = feats

    # Build global files
    build_global_summary(all_catalog_summaries, all_annot_summaries, all_coact_summaries)
    build_modules_file(all_features)
    build_cross_layer_graph()
    build_causal_patching()
    build_perturbation_response()
    build_svd_comparison()
    build_cross_layer_tracking()
    build_novel_clusters()
    build_gene_index(all_features)
    build_ontology_index()
    build_celltype_enrichments()

    print("\n" + "=" * 60)
    print("DONE! All files written to", OUT)
    print("=" * 60)

    # Print file inventory
    total = 0
    for f in sorted(OUT.glob("*.json")):
        size = os.path.getsize(f)
        total += size
        print(f"  {f.name:40s} {size/1024:8.1f} KB")
    print(f"  {'TOTAL':40s} {total/1024/1024:8.1f} MB")


if __name__ == "__main__":
    main()
