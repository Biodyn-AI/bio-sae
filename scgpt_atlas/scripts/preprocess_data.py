#!/usr/bin/env python3
"""Preprocess scGPT SAE analysis data into compact JSON for the interactive atlas.

Reads all raw analysis JSON from experiments/scgpt_atlas/ and produces
web-optimized JSON files in scgpt_atlas/public/data/.
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import sys, os

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

BASE = Path(__file__).resolve().parent.parent.parent
SCGPT = BASE / "experiments" / "scgpt_atlas"
OUT = Path(__file__).resolve().parent.parent / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

N_FEATURES = 2048
N_LAYERS = 12


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
    path = SCGPT / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_catalog.json"
    with open(path) as f:
        return json.load(f)


def load_annotations(layer):
    path = SCGPT / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_annotations.json"
    with open(path) as f:
        return json.load(f)


def load_coactivation(layer):
    path = SCGPT / "coactivation" / f"coactivation_layer{layer:02d}.json"
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
        results_path = SCGPT / "sae_models" / f"layer{layer:02d}_x4_k32" / "results.json"
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
            for fi in mod_feats[:100]:
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
    graph_dir = SCGPT / "computational_graph"
    # scGPT uses L0→L4, L4→L8, L8→L11
    pairs = [("00", "04"), ("04", "08"), ("08", "11")]
    graph = {}

    for la, lb in pairs:
        path = graph_dir / f"graph_L{la}_L{lb}.json"
        if not path.exists():
            print(f"  Skipping {path.name} (not found)")
            continue

        with open(path) as f:
            data = json.load(f)

        key = f"L{la}_L{lb}"
        edges = []
        for e in data.get('top_edges', [])[:200]:
            edges.append({
                'a': e['upstream'],
                'b': e['downstream'],
                'pmi': round(e['pmi'], 2),
                'c': e['count'],
            })

        graph[key] = {
            'summary': data['summary'],
            'edges': edges,
            'upstream_hubs': data.get('upstream_hubs', [])[:20],
            'downstream_hubs': data.get('downstream_hubs', [])[:20],
        }

    save_json(graph, OUT / "cross_layer_graph.json")


# ── Step 7: Causal patching ──────────────────────────────────────────────────

def build_causal_patching():
    print("\nBuilding causal patching data...")
    # Try multiple layers (default L7 for scGPT)
    for layer in [7, 6, 8, 5]:
        path = SCGPT / "causal_patching" / f"layer_{layer:02d}_patching.json"
        if path.exists():
            break
    else:
        print("  No causal patching data found, skipping")
        return

    print(f"  Loading from layer {layer}...")
    with open(path) as f:
        data = json.load(f)

    compact = {
        'summary': data['summary'],
        'layer': layer,
        'features': []
    }

    for fid, fdata in data.get('features', {}).items():
        compact['features'].append({
            'i': fdata['feature_idx'],
            'nc': fdata['n_cells_tested'],
            'td': round(fdata['mean_feat_logit_change'], 4),
            'od': round(fdata['mean_ctrl_logit_change'], 4),
            'sr': round(fdata['specificity'], 2),
            'tg': fdata.get('top_genes', [])[:10],
        })

    compact['features'].sort(key=lambda f: -f['sr'])
    save_json(compact, OUT / "causal_patching.json")


# ── Step 8: Cell type enrichments ────────────────────────────────────────────

def build_celltype_enrichments():
    """Convert raw celltype enrichment JSONs to compact web format."""
    CT_DIR = SCGPT / "celltype_enrichments"
    MAX_CT = 10
    MAX_TC = 5

    found = 0
    for layer in range(N_LAYERS):
        raw_path = CT_DIR / f"celltype_enrichment_layer{layer:02d}.json"
        if not raw_path.exists():
            print(f"  [celltype] Layer {layer}: no raw file, skipping")
            continue

        print(f"  [celltype] Processing layer {layer}...")
        with open(raw_path) as f:
            raw = json.load(f)

        compact_features = {}
        n_with_enrichment = 0

        for fi, fdata in raw.get('features', {}).items():
            ct_list = []
            for ct in fdata.get('cell_type_enrichments', [])[:MAX_CT]:
                ct_list.append({
                    'c': ct['cell_type'],
                    'p': ct.get('p_adjusted', ct.get('p_value', 1.0)),
                    'or': ct['odds_ratio'],
                    'n': ct['n_in_top'],
                })

            ti_list = []
            for ti in fdata.get('tissue_enrichments', []):
                ti_list.append({
                    't': ti['tissue'],
                    'p': ti.get('p_adjusted', ti.get('p_value', 1.0)),
                    'or': ti['odds_ratio'],
                })

            tc_list = []
            for tc in fdata.get('top_cells', [])[:MAX_TC]:
                tc_list.append({
                    'ct': tc['cell_type'],
                    't': tc['tissue'],
                    'a': tc['activation'],
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
                'tissues': raw.get('tissues', []),
                'n_cell_types': len(raw.get('cell_types', [])),
                'n_features_with_enrichment': n_with_enrichment,
            },
            'features': compact_features,
        }

        out_path = OUT / f"layer_{layer:02d}_celltypes.json"
        save_json(output, out_path)
        found += 1

    if found == 0:
        print("  [celltype] No enrichment files found.")
    else:
        print(f"  [celltype] {found} layers processed")


# ── Step 9: Gene index ──────────────────────────────────────────────────────

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


# ── Step 10: Ontology index ──────────────────────────────────────────────────

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

def main():
    print("=" * 60)
    print("Preprocessing scGPT SAE data for interactive atlas")
    print(f"  Source: {SCGPT}")
    print(f"  Output: {OUT}")
    print(f"  Layers: {N_LAYERS}, Features/layer: {N_FEATURES}")
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
    build_celltype_enrichments()
    build_gene_index(all_features)
    build_ontology_index()

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
