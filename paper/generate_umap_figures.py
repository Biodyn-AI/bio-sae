#!/usr/bin/env python3
"""Generate feature projection figures for SAE paper.

Uses two complementary approaches:
- Co-activation network layout (spring-directed) for module structure
- TF-IDF annotation term UMAP for biological function clustering
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from pathlib import Path
import networkx as nx
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# ── Paths ──────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
K562 = BASE / "experiments" / "phase1_k562"
FIGDIR = BASE / "paper" / "figures"
FIGDIR.mkdir(exist_ok=True)

LAYERS = [0, 5, 11, 17]

# ── Style ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

MODULE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
    '#8da0cb', '#e78ac3',
]

ONTOLOGY_COLORS = {
    'GO_BP': '#1b9e77',
    'KEGG': '#d95f02',
    'Reactome': '#7570b3',
    'STRING_edges': '#e7298a',
    'TRRUST_TF_enrichment': '#66a61e',
    'TRRUST_edges': '#e6ab02',
    'none': '#cccccc',
}

LAYER_COLORS = {0: '#2166ac', 5: '#92c5de', 11: '#f4a582', 17: '#b2182b'}


# ── Data loading ───────────────────────────────────────────────────────────

def load_catalog(layer):
    path = K562 / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_catalog.json"
    with open(path) as f:
        return json.load(f)

def load_annotations(layer):
    path = K562 / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_annotations.json"
    with open(path) as f:
        return json.load(f)

def load_modules(layer):
    path = K562 / "coactivation" / f"coactivation_layer{layer:02d}.json"
    with open(path) as f:
        coact = json.load(f)
    module_ids = np.full(4608, -1, dtype=int)
    for mod in coact['modules']:
        for fi in mod['features']:
            if fi < 4608:
                module_ids[fi] = mod['module_id']
    return module_ids, coact['modules']

def get_feature_metadata(layer):
    catalog = load_catalog(layer)
    annot_data = load_annotations(layer)
    fa = annot_data['feature_annotations']
    features = catalog['features']
    n = len(features)

    meta = {
        'is_dead': np.array([f.get('is_dead', False) for f in features]),
        'activation_freq': np.array([f.get('activation_freq', 0.0) for f in features]),
        'is_svd_aligned': np.array([f.get('is_svd_aligned', False) for f in features]),
    }

    n_annotations = np.zeros(n, dtype=int)
    has_annotation = np.zeros(n, dtype=bool)
    top_ontology = ['none'] * n

    for i in range(n):
        anns = fa.get(str(i), [])
        n_annotations[i] = len(anns)
        has_annotation[i] = len(anns) > 0
        if anns:
            best_score = 1e10
            best_ont = 'none'
            for a in anns:
                if 'p_adjusted' in a:
                    score = a['p_adjusted']
                elif 'density' in a:
                    score = 1.0 / (a['density'] + 1e-10)
                else:
                    continue
                if score < best_score:
                    best_score = score
                    best_ont = a['ontology']
            top_ontology[i] = best_ont

    meta['n_annotations'] = n_annotations
    meta['has_annotation'] = has_annotation
    meta['top_ontology'] = top_ontology
    return meta


# ── Network layout ─────────────────────────────────────────────────────────

NET_CACHE = {}

def compute_network_layout(layer):
    """Force-directed spring layout from co-activation module structure.

    Only intra-module edges are used. Unassigned features are placed near
    the centroid with small random offsets so they don't distort the layout.
    """
    if layer in NET_CACHE:
        return NET_CACHE[layer]

    module_ids, modules = load_modules(layer)
    G = nx.Graph()

    np.random.seed(42)
    for mod in modules:
        feats = [f for f in mod['features'] if f < 4608]
        for fi in feats:
            G.add_node(fi, module=mod['module_id'])
            # Connect each node to ~10 random same-module neighbors
            neighbors = np.random.choice(feats, size=min(10, len(feats) - 1), replace=False)
            for fj in neighbors:
                if fi != fj:
                    G.add_edge(fi, fj)

    # NO cross-module edges for unassigned nodes
    print(f"    Graph L{layer}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    pos = nx.spring_layout(G, k=0.3, iterations=150, seed=42)

    # Build full embedding: module features use layout, unassigned go to center
    emb = np.zeros((4608, 2), dtype=np.float64)
    all_xy = np.array(list(pos.values()))
    cx, cy = all_xy.mean(axis=0)
    spread = all_xy.std() * 0.3  # small spread around centroid

    for i in range(4608):
        if i in pos:
            emb[i] = pos[i]
        else:
            # Unassigned: random position near center
            emb[i] = [cx + np.random.randn() * spread,
                       cy + np.random.randn() * spread]

    NET_CACHE[layer] = emb
    return emb


# ── TF-IDF UMAP ───────────────────────────────────────────────────────────

TFIDF_CACHE = {}

def compute_tfidf_umap(layer):
    """UMAP on TF-IDF weighted annotation term vectors."""
    if layer in TFIDF_CACHE:
        return TFIDF_CACHE[layer]

    annot_data = load_annotations(layer)
    fa = annot_data['feature_annotations']

    # Build term vocabulary
    all_terms = set()
    for idx_str, anns in fa.items():
        for a in anns:
            all_terms.add(f"{a['ontology']}:{a['term']}")
    term_list = sorted(all_terms)
    term_idx = {t: i for i, t in enumerate(term_list)}

    # Build binary matrix
    mat = np.zeros((4608, len(term_list)), dtype=np.float32)
    for i in range(4608):
        for a in fa.get(str(i), []):
            mat[i, term_idx[f"{a['ontology']}:{a['term']}"]] = 1.0

    # TF-IDF
    tfidf = TfidfTransformer()
    mat_tfidf = tfidf.fit_transform(mat).toarray()

    emb = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
               random_state=42).fit_transform(mat_tfidf)
    TFIDF_CACHE[layer] = emb
    return emb


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9: 4-layer overview — network layout (modules + annotation richness)
# ══════════════════════════════════════════════════════════════════════════

def fig9_overview():
    print("Generating Fig 9: 4-layer network layout overview...")

    fig, axes = plt.subplots(4, 2, figsize=(7, 12))

    for row, layer in enumerate(LAYERS):
        print(f"  Layer {layer}...")
        emb = compute_network_layout(layer)
        module_ids, modules = load_modules(layer)
        meta = get_feature_metadata(layer)
        n_modules = len(modules)

        # ── Left: modules ──
        ax = axes[row, 0]
        mask_un = module_ids == -1
        if mask_un.sum() > 0:
            ax.scatter(emb[mask_un, 0], emb[mask_un, 1],
                       c='#dddddd', s=2, alpha=0.15, rasterized=True)
        mask_dead = meta['is_dead']
        if mask_dead.sum() > 0:
            ax.scatter(emb[mask_dead, 0], emb[mask_dead, 1],
                       c='black', s=4, marker='x', alpha=0.5, linewidths=0.4,
                       rasterized=True, zorder=5)
        for mod_id in range(n_modules):
            mask = module_ids == mod_id
            color = MODULE_COLORS[mod_id % len(MODULE_COLORS)]
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, s=5, alpha=0.7, rasterized=True,
                       edgecolors='none',
                       label=f'M{mod_id} ({mask.sum()})')
        ax.set_title(f'Layer {layer} — modules ({n_modules})', fontweight='bold', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(fontsize=4.5, ncol=2, loc='upper right',
                      markerscale=2, handletextpad=0.1, columnspacing=0.3)

        # ── Right: annotation richness ──
        ax = axes[row, 1]
        richness = np.log1p(meta['n_annotations'])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=richness,
                        cmap='viridis', s=5, alpha=0.7, rasterized=True,
                        edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7, label='log(1 + n_annotations)')
        ax.set_title(f'Layer {layer} — annotation richness', fontweight='bold', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig9_umap_overview.pdf')
    fig.savefig(FIGDIR / 'fig9_umap_overview.png')
    plt.close(fig)
    print("  Done: fig9_umap_overview")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10: L11 detail — 6 panels using network layout
# ══════════════════════════════════════════════════════════════════════════

def fig10_l11_detail():
    print("Generating Fig 10: L11 detailed (6 panels)...")

    layer = 11
    emb = compute_network_layout(layer)
    module_ids, modules = load_modules(layer)
    meta = get_feature_metadata(layer)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))

    # (a) Modules
    ax = axes[0, 0]
    mask_un = module_ids == -1
    if mask_un.sum() > 0:
        ax.scatter(emb[mask_un, 0], emb[mask_un, 1],
                   c='#dddddd', s=4, alpha=0.15, rasterized=True)
    for mod_id in range(len(modules)):
        mask = module_ids == mod_id
        color = MODULE_COLORS[mod_id % len(MODULE_COLORS)]
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=6, alpha=0.7,
                   rasterized=True, edgecolors='none', label=f'M{mod_id}')
    ax.set_title('(a) Co-activation modules', fontweight='bold')
    ax.legend(fontsize=5, ncol=2, markerscale=2, loc='upper right',
              handletextpad=0.1, columnspacing=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) Annotation status
    ax = axes[0, 1]
    dead = meta['is_dead']
    ann = meta['has_annotation'] & ~dead
    unann = ~meta['has_annotation'] & ~dead
    ax.scatter(emb[unann, 0], emb[unann, 1], c='#cccccc', s=4, alpha=0.3,
               label=f'Unannotated ({unann.sum()})', rasterized=True)
    ax.scatter(emb[ann, 0], emb[ann, 1], c='#2166ac', s=5, alpha=0.6,
               label=f'Annotated ({ann.sum()})', rasterized=True)
    if dead.sum() > 0:
        ax.scatter(emb[dead, 0], emb[dead, 1], c='black', s=6, marker='x',
                   alpha=0.5, linewidths=0.5, label=f'Dead ({dead.sum()})',
                   rasterized=True, zorder=5)
    ax.set_title('(b) Annotation status', fontweight='bold')
    ax.legend(fontsize=6, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

    # (c) Top ontology
    ax = axes[0, 2]
    for ont_name, ont_color in ONTOLOGY_COLORS.items():
        mask = np.array([t == ont_name for t in meta['top_ontology']])
        if mask.sum() == 0:
            continue
        display = ont_name.replace('_edges', '').replace('_enrichment', '').replace('_', ' ')
        ax.scatter(emb[mask, 0], emb[mask, 1], c=ont_color, s=5, alpha=0.6,
                   label=f'{display} ({mask.sum()})', rasterized=True,
                   zorder=1 if ont_name == 'none' else 2, edgecolors='none')
    ax.set_title('(c) Top ontology source', fontweight='bold')
    ax.legend(fontsize=5, markerscale=2, loc='upper right', handletextpad=0.1)
    ax.set_xticks([]); ax.set_yticks([])

    # (d) Annotation richness
    ax = axes[1, 0]
    richness = np.log1p(meta['n_annotations'])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=richness,
                    cmap='viridis', s=5, alpha=0.7, rasterized=True,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='log(1 + n_annotations)')
    ax.set_title('(d) Annotation richness', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # (e) Activation frequency
    ax = axes[1, 1]
    freq = meta['activation_freq']
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.log10(freq + 1e-6),
                    cmap='magma', s=5, alpha=0.7, rasterized=True,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='log\u2081\u2080(activation freq)')
    ax.set_title('(e) Activation frequency', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # (f) SVD alignment
    ax = axes[1, 2]
    svd = meta['is_svd_aligned']
    novel = ~svd & ~dead
    ax.scatter(emb[novel, 0], emb[novel, 1], c='#2166ac', s=4, alpha=0.25,
               label=f'Novel ({novel.sum()})', rasterized=True, edgecolors='none')
    if svd.sum() > 0:
        ax.scatter(emb[svd, 0], emb[svd, 1], c='#e66101', s=25, alpha=0.9,
                   label=f'SVD-aligned ({svd.sum()})', rasterized=True,
                   edgecolors='black', linewidths=0.3, zorder=5)
    if dead.sum() > 0:
        ax.scatter(emb[dead, 0], emb[dead, 1], c='black', s=6, marker='x',
                   alpha=0.5, linewidths=0.5, label=f'Dead ({dead.sum()})',
                   rasterized=True)
    ax.set_title('(f) SVD alignment', fontweight='bold')
    ax.legend(fontsize=6, markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig10_l11_detail.pdf')
    fig.savefig(FIGDIR / 'fig10_l11_detail.png')
    plt.close(fig)
    print("  Done: fig10_l11_detail")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 11: TF-IDF UMAP (annotation-based) + t-SNE comparison
# ══════════════════════════════════════════════════════════════════════════

def fig11_annotation_projections():
    print("Generating Fig 11: Annotation-based projections...")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    layer = 11
    module_ids, modules = load_modules(layer)

    # (a) TF-IDF UMAP colored by module
    print("  Computing TF-IDF UMAP for L11...")
    emb_umap = compute_tfidf_umap(layer)

    ax = axes[0]
    mask_un = module_ids == -1
    if mask_un.sum() > 0:
        ax.scatter(emb_umap[mask_un, 0], emb_umap[mask_un, 1],
                   c='#dddddd', s=3, alpha=0.15, rasterized=True)
    for mod_id in range(len(modules)):
        mask = module_ids == mod_id
        color = MODULE_COLORS[mod_id % len(MODULE_COLORS)]
        ax.scatter(emb_umap[mask, 0], emb_umap[mask, 1],
                   c=color, s=5, alpha=0.6, rasterized=True,
                   edgecolors='none', label=f'M{mod_id}')
    ax.set_title('(a) L11 TF-IDF annotation UMAP', fontweight='bold')
    ax.legend(fontsize=5, ncol=2, markerscale=2, loc='upper right',
              handletextpad=0.1, columnspacing=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) TF-IDF t-SNE for comparison (annotated features only)
    print("  Computing TF-IDF t-SNE for L11 (annotated only)...")
    annot_data = load_annotations(layer)
    fa = annot_data['feature_annotations']

    all_terms = set()
    for idx_str, anns in fa.items():
        for a in anns:
            all_terms.add(f"{a['ontology']}:{a['term']}")
    term_list = sorted(all_terms)
    term_idx = {t: i for i, t in enumerate(term_list)}

    # Build matrix for annotated features
    ann_indices = []
    rows = []
    for i in range(4608):
        anns = fa.get(str(i), [])
        if not anns:
            continue
        ann_indices.append(i)
        row = np.zeros(len(term_list), dtype=np.float32)
        for a in anns:
            row[term_idx[f"{a['ontology']}:{a['term']}"]] = 1.0
        rows.append(row)

    mat_ann = np.stack(rows)
    tfidf = TfidfTransformer()
    mat_tfidf = tfidf.fit_transform(mat_ann).toarray()
    mids_ann = module_ids[np.array(ann_indices)]

    emb_tsne = TSNE(perplexity=20, random_state=42, metric='cosine',
                    init='random', learning_rate='auto').fit_transform(mat_tfidf)

    ax = axes[1]
    for mod_id in sorted(set(mids_ann[mids_ann >= 0])):
        mask = mids_ann == mod_id
        color = MODULE_COLORS[mod_id % len(MODULE_COLORS)]
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                   c=color, s=5, alpha=0.6, rasterized=True,
                   edgecolors='none', label=f'M{mod_id}')
    mask_un = mids_ann == -1
    if mask_un.sum() > 0:
        ax.scatter(emb_tsne[mask_un, 0], emb_tsne[mask_un, 1],
                   c='#dddddd', s=3, alpha=0.2, rasterized=True)
    ax.set_title(f'(b) L11 TF-IDF t-SNE (n={len(ann_indices)} annotated)',
                 fontweight='bold')
    ax.legend(fontsize=5, ncol=2, markerscale=2, loc='upper right',
              handletextpad=0.1, columnspacing=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig11_cross_layer.pdf')
    fig.savefig(FIGDIR / 'fig11_cross_layer.png')
    plt.close(fig)
    print("  Done: fig11_cross_layer")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Generating feature projection figures")
    print("=" * 60)
    fig9_overview()
    fig10_l11_detail()
    fig11_annotation_projections()
    print(f"\nAll figures saved to {FIGDIR}")
