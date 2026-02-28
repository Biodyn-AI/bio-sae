#!/usr/bin/env python3
"""Generate all figures for the SAE paper v2 (Geneformer + scGPT).

Reads JSON data from experiments/ and produces publication-quality PDF figures
in paper/figures/.

Changes from v1:
  - Fig 2C: Replace uninformative 99.9/0.1% pie with variance explained comparison
  - Fig 5A: Fix legend overlap on histogram
  - Fig 6: Fix overlapping panel titles (A/B)
  - New Fig S1: scGPT atlas overview (training stats across 12 layers)
  - New Fig S2: Cross-model comparison (Geneformer vs scGPT side-by-side)
  - New Fig S3: scGPT cross-layer highways + progressive concentration
  - New Fig S4: Cell type enrichment overview
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent
K562 = BASE / "experiments" / "phase1_k562"
MT = BASE / "experiments" / "phase3_multitissue"
SCGPT = BASE / "experiments" / "scgpt_atlas"
FIGDIR = BASE / "paper" / "figures"
FIGDIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#2166ac',
    'red': '#b2182b',
    'green': '#1b7837',
    'orange': '#e66101',
    'purple': '#7b3294',
    'gray': '#878787',
    'light_blue': '#92c5de',
    'light_red': '#f4a582',
    'gold': '#dfc27d',
    'teal': '#018571',
}

# ── Load all data ──────────────────────────────────────────────────────────

def load_all_results():
    data = {}
    for layer in range(18):
        p = K562 / "sae_models" / f"layer{layer:02d}_x4_k32" / "results.json"
        with open(p) as f:
            data[layer] = json.load(f)
    return data

def load_svd_comparison():
    with open(K562 / "svd_vs_sae_comparison.json") as f:
        return json.load(f)

def load_cross_layer():
    with open(K562 / "cross_layer_tracking.json") as f:
        return json.load(f)

def load_coactivation(layer):
    with open(K562 / "coactivation" / f"coactivation_layer{layer:02d}.json") as f:
        return json.load(f)

def load_causal():
    with open(K562 / "causal_patching" / "causal_patching_layer11.json") as f:
        return json.load(f)

def load_perturbation():
    with open(K562 / "perturbation_response" / "perturbation_response_layer11.json") as f:
        return json.load(f)

def load_comparison():
    with open(MT / "comparison" / "comparison_results.json") as f:
        return json.load(f)

def load_mt_results():
    data = {}
    for layer in [0, 5, 11, 17]:
        p = MT / "sae_models" / f"layer{layer:02d}_x4_k32" / "results.json"
        if p.exists():
            with open(p) as f:
                data[layer] = json.load(f)
    return data

# ── Annotation data (hardcoded from verified results) ──

LAYER_DATA = {
    'layers': list(range(18)),
    'var_expl': [0.839, 0.846, 0.848, 0.852, 0.853, 0.846, 0.833, 0.814,
                 0.804, 0.802, 0.810, 0.816, 0.810, 0.801, 0.800, 0.788, 0.769, 0.768],
    'alive': [4608, 4606, 4601, 4595, 4583, 4576, 4580, 4591,
              4586, 4595, 4602, 4598, 4592, 4583, 4568, 4543, 4538, 4580],
    'dead': [10, 18, 70, 76, 101, 133, 134, 166,
             180, 115, 88, 80, 94, 107, 144, 215, 314, 209],
    'svd_aligned': [41, 27, 29, 12, 13, 8, 5, 6, 3, 4, 7, 4, 3, 3, 2, 5, 5, 12],
    'novel': [4567, 4579, 4572, 4583, 4570, 4568, 4575, 4585,
              4583, 4591, 4595, 4594, 4589, 4580, 4566, 4538, 4533, 4568],
    'ann_pct': [58.6, 57.4, 55.5, 56.4, 53.9, 52.1, 49.1, 47.7,
                45.4, 50.3, 55.8, 56.2, 53.2, 50.7, 50.6, 49.0, 54.7, 47.0],
    'GO_BP': [10153, 10022, 9948, 9726, 8537, 7695, 7180, 6628,
              6850, 7299, 8461, 8785, 8217, 7158, 7615, 6790, 7040, 7002],
    'KEGG': [2650, 2433, 2495, 2514, 2045, 1845, 1555, 1637,
             1570, 1643, 1751, 2089, 1915, 1686, 1595, 1520, 1781, 1762],
    'Reactome': [11001, 10512, 10790, 9525, 9195, 8189, 7871, 7080,
                 7169, 7880, 9247, 8957, 8856, 8393, 8412, 7135, 7172, 6869],
    'STRING': [302, 258, 283, 273, 248, 216, 182, 181,
               199, 207, 214, 227, 210, 202, 221, 150, 158, 193],
    'TRRUST_TF': [155, 164, 150, 157, 133, 136, 110, 125,
                  90, 103, 128, 112, 117, 101, 97, 126, 87, 131],
    'TRRUST_edges': [42, 48, 32, 37, 30, 24, 28, 30, 27, 29, 30, 31, 35, 28, 27, 34, 25, 25],
}

COACT_DATA = {
    'layers': list(range(18)),
    'modules': [6, 8, 7, 8, 9, 12, 7, 8, 7, 7, 9, 8, 7, 7, 8, 8, 8, 7],
    'coverage': [99.3, 99.0, 98.6, 98.3, 98.2, 97.7, 97.3, 96.7,
                 97.6, 98.7, 99.3, 99.3, 99.5, 99.5, 99.5, 98.2, 96.0, 97.7],
    'pmi_edges': [446324, 440681, 404403, 393574, 393194, 390845, 383033, 371832,
                  369280, 380304, 388498, 388103, 388977, 383779, 379595, 340269, 327895, 343059],
}

# scGPT data (from global_summary.json and experiment results)
SCGPT_DATA = {
    'layers': list(range(12)),
    'var_expl': [0.920, 0.930, 0.932, 0.932, 0.935, 0.921, 0.903, 0.857,
                 0.863, 0.869, 0.874, 0.886],
    'alive': [2027, 2035, 2038, 2045, 2048, 2048, 2048, 2048,
              2048, 2047, 2047, 2048],  # from annotation step
    'dead': [21, 13, 10, 3, 0, 0, 0, 0, 0, 1, 1, 0],  # from annotation-step alive counts (2048 - n_alive)
    'ann_pct': [32.7, 30.8, 28.7, 31.6, 30.4, 29.6, 28.9, 32.4,
                28.9, 33.9, 31.7, 32.0],
    'modules': [6, 6, 5, 7, 6, 7, 7, 6, 7, 5, 7, 7],
    'mean_cos': [0.038, 0.041, 0.040, 0.041, 0.041, 0.042, 0.046, 0.049,
                 0.046, 0.045, 0.044, 0.043],
    'GO_BP': [2125, 1806, 1635, 1777, 1594, 1654, 1495, 1603,
              1549, 1571, 1620, 1683],
    'KEGG': [541, 488, 497, 577, 449, 512, 561, 581,
             456, 438, 423, 398],
    'Reactome': [2443, 2310, 2085, 2164, 1992, 1978, 1953, 2082,
                 2071, 2023, 2268, 2155],
    'STRING': [52, 49, 46, 60, 50, 44, 52, 52, 39, 44, 39, 32],
    'TRRUST': [122, 119, 120, 130, 121, 117, 124, 121, 117, 223, 121, 131],
}

SCGPT_HIGHWAYS = {
    'L0→L4': {'pmi_edges': 75305, 'upstream': 1935, 'upstream_total': 2027,
              'downstream': 1960, 'downstream_total': 2048, 'max_pmi': 9.15},
    'L4→L8': {'pmi_edges': 61263, 'upstream': 1955, 'upstream_total': 2048,
              'downstream': 1723, 'downstream_total': 2048, 'max_pmi': 9.26},
    'L8→L11': {'pmi_edges': 45258, 'upstream': 1773, 'upstream_total': 2048,
               'downstream': 1289, 'downstream_total': 2048, 'max_pmi': 10.78},
}


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: SAE Atlas Overview (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def fig1_atlas_overview():
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    layers = LAYER_DATA['layers']

    ax = axes[0, 0]
    ax.plot(layers, [v*100 for v in LAYER_DATA['var_expl']], 'o-',
            color=COLORS['blue'], markersize=4, linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('A  SAE reconstruction quality', fontweight='bold', loc='left')
    ax.set_ylim(74, 88)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
    ax.axhline(80, color=COLORS['gray'], linestyle='--', alpha=0.3, linewidth=0.8)

    ax = axes[0, 1]
    ax.bar(layers, LAYER_DATA['dead'], color=COLORS['red'], alpha=0.7, width=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Dead features (of 4608)')
    ax.set_title('B  Dead features by layer', fontweight='bold', loc='left')
    ax.set_xticks([0, 3, 6, 9, 12, 15, 17])

    ax = axes[1, 0]
    ax.plot(layers, LAYER_DATA['ann_pct'], 's-', color=COLORS['green'],
            markersize=4, linewidth=1.5)
    ax.fill_between(layers, LAYER_DATA['ann_pct'], alpha=0.15, color=COLORS['green'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Annotated features (%)')
    ax.set_title('C  Biological annotation rate', fontweight='bold', loc='left')
    ax.set_ylim(42, 62)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
    min_idx = np.argmin(LAYER_DATA['ann_pct'])
    ax.annotate(f'L{min_idx}: {LAYER_DATA["ann_pct"][min_idx]}%',
                xy=(min_idx, LAYER_DATA['ann_pct'][min_idx]),
                xytext=(min_idx+2.5, LAYER_DATA['ann_pct'][min_idx]-2),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray']),
                fontsize=7, color=COLORS['gray'])

    ax = axes[1, 1]
    ax.bar(layers, LAYER_DATA['svd_aligned'], color=COLORS['orange'], alpha=0.8,
           label='SVD-aligned', width=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('SVD-aligned features')
    ax.set_title('D  SVD-aligned features (of ~4600)', fontweight='bold', loc='left')
    ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
    ax.set_ylim(0, 50)
    ax.text(0.98, 0.95, '99.8% novel\nacross all layers',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color=COLORS['gray'], style='italic')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig1_atlas_overview.pdf')
    fig.savefig(FIGDIR / 'fig1_atlas_overview.png')
    plt.close(fig)
    print("  Fig 1: atlas overview")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: SVD vs SAE — FIXED: Replace useless pie with variance bar
# ══════════════════════════════════════════════════════════════════════════

def fig2_svd_comparison():
    svd = load_svd_comparison()
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    layers = list(range(18))

    # Panel A: SVD-aligned count per layer (unchanged)
    ax = axes[0]
    svd_counts = [svd['per_layer'][str(l)]['feature_counts']['n_svd_aligned'] for l in layers]
    ax.bar(layers, svd_counts, color=COLORS['orange'], alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_title('A  SVD-aligned features', fontweight='bold', loc='left')
    ax.set_xticks([0, 5, 10, 15, 17])

    # Panel B: Annotation rate comparison (unchanged)
    ax = axes[1]
    total_ann_svd = sum(svd['per_layer'][str(l)]['annotation_rates']['n_annotated_svd'] for l in layers)
    total_svd = sum(svd['per_layer'][str(l)]['feature_counts']['n_svd_aligned'] for l in layers)
    total_ann_novel = sum(svd['per_layer'][str(l)]['annotation_rates']['n_annotated_novel'] for l in layers)
    total_novel = sum(svd['per_layer'][str(l)]['feature_counts']['n_novel'] for l in layers)
    ann_svd_pct = total_ann_svd / max(total_svd, 1) * 100
    ann_novel_pct = total_ann_novel / max(total_novel, 1) * 100
    bars = ax.bar([f'SVD-aligned\n({total_svd})', f'Novel\n({total_novel:,})'],
                  [ann_svd_pct, ann_novel_pct],
                  color=[COLORS['orange'], COLORS['blue']], alpha=0.8, width=0.5)
    ax.set_ylabel('Annotation rate (%)')
    ax.set_title('B  Annotation rate', fontweight='bold', loc='left')
    for bar, val in zip(bars, [ann_svd_pct, ann_novel_pct]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # Panel C: FIXED — Replace pie with variance explained comparison (SAE vs SVD)
    ax = axes[2]
    # Use per-layer variance data from svd comparison
    sae_var = [svd['per_layer'][str(l)]['variance_explained']['sae_4x_k32'] * 100
               for l in layers]
    svd_var = [svd['per_layer'][str(l)]['variance_explained']['svd_top50'] * 100
               for l in layers]
    ax.plot(layers, sae_var, 'o-', color=COLORS['blue'], markersize=3, linewidth=1.5,
            label='SAE')
    ax.plot(layers, svd_var, 's-', color=COLORS['orange'], markersize=3, linewidth=1.5,
            label='SVD (top-50)')
    ax.fill_between(layers, svd_var, sae_var, alpha=0.1, color=COLORS['blue'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('C  SAE vs SVD variance', fontweight='bold', loc='left')
    ax.set_xticks([0, 5, 10, 15, 17])
    ax.legend(fontsize=7, loc='lower left')
    # Add 2.4x annotation
    mid = 9
    gap = sae_var[mid] - svd_var[mid]
    ax.annotate(f'{sae_var[mid]/svd_var[mid]:.1f}×',
                xy=(mid, (sae_var[mid] + svd_var[mid])/2),
                fontsize=8, ha='center', color=COLORS['blue'], fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig2_svd_comparison.pdf')
    fig.savefig(FIGDIR / 'fig2_svd_comparison.png')
    plt.close(fig)
    print("  Fig 2: SVD comparison (FIXED: replaced pie with variance)")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Per-ontology Enrichment Heatmap + U-shape (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def fig3_ontology_heatmap():
    fig = plt.figure(figsize=(7, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.35)

    ax_heat = fig.add_subplot(gs[0])
    ontologies = ['GO_BP', 'Reactome', 'KEGG', 'STRING', 'TRRUST_TF', 'TRRUST_edges']
    display_names = ['GO BP', 'Reactome', 'KEGG', 'STRING', 'TRRUST TF', 'TRRUST edges']

    matrix = []
    for ont in ontologies:
        vals = np.array(LAYER_DATA[ont], dtype=float)
        matrix.append((vals - vals.min()) / (vals.max() - vals.min() + 1e-10))
    matrix = np.array(matrix)

    im = ax_heat.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_heat.set_xticks(range(18))
    ax_heat.set_xticklabels([str(i) for i in range(18)], fontsize=7)
    ax_heat.set_yticks(range(len(display_names)))
    ax_heat.set_yticklabels(display_names)
    ax_heat.set_xlabel('Layer')
    ax_heat.set_title('A  Per-ontology enrichment (row-normalized)', fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label='Relative enrichment')

    ax_bar = fig.add_subplot(gs[1])
    totals = []
    for i in range(18):
        t = sum(LAYER_DATA[ont][i] for ont in ontologies)
        totals.append(t)
    ax_bar.barh(range(18), totals, color=COLORS['teal'], alpha=0.7, height=0.8)
    ax_bar.set_yticks(range(18))
    ax_bar.set_yticklabels([str(i) for i in range(18)], fontsize=7)
    ax_bar.set_xlabel('Total enrichments')
    ax_bar.set_title('B  Total per layer', fontweight='bold', loc='left')
    ax_bar.invert_yaxis()

    fig.savefig(FIGDIR / 'fig3_ontology_heatmap.pdf')
    fig.savefig(FIGDIR / 'fig3_ontology_heatmap.png')
    plt.close(fig)
    print("  Fig 3: ontology heatmap")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Co-activation Modules (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def fig4_coactivation_modules():
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    layers = COACT_DATA['layers']

    ax = axes[0]
    ax.bar(layers, COACT_DATA['modules'], color=COLORS['purple'], alpha=0.7, width=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of modules')
    ax.set_title('A  Modules per layer', fontweight='bold', loc='left')
    ax.set_xticks([0, 5, 10, 15, 17])
    ax.axhline(np.mean(COACT_DATA['modules']), color=COLORS['gray'], ls='--', lw=0.8)

    ax = axes[1]
    ax.plot(layers, COACT_DATA['coverage'], 'o-', color=COLORS['teal'],
            markersize=3, linewidth=1.5)
    ax.fill_between(layers, 95, COACT_DATA['coverage'], alpha=0.1, color=COLORS['teal'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Feature coverage (%)')
    ax.set_title('B  Module coverage', fontweight='bold', loc='left')
    ax.set_ylim(95, 100)
    ax.set_xticks([0, 5, 10, 15, 17])

    ax = axes[2]
    edges_k = [e/1000 for e in COACT_DATA['pmi_edges']]
    ax.plot(layers, edges_k, 's-', color=COLORS['red'], markersize=3, linewidth=1.5)
    ax.fill_between(layers, edges_k, alpha=0.1, color=COLORS['red'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('PMI edges (×1000)')
    ax.set_title('C  Co-activation density', fontweight='bold', loc='left')
    ax.set_xticks([0, 5, 10, 15, 17])

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig4_coactivation.pdf')
    fig.savefig(FIGDIR / 'fig4_coactivation.png')
    plt.close(fig)
    print("  Fig 4: co-activation modules")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Causal Patching — FIXED: legend placement
# ══════════════════════════════════════════════════════════════════════════

def fig5_causal_patching():
    causal = load_causal()
    features = causal['feature_results']

    fig, axes = plt.subplots(1, 3, figsize=(7, 3.0))

    # Panel A: FIXED — move legend below axis, use loc='upper right' outside data
    ax = axes[0]
    ratios = [f['specificity_ratio'] for f in features if f['specificity_ratio'] is not None]
    ratios_clipped = np.clip(ratios, 0, 50)
    ax.hist(ratios_clipped, bins=25, color=COLORS['blue'], alpha=0.7, edgecolor='white')
    ax.axvline(1.0, color=COLORS['red'], linestyle='--', linewidth=1.2, label='1× (baseline)')
    ax.axvline(np.median(ratios), color=COLORS['green'], linestyle='-', linewidth=1.2,
               label=f'Median: {np.median(ratios):.1f}×')
    ax.set_xlabel('Specificity ratio')
    ax.set_ylabel('Features')
    ax.set_title('A  Specificity distribution', fontweight='bold', loc='left')
    # FIXED: place legend in upper right with small font, no frame overlap
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9,
              edgecolor='none', facecolor='white')

    # Panel B: unchanged
    ax = axes[1]
    target_diffs = [f['target_logit_diff_mean'] for f in features]
    other_diffs = [f['other_logit_diff_mean'] for f in features]
    ratios_raw = [f['specificity_ratio'] if f['specificity_ratio'] else 1 for f in features]
    sc = ax.scatter(other_diffs, target_diffs, c=np.log10(np.clip(ratios_raw, 0.1, 200)),
                    cmap='RdYlBu_r', s=25, alpha=0.8, edgecolors='gray', linewidths=0.3)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Other genes logit diff')
    ax.set_ylabel('Target genes logit diff')
    ax.set_title('B  Target vs other effect', fontweight='bold', loc='left')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='log₁₀(specificity)')

    # Panel C: unchanged
    ax = axes[2]
    sorted_feats = sorted(features, key=lambda x: x['specificity_ratio'] or 0, reverse=True)[:10]
    labels = [f['label'][:20] if f['label'] else f"F{f['feature_idx']}" for f in sorted_feats]
    vals = [f['specificity_ratio'] for f in sorted_feats]
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, vals, color=COLORS['blue'], alpha=0.7, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel('Specificity ratio')
    ax.set_title('C  Top 10 causal features', fontweight='bold', loc='left')
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}×', va='center', fontsize=6.5)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig5_causal_patching.pdf')
    fig.savefig(FIGDIR / 'fig5_causal_patching.png')
    plt.close(fig)
    print("  Fig 5: causal patching (FIXED: legend placement)")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cross-Layer — FIXED: overlapping panel titles
# ══════════════════════════════════════════════════════════════════════════

def fig6_cross_layer():
    xlt = load_cross_layer()

    # FIXED: increase figure width and use gridspec for better spacing
    fig = plt.figure(figsize=(7.5, 2.8))
    gs = gridspec.GridSpec(1, 3, wspace=0.45)  # More horizontal space between panels

    # Panel A: Adjacent persistence rate
    ax = fig.add_subplot(gs[0])
    adj = xlt['adjacent_persistence']
    from_layers = [d['from_layer'] for d in adj]
    rates = [d['persistence_rate']*100 for d in adj]
    ax.plot(from_layers, rates, 'o-', color=COLORS['blue'], markersize=3, linewidth=1.5)
    ax.fill_between(from_layers, rates, alpha=0.15, color=COLORS['blue'])
    ax.set_xlabel('Layer transition (L → L+1)')
    ax.set_ylabel('Persistence rate (%)')
    # FIXED: shorter title to avoid overlap
    ax.set_title('A  Adjacent persistence', fontweight='bold', loc='left')
    ax.set_ylim(0, 5)
    ax.set_xticks([0, 4, 8, 12, 16])

    # Panel B: Long-range decay from L0
    ax = fig.add_subplot(gs[1])
    lr = xlt['long_range_persistence']
    target_layers = [d['target_layer'] for d in lr]
    lr_rates = [d['persistence_rate']*100 for d in lr]
    ax.plot(target_layers, lr_rates, 's-', color=COLORS['red'], markersize=3, linewidth=1.5)
    ax.fill_between(target_layers, lr_rates, alpha=0.1, color=COLORS['red'])
    ax.set_xlabel('Target layer')
    ax.set_ylabel('L0 features matched (%)')
    ax.set_title('B  L0 feature decay', fontweight='bold', loc='left')
    ax.set_xticks([0, 4, 8, 12, 16])

    # Panel C: Cross-layer highways
    ax = fig.add_subplot(gs[2])
    highway_data = {
        'L0→L5': {'pct': 98.4, 'mean_pmi': 6.61},
        'L5→L11': {'pct': 97.4, 'mean_pmi': 6.63},
        'L11→L17': {'pct': 99.8, 'mean_pmi': 6.79},
    }
    pairs = list(highway_data.keys())
    pcts = [highway_data[p]['pct'] for p in pairs]
    mean_pmis = [highway_data[p]['mean_pmi'] for p in pairs]

    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, pcts, w, color=COLORS['teal'], alpha=0.7, label='% highways')
    ax2 = ax.twinx()
    ax2.bar(x + w/2, mean_pmis, w, color=COLORS['orange'], alpha=0.7, label='Mean max PMI')
    ax2.spines['right'].set_visible(True)

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=7)
    ax.set_ylabel('Highways (%)')
    ax.set_ylim(95, 100.5)
    ax2.set_ylabel('Mean max PMI')
    ax2.set_ylim(6, 7.2)
    ax.set_title('C  Information highways', fontweight='bold', loc='left')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc='lower left')

    fig.savefig(FIGDIR / 'fig6_cross_layer.pdf')
    fig.savefig(FIGDIR / 'fig6_cross_layer.png')
    plt.close(fig)
    print("  Fig 6: cross-layer (FIXED: panel title overlap)")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Perturbation Response (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def fig7_perturbation():
    pert = load_perturbation()

    fig, axes = plt.subplots(1, 3, figsize=(7, 3.0))

    ax = axes[0]
    n_total = pert['summary']['n_targets_processed']
    n_responding = sum(1 for t in pert['target_results'] if t['n_responding_features'] > 0)
    n_tfs = pert['summary']['n_trrust_tfs']
    n_specific = pert['summary']['n_tfs_with_specific_response']

    sizes = [n_responding, n_total - n_responding]
    ax.pie(sizes, labels=[f'Detected\n({n_responding})', f'No response\n({n_total - n_responding})'],
           colors=[COLORS['blue'], COLORS['light_blue']],
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
    ax.set_title('A  Perturbation detection', fontweight='bold', loc='left')

    ax = axes[1]
    bars = ax.bar(['Non-specific\nTFs', 'Specific\nTFs'],
                  [n_tfs - n_specific, n_specific],
                  color=[COLORS['gray'], COLORS['red']], alpha=0.8, width=0.5)
    ax.set_ylabel(f'TRRUST TFs (of {n_tfs})')
    ax.set_title('B  Regulatory specificity', fontweight='bold', loc='left')
    for bar, val in zip(bars, [n_tfs - n_specific, n_specific]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val} ({val/n_tfs*100:.1f}%)', ha='center', va='bottom', fontsize=8)

    ax = axes[2]
    n_resp_list = [t['n_responding_features'] for t in pert['target_results']]
    ax.hist(n_resp_list, bins=range(0, max(n_resp_list)+2), color=COLORS['purple'],
            alpha=0.7, edgecolor='white')
    ax.set_xlabel('Responding features per target')
    ax.set_ylabel('Perturbation targets')
    ax.set_title('C  Response breadth', fontweight='bold', loc='left')
    ax.axvline(np.mean(n_resp_list), color=COLORS['red'], ls='--', lw=1,
               label=f'Mean: {np.mean(n_resp_list):.1f}')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig7_perturbation.pdf')
    fig.savefig(FIGDIR / 'fig7_perturbation.png')
    plt.close(fig)
    print("  Fig 7: perturbation response")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Multi-Tissue SAE Comparison (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def fig8_multitissue():
    comp = load_comparison()

    fig, axes = plt.subplots(1, 3, figsize=(7, 3.0))

    ax = axes[0]
    mt_layers = [0, 5, 11, 17]
    mt_spec = [comp['multi_tissue'][f'layer{l:02d}']['specificity_rate']*100 for l in mt_layers]
    k562_spec = comp['k562_only']['specificity_rate'] * 100

    ax.bar(range(4), mt_spec, color=COLORS['blue'], alpha=0.7, width=0.6,
           label='Multi-tissue SAE')
    ax.axhline(k562_spec, color=COLORS['red'], linestyle='--', linewidth=1.2,
               label=f'K562-only L11 ({k562_spec:.1f}%)')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'L{l}' for l in mt_layers])
    ax.set_ylabel('TF specificity (%)')
    ax.set_title('A  Specificity by layer', fontweight='bold', loc='left')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 15)

    ax = axes[1]
    per_tf = comp['per_tf_comparison_layer11']
    gained = sum(1 for t in per_tf if not t['k562_specific'] and t['mt_specific'])
    lost = sum(1 for t in per_tf if t['k562_specific'] and not t['mt_specific'])
    unchanged_spec = sum(1 for t in per_tf if t['k562_specific'] and t['mt_specific'])
    unchanged_non = sum(1 for t in per_tf if not t['k562_specific'] and not t['mt_specific'])

    categories = ['Gained', 'Lost', 'Both\nspecific', 'Neither\nspecific']
    counts = [gained, lost, unchanged_spec, unchanged_non]
    colors_bar = [COLORS['green'], COLORS['red'], COLORS['blue'], COLORS['gray']]
    ax.bar(categories, counts, color=colors_bar, alpha=0.7, width=0.6)
    ax.set_ylabel('Number of TFs')
    ax.set_title('B  Per-TF comparison (L11)', fontweight='bold', loc='left')
    for i, (cat, cnt) in enumerate(zip(categories, counts)):
        ax.text(i, cnt + 0.5, str(cnt), ha='center', va='bottom', fontsize=8)

    ax = axes[2]
    k562_rate = comp['k562_only']['specificity_rate'] * 100
    best_mt_rate = comp['multi_tissue']['layer11']['specificity_rate'] * 100
    k562_n = comp['k562_only']['n_tfs_specific']
    best_mt_n = comp['multi_tissue']['layer11']['n_tfs_specific']

    x = np.arange(2)
    bars = ax.bar(x, [k562_rate, best_mt_rate],
                  color=[COLORS['red'], COLORS['blue']], alpha=0.7, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['K562-only\nSAE (L11)', 'Multi-tissue\nSAE (L11)'], fontsize=8)
    ax.set_ylabel('TF specificity (%)')
    ax.set_title('C  Best layer comparison', fontweight='bold', loc='left')
    ax.set_ylim(0, 15)
    for bar, val, n in zip(bars, [k562_rate, best_mt_rate], [k562_n, best_mt_n]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{n}/48\n({val:.1f}%)', ha='center', va='bottom', fontsize=7.5)
    ax.annotate(f'+{best_mt_rate - k562_rate:.1f}pp',
                xy=(1, best_mt_rate), xytext=(0.5, best_mt_rate + 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['green']),
                fontsize=8, color=COLORS['green'], fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig8_multitissue.pdf')
    fig.savefig(FIGDIR / 'fig8_multitissue.png')
    plt.close(fig)
    print("  Fig 8: multi-tissue comparison")


# ══════════════════════════════════════════════════════════════════════════
# NEW FIGURE S1: scGPT Atlas Overview (4-panel, mirrors Fig 1)
# ══════════════════════════════════════════════════════════════════════════

def figS1_scgpt_atlas_overview():
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    layers = SCGPT_DATA['layers']

    # Panel A: Variance Explained
    ax = axes[0, 0]
    ax.plot(layers, [v*100 for v in SCGPT_DATA['var_expl']], 'o-',
            color=COLORS['blue'], markersize=4, linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('A  scGPT reconstruction quality', fontweight='bold', loc='left')
    ax.set_ylim(83, 95)
    ax.set_xticks(range(0, 12, 2))
    ax.axhline(90, color=COLORS['gray'], linestyle='--', alpha=0.3, linewidth=0.8)

    # Panel B: Dead features
    ax = axes[0, 1]
    ax.bar(layers, SCGPT_DATA['dead'], color=COLORS['red'], alpha=0.7, width=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Dead features (of 2048)')
    ax.set_title('B  Dead features by layer', fontweight='bold', loc='left')
    ax.set_xticks(range(0, 12, 2))
    total_dead = sum(SCGPT_DATA['dead'])
    ax.text(0.98, 0.95, f'Only {total_dead} dead\nacross all layers',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color=COLORS['gray'], style='italic')

    # Panel C: Annotation rate
    ax = axes[1, 0]
    ax.plot(layers, SCGPT_DATA['ann_pct'], 's-', color=COLORS['green'],
            markersize=4, linewidth=1.5)
    ax.fill_between(layers, SCGPT_DATA['ann_pct'], alpha=0.15, color=COLORS['green'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Annotated features (%)')
    ax.set_title('C  Biological annotation rate', fontweight='bold', loc='left')
    ax.set_ylim(25, 38)
    ax.set_xticks(range(0, 12, 2))

    # Panel D: Modules per layer
    ax = axes[1, 1]
    ax.bar(layers, SCGPT_DATA['modules'], color=COLORS['purple'], alpha=0.7, width=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of modules')
    ax.set_title('D  Co-activation modules', fontweight='bold', loc='left')
    ax.set_xticks(range(0, 12, 2))
    ax.axhline(np.mean(SCGPT_DATA['modules']), color=COLORS['gray'], ls='--', lw=0.8)
    ax.text(0.98, 0.95, f'76 modules total\n(mean {np.mean(SCGPT_DATA["modules"]):.1f}/layer)',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color=COLORS['gray'], style='italic')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'figS1_scgpt_overview.pdf')
    fig.savefig(FIGDIR / 'figS1_scgpt_overview.png')
    plt.close(fig)
    print("  Fig S1: scGPT atlas overview")


# ══════════════════════════════════════════════════════════════════════════
# NEW FIGURE S2: Cross-Model Comparison (Geneformer vs scGPT)
# ══════════════════════════════════════════════════════════════════════════

def figS2_cross_model_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    # Panel A: Variance explained — both models on same plot
    ax = axes[0, 0]
    gf_layers_norm = np.array(LAYER_DATA['layers']) / 17  # Normalize to [0,1]
    sc_layers_norm = np.array(SCGPT_DATA['layers']) / 11
    ax.plot(gf_layers_norm, [v*100 for v in LAYER_DATA['var_expl']], 'o-',
            color=COLORS['blue'], markersize=3, linewidth=1.5, label='Geneformer (18L)')
    ax.plot(sc_layers_norm, [v*100 for v in SCGPT_DATA['var_expl']], 's-',
            color=COLORS['orange'], markersize=3, linewidth=1.5, label='scGPT (12L)')
    ax.set_xlabel('Relative depth (0=input, 1=output)')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('A  Reconstruction quality', fontweight='bold', loc='left')
    ax.legend(fontsize=7)
    ax.set_ylim(74, 95)

    # Panel B: Annotation rate — both models
    ax = axes[0, 1]
    ax.plot(gf_layers_norm, LAYER_DATA['ann_pct'], 'o-',
            color=COLORS['blue'], markersize=3, linewidth=1.5, label='Geneformer')
    ax.plot(sc_layers_norm, SCGPT_DATA['ann_pct'], 's-',
            color=COLORS['orange'], markersize=3, linewidth=1.5, label='scGPT')
    ax.set_xlabel('Relative depth')
    ax.set_ylabel('Annotated features (%)')
    ax.set_title('B  Annotation rate', fontweight='bold', loc='left')
    ax.legend(fontsize=7)

    # Panel C: Summary comparison bar chart
    ax = axes[1, 0]
    metrics = ['VarExpl\n(mean)', 'Alive\n(%)', 'Annotated\n(%)', 'Modules\n/layer']
    gf_vals = [np.mean(LAYER_DATA['var_expl'])*100,
               sum(LAYER_DATA['alive'])/len(LAYER_DATA['alive'])/4608*100,
               52.4,
               np.mean(COACT_DATA['modules'])]
    sc_vals = [np.mean(SCGPT_DATA['var_expl'])*100,
               sum(SCGPT_DATA['alive'])/len(SCGPT_DATA['alive'])/2048*100,
               31.0,
               np.mean(SCGPT_DATA['modules'])]

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, gf_vals, w, color=COLORS['blue'], alpha=0.7, label='Geneformer')
    ax.bar(x + w/2, sc_vals, w, color=COLORS['orange'], alpha=0.7, label='scGPT')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel('Value')
    ax.set_title('C  Key metrics comparison', fontweight='bold', loc='left')
    ax.legend(fontsize=7)

    # Panel D: Dead features comparison
    ax = axes[1, 1]
    ax.plot(gf_layers_norm, LAYER_DATA['dead'], 'o-',
            color=COLORS['blue'], markersize=3, linewidth=1.5, label='Geneformer')
    ax.plot(sc_layers_norm, SCGPT_DATA['dead'], 's-',
            color=COLORS['orange'], markersize=3, linewidth=1.5, label='scGPT')
    ax.set_xlabel('Relative depth')
    ax.set_ylabel('Dead features')
    ax.set_title('D  Dead features', fontweight='bold', loc='left')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'figS2_cross_model.pdf')
    fig.savefig(FIGDIR / 'figS2_cross_model.png')
    plt.close(fig)
    print("  Fig S2: cross-model comparison")


# ══════════════════════════════════════════════════════════════════════════
# NEW FIGURE S3: scGPT Cross-Layer Highways + Progressive Concentration
# ══════════════════════════════════════════════════════════════════════════

def figS3_scgpt_highways():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))

    # Panel A: Upstream vs downstream connectivity
    ax = axes[0]
    pairs = list(SCGPT_HIGHWAYS.keys())
    upstream_pct = [SCGPT_HIGHWAYS[p]['upstream']/SCGPT_HIGHWAYS[p]['upstream_total']*100
                    for p in pairs]
    downstream_pct = [SCGPT_HIGHWAYS[p]['downstream']/SCGPT_HIGHWAYS[p]['downstream_total']*100
                      for p in pairs]

    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, upstream_pct, w, color=COLORS['blue'], alpha=0.7, label='Upstream')
    ax.bar(x + w/2, downstream_pct, w, color=COLORS['orange'], alpha=0.7, label='Downstream')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=7)
    ax.set_ylabel('Connected features (%)')
    ax.set_title('A  scGPT connectivity', fontweight='bold', loc='left')
    ax.legend(fontsize=7)
    ax.set_ylim(50, 100)
    # Annotate the drop
    ax.annotate(f'{downstream_pct[-1]:.0f}%', xy=(2 + w/2, downstream_pct[-1]),
                xytext=(2 + w/2, downstream_pct[-1] - 8),
                ha='center', fontsize=7, color=COLORS['red'], fontweight='bold')

    # Panel B: PMI edges declining
    ax = axes[1]
    edges_k = [SCGPT_HIGHWAYS[p]['pmi_edges']/1000 for p in pairs]
    ax.bar(range(len(pairs)), edges_k, color=COLORS['teal'], alpha=0.7, width=0.6)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=7)
    ax.set_ylabel('PMI edges (×1000)')
    ax.set_title('B  Edge density', fontweight='bold', loc='left')
    for i, v in enumerate(edges_k):
        ax.text(i, v + 1, f'{v:.0f}K', ha='center', fontsize=7)

    # Panel C: Geneformer vs scGPT downstream connectivity comparison
    ax = axes[2]
    gf_downstream = [98.4, 97.4, 99.8]  # From Geneformer data
    sc_downstream = downstream_pct
    gf_pairs = ['L0→L5', 'L5→L11', 'L11→L17']
    sc_pairs = pairs

    ax.plot([0, 1, 2], gf_downstream, 'o-', color=COLORS['blue'],
            markersize=5, linewidth=1.5, label='Geneformer')
    ax.plot([0, 1, 2], sc_downstream, 's-', color=COLORS['orange'],
            markersize=5, linewidth=1.5, label='scGPT')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Early→Mid', 'Mid→Late', 'Late→Output'], fontsize=7)
    ax.set_ylabel('Downstream connectivity (%)')
    ax.set_title('C  Progressive concentration', fontweight='bold', loc='left')
    ax.legend(fontsize=7)
    ax.set_ylim(55, 105)
    # Highlight the divergence
    ax.fill_between([0, 1, 2], sc_downstream, gf_downstream,
                    alpha=0.1, color=COLORS['red'])

    plt.tight_layout()
    fig.savefig(FIGDIR / 'figS3_scgpt_highways.pdf')
    fig.savefig(FIGDIR / 'figS3_scgpt_highways.png')
    plt.close(fig)
    print("  Fig S3: scGPT cross-layer highways")


# ══════════════════════════════════════════════════════════════════════════
# NEW FIGURE S4: scGPT Ontology Enrichment Heatmap
# ══════════════════════════════════════════════════════════════════════════

def figS4_scgpt_ontology_heatmap():
    fig = plt.figure(figsize=(7, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.35)

    # Panel A: Heatmap
    ax_heat = fig.add_subplot(gs[0])
    ontologies = ['GO_BP', 'Reactome', 'KEGG', 'STRING', 'TRRUST']
    display_names = ['GO BP', 'Reactome', 'KEGG', 'STRING', 'TRRUST']

    matrix = []
    for ont in ontologies:
        vals = np.array(SCGPT_DATA[ont], dtype=float)
        matrix.append((vals - vals.min()) / (vals.max() - vals.min() + 1e-10))
    matrix = np.array(matrix)

    im = ax_heat.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels([str(i) for i in range(12)], fontsize=7)
    ax_heat.set_yticks(range(len(display_names)))
    ax_heat.set_yticklabels(display_names)
    ax_heat.set_xlabel('Layer')
    ax_heat.set_title('A  scGPT per-ontology enrichment', fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label='Relative enrichment')

    # Panel B: Total enrichments per layer
    ax_bar = fig.add_subplot(gs[1])
    totals = []
    for i in range(12):
        t = sum(SCGPT_DATA[ont][i] for ont in ontologies)
        totals.append(t)
    ax_bar.barh(range(12), totals, color=COLORS['teal'], alpha=0.7, height=0.8)
    ax_bar.set_yticks(range(12))
    ax_bar.set_yticklabels([str(i) for i in range(12)], fontsize=7)
    ax_bar.set_xlabel('Total enrichments')
    ax_bar.set_title('B  Total per layer', fontweight='bold', loc='left')
    ax_bar.invert_yaxis()

    fig.savefig(FIGDIR / 'figS4_scgpt_ontology.pdf')
    fig.savefig(FIGDIR / 'figS4_scgpt_ontology.png')
    plt.close(fig)
    print("  Fig S4: scGPT ontology heatmap")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating figures (v2)...")
    print("\n--- Original figures (with fixes) ---")
    fig1_atlas_overview()
    fig2_svd_comparison()
    fig3_ontology_heatmap()
    fig4_coactivation_modules()
    fig5_causal_patching()
    fig6_cross_layer()
    fig7_perturbation()
    fig8_multitissue()
    print("\n--- New scGPT figures ---")
    figS1_scgpt_atlas_overview()
    figS2_cross_model_comparison()
    figS3_scgpt_highways()
    figS4_scgpt_ontology_heatmap()
    print(f"\nAll figures saved to {FIGDIR}")
