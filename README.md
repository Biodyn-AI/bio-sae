# Sparse Autoencoders Reveal Organized Biological Knowledge but Minimal Regulatory Logic in Single-Cell Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the complete analysis pipeline and interactive web atlases for applying sparse autoencoders (SAEs) to decompose the residual streams of two single-cell foundation models — **Geneformer V2-316M** (18 layers, d=1,152) and **scGPT whole-human** (12 layers, d=512, 33M training cells) — producing atlases of **82,525** and **24,527** interpretable biological features, respectively.

**Paper**: [Kendiukhov (2025) — Genome Biology submission]

**Interactive Atlases**:
- Geneformer Feature Atlas: https://biodyn-ai.github.io/geneformer-atlas/
- scGPT Feature Atlas: https://biodyn-ai.github.io/scgpt-atlas/

## Key Findings

- **Massive superposition**: 99.8% of SAE features are invisible to SVD; models compress 82,525+ biological concepts into 1,152 dimensions (>70x compression ratio)
- **Rich biological annotation**: 52.4% (Geneformer) and 31.0% (scGPT) of features annotate to Gene Ontology, KEGG, Reactome, STRING, and TRRUST databases
- **U-shaped annotation profile**: Biological interpretability follows a characteristic U-shape across layers, reflecting hierarchical abstraction from molecular programs to abstract computation to integrative programs
- **Modular organization**: 141 co-activation modules (Geneformer) and 76 modules (scGPT) with 96-99.5% feature coverage
- **Causal specificity**: Median 2.36x specificity at feature level (vs. null at attention head level), with top feature at 114.5x
- **Cross-layer information highways**: 97-99.8% of features participate despite complete representational turnover between layers
- **Minimal regulatory logic**: Only 6.2% of transcription factors (3/48) show regulatory-target-specific feature responses; multi-tissue SAE yields marginal improvement (10.4%, 5/48), establishing model representations as the bottleneck
- **Cross-model convergence**: Both architecturally distinct models develop qualitatively similar feature organization, confirming universal superposition in biological foundation models

## Repository Structure

```
bio-sae/
├── README.md                          # This file
├── LICENSE                            # MIT license
├── requirements.txt                   # Python dependencies
│
├── src/                               # Geneformer SAE pipeline
│   ├── sae_model.py                   # Core TopK SAE architecture + trainer
│   ├── 01_extract_activations.py      # Extract K562 residual stream activations
│   ├── 01b_extract_remaining_layers.py # Complete extraction for all 18 layers
│   ├── 02_train_sae.py               # Train TopK SAE (d=1152, 4608 features, k=32)
│   ├── 02b_train_all_layers.py       # Orchestrate training across all layers
│   ├── 03_analyze_features.py         # Feature analysis (top genes, SVD comparison)
│   ├── 04_annotate_features.py        # Ontology annotation (GO, KEGG, Reactome, STRING, TRRUST)
│   ├── 05_compare_svd.py             # Systematic SAE vs SVD comparison
│   ├── 06_cross_layer.py             # Cross-layer feature persistence tracking
│   ├── 07_feature_coactivation.py     # PMI-based co-activation modules (Leiden clustering)
│   ├── 08_causal_patching.py          # Single-feature causal ablation experiments
│   ├── 09_perturbation_response.py    # CRISPRi perturbation response mapping
│   ├── 10_novel_features.py           # Characterization of unannotated features
│   ├── 11_computational_graph.py      # Cross-layer PMI information highways
│   ├── 12a_extract_tabula_sapiens.py  # Multi-tissue activation extraction
│   ├── 12b_pool_and_train.py          # Multi-tissue SAE training
│   ├── 12c_analyze_and_annotate.py    # Multi-tissue feature analysis
│   ├── 12d_perturbation_test.py       # Multi-tissue perturbation specificity test
│   └── 12e_compare_results.py         # K562-only vs multi-tissue comparison
│
├── scgpt_src/                         # scGPT SAE pipeline (parallel analysis)
│   ├── 01_extract_activations.py      # Extract scGPT residual stream activations
│   ├── 02_train_sae.py               # Train TopK SAE (d=512, 2048 features, k=32)
│   ├── 03_analyze_features.py         # scGPT feature analysis
│   ├── 04_annotate_features.py        # scGPT ontology annotation
│   ├── 07_coactivation.py            # scGPT co-activation modules
│   ├── 08_causal_patching.py          # scGPT causal patching
│   ├── 11_computational_graph.py      # scGPT cross-layer highways
│   └── compute_celltype_enrichments.py # Cell type enrichment analysis
│
├── atlas/                             # Geneformer Feature Atlas (React web app)
│   ├── src/                           # React + TypeScript source code
│   ├── scripts/                       # Data preprocessing scripts
│   ├── package.json                   # Dependencies
│   └── vite.config.ts                 # Build configuration
│
├── scgpt_atlas/                       # scGPT Feature Atlas (React web app)
│   ├── src/                           # React + TypeScript source code
│   ├── scripts/                       # Data preprocessing scripts
│   ├── package.json
│   └── vite.config.ts
│
├── paper/                             # Manuscript files
│   ├── sae_paper_v2.tex               # arXiv preprint version
│   ├── genome_biology/                # Genome Biology submission
│   ├── references_v2.bib             # Bibliography
│   ├── generate_figures_v2.py         # Figure generation script
│   └── figures/                       # Generated figure PDFs
│
└── docs/                              # Detailed results documentation
    ├── results_phase1_geneformer_k562.md
    ├── results_phase2_geneformer_advanced.md
    ├── results_phase3_multitissue.md
    └── knowledge_map_summary.md
```

## Pipeline Overview

The analysis proceeds in three phases, applied identically to both models:

### Phase 1: Feature Extraction and Annotation
1. **Activation extraction** (`01_extract_activations.py`): Extract per-position residual stream activations from all transformer layers
2. **SAE training** (`02_train_sae.py`): Train TopK sparse autoencoders (4x overcomplete, k=32)
3. **Feature analysis** (`03_analyze_features.py`): Identify top-activating genes, compute statistics
4. **Ontology annotation** (`04_annotate_features.py`): Fisher's exact test against GO, KEGG, Reactome, STRING, TRRUST
5. **SVD comparison** (`05_compare_svd.py`): Quantify superposition (SAE vs top-50 SVD axes)
6. **Cross-layer tracking** (`06_cross_layer.py`): Track feature persistence via decoder cosine similarity

### Phase 2: Advanced Analysis
7. **Co-activation modules** (`07_feature_coactivation.py`): PMI-based co-activation graphs + Leiden clustering
8. **Causal patching** (`08_causal_patching.py`): Single-feature ablation measuring logit specificity
9. **Perturbation response** (`09_perturbation_response.py`): CRISPRi knockdown response mapping
10. **Novel features** (`10_novel_features.py`): Characterize unannotated features (clustering, guilt-by-association)
11. **Computational graph** (`11_computational_graph.py`): Cross-layer PMI information highways

### Phase 3: Multi-tissue Control
12. **Multi-tissue SAE** (`12a-12e`): Pool K562 + Tabula Sapiens activations, train SAE, test perturbation specificity to determine whether model or training data is the bottleneck

## Requirements

### Hardware
- GPU recommended (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback)
- 32+ GB RAM for activation extraction
- ~400 GB disk space for full activation datasets

### Software
```bash
# Create conda environment
conda create -n bio-sae python=3.10
conda activate bio-sae

# Install dependencies
pip install -r requirements.txt
```

### Data Sources
- **Geneformer V2-316M**: [HuggingFace ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) (subfolder `Geneformer-V2-316M`)
- **scGPT whole-human**: Available from the [scGPT authors](https://github.com/bowang-lab/scGPT)
- **Replogle CRISPRi**: [Replogle et al. (2022)](https://doi.org/10.1016/j.cell.2022.05.013) -- K562 genome-scale Perturb-seq
- **Tabula Sapiens**: [The Tabula Sapiens Consortium (2022)](https://doi.org/10.1126/science.abl4896) -- multi-organ single-cell atlas
- **TRRUST v2**: [Han et al. (2018)](https://doi.org/10.1093/nar/gkx1013) -- human transcriptional regulatory network
- **Gene Ontology**: [Ashburner et al. (2000)](https://doi.org/10.1038/75556)
- **KEGG**: [Kanehisa & Goto (2000)](https://doi.org/10.1093/nar/28.1.27)
- **Reactome**: [Jassal et al. (2020)](https://doi.org/10.1093/nar/gkz1031)
- **STRING**: [Szklarczyk et al. (2023)](https://doi.org/10.1093/nar/gkac1000)

## Quick Start

```bash
# Note: Scripts contain hardcoded paths that need to be adjusted for your setup.
# See the BASE variable at the top of each script.

# Phase 1: Extract activations (requires Geneformer model)
python src/01_extract_activations.py

# Phase 1: Train SAE for a specific layer
python src/02_train_sae.py --layer 11 --expansion 4 --k 32

# Phase 1: Analyze and annotate features
python src/03_analyze_features.py --layer 11
python src/04_annotate_features.py --layer 11

# Phase 2: Co-activation modules
python src/07_feature_coactivation.py --layer 11

# Phase 2: Causal patching
python src/08_causal_patching.py --layer 11

# Train and analyze all 18 layers automatically
python src/02b_train_all_layers.py
```

## SAE Architecture

The core SAE (`src/sae_model.py`) implements a TopK sparse autoencoder:

- **Encoder**: Linear projection from d_model to n_features (4x overcomplete)
- **Sparsity**: TopK activation (k=32), keeping only the 32 largest activations per position
- **Decoder**: Unit-normalized linear projection back to d_model
- **Loss**: MSE reconstruction loss (no L1 penalty; TopK enforces exact sparsity)
- **Training**: Adam optimizer, lr=3e-4, batch size 4096, 5 epochs

The same architecture is used for both Geneformer (d=1152 to 4608 features) and scGPT (d=512 to 2048 features) without modification.

## Interactive Feature Atlases

Both atlases are deployed as single-page web applications:

- **Geneformer Feature Atlas**: https://biodyn-ai.github.io/geneformer-atlas/
  - 82,525 features across 18 layers
  - Source: https://github.com/Biodyn-AI/geneformer-atlas

- **scGPT Feature Atlas**: https://biodyn-ai.github.io/scgpt-atlas/
  - 24,527 features across 12 layers
  - Source: https://github.com/Biodyn-AI/scgpt-atlas

Features include: layer exploration, feature detail pages, co-activation module maps, cross-layer flow visualization, gene search, and ontology search.

## Citation

```bibtex
@article{kendiukhov2025sparse,
  title={Sparse Autoencoders Reveal Organized Biological Knowledge but Minimal Regulatory Logic in Single-Cell Foundation Models: A Comparative Atlas of Geneformer and scGPT},
  author={Kendiukhov, Ihor},
  journal={Genome Biology},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
