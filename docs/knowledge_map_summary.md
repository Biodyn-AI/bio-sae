# Complete Knowledge Map: What Does Geneformer Encode?

## The Big Picture

Across three phases of SAE analysis, we decomposed Geneformer V2-316M's internal representations into 82,525 interpretable features and mapped their biological identity, modular organization, causal roles, cross-layer connectivity, perturbation responses, and novelty. The answer to "what does it encode?" is rich and multi-layered.

**In short: Geneformer encodes a hierarchical biological knowledge base — from molecular-level gene programs at early layers, through abstract computational representations at middle layers, to integrative cellular programs at late layers. Approximately 45-59% of features have confirmed biological annotations (pathway membership, protein-protein interaction structure, functional gene modules). The remaining 41-55% co-activate with annotated features (suggesting they are not noise) but lack independent validation of biological meaning. The model does NOT encode directed transcription factor → target regulatory wiring.**

---

## 1. The Feature Atlas: 82,525 Biological Concepts

### 1.1 Scale of Superposition

The model's 1,152-dimensional hidden state encodes at least **82,525 distinct biological features** across 18 layers — a compression ratio exceeding 70×. This was invisible to all prior analysis methods:

- **99.8%** of features (82,336/82,525) are invisible to SVD/PCA
- Only **189** features align with standard linear decomposition axes
- Novel (non-SVD) features carry **98.7%** of all biological annotations
- Novel features are annotated at **52.5%** vs only **14.3%** for SVD-aligned features
- SAE explains **2.4×** more variance than top-50 SVD (77-85% vs 31-38%)

**Implication:** Every previous study that probed Geneformer's representations with PCA/UMAP/linear probes was seeing less than 1% of the encoded structure. The model's knowledge is overwhelmingly stored in superposed directions.

### 1.2 Feature Quality

- Mean inter-feature cosine similarity: **0.033-0.040** (features are well-separated)
- Dead features: 0.2% (L0) to 6.8% (L16) — the vast majority of dictionary entries are utilized
- Variance explained: 83.9-85.3% (early layers) to 76.8% (L17) — high-fidelity reconstruction

### 1.3 Annotated Biological Content

Total ontology enrichments across all 18 layers:

| Ontology | Total Enrichments | What It Captures |
|----------|-------------------|------------------|
| GO Biological Process | ~141,000 | Cellular processes, signaling, metabolism |
| Reactome | ~153,000 | Biochemical pathways, cascades |
| KEGG | ~33,000 | Metabolic and signaling pathways |
| STRING PPI | ~3,800 | Protein-protein interaction modules |
| TRRUST TF-target | ~2,100 | Transcription factor target sets |
| TRRUST TF edges | ~560 | Specific TF→target relationships |

**43,241 features** (52.4%) have at least one significant biological annotation. These are not vague associations — they are specific, FDR-corrected enrichments against curated databases.

---

## 2. What Specific Biological Knowledge Is Encoded?

### 2.1 Exemplary Features (Layer 0 — Molecular Programs)

| Feature | Top Genes | Biological Identity |
|---------|-----------|---------------------|
| F3717 | CDK1, CDC20, DLGAP5, PBK | **Cell cycle G2/M transition** |
| F3607 | RRM1, E2F1, MCM4, MCM6, RAD51 | **DNA replication & repair** |
| F1116 | ARL6IP4, SQSTM1, JAK1 | **B cell activation / differentiation** |
| F4536 | DYNC1H1, TLN1, MYH9, SPTAN1 | **Cytoskeleton / focal adhesion** |
| F2829 | TGFB1, PKN1, GADD45A, ZBTB7A | **MAPK/TGFβ signaling** |
| F1573 | MKI67, CENPF, TOP2A, CDK1, AURKB | **Mitosis / chromosome segregation** |

These are not "correlated gene lists" — they are recognizable, textbook biological programs. Feature F3717 essentially IS the G2/M checkpoint program; F3607 IS the DNA replication fork machinery.

### 2.2 Layer-Specific Knowledge Types

The model encodes different kinds of biology at different depths:

**Early layers (0-4): Molecular machinery**
- Gene-centric features mapping to specific pathways
- Highest annotation rates (53-59%)
- Examples: DNA replication, cell cycle checkpoints, ribosomal subunits, specific signaling cascades
- 10,000+ GO BP enrichments per layer
- Most STRING PPI associations (248-302 per layer)

**Middle layers (5-9): Abstract computation**
- Lowest annotation rates (45-52%)
- Features harder to map to single ontology terms
- May represent intermediate computational states
- Fewer but still substantial pathway associations
- Module counts peak here (12 at L5) — most functionally compartmentalized

**Mid-late layers (10-12): Re-specialization**
- Annotation recovery to 53-56%
- More integrative programs: cell differentiation, intracellular signaling, organelle organization
- Strong causal specificity (tested at L11)
- Peak perturbation detection

**Terminal layers (15-17): Prediction-focused**
- Annotation drops again (47-55%)
- Features respond broadly to perturbation (10-20 per target) but non-specifically
- Most dead features (215-314)
- Most distributed representations

### 2.3 The U-Shaped Annotation Profile

```
Annotation Rate by Layer:
L0:  58.6% ████████████████████████████▊
L1:  57.4% ████████████████████████████▍
L2:  55.5% ███████████████████████████▊
L3:  56.4% ████████████████████████████▏
L4:  53.9% ██████████████████████████▉
L5:  52.1% ██████████████████████████
L6:  49.1% ████████████████████████▌
L7:  47.7% ███████████████████████▊
L8:  45.4% ██████████████████████▋   ← minimum
L9:  50.3% █████████████████████████▏
L10: 55.8% ███████████████████████████▉
L11: 56.2% ████████████████████████████
L12: 53.2% ██████████████████████████▌
L13: 50.7% █████████████████████████▎
L14: 50.6% █████████████████████████▎
L15: 49.0% ████████████████████████▌
L16: 54.7% ███████████████████████████▎
L17: 47.0% ███████████████████████▌
```

This U-shape tells a story: the model starts with recognizable biology, transforms it through abstract intermediate representations, partially re-specializes, then optimizes for prediction. This is biological abstraction in action.

---

## 3. Modular Organization: 141 Biological Modules

### 3.1 Module Statistics

Across all 18 layers, Leiden clustering on PMI co-activation graphs identified **141 distinct biological modules**:

- **6-12 modules per layer** (not a single blob — genuinely compartmentalized)
- **96-99.5% coverage** (almost all features belong to modules)
- **327K-446K PMI edges** per layer
- Peak compartmentalization at **L5 (12 modules)**
- Edge density declines with depth (446K at L0 → 328K at L16)

### 3.2 Module Biological Identity

Layer 0 modules (6 modules):
1. **Cell Cycle / DNA Replication** — CDK1, MCM family, E2F transcription factors
2. **Immune Signaling** — JAK-STAT, cytokine receptors, B/T cell markers
3. **Metabolism** — glycolytic enzymes, TCA cycle, oxidative phosphorylation
4. **Translation** — ribosomal proteins, translation initiation/elongation factors
5. **Protein Quality Control** — ubiquitin-proteasome, ER stress, chaperones
6. **Cytoskeleton / Adhesion** — actin regulators, focal adhesion, integrins

Layer 11 modules (8 modules) — shift to integrative programs:
1. **Cell Differentiation** — including negative regulation
2. **Intracellular Signaling** — second messengers, kinase cascades
3. **Mitochondrial Organization** — complex assembly, mitochondrial dynamics
4. **Vesicular Transport** — Golgi, COPII, endocytosis
5. **Chromatin/Transcription** — histone modification, RNA processing
6. **Protein Modification** — ubiquitination, SUMOylation
7. **Cell Motility** — actomyosin, locomotion
8. **Stress Response** — DNA damage, unfolded protein response

**Key finding:** Module themes shift from **molecular machinery** (early) to **integrative cellular programs** (mid-late). This is hierarchical biological abstraction.

### 3.3 Module Connectivity

Modules are not isolated. Within each layer, features within modules have strong PMI connections, but there are also inter-module edges that reflect cross-pathway crosstalk (e.g., cell cycle ↔ DNA repair, metabolism ↔ translation).

---

## 4. Causal Structure: Features Have Genuine Computational Roles

### 4.1 Causal Patching Results (50 Features, Layer 11)

| Metric | Value |
|--------|-------|
| Median specificity ratio | **2.36×** |
| Features with >2× specificity | **60%** (30/50) |
| Features with >5× specificity | **24%** (12/50) |
| Features with >10× specificity | **12%** (6/50) |
| Maximum specificity | **114.5×** (F2035, Cell Differentiation) |
| Mean target logit disruption | −0.116 |
| Mean off-target disruption | −0.005 |

### 4.2 Top Causally Specific Features

| Feature | Biology | Specificity | What Happens When Ablated |
|---------|---------|-------------|---------------------------|
| F2035 | Cell Differentiation (neg. reg.) | 114.5× | Differentiation gene predictions collapse; others unaffected |
| F3692 | ERAD Pathway | 108.1× | ER quality control predictions disrupted |
| F3933 | Intracellular Signaling (neg. reg.) | 55.7× | Signaling cascade predictions fail |
| F157 | Golgi Vesicle Transport | 25.4× | Secretory pathway predictions disrupted |
| F3532 | Protein Metabolic Process (pos. reg.) | 11.2× | Protein homeostasis predictions disrupted |
| F4516 | Mitotic Spindle Microtubules | 10.6× | Mitosis gene predictions affected |
| F1337 | Mitotic Cell Cycle Phase Transition | 9.4× | Cell cycle transition predictions disrupted |
| F1023 | Mitotic Spindle Microtubules | 7.6× | Largest absolute effect (−2.799 on targets) |
| F2936 | Mitochondrion Organization | 7.1× | Mitochondrial gene predictions disrupted |
| F3962 | Endocytosis | 6.9× | Vesicular trafficking predictions disrupted |

**Key insight:** These features are not just correlated with biology — they are **causally necessary** for the model's predictions about that biology. Removing the "Cell Differentiation" feature specifically disrupts cell differentiation predictions while leaving everything else intact.

This is qualitatively different from the NMI paper's finding that ablating entire attention heads produces null effects. The model's computations are organized at the **feature level**, not the head level.

---

## 5. Cross-Layer Information Flow

### 5.1 The Paradox: Complete Turnover, Full Connectivity

- **Representational persistence**: Only 2-3% of features match across adjacent layers. 0% of L0 features survive past L11.
- **Functional connectivity**: 97-99.8% of features are "information highways" with strong PMI dependencies on downstream features.

The model completely rebuilds its representation at every layer, yet information flows through unbroken.

### 5.2 Biologically Meaningful Cascades

| Source (L0) | Target (L5) | PMI | Biological Meaning |
|-------------|-------------|-----|-------------------|
| mTORC1 Regulation | Autophagy | 9.55 | Known mTORC1→autophagy axis |
| Wnt Signaling | [unlabeled] | 9.48 | Wnt pathway processing |
| Protein Processing in ER | [unlabeled] | 11.10 | ER stress cascade |

| Source (L5) | Target (L11) | PMI | Biological Meaning |
|-------------|------------|-----|-------------------|
| Protein Polyubiquitination | [unlabeled] | 10.87 | Protein quality control |
| Translation | [unlabeled] | 10.35 | Translational regulation |
| RNA Splicing (neg. reg.) | [unlabeled] | 10.21 | Post-transcriptional processing |

| Source (L11) | Target (L17) | PMI | Biological Meaning |
|------------|------------|-----|-------------------|
| Protein Modification | Angiogenesis (pos. reg.) | 10.62 | Post-translational→phenotype |
| COPII Vesicle Budding | Thermogenesis | 10.29 | Secretory→metabolic |
| Actomyosin Organization | Cellular Locomotion (neg. reg.) | 10.14 | Cytoskeleton→motility |

The model builds biological pathways through chains of distinct feature sets — a form of **hierarchical biological abstraction** where molecular processes at early layers are progressively integrated into cellular programs at later layers.

### 5.3 Information Highway Statistics

- L0→L5: 4,530/4,604 highways (98.4%), mean max PMI = 6.61, max = 11.10
- L5→L11: 4,401/4,518 highways (97.4%), mean max PMI = 6.63, max = 10.87
- L11→L17: 4,544/4,555 highways (99.8%), mean max PMI = 6.79, max = 10.66

Almost every feature at every layer has a strong functional dependency on at least one feature at the next processing stage. The model's computation is densely interconnected despite surface-level representational independence.

---

## 6. Unannotated Features: Co-activation Evidence and Its Limits

### 6.1 The Unannotated Feature Problem

41-55% of features lack ontology annotations. Are they noise, biologically meaningful, or something in between?

**What we observe:** 95-98.5% of unannotated features co-activate with annotated features in established biological modules ("guilt by association"). Only 0.6-2.3% are completely isolated.

| Layer | Unannotated | Co-activate w/ Annotated | Truly Isolated |
|-------|-------------|--------------------------|----------------|
| 0 | 1,906 | 1,876 (98.4%) | 30 (1.6%) |
| 5 | 2,193 | 2,090 (95.3%) | 103 (4.7%) |
| 11 | 2,015 | 1,984 (98.5%) | 28 (1.4%) |
| 17 | 2,426 | 2,334 (96.2%) | 75 (3.1%) |

### 6.2 Caveats: Co-activation ≠ Biological Meaning

**This co-activation evidence is suggestive but not conclusive.** Several issues limit interpretation:

1. **Module coverage is near-total.** With only 6-12 modules per layer covering 96-99.5% of all features, the modules are large (hundreds of features each). At this granularity, co-membership is nearly unavoidable — features would need to be extremely unusual to NOT end up in a module.

2. **Co-activation has multiple explanations.** Two features firing on the same cells could reflect shared biological function, but also shared statistical properties (similar activation thresholds, correlated noise, positional biases in the tokenization).

3. **We tested the easy direction.** We showed unannotated features are *near* annotated features in activation space. We did NOT show they encode specific, independently testable biological hypotheses.

4. **No perturbation validation.** We did not test whether unannotated features respond specifically when genes in their top gene sets are perturbed. Given that even annotated TF features show only 6.2% perturbation specificity, the bar for validating unannotated features is high.

**What we can say:** The high co-activation rate is evidence that most unannotated features are **not random noise** — they capture some systematic pattern in the activation space. But the gap between "not noise" and "biologically meaningful" is substantial. Bridging it would require:
- Perturbation experiments targeting genes in unannotated feature clusters
- Cross-dataset replication (do the same features emerge in independently trained SAEs?)
- Experimental validation of specific predictions from novel clusters

### 6.3 Novel Clusters — Potentially Interesting, Unvalidated

Layer 11 novel feature clusters that form coherent gene programs (by Jaccard similarity):

- **Ribosomal + Iron Metabolism** (11 features): RPL11, RPL5, RPL23, RPS19... — ribosomal biogenesis intersecting with iron homeostasis. Not a single GO/KEGG term, but the genes are plausibly functionally related (ribosomal stress activates iron regulatory pathways).
- **Ribosomal + Mitochondrial** (6 features): RPS3, RPL3, NDUFS3, UQCRC1... — translation-respiration coupling. Known biology, poorly captured by discrete pathway terms.
- **Metabolite Transport** (4 features): SLC25A5, ATP5F1B, VDAC1... — mitochondrial metabolite exchange. Specific solute carrier combinations not represented in pathway databases.

These clusters are **hypothesis-generating**: they suggest potential functional groupings that could be tested experimentally, but they have not been validated beyond gene-set coherence. The fact that a cluster's top genes are recognizable to a biologist does not prove the feature itself is biologically meaningful — the model may group these genes for statistical reasons unrelated to their shared biology.

### 6.4 The Truly Isolated Features

Only 28-103 features per layer (0.6-2.3%) are genuinely isolated — no co-activation partners, no cluster membership. These are the strongest candidates for noise or technical artifacts. They may also represent:
- Extremely rare cell states present in few cells
- Technical artifacts (batch effects, sequencing noise)
- Biological programs specific to K562 not captured in general databases

---

## 7. What Does It NOT Encode? The Regulatory Boundary

### 7.1 The Perturbation Specificity Result

- 92% of CRISPRi knockdowns cause SAE feature changes (detection: YES)
- Only 6.2% of TFs produce target-specific feature responses (regulatory specificity: NO)
- Mean 2.54 features respond per perturbation, but mean only 0.03 are specific

### 7.2 The Multi-Tissue Control

- K562-only SAE: 6.2% (3/48 TFs)
- Multi-tissue SAE (best layer): 10.4% (5/48 TFs)
- Non-systematic: 5 gained, 3 lost, 40 unchanged
- TF feature representation actually DECREASED (64.5% → 60.5%)

### 7.3 Which TFs Do Show Specificity?

The few TFs with specific responses (across both experiments):
- **GATA1** — master hematopoietic TF (specific in multi-tissue L5, L11)
- **DNMT1** — DNA methyltransferase (specific in multi-tissue L5)
- **SRF** — serum response factor (specific in K562-only, lost in multi-tissue)
- **MAX** — MYC partner (specific in K562-only, lost in multi-tissue)
- **ATF5, BRCA1, RBMX, NFRKB** — gained in multi-tissue

These are among the most heavily studied TFs with the largest target sets — suggesting that specificity requires overwhelmingly strong regulatory signal to overcome the co-expression background.

---

## 8. Summary: The Complete Knowledge Inventory

### What Geneformer V2-316M DOES encode (confirmed by SAE analysis):

1. **Gene co-expression programs** — thousands of distinct co-expression modules
2. **Pathway membership** — GO BP, KEGG, Reactome pathway structure (45-59% of features have FDR-corrected enrichments)
3. **Protein-protein interaction topology** — STRING network neighborhoods
4. **Cell cycle programs** — G1/S, G2/M, mitosis with distinct feature sets
5. **Metabolic programs** — glycolysis, TCA, oxidative phosphorylation, amino acid metabolism
6. **Signaling cascades** — MAPK, TGFβ, Wnt, mTOR, JAK-STAT
7. **Organelle-specific programs** — ER quality control, Golgi transport, mitochondrial organization
8. **Immune programs** — B/T cell activation, cytokine signaling
9. **Structural programs** — cytoskeleton, focal adhesion, actomyosin
10. **Hierarchical biological abstraction** — molecular → integrative across layers
11. **Cross-pathway crosstalk** — functional connections between biological modules
12. **Causal computational structure** — individual features are causally necessary for specific biological predictions (median 2.36× specificity, up to 114.5×)

### What it does NOT encode (or encodes minimally):

1. **Directed TF→target regulatory edges** — only 6-10% specificity
2. **Causal regulatory logic** — cannot distinguish regulator from co-regulated
3. **Perturbation-specific consequences** — detects change but not regulatory mechanism

### What remains uncertain:

1. **Biological meaning of unannotated features** — 41-55% of features lack direct ontology annotations. 95-98.5% co-activate with annotated features, which suggests they are not noise, but co-activation alone does not prove biological function. Experimental validation is needed.
2. Whether features encode **chromatin state** (would need ATAC-seq integration)
3. Whether features encode **temporal dynamics** (would need time-course data)
4. Whether features encode **spatial information** (would need spatial transcriptomics)
5. Whether higher-expansion SAEs (8×, 16×) would resolve finer biological structure
5. Whether other scFMs (scGPT, scBERT, UCE) have similar or different feature atlases

---

## 9. Potential Utility Despite Regulatory Limitation

Even without regulatory logic, the encoded knowledge may be useful for:

1. **Cell type classification** — features represent cell-state programs directly
2. **Pathway activity scoring** — individual features serve as pathway activity readouts
3. **Gene function hypothesis generation** — novel feature clusters suggest potentially unstudied gene groupings (requires experimental validation)
4. **Drug target identification** — causally specific features point to genes with non-redundant roles
5. **Biomarker discovery** — features with strong causal specificity identify non-obvious gene sets
6. **Cross-tissue comparison** — multi-tissue SAE features could map shared vs tissue-specific programs
7. **Model debugging** — understanding what the model computes helps identify failure modes

---

## 10. Visualization and Presentation Opportunities

### 10.1 Interactive Feature Atlas Browser
- Searchable database of all 82,525 features
- Per feature: top genes, ontology annotations, causal specificity, module membership
- Filter by layer, annotation type, specificity threshold

### 10.2 Network Visualizations
- **Intra-layer co-activation networks** (141 modules, colored by biological theme)
- **Cross-layer information highways** (bipartite graphs showing L0→L5→L11→L17 flow)
- **Module theme evolution** across layers (Sankey/alluvial diagram)

### 10.3 Layer Profile Dashboards
- U-shaped annotation heatmap
- Per-ontology enrichment across layers
- Dead feature / alive feature distributions
- Variance explained decay

### 10.4 Perturbation Response Maps
- TF × feature heatmaps showing response patterns
- K562 vs multi-tissue comparison panels
- Specificity distributions across TFs

### 10.5 Causal Specificity Showcase
- Individual feature ablation case studies
- Target vs off-target logit disruption scatter plots
- Top-10 most specific features with gene-level detail

---

## Data Inventory

All raw results are available in:

| Directory | Contents |
|-----------|----------|
| `experiments/phase1_k562/layer_*_activations.npy` | Raw activations (336 GB) |
| `experiments/phase1_k562/sae_models/layer*_x4_k32/` | 18 trained SAE models + catalogs |
| `experiments/phase1_k562/coactivation/` | 18 co-activation graphs with modules |
| `experiments/phase1_k562/causal_patching/` | 50-feature causal patching results |
| `experiments/phase1_k562/perturbation_response/` | 100-target perturbation mapping |
| `experiments/phase1_k562/novel_features/` | Novel feature clustering (4 layers) |
| `experiments/phase1_k562/computational_graph/` | Cross-layer PMI dependencies |
| `experiments/phase1_k562/svd_vs_sae_comparison.json` | SVD vs SAE comparison |
| `experiments/phase1_k562/cross_layer_tracking.json` | Cross-layer feature persistence |
| `experiments/phase3_multitissue/` | Multi-tissue SAE models + perturbation results |
