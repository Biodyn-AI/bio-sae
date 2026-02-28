# Phase 3 Results: Multi-Tissue SAE — Does Geneformer Know Regulation?

## Summary

**Verdict: WEAK POSITIVE** — Training SAEs on diverse multi-tissue data provides marginal improvement in TF perturbation specificity (6.2% → 10.4%), but the effect is small. Geneformer's regulatory knowledge is minimal.

## Experiment Design

**Hypothesis:** A SAE trained on diverse cells (where TFs vary across cell types) will develop TF-sensitive features that respond specifically to CRISPRi TF knockdowns.

**Data:**
- 3,000 Tabula Sapiens cells (1,000 immune × 43 cell types, 1,000 kidney, 1,000 lung)
- 5,623,164 token positions extracted through Geneformer V2-316M
- Pooled with K562 activations: 500K K562 + 500K Tabula Sapiens = 1M training positions per layer
- SAEs trained at 4 layers (0, 5, 11, 17) with identical architecture: 4608 features, k=32

## Results

### SAE Training Quality

| Layer | Var. Explained | Alive Features | Annotation Rate |
|-------|---------------|----------------|-----------------|
| L0  | 79.1% | 4607/4608 | 54.3% |
| L5  | 79.5% | 4477/4608 | 43.4% |
| L11 | 77.3% | 4546/4608 | 51.2% |
| L17 | 73.2% | 4413/4608 | 42.3% |

All SAEs trained well — comparable quality to Phase 1 K562-only SAEs.

### Perturbation Specificity (Main Result)

| Condition | Layer | TFs Specific | Specificity Rate |
|-----------|-------|-------------|-----------------|
| K562-only SAE (Phase 2) | 11 | 3/48 | **6.2%** |
| Multi-tissue SAE | 0 | 0/48 | 0.0% |
| Multi-tissue SAE | 5 | 4/48 | 8.3% |
| **Multi-tissue SAE** | **11** | **5/48** | **10.4%** |
| Multi-tissue SAE | 17 | 1/48 | 2.1% |

**Best multi-tissue layer: L11 at 10.4%** — improvement of +4.2 percentage points over K562-only.

### Per-TF Head-to-Head (Layer 11)

| TF | K562-SAE Specific | MT-SAE Specific | Change |
|----|-------------------|-----------------|--------|
| ATF5 | 0 | 1 | + gained |
| BRCA1 | 0 | 1 | + gained |
| GATA1 | 0 | 1 | + gained |
| RBMX | 0 | 1 | + gained |
| NFRKB | 0 | 1 | + gained |
| MAX | 1 | 0 | - lost |
| PHB2 | 1 | 0 | - lost |
| SRF | 1 | 0 | - lost |
| *40 others* | 0 | 0 | unchanged |

5 TFs gained specificity, 3 lost it — largely a **different set** rather than a consistent improvement.

### TF Feature Diagnostics

| SAE | Features with TFs in top genes | TF-dominant (≥3 TFs) |
|-----|-------------------------------|----------------------|
| K562-only L11 | 2967/4598 (64.5%) | 424 |
| Multi-tissue L0 | 2796/4608 (60.7%) | 452 |
| Multi-tissue L5 | 2777/4568 (60.8%) | 337 |
| Multi-tissue L11 | 2782/4601 (60.5%) | 343 |
| Multi-tissue L17 | 2680/4603 (58.2%) | 346 |

Surprisingly, the K562-only SAE has **more** TF-associated features than the multi-tissue SAE. Multi-tissue training did not increase TF feature representation.

### Layer Pattern

Striking layer-dependent behavior:
- **L0 (0%):** Embedding layer — multi-tissue SAE features barely respond to perturbation at all
- **L5 (8.3%):** Early processing — some TF-specific responses emerge (GATA1, DNMT1, SRF, MAX)
- **L11 (10.4%):** Mid-network — best specificity, consistent with Phase 1 finding that L11 has peak biological annotation
- **L17 (2.1%):** Late layers — many features respond broadly (10-20 per target) but almost none specifically. Features at L17 are too abstract/non-specific for regulatory mapping.

## Interpretation

The +4.2pp improvement (6.2% → 10.4%) falls in the **WEAK POSITIVE** category:

1. **Multi-tissue SAE does marginally better**, but 10.4% (5/48 TFs) is still far from the >20% threshold for "strong positive"
2. **The improvement is not systematic** — different TFs gain and lose specificity, suggesting stochastic variation rather than unlocked regulatory knowledge
3. **TF feature representation actually decreased** — multi-tissue SAE has fewer TF-associated features (60.5%) than K562-only (64.5%)
4. **The fundamental bottleneck is Geneformer itself**, not the SAE training data. If Geneformer had encoded TF→target regulatory relationships, they would have been extractable regardless of SAE training diversity

## Conclusion

Geneformer V2-316M has minimal regulatory knowledge that can be extracted via SAE perturbation analysis. The multi-tissue training provided marginal improvement but did not unlock hidden regulatory features. This is consistent with the broader finding from the main paper: Geneformer captures **co-expression patterns** (gene proximity, pathway membership) rather than **causal regulatory relationships** (TF→target activation).

## Compute

| Step | Time | Disk |
|------|------|------|
| 12a: Extract Tabula Sapiens | 61.6 min | 103.6 GB |
| 12b: Pool and train SAEs | 23.6 min | ~4 GB |
| 12c: Analyze and annotate | 46.6 min | ~2 GB |
| 12d: Perturbation test | 88.9 min | ~0.1 GB |
| 12e: Compare results | 0.6 s | ~0.01 GB |
| **Total** | **~221 min (~3.7 hr)** | **~110 GB** |

## Files

| File | Description |
|------|-------------|
| `src/12a_extract_tabula_sapiens.py` | Extract TS activations |
| `src/12b_pool_and_train.py` | Pool + train SAEs |
| `src/12c_analyze_and_annotate.py` | Feature analysis + ontology annotation |
| `src/12d_perturbation_test.py` | Perturbation response test |
| `src/12e_compare_results.py` | Head-to-head comparison |
| `experiments/phase3_multitissue/ts_activations/` | TS activation memmaps (4 layers) |
| `experiments/phase3_multitissue/sae_models/` | Trained SAE models (4 layers) |
| `experiments/phase3_multitissue/perturbation_response/` | Per-layer perturbation results |
| `experiments/phase3_multitissue/comparison/comparison_results.json` | Final comparison |
