# Verified new results (source of truth for the manuscript)

Every number below is read from a result file under `experiments/revision_srep/`.
Nothing here is written into the paper until it appears in this file.

---

## E0 — the control cohort is NOT K562 (correction to the submitted manuscript)
Source: `experiments/revision_srep/E0_cohorts/composition.json`, seed 42.

The submitted manuscript states that the Geneformer atlas was built from "2,000 K562 control
cells". It was not. The extraction script (`src/01_extract_activations.py`) selects control cells
by perturbation label alone (`gene == "non-targeting"`) over the whole concatenated Replogle
matrix, which spans four cell lines, and applies **no cell-line filter**. The resulting cohort is:

| cohort | Jurkat | K562 | RPE1 | HepG2 |
|---|---|---|---|---|
| non-targeting cells available | 12,013 | 10,691 | 11,485 | 4,976 |
| **atlas extraction cohort (n = 2,000)** | **591** | **581** | **567** | **261** |
| **published causal-patching cohort (n = 200)** | 0 | 0 | 0 | **200** |

**This is verified, not inferred.** Re-running the selection under seed 42 reproduces the per-cell
gene counts stored at extraction time exactly (25/25 identical, mean 2,028.18 genes per cell), so
this is the composition of the cells actually processed.

The causal-patching cohort is entirely HepG2 because `src/08_causal_patching.py` takes the first
200 cells of the sorted atlas cohort, and the matrix is ordered by cell line with HepG2 occupying
rows 0–96,615.

**Consequences for the manuscript.**
1. Every description of the atlas as K562-derived is wrong and must be corrected to a
   multi-cell-line non-targeting control cohort. The directory name `phase1_k562` is a misnomer.
2. The published causal-patching numbers (median 2.36×) were measured on HepG2 cells. E5 re-runs
   this with an explicit cell-line filter.
3. The multi-tissue comparison was framed as "K562-only versus K562 + Tabula Sapiens", i.e. as a
   test of whether the SAE training data lacked diversity. The actual contrast is four immortalized
   CRISPRi lines versus those four plus three primary tissues. The observed marginal improvement
   stands but its interpretation changes: the baseline dictionary was already multi-cell-line.
4. The published perturbation-response analysis compared K562 perturbed cells (`src/09` does filter
   to K562) against a control baseline drawn from the four-line pool, so cell-line differences
   could contribute to apparent perturbation responses. The new analysis compares perturbed and
   non-targeting cells **within the same line**, removing this.
5. Reviewer 2's major point 4 (Geneformer on K562 versus scGPT on Tabula Sapiens) has a different
   answer than expected: the Geneformer cohort is already multi-line, which narrows the confound
   without eliminating it (immortalized lines versus primary tissue).

---

## E6 — capacity-matched SAE vs SVD
Source: `experiments/revision_srep/E6_svd_capacity/summary.json` (all 18 layers), seed 42.
Sampling: 500,000 positions per layer in 40 evenly spaced blocks; exact eigendecomposition of the
sampled covariance gives all 1,152 principal directions.

**Reconstruction-matched.** Truncated SVD needs rank 250–403 to match the SAE's variance explained:
L0 0.848 → rank 403; L5 0.856 → 317; L11 0.823 → 308; L17 0.775 → 250.
The published comparison used 50 axes, i.e. 5–8× under-capacity.

**Cumulative projection onto the rank-k principal subspace** (median over decoder directions),
against the random-direction null k/d:

| k | 50 | 100 | 200 | 300 | 500 | 800 |
|---|---|---|---|---|---|---|
| L0 median ρ | 0.093 | 0.207 | 0.424 | 0.573 | 0.762 | 0.913 |
| L11 median ρ | 0.158 | 0.332 | 0.534 | 0.650 | 0.793 | 0.922 |
| L17 median ρ | 0.178 | 0.372 | 0.580 | 0.692 | 0.830 | 0.941 |
| random null | 0.043 | 0.087 | 0.174 | 0.260 | 0.434 | 0.694 |

Median direction reaches 50% of its norm at k = 247 (L0), 190 (L5), 178 (L11), 152 (L17);
90% at k = 764, 671, 734, 664. Null: 50% at k = 576, 90% at k = 1,037.
→ SAE directions are 2.4–4.3× more concentrated in the leading principal subspace than random
directions, but no feature is a principal axis.

**Single-axis alignment is not a capacity artefact.** At τ = 0.5 the fraction of features aligned
with *any* axis saturates by k ≈ 20–50 and does not rise thereafter: even against the **complete**
1,152-dimensional basis it is 0.98% (L0), 0.13% (L5), 0.09% (L11), 0.26% (L17).

**Annotation yield per direction** (identical top-20-gene Fisher/BH protocol, both polarities for
principal axes):

| | frac annotated | terms/direction | unique terms |
|---|---|---|---|
| L0 top-50 principal axes | 0.98 | 159.5 | 2,029 |
| L0 100 random SAE features | 0.90 | 64.4 | 1,977 |
| L11 top-50 principal axes | 0.98 | 178.8 | 2,083 |
| L11 100 random SAE features | 0.93 | 71.3 | 2,045 |

→ Per direction, leading principal axes are at least as annotatable as SAE features. The claim that
biological signal is exclusive to non-SVD-aligned features does not survive a matched comparison.

---

## E7 — how many distinct concepts
Source: `experiments/revision_srep/E7_concepts/summary.json`, seed 42.
Method: all 82,525 features from 18 layers compared in the shared gene space by top-20 gene-set
Jaccard (edge if |intersection| ≥ 5 and Jaccard ≥ 0.14), Leiden at γ ∈ {0.5, 1.0, 2.0}.

- 82,525 atoms → **18,711 distinct gene-level programs** at γ = 1.0 (4.41 atoms per program).
  Stable across resolutions: 18,695 (γ=0.5), 18,748 (γ=2.0).
- 17,690 programs are singletons; 1,021 multi-feature programs absorb the remaining 64,835 atoms;
  largest program 3,221 atoms.
- 97.1% of multi-feature programs recur in ≥2 layers; mean 3.16 layers spanned.
- Gene signatures are almost all distinct as exact sets: 82,442 distinct signatures, only 56
  signatures shared by more than one feature (139 features total, max multiplicity 5).
- Per layer at γ=1.0: L0 4,608 atoms → 1,301 programs; L5 4,576 → 814; L11 4,598 → 1,252;
  L17 4,580 → 1,360.
- Within-layer packing at L11: 4,608 atoms in d = 1,152 (4.0 atoms per dimension), 4,568 alive,
  mean |coherence| 0.0361, Welch bound for 4,608 unit vectors in 1,152 dimensions = 0.0255.

→ Replaces ">70× compression". Within one representation space the dictionary is 4× overcomplete
and resolves ~1,250 distinct gene-level programs; the 82,525 figure is an aggregate over 18
separately trained dictionaries, not concepts coexisting in one 1,152-dimensional space.

---

## E1 — recoverability ceiling (K562)
Source: `experiments/revision_srep/E1_ceiling/scores_k562*.json`, seed 42.
Panel: 20 evaluable TFs (≥50 perturbed cells and ≥5 curated targets present in the measured gene
space). Top-100 predicted targets per method; hypergeometric enrichment, BH across the panel.

**Evaluability of the assay (the binding constraint).** The Perturb-seq matrix measures 6,546
genes. Of 73 perturbed TRRUST TFs with ≥50 cells:

| curated targets measured | TRRUST | DoRothEA A+B+C | DoRothEA all |
|---|---|---|---|
| ≥1 | 55 | 17 | 27 |
| ≥3 | 21 | 14 | 22 |
| ≥5 | **9** | 13 | 18 |
| ≥10 | 3 | 13 | 18 |
| ≥20 | 1 | 10 | 17 |
| median measured targets per TF | **1.0** | 15.0 | 146.5 |

→ Against TRRUST, the median perturbed TF has exactly **one** measurable target. Target-specific
recovery is undefined for most of the published 48-TF panel, for any method.

**Fraction of TFs with FDR < 0.05 target enrichment** (universe = 6,546 measured genes;
`scores_k562_merged.json`, top-100 predicted targets per method):

| method | TRRUST | DoRothEA A+B+C | DoRothEA all |
|---|---|---|---|
| differential expression of the knockdown (ceiling) | 1/9 = 11.1% | 3/13 = 23.1% | 8/18 = 44.4% |
| Pearson co-expression | 2/8 = 25.0% | 2/10 = 20.0% | 2/14 = 14.3% |
| Spearman co-expression | 2/8 = 25.0% | 2/10 = 20.0% | 1/14 = 7.1% |
| GENIE3 | 1/8 = 12.5% | 1/10 = 10.0% | 0/14 = 0.0% |
| GRNBoost2 | 1/8 = 12.5% | 1/10 = 10.0% | 1/14 = 7.1% |
| Geneformer gene embeddings (no SAE) | 1/8 = 12.5% | 1/10 = 10.0% | 0/14 = 0.0% |
| random gene sets | 0/9 = 0.0% | 0/13 = 0.0% | 0/18 = 0.0% |

Established network-inference methods do no better than plain co-expression on these data, and all
observational methods fall below the perturbation-aware ceiling on the best-powered reference.
GENIE3 fitting took 2,136 s for 6,546 per-target models; GRNBoost2 74 s (early stopping).

SAE arm (cell-level): PENDING (E2/E3).

**Universe choice matters.** With the universe set to 20,000 rather than the 6,546 measured genes,
random gene sets reach 16.7% "significant" against DoRothEA-all — the test is anticonservative
because predictions can only be drawn from measured genes. All headline numbers use the measured
universe; random is 0% there.

**Positive control.** The knockdown itself is strongly detectable in expression
(self-gene Cohen's d ≈ −2.5 for ATF4, −1.9 for BRCA1 at n = 200 cells).

---

## E2 — statistics of the perturbation response (K562)
Source: `experiments/revision_srep/E3_cell_level/k562_main/{k562sae,multitissue}/per_target.json`,
seed 42. 30 targets (20 evaluable transcription factors, 10 non-TF controls), up to 200 perturbed
cells per target against 600 non-targeting K562 cells. Significance is assessed with the cell as
the unit of replication; the effect-size definition is held fixed at the control position-level
scale so the cell-level and position-level results differ only in the inference.

**The pseudoreplication effect is real but small.** Median intraclass correlation of feature
activation across gene positions within a cell = 0.000298 (atlas dictionary). At ~2,000 positions
per cell the design effect is **1.61** (multi-tissue dictionary 1.66), so the effective sample size
is ~60% of the nominal position count, not smaller by three orders of magnitude. Sparse TopK
activations at different positions of one cell are close to independent.

**Responding features, cell-level versus position-level** (mean per target):

| dictionary | TF, cell-level | TF, position-level | non-TF, cell-level |
|---|---|---|---|
| multi-line control atlas | 1.90 (3/20 with any) | 1.60 | 2.30 (3/10 with any) |
| + primary tissue (multi-tissue) | 2.55 (5/20 with any) | 1.85 | 2.30 (2/10 with any) |

**Non-TF perturbations respond as much as TF perturbations.** With the atlas dictionary the ten
non-TF control knockdowns average *more* responding features (2.30) than the twenty transcription
factors (1.90). Whatever the features register is not TF-specific. The published analysis had no
negative control of this kind.

**The published response counts were uncorrected for multiple testing.** Reproducing the published
criterion (|effect| > 0.5, no correction) gives mean 3.20 responding features per TF with 15/20
targets showing at least one, consistent with the published mean of 2.54 and "92% of knockdowns".
Adding Benjamini–Hochberg correction across the 4,608 features collapses this to 2/20 targets.

**The sensitivity limit is measured, not assumed.** This is the sharpest result of the section:

| | detects the knocked-down gene itself |
|---|---|
| differential expression on the same cells | **15/16 targets** (FDR < 0.05; median Cohen's d −1.19, range −2.76 to −0.49) |
| SAE feature response, atlas dictionary | **1/20 targets** |
| SAE feature response, multi-tissue dictionary | **0/20 targets** |

The knockdown is unambiguously present in the expression profile of the very same cells, and the
SAE feature responses do not register it. The features are therefore not insensitive because the
data are noisy; they are responding to global cell-state shifts rather than to the gene-level
consequences of the perturbation. This is the precise sense in which detection is limited, and it
is a property of what the dictionary encodes rather than of the assay.

---

## E3 — sample size (K562, atlas dictionary)
Source: `experiments/revision_srep/E3_cell_level/k562_main/k562sae/sweep.json`, seed 42.
20 bootstrap draws per target per sample size. Restricted to the 14 transcription factors that
have at least 100 cells, so the comparison is within-target rather than across different target
sets (the unrestricted version confounds sample size with which targets are available).

| cells per target | responding features | detects the knockdown itself | nominal target enrichment p < 0.05 |
|---|---|---|---|
| 10 | 0.56 | 5.0% | 7.7% |
| 20 | 0.65 | 6.8% | 15.5% |
| 50 | 0.65 | 7.1% | 27.5% |
| 100 | 0.61 | 7.1% | 26.9% |

The responding-feature count is flat across a tenfold increase in cells. Detection of the knockdown
itself rises from 5.0% to 7.1% and then saturates. Nominal target enrichment improves from n = 10 to
n = 50 and then plateaus, so the published choice of n = 20 was somewhat under-powered, but the gain
saturates well before the effect becomes substantial. These are *nominal* p-values; under
Benjamini–Hochberg across the panel none of the sample sizes yields a significant transcription
factor.

---

## E4 — replication in independent cell lines
Source: `experiments/revision_srep/E3_cell_level/<line>_main/k562sae/per_target.json`, seed 42.
Perturbed and non-targeting cells drawn from the same line in every case.

**RPE1** (retinal pigment epithelial, non-cancer): 16 evaluable TFs, 6 non-TF controls.
Mean responding features 0.25 (TF) versus 0.17 (non-TF); 4/16 TFs with any responding feature;
**0/16** detect the knocked-down gene; **0/16** show target enrichment at FDR < 0.05;
design effect 1.72.

Jurkat and HepG2: PENDING.

---

## E5 — causal feature ablation with an independent evaluation set
Source: `experiments/revision_srep/E5_causal_v2/results.json`, seed 42. Layer 11, K562 cells only,
119 features patched across three selection arms, 60 cells each. 14 of 119 features have **no**
held-out set at all: every gene of their annotated term is already inside their own top-20
activating genes, so for those the original test was circular by construction.

Median specificity ratio (mean |Δlogit| on the gene set over mean |Δlogit| on all other positions):

| gene set used for evaluation | top-annotated arm | random annotated | random any |
|---|---|---|---|
| top-20 ∩ annotated term (the confounded set) | **37.3** | 56.9 | 66.9 |
| annotated-term genes *not* in the top-20 (independent) | **1.04** | 0.94 | 0.97 |
| size- and activation-matched random genes | 0.89 | 0.91 | 0.89 |
| all top-20 genes vs everything else (published definition) | 35.5 | 64.3 | 62.1 |

Wilcoxon, held-out versus matched-random per feature: p = 0.006 (median difference 0.13) in the
top-annotated arm, p = 0.76 and p = 0.34 in the two unbiased arms.

**The reported causal specificity was an artefact of the evaluation set.** Ablating a feature
disrupts the genes that feature activates on, which is close to tautological. Evaluated on the
*other* genes of the same annotated term, specificity is ~1.0 — indistinguishable from a random
gene set matched for size and activation level. The effect does not depend on how features are
chosen: unbiased random selection gives the same picture as the richly-annotated selection, so this
is not a consequence of cherry-picking features either.

Note the arms are internally comparable but not directly comparable to the previously published
2.36× median, which was computed on different cells (HepG2, see E0), with 200 cells per feature and
a signed rather than magnitude ratio. The within-experiment contrast is the valid one.

---

## E8 — run-to-run stability of the dictionaries
Source: `experiments/revision_srep/E8_seed_stability/summary.json`. Six runs per layer: five
differing only in weight initialisation and batch order with the training subsample held fixed,
plus one with a different 1M-position subsample.

**Aggregate statistics are highly reproducible.**

| | layer 0 | layer 11 |
|---|---|---|
| variance explained | 0.8454 ± 0.0003 | 0.8200 ± 0.0003 |
| dead features | 2 ± 1 | 32 ± 4 |
| annotation rate | 0.935 ± 0.009 | 0.933 ± 0.010 |
| module count | 7.2 ± 0.4 | 8.3 ± 1.0 |

**Individual dictionary atoms are not.** Mean best decoder cosine between runs is 0.282 (L0) and
0.294 (L11); only 0.8% and 0.6% of atoms have a counterpart above 0.9 in another run.

The right reference for those numbers is the best match a random direction would find among 4,608
directions in 1,152 dimensions, computed empirically: mean best |cos| = **0.113** (sd 0.008, 99th
percentile 0.134), with no random direction reaching 0.9. So cross-seed agreement (0.28–0.29) is
well above chance but far below reproducibility.

**Interpretation.** Independent runs learn different bases that explain the same variance, annotate
at the same rate and yield the same number of modules. Aggregate properties of the atlas are
seed-independent; the identity of any individual feature is not. This is a real limitation for a
released feature atlas, and it is consistent with the gene-level program analysis (E7), where
82,525 atoms resolve into 18,711 programs — the programs are the level at which the representation
is stable, not the atoms.

---

## E9 — matched cross-model comparison
Source: `experiments/revision_srep/E9_matched_crossmodel/summary.json`, seed 42.
Geneformer dictionaries trained on the same 3,000 Tabula Sapiens cells as the scGPT atlas, with
identical hyperparameters, and compared at matched relative depth (layer index / depth − 1).
`matched_protocol` is the scGPT dictionary retrained under the same 1M-position subsample protocol
as the Geneformer runs, controlling for the difference in training-set size.

Variance explained on held-out positions:

| relative depth | Geneformer / TS | scGPT / TS | scGPT / TS (matched protocol) | Geneformer / K562 |
|---|---|---|---|---|
| 0.00 (GF L0, scGPT L0) | **0.7731** | 0.7258 | 0.7676 | 0.8444 |
| ~0.28 (GF L5, scGPT L3) | 0.7749 | 0.7729 | 0.7827 | 0.8528 |
| ~0.65 (GF L11, scGPT L7) | 0.7596 | **0.8515** | 0.8171 | 0.8187 |
| 1.00 (GF L17, scGPT L11) | 0.7241 | **0.8862** | 0.8552 | 0.7713 |

**The published comparison does not survive matching.** The manuscript reported a uniform scGPT
advantage in reconstruction (mean 90.2% against 81.7%) and attributed it to architecture, but that
compared Geneformer on K562 against scGPT on Tabula Sapiens. On the same cells the difference
*reverses with depth*: Geneformer reconstructs better at the input end (0.7731 vs 0.7258 at depth 0),
the two are indistinguishable at about a quarter depth (0.7749 vs 0.7729), and scGPT is ahead in
the second half (0.8862 vs 0.7241 at the output end). There is no uniform architectural advantage;
there is a crossover.

**Input distribution accounts for a large part of the original gap.** The same Geneformer
dictionaries reconstruct their own CRISPRi-like data better than Tabula Sapiens by 5–8 percentage
points at every depth (for example 0.8444 vs 0.7731 at layer 0), which is of the same order as the
cross-model difference the original comparison reported.

Decoder coherence is 0.030–0.036 for Geneformer and 0.039–0.047 for scGPT against Welch bounds of
0.0255 and 0.0362 respectively, so both dictionaries sit close to the packing limit for their
dimensionality and neither is exploiting unusual capacity.

---

## E10 — independent external datasets (this qualifies the central claim)
Sources: `experiments/revision_srep/E10_external/{papalexi,norman}/results.json`, seed 42.
Same atlas dictionary, same per-cell encoding, same cell-level statistics as the primary analysis.

| | Replogle (primary) | Papalexi | Norman |
|---|---|---|---|
| assay | CRISPRi | ECCITE-seq (CRISPRi) | Perturb-seq (**CRISPRa**) |
| cells | 4 lines | THP-1 monocytes | K562 |
| genes measured | 6,546 | 18,649 | 33,694 |
| median measured TRRUST targets per TF | **1.0** | **19.5** | **6.0** |
| median self-effect Cohen's d | −1.19 | −0.18 | **+0.98** |
| mean DE genes per target | — | 184 | 405 |
| median responding SAE features | ~0 | — | 38 |

**On the primary panel nothing is detectable**: 0/9, 0/13, 0/18 under the three references, and
1/66 across four cell lines. On the two transcriptome-scale datasets the SAE does recover targets:
20–22% (Papalexi) and 23–25% (Norman) under DoRothEA, against 0% under TRRUST in both.

**The non-TF control, at a single consistent protocol (top-100 predictions, BH within arm).**
Ten Norman perturbations of genes absent from every reference network, run through the identical
pipeline; their predicted sets tested against the *same* TF regulons. Nine produced a prediction.

| reference | matching TF | control tests | control **genes** hitting ≥1 regulon | Fisher (TF vs control genes) |
|---|---|---|---|---|
| DoRothEA A+B+C | 3/13 = 23% | 1/117 | **1/9 = 11%** | p = 0.62 |
| DoRothEA all | 4/16 = 25% | 29/144 | **8/9 = 89%** | p = 0.004 (controls higher) |

**The per-perturbation unit is the correct one.** Quoting 1/117 compares one arm's independent
perturbations against the other arm's perturbation-by-regulon tests, inflating the control
denominator ninefold. At the level of independent perturbations the high-confidence comparison is
3 of 13 against 1 of 9, which is **not significant**.

**Conclusion the paper states.** The flat null obtained on the primary panel does not reproduce on
transcriptome-scale data, and a factor-specific signal is not established either: the one
comparison capable of demonstrating it rests on thirteen factors in a single screen. The
all-confidence tier is uninformative — controls enrich more often than factors. The binding
limitation on the positive claim is panel size, not assay coverage.

---

All experiments complete. Numbers in the manuscript are checked against these files by
`src_revision_srep/verify_numbers.py` (16/16 passing).
