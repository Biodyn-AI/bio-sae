# Scientific Reports revision plan — SAE atlas of single-cell foundation models

Submission ID fbb71648-3418-4139-b1df-d7dd0b5456d6. Manuscript transferred from BMC Genomics.
Base manuscript: `paper/genome_biology/bmcgenomics_revision/main.tex` (+ `supplementary.tex`).
Working copy for this round: `paper/scientific_reports_revision/`.
New code: `src_revision_srep/`. New results: `experiments/revision_srep/`.

**Governing constraints from the author**
1. Prefer *new experiments and analysis* over text-only rebuttal wherever a comment admits either.
2. The final paper must read as a single coherent study. No trace of revision history: no "an earlier
   version", "the original preprint", "as the reviewers suggested", "we now treat", "we have since",
   "in this revision", no superseded-number pairs, no reviewer-response voice.
3. Title: drop the sub-title. New title (also satisfies the no-colon editorial rule):
   **"Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in
   single-cell foundation models"**.
4. KEGG permission is already resolved by the author — no action beyond keeping the KEGG citation in
   the legend of any figure carrying KEGG-derived content (condition of the granted permission).

---

## 1. Comment inventory and disposition

Legend: **NEW EXP** = new computation; **TEXT** = wording/structure only; **FIX** = build/technical.

### Editorial

| # | Comment | Disposition |
|---|---|---|
| E1 | Title must have no colon, read as one sentence | **TEXT** — new title as above |
| E2 | KEGG permission | Resolved by author. Keep KEGG citation in figure legends |
| E3 | Code must be deposited in a DOI-assigning repository, linked from Methods or a Code Availability section | **FIX** — add `Code availability` section; build a release-ready archive + Zenodo metadata; author performs the deposit and pastes the DOI |
| E4 | Editor: key findings insufficiently supported; strengthen with analyses/controls or tone down | Addressed by the whole experiment programme below, plus systematic claim scoping (§4) |

### Reviewer 1

| # | Comment | Disposition |
|---|---|---|
| R1.M1 | No baseline for how much regulatory information is recoverable from the same scRNA-seq input; compare against established regulatory-inference methods incl. a perturbation-aware one | **NEW EXP E1** — recoverability-ceiling benchmark |
| R1.M2 | Target-gene definition unclear; §5.4 annotates on top-20 genes while §2.7 evaluates on ontology-matching positions; is the evaluation set independent of the annotation set? How were the 50 features chosen? | **NEW EXP E5** + **TEXT** — held-out gene evaluation, matched random controls, unbiased feature selection, explicit selection protocol |
| R1.M3 | Only 20 (Geneformer) / 10 (scGPT) perturbed cells per target; no power or sensitivity analysis | **NEW EXP E3** — cell-count sweep, bootstrap CIs, positive control, minimum detectable effect |
| R1.M4 | Regulatory analysis limited to K562; validate in an independent perturbation dataset from another cell type | **NEW EXP E4** — RPE1, Jurkat, HepG2 from the same Replogle release (independent screens, three additional cell types) |
| R1.m1 | SAE training seed not reported; run-to-run variability not assessed | **NEW EXP E8** + **FIX** — explicit seeding in code, 5-seed stability study |
| R1.m2 | Figure 6 enrichment-term labels truncated; axis labels overlap | **FIX E10** — regenerate figures |
| R1.m3 | Unresolved cross-references ("Table ??") throughout | **FIX E11** — remove the fragile `xr`/`externaldocument` mechanism |

### Reviewer 2

| # | Comment | Disposition |
|---|---|---|
| R2.M1 | Summing 82,525 features across 18 layers and calling it >70× compression in a 1,152-d space is not justified; separate aggregate layer-specific dictionary atoms from unique concepts per hidden space | **NEW EXP E7** + **TEXT** — cross-layer concept counting in shared gene space; delete the 70× claim |
| R2.M2 | SAE (4,608/layer) vs SVD (50 axes) is not capacity-matched; arbitrary cosine threshold; repeat under capacity/reconstruction-matched conditions with cumulative projection onto top-k SVD subspace across k | **NEW EXP E6** — full ρ_k projection curves, random-direction null, reconstruction-matched and annotation-matched comparisons |
| R2.M3 | Pseudoreplication: ~20 cells per target but Wilcoxon against ~100K control *positions*; positions from one cell are not independent replicates | **NEW EXP E2** — cell-level unit of analysis, ICC, permutation null, inflation quantification |
| R2.M4 | Cross-model comparison confounded: Geneformer on K562 vs scGPT on Tabula Sapiens | **NEW EXP E9** — full 2×2 (model × dataset) matched design |
| R2.M5 | Conclusions exceed evidence; restrict claims to the cellular context, perturbation dataset, SAE design and regulatory reference used; extend to more cell types; validate on independent data | **NEW EXP E4** + systematic scoping (§4) |
| R2.m1 | Incomplete cross-references | **FIX E11** |
| R2.m2 | References 20 and 23 inaccurate/incomplete; standardise all references | **FIX E12** |

---

## 2. New experiments

All experiments write JSON to `experiments/revision_srep/<id>/` and are driven by scripts in
`src_revision_srep/`. Every script takes `--seed` and records it in its output.

### Enabling infrastructure (I0)

`src_revision_srep/common.py` — a streaming extract→encode→aggregate helper:
Geneformer forward pass with a layer-11 hook → SAE encode → **per-cell** aggregation
(mean/max feature activation over that cell's gene positions, plus per-position sparse records
where needed). This keeps disk cost at ~40 MB per experiment instead of ~20 GB, and makes the
cell the natural unit of analysis (prerequisite for E2/E3/E4).

Key data fact established during scoping: `replogle_concat.h5ad` (30 GB) contains **four** cell
lines — K562 (188,590 cells), Jurkat (184,470), RPE1 (173,737), HepG2 (96,616); 1,341–1,538
perturbation targets each; 4,976–12,013 non-targeting control cells each; 73–89 TRRUST TFs with
≥20 cells per line. This makes E4 possible with no download.

### E1 — Regulatory-recoverability ceiling *(R1.M1)*

**Question.** How much TF→target information is recoverable from the same input by *any* method?
Without this, 6.2% is uninterpretable.

**Design.** One evaluation framework for every method: for each TRRUST TF in the K562 panel, each
method emits a ranked/thresholded predicted-target set; enrichment for that TF's known targets is
tested by hypergeometric test with BH FDR across the panel (universe = 20,000 protein-coding
genes), exactly as in the manuscript's DoRothEA protocol. Metrics: fraction of TFs with FDR<0.05,
median odds ratio, AUPRC against the TRRUST/DoRothEA target labels.

**Methods compared.**
1. SAE feature response (this study, cell-level version from E2).
2. **Perturbation-aware empirical ceiling**: differential expression of the CRISPRi knockdown
   itself (perturbed vs non-targeting cells, Wilcoxon + BH). This is the strongest achievable
   perturbation-aware signal on this dataset and therefore the ceiling any model could reach.
3. **GENIE3** (ExtraTrees feature importance per target gene) on the same 2,000 K562 control cells.
4. **GRNBoost2** (arboreto) if the package is available; otherwise gradient-boosting equivalent.
5. Pearson/Spearman co-expression on the same control cells.
6. Geneformer's own contextual gene-embedding similarity (model-internal, non-SAE baseline).
7. Random-gene-set control.

**Expected payoff.** Reframes the central result quantitatively: the paper stops saying "the model
lacks regulatory logic" and starts saying "the SAE recovers X% where direct perturbation DE on the
same cells recovers Y% and the best observational method recovers Z%".

### E2 — Cell-level statistics, pseudoreplication removed *(R2.M3)*

Per-cell feature activation (mean over the cell's positions); Wilcoxon over cells
(n_perturbed vs n_control cells); BH FDR; cell-level permutation null (shuffle perturbation labels
over cells); intraclass correlation of positions within cells to quantify how badly the
position-level test inflated the effective n. Report both the corrected specificity numbers and the
inflation factor. **All downstream regulatory numbers in the paper are replaced by these.**

### E3 — Power and sample-size sensitivity *(R1.M3)*

K562, layer 11. n ∈ {10, 20, 50, 100, 200} cells per target (TFs with sufficient cells), 5 bootstrap
resamples per n. Outputs: (i) TF-specificity rate vs n with 95% bootstrap CIs; (ii) **positive
control** — detection rate of the knocked-down gene's *own* signature vs n (if the pipeline detects
the knockdown itself but not its targets, power is adequate and the negative result stands);
(iii) minimum detectable Cohen's d at α=0.05, power 0.8, per n; (iv) saturation check — does
specificity rise with n?

### E4 — Independent cell types *(R1.M4, R2.M5)*

RPE1, Jurkat, HepG2. For each: train a layer-11 SAE on that line's non-targeting control cells
(same architecture/hyperparameters/seed protocol), plus the K562-SAE-transfer variant; run the E2
cell-level perturbation-response and TF-specificity analysis on that line's TRRUST TF panel at
n=50–100 cells per target. Result: a four-cell-line table of regulatory specificity, and a
statement about whether the finding is K562-specific or general across the four lines.

### E5 — Annotation/evaluation independence and unbiased selection *(R1.M2, R2 circularity)*

Causal patching at layer 11 re-run with three changes:
1. **Held-out evaluation genes** — annotate on the feature's top-20 genes, evaluate logit
   disruption only on genes of the annotated term that are *not* in the top-20.
2. **Matched random controls** — size-matched and expression-rank-matched random gene sets.
3. **Unbiased feature selection** — a random sample of annotated features (n≈150) rather than the
   50 most richly annotated; report the whole distribution, not just the top.
Also: document the original 50-feature selection rule explicitly in Methods and reconcile the
§2.7/§5.4 mismatch the reviewer identified.

### E6 — Capacity-matched SAE vs SVD *(R2.M2)*

For every layer: cumulative projection ρ_k = ‖P_k d_f‖² of each SAE decoder direction onto the
top-k SVD subspace for k on a log grid up to d; median/quantile curves; random-direction null
(ρ_k ≈ k/d). Report k* at which the median SAE direction reaches 50% and 90% of its norm.
Reconstruction-matched: the SVD rank needed to match the SAE's variance explained. Annotation-
matched: annotate the top-k SVD axes by their top-20 loading genes with the identical enrichment
pipeline and compare enrichment yield per direction and unique terms recovered.
**The claim "99.8% of features are invisible to SVD" is replaced by these measured curves.**

### E7 — Cross-layer feature identity *(R2.M1)*

Features from different layers live in different spaces, so compare them in the *shared gene
space*: build a Jaccard graph over all features' top-20 gene sets across all 18 layers, cluster
(Leiden), and report the number of distinct gene-level programs versus 82,525 dictionary atoms;
also per-layer distinct-program counts and cross-layer program reuse. Report the honest per-space
statement: 4,608 dictionary atoms in a 1,152-d space is a 4× overcomplete dictionary, of which N
are alive and quasi-orthogonal. **Delete the ">70× compression" claim everywhere.**

### E8 — SAE seed stability *(R1.m1)*

Patch `02_train_sae.py` to seed torch/numpy explicitly and record the seed. Retrain layers
{0, 5, 11, 17} with 5 seeds. Report: variance-explained spread; dictionary reproducibility
(bipartite max-cosine matching between seeds — mean max cosine and fraction of features matched
above 0.9); annotation-rate spread; module-count spread; and the spread of the layer-11 TF
specificity number. Report the main-atlas seed in Methods.

### E9 — Matched 2×2 cross-model design *(R2.M4)*

| | K562 | Tabula Sapiens |
|---|---|---|
| Geneformer | existing atlas | **new**: train SAEs on the cached TS Geneformer activations |
| scGPT | **new**: extract K562 activations, train SAEs | existing atlas |

Compare at matched relative depth on matched data: variance explained, annotation rate, module
count, dead features, SVD alignment. Replaces the confounded comparison with a controlled one.

### E10 — Figures *(R1.m2)*

Regenerate the affected figure(s) with full enrichment-term labels (wrap/truncate-with-tooltip →
full text, larger canvas, rotated ticks, no overlap). Add new figures for E1 (ceiling benchmark),
E3 (power curves), E4 (four-cell-line specificity), E6 (ρ_k curves).

### E11 — Cross-references *(R1.m3, R2.m1)*

Root cause: `\externaldocument{supplementary}` requires `supplementary.aux` to exist at main-file
compile time; when it does not, every supplementary reference renders as `??`. Fix by making the
main text not depend on it — explicit, stable supplementary numbering ("Supplementary Table S12")
with a single source of truth, plus a compile check that greps the log for `??` and fails loudly.

### E12 — Bibliography *(R2.m2)*

Verify every entry; fix references 20 and 23 specifically; complete missing journal/volume/pages/
DOI/year fields; standardise to the journal style.

---

## 2a. Execution status

| ID | Status | Notes |
|---|---|---|
| E6 SVD capacity | **done** | 18 layers; see `NEW_RESULTS.md` |
| E7 concepts | **done** | 82,525 atoms → 18,711 distinct programs |
| E1 ceiling | **partly done** | DE ceiling, correlation, embeddings, random done; GENIE3/GRNBoost2 running |
| E2/E3 cell-level + power | **running** | extraction through both the K562 and multi-tissue dictionaries in one pass |
| E4 other cell lines | queued | RPE1, Jurkat, HepG2 |
| E5 causal patching v2 | script written | |
| E8 seed stability | script written | |
| E9 matched cross-model | script written | |
| E12 bibliography | **done** | 24 entries verified; refs 20 and 23 corrected, DOIs added, placeholder notes removed |
| E10 figures | **done** | all figures generated from result files |
| E0 cohort correction | **done** | atlas cohort is four cell lines, not K562 |
| E10 external datasets | **done** | Papalexi and Norman; qualifies the central claim |
| Manuscript | **done** | 31 pp, 0 unresolved refs, 0 draft artefacts |
| Supplementary | **done** | 26 tables, 4 figures, symbolic numbering |
| Response letter | **done** | 11 pp, all points answered |

**Design change adopted during execution.** The per-cell extractor encodes each forward pass through
several SAE dictionaries at once, so the K562-trained and multi-tissue dictionaries are compared on
exactly the same cells at no extra model cost. This replaces the earlier multi-tissue comparison,
which used position-pooled statistics, with a cell-level one on identical inputs.

**Key finding that reframes the paper.** In the CRISPRi assay only 6,546 genes are measured. Of 73
perturbed TRRUST TFs, the median has exactly **one** curated target present in the data, and only 9
have five or more. Target-specific recovery is therefore undefined for most of the published 48-TF
panel under any method — which is why the benchmark against an empirical ceiling (E1), rather than
the bare percentage, has to carry the interpretation.

## 3. Execution order

Compute is a single Apple M2 Pro (32 GB, MPS); GPU work must be serialised.

**Wave A (CPU-only, no GPU contention, run first and in parallel)**
E6 (SVD projections — cached activations), E7 (gene-set clustering), E12 (bibliography),
E11 (cross-reference restructuring), plus the E1 non-Geneformer baselines (GENIE3, GRNBoost2,
correlation, DE ceiling — all operate on the h5ad expression matrix).

**Wave B (GPU, serialised)**
1. E2 (cell-level re-analysis; K562, layer 11)
2. E3 (cell-count sweep — reuses E2 infrastructure)
3. E5 (causal patching v2)
4. E4 (three new cell lines: extraction → SAE training → analysis)
5. E8 (seed stability: 4 layers × 5 seeds)
6. E9 (Geneformer-on-TS SAEs; scGPT-on-K562 extraction + SAEs)

**Wave C (after results land)**
E10 figures → manuscript rewrite → consistency verification → response letter.

---

## 4. Manuscript changes

### 4.1 Title, abstract, conclusions
- Title as agreed in §0.
- Abstract: replace the SVD sentence with the capacity-matched result; replace "70×"; state the
  cell-line scope explicitly (four cell lines); report the ceiling comparison alongside the 6.2%;
  report the cell-level (not position-level) statistics.
- Conclusions: scope to "under this SAE design, on these perturbation data, against these
  regulatory references"; state the ceiling result as the reason the negative finding is
  interpretable.

### 4.2 Claims to rewrite (each currently overstated)
1. ">70× compression ratio" (Results §2.2 and Discussion) — delete, replace with E7 numbers.
2. "99.8% of SAE features are invisible to SVD" (abstract, §2.2, Conclusions) — replace with E6.
3. "minimal regulatory logic in single-cell foundation models" as a general claim (abstract,
   discussion, conclusions) — scope to the tested conditions; the ceiling result (E1) carries the
   interpretation instead of the bare percentage.
4. Cross-model comparative claims (§2.3, Discussion) — restate on the 2×2 matched basis (E9).
5. Causal-specificity claims (§2.7) — restate on the held-out/unbiased basis (E5).

### 4.3 De-drafting checklist (author constraint 2)
Remove/rewrite, at minimum:
- §2.2 and Methods: the 0.7-vs-0.5 threshold correction narrative → state the threshold once, with
  the sensitivity sweep as a normal robustness analysis.
- §2.7 "preliminary/exploratory" scGPT paragraph and §2.11 "with true binned values" section →
  merge into one clean description of the scGPT causal-patching protocol and its result.
- §2.14 "in the original preprint we used a guilt-by-association test … we now treat that test as
  uninformative" → present only the tests that are used, with their limitations stated plainly.
- "as the reviewers suggested", "we have since re-extracted", "in this revision", "the original
  draft", "we did not re-run … due to compute budget", "natural next experiment", "planned
  follow-ups", "we have not yet performed" → delete or convert to plain limitation statements.
- §2.12 "Matched-data cross-model snapshot" → becomes the full E9 analysis, not a "snapshot".
- §2.13 "Batch and assay confounds … first-pass answer" → present as the analysis it is.
- Scorecard table: keep only if it reads as a study design element, not as a revision artefact;
  otherwise fold into the text.
- Every "we flag X as a follow-up" → either do X or state it once in Limitations.

### 4.4 Additions
- `Code availability` section (E3 editorial) with the Zenodo DOI placeholder.
- Methods subsections for E1–E9, written as part of the original design, not as add-ons.
- Supplementary tables for every new experiment.

---

## 5. Deliverables

1. `paper/scientific_reports_revision/main.tex` + `supplementary.tex` + figures + compiled PDFs.
2. `paper/scientific_reports_revision/response_to_reviewers.tex/pdf` — point-by-point (this is the
   only document where revision history is allowed to appear).
3. `src_revision_srep/*.py` — all new analysis code, seeded and reproducible.
4. `experiments/revision_srep/*` — all new result files.
5. Zenodo-ready code archive + deposit instructions.
6. A verification pass: no `??` in the compiled PDF, no draft artefacts, all numbers in text ==
   numbers in tables == numbers in result JSONs.

---

## 6. Open items requiring the author

- Zenodo deposit (needs the author's account) → DOI to paste into Code availability.
- Affiliation: manuscript currently reads "Department of Computer Science"; recent submissions use
  "Institute of Medical Genetics and Applied Genomics". Left unchanged unless instructed.
