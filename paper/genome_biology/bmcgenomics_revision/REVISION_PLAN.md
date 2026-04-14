# BMC Genomics Revision Plan — Response to Reviewers

**Manuscript:** "Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT"
**Target:** `paper/genome_biology/bmcgenomics_revision/main.tex`
**Plan created:** 2026-04-13

---

## 0. Executive summary

Three reviewers raise a consistent set of issues, ordered roughly from cheapest to most expensive:

1. **Factual errors about scGPT architecture and training objective** (R3) — text-only fix, must do first.
2. **Writing polish / LLM-smell / figure captions / reproducibility scripts** (R2 minor) — low effort, high yield.
3. **Missing methodological justifications** (SVD threshold 0.7, PMI graph sizes, Leiden resolution 1.0, hyperparameter choice) (R1-2, R2-2, R2-3) — text + small ablation.
4. **Softening over-reaching claims** ("minimal regulatory logic", "model is the bottleneck") (R3) — requires adding caveats, not new experiments.
5. **New experiments**, in increasing order of cost:
   - Permutation null for cross-layer PMI + consecutive-layer analysis (R1-4).
   - Unannotated-feature characterization with cell-type + CRISPRi tests (R1-5).
   - Re-run perturbation mapping with DoRothEA and/or ChIP-Atlas (R1-1).
   - Hyperparameter ablation on one layer (R1-2, R2-2).
   - scGPT perturbation response mapping on Replogle (R2-4).
   - scGPT true-value re-extraction + causal patching (R1-3, R2-1).
   - Matched-data cross-model comparison (R3-2).
   - Optional / stretch: finetuned-scGPT on Replogle (R3-6), batch/assay feature analysis (R3-5), second CRISPRi cell line (R1-1).

The core narrative ("rich co-expression structure, limited regulatory specificity") survives. The revision's job is to shore up controls, soften absolute claims, fix factual errors, and add missing ablations.

---

## 1. Factual corrections (must do first, blocks everything else)

### 1.1 scGPT uses binned, not continuous, expression values
- **Raised by:** R3.
- **Locations:**
  - `main.tex:162` — "continuous-value gene encoding (vs. Geneformer's rank-value tokens)"
  - `main.tex:209` (Table 3, `tab:model_comparison`) — "Continuous expression"
  - `main.tex:567` — Methods paragraph describing scGPT inputs.
  - `README.md` — any similar phrasing in the overview.
- **Fix:** Replace "continuous-value" with "binned expression values (default 51 bins)" everywhere. Cite the scGPT paper's binning scheme explicitly in Methods.
- **Add:** a short Methods paragraph ("Input handling for each model") describing:
  - Geneformer: rank-value tokenization (genes sorted by expression, rank becomes token ID); no explicit expression magnitude fed to the model.
  - scGPT: dual input of gene-ID tokens + binned expression value embeddings; our activation extraction fed the actual per-cell binned values (or, if we used uniform values anywhere, state this explicitly and bound the affected analysis).

### 1.2 Training objectives are swapped
- **Raised by:** R3.
- **Locations:**
  - `main.tex:162` — "masked gene prediction training objective (vs. next-token prediction)" — this has the two models reversed.
  - `main.tex:209` (Table 3).
- **Correct assignment:**
  - Geneformer: masked language modeling (BERT-style masked gene prediction).
  - scGPT: generative / specialized causal attention ("next-token"-style) gene prediction.
- **Fix:** Swap the two in both the prose and the table. Cite the Geneformer and scGPT papers on the exact wording of their objectives.

### 1.3 Sanity-check every other architectural claim
- Re-read sections 2.3 and Methods against the Geneformer V2-316M and scGPT whole-human model cards / papers. Flag any other inaccuracies (attention head counts, vocab sizes, positional encoding, etc.) before submitting the revision.

---

## 2. Writing, figures, and reproducibility (low cost, high value)

### 2.1 Language polish
- **Raised by:** R2 minor #1.
- **Action:** Do a top-to-bottom editing pass removing LLM-register phrases ("It is worth noting that", "rich biological organization", "in stark contrast", repeated "massive", etc.). Tighten Background and Conclusions in particular — R2 explicitly flagged those two. Target ~15% word reduction in both.
- **Deliverable:** cleaned `main.tex`. If a native-speaker editor is available, route it through them; otherwise self-edit with an explicit style target.

### 2.2 Figure S2–S3 captions
- **Raised by:** R2 minor #2.
- **Action:** In `supplementary.tex`, update the captions for S2–S3 (and S4 if applicable) to state explicitly: "Force-directed (Fruchterman–Reingold) layout of the intra-module co-activation graph; this is **not** a UMAP or t-SNE projection." Mention NetworkX as the implementation.

### 2.3 Software/hardware reporting
- **Raised by:** R2 minor #3.
- **Action:** Add a "Computational environment" subsection to Methods listing:
  - Python, PyTorch, NumPy, scikit-learn, scanpy, leidenalg, networkx, umap-learn, statsmodels versions (pull from `requirements.txt` / the env used).
  - Exact hardware: Apple Silicon model (M-series chip, core count), unified memory size, macOS version, MPS availability.
  - Wall-clock time for the longest steps (activation extraction, SAE training per layer, PMI graph construction, perturbation mapping).
- **Deliverable:** new `\subsection{Computational environment}` near the end of Methods + a mirror entry in `README.md`.

### 2.4 Single-command reproducibility
- **Raised by:** R2 minor #4.
- **Action:** Create two deliverables in the main repo (not the paper folder):
  - `run_all.sh` — thin wrapper that runs Phase 1 → Phase 2 → Phase 3 in order, with clear `# edit these paths` markers at the top. It should call the existing `src/01_*.py` → `src/12e_*.py` scripts in sequence and do the same for `scgpt_src/`.
  - `notebooks/walkthrough.ipynb` — a narrated notebook that re-runs one layer end-to-end (extract → train → annotate → causal patch → perturb) on a small subsample, so reviewers can reproduce the pipeline without 400 GB of disk.
- **Deliverable path:** `subproject_42_sparse_autoencoder_biological_map/run_all.sh` and `subproject_42_sparse_autoencoder_biological_map/notebooks/walkthrough.ipynb`. Reference both in `README.md` Quick Start.

---

## 3. Missing methodological justifications (text + tiny experiments)

### 3.1 SVD threshold of 0.7
- **Raised by:** R2 major #2.
- **Action:** In Section 2.2 (the SVD comparison), add one paragraph stating:
  - The 0.7 cosine threshold is motivated by standard practice in linear-probe / feature-alignment studies (cite: Bricken et al. 2023, Cunningham et al. 2023, Gao et al. 2024 — verify each actually uses this threshold; if not, say "following prior SAE work which treats cosine > 0.7 as the onset of strong directional alignment").
  - Report a sweep over thresholds {0.3, 0.5, 0.7, 0.9} at layer 11 showing how the "SVD-aligned fraction" and annotation-exclusivity conclusions vary. The headline result is robust only if the 0.9 threshold still leaves >99% of features non-aligned and the 0.3 threshold still leaves the vast majority of ontology enrichments in the "novel" set.
- **Deliverable:** new paragraph in Section 2.2 + a short table in Additional file 1 (new Table S-SVD).
- **Code:** extend `src/05_compare_svd.py` to loop over thresholds and emit the sweep table.

### 3.2 PMI graph dimensions and memory footprint
- **Raised by:** R2 major #3.
- **Action:** In Section 2.6 / Methods add:
  - Number of nodes (alive features per layer, ~4608 Geneformer / ~2048 scGPT).
  - Number of edges before and after the significance threshold.
  - Peak memory for PMI computation (the dense activation matrix, the joint-frequency table, and the Leiden input graph).
  - Runtime per layer.
- **Deliverable:** Methods paragraph + Additional file 1 table (Table S-PMI) with per-layer edge counts and memory usage.
- **Code:** add logging to `src/07_feature_coactivation.py` (or read off from existing run logs) to report `psutil` peak RSS and `len(edges)`.

### 3.3 Leiden resolution = 1.0
- **Raised by:** R2 major #3.
- **Action:** Justify choice of resolution = 1.0. Options:
  - Cite Traag et al. 2019 and standard scanpy practice (default 1.0 for Leiden).
  - **And** sweep resolution over {0.5, 0.75, 1.0, 1.5, 2.0} at layer 0 and layer 11 of Geneformer, reporting module count, mean module size, and annotation coverage. Show that module identity is stable (large Adjusted Rand Index across resolutions ~ >0.7).
- **Deliverable:** new Methods paragraph + Table S-Leiden in supplementary.

### 3.4 Hyperparameter ablation: expansion ratio × k
- **Raised by:** R1 #2, R2 #2.
- **Action:** Grid-train SAEs at a representative layer (Geneformer L11) over:
  - Expansion ratio ∈ {2×, 4×, 8×, 16×} → {2304, 4608, 9216, 18432} features.
  - k ∈ {16, 32, 64, 128}.
  - 16 configurations total (or drop to 9 if compute is tight: {2×, 4×, 8×} × {16, 32, 64}).
- For each configuration report:
  - Variance explained, dead-feature count, mean |cos|.
  - Annotation rate (fraction of alive features with ≥1 FDR<0.05 enrichment).
  - Module count (Leiden at fixed resolution 1.0).
  - Perturbation TF specificity on the TRRUST 48-TF panel.
- **Expected outcome:** qualitative conclusions survive (TF specificity stays single-digit across the whole grid). If they don't, we will need to reframe the headline number.
- **Deliverable:** new subsection in Results ("SAE hyperparameter ablation") or in Additional file 1 as a small self-contained section, with a heatmap figure and a summary table.
- **Code:** new script `src/13_hyperparam_ablation.py` that wraps existing training/annotation/perturbation routines in a grid loop. Reuse the L11 K562 activations already on disk.
- **Compute estimate:** ~16 × (SAE train + annotate + perturb) ≈ half a day on the existing Apple Silicon setup, assuming ≤30 min per SAE.

---

## 4. Claims to soften (text-only but central)

### 4.1 "Minimal regulatory logic" is too strong
- **Raised by:** R3.
- **Rewording targets:**
  - Title: leave as-is but add a qualifying phrase in the abstract Conclusions sentence ("encode organized biological knowledge but, under the conditions tested, minimal causal regulatory logic").
  - Abstract + Conclusions: replace "establishes model representations as the bottleneck" with "suggests, but does not fully establish, that the limitation lies in model representations rather than in SAE training data or methodology. A perturbation-rich training regime or an alternative SAE design could behave differently."
  - Discussion paragraph "The multi-tissue control establishes the model as the bottleneck" — rewrite to "The multi-tissue control is consistent with, but does not prove, a model-level limitation. Alternative explanations remain open: (a) the pooling strategy is too simple, (b) SAEs trained on baseline/control activations may be inherently blind to TF→target wiring, (c) a different SAE objective (e.g., supervised-by-perturbation) could recover more signal."
- **Deliverable:** coordinated edits across abstract, Discussion section "The co-expression–regulation dichotomy...", and Conclusions.

### 4.2 Acknowledge the control-state training-data caveat
- **Raised by:** R3.
- **Action:** Add a dedicated paragraph to the Discussion, before "Limitations":
  - Both K562 CRISPRi-control cells and Tabula Sapiens are unperturbed / baseline populations. SAEs trained on such data are structurally biased toward cell-state / co-expression features; expecting them to recover TF→target regulatory wiring in a strong sense is arguably asking for too much.
  - Point forward: a fruitful next step is to train SAEs on pooled perturbed + control activations (where the label is the perturbation identity) and see whether TF-specific features emerge.

### 4.3 TRRUST is only one reference; expand Limitations
- **Raised by:** R1 #1.
- **Action:** Add explicit limitations bullet: TRRUST is literature-mined and covers a small slice of known TF biology; DoRothEA, ChIP-Atlas, CellOracle priors, and CollecTRI cover complementary relationships. See §5.3 for the experimental complement.

### 4.4 Cross-model comparison interpretation (Fig 3)
- **Raised by:** R3.
- **Action:** Add a sentence in Section 2.3 and at Figure 3's caption: "Because Geneformer activations were extracted from K562 and scGPT activations from Tabula Sapiens, cross-model differences in variance explained, annotation rate, and module count conflate architecture with input distribution. See Section X.Y for a matched-data comparison." If §5.7 is not performed, replace the last clause with "and should be interpreted qualitatively rather than as a controlled architecture contrast."

---

## 5. New experiments (ordered by priority)

Each entry lists: trigger, deliverable, rough effort, and what would falsify the current claims.

### 5.1 Cross-layer PMI permutation null + consecutive-layer sweep — **high priority**
- **Trigger:** R1 #4.
- **Problem:** "97–99.8% of features are information highways" is suspiciously high when TopK k=32 forces dense co-activation.
- **Experiment A (permutation null):**
  - For each layer pair, build 100 shuffled matrices by permuting feature activations within each row (destroys cross-feature co-occurrence while preserving marginals and TopK structure).
  - Recompute PMI and report the 99th-percentile null PMI.
  - Re-report the "highway" fraction using the data-driven threshold (max of 3.0 and 99th-percentile null) rather than the fixed 3.0.
- **Experiment B (consecutive layers):**
  - Compute cross-layer PMI for every consecutive pair (L0→L1, L1→L2, …, L16→L17) in Geneformer and (L0→L1, …, L10→L11) in scGPT.
  - Report a layer-by-layer "highway fraction" curve; check whether connectivity is smooth or has sharp drops (matching intuitions from the Wang et al. 2025 scDrugMap paper about layer-specific representation quality).
- **Deliverables:**
  - New Figure 7 panel or supplementary figure showing (i) null PMI distribution vs. observed, (ii) per-layer highway curve.
  - Updated numbers in Section 2.7. Expect the 97–99.8% range to drop after null correction; state the revised figures honestly.
  - New Additional file 1 Table S-Highways.
- **Code:** extend `src/11_computational_graph.py` with a `--null-permutations` flag and a consecutive-layer loop.
- **Effort:** 1–2 days compute + analysis.

### 5.2 Unannotated-feature characterization — **high priority**
- **Trigger:** R1 #5.
- **Problem:** "95–98.5% of unannotated features co-activate with annotated ones" is tautological because modules cover 96–99.5% of features.
- **New tests for unannotated features at layers 0, 5, 11, 17:**
  1. **Cell-type specificity:** for each unannotated feature, compute enrichment against the 56 Tabula Sapiens cell types (Fisher's exact, BH FDR<0.05). Report fraction that are cell-type-specific at each layer.
  2. **Perturbation responsiveness:** for each unannotated feature, test whether it responds (Wilcoxon FDR<0.05, |effect|>0.5) to any of the 100 Replogle CRISPRi targets. Report fraction responsive and distribution of response counts.
  3. **Tightened noise control:** report each of the above against a matched random-permutation null for unannotated features (to guard against "anything will look non-random in TopK space").
- **Optional (R1 #5 also suggests):** compare SAE-feature gene lists to representations from an orthogonal method (Wang et al. 2025 HECLIP is cited; a simpler check is gene–gene similarity via scVI or the raw expression covariance) and report agreement.
- **Deliverables:**
  - Table S-Unannotated with the three fractions per layer.
  - New subsection "Cell-type and perturbation evidence for unannotated features" replacing the guilt-by-association paragraph in Section 2.11.
- **Code:** new script `src/14_unannotated_characterize.py` reusing `09_perturbation_response.py` and the cell-type enrichment code.
- **Effort:** 1 day.

### 5.3 Alternative regulatory databases: DoRothEA and ChIP-Atlas — **high priority**
- **Trigger:** R1 #1.
- **Experiment:** re-run the perturbation specificity test using:
  - **DoRothEA** (A+B+C confidence levels, ~278 TFs for human).
  - **CollecTRI** (~1200 TFs, broader coverage, modern curation).
  - Optionally **ChIP-Atlas** peak-based TF targets (filter by ChIP-seq peak score).
- **Report:** per-database specificity rate, per-TF concordance (how many TFs are called "specific" under multiple databases), and a combined-database panel.
- **Expected outcome:** specificity should stay in the 5–15% range. If it shoots up under DoRothEA, that strengthens the "training-data-dependent" interpretation and weakens the model-bottleneck claim — either way, report honestly.
- **Deliverables:**
  - New Table in Section 2.8 (or S-Regulatory) listing per-database specificity.
  - Updated headline framing to present the numbers as a range, not a single "6.2%".
- **Code:** new script `src/15_perturbation_multi_db.py` parameterizing the database argument inside the existing Fisher test. Downloads:
  - DoRothEA: Bioconductor or `decoupler-py`.
  - ChIP-Atlas: REST API or bulk download.
- **Effort:** 1 day per database.

### 5.4 scGPT perturbation mapping on Replogle — **high priority**
- **Trigger:** R2 major #4.
- **Problem:** Section 2.9 runs perturbation mapping only on Geneformer; the central negative finding is not cross-validated on scGPT.
- **Experiment:**
  - Extract scGPT activations on Replogle K562 perturbed cells (reuse gene-ID + binned-expression input pipeline). **Critical:** preserve true binned expression values this time (see §5.5 — merge this extraction run with it).
  - Encode through scGPT SAEs at L4, L7, L11 and run the same perturbation response pipeline.
  - Report TF specificity for scGPT (48 TRRUST TFs; also DoRothEA if §5.3 is done).
- **Expected outcome:** similar low specificity. If scGPT is dramatically better or worse, that is itself a finding.
- **Deliverables:**
  - Extend Table 6 (perturbation) with scGPT columns.
  - New figure panel or update Fig 8.
  - One Discussion paragraph comparing.
- **Code:** new `scgpt_src/09_perturbation_response.py` mirroring the Geneformer version.
- **Effort:** 2–3 days including extraction.

### 5.5 scGPT true-value causal patching — **high priority**
- **Trigger:** R1 #3, R2 major #1.
- **Problem:** current scGPT causal patching used uniform-1.0 proxy values, giving median 0.98× specificity; this makes the side-by-side table misleading.
- **Experiment:**
  - Re-extract scGPT activations for ~3000 Tabula Sapiens cells with the actual per-cell binned expression values (confirm the scGPT forward hook exposes the value-embedding inputs).
  - Retrain SAEs at layer 7 (at minimum) on these activations, or re-encode using existing SAEs if the distribution shift is small.
  - Re-run `08_causal_patching.py` with real inputs.
  - Report new median specificity ratio and compare to Geneformer.
- **Deliverables:**
  - Updated Table 4 / causal-patching scGPT paragraph.
  - If specificity stays near 1.0×, interpret as a genuine property of scGPT, not a data artifact. If it jumps, acknowledge the original 0.98× was artifactual and update Figure 6 accordingly.
- **Fallback (if re-extraction blocked):** R1 offers this explicitly — *label the scGPT causal results as preliminary / exploratory in Results and Table 4 and remove them from the cross-model comparison table.* This is the minimum acceptable fix even if no new compute runs.
- **Code:** audit `scgpt_src/01_extract_activations.py` to locate the uniform-1.0 line, fix, re-extract, re-run `08_causal_patching.py`.
- **Effort:** 2–4 days depending on storage and whether SAE retraining is needed.

### 5.6 Batch / assay-specific features in scGPT — **medium priority**
- **Trigger:** R3.
- **Experiment:** annotate each scGPT feature with:
  - Mutual information between feature activation and Tabula Sapiens donor ID.
  - MI between feature activation and sequencing assay / 10x chemistry version.
- **Report:** fraction of features whose top-1 categorical correlate is donor or assay rather than biology. Cross-tabulate with ontology annotation — are "annotated" features less batch-driven?
- **Deliverables:** one new supplementary table (Table S-Batch), one paragraph in Section 2.10 (cell-type enrichment) or a new subsection "Batch and assay confounds".
- **Code:** `scgpt_src/16_batch_confound.py` using metadata from the Tabula Sapiens `.obs` frame.
- **Effort:** half a day.

### 5.7 Matched-data cross-model comparison — **medium priority**
- **Trigger:** R3.
- **Problem:** Geneformer activations come from K562; scGPT activations from Tabula Sapiens. The cross-model "architecture" comparison in Fig 3 confounds architecture with input distribution.
- **Experiment:**
  - Extract Geneformer activations on the same 3000 Tabula Sapiens cells used for scGPT (already done for cell-type analysis per line 499; confirm the storage is still there).
  - Retrain Geneformer SAEs on Tabula-Sapiens-derived activations at layers 0, 5, 11, 17.
  - OR more cheaply: encode Tabula Sapiens Geneformer activations through the existing K562-trained Geneformer SAEs and report the matched-data metrics (variance explained, annotation rate, module count). This is less clean but does not require retraining.
- **Report:** a matched-input version of Table 3 and Figure 3, making clear which columns are K562-vs-Tabula-Sapiens and which are matched-input.
- **Fallback (cheapest):** just add a prominent caveat to Section 2.3 and Figure 3 caption (see §4.4) and defer the matched experiment to future work.
- **Effort:** 2–3 days if retraining; <1 day for encode-only.

### 5.8 Second CRISPRi cell line — **low priority / stretch**
- **Trigger:** R1 #1.
- **Data source:** Replogle 2022 RPE1 essentials (smaller panel) or Nadig/Adamson datasets.
- **Experiment:** rerun perturbation mapping in a non-K562 context to see whether tissue origin shifts TF specificity.
- **Effort:** 3–5 days (new extraction, retraining optional). **Recommended only if we want to elevate the paper's claims about generality.** Otherwise, acknowledge the K562-only limitation explicitly in §4.3 and §4.1.

### 5.9 Finetuned scGPT on Replogle — **low priority / stretch**
- **Trigger:** R3 final question.
- **Experiment:** use the published scGPT perturbation-finetuned checkpoint (if released) or finetune scGPT on Replogle CRISPRi, retrain SAEs, rerun perturbation specificity. Hypothesis: perturbation-aware training raises TF specificity substantially; this would directly support the paper's recommendation in its Discussion.
- **Effort:** 5–7 days (finetuning + full downstream pipeline).
- **Recommendation:** mention in Discussion as the clearest next experiment and attempt only if time permits. Framing the paper as "here is the baseline; the next paper explores whether finetuning closes the gap" is a legitimate stance.

---

## 6. Manuscript-section edit map

For each section, list the concrete edits required.

| Section | Edits |
| --- | --- |
| **Abstract** | Fix scGPT architecture wording (§1.1); soften "minimal regulatory logic" (§4.1). |
| **Background** | Language polish (§2.1); add the "input handling for each model" sentence or forward-reference to Methods (§1.1). |
| **§2.1 Atlas overview** | No changes expected beyond polish. |
| **§2.2 SVD comparison** | Add threshold justification paragraph + cite sweep table (§3.1). |
| **§2.3 scGPT atlas + cross-model Table 3** | Fix encoding & objective errors (§1.1, §1.2); add data-confound caveat for Fig 3 (§4.4). |
| **§2.4 U-shape profile** | No changes. |
| **§2.5 Cross-layer tracking** | No changes. |
| **§2.6 Co-activation modules** | Add PMI graph dimension / memory / resolution-sweep methods pointer (§3.2, §3.3). |
| **§2.7 Causal patching** | Re-present scGPT results as preliminary **or** with true-value inputs (§5.5); add labelling caveat. |
| **§2.8 Perturbation mapping** | Add DoRothEA / ChIP-Atlas numbers (§5.3); extend with scGPT perturbation results (§5.4). |
| **§2.9 Multi-tissue control** | Soften "establishes the model as the bottleneck" claim (§4.1). |
| **§2.10 Cross-layer highways** | Replace highway fractions with null-corrected values (§5.1); add consecutive-layer panel. |
| **§2.11 Unannotated features** | Replace guilt-by-association with cell-type + CRISPRi tests (§5.2). |
| **Discussion** | Add control-state caveat (§4.2); add TRRUST-coverage caveat (§4.3); soften bottleneck language (§4.1). |
| **Conclusions** | Soften "minimal regulatory logic" and "establish model as bottleneck". |
| **Methods** | Add input-handling paragraph (§1.1); SVD threshold rationale (§3.1); PMI memory + graph sizes (§3.2); Leiden resolution justification (§3.3); Computational environment subsection (§2.3). |
| **Supplementary** | New tables: S-SVD, S-PMI, S-Leiden, S-Hyperparam, S-Highways, S-Unannotated, S-Regulatory, S-Batch. Fix S2–S3 captions (§2.2). |

---

## 7. Execution order and dependencies

**Phase A — text-only, no compute (target: 1–2 days).** Do these first so the revision is coherent even if later experiments slip.
1. Factual corrections §1.1–§1.3.
2. Soften claims §4.
3. Figure caption fixes §2.2.
4. Software/hardware reporting §2.3 (collect info from existing logs).
5. Language polish §2.1.

**Phase B — small ablations and analyses on existing activations (target: 3–5 days).**
6. SVD threshold sweep §3.1.
7. PMI memory logging + Leiden resolution sweep §3.2, §3.3.
8. Permutation null + consecutive-layer PMI §5.1.
9. Unannotated feature characterization §5.2.
10. Batch/assay analysis §5.6.
11. Reproducibility scripts §2.4.

**Phase C — new SAE training and re-extraction (target: 1–2 weeks).**
12. Hyperparameter ablation at L11 §3.4.
13. DoRothEA / ChIP-Atlas perturbation rerun §5.3.
14. scGPT perturbation mapping §5.4 *(depends on §5.5 for clean inputs).* 
15. scGPT true-value re-extraction + causal patching §5.5.
16. Matched-data cross-model comparison §5.7.

**Phase D — stretch (only if Phase C completes with buffer).**
17. Second CRISPRi cell line §5.8.
18. Finetuned scGPT §5.9.

**Critical path:** Phase A → §5.5 (scGPT true-value) → §5.4 (scGPT perturbation). §5.3 (DoRothEA) and §3.4 (hyperparam ablation) run in parallel with the scGPT extraction.

---

## 8. Response-to-reviewers skeleton

When writing the cover letter / point-by-point response, structure it as:

- Reviewer 1, point 1 (TRRUST/cell line) → §4.3 + §5.3 (+ §5.8 if done).
- Reviewer 1, point 2 (hyperparams) → §3.4.
- Reviewer 1, point 3 (scGPT causal proxy) → §5.5 (or preliminary-label fallback).
- Reviewer 1, point 4 (PMI baseline + consecutive layers + scDrugMap citation) → §5.1; add citation.
- Reviewer 1, point 5 (unannotated features + HECLIP citation) → §5.2; add citation.
- Reviewer 2, major 1 (scGPT proxy) → §5.5.
- Reviewer 2, major 2 (SVD threshold) → §3.1.
- Reviewer 2, major 3 (PMI graph + Leiden) → §3.2, §3.3.
- Reviewer 2, major 4 (scGPT perturbation) → §5.4.
- Reviewer 2, minor 1–4 → §2.1–§2.4.
- Reviewer 3, factual errors → §1.
- Reviewer 3, cross-model confound → §5.7 + §4.4.
- Reviewer 3, control-state caveat → §4.2.
- Reviewer 3, bottleneck overclaim → §4.1.
- Reviewer 3, batch/assay features → §5.6.
- Reviewer 3, finetuning → §5.9 (stretch) + Discussion paragraph.

Every change should be highlighted in the revised manuscript (e.g., via `\textcolor{blue}{...}` or line-numbered diffs) and cross-referenced in the response document.

---

## 9. Risk register

- **Hyperparameter ablation kills the narrative.** If e.g. 8× expansion with k=64 produces 25% TF specificity, the "minimal regulatory logic" framing collapses. Mitigation: run the ablation early (Phase B/C boundary) so we can reframe before writing the response letter.
- **scGPT true-value re-extraction reveals a bug elsewhere.** Mitigation: keep the preliminary-label fallback §5.5 ready.
- **Permutation null invalidates the 97–99.8% highway claim.** Expected; the revised numbers will likely drop but should still support a "pervasive cross-layer connectivity" claim.
- **DoRothEA specificity is much higher than TRRUST.** Treat as a finding, not a problem — it strengthens the "training data / annotation matters" angle and slightly weakens the model-bottleneck claim. Discussion §4.1 is already softened to accommodate.
- **Compute budget.** Phase C is the bottleneck. If storage runs out, prioritize §5.5 > §5.4 > §5.3 > §3.4 > §5.7.

---

## 10. Deliverables checklist

- [ ] `main.tex` — all edits per §6 edit map.
- [ ] `supplementary.tex` — new tables S-SVD, S-PMI, S-Leiden, S-Hyperparam, S-Highways, S-Unannotated, S-Regulatory, S-Batch; fixed S2–S3 captions.
- [ ] `response_to_reviewers.tex` — point-by-point per §8.
- [ ] Figures: updated Fig 3 (caveat), Fig 6 (scGPT causal), Fig 7 (null-corrected), Fig 8 (DoRothEA / scGPT panels).
- [ ] `src/13_hyperparam_ablation.py`, `src/14_unannotated_characterize.py`, `src/15_perturbation_multi_db.py`.
- [ ] `scgpt_src/09_perturbation_response.py`, `scgpt_src/16_batch_confound.py`; fixed `scgpt_src/01_extract_activations.py` (true expression values).
- [ ] Extended `src/05_compare_svd.py`, `src/07_feature_coactivation.py`, `src/11_computational_graph.py`.
- [ ] `run_all.sh` and `notebooks/walkthrough.ipynb`.
- [ ] Updated `README.md` with the Computational environment block and pointers to the new scripts.
