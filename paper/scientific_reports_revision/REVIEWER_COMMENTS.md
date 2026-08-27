# Reviewer and editor comments as received

Journal: *Scientific Reports* (manuscript transferred from BMC Genomics)
Submission ID: fbb71648-3418-4139-b1df-d7dd0b5456d6
Manuscript as submitted: *"Sparse autoencoders reveal organized biological knowledge but minimal
regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT"*
Handling editor: Varodom Charoensawan. Assistant editor: Trupti Bodas.

This file records the comments verbatim, as the reference the revision was written against.
The point-by-point replies are in `response_to_reviewers.tex`.

---

## Decision letter

> Dear Dr Kendiukhov,
>
> Your manuscript, "Sparse autoencoders reveal organized biological knowledge but minimal
> regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT",
> has now been assessed.
>
> We invite you to revise your paper, carefully addressing the comments from the reviewers and the
> editor. Please ensure the results are accurately reported, any overstated conclusions are
> rewritten and the limitations of the work fully explained. When your revision is ready, please
> submit the updated manuscript and a point-by-point response. This will help us move to a swift
> decision.
>
> Please note that if your manuscript uses any custom or bespoke computational tool or code, or
> reports a new algorithm, tool, software, or a pipeline (even if individual components are not
> new), the underlying code must be deposited in a recognised DOI-assigning repository (e.g.
> zenodo) and linked either from Methods or a dedicated Code Availability section.

---

## Editor comments

> The reviewers found the work to be useful to the field, especially on the interpretation of
> single-cell foundation models. However, there are a number of key findings that were not
> sufficiently supported by the analyses, and hence some of the conclusions somewhat limited.
> Please find point-by-point comments from the reviewers within this email, as well as in an
> additional attached file. Please highlight additional analyses/re-analyses/additional datasets
> (and appropriate matched controls), and how they strengthen your conclusions, or otherwise tone
> them down.

— Varodom Charoensawan

## In-house editorial comments

> 1. To aid our readers, and to maximize the accessibility of your manuscript, the title should
> have a clear, precise scientific meaning and should not contain a colon. Where possible, the
> title should be read as one concise sentence. Please could you re-write the title ensuring that
> it is informative and appropriate.

> 2. The KEGG pathway database is copyrighted by Kanehisa laboratories and we do require formal
> permission from them to publish this material commercially under an Open Access license. We would
> therefore be grateful if you could get permission to use the KEGG software from the Kanehisa
> laboratory. To obtain this permission please submit the form at the following URL:
> www.kegg.jp/feedback/copyright.html . Please make sure to select 'Scientific Reports' in the
> Publication Detail section.
>
> When using KEGG imagery please ensure that you cite this source in the appropriate figure legend,
> as per the citation guidelines: www.kegg.jp/kegg/kegg1.html . For previous uses, the Kanehisa
> laboratory have happily provided permission.
>
> Please send the permissions document to srep@nature.com

---

## Reviewer 1

> Kendiukhov presents a systematic study using sparse autoencoder (SAE) to investigate the internal
> representations learned by Geneformer and scGPT. The results suggest that both models encode
> organized biological information, including pathways, protein interactions, cell-type programs,
> and functional modules, while showing limited evidence of TF-target-specific regulatory logic. The
> topic is interesting and relevant to the interpretation of single-cell foundation models. However,
> the current evidence does not yet fully support the breadth of the central conclusion. The
> analysis does not sufficiently distinguish limitations of the learned representations from
> limitations arising from the input data, statistical power, and evaluation framework. Please find
> my detailed comments below.

### Major comments

> 1. The manuscript interprets the low TF–target specificity as evidence of limited causal
> regulatory information, but provides no baseline showing how much such information is recoverable
> from the same scRNA-seq input. The authors should compare Geneformer and scGPT with established
> regulatory-inference methods, ideally including a perturbation-aware model, under the same
> evaluation framework.

> 2. The definition of the target genes in the causal-patching analysis is unclear. Section 5.4 uses
> each feature's top-20 activated genes to assign ontology annotations, whereas Section 2.7 defines
> targets as gene positions matching the selected ontology annotation. The manuscript should clarify
> how the target gene set and the 50 richly annotated features were selected, and whether the
> evaluation gene set is independent of the genes used for annotation.

> 3. In Sections 2.9 and 5.7, the analyses use only 20 perturbed cells per target for Geneformer and
> 10 for scGPT, without power or sample-size sensitivity analyses. The authors should test whether
> the reported low TF-target specificity is robust to larger cell numbers and provide bootstrap or
> power estimates.

> 4. The regulatory-specificity analysis is limited to K562 cells. Because both the Geneformer and
> scGPT evaluations use the same Replogle K562 dataset, the authors should validate the finding in
> at least one independent perturbation dataset from another cell type or tissue.

### Minor comments

> 1. The random seed used for SAE training is not reported, and run-to-run variability is not
> assessed. The authors should report the seed and evaluate the stability of the main SAE results
> across repeated training runs.

> 2. The enrichment-term labels in Figure 6 are truncated, and several axis labels overlap.

> 3. Many internal cross-references remain unresolved and are displayed as "Table ??" or
> "Additional file 1: Table ??" throughout the manuscript, including in the Discussion and Methods.
> All cross-references should be corrected before publication.

---

## Reviewer 2

### Summary in the decision letter

> This study applies sparse autoencoders (SAEs) to Geneformer and scGPT and presents what the
> authors describe as the largest interpretability-oriented feature atlas of single-cell foundation
> models to date. The study is innovative and potentially valuable, lying at the intersection of
> single-cell foundation models, biological interpretability, and gene-regulatory analysis. However,
> several substantive methodological and interpretative concerns should be addressed before the
> manuscript can be considered further. Detailed major and minor comments are provided in the
> attached review file.
>
> Recommendation: Major Revision.

> This study applies TopK sparse autoencoders to the residual streams of Geneformer and scGPT. The
> resulting feature atlases show substantial biological annotation, co-activation modules, and
> cross-layer organization. However, CRISPRi analyses reveal limited TF–target-specific responses,
> suggesting that these models capture co-expression and pathway structure more strongly than causal
> regulatory logic.

**Major weaknesses, as listed in the letter**

> The claim that 99.8% of SAE features are "invisible to SVD" is not fully convincing. Comparing
> thousands of SAE directions with only 50 SVD axes using an arbitrary cosine threshold may
> naturally produce low alignment. The reported cross-layer feature count also should not be
> interpreted as a compression ratio within one representation space.

> The causal-patching analysis may be partly circular. Features are annotated using their top genes,
> and specificity is then evaluated on genes belonging to those annotations. Independent gene sets,
> matched random controls, and an unbiased feature-selection procedure are needed.

> The regulatory conclusion is limited by the experimental design. SAEs were trained mainly on
> unperturbed cells, while validation used a small K562-only CRISPRi sample and inconsistent
> specificity definitions. The conclusion should therefore be framed as limited detection by the
> current SAE pipeline rather than evidence that the models themselves lack regulatory logic.

### Attached review file

> This study applies sparse autoencoders (SAEs) to Geneformer and scGPT and constructs what the
> authors describe as the largest interpretability-oriented feature atlas of single-cell foundation
> models to date. The work is innovative and potentially valuable, lying at the intersection of
> single-cell foundation models, biological interpretability, and gene-regulatory analysis.
> Nevertheless, several substantive concerns and limitations should be addressed before the
> manuscript can be considered further.
>
> Recommendation: Major Revision

#### Major

> **1. Overinterpretation of the Number of Cross-Layer SAE Features**
>
> The authors sum the 82525 SAE features identified across different Geneformer layers and infer
> that the model compresses more than 70-fold as many independent concepts into a 1152-dimensional
> representation space. This calculation is not conceptually justified. Each layer has its own
> 1152-dimensional hidden-state space and its own independently trained SAE dictionary.
> Consequently, features obtained across the 18 layers cannot be regarded as independent concepts
> that coexist within a single 1152-dimensional vector space. The authors should revise this
> interpretation and distinguish the aggregate number of layer-specific active dictionary features
> from the number of unique biological concepts represented within any one hidden-state space.

> **2. The Comparison Between SAE and SVD Is Not Capacity-Matched**
>
> The manuscript directly compares approximately 4608 SAE features per layer with only 50
> singular-vector directions. Because the representational capacities differ by nearly two orders of
> magnitude, the observation that 99.8% of SAE features are not aligned with the top-50 SVD axes
> does not establish that the corresponding information is invisible to SVD. The comparison should
> therefore be repeated under capacity- or reconstruction-matched conditions, ideally by quantifying
> the cumulative projection of each SAE direction onto the top-k SVD subspace across a range of k
> values.

> **3. Potentially Serious Pseudoreplication in the CRISPRi Perturbation Analysis**
>
> The Methods indicate that each perturbation target is represented by only approximately 20 cells,
> yet the Wilcoxon tests appear to compare these observations with "100K control positions." If
> gene-token positions are treated as independent observations, thousands of positions originating
> from the same cell are effectively counted as independent biological replicates. This would
> markedly inflate the effective sample size and underestimate statistical uncertainty.

> **4. Substantial Confounding in the Cross-Model Comparison of Geneformer and scGPT**
>
> The Geneformer activations are derived from the K562 cell line, whereas the scGPT activations are
> derived from the multi-tissue Tabula Sapiens dataset. This introduces substantial confounding into
> the cross-model comparison and makes it difficult to determine whether the reported differences
> arise from model architecture or from the underlying input data. Accordingly, conclusions
> regarding relative model performance are not sufficiently supported by the current analysis.

> **5. The Conclusions Extend Beyond the Evidence Provided**
>
> The analysis of regulatory specificity is based primarily on K562 cells, a limited set of CRISPRi
> perturbations, and TF-target relationships curated in TRRUST and DoRothEA. Although the study does
> not detect broad and reproducible TF-target specificity under this analytical framework, this
> negative finding cannot be generalized to conclude that Geneformer, scGPT, or single-cell
> foundation models as a class lack regulatory logic. The claims in the title, abstract, and
> Discussion should therefore be restricted to the specific cellular context, perturbation dataset,
> SAE design, and regulatory-reference framework evaluated here. The authors are encouraged to
> extend the analysis to multiple representative cell types and tissues to establish the
> generalizability of the findings. And the key results should be further validated by using
> independent external datasets or wet-lab experiments.

#### Minor

> 1. The manuscript contains incomplete cross-references, including "Additional file 1: Table ??" at
> Lines 96, 239, 245, and 252, among other locations.

> 2. Several references appear to be inaccurate or incompletely formatted, including References 20
> and 23. The authors should verify the bibliographic details and standardize all references
> according to the journal's required format.
