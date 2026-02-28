# Genome Biology Submission Instructions

## Pre-Submission Checklist

- [ ] Main manuscript (`main.tex`) compiles cleanly (0 warnings, 0 errors)
- [ ] Supplementary file (`supplementary.tex`) compiles cleanly
- [ ] Abstract is structured (Background / Results / Conclusions) and ≤250 words
- [ ] Keywords listed (3–10) after abstract
- [ ] All sections present: Background → Results → Discussion → Conclusions → Methods → Declarations
- [ ] All Declarations subsections present (Ethics, Consent, Competing interests, Authors' contributions, Funding, Availability, Acknowledgements)
- [ ] References in Vancouver numbered style [1], [2], ...
- [ ] Line numbers enabled (`\linenumbers`)
- [ ] Figure legends: titles ≤15 words, legends ≤300 words each
- [ ] All supplementary references use "Additional file 1: Table S1" format
- [ ] GitHub repository published and accessible: https://github.com/Biodyn-AI/bio-sae
- [ ] Interactive atlases live and accessible:
  - https://biodyn-ai.github.io/geneformer-atlas/
  - https://biodyn-ai.github.io/scgpt-atlas/
- [ ] ORCID ready for author metadata

---

## Step 1: Create Account / Log In

1. Go to **https://www.editorialmanager.com/gbiomed/** (Genome Biology submission portal via Editorial Manager)
2. Create an account if you don't have one, or log in
3. Complete your profile (affiliation, ORCID, etc.)

---

## Step 2: Start New Submission

1. Click **"Submit New Manuscript"**
2. Select article type: **Research** (also called "Research Article")
3. This is the standard full-length article type for Genome Biology

---

## Step 3: Enter Metadata

### Title
```
Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT
```

### Authors
- **Ihor Kendiukhov** — Department of Computer Science, University of Tübingen, Tübingen, Germany
- Email: kendiukhov@gmail.com
- ORCID: [your ORCID]

### Abstract
Copy the structured abstract from `main.tex` (Background / Results / Conclusions sections). Ensure ≤250 words total. Remove LaTeX formatting (math symbols → plain text).

### Keywords
```
sparse autoencoders, single-cell foundation models, Geneformer, scGPT, mechanistic interpretability, superposition, gene regulatory networks, co-expression, feature atlas
```

---

## Step 4: Upload Files

Upload in this order:

| Order | File | Type | Description |
|-------|------|------|-------------|
| 1 | `main.pdf` | Manuscript | Main manuscript (compiled from main.tex) |
| 2 | `supplementary.pdf` | Additional file 1 | Supplementary tables and figures (compiled from supplementary.tex) |
| 3 | `figures/fig1_atlas_overview.pdf` | Figure | Fig. 1 (may be requested separately) |
| 4 | ... | Figure | Figs. 2–8 (individual PDFs if required) |

**Note:** Some journals accept a single PDF with embedded figures. Check the submission system's guidance — Genome Biology typically accepts a single manuscript PDF with figures inline, plus separate Additional files.

### Figure files (if needed separately)
- `fig1_atlas_overview.pdf` — Geneformer SAE feature atlas overview
- `fig2_svd_comparison.pdf` — SVD vs SAE comparison
- `figS2_cross_model.pdf` — Cross-model comparison (Fig. 3 in main)
- `fig3_ontology_heatmap.pdf` — U-shaped annotation profile (Fig. 4 in main)
- `fig4_coactivation.pdf` — Co-activation modules (Fig. 5 in main)
- `fig5_causal_patching.pdf` — Causal patching (Fig. 6 in main)
- `fig6_cross_layer.pdf` — Information highways (Fig. 7 in main)
- `fig7_perturbation.pdf` — Perturbation response (Fig. 8 in main)
- `fig8_multitissue.pdf` — Multi-tissue SAE (Fig. 9 in main)

---

## Step 5: Cover Letter

Use the template below (modify as needed):

```
Dear Editors,

We are pleased to submit our manuscript entitled "Sparse autoencoders reveal organized
biological knowledge but minimal regulatory logic in single-cell foundation models: a
comparative atlas of Geneformer and scGPT" for consideration as a Research article in
Genome Biology.

Single-cell foundation models such as Geneformer and scGPT are increasingly used for
gene network inference, yet whether they encode causal regulatory logic remains unclear.
We address this question by applying sparse autoencoders — a technique from AI
interpretability — to decompose the internal representations of both models, producing
the first comprehensive feature atlases of single-cell foundation models.

Our key findings are:
• Both models exhibit massive superposition: 99.8% of features are invisible to SVD,
  yet carry 98.7% of biological annotations
• Features organize into biologically coherent modules with hierarchical abstraction
  across layers
• Despite this rich biological organization, only 6.2% of transcription factors show
  regulatory-target-specific feature responses, confirming that these models encode
  co-expression rather than causal regulation
• A multi-tissue control experiment establishes the model — not the analysis method —
  as the bottleneck

We release two interactive feature atlases (107,000+ features across 30 layers) as
community resources, along with all analysis code. We believe this work will be of
broad interest to Genome Biology's readership, as it provides the first mechanistic
understanding of what single-cell foundation models actually learn, with direct
implications for how these models should (and should not) be used for gene regulatory
network inference.

This manuscript has not been published elsewhere and is not under consideration at
another journal. A companion paper focusing on attention-based interpretability is
currently available as a preprint (Kendiukhov, 2025).

Thank you for your consideration.

Sincerely,
Ihor Kendiukhov
Department of Computer Science
University of Tübingen
```

---

## Step 6: Data and Code Availability Statement

For the submission form, use:

```
All analysis code, trained SAE models, and feature catalogs are available at
https://github.com/Biodyn-AI/bio-sae. Interactive feature atlases: Geneformer
(https://biodyn-ai.github.io/geneformer-atlas/) and scGPT
(https://biodyn-ai.github.io/scgpt-atlas/). Geneformer V2-316M: HuggingFace
ctheodoris/Geneformer. scGPT whole-human: available from the scGPT authors
(Cui et al., 2024). Replogle CRISPRi data: Replogle et al., 2022. Tabula
Sapiens: The Tabula Sapiens Consortium, 2022. TRRUST: Han et al., 2018.
```

---

## Step 7: Zenodo DOI (Optional but Recommended)

To create a permanent DOI for your code/data:

1. Go to **https://zenodo.org**
2. Log in with GitHub
3. Enable the `Biodyn-AI/bio-sae` repository in Zenodo's GitHub integration
4. Create a new release on GitHub (e.g., `v1.0.0`)
5. Zenodo will automatically mint a DOI
6. Add the DOI to the manuscript's data availability section

---

## Step 8: Review and Submit

1. Review all uploaded files in the submission system
2. Verify the PDF rendering (check figures, tables, references)
3. Confirm all declarations (competing interests, ethics, etc.)
4. Submit

---

## APC Information

Genome Biology is an open access journal published by BioMed Central (Springer Nature).

- **Article Processing Charge (APC):** £3,690 / ~$5,490 USD (as of 2025)
- APC is charged upon acceptance, not submission
- Check for institutional agreements: many universities have Springer Nature Read & Publish deals that cover APCs
- University of Tübingen may have a Springer Nature agreement — check with the library

---

## Review Timeline (Approximate)

| Stage | Expected Duration |
|-------|-------------------|
| Initial editorial assessment | 1–2 weeks |
| Peer review | 4–8 weeks |
| First decision | 6–10 weeks from submission |
| Revision (if requested) | 4–8 weeks allowed |
| Final decision after revision | 2–4 weeks |
| Publication after acceptance | 1–2 weeks (online first) |

---

## Useful Links

- Genome Biology author guidelines: https://genomebiology.biomedcentral.com/submission-guidelines
- Editorial Manager portal: https://www.editorialmanager.com/gbiomed/
- Springer Nature APC page: https://www.springernature.com/gp/open-research/journals-books/journals
- Zenodo: https://zenodo.org
