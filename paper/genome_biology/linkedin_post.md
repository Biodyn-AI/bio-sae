# LinkedIn Post Draft

---

## Post Text

What do single-cell foundation models actually know about biology? To find out, I applied sparse autoencoders — the same interpretability technique Anthropic and OpenAI use to peer inside large language models — to two leading single-cell foundation models: Geneformer and scGPT.

The result: two open interactive feature atlases covering 107,000+ biological features across 30 transformer layers.

**What we found:**

These models use massive superposition to pack biological knowledge into their hidden dimensions — 82,525 features crammed into just 1,152 dimensions in Geneformer (a 70x compression ratio). 99.8% of these features are completely invisible to standard methods like PCA or SVD. You literally cannot see them without sparse autoencoders.

The features that SAEs uncover are biologically rich: over half map to known pathways (GO, KEGG, Reactome), protein interaction networks (STRING), or transcription factor targets (TRRUST). They self-organize into co-activation modules, form cross-layer information highways, and show causal specificity when patched — zeroing a single feature shifts model predictions for exactly the genes you'd expect.

**But here's the critical finding:** when tested against genome-scale CRISPRi perturbation data, only 6.2% of transcription factors show regulatory-target-specific feature responses. The models have learned rich co-expression structure and pathway biology — but not causal gene regulation. A multi-tissue control experiment confirmed this is a model-level limitation, not a methodological one.

**Why this matters:**

1. It tells us exactly what current foundation models know and don't know — critical for anyone building downstream applications
2. It suggests that learning causal regulation will require perturbation-aware training objectives, not just more scRNA-seq data
3. The atlases themselves are a resource: search any gene, explore any pathway, trace information flow across layers

**Explore the atlases (no installation needed):**

Geneformer Feature Atlas (82,525 features, 18 layers):
https://biodyn-ai.github.io/geneformer-atlas/

scGPT Feature Atlas (24,527 features, 12 layers):
https://biodyn-ai.github.io/scgpt-atlas/

Code & data: https://github.com/Biodyn-AI/bio-sae

Paper: [arXiv link once live]

I'd love to hear thoughts from anyone working on single-cell models, biological foundation models, or mechanistic interpretability. What other models should we open up next?

---

## People to Tag

### Geneformer authors (Nature, 2023)
- Christina V. Theodoris — https://www.linkedin.com/in/christina-theodoris-a77093245/ (PI, Gladstone Institutes / UCSF)
- X. Shirley Liu — (Dana-Farber / Harvard; search LinkedIn)
- Patrick T. Ellinor — (Broad Institute; search LinkedIn)
- Anant Chopra — (search LinkedIn)
- Mark D. Chaffin — (search LinkedIn)
- Zeina R. Al Sayed — (search LinkedIn)
- Matthew C. Hill — (search LinkedIn)
- Helene Mantineo — (search LinkedIn)
- Elizabeth M. Brydon — (search LinkedIn)
- Zexian Zeng — (search LinkedIn)
- Ling Xiao — (search LinkedIn)

### scGPT authors (Nature Methods, 2024)
- Bo Wang — https://www.linkedin.com/in/bo-wang-a6065240/ (U of T / Xaira Therapeutics, senior author)
- Haotian Cui — https://www.linkedin.com/in/haotiancui/ (U of T / Xaira Therapeutics, first author)
- Chloe Wang — (search LinkedIn)
- Hassaan Maan — (search LinkedIn)
- Kuan Pang — (search LinkedIn)
- Fengning Luo — (search LinkedIn)
- Nan Duan — (Microsoft Research Asia; search LinkedIn)

### SAE / Mechanistic Interpretability pioneers (suggested)
- Lee Sharkey — https://www.linkedin.com/in/lee-sharkey-62a0a19b/ (Goodfire; co-author of "Sparse Autoencoders Find Highly Interpretable Features in Language Models")
- Trenton Bricken — (Anthropic; lead author of "Towards Monosemanticity"; search LinkedIn)
- Hoagy Cunningham — (first author of "SAEs Find Highly Interpretable Features"; search LinkedIn)
- Adly Templeton — (Anthropic; lead author of "Scaling Monosemanticity"; search LinkedIn)
- Leo Gao — (OpenAI; lead author of "Scaling and Evaluating Sparse Autoencoders"; search LinkedIn)

### SAEs for protein/biological models (suggested — most directly related)
- Bonnie Berger — (MIT; senior author of "Sparse autoencoders uncover biologically interpretable features in protein language model representations", PNAS 2025)
- Onkar Gujral — (MIT; first author of same PNAS paper; search LinkedIn)
- InterPLM authors — (Nature Methods 2025; SAEs on ESM-2 protein language model; search LinkedIn for authors)

### Other prominent single-cell FM researchers (suggested)
- Jure Leskovec — (Stanford; UCE / Universal Cell Embeddings; search LinkedIn)
- Fabian Theis — (Helmholtz Munich; scVI / scArches ecosystem, prominent scFM commentator; search LinkedIn)
- Aviv Regev — (Genentech; former Broad co-director, key figure in single-cell genomics; search LinkedIn)

---

## Notes for posting

- LinkedIn has a ~3,000 character limit for posts before the "see more" fold. The first ~2-3 sentences should hook the reader. The post above is ~2,800 characters before links, which is close to the fold. Consider trimming if needed.
- Tag people using @ mentions when actually posting (LinkedIn will auto-suggest from the names above).
- Add a relevant image — consider a screenshot of one of the atlases showing an interesting feature, or the paper's overview figure (Fig. 1).
- Suggested hashtags: #MachineLearning #Bioinformatics #SingleCell #FoundationModels #MechanisticInterpretability #SparseAutoencoders #Genomics #AIforScience #ComputationalBiology
