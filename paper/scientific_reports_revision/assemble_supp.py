#!/usr/bin/env python3
"""Assemble supplementary.tex.

Tables reporting new analyses are generated from result files by build_supp_tables.py and
included here as fragments. Tables carried over from the previous manuscript are lifted out
of it by label so their bodies are never re-keyed, with caption text normalised to this
journal's supplement convention.

Supplementary items are numbered by position in this file, and the main text cites them by
that number; check_refs.py verifies the correspondence.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPP = HERE / "sections" / "supp"
SOURCE = HERE / "supplementary_source.tex"

# (kind, key) in final order. "gen" = generated fragment file, "old" = label in the source.
ORDER = [
    ("gen", "supp_cohorts.tex"),
    ("old", "tab:perontology"),
    ("old", "tab:persistence"),
    ("old", "tab:modules_full"),
    ("old", "tab:highways"),
    ("old", "tab:highways_null"),
    ("old", "tab:highways_consecutive"),
    ("old", "tab:scgpt_highways"),
    ("old", "tab:cascades"),
    ("old", "tab:pmi_memory"),
    ("old", "tab:leiden_sweep"),
    ("old", "tab:hyperparam"),
    ("old", "tab:svd_sweep"),
    ("gen", "supp_svd_capacity.tex"),
    ("gen", "supp_concepts.tex"),
    ("gen", "supp_seed_stability.tex"),
    ("gen", "supp_crossmodel.tex"),
    ("gen", "supp_evaluability.tex"),
    ("gen", "supp_ceiling.tex"),
    ("gen", "supp_cell_level.tex"),
    ("gen", "supp_sweep.tex"),
    ("gen", "supp_crossline.tex"),
    ("gen", "supp_causal_arms.tex"),
    ("old", "tab:unannotated_new"),
    ("old", "tab:novel"),
    ("old", "tab:batch"),
]

FIGURES = [
    ("gb_figS3_scgpt_highways.pdf",
     r"\textbf{scGPT cross-layer connectivity.} Upstream connectivity stays high while "
     r"downstream connectivity falls with depth, a pattern Geneformer does not show.",
     "fig:scgpt_highways"),
    ("gb_figS4_umap_overview.pdf",
     r"\textbf{Force-directed layout of the co-activation graph.} Each module forms a "
     r"spatially coherent community. This is a layout of the co-activation graph, not a "
     r"UMAP or t-SNE projection of the decoder directions.", "fig:umap_overview"),
    ("gb_figS5_l11_detail.pdf",
     r"\textbf{Layer-11 co-activation graph in detail.} Annotated features distribute "
     r"across all communities while sparsely annotated features concentrate centrally.",
     "fig:l11_detail"),
    ("gb_figS6_annotation_projections.pdf",
     r"\textbf{Annotation-based projections.} TF-IDF weighted ontology vectors projected "
     r"with UMAP and t-SNE, an independent check that module structure reflects biological "
     r"similarity.", "fig:annotation_projections"),
]

FIXUPS = [
    ("Additional file~1: Table~", "Supplementary Table~"),
    ("Additional file~1: ", "Supplementary "),
    ("of the main text", "of the main text"),
    # revision-era caption phrasing
    ("original-preprint analysis", "gene-set clustering analysis"),
    ("As discussed in \\S2.11 of the main text, the co-activate/isolated column is now "
     "regarded as weak evidence", "The co-activate/isolated column is weak evidence"),
    ("in this revision", ""),
    ("as the reviewers suggested", ""),
    # cross-document references cannot resolve inside a separate document; state them in words
    ("Table~\\ref{tab:full18}", "Table~1 of the main text"),
    ("\\S\\ref{sec:ushape}", "the section on layer-wise annotation in the main text"),
    ("\\S\\ref{sec:hyperparam}", "the Methods of the main text"),
    ("\\ref{sec:hyperparam}", "the Methods of the main text"),
    ("\\S2.11 of the main text", "the section on unannotated features in the main text"),
    ("\\S2.7 of the main text", "the section on causal feature ablation in the main text"),
    # revision-era caption phrasing in carried tables
    ("(the original preprint quoted $\\tau{=}0.7$ in the text but the pipeline used "
     "$\\tau{=}0.5$)", ""),
    ("the original preprint quoted $\\tau{=}0.7$ in the text but t", "t"),
    ("(original preprint)", "(gene-set clustering)"),
    ("original preprint", "gene-set clustering analysis"),
    ("are beyond this revision's compute budget", "are not reported here"),
    ("this revision's compute budget", "the reported scope"),
]

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{float}

\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}

\title{\textbf{Supplementary Information}\\[6pt]
\large Sparse autoencoders reveal organized biological knowledge but minimal regulatory
logic in single-cell foundation models}
\author{Ihor Kendiukhov}
\date{}
"""


def extract_env(text, label, env="table"):
    pattern = re.compile(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", re.S)
    for m in pattern.finditer(text):
        if f"\\label{{{label}}}" in m.group(0):
            return m.group(0)
    return None


def fixup(text):
    for old, new in FIXUPS:
        text = text.replace(old, new)
    return text


def main():
    source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
    blocks, missing, number = [], [], 0
    numbering = []

    for kind, key in ORDER:
        if kind == "gen":
            p = SUPP / key
            if not p.exists():
                missing.append(key)
                continue
            body = p.read_text(encoding="utf-8")
        else:
            body = extract_env(source, key)
            if body is None:
                missing.append(key)
                continue
            body = fixup(body)
        number += 1
        key = key[len("supp_"):-len(".tex")] if kind == "gen" else key.split(":", 1)[-1]
        numbering.append((f"tab-{key}", number))
        blocks.append(body.rstrip())

    fig_blocks = []
    for fig_i, (fname, caption, label) in enumerate(FIGURES, start=1):
        numbering.append((f"fig-{label.split(':', 1)[-1]}", fig_i))
    for fname, caption, label in FIGURES:
        fig_blocks.append("\n".join([
            r"\begin{figure}[htbp]\centering",
            r"\IfFileExists{figures/%s}{\includegraphics[width=\textwidth]{figures/%s}}"
            % (fname, fname),
            r"{\fbox{\parbox[c][3cm][c]{0.9\textwidth}{\centering FIGURE PENDING: "
            r"\texttt{\detokenize{%s}}}}}" % fname,
            r"\caption{%s}" % caption,
            r"\label{%s}" % label,
            r"\end{figure}",
        ]))

    # Emit the key -> number mapping so the main text can cite items symbolically and can
    # never drift when the supplement is reordered or an item is added.
    macro_lines = [r"% generated by assemble_supp.py - do not edit",
                   r"\makeatletter",
                   r"\newcommand{\supref}[1]{%",
                   r"  \expandafter\ifx\csname suprefnum@#1\endcsname\relax",
                   r"    \textbf{??SUPP:#1??}%",
                   r"  \else\csname suprefnum@#1\endcsname\fi}",
                   r"\newcommand{\suptab}[1]{Supplementary Table~S\supref{tab-#1}}",
                   r"\newcommand{\supfig}[1]{Supplementary Fig.~S\supref{fig-#1}}"]
    for key, num in numbering:
        macro_lines.append(r"\expandafter\def\csname suprefnum@%s\endcsname{%d}" % (key, num))
    macro_lines.append(r"\makeatother")
    (HERE / "sections" / "supp_numbers.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8")
    print(f"  wrote sections/supp_numbers.tex with {len(numbering)} entries")

    doc = "\n\n".join([
        PREAMBLE,
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Supplementary Tables}",
        "\n\n".join(blocks),
        r"\clearpage",
        r"\section*{Supplementary Figures}",
        "\n\n".join(fig_blocks),
        r"\end{document}",
        "",
    ])
    (HERE / "supplementary.tex").write_text(doc, encoding="utf-8")
    print(f"  wrote supplementary.tex with {number} tables and {len(fig_blocks)} figures")
    if missing:
        print(f"  pending ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
