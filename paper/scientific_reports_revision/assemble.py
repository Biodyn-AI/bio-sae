#!/usr/bin/env python3
"""Assemble main.tex from its section files.

The manuscript is kept as sections so that each analysis owns its prose, and the tables
that report new analyses are generated from result files rather than transcribed. This
script stitches them together in reading order and pulls the retained main-text tables out
of the previous manuscript by label, so their bodies are never re-keyed by hand.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = HERE / "sections"
SOURCE = HERE / "main_source.tex"

# Main-text tables carried over unchanged, in the order they are first referenced.
CARRIED_TABLES = ["tab:full18", "tab:scgpt_full12", "tab:examples"]


def extract_env(text, label, env="table"):
    """Return the full environment block that defines the given label."""
    pattern = re.compile(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", re.S)
    for m in pattern.finditer(text):
        if f"\\label{{{label}}}" in m.group(0):
            return m.group(0)
    return None


# Carried tables were written for a different journal's supplement convention and for the
# earlier framing of the SVD comparison; these substitutions bring them into line.
FIXUPS = [
    ("Additional file~1: Table~S1", "\\suptab{perontology}"),
    ("Additional file~1: Table~", "Supplementary Table~"),
    ("Additional file~1: ", "Supplementary "),
    ("Layer & VarExpl & Alive & Dead & SVD & Novel & MeanCos & Ann\\% & Enrichments",
     "Layer & VarExpl & Alive & Dead & Aligned & Non-aligned & MeanCos & Ann\\% & Enrichments"),
    ("Dead = $4{,}608 - \\text{Alive}$.",
     "Dead = $4{,}608 - \\text{Alive}$. Aligned = features with $|\\cos| > 0.5$ against any of "
     "the leading 50 principal axes; see \\suptab{svd_capacity} for the capacity-matched "
     "comparison."),
]


def apply_fixups(text):
    for old, new in FIXUPS:
        text = text.replace(old, new)
    return text


def read(name):
    p = SECTIONS / name
    if not p.exists():
        print(f"  missing section: {name}")
        return ""
    return p.read_text(encoding="utf-8")


def main():
    source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
    tables = {}
    for label in CARRIED_TABLES:
        body = extract_env(source, label)
        if body is None:
            print(f"  WARNING: could not find {label} in {SOURCE.name}")
        tables[label] = apply_fixups(body or "")

    preamble = read("preamble.tex")
    abstract = read("abstract.tex")
    intro = read("introduction.tex")
    results = "\n\n".join(x for x in [
        read("results_stable.tex"),
        read("results_svd_concepts.tex"),
        read("results_causal.tex"),
        read("results_robustness.tex"),
        read("results_regulatory.tex"),
        read("figures.tex"),
    ] if x)
    discussion = read("discussion.tex")
    methods = read("methods.tex")
    decl = read("declarations.tex")

    # Insert carried tables after the paragraph that first cites them.
    for label in CARRIED_TABLES:
        if not tables[label]:
            continue
        anchor = f"Table~\\ref{{{label}}}"
        idx = results.find(anchor)
        if idx == -1:
            results += "\n\n" + tables[label]
            continue
        end = results.find("\n\n", idx)
        end = len(results) if end == -1 else end
        results = results[:end] + "\n\n" + tables[label] + results[end:]

    doc = "\n".join([
        preamble.rstrip(),
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        abstract.strip(),
        "\\end{abstract}",
        "",
        "\\section{Introduction}",
        intro.strip(),
        "",
        "\\section{Results}",
        results.strip(),
        "",
        "\\section{Discussion}",
        discussion.strip(),
        "",
        "\\section{Methods}",
        methods.strip(),
        "",
        decl.strip(),
        "",
        "\\bibliography{references}",
        "",
        "\\end{document}",
        "",
    ])
    (HERE / "main.tex").write_text(doc, encoding="utf-8")
    print(f"  wrote {HERE / 'main.tex'} ({doc.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
