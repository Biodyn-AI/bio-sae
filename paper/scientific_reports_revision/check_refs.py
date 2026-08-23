#!/usr/bin/env python3
"""Pre-submission consistency check for the manuscript.

The main text refers to supplementary items by explicit number rather than through a
cross-document label mechanism, because that mechanism resolves correctly only under a
particular compile order and renders every reference as "??" in a single-pass build. This
script is the replacement guarantee: it fails loudly if any referenced supplementary item
does not exist, if any supplementary item is never cited, or if a compiled PDF still
contains an unresolved reference.

Usage:  python3 check_refs.py [--dir .]
Exit code 0 = clean, 1 = problems found.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def supplementary_items(text):
    """Ordered (kind, number, caption-start) for every supplementary table and figure."""
    items = {"table": [], "figure": []}
    for env, kind in (("table", "table"), ("figure", "figure")):
        pattern = re.compile(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", re.S)
        for i, m in enumerate(pattern.finditer(text), start=1):
            body = m.group(1)
            cap = re.search(r"\\caption\{(.{0,80})", body, re.S)
            items[kind].append((i, cap.group(1).replace("\n", " ").strip() if cap else ""))
    return items


def cited_items(text, mapping):
    """Symbolic citations resolved through the generated mapping, plus any literal numbers
    that slipped through (which are exactly the ones that silently drift)."""
    tabs, figs, unknown, literal = set(), set(), [], []
    for key in re.findall(r"\\suptab\{([^}]*)\}", text):
        n = mapping.get(f"tab-{key}")
        (tabs.add(n) if n else unknown.append(f"\\suptab{{{key}}}"))
    for key in re.findall(r"\\supfig\{([^}]*)\}", text):
        n = mapping.get(f"fig-{key}")
        (figs.add(n) if n else unknown.append(f"\\supfig{{{key}}}"))
    for n in re.findall(r"\\supref\{(fig-[^}]*)\}", text):
        m = mapping.get(n)
        if m:
            figs.add(m)
        else:
            unknown.append(f"\\supref{{{n}}}")
    literal += re.findall(r"Supplementary\s+Table~?\s*S(\d+)", text)
    literal += re.findall(r"Supplementary\s+Fig(?:\.|ure)?~?\s*S(\d+)", text)
    return tabs, figs, unknown, literal


def load_mapping(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return {k: int(v) for k, v in
            re.findall(r"suprefnum@([A-Za-z0-9_-]+)\\endcsname\{(\d+)\}", text)}


def check_pdf(path):
    if not path.exists():
        return None
    try:
        out = subprocess.run(["pdftotext", str(path), "-"], capture_output=True,
                             text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.count("??")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    args = ap.parse_args()
    d = Path(args.dir).resolve()

    main_tex = read(d / "main.tex")
    supp_tex = read(d / "supplementary.tex")
    if not main_tex or not supp_tex:
        print(f"ERROR: main.tex or supplementary.tex missing in {d}")
        return 1

    problems, notes = [], []

    items = supplementary_items(supp_tex)
    n_tables, n_figures = len(items["table"]), len(items["figure"])
    mapping = load_mapping(d / "sections" / "supp_numbers.tex")
    cited_tabs, cited_figs, unknown_keys, literal_nums = cited_items(main_tex, mapping)
    if not mapping:
        problems.append("sections/supp_numbers.tex missing - run assemble_supp.py")
    for k in unknown_keys:
        problems.append(f"main.tex cites {k}, which has no supplementary item")
    if literal_nums:
        problems.append("main.tex uses hard-coded supplementary numbers "
                        f"({', '.join('S' + n for n in sorted(set(literal_nums)))}); "
                        "these drift silently when the supplement is reordered - "
                        "use \\suptab{key} / \\supfig{key} instead")

    print(f"supplementary: {n_tables} tables, {n_figures} figures")
    print(f"main text cites: {len(cited_tabs)} tables, {len(cited_figs)} figures")

    for n in sorted(cited_tabs):
        if n > n_tables:
            problems.append(f"main.tex cites Supplementary Table S{n}, "
                            f"but only {n_tables} tables exist")
    for n in sorted(cited_figs):
        if n > n_figures:
            problems.append(f"main.tex cites Supplementary Fig. S{n}, "
                            f"but only {n_figures} figures exist")

    uncited_t = [n for n, _ in items["table"] if n not in cited_tabs]
    uncited_f = [n for n, _ in items["figure"] if n not in cited_figs]
    if uncited_t:
        problems.append(f"supplementary tables never cited in the main text: "
                        f"{', '.join('S%d' % n for n in uncited_t)}")
    if uncited_f:
        problems.append(f"supplementary figures never cited in the main text: "
                        f"{', '.join('S%d' % n for n in uncited_f)}")

    # legacy mechanisms that reintroduce the "??" failure mode
    for bad, why in ((r"\\usepackage\{xr\}", "xr package"),
                     (r"\\externaldocument", "externaldocument"),
                     (r"Additional file", "BMC-style 'Additional file' reference")):
        for name, text in (("main.tex", main_tex), ("supplementary.tex", supp_tex)):
            if re.search(bad, text):
                problems.append(f"{name} still contains {why}")

    # draft-like artefacts
    artefacts = [
        r"earlier version", r"original preprint", r"in this revision", r"the reviewers?",
        r"we (?:now|have since)\b", r"previous(?:ly)? reported", r"as suggested",
        r"follow-?up experiment", r"natural next experiment", r"compute budget",
        r"we have not yet", r"preliminary\s*/\s*exploratory", r"PLACEHOLDER",
        r"RESPONSE_[A-Z0-9]+", r"TODO", r"XXX", r"to be inserted", r"NEW_ANALYSES", r"DISCUSSION-[A-Z-]+", r"RESULTS-[A-Z-]+",
    ]
    for name, text in (("main.tex", main_tex), ("supplementary.tex", supp_tex)):
        for pat in artefacts:
            for m in re.finditer(pat, text, re.I):
                line = text[:m.start()].count("\n") + 1
                snippet = text[max(0, m.start() - 45):m.start() + 55].replace("\n", " ")
                problems.append(f"{name}:{line} draft artefact "
                                f"[{m.group(0)}] ... {snippet.strip()} ...")

    for pdf in ("main.pdf", "supplementary.pdf"):
        n = check_pdf(d / pdf)
        if n is None:
            notes.append(f"{pdf} not compiled (or pdftotext unavailable) - not checked")
        elif n:
            problems.append(f"{pdf} contains {n} unresolved '??' reference(s)")
        else:
            print(f"{pdf}: no unresolved references")
        # the graphic-guard fallback only matters if it actually rendered
        try:
            text = subprocess.run(["pdftotext", str(d / pdf), "-"], capture_output=True,
                                  text=True, timeout=120).stdout
            if "FIGURE PENDING" in text:
                missing = text.count("FIGURE PENDING")
                problems.append(f"{pdf} renders {missing} placeholder box(es) for missing figures")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    for n in notes:
        print(f"note: {n}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
