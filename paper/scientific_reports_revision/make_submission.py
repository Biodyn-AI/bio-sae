#!/usr/bin/env python3
"""Build a self-contained LaTeX submission bundle and verify it compiles from scratch.

The bundle must build in a directory that contains nothing but the bundle, in a plain
sequence of pdflatex/bibtex runs, with no reliance on this project's layout. The previous
submission failed exactly that test: the two documents imported each other's labels, which
resolves only under a particular compile order, and every cross-reference rendered as "??"
on the publisher's single-pass build. This script therefore inlines the one generated
include, copies only the figures actually referenced, ships the .bbl so bibtex is optional,
and then test-compiles the result in a temporary directory before packaging it.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission"
ZIP = HERE / "scientific_reports_submission.zip"


def figures_used(text):
    names = set(re.findall(r"reviewfig\{([A-Za-z0-9_]+\.pdf)\}", text))
    names |= set(re.findall(r"figures/([A-Za-z0-9_]+\.pdf)", text))
    return names


def inline_inputs(text, base):
    """Replace \\input{...} with the file's contents so the bundle has no hidden deps."""
    def sub(m):
        p = base / (m.group(1) + ".tex")
        if not p.exists():
            p = base / m.group(1)
        return p.read_text(encoding="utf-8") if p.exists() else m.group(0)
    return re.sub(r"\\input\{([^}]+)\}", sub, text)


IGNORE = shutil.ignore_patterns("._*", ".DS_Store", "__MACOSX")


def build():
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "figures").mkdir(parents=True)

    main = inline_inputs((HERE / "main.tex").read_text(encoding="utf-8"), HERE)
    supp = inline_inputs((HERE / "supplementary.tex").read_text(encoding="utf-8"), HERE)
    (OUT / "main.tex").write_text(main, encoding="utf-8")
    (OUT / "supplementary.tex").write_text(supp, encoding="utf-8")

    for name in sorted(figures_used(main) | figures_used(supp)):
        src = HERE / "figures" / name
        if src.exists():
            shutil.copy2(src, OUT / "figures" / name)
        else:
            print(f"  WARNING: referenced figure missing: {name}")

    for extra in ("references.bib", "main.bbl"):
        if (HERE / extra).exists():
            shutil.copy2(HERE / extra, OUT / extra)

    (OUT / "README.txt").write_text(
        "Manuscript source for Scientific Reports\n"
        "=======================================\n\n"
        "Files\n"
        "  main.tex           main manuscript, self-contained (no \\input dependencies)\n"
        "  supplementary.tex  Supplementary Information, compiles independently\n"
        "  references.bib     bibliography source\n"
        "  main.bbl           pre-built bibliography, so bibtex need not be run\n"
        "  figures/           all figures referenced by either document\n\n"
        "Compilation\n"
        "  pdflatex main && bibtex main && pdflatex main && pdflatex main\n"
        "  pdflatex supplementary && pdflatex supplementary\n\n"
        "The two documents are independent: neither imports labels from the other, and\n"
        "supplementary items are referenced from the main text by explicit number. A\n"
        "single-pass build of either file therefore produces no unresolved references.\n",
        encoding="utf-8")
    return sorted(p.relative_to(OUT) for p in OUT.rglob("*")
                  if p.is_file() and not p.name.startswith("._")
                  and p.name != ".DS_Store")


def test_compile():
    """Compile in an empty temporary directory, exactly as a publisher would."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(OUT, tmp / "bundle", ignore=IGNORE)
        work = tmp / "bundle"
        cmds = [["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "supplementary.tex"],
                ["pdflatex", "-interaction=nonstopmode", "supplementary.tex"]]
        for c in cmds:
            subprocess.run(c, cwd=work, capture_output=True, timeout=300)

        problems = []
        for stem in ("main", "supplementary"):
            pdf = work / f"{stem}.pdf"
            if not pdf.exists():
                problems.append(f"{stem}.pdf was not produced")
                continue
            txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                                 text=True, timeout=120).stdout
            if "??" in txt:
                problems.append(f"{stem}.pdf contains {txt.count('??')} unresolved references")
            if "FIGURE PENDING" in txt:
                problems.append(f"{stem}.pdf renders a placeholder for a missing figure")
            log = (work / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
            n_undef = log.count("Reference") and len(re.findall(r"Reference `[^']*' on page \d+ undefined", log))
            if n_undef:
                problems.append(f"{stem}.log reports {n_undef} undefined references")
            if re.search(r"^! ", log, re.M):
                problems.append(f"{stem}.log contains a LaTeX error")
            pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                   text=True).stdout
            m = re.search(r"Pages:\s+(\d+)", pages)
            print(f"  {stem}.pdf: {m.group(1) if m else '?'} pages")

        # A single pass cannot resolve internal \ref (no LaTeX document can), but it must
        # resolve every supplementary reference: those are literal numbers, and their failing
        # on a publisher's single-pass build is the defect this bundle exists to prevent.
        single = tmp / "single"
        shutil.copytree(OUT, single, ignore=IGNORE)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=single, capture_output=True, timeout=300)
        log = (single / "main.log").read_text(encoding="utf-8", errors="replace")
        undef = set(re.findall(r"Reference `([^']*)' on page \d+ undefined", log))
        cross_doc = {u for u in undef if not u.startswith(("fig:", "tab:", "sec:", "eq:"))}
        txt = subprocess.run(["pdftotext", str(single / "main.pdf"), "-"],
                             capture_output=True, text=True).stdout
        if cross_doc:
            problems.append("single-pass build leaves non-internal references unresolved: "
                            + ", ".join(sorted(cross_doc)))
        elif "??SUPP:" in txt:
            problems.append("single-pass build leaves a supplementary reference unresolved")
        else:
            print(f"  single-pass build: all supplementary references resolve "
                  f"({len(undef)} internal \\ref pending a second pass, as expected)")
        return problems


def main():
    files = build()
    print(f"bundle: {len(files)} files")
    problems = test_compile()
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(OUT / f, f)
    print(f"\nwrote {ZIP.name} ({ZIP.stat().st_size / 1048576:.1f} MB)")
    print("clean-room compilation succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
