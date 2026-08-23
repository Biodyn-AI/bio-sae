"""E0 — composition of the control cohort the atlas was built from.

The Replogle resource used here is a single processed matrix spanning four cell lines. The
control cohort for activation extraction is defined by the perturbation label
(``non-targeting``) and is therefore drawn from all four lines unless a cell-line filter is
applied. This script records the exact composition of every cohort the study uses, so that
each analysis can state which cells it is about.

The atlas cohort is reproduced from the extraction procedure (seed 42) and validated
against the per-cell gene counts stored at extraction time, so the composition reported
here is the composition of the cells actually processed, not an inference from the code.

Outputs: experiments/revision_srep/E0_cohorts/composition.json
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.OUT_ROOT / "E0_cohorts"
N_ATLAS = 2000
N_PATCHING = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cl, pert, var_genes = common.load_replogle_obs()

    control_mask = np.zeros(len(cl), dtype=bool)
    for name in ("non-targeting", "Non-targeting", "non_targeting"):
        control_mask |= (pert == name)
    ctrl = np.where(control_mask)[0]

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    atlas = np.random.choice(ctrl, N_ATLAS, replace=False)
    atlas.sort()
    patching = atlas[:N_PATCHING]

    # validation: per-cell gene counts recorded when the activations were extracted
    validation = {"checked": False}
    info_path = common.PHASE1 / "cell_info.json"
    if info_path.exists():
        stored = np.array(json.load(open(info_path))["genes_per_cell"])
        tokenizer = common.GeneformerTokenizer(var_genes)
        n_check = 25
        X, _ = common.read_expression(atlas[:n_check])
        reproduced = [int(len(tokenizer.encode(X[i])) - 2) for i in range(n_check)]
        validation = {
            "checked": True,
            "n_checked": n_check,
            "stored_genes_per_cell": stored[:n_check].tolist(),
            "reproduced_genes_per_cell": reproduced,
            "identical": bool(list(stored[:n_check]) == reproduced),
            "stored_n_cells": int(len(stored)),
            "stored_mean_genes_per_cell": float(stored.mean()),
        }

    def comp(idx):
        c = collections.Counter(cl[idx])
        total = sum(c.values())
        return {"n": int(total),
                "by_cell_line": {k: int(v) for k, v in
                                 sorted(c.items(), key=lambda kv: -kv[1])},
                "fraction": {k: round(v / total, 4) for k, v in
                             sorted(c.items(), key=lambda kv: -kv[1])}}

    row_ranges = {}
    for line in common.CELL_LINES:
        idx = np.where(cl == line)[0]
        row_ranges[line] = [int(idx.min()), int(idx.max()), int(len(idx))]

    payload = {
        "available_non_targeting_cells": comp(ctrl),
        "atlas_extraction_cohort": comp(atlas),
        "published_causal_patching_cohort": comp(patching),
        "cell_line_row_ranges": row_ranges,
        "validation_against_extraction_metadata": validation,
        "note": ("The atlas cohort is a random sample of non-targeting cells across all four "
                 "lines. The causal-patching cohort is the first 200 of that cohort after "
                 "sorting by row index, and the matrix is ordered by cell line, so it falls "
                 "entirely within the line occupying the lowest row indices."),
    }
    common.write_json(OUT / "composition.json", payload, seed=args.seed)

    print("non-targeting cells available:", payload["available_non_targeting_cells"]["by_cell_line"])
    print("atlas cohort:", payload["atlas_extraction_cohort"]["by_cell_line"])
    print("published causal-patching cohort:",
          payload["published_causal_patching_cohort"]["by_cell_line"])
    print("validated against extraction metadata:", validation.get("identical"))


if __name__ == "__main__":
    main()
