"""E1b — merge every method's predictions into one scored benchmark.

The benchmark arms are produced by separate runs (the tree-ensemble methods take far longer
than the rest, and the SAE arm depends on the per-cell extraction), so this step collects
the prediction files, adds the SAE arm from the cell-level analysis, and re-scores all
methods together under one framework. Scoring is per method, so combining after the fact is
equivalent to having run them in a single process.

Outputs: experiments/revision_srep/E1_ceiling/scores_<cell_line>_merged.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from e1_recoverability_ceiling import score_method  # noqa: E402

OUT = common.OUT_ROOT / "E1_ceiling"


def sae_predictions(cell_line, tag, references, measured, top_n):
    """Predicted target sets from the cell-level SAE responses, built the same way as for
    every other method: a ranked gene list truncated to the common length."""
    p = common.OUT_ROOT / f"E3_cell_level/{cell_line}_{tag}/per_target.json"
    if not p.exists():
        return None, None
    rows = json.load(open(p))["per_target"]
    preds, detail = {}, {}
    for r in rows:
        if r["role"] != "tf":
            continue
        genes = r.get("predicted_genes_cell_level")
        if genes is None:
            continue
        preds[r["gene"]] = genes[:top_n]
        detail[r["gene"]] = {
            "n_responding_cell_level": r["n_responding_cell_level"],
            "n_predicted_genes": r["n_predicted_genes_cell_level"],
            "self_gene_in_prediction": r["self_gene_in_prediction"],
        }
    return preds, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", default="k562")
    ap.add_argument("--sae-tag", default="main/k562sae")
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, _, var_genes = common.load_replogle_obs()
    measured = {str(g) for g in var_genes}
    trrust = common.load_trrust()
    dorothea_abc = common.load_dorothea(("A", "B", "C"))
    dorothea_all = common.load_dorothea(None)
    union_ref = {}
    for ref in (trrust, dorothea_all):
        for tf, tg in ref.items():
            union_ref.setdefault(tf, set()).update(tg)
    references = {"TRRUST": trrust, "DoRothEA_ABC": dorothea_abc,
                  "DoRothEA_all": dorothea_all, "union": union_ref}

    predictions, panel, evaluability, extras = {}, None, None, {}
    for name in (f"predictions_{args.cell_line}_part1.json",
                 f"predictions_{args.cell_line}.json"):
        p = OUT / name
        if not p.exists():
            continue
        d = json.load(open(p))
        predictions.update(d["predictions"])
        panel = panel or d["panel"]
        print(f"  {name}: {list(d['predictions'])}")
    for name in (f"scores_{args.cell_line}_part1.json", f"scores_{args.cell_line}.json"):
        p = OUT / name
        if p.exists():
            d = json.load(open(p))
            evaluability = evaluability or d.get("evaluability")
            if "de_detail" in d:
                extras["de_detail"] = d["de_detail"]

    sae_preds, sae_detail = sae_predictions(args.cell_line, args.sae_tag, references,
                                            measured, args.top_n)
    if sae_preds:
        predictions["sae_features"] = sae_preds
        extras["sae_detail"] = sae_detail
        print(f"  SAE arm: {len(sae_preds)} transcription factors")
    else:
        print("  SAE arm: cell-level results not available yet")

    if panel is None:
        print("no prediction files found")
        return 1

    scores = {}
    for ref_name, ref in references.items():
        for universe_mode in ("measured", "20000"):
            scores[f"{ref_name}|{universe_mode}"] = {
                m: score_method(p, panel, ref, measured, universe_mode, args.min_targets)
                for m, p in predictions.items()}

    print("\n=== fraction of TFs with FDR<0.05 target enrichment (measured universe) ===")
    for ref_name in references:
        key = f"{ref_name}|measured"
        print(f"  [{ref_name}]")
        for m, s in sorted(scores[key].items(), key=lambda kv: -kv[1]["frac_significant"]):
            print(f"    {m:18s} {s['n_significant']:3d}/{s['n_tfs']:3d} "
                  f"= {100 * s['frac_significant']:5.1f}%")

    common.write_json(OUT / f"scores_{args.cell_line}_merged.json",
                      {"scores": scores, "panel": panel, "panel_size": len(panel),
                       "evaluability": evaluability, "methods": sorted(predictions),
                       "top_n": args.top_n, **extras}, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
