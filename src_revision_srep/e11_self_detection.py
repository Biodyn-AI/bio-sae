"""E11 — detection of the perturbed gene itself, under one explicit definition per readout.

Comparing how often each readout recovers the perturbed gene is only meaningful if both are
scored on the same targets and each rule is stated. This script fixes both:

Evaluable set
    A target counts only if the perturbed gene could in principle be detected by either
    readout, which requires that it is measured in the expression assay and that it exists
    in the model's gene vocabulary and so can appear in a feature's gene list. Targets
    failing either condition are excluded from both numerator and denominator rather than
    counted as failures by one readout and ignored by the other.

Differential expression, detection
    The perturbed gene's own expression differs between its perturbed cells and the
    non-targeting cells of the same line (Mann-Whitney over cells, Benjamini-Hochberg
    across all measured genes, FDR < 0.05).

SAE feature response, detection
    The perturbed gene appears in the predicted target set, that is among the top-20
    activating genes of at least one feature that responded to the perturbation under the
    criterion used throughout (cell-level FDR < 0.05 and a shift exceeding half a control
    standard deviation).

Those two rules ask different questions, so the script additionally reports a matched
comparison in which both readouts emit a ranked gene list and detection means the perturbed
gene falls within the first K entries, with K set to the same value for both.

Both readouts are evaluated on exactly the cells the main analysis used: the per-cell
feature matrices record the row indices they were built from, and the differential
expression here is recomputed on those same rows.

Outputs: experiments/revision_srep/E11_self_detection/summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.OUT_ROOT / "E11_self_detection"
FDR = 0.05
LINES = ("k562", "rpe1", "jurkat", "hepg2")


def de_for_target(Xp, Xc, gene_names, gene):
    """Mann-Whitney over cells for every measured gene, BH across genes."""
    from scipy.stats import mannwhitneyu
    _, p = mannwhitneyu(Xp, Xc, axis=0, alternative="two-sided")
    q = common.bh_fdr(p)
    diff = Xp.mean(axis=0) - Xc.mean(axis=0)
    pooled = np.sqrt((Xp.var(axis=0) + Xc.var(axis=0)) / 2.0)
    pooled[pooled == 0] = 1e-9
    d = diff / pooled
    idx = np.where(gene_names == gene)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    order = np.argsort(-np.abs(d))                 # rank over all measured genes
    rank = int(np.where(order == i)[0][0]) + 1
    return {"cohens_d": float(d[i]), "fdr": float(q[i]), "rank": rank,
            "detected": bool(q[i] < FDR), "n_genes_ranked": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--tag", default="main/k562sae")
    ap.add_argument("--match-k", type=int, default=100,
                    help="list length for the matched ranked-list comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    _, _, var_genes = common.load_replogle_obs()
    gene_names = np.array([str(g) for g in var_genes])
    measured = set(gene_names.tolist())

    _, _, run_dir = common.load_sae(args.layer)
    catalog = common.load_feature_catalog(run_dir)
    vocabulary = {g for genes in catalog.values() for g in genes}

    per_line, rows_all = {}, []
    for line in LINES:
        src = common.OUT_ROOT / "cell_features" / f"{line}_L{args.layer:02d}_{args.tag}"
        res = common.OUT_ROOT / f"E3_cell_level/{line}_{args.tag}/per_target.json"
        if not src.exists() or not res.exists():
            continue
        sae_rows = {r["gene"]: r for r in json.load(open(res))["per_target"]
                    if r["role"] == "tf"}
        ctrl = np.load(src / "control.npz")
        Xc, _ = common.read_expression(ctrl["cell_idx"])

        rows = []
        for gene, r in sae_rows.items():
            f = src / f"target_{gene}.npz"
            if not f.exists():
                continue
            in_assay = gene in measured
            in_vocab = gene in vocabulary
            row = {"line": line, "gene": gene,
                   "perturbed_gene_measured": in_assay,
                   "perturbed_gene_in_vocabulary": in_vocab,
                   "evaluable": bool(in_assay and in_vocab),
                   "n_cells": int(r["n_cells"])}
            if row["evaluable"]:
                Xp, _ = common.read_expression(np.load(f)["cell_idx"])
                de = de_for_target(Xp, Xc, gene_names, gene)
                row["de"] = de
                predicted = r.get("predicted_genes_cell_level") or []
                sae_rank = (predicted.index(gene) + 1) if gene in predicted else None
                row["sae"] = {
                    "detected": bool(r["self_gene_in_prediction"]),
                    "rank": sae_rank,
                    "n_predicted": len(predicted),
                    "n_responding_features": int(r["n_responding_cell_level"]),
                }
                row["matched_top_k"] = {
                    "k": args.match_k,
                    "de": bool(de and de["rank"] <= args.match_k),
                    "sae": bool(sae_rank is not None and sae_rank <= args.match_k),
                }
            rows.append(row)
            rows_all.append(row)

        ev = [r for r in rows if r["evaluable"]]
        per_line[line] = {
            "n_targets": len(rows),
            "n_evaluable": len(ev),
            "excluded_not_measured": sorted(r["gene"] for r in rows
                                            if not r["perturbed_gene_measured"]),
            "excluded_not_in_vocabulary": sorted(r["gene"] for r in rows
                                                 if r["perturbed_gene_measured"]
                                                 and not r["perturbed_gene_in_vocabulary"]),
            "de_detected": sum(r["de"]["detected"] for r in ev),
            "sae_detected": sum(r["sae"]["detected"] for r in ev),
            "matched_de": sum(r["matched_top_k"]["de"] for r in ev),
            "matched_sae": sum(r["matched_top_k"]["sae"] for r in ev),
            "median_de_rank": (float(np.median([r["de"]["rank"] for r in ev]))
                               if ev else None),
        }
        print(f"{line:7s} evaluable {len(ev):2d}/{len(rows):2d} | "
              f"DE detects {per_line[line]['de_detected']:2d} | "
              f"SAE detects {per_line[line]['sae_detected']:2d} | "
              f"matched top-{args.match_k}: DE {per_line[line]['matched_de']:2d}, "
              f"SAE {per_line[line]['matched_sae']:2d} | "
              f"median DE rank {per_line[line]['median_de_rank']}", flush=True)

    ev_all = [r for r in rows_all if r["evaluable"]]
    pooled = {
        "n_targets": len(rows_all),
        "n_evaluable": len(ev_all),
        "de_detected": sum(r["de"]["detected"] for r in ev_all),
        "sae_detected": sum(r["sae"]["detected"] for r in ev_all),
        "matched_de": sum(r["matched_top_k"]["de"] for r in ev_all),
        "matched_sae": sum(r["matched_top_k"]["sae"] for r in ev_all),
        "median_de_rank": float(np.median([r["de"]["rank"] for r in ev_all])) if ev_all else None,
        "median_de_cohens_d": float(np.median([r["de"]["cohens_d"] for r in ev_all])) if ev_all else None,
    }
    print(f"\npooled: evaluable {pooled['n_evaluable']}/{pooled['n_targets']} | "
          f"DE {pooled['de_detected']} | SAE {pooled['sae_detected']} | "
          f"matched top-{args.match_k}: DE {pooled['matched_de']}, SAE {pooled['matched_sae']}")

    common.write_json(OUT / "summary.json",
                      {"definitions": {
                          "evaluable": "perturbed gene is measured in the assay and present "
                                       "in the model gene vocabulary, so either readout could "
                                       "detect it",
                          "de_detected": "perturbed gene's own expression differs between its "
                                         "perturbed cells and the non-targeting cells of the "
                                         "same line (Mann-Whitney over cells, BH across all "
                                         f"measured genes, FDR < {FDR})",
                          "sae_detected": "perturbed gene appears among the top-20 activating "
                                          "genes of at least one responding feature",
                          "matched_top_k": "perturbed gene falls within the first K entries of "
                                           "each readout's ranked gene list, same K for both"},
                       "per_line": per_line, "pooled": pooled,
                       "per_target": rows_all, "config": vars(args)}, seed=args.seed)


if __name__ == "__main__":
    main()
