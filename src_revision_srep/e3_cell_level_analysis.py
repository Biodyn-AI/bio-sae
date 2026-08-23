"""E2/E3 analysis — cell-level perturbation response, power, and regulatory specificity.

Reads the cached per-cell feature matrices and produces, for one cell line:

  1. cell-level responding-feature counts (Mann-Whitney over cells, BH across features)
     next to the position-pooled statistic computed on exactly the same cells, so the
     effect of treating gene positions as independent replicates is quantified rather
     than argued about;
  2. the intraclass correlation of feature activation within cells, which is what sets
     the effective sample size of a position-pooled test;
  3. TF target specificity under the same hypergeometric framework used for every other
     method in the ceiling benchmark, with a label-permutation null;
  4. a sample-size sweep with bootstrap confidence intervals, plus a positive control
     (recovery of the knocked-down gene itself) that separates "no power" from
     "power but no target specificity".

Outputs: experiments/revision_srep/E3_cell_level/<cell_line>_<tag>/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EFFECT_THRESHOLD = 0.5
FDR_THRESHOLD = 0.05


# ---------------------------------------------------------------------------

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    pooled[pooled <= 0] = np.inf
    return (a.mean(axis=0) - b.mean(axis=0)) / pooled


def pooled_scale_effect(pert_cell_mean, ctrl):
    """Effect on the scale used by the position-pooled analysis: the shift in mean feature
    activation expressed in control position-level standard deviations. Holding this
    definition fixed means the cell-level and position-level results differ only in what is
    treated as an independent observation."""
    ctrl_mean = ctrl["pooled_sum"] / ctrl["pooled_n"]
    ctrl_var = ctrl["pooled_sqsum"] / ctrl["pooled_n"] - ctrl_mean ** 2
    ctrl_sd = np.sqrt(np.maximum(ctrl_var, 1e-12))
    return (pert_cell_mean.mean(axis=0) - ctrl_mean) / ctrl_sd


def design_effect(icc, positions_per_cell):
    """Variance inflation from treating correlated within-cell positions as independent."""
    return 1.0 + (positions_per_cell - 1.0) * icc


def cell_level_test(pert_mean, ctrl_mean):
    """Mann-Whitney over cells for every feature, BH across features."""
    from scipy.stats import mannwhitneyu
    keep = ~((pert_mean.std(axis=0) == 0) & (ctrl_mean.std(axis=0) == 0))
    p = np.ones(pert_mean.shape[1])
    if keep.any():
        _, p_keep = mannwhitneyu(pert_mean[:, keep], ctrl_mean[:, keep],
                                 axis=0, alternative="two-sided")
        p[keep] = p_keep
    q = common.bh_fdr(p)
    d = cohens_d(pert_mean, ctrl_mean)
    return p, q, d


def position_pooled_stat(pert, ctrl):
    """The position-pooled statistic: effect = (mean_pert - mean_ctrl) / sd_ctrl, where
    both moments are taken over gene positions pooled across cells."""
    ctrl_mean = ctrl["pooled_sum"] / ctrl["pooled_n"]
    ctrl_var = ctrl["pooled_sqsum"] / ctrl["pooled_n"] - ctrl_mean ** 2
    ctrl_sd = np.sqrt(np.maximum(ctrl_var, 1e-12))
    pert_mean = pert["pooled_sum"] / pert["pooled_n"]
    effect = (pert_mean - ctrl_mean) / ctrl_sd

    # Normal-approximation p-value using the position count as if positions were
    # independent - this is the assumption whose consequence we are quantifying.
    pert_var = pert["pooled_sqsum"] / pert["pooled_n"] - pert_mean ** 2
    se = np.sqrt(np.maximum(pert_var, 1e-12) / pert["pooled_n"] +
                 np.maximum(ctrl_var, 1e-12) / ctrl["pooled_n"])
    from scipy.stats import norm
    z = (pert_mean - ctrl_mean) / np.maximum(se, 1e-12)
    p = 2 * norm.sf(np.abs(z))
    return effect, p, common.bh_fdr(p)


def icc_per_feature(cell_mean, cell_msq, n_pos):
    """ICC(1) per feature from per-cell means and per-cell mean squares."""
    within = np.maximum(cell_msq - cell_mean ** 2, 0.0)          # (cells, features)
    mean_within = within.mean(axis=0)
    between = cell_mean.var(axis=0, ddof=1)
    total = between + mean_within
    out = np.zeros_like(between)
    nz = total > 0
    out[nz] = between[nz] / total[nz]
    return out


def predicted_targets(responding_idx, catalog, effect, top_features=None):
    """Union of the top-20 gene lists of responding features, ranked by |effect|."""
    order = responding_idx[np.argsort(-np.abs(effect[responding_idx]))]
    if top_features:
        order = order[:top_features]
    genes, seen = [], set()
    for fi in order:
        for g in catalog.get(int(fi), []):
            if g not in seen:
                seen.add(g)
                genes.append(g)
    return genes, order


def specificity_for_target(genes, known, measured, universe_mode="measured"):
    universe = len(measured) if universe_mode == "measured" else 20_000
    return common.hypergeom_enrichment(set(genes), known, universe)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", default="k562")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--sae-dir", default=None)
    ap.add_argument("--n-sweep", type=int, nargs="*", default=[10, 20, 50, 100, 200])
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = common.OUT_ROOT / "cell_features" / f"{args.cell_line}_L{args.layer:02d}_{args.tag}"
    out = common.OUT_ROOT / "E3_cell_level" / f"{args.cell_line}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(src / "manifest.json"))

    _, _, var_genes = common.load_replogle_obs()
    measured = {str(g) for g in var_genes}
    references = {"TRRUST": common.load_trrust(),
                  "DoRothEA_ABC": common.load_dorothea(("A", "B", "C")),
                  "DoRothEA_all": common.load_dorothea(None)}

    # the manifest records the SAE *run* directory; load_sae wants its parent
    sae_base = args.sae_dir
    if sae_base is None and manifest.get("sae_dir"):
        sae_base = str(Path(manifest["sae_dir"]).parent)
    _, _, run_dir = common.load_sae(args.layer, sae_base)
    catalog = common.load_feature_catalog(run_dir)

    ctrl = dict(np.load(src / "control.npz"))
    ctrl_mean = ctrl["mean"]
    print(f"control: {ctrl_mean.shape[0]} cells x {ctrl_mean.shape[1]} features")

    rng = np.random.RandomState(args.seed)
    per_target, sweep_rows, perm_rows = [], [], []

    for entry in manifest["targets"]:
        path = src / entry["file"]
        if not path.exists():
            continue
        pert = dict(np.load(path))
        pm = pert["mean"]
        if pm.shape[0] < 5:
            continue
        gene = entry["gene"]
        t0 = time.time()

        p_cell, q_cell, d_cell = cell_level_test(pm, ctrl_mean)
        eff_scale = pooled_scale_effect(pm, ctrl)
        responding_cell = np.where((q_cell < FDR_THRESHOLD) &
                                   (np.abs(eff_scale) > EFFECT_THRESHOLD))[0]

        eff_pos, p_pos, q_pos = position_pooled_stat(pert, ctrl)
        responding_pos_effect = np.where(np.abs(eff_pos) > EFFECT_THRESHOLD)[0]
        responding_pos_fdr = np.where((q_pos < FDR_THRESHOLD) &
                                      (np.abs(eff_pos) > EFFECT_THRESHOLD))[0]

        icc = icc_per_feature(pm, pert["msq"], pert["n_pos"])
        active = pm.mean(axis=0) > 0

        genes_cell, ranked_cell = predicted_targets(responding_cell, catalog, eff_scale)
        genes_pos, _ = predicted_targets(responding_pos_effect, catalog, eff_pos)

        row = {
            "gene": gene, "role": entry["role"], "is_tf": entry["is_tf"],
            "n_cells": int(pm.shape[0]), "n_positions": int(pert["pooled_n"]),
            "n_responding_cell_level": int(len(responding_cell)),
            "n_responding_position_effect": int(len(responding_pos_effect)),
            "n_responding_position_fdr": int(len(responding_pos_fdr)),
            "inflation_ratio": float(len(responding_pos_fdr) /
                                     max(len(responding_cell), 1)),
            "median_icc_active_features": float(np.median(icc[active])) if active.any() else None,
            "mean_icc_active_features": float(icc[active].mean()) if active.any() else None,
            "mean_positions_per_cell": float(pert["n_pos"].mean()),
            "design_effect": float(design_effect(
                float(np.median(icc[active])) if active.any() else 0.0,
                float(pert["n_pos"].mean()))),
            "effective_n_fraction": float(1.0 / design_effect(
                float(np.median(icc[active])) if active.any() else 0.0,
                float(pert["n_pos"].mean()))),
            "n_significant_cell_level_any_effect": int((q_cell < FDR_THRESHOLD).sum()),
            "n_significant_position_level_any_effect": int((q_pos < FDR_THRESHOLD).sum()),
            "max_abs_effect_pooled_scale": float(np.abs(eff_scale).max()),
            "n_predicted_genes_cell_level": len(genes_cell),
            "predicted_genes_cell_level": genes_cell[:200],
            "n_predicted_genes_position_level": len(genes_pos),
            "self_gene_in_prediction": bool(gene in set(genes_cell)),
            "self_gene_rank": (genes_cell.index(gene) + 1) if gene in genes_cell else None,
            "max_abs_d_cell": float(np.abs(d_cell).max()),
            "top_features": [
                {"feature_idx": int(fi), "cohens_d": float(d_cell[fi]),
                 "fdr": float(q_cell[fi]), "top_genes": catalog.get(int(fi), [])[:10]}
                for fi in ranked_cell[:10]],
        }

        for ref_name, ref in references.items():
            known = {g for g in ref.get(gene, set()) if g in measured} - {gene}
            row[f"n_known_{ref_name}"] = len(known)
            if len(known) >= 5:
                p_c, ov_c, fold_c = specificity_for_target(genes_cell, known, measured)
                p_p, ov_p, fold_p = specificity_for_target(genes_pos, known, measured)
                row[f"{ref_name}_cell_p"] = p_c
                row[f"{ref_name}_cell_overlap"] = ov_c
                row[f"{ref_name}_cell_fold"] = fold_c
                row[f"{ref_name}_position_p"] = p_p
                row[f"{ref_name}_position_overlap"] = ov_p

        # --- label-permutation null on the same cells ------------------------
        if entry["role"] == "tf" and args.n_perm:
            pool = np.vstack([pm, ctrl_mean])
            labels = np.array([1] * pm.shape[0] + [0] * ctrl_mean.shape[0])
            null_counts, null_sig = [], []
            for b in range(args.n_perm):
                prng = np.random.RandomState(args.seed + 7000 + b)
                shuf = prng.permutation(labels)
                a = pool[shuf == 1]
                c = pool[shuf == 0]
                _, q_b, _ = cell_level_test(a, c)
                eff_b = pooled_scale_effect(a, ctrl)
                resp_b = np.where((q_b < FDR_THRESHOLD) &
                                  (np.abs(eff_b) > EFFECT_THRESHOLD))[0]
                null_counts.append(len(resp_b))
                gb, _ = predicted_targets(resp_b, catalog, eff_b)
                known = {g for g in references["DoRothEA_all"].get(gene, set())
                         if g in measured} - {gene}
                if len(known) >= 5 and gb:
                    pb, _, _ = specificity_for_target(gb, known, measured)
                    null_sig.append(pb < 0.05)
            row["perm_null_mean_responding"] = float(np.mean(null_counts))
            row["perm_null_frac_nominally_specific"] = (
                float(np.mean(null_sig)) if null_sig else None)
            perm_rows.append({"gene": gene, "null_responding": null_counts})

        # --- sample-size sweep ----------------------------------------------
        for n in args.n_sweep:
            if n > pm.shape[0]:
                continue
            for b in range(args.n_boot):
                brng = np.random.RandomState(args.seed + 100 * n + b)
                take = brng.choice(pm.shape[0], size=n, replace=False)
                _, q_b, _ = cell_level_test(pm[take], ctrl_mean)
                eff_b = pooled_scale_effect(pm[take], ctrl)
                resp_b = np.where((q_b < FDR_THRESHOLD) &
                                  (np.abs(eff_b) > EFFECT_THRESHOLD))[0]
                gb, _ = predicted_targets(resp_b, catalog, eff_b)
                rec = {"gene": gene, "role": entry["role"], "n": n, "boot": b,
                       "n_responding": int(len(resp_b)),
                       "self_detected": bool(gene in set(gb))}
                for ref_name, ref in references.items():
                    known = {g for g in ref.get(gene, set()) if g in measured} - {gene}
                    if len(known) >= 5 and gb:
                        pb, ovb, _ = specificity_for_target(gb, known, measured)
                        rec[f"{ref_name}_p"] = pb
                        rec[f"{ref_name}_overlap"] = ovb
                sweep_rows.append(rec)

        per_target.append(row)
        print(f"  {gene}: n={row['n_cells']} cells, responding cell-level="
              f"{row['n_responding_cell_level']}, position-level(FDR)="
              f"{row['n_responding_position_fdr']}, ICC="
              f"{row['median_icc_active_features']}, self-detected="
              f"{row['self_gene_in_prediction']} ({time.time() - t0:.0f}s)", flush=True)

    common.write_json(out / "per_target.json",
                      {"cell_line": args.cell_line, "layer": args.layer,
                       "config": vars(args), "per_target": per_target}, seed=args.seed)
    common.write_json(out / "sweep.json",
                      {"rows": sweep_rows, "config": vars(args)}, seed=args.seed)
    common.write_json(out / "permutation_null.json",
                      {"rows": perm_rows, "config": vars(args)}, seed=args.seed)
    print(f"\n{len(per_target)} targets analysed -> {out}")


if __name__ == "__main__":
    main()
