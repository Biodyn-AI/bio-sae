"""E1 — how much TF->target information is recoverable from these data at all?

A specificity rate is only interpretable against a ceiling. This script scores several
methods on one common footing: each emits a ranked list of putative targets for every
TF in the panel, and every list is tested for enrichment of that TF's curated targets by
hypergeometric test with BH correction across the panel.

Methods
  de_perturbation   differential expression of the CRISPRi knockdown itself (cell-level
                    Wilcoxon, perturbed vs non-targeting). Perturbation-aware, and the
                    strongest signal obtainable from this dataset -> the empirical ceiling.
  genie3            random-forest variable importance per target gene, control cells only.
  grnboost2         stochastic gradient boosting importance, control cells only.
  pearson/spearman  co-expression with the TF, control cells only.
  geneformer_emb    cosine similarity of mean contextual gene embeddings from the
                    residual stream (model-internal, no SAE).
  random            size-matched random gene sets.

The SAE arm is scored by e2_cell_level_perturbation.py and merged in by e1b_merge.py so
that both use identical cell selections.

Outputs: experiments/revision_srep/E1_ceiling/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.OUT_ROOT / "E1_ceiling"
TOP_N = 100


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_panel(cell_line, min_cells, min_targets, references, measured_genes,
                paper_panel=None):
    """TFs perturbed in this line that are *evaluable*: enough cells, and enough curated
    targets actually present in the measured gene space. The second condition is the
    binding one and is reported explicitly - a TF whose curated targets are not measured
    cannot show target-specific anything, by any method."""
    cl, pert, _ = common.load_replogle_obs()
    mask = cl == cell_line
    idx = np.where(mask)[0]
    pert_line = pert[mask]

    control_idx = idx[pert_line == "non-targeting"]
    counts = {}
    for g in np.unique(pert_line):
        if g == "non-targeting":
            continue
        counts[g] = int((pert_line == g).sum())

    any_ref_tfs = set().union(*[set(r) for r in references.values()])
    panel, diagnostics = [], {"n_perturbed_targets": len(counts)}
    enough_cells = [tf for tf, n in counts.items() if n >= min_cells]
    diagnostics["n_targets_with_enough_cells"] = len(enough_cells)
    diagnostics["n_tf_in_any_reference"] = sum(1 for tf in enough_cells if tf in any_ref_tfs)

    per_ref_counts = {}
    for ref_name, ref in references.items():
        measured_target_counts = {
            tf: len({g for g in ref.get(tf, set()) if g in measured_genes} - {tf})
            for tf in enough_cells if tf in ref
        }
        per_ref_counts[ref_name] = measured_target_counts
        diagnostics[f"{ref_name}_tfs_perturbed"] = len(measured_target_counts)
        for thresh in (1, 3, 5, 10, 20):
            diagnostics[f"{ref_name}_tfs_with_ge{thresh}_measured_targets"] = int(
                sum(1 for v in measured_target_counts.values() if v >= thresh))
        vals = sorted(measured_target_counts.values())
        diagnostics[f"{ref_name}_median_measured_targets_per_tf"] = (
            float(np.median(vals)) if vals else 0.0)

    for tf in sorted(enough_cells):
        entry = {"tf": tf, "n_cells": counts[tf],
                 "in_paper_panel": bool(paper_panel and tf in paper_panel)}
        evaluable = False
        for ref_name in references:
            n = per_ref_counts[ref_name].get(tf, 0)
            entry[f"n_known_{ref_name}"] = int(n)
            if n >= min_targets:
                evaluable = True
        if evaluable:
            panel.append(entry)
    diagnostics["panel_size"] = len(panel)
    return panel, control_idx, idx, pert_line, diagnostics


# ---------------------------------------------------------------------------
# Expression matrices
# ---------------------------------------------------------------------------

def control_matrix(control_idx, n_cells, seed):
    rng = np.random.RandomState(seed)
    take = rng.choice(control_idx, size=min(n_cells, len(control_idx)), replace=False)
    X, used = common.read_expression(take)
    return X, used


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def method_correlation(X, gene_names, panel, kind="pearson"):
    """Rank genes by |correlation| with the TF across control cells."""
    from scipy.stats import rankdata
    name_to_col = {g: i for i, g in enumerate(gene_names)}
    Z = X
    if kind == "spearman":
        Z = np.apply_along_axis(rankdata, 0, X)
    Z = Z - Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Z = Z / sd
    n = Z.shape[0]
    out = {}
    for entry in panel:
        tf = entry["tf"]
        if tf not in name_to_col:
            continue
        v = Z[:, name_to_col[tf]]
        r = (Z.T @ v) / n
        r[name_to_col[tf]] = 0.0
        order = np.argsort(-np.abs(r))
        out[tf] = [gene_names[i] for i in order[:TOP_N]]
    return out


def method_tree_ensemble(X, gene_names, panel, algorithm, seed, n_jobs=5):
    """GENIE3 and GRNBoost2 in their published formulation: regress every target gene on
    the candidate regulators with a tree ensemble and rank regulators by variable
    importance. Hyperparameters mirror the arboreto reference implementation
    (max_features='sqrt' for the forest; lr 0.01, max_features 0.1, subsample 0.9 and
    early stopping for the boosting variant). Ensemble size is reduced from arboreto's
    default to keep 6,546 per-target models tractable; importances are stable well below
    the default size."""
    from joblib import Parallel, delayed as joblib_delayed
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    name_to_col = {g: i for i, g in enumerate(gene_names)}
    tf_names = [e["tf"] for e in panel if e["tf"] in name_to_col]
    tf_cols = [name_to_col[t] for t in tf_names]
    TF = X[:, tf_cols]

    def fit_one(j):
        y = X[:, j]
        if y.std() == 0:
            return np.zeros(len(tf_cols), dtype=np.float32)
        Xj, cols = TF, list(range(len(tf_cols)))
        if j in tf_cols:                      # a regulator never predicts itself
            drop = tf_cols.index(j)
            keep = [c for c in cols if c != drop]
            Xj = TF[:, keep]
        else:
            keep = cols
        if algorithm == "genie3":
            reg = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                        random_state=seed, n_jobs=1)
        else:
            reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000,
                                            max_features=0.1, subsample=0.9,
                                            n_iter_no_change=25, validation_fraction=0.1,
                                            random_state=seed)
        reg.fit(Xj, y)
        imp = np.zeros(len(tf_cols), dtype=np.float32)
        imp[keep] = reg.feature_importances_.astype(np.float32)
        return imp

    n_targets = X.shape[1]
    print(f"    fitting {n_targets} target models on {len(tf_cols)} regulators "
          f"({algorithm}) ...", flush=True)
    importances = Parallel(n_jobs=n_jobs, verbose=1)(
        joblib_delayed(fit_one)(j) for j in range(n_targets))
    M = np.stack(importances)                 # (n_targets, n_tfs)

    out = {}
    for ti, tf in enumerate(tf_names):
        scores = M[:, ti].copy()
        scores[name_to_col[tf]] = -np.inf
        order = np.argsort(-scores)
        out[tf] = [gene_names[i] for i in order[:TOP_N]]
    return out


def method_de_perturbation(cell_line, panel, control_idx, line_idx, pert_line,
                           gene_names, n_pert_cells, n_ctrl_cells, seed, min_cells):
    """Cell-level Wilcoxon of perturbed vs non-targeting cells: the empirical ceiling."""
    from scipy.stats import mannwhitneyu
    rng = np.random.RandomState(seed)
    ctrl_take = rng.choice(control_idx, size=min(n_ctrl_cells, len(control_idx)),
                           replace=False)
    Xc, _ = common.read_expression(ctrl_take)

    out, detail = {}, {}
    for entry in panel:
        tf = entry["tf"]
        tf_cells = line_idx[pert_line == tf]
        if len(tf_cells) < min_cells:
            continue
        if len(tf_cells) > n_pert_cells:
            tf_cells = rng.choice(tf_cells, size=n_pert_cells, replace=False)
        Xp, _ = common.read_expression(tf_cells)

        stat, p = mannwhitneyu(Xp, Xc, axis=0, alternative="two-sided")
        q = common.bh_fdr(p)
        diff = Xp.mean(axis=0) - Xc.mean(axis=0)
        pooled = np.sqrt((Xp.var(axis=0) + Xc.var(axis=0)) / 2.0)
        pooled[pooled == 0] = 1e-9
        d = diff / pooled

        self_col = {g: i for i, g in enumerate(gene_names)}.get(tf)
        score = np.abs(d)
        if self_col is not None:
            score[self_col] = -np.inf          # exclude the knocked-down gene itself
        order = np.argsort(-score)
        out[tf] = [gene_names[i] for i in order[:TOP_N]]

        sig = np.where((q < 0.05) & (np.abs(d) > 0.2))[0]
        detail[tf] = {
            "n_perturbed_cells": int(len(tf_cells)),
            "n_control_cells": int(len(ctrl_take)),
            "n_de_genes_fdr05": int(len(sig)),
            "de_genes_fdr05": [gene_names[i] for i in sig[:500]],
            "self_knockdown_cohens_d": float(d[self_col]) if self_col is not None else None,
            "self_knockdown_fdr": float(q[self_col]) if self_col is not None else None,
        }
        self_d = detail[tf]["self_knockdown_cohens_d"]
        print(f"    DE {tf}: n={len(tf_cells)}, DE genes={len(sig)}, "
              f"self d={self_d:.2f}" if self_d is not None else
              f"    DE {tf}: n={len(tf_cells)}, DE genes={len(sig)}, self d=n/a",
              flush=True)
    return out, detail


def method_geneformer_embeddings(layer, panel, gene_names, seed, n_blocks=40,
                                 block_rows=12_500):
    """Cosine similarity of mean per-gene residual-stream vectors (no SAE involved)."""
    act_path = common.PHASE1 / f"layer_{layer:02d}_activations.npy"
    mm = np.lib.format.open_memmap(act_path, mode="r")
    gene_ids = np.load(common.PHASE1 / f"layer_{layer:02d}_gene_ids.npy", mmap_mode="r")
    with open(common.PHASE1 / "token_id_to_gene_name.json") as fh:
        token_to_gene = {int(k): v for k, v in json.load(fh).items()}

    rng = np.random.RandomState(seed)
    starts = np.linspace(0, max(mm.shape[0] - block_rows, 1), n_blocks).astype(np.int64)
    starts = np.clip(starts + rng.randint(0, block_rows // 2, size=n_blocks),
                     0, max(mm.shape[0] - block_rows, 0))

    d = mm.shape[1]
    sums, counts = {}, {}
    for s in sorted(set(starts.tolist())):
        block = np.asarray(mm[s:s + block_rows], dtype=np.float32)
        toks = np.asarray(gene_ids[s:s + block.shape[0]])
        for t in np.unique(toks):
            sel = toks == t
            sums[int(t)] = sums.get(int(t), np.zeros(d, dtype=np.float64)) + \
                block[sel].sum(axis=0, dtype=np.float64)
            counts[int(t)] = counts.get(int(t), 0) + int(sel.sum())

    tokens = sorted(sums)
    names = [token_to_gene.get(t) for t in tokens]
    M = np.stack([sums[t] / counts[t] for t in tokens]).astype(np.float32)
    M = M - M.mean(axis=0, keepdims=True)
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

    name_to_row = {n: i for i, n in enumerate(names) if n}
    out = {}
    for entry in panel:
        tf = entry["tf"]
        if tf not in name_to_row:
            continue
        sim = M @ M[name_to_row[tf]]
        sim[name_to_row[tf]] = -np.inf
        order = np.argsort(-sim)
        picked = [names[i] for i in order if names[i]][:TOP_N]
        out[tf] = picked
    return out


def method_random(panel, gene_names, seed):
    rng = np.random.RandomState(seed)
    out = {}
    for entry in panel:
        out[entry["tf"]] = list(rng.choice(gene_names, size=TOP_N, replace=False))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_method(predictions, panel, reference, measured_genes, universe_mode,
                 min_targets=5, missing_counts_as_failure=True):
    """Hypergeometric enrichment of each TF's predicted set for its curated targets.

    A method that emits no prediction for an evaluable transcription factor has failed to
    recover its targets, so by default such a factor stays in the denominator with p = 1.
    Dropping it instead would score every method only on the factors where it happened to
    fire, which favours methods that abstain."""
    universe = len(measured_genes) if universe_mode == "measured" else 20_000
    rows, pvals = [], []
    for entry in panel:
        tf = entry["tf"]
        known = {g for g in reference.get(tf, set()) if g in measured_genes} - {tf}
        if len(known) < min_targets:
            continue
        pred = predictions.get(tf)
        if not pred:
            if not missing_counts_as_failure:
                continue
            rows.append({"tf": tf, "n_predicted": 0, "n_known": len(known),
                         "overlap": 0, "fold_enrichment": 0.0, "p": 1.0,
                         "no_prediction": True})
            pvals.append(1.0)
            continue
        p, overlap, fold = common.hypergeom_enrichment(set(pred) - {tf}, known, universe)
        rows.append({"tf": tf, "n_predicted": len(set(pred) - {tf}),
                     "n_known": len(known), "overlap": overlap,
                     "fold_enrichment": fold, "p": p})
        pvals.append(p)
    if not rows:
        return {"n_tfs": 0, "n_significant": 0, "frac_significant": 0.0, "per_tf": []}
    q = common.bh_fdr(pvals)
    for r, qq in zip(rows, q):
        r["fdr"] = float(qq)
        r["significant"] = bool(qq < 0.05)
    n_sig = sum(r["significant"] for r in rows)
    return {
        "n_tfs": len(rows),
        "n_significant": int(n_sig),
        "frac_significant": float(n_sig / len(rows)),
        "median_fold_enrichment": float(np.median([r["fold_enrichment"] for r in rows])),
        "mean_overlap": float(np.mean([r["overlap"] for r in rows])),
        "universe": universe,
        "per_tf": rows,
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", default="k562")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--n-control-cells", type=int, default=2000)
    ap.add_argument("--n-pert-cells", type=int, default=200)
    ap.add_argument("--methods", nargs="*",
                    default=["random", "pearson", "spearman", "geneformer_emb",
                             "de_perturbation", "genie3", "grnboost2"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    _, _, var_genes = common.load_replogle_obs()
    gene_names = np.array([str(g) for g in var_genes])
    measured = set(gene_names.tolist())
    trrust = common.load_trrust()
    dorothea_abc = common.load_dorothea(("A", "B", "C"))
    dorothea_all = common.load_dorothea(None)
    union_ref = {}
    for ref in (trrust, dorothea_all):
        for tf, tgts in ref.items():
            union_ref.setdefault(tf, set()).update(tgts)
    references = {"TRRUST": trrust, "DoRothEA_ABC": dorothea_abc,
                  "DoRothEA_all": dorothea_all, "union": union_ref}

    paper_panel = None
    pr = common.PHASE1 / "perturbation_response/perturbation_response_layer11.json"
    if pr.exists():
        paper_panel = {r["target_gene"] for r in json.load(open(pr))["target_results"]
                       if r.get("is_trrust_tf")}

    panel, control_idx, line_idx, pert_line, diagnostics = build_panel(
        args.cell_line, args.min_cells, args.min_targets, references, measured, paper_panel)
    print(f"panel: {len(panel)} evaluable TFs in {args.cell_line} "
          f"(>= {args.min_cells} cells, >= {args.min_targets} measured targets in >=1 "
          f"reference); {sum(e['in_paper_panel'] for e in panel)} of them in the "
          f"published panel")
    print("evaluability diagnostics:")
    for k, v in diagnostics.items():
        print(f"    {k}: {v}")
    print(f"control cells available: {len(control_idx)}")

    predictions = {}
    extras = {}

    if "random" in args.methods:
        predictions["random"] = method_random(panel, gene_names, args.seed)

    need_ctrl = {"pearson", "spearman", "genie3", "grnboost2"} & set(args.methods)
    if need_ctrl:
        print(f"\nloading {args.n_control_cells} control cells ...", flush=True)
        Xc, used = control_matrix(control_idx, args.n_control_cells, args.seed)
        print(f"  control matrix {Xc.shape} ({time.time() - t_start:.0f}s)", flush=True)
        for kind in ("pearson", "spearman"):
            if kind in args.methods:
                t0 = time.time()
                predictions[kind] = method_correlation(Xc, gene_names, panel, kind)
                print(f"  {kind}: {len(predictions[kind])} TFs ({time.time() - t0:.0f}s)",
                      flush=True)
        for algo in ("genie3", "grnboost2"):
            if algo in args.methods:
                t0 = time.time()
                print(f"  running {algo} ...", flush=True)
                predictions[algo] = method_tree_ensemble(Xc, gene_names, panel, algo,
                                                        args.seed)
                print(f"  {algo}: {len(predictions[algo])} TFs "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del Xc

    if "geneformer_emb" in args.methods and args.cell_line == "k562":
        t0 = time.time()
        predictions["geneformer_emb"] = method_geneformer_embeddings(
            args.layer, panel, gene_names, args.seed)
        print(f"  geneformer_emb: {len(predictions['geneformer_emb'])} TFs "
              f"({time.time() - t0:.0f}s)", flush=True)

    if "de_perturbation" in args.methods:
        print("\nrunning perturbation DE (empirical ceiling) ...", flush=True)
        preds, detail = method_de_perturbation(
            args.cell_line, panel, control_idx, line_idx, pert_line, gene_names,
            args.n_pert_cells, args.n_control_cells, args.seed, args.min_cells)
        predictions["de_perturbation"] = preds
        extras["de_detail"] = detail

    scores = {}
    for ref_name, ref in references.items():
        for universe_mode in ("measured", "20000"):
            key = f"{ref_name}|{universe_mode}"
            scores[key] = {m: score_method(p, panel, ref, measured, universe_mode,
                                           args.min_targets)
                           for m, p in predictions.items()}

    print("\n=== fraction of TFs with FDR<0.05 target enrichment ===")
    for key, per_method in scores.items():
        print(f"  [{key}]")
        for m, s in sorted(per_method.items(),
                           key=lambda kv: -kv[1]["frac_significant"]):
            print(f"    {m:18s} {s['n_significant']:3d}/{s['n_tfs']:3d} "
                  f"= {100 * s['frac_significant']:5.1f}%   "
                  f"median fold {s.get('median_fold_enrichment', 0):.2f}")

    common.write_json(OUT / f"predictions_{args.cell_line}.json",
                      {"panel": panel, "predictions": predictions, "top_n": TOP_N,
                       "config": vars(args)}, seed=args.seed)
    common.write_json(OUT / f"scores_{args.cell_line}.json",
                      {"scores": scores, "panel_size": len(panel),
                       "evaluability": diagnostics,
                       "config": vars(args), **extras}, seed=args.seed)
    print(f"\ntotal {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
