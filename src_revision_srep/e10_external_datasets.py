"""E10 — the regulatory measurement on independent external perturbation datasets.

The primary CRISPRi resource measures a 6,546-gene panel, which bounds how much curated
TF-target structure any method can recover from it. This script repeats the measurement on
perturbation datasets produced by other groups with other protocols and, critically, other
gene panels:

  papalexi   THP-1 monocytes, ECCITE-seq, 18,649 genes  (different cell type, different assay)
  norman     K562, Perturb-seq, 33,694 genes            (near-complete transcriptome)

The Norman panel is roughly five times the size of the primary one, so comparing the
evaluability diagnostics and the empirical ceiling across datasets separates a property of
the assay from a property of the model.

Everything downstream is identical to the primary analysis: the same atlas dictionary, the
same per-cell encoding, the cell as the unit of replication, and the same hypergeometric
enrichment of a size-matched predicted target set.

Outputs: experiments/revision_srep/E10_external/<dataset>/
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from e3_cell_level_analysis import (cell_level_test, icc_per_feature,  # noqa: E402
                                    position_pooled_stat, predicted_targets)

PERTURB_ROOT = Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/perturb")
DATASETS = {
    "papalexi": {
        "path": PERTURB_ROOT / "papalexi/PapalexiSatija2021_eccite_RNA.h5ad",
        "pert_col": "perturbation",
        "control": "control",
        "strip_guide": True,          # labels are guide-level, e.g. ATF2g1
        "var_symbol": "gene_symbol",
        "description": "THP-1 monocytes, ECCITE-seq (Papalexi et al. 2021)",
    },
    "norman": {
        "path": PERTURB_ROOT / "norman/NormanWeissman2019_filtered.h5ad",
        "pert_col": "perturbation",
        "control": "control",
        "strip_guide": False,
        "drop_combinations": True,    # labels like A_B are two-gene perturbations
        "var_symbol": None,           # var index holds symbols
        "description": "K562, Perturb-seq (Norman et al. 2019)",
    },
}
TOP_N = 100
EFFECT_THRESHOLD = 0.5
FDR_THRESHOLD = 0.05


def clean_label(label, cfg):
    if cfg.get("strip_guide"):
        m = re.match(r"^(.*?)g\d+$", label)
        if m and len(m.group(1)) >= 2:
            return m.group(1)
    return label


def load_dataset(cfg):
    import scanpy as sc
    print(f"  reading {cfg['path'].name} ...", flush=True)
    adata = sc.read_h5ad(cfg["path"])
    if cfg["var_symbol"] and cfg["var_symbol"] in adata.var:
        symbols = np.asarray(adata.var[cfg["var_symbol"]].astype(str))
    else:
        symbols = np.asarray(adata.var_names.astype(str))
    labels_raw = np.asarray(adata.obs[cfg["pert_col"]].astype(str))
    labels = np.array([clean_label(l, cfg) for l in labels_raw])
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes; "
          f"{len(set(labels))} perturbation labels", flush=True)
    return adata, symbols, labels


def normalise_rows(X):
    X = np.asarray(X, dtype=np.float32)
    row = X.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    np.log1p(X / row * 1e4, out=X)
    return X


def build_panel(labels, cfg, references, measured, min_cells, min_targets,
                n_nontf=0, seed=42):
    import collections
    counts = collections.Counter(labels)
    control = cfg["control"]
    diagnostics = {"n_labels": len(counts),
                   "n_control_cells": int(counts.get(control, 0))}
    candidates = []
    for label, n in counts.items():
        if label == control or n < min_cells:
            continue
        if cfg.get("drop_combinations") and "_" in label:
            continue
        candidates.append((label, n))
    diagnostics["n_targets_with_enough_cells"] = len(candidates)

    per_ref = {}
    for ref_name, ref in references.items():
        measured_counts = {lab: len({g for g in ref.get(lab, set()) if g in measured} - {lab})
                           for lab, _ in candidates if lab in ref}
        per_ref[ref_name] = measured_counts
        diagnostics[f"{ref_name}_tfs_perturbed"] = len(measured_counts)
        for t in (1, 3, 5, 10, 20):
            diagnostics[f"{ref_name}_tfs_with_ge{t}_measured_targets"] = int(
                sum(1 for v in measured_counts.values() if v >= t))
        vals = sorted(measured_counts.values())
        diagnostics[f"{ref_name}_median_measured_targets_per_tf"] = (
            float(np.median(vals)) if vals else 0.0)

    panel = []
    for label, n in sorted(candidates, key=lambda kv: -kv[1]):
        entry = {"gene": label, "n_cells": int(n), "role": "tf"}
        evaluable = False
        for ref_name in references:
            v = per_ref[ref_name].get(label, 0)
            entry[f"n_known_{ref_name}"] = int(v)
            if v >= min_targets:
                evaluable = True
        if evaluable:
            panel.append(entry)
    diagnostics["panel_size"] = len(panel)

    # Perturbations of genes absent from every reference network. If the predicted target
    # sets enrich for a factor's regulon only because they are broad, expression-biased
    # signatures, these will enrich just as often as the transcription factors do.
    if n_nontf:
        import numpy as _np
        rng = _np.random.RandomState(seed)
        chosen = {e["gene"] for e in panel}
        non_tf = sorted(lab for lab, n in candidates
                        if lab not in chosen
                        and not any(lab in r for r in references.values()))
        for g in rng.choice(non_tf, size=min(n_nontf, len(non_tf)), replace=False):
            panel.append({"gene": str(g), "n_cells": int(dict(candidates)[str(g)]),
                          "role": "non_tf_control"})
    return panel, diagnostics


def de_ceiling(Xp, Xc, symbols, gene):
    from scipy.stats import mannwhitneyu
    _, p = mannwhitneyu(Xp, Xc, axis=0, alternative="two-sided")
    q = common.bh_fdr(p)
    diff = Xp.mean(axis=0) - Xc.mean(axis=0)
    pooled = np.sqrt((Xp.var(axis=0) + Xc.var(axis=0)) / 2.0)
    pooled[pooled == 0] = 1e-9
    d = diff / pooled
    score = np.abs(d)
    self_idx = np.where(symbols == gene)[0]
    self_d = float(d[self_idx[0]]) if len(self_idx) else None
    if len(self_idx):
        score[self_idx[0]] = -np.inf
    order = np.argsort(-score)
    return ([symbols[i] for i in order[:TOP_N]], int(((q < 0.05) & (np.abs(d) > 0.2)).sum()),
            self_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--sae-dir", default=None)
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--max-targets", type=int, default=16)
    ap.add_argument("--n-nontf-controls", type=int, default=0)
    ap.add_argument("--cells-per-target", type=int, default=100)
    ap.add_argument("--control-cells", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    out = common.OUT_ROOT / "E10_external" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    common.seed_everything(args.seed)
    t_start = time.time()

    references = {"TRRUST": common.load_trrust(),
                  "DoRothEA_ABC": common.load_dorothea(("A", "B", "C")),
                  "DoRothEA_all": common.load_dorothea(None)}

    adata, symbols, labels = load_dataset(cfg)
    measured = set(symbols.tolist())
    panel, diagnostics = build_panel(labels, cfg, references, measured,
                                     args.min_cells, args.min_targets,
                                     n_nontf=args.n_nontf_controls, seed=args.seed)
    diagnostics["n_measured_genes"] = int(len(measured))
    print(f"  evaluable panel: {len(panel)} transcription factors "
          f"out of {diagnostics['n_targets_with_enough_cells']} perturbations with "
          f"enough cells", flush=True)
    for k, v in diagnostics.items():
        print(f"    {k}: {v}")
    tf_panel = [e for e in panel if e.get("role") == "tf"][:args.max_targets]
    panel = tf_panel + [e for e in panel if e.get("role") == "non_tf_control"]

    rng = np.random.RandomState(args.seed)
    ctrl_idx = np.where(labels == cfg["control"])[0]
    ctrl_take = np.sort(rng.choice(ctrl_idx, min(args.control_cells, len(ctrl_idx)),
                                   replace=False))
    Xc = normalise_rows(adata[ctrl_take].X.toarray())

    sae, act_mean_t, run_dir = common.load_sae(args.layer, args.sae_dir)
    catalog = common.load_feature_catalog(run_dir)
    tokenizer = common.GeneformerTokenizer(symbols)
    print(f"  {len(tokenizer.var_indices):,} of {len(symbols):,} genes map to the model "
          f"vocabulary", flush=True)
    model, device = common.load_geneformer()

    ctrl_res = common.encode_cells(model, device, sae, act_mean_t, tokenizer, Xc,
                                   args.layer, tag="control ")[0]
    print(f"  control: {ctrl_res['mean'].shape[0]} cells "
          f"({time.time() - t_start:.0f}s)", flush=True)

    rows, sae_preds, de_preds = [], {}, {}
    for ti, entry in enumerate(panel):
        gene = entry["gene"]
        idx = np.where(labels == gene)[0]
        trng = np.random.RandomState(args.seed + 100 + ti)
        if len(idx) > args.cells_per_target:
            idx = trng.choice(idx, args.cells_per_target, replace=False)
        idx = np.sort(idx)
        Xp = normalise_rows(adata[idx].X.toarray())

        res = common.encode_cells(model, device, sae, act_mean_t, tokenizer, Xp,
                                  args.layer, progress_every=0)[0]
        if res["mean"].shape[0] < 5:
            continue
        _, q_cell, d_cell = cell_level_test(res["mean"], ctrl_res["mean"])
        responding = np.where((q_cell < FDR_THRESHOLD) &
                              (np.abs(d_cell) > EFFECT_THRESHOLD))[0]
        genes_cell, _ = predicted_targets(responding, catalog, d_cell)
        sae_preds[gene] = genes_cell[:TOP_N]

        de_genes, n_de, self_d = de_ceiling(Xp, Xc, symbols, gene)
        de_preds[gene] = de_genes

        eff_pos, _, q_pos = position_pooled_stat(res, ctrl_res)
        icc = icc_per_feature(res["mean"], res["msq"], res["n_pos"])
        active = res["mean"].mean(axis=0) > 0

        row = {"gene": gene, "role": entry.get("role", "tf"),
               "n_cells": int(res["mean"].shape[0]),
               "n_responding_cell_level": int(len(responding)),
               "n_responding_position_fdr": int(((q_pos < FDR_THRESHOLD) &
                                                 (np.abs(eff_pos) > EFFECT_THRESHOLD)).sum()),
               "median_icc_active_features": float(np.median(icc[active])) if active.any() else None,
               "n_de_genes": n_de, "self_knockdown_cohens_d": self_d,
               "self_gene_in_prediction": bool(gene in set(genes_cell)),
               "predicted_genes_cell_level": genes_cell[:200]}
        for ref_name, ref in references.items():
            row[f"n_known_{ref_name}"] = entry.get(f"n_known_{ref_name}", 0)
        rows.append(row)
        print(f"  [{ti + 1}/{len(panel)}] {gene}: n={row['n_cells']}, "
              f"responding={row['n_responding_cell_level']}, DE genes={n_de}, "
              f"self d={self_d if self_d is None else round(self_d, 2)}", flush=True)

    from e1_recoverability_ceiling import score_method
    scores = {}
    for ref_name, ref in references.items():
        scores[f"{ref_name}|measured"] = {
            "sae_features": score_method(sae_preds, [{"tf": r["gene"]} for r in rows],
                                         ref, measured, "measured", args.min_targets),
            "de_perturbation": score_method(de_preds, [{"tf": r["gene"]} for r in rows],
                                            ref, measured, "measured", args.min_targets),
        }

    print("\n=== fraction of TFs with FDR<0.05 target enrichment ===")
    for key, per_method in scores.items():
        for m, s in per_method.items():
            print(f"  {key:22s} {m:16s} {s['n_significant']}/{s['n_tfs']} "
                  f"= {100 * s['frac_significant']:.1f}%")

    common.write_json(out / "results.json",
                      {"dataset": args.dataset, "description": cfg["description"],
                       "evaluability": diagnostics, "config": vars(args),
                       "per_target": rows, "scores": scores}, seed=args.seed)
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
