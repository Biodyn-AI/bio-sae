"""E2/E3/E4 extractor — per-cell SAE feature activations for CRISPRi perturbations.

One forward pass per cell, hooked at the target layer, encoded through the layer's SAE,
and aggregated to one row per cell. Caching the per-cell matrix (rather than pooled
position statistics) is what allows the downstream analysis to

  * treat the cell as the unit of replication (removing the pseudoreplication that
    inflates a position-pooled test),
  * resample cells to any n <= n_extracted for the power analysis,
  * and reproduce the position-pooled statistic side by side for comparison.

Runs for any of the four Replogle cell lines and any SAE directory, so the same code
serves the K562 analysis and the independent-cell-line replications.

Outputs: experiments/revision_srep/cell_features/<cell_line>_L<layer>_<tag>/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def select_targets(cell_line, min_cells, max_targets, references, measured, seed,
                   include_paper_panel=True, n_nontf_controls=20):
    cl, pert, _ = common.load_replogle_obs()
    mask = cl == cell_line
    line_idx = np.where(mask)[0]
    pert_line = pert[mask]
    control_idx = line_idx[pert_line == "non-targeting"]

    counts = {}
    for g in np.unique(pert_line):
        if g != "non-targeting":
            counts[g] = int((pert_line == g).sum())

    paper_panel = set()
    if include_paper_panel:
        pr = common.PHASE1 / "perturbation_response/perturbation_response_layer11.json"
        if pr.exists():
            paper_panel = {r["target_gene"] for r in
                           json.load(open(pr))["target_results"]}

    def measured_targets(tf, ref):
        return len({g for g in ref.get(tf, set()) if g in measured} - {tf})

    scored = []
    for tf, n in counts.items():
        if n < min_cells:
            continue
        best = max((measured_targets(tf, r) for r in references.values()), default=0)
        is_tf = any(tf in r for r in references.values())
        if best >= 5 or (include_paper_panel and tf in paper_panel and is_tf):
            scored.append({"gene": tf, "n_cells": n, "is_tf": is_tf,
                           "max_measured_targets": best,
                           "in_paper_panel": tf in paper_panel,
                           "role": "tf"})
    scored.sort(key=lambda r: (-r["max_measured_targets"], -r["n_cells"]))
    selected = scored[:max_targets]

    # Non-TF perturbations act as a negative control for the specificity test.
    rng = np.random.RandomState(seed)
    chosen = {r["gene"] for r in selected}
    non_tf = [g for g, n in counts.items()
              if n >= min_cells and g not in chosen
              and not any(g in r for r in references.values())]
    for g in rng.choice(sorted(non_tf), size=min(n_nontf_controls, len(non_tf)),
                        replace=False):
        selected.append({"gene": str(g), "n_cells": counts[str(g)], "is_tf": False,
                         "max_measured_targets": 0, "in_paper_panel": False,
                         "role": "non_tf_control"})
    return selected, control_idx, line_idx, pert_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", default="k562")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--sae-dirs", nargs="*", default=[None],
                    help="one or more SAE base directories; None = published K562 atlas")
    ap.add_argument("--sae-names", nargs="*", default=["k562sae"],
                    help="short name per SAE, used as the output sub-directory")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--cells-per-target", type=int, default=200)
    ap.add_argument("--control-cells", type=int, default=600)
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--max-targets", type=int, default=48)
    ap.add_argument("--n-nontf-controls", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = common.OUT_ROOT / "cell_features" / f"{args.cell_line}_L{args.layer:02d}_{args.tag}"
    outs = [root / name for name in args.sae_names]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    common.seed_everything(args.seed)

    _, _, var_genes = common.load_replogle_obs()
    measured = {str(g) for g in var_genes}
    references = {"TRRUST": common.load_trrust(),
                  "DoRothEA_ABC": common.load_dorothea(("A", "B", "C")),
                  "DoRothEA_all": common.load_dorothea(None)}

    targets, control_idx, line_idx, pert_line = select_targets(
        args.cell_line, args.min_cells, args.max_targets, references, measured,
        args.seed, n_nontf_controls=args.n_nontf_controls)
    n_tf = sum(1 for t in targets if t["role"] == "tf")
    print(f"{args.cell_line}: {len(targets)} targets "
          f"({n_tf} TFs + {len(targets) - n_tf} non-TF controls), "
          f"{len(control_idx)} non-targeting control cells available")

    saes, act_means, run_dirs = [], [], []
    for sd in args.sae_dirs:
        sd = None if sd in (None, "None", "default") else sd
        sae, mean_t, run_dir = common.load_sae(args.layer, sd)
        saes.append(sae)
        act_means.append(mean_t)
        run_dirs.append(run_dir)
        print(f"SAE: {run_dir} ({sae.n_features} features, k={sae.k})")
    tokenizer = common.GeneformerTokenizer(var_genes)
    model, device = common.load_geneformer()
    print(f"device: {device}")

    rng = np.random.RandomState(args.seed)

    # ---- control cells -----------------------------------------------------
    take = rng.choice(control_idx,
                      size=min(args.control_cells, len(control_idx)), replace=False)
    if all((o / "control.npz").exists() for o in outs):
        print("control cells already extracted")
    else:
        X, used = common.read_expression(take)
        t0 = time.time()
        res = common.encode_cells(model, device, saes, act_means, tokenizer, X,
                                  args.layer, tag="control ")
        for o, r in zip(outs, res):
            np.savez_compressed(
                o / "control.npz", mean=r["mean"], freq=r["freq"], msq=r["msq"],
                n_pos=r["n_pos"], cell_idx=used[r["kept"]],
                pooled_sum=r["pooled_sum"], pooled_sqsum=r["pooled_sqsum"],
                pooled_active=r["pooled_active"], pooled_n=r["pooled_n"])
        print(f"  control: {res[0]['mean'].shape[0]} cells, {res[0]['pooled_n']} positions "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- perturbed cells ---------------------------------------------------
    manifest = []
    for ti, target in enumerate(targets):
        gene = target["gene"]
        paths = [o / f"target_{gene}.npz" for o in outs]
        manifest.append({**target, "file": f"target_{gene}.npz"})
        if all(p.exists() for p in paths):
            continue
        cells = line_idx[pert_line == gene]
        trng = np.random.RandomState(args.seed + 1000 + ti)
        if len(cells) > args.cells_per_target:
            cells = trng.choice(cells, size=args.cells_per_target, replace=False)
        X, used = common.read_expression(cells)
        t0 = time.time()
        res = common.encode_cells(model, device, saes, act_means, tokenizer, X,
                                  args.layer, progress_every=0)
        for path, r in zip(paths, res):
            np.savez_compressed(
                path, mean=r["mean"], freq=r["freq"], msq=r["msq"],
                n_pos=r["n_pos"], cell_idx=used[r["kept"]],
                pooled_sum=r["pooled_sum"], pooled_sqsum=r["pooled_sqsum"],
                pooled_active=r["pooled_active"], pooled_n=r["pooled_n"])
        print(f"  [{ti + 1}/{len(targets)}] {gene}: {res[0]['mean'].shape[0]} cells, "
              f"{res[0]['pooled_n']} positions ({time.time() - t0:.0f}s)", flush=True)

    for o, run_dir in zip(outs, run_dirs):
        common.write_json(o / "manifest.json",
                          {"cell_line": args.cell_line, "layer": args.layer,
                           "sae_dir": str(run_dir), "config": vars(args),
                           "targets": manifest}, seed=args.seed)


if __name__ == "__main__":
    main()
