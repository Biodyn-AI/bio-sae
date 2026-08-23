"""E8 — run-to-run stability of the SAE dictionaries.

Retrains the atlas SAE at several layers under independent random seeds and asks how much
of the reported structure is reproducible. Two sources of randomness are separated:

  init      identical training subsample, different weight initialisation and batch order
  data      different 1M-position training subsample as well

For every pair of runs the dictionaries are matched by decoder cosine similarity (greedy
one-to-one assignment), giving a reproducibility curve rather than a single number. The
downstream quantities the paper reports - variance explained, dead features, decoder
coherence, annotation rate, module count - are recomputed per run so their spread is
reported alongside.

Outputs: experiments/revision_srep/E8_seed_stability/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from sae_model import TopKSAE, SAETrainer  # noqa: E402

OUT = common.OUT_ROOT / "E8_seed_stability"
ONTOLOGY_DIR = common.BASE / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets"
TRAIN_SUBSAMPLE = 1_000_000
VAL_HOLDOUT = 100_000
TOPGENE_SUBSAMPLE = 300_000
MODULE_SUBSAMPLE = 200_000
EPOCHS = 5
LR = 3e-4
BATCH_SIZE = 4096
MIN_COACT = 20
PMI_THRESHOLD = 2.0


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def subsample_indices(n_total, size, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(size, n_total), replace=False))


def train_run(layer, seed, data_seed, act_mm, act_mean, n_features, k):
    common.seed_everything(seed)
    train_idx = subsample_indices(act_mm.shape[0], TRAIN_SUBSAMPLE, data_seed)
    train_act = np.asarray(act_mm[train_idx], dtype=np.float32) - act_mean[None, :]

    sae = TopKSAE(d_model=act_mm.shape[1], n_features=n_features, k=k)
    trainer = SAETrainer(sae, lr=LR, device=device())
    t0 = time.time()
    for _ in range(EPOCHS):
        trainer.train_epoch(train_act, batch_size=BATCH_SIZE, log_every=10 ** 9)
    train_seconds = time.time() - t0
    del train_act

    val_idx = subsample_indices(act_mm.shape[0], VAL_HOLDOUT, data_seed + 999_983)
    x_val = torch.tensor(np.asarray(act_mm[val_idx], dtype=np.float32) - act_mean[None, :])
    sae.eval().to("cpu")
    with torch.no_grad():
        x_hat, h_sparse, _ = sae(x_val)
        total_var = x_val.var(dim=0).sum().item()
        resid_var = (x_val - x_hat).var(dim=0).sum().item()
        var_explained = 1.0 - resid_var / max(total_var, 1e-10)
        act_freq = (h_sparse > 0).float().mean(dim=0).numpy()
    dead = int((act_freq == 0).sum())

    W = sae.W_dec.weight.detach().numpy()
    W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, n_features, size=6000)
    ib = rng.integers(0, n_features, size=6000)
    m = ia != ib
    coh = float(np.abs((W[:, ia[m]] * W[:, ib[m]]).sum(axis=0)).mean())

    return sae, W, {
        "seed": seed, "data_seed": data_seed,
        "variance_explained": float(var_explained),
        "dead_features": dead,
        "alive_features": int(n_features - dead),
        "mean_abs_cos_decoder": coh,
        "training_seconds": round(train_seconds, 1),
    }


def top_genes_per_feature(sae, act_mm, act_mean, gene_ids, token_to_gene, seed, n_top=20):
    idx = subsample_indices(act_mm.shape[0], TOPGENE_SUBSAMPLE, seed + 31)
    mean_t = torch.tensor(act_mean)
    n_features = sae.n_features
    tokens = np.asarray(gene_ids[idx])
    uniq = np.unique(tokens)
    tok_to_row = {int(t): i for i, t in enumerate(uniq)}
    sums = np.zeros((len(uniq), n_features), dtype=np.float32)
    counts = np.zeros(len(uniq), dtype=np.int64)

    for start in range(0, len(idx), 8192):
        chunk = np.asarray(act_mm[idx[start:start + 8192]], dtype=np.float32)
        with torch.no_grad():
            h, _ = sae.encode(torch.tensor(chunk) - mean_t)
        h = h.numpy()
        rows = np.array([tok_to_row[int(t)] for t in tokens[start:start + len(chunk)]])
        np.add.at(sums, rows, h)
        np.add.at(counts, rows, 1)
    means = sums / np.maximum(counts[:, None], 1)

    out = {}
    for f in range(n_features):
        order = np.argsort(-means[:, f])[:n_top]
        genes = [token_to_gene.get(int(uniq[r])) for r in order]
        out[f] = [g for g in genes if g]
    return out


def annotation_rate(top_genes, ontologies, sample_features, seed):
    from scipy.stats import fisher_exact
    rng = np.random.default_rng(seed)
    keys = rng.choice(sorted(top_genes), size=min(sample_features, len(top_genes)),
                      replace=False)
    annotated = 0
    for f in keys:
        genes = {g.upper() for g in top_genes[int(f)]}
        pvals = []
        for term_map in ontologies.values():
            for members in term_map.values():
                overlap = len(genes & members)
                if overlap == 0:
                    continue
                table = [[overlap, len(genes) - overlap],
                         [len(members) - overlap,
                          20000 - len(genes) - len(members) + overlap]]
                pvals.append(fisher_exact(table, alternative="greater")[1])
        if pvals and (common.bh_fdr(pvals) < 0.05).any():
            annotated += 1
    return annotated / len(keys), len(keys)


def module_count(sae, act_mm, act_mean, seed):
    import igraph as ig
    import leidenalg
    idx = subsample_indices(act_mm.shape[0], MODULE_SUBSAMPLE, seed + 17)
    mean_t = torch.tensor(act_mean)
    k = sae.k
    n_features = sae.n_features
    top_idx = np.zeros((len(idx), k), dtype=np.int32)
    for start in range(0, len(idx), 8192):
        chunk = np.asarray(act_mm[idx[start:start + 8192]], dtype=np.float32)
        with torch.no_grad():
            _, ti = sae.encode(torch.tensor(chunk) - mean_t)
        top_idx[start:start + len(chunk)] = ti.numpy()

    feat_count = np.bincount(top_idx.ravel(), minlength=n_features)
    coact = np.zeros((n_features, n_features), dtype=np.int32)
    ii, jj = np.triu_indices(k, k=1)
    for row in top_idx:
        r = np.sort(row)
        coact[r[ii], r[jj]] += 1
    rows, cols = np.where(coact >= MIN_COACT)
    keep = rows < cols
    rows, cols = rows[keep], cols[keep]
    if len(rows) == 0:
        return 0
    n_pos = len(idx)
    p_i = feat_count / n_pos
    joint = coact[rows, cols] / n_pos
    pmi = np.log2(joint / np.maximum(p_i[rows] * p_i[cols], 1e-15))
    sel = pmi > PMI_THRESHOLD
    if sel.sum() == 0:
        return 0
    g = ig.Graph(n=n_features, edges=list(zip(rows[sel].tolist(), cols[sel].tolist())))
    g.es["weight"] = pmi[sel].tolist()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    weights="weight", resolution_parameter=1.0, seed=0)
    sizes = np.bincount(part.membership)
    return int((sizes >= 3).sum())


def match_dictionaries(Wa, Wb, chunk=512):
    """Greedy one-to-one matching by decoder cosine, plus the unconstrained best match."""
    best = np.zeros(Wa.shape[1], dtype=np.float32)
    best_j = np.zeros(Wa.shape[1], dtype=np.int64)
    for start in range(0, Wa.shape[1], chunk):
        C = np.abs(Wa[:, start:start + chunk].T @ Wb)
        best[start:start + C.shape[0]] = C.max(axis=1)
        best_j[start:start + C.shape[0]] = C.argmax(axis=1)
    order = np.argsort(-best)
    used = np.zeros(Wb.shape[1], dtype=bool)
    matched = np.zeros(Wa.shape[1], dtype=np.float32)
    for i in order:
        j = best_j[i]
        if not used[j]:
            used[j] = True
            matched[i] = best[i]
        else:
            C = np.abs(Wa[:, i] @ Wb)
            C[used] = -1
            jj = int(C.argmax())
            if C[jj] > 0:
                used[jj] = True
                matched[i] = C[jj]
    return best, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 5, 11, 17])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--extra-data-seed", type=int, default=7,
                    help="one additional run per layer with a different training subsample")
    ap.add_argument("--annotate-sample", type=int, default=400)
    ap.add_argument("--skip-annotation", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ontologies = {}
    if not args.skip_annotation:
        for name, fn in (("GO_BP", "go_bp_gene_sets.json"), ("KEGG", "kegg_gene_sets.json"),
                         ("Reactome", "reactome_gene_sets.json")):
            raw = json.load(open(ONTOLOGY_DIR / fn))
            ontologies[name] = {k: {g.upper() for g in v} for k, v in raw.items()
                                if 5 <= len(v) <= 500}

    with open(common.PHASE1 / "token_id_to_gene_name.json") as fh:
        token_to_gene = {int(k): v for k, v in json.load(fh).items()}

    summary = {}
    for layer in args.layers:
        print(f"\n=== layer {layer} ===", flush=True)
        act_mm = np.lib.format.open_memmap(
            common.PHASE1 / f"layer_{layer:02d}_activations.npy", mode="r")
        gene_ids = np.load(common.PHASE1 / f"layer_{layer:02d}_gene_ids.npy", mmap_mode="r")
        _, act_mean_t, run_dir = common.load_sae(layer)
        act_mean = act_mean_t.numpy()
        n_features = common.EXPANSION * act_mm.shape[1]

        runs, dicts = [], []
        specs = [(s, args.data_seed) for s in args.seeds]
        specs.append((args.seeds[0], args.extra_data_seed))
        for seed, data_seed in specs:
            tag = f"seed{seed}_data{data_seed}"
            wpath = OUT / f"layer{layer:02d}_{tag}_decoder.npy"
            mpath = OUT / f"layer{layer:02d}_{tag}_metrics.json"
            if wpath.exists() and mpath.exists():
                W = np.load(wpath)
                metrics = json.load(open(mpath))
                sae = None
                print(f"  {tag}: cached", flush=True)
            else:
                sae, W, metrics = train_run(layer, seed, data_seed, act_mm, act_mean,
                                            n_features, common.K_VAL)
                if not args.skip_annotation:
                    tg = top_genes_per_feature(sae, act_mm, act_mean, gene_ids,
                                               token_to_gene, seed)
                    rate, n_sampled = annotation_rate(tg, ontologies,
                                                      args.annotate_sample, seed)
                    metrics["annotation_rate"] = float(rate)
                    metrics["annotation_sample"] = int(n_sampled)
                metrics["module_count"] = module_count(sae, act_mm, act_mean, seed)
                np.save(wpath, W)
                json.dump(metrics, open(mpath, "w"), indent=2)
                print(f"  {tag}: varexpl={metrics['variance_explained']:.4f}, "
                      f"dead={metrics['dead_features']}, "
                      f"modules={metrics.get('module_count')}, "
                      f"ann={metrics.get('annotation_rate')} "
                      f"({metrics['training_seconds']}s)", flush=True)
            runs.append(metrics)
            dicts.append(W)

        # cross-run dictionary agreement
        pairs = []
        n_init = len(args.seeds)
        for a in range(len(dicts)):
            for b in range(a + 1, len(dicts)):
                best, matched = match_dictionaries(dicts[a], dicts[b])
                kind = "init" if (a < n_init and b < n_init) else "data"
                pairs.append({
                    "a": runs[a]["seed"], "a_data": runs[a]["data_seed"],
                    "b": runs[b]["seed"], "b_data": runs[b]["data_seed"],
                    "kind": kind,
                    "mean_best_cosine": float(best.mean()),
                    "median_best_cosine": float(np.median(best)),
                    "frac_best_above_0.9": float((best > 0.9).mean()),
                    "frac_best_above_0.7": float((best > 0.7).mean()),
                    "frac_best_above_0.5": float((best > 0.5).mean()),
                    "mean_matched_cosine": float(matched.mean()),
                    "frac_matched_above_0.9": float((matched > 0.9).mean()),
                    "frac_matched_above_0.7": float((matched > 0.7).mean()),
                })
                print(f"    pair {runs[a]['seed']}/{runs[b]['seed']} ({kind}): "
                      f"mean best cos {best.mean():.3f}, "
                      f">0.9: {100 * (best > 0.9).mean():.1f}%", flush=True)

        def spread(key):
            vals = [r[key] for r in runs if key in r]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                    "n": len(vals)}

        summary[str(layer)] = {
            "runs": runs, "pairs": pairs,
            "variance_explained": spread("variance_explained"),
            "dead_features": spread("dead_features"),
            "mean_abs_cos_decoder": spread("mean_abs_cos_decoder"),
            "module_count": spread("module_count"),
            "annotation_rate": spread("annotation_rate"),
            "init_pairs_mean_best_cosine": float(np.mean(
                [p["mean_best_cosine"] for p in pairs if p["kind"] == "init"])),
            "data_pairs_mean_best_cosine": float(np.mean(
                [p["mean_best_cosine"] for p in pairs if p["kind"] == "data"])),
        }
        common.write_json(OUT / "summary.json", summary, seed=args.data_seed)


if __name__ == "__main__":
    main()
