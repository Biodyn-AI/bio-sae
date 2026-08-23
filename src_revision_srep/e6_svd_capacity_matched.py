"""E6 — capacity-matched comparison of SAE features against the SVD basis.

Comparing 4,608 dictionary atoms with 50 singular axes is not a like-for-like
comparison, so "not aligned with the top-50 axes" cannot by itself mean "invisible to
linear decomposition". This script replaces that statement with measured quantities:

  1. Cumulative projection rho_k = ||P_k d_f||^2 of every SAE decoder direction onto the
     rank-k principal subspace, for k = 1 .. d, against the random-direction null k/d.
  2. The rank k* at which truncated SVD matches the SAE's own reconstruction quality
     (reconstruction-matched comparison).
  3. Single-axis alignment counts as a joint function of subspace size k and cosine
     threshold tau, so the headline percentage can be read off at any (k, tau).
  4. Annotation yield per direction: the top-k principal axes annotated with the identical
     top-20-gene enrichment protocol used for SAE features (both polarities), compared
     against equally many SAE features.

Outputs: experiments/revision_srep/E6_svd_capacity/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.OUT_ROOT / "E6_svd_capacity"
ONTOLOGY_DIR = common.BASE / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets"
MIN_TERM_SIZE, MAX_TERM_SIZE = 5, 500
N_BLOCKS = 40
BLOCK_ROWS = 12_500          # 40 x 12,500 = 500,000 sampled positions per layer
TOP_GENES = 20
BACKGROUND = 20_000


# ---------------------------------------------------------------------------

def sample_positions(n_total, seed):
    """Evenly spaced contiguous blocks — sequential reads on a memmap, unbiased in cells."""
    rng = np.random.RandomState(seed)
    starts = np.linspace(0, max(n_total - BLOCK_ROWS, 1), N_BLOCKS).astype(np.int64)
    jitter = rng.randint(0, max(BLOCK_ROWS // 2, 1), size=N_BLOCKS)
    starts = np.clip(starts + jitter, 0, max(n_total - BLOCK_ROWS, 0))
    return sorted(set(starts.tolist()))


def stream_blocks(mm, starts):
    for s in starts:
        yield int(s), np.asarray(mm[s:s + BLOCK_ROWS], dtype=np.float32)


def layer_eigenbasis(layer, seed):
    """Exact eigenbasis of the sampled covariance (all d principal directions)."""
    act_path = common.PHASE1 / f"layer_{layer:02d}_activations.npy"
    mm = np.lib.format.open_memmap(act_path, mode="r")
    starts = sample_positions(mm.shape[0], seed)

    d = mm.shape[1]
    total = np.zeros(d, dtype=np.float64)
    gram = np.zeros((d, d), dtype=np.float64)
    n = 0
    for _, block in stream_blocks(mm, starts):
        total += block.sum(axis=0, dtype=np.float64)
        gram += block.T.astype(np.float64) @ block.astype(np.float64)
        n += block.shape[0]
    mean = total / n
    cov = gram / n - np.outer(mean, mean)
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order].astype(np.float64), eigvecs[:, order].astype(np.float32), mean.astype(np.float32), n, starts


def projection_curves(eigvecs, decoder):
    """rho_k for every decoder direction. decoder: (d, n_features), unit-norm columns."""
    coeff = eigvecs.T.astype(np.float32) @ decoder          # (d, n_features)
    power = coeff ** 2
    return np.cumsum(power, axis=0)                         # (d, n_features)


def alignment_grid(eigvecs, decoder, k_values, taus):
    """Fraction of features whose |cos| with SOME axis in the top-k exceeds tau."""
    coeff = np.abs(eigvecs.T.astype(np.float32) @ decoder)  # (d, n_features)
    running = np.maximum.accumulate(coeff, axis=0)          # max over the first k axes
    out = {}
    for k in k_values:
        best = running[min(k, running.shape[0]) - 1]
        out[str(k)] = {f"{t:g}": float((best > t).mean()) for t in taus}
    return out


def top_genes_for_axes(layer, eigvecs, mean, starts, n_axes, token_to_gene):
    """Mean signed projection per gene token for the leading axes (both polarities)."""
    act_path = common.PHASE1 / f"layer_{layer:02d}_activations.npy"
    mm = np.lib.format.open_memmap(act_path, mode="r")
    gene_ids = np.load(common.PHASE1 / f"layer_{layer:02d}_gene_ids.npy", mmap_mode="r")

    axes = eigvecs[:, :n_axes].astype(np.float32)           # (d, n_axes)
    uniq_tokens = {}
    sums = None
    counts = None
    for s, block in stream_blocks(mm, starts):
        proj = (block - mean) @ axes                        # (rows, n_axes)
        toks = np.asarray(gene_ids[s:s + block.shape[0]])
        for t in np.unique(toks):
            if t not in uniq_tokens:
                uniq_tokens[t] = len(uniq_tokens)
        if sums is None:
            sums = np.zeros((len(uniq_tokens) * 4 + 1024, n_axes), dtype=np.float64)
            counts = np.zeros(sums.shape[0], dtype=np.int64)
        if len(uniq_tokens) >= sums.shape[0]:
            grow = np.zeros((len(uniq_tokens) * 2, n_axes), dtype=np.float64)
            grow[:sums.shape[0]] = sums
            sums = grow
            gc_ = np.zeros(sums.shape[0], dtype=np.int64)
            gc_[:len(counts)] = counts
            counts = gc_
        rows = np.array([uniq_tokens[t] for t in toks], dtype=np.int64)
        np.add.at(sums, rows, proj.astype(np.float64))
        np.add.at(counts, rows, 1)

    idx_to_tok = {v: k for k, v in uniq_tokens.items()}
    n_used = len(uniq_tokens)
    means = sums[:n_used] / np.maximum(counts[:n_used, None], 1)

    axis_genes = {}
    for a in range(n_axes):
        col = means[:, a]
        for sign, label in ((1.0, "+"), (-1.0, "-")):
            order = np.argsort(-sign * col)[:TOP_GENES]
            genes = []
            for r in order:
                g = token_to_gene.get(int(idx_to_tok[int(r)]))
                if g:
                    genes.append(g)
            axis_genes[f"PC{a}{label}"] = genes
    return axis_genes


def load_ontologies():
    onts = {}
    for name, fn in (("GO_BP", "go_bp_gene_sets.json"),
                     ("KEGG", "kegg_gene_sets.json"),
                     ("Reactome", "reactome_gene_sets.json")):
        p = ONTOLOGY_DIR / fn
        if p.exists():
            raw = json.load(open(p))
            onts[name] = {k: set(v) for k, v in raw.items()
                          if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
    return onts


def annotate(gene_sets, ontologies):
    """Identical protocol to the atlas: Fisher's exact per term, BH FDR < 0.05."""
    from scipy.stats import fisher_exact
    results = {}
    for name, genes in gene_sets.items():
        genes = set(g.upper() for g in genes if g)
        hits, pvals, terms = [], [], []
        if genes:
            for ont, term_map in ontologies.items():
                for term, members in term_map.items():
                    members_u = set(m.upper() for m in members)
                    overlap = len(genes & members_u)
                    if overlap == 0:
                        continue
                    table = [[overlap, len(genes) - overlap],
                             [len(members_u) - overlap,
                              BACKGROUND - len(genes) - len(members_u) + overlap]]
                    _, p = fisher_exact(table, alternative="greater")
                    pvals.append(p)
                    terms.append((ont, term, overlap))
        if pvals:
            q = common.bh_fdr(pvals)
            hits = [{"ontology": o, "term": t, "overlap": ov, "fdr": float(qq)}
                    for (o, t, ov), qq in zip(terms, q) if qq < 0.05]
        results[name] = hits
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(18)))
    ap.add_argument("--annotate-layers", type=int, nargs="*", default=[0, 11])
    ap.add_argument("--n-annotate-axes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    k_grid = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 800, 1152]
    taus = [0.3, 0.5, 0.7, 0.9]
    summary = {}

    _, _, var_genes = common.load_replogle_obs()
    tokenizer = common.GeneformerTokenizer(var_genes)
    ontologies = load_ontologies()
    print(f"ontologies: {[(k, len(v)) for k, v in ontologies.items()]}")

    for layer in args.layers:
        t0 = time.time()
        print(f"\n=== layer {layer} ===", flush=True)
        eigvals, eigvecs, mean, n_pos, starts = layer_eigenbasis(layer, args.seed + layer)
        d = len(eigvals)
        var_cum = np.cumsum(eigvals) / eigvals.sum()

        sae, act_mean_t, run_dir = common.load_sae(layer)
        decoder = sae.W_dec.weight.detach().numpy().astype(np.float32)   # (d, n_features)
        decoder = decoder / np.maximum(np.linalg.norm(decoder, axis=0, keepdims=True), 1e-12)

        rho = projection_curves(eigvecs, decoder)            # (d, n_features)
        med = np.median(rho, axis=1)
        q25 = np.percentile(rho, 25, axis=1)
        q75 = np.percentile(rho, 75, axis=1)

        def first_k(curve, thresh):
            idx = np.argmax(curve >= thresh)
            return int(idx + 1) if curve[idx] >= thresh else None

        sae_var = json.load(open(run_dir / "results.json"))["results"]["variance_explained"]
        k_match = first_k(var_cum, sae_var) if sae_var else None

        summary[str(layer)] = {
            "n_positions_sampled": int(n_pos),
            "d": int(d),
            "sae_variance_explained": sae_var,
            "svd_rank_matching_sae_varexpl": k_match,
            "svd_varexpl_at_k": {str(k): float(var_cum[min(k, d) - 1]) for k in k_grid},
            "median_rho_at_k": {str(k): float(med[min(k, d) - 1]) for k in k_grid},
            "q25_rho_at_k": {str(k): float(q25[min(k, d) - 1]) for k in k_grid},
            "q75_rho_at_k": {str(k): float(q75[min(k, d) - 1]) for k in k_grid},
            "random_null_rho_at_k": {str(k): float(min(k, d) / d) for k in k_grid},
            "k_median_rho_50pct": first_k(med, 0.5),
            "k_median_rho_90pct": first_k(med, 0.9),
            "alignment_fraction": alignment_grid(eigvecs, decoder, k_grid, taus),
        }
        np.savez_compressed(OUT / f"rho_curves_layer{layer:02d}.npz",
                            median=med, q25=q25, q75=q75,
                            var_cum=var_cum, eigvals=eigvals)
        print(f"  varexpl(SAE)={sae_var}, SVD rank matching it = {k_match}; "
              f"median rho reaches 50% at k={summary[str(layer)]['k_median_rho_50pct']}, "
              f"90% at k={summary[str(layer)]['k_median_rho_90pct']} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if layer in args.annotate_layers:
            print("  annotating principal axes with the atlas protocol ...", flush=True)
            axis_genes = top_genes_for_axes(layer, eigvecs, mean, starts,
                                            args.n_annotate_axes, tokenizer.token_to_gene)
            axis_hits = annotate(axis_genes, ontologies)

            catalog = common.load_feature_catalog(run_dir)
            rng = np.random.RandomState(args.seed + 1000 + layer)
            feat_ids = rng.choice(sorted(catalog.keys()),
                                  size=min(args.n_annotate_axes * 2, len(catalog)),
                                  replace=False)
            feat_sets = {f"F{int(fi)}": catalog[int(fi)] for fi in feat_ids}
            feat_hits = annotate(feat_sets, ontologies)

            def yield_stats(hits):
                per = [len(v) for v in hits.values()]
                terms = {(h["ontology"], h["term"]) for v in hits.values() for h in v}
                return {"n_directions": len(hits),
                        "frac_annotated": float(np.mean([p > 0 for p in per])) if per else 0.0,
                        "mean_terms_per_direction": float(np.mean(per)) if per else 0.0,
                        "unique_terms": len(terms)}

            pol = {p: {k: v for k, v in axis_hits.items() if k.endswith(p)} for p in "+-"}
            summary[str(layer)]["annotation_matched"] = {
                "svd_axes_positive": yield_stats(pol["+"]),
                "svd_axes_negative": yield_stats(pol["-"]),
                "svd_axes_either": yield_stats(
                    {f"PC{a}": (axis_hits.get(f"PC{a}+", []) + axis_hits.get(f"PC{a}-", []))
                     for a in range(args.n_annotate_axes)}),
                "sae_features_random_sample": yield_stats(feat_hits),
            }
            common.write_json(OUT / f"axis_annotations_layer{layer:02d}.json",
                              {"axis_top_genes": axis_genes, "axis_hits": axis_hits},
                              seed=args.seed)
            print(f"    SVD axes annotated: "
                  f"{summary[str(layer)]['annotation_matched']['svd_axes_either']}")
            print(f"    SAE features annotated: "
                  f"{summary[str(layer)]['annotation_matched']['sae_features_random_sample']}")

    common.write_json(OUT / "summary.json",
                      {"k_grid": k_grid, "taus": taus, "layers": summary}, seed=args.seed)


if __name__ == "__main__":
    main()
