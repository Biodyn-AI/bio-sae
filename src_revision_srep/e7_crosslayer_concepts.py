"""E7 — how many distinct concepts does the atlas contain?

Dictionary atoms from different layers live in different vector spaces, so counting them
across layers and dividing by the width of one space is not a meaningful compression
figure. What can be compared across layers is the *gene-level signature* of each feature,
which lives in the shared gene space. This script therefore

  1. clusters all features from all layers by top-20 gene-set similarity and reports the
     number of distinct gene-level programs, versus the raw dictionary-atom count;
  2. reports the same counts within a single layer, which is the only place a
     "features per representation space" ratio is defined;
  3. measures how much a program discovered at one layer is re-discovered at other
     layers (program reuse), and the best-match Jaccard between adjacent layers;
  4. quantifies quasi-orthogonal packing within one layer: the number of alive atoms,
     their mutual coherence, and the Welch bound for that many unit vectors in d
     dimensions, which is what actually bounds how many near-orthogonal directions a
     d-dimensional space can host.

Outputs: experiments/revision_srep/E7_concepts/
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.OUT_ROOT / "E7_concepts"


def load_all_catalogs(layers, sae_base=None):
    features, meta = [], []
    for layer in layers:
        _, _, run_dir = common.load_sae(layer, sae_base)
        catalog = common.load_feature_catalog(run_dir)
        for fi, genes in sorted(catalog.items()):
            if genes:
                features.append(genes)
                meta.append((layer, fi))
    return features, meta


def build_membership(features):
    vocab = {}
    rows, cols = [], []
    for i, genes in enumerate(features):
        for g in set(genes):
            j = vocab.setdefault(g, len(vocab))
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    M = sp.csr_matrix((data, (rows, cols)), shape=(len(features), len(vocab)))
    return M, vocab


def jaccard_graph(M, min_intersection, min_jaccard, chunk=4000):
    """Edges between features whose top-20 gene sets overlap enough. Sparse product is
    computed in row chunks so the intermediate never materialises in full."""
    sizes = np.asarray(M.sum(axis=1)).ravel()
    n = M.shape[0]
    Mt = M.T.tocsc()
    ei, ej, ew = [], [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        inter = (M[start:end] @ Mt).tocoo()
        keep = (inter.data >= min_intersection) & (inter.row + start < inter.col)
        r = inter.row[keep] + start
        c = inter.col[keep]
        v = inter.data[keep]
        union = sizes[r] + sizes[c] - v
        jac = v / np.maximum(union, 1)
        sel = jac >= min_jaccard
        ei.append(r[sel])
        ej.append(c[sel])
        ew.append(jac[sel])
        print(f"    rows {start}-{end}: {int(sel.sum())} edges", flush=True)
    return (np.concatenate(ei), np.concatenate(ej), np.concatenate(ew)) if ei else \
        (np.array([]), np.array([]), np.array([]))


def leiden_communities(n_nodes, ei, ej, ew, resolution, seed):
    import igraph as ig
    import leidenalg
    g = ig.Graph(n=n_nodes, edges=list(zip(ei.tolist(), ej.tolist())))
    g.es["weight"] = ew.tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=resolution, seed=seed)
    return np.array(part.membership)


def welch_bound(n_vectors, d):
    """Minimum achievable maximum coherence for n unit vectors in R^d."""
    if n_vectors <= d:
        return 0.0
    return float(np.sqrt((n_vectors - d) / (d * (n_vectors - 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(18)))
    ap.add_argument("--min-intersection", type=int, default=5)
    ap.add_argument("--min-jaccard", type=float, default=0.14)
    ap.add_argument("--resolution", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    features, meta = load_all_catalogs(args.layers)
    layers_arr = np.array([m[0] for m in meta])
    print(f"{len(features)} features with top-gene lists across {len(args.layers)} layers "
          f"({time.time() - t0:.0f}s)")

    M, vocab = build_membership(features)
    print(f"gene vocabulary: {len(vocab)}")

    print("building gene-set similarity graph ...", flush=True)
    ei, ej, ew = jaccard_graph(M, args.min_intersection, args.min_jaccard)
    print(f"{len(ei)} edges ({time.time() - t0:.0f}s)")

    summary = {
        "n_features_total": len(features),
        "n_layers": len(args.layers),
        "features_per_layer": {str(l): int((layers_arr == l).sum())
                               for l in args.layers},
        "gene_vocabulary": len(vocab),
        "n_edges": int(len(ei)),
        "min_intersection": args.min_intersection,
        "min_jaccard": args.min_jaccard,
        "exact_duplicate_gene_sets": None,
        "resolutions": {},
    }

    # exact / near-exact duplicate signatures
    sig = Counter(tuple(sorted(set(g))) for g in features)
    dup = {k: v for k, v in sig.items() if v > 1}
    summary["exact_duplicate_gene_sets"] = {
        "n_distinct_signatures": len(sig),
        "n_signatures_shared_by_more_than_one_feature": len(dup),
        "n_features_in_shared_signatures": int(sum(dup.values())),
        "largest_signature_multiplicity": int(max(dup.values())) if dup else 1,
    }

    for res in args.resolution:
        memb = leiden_communities(len(features), ei, ej, ew, res, args.seed)
        n_comm = len(set(memb.tolist()))
        sizes = np.bincount(memb)
        singleton = int((sizes == 1).sum())
        # cross-layer spread of each program
        spread = []
        for c in range(n_comm):
            ls = layers_arr[memb == c]
            if len(ls) > 1:
                spread.append(len(set(ls.tolist())))
        summary["resolutions"][str(res)] = {
            "n_programs": int(n_comm),
            "n_singleton_programs": singleton,
            "n_multi_feature_programs": int(n_comm - singleton),
            "median_program_size": float(np.median(sizes)),
            "largest_program_size": int(sizes.max()),
            "mean_layers_spanned_per_program": float(np.mean(spread)) if spread else 0.0,
            "frac_programs_spanning_ge2_layers": (
                float(np.mean([s >= 2 for s in spread])) if spread else 0.0),
            "atoms_per_program": float(len(features) / n_comm),
        }
        print(f"  resolution {res}: {n_comm} programs "
              f"({summary['resolutions'][str(res)]['atoms_per_program']:.1f} atoms each)",
              flush=True)
        np.save(OUT / f"membership_res{res}.npy", memb)

    # per-layer program counts at the reference resolution
    ref_res = str(args.resolution[len(args.resolution) // 2])
    memb = np.load(OUT / f"membership_res{ref_res}.npy")
    per_layer = {}
    for l in args.layers:
        sel = layers_arr == l
        per_layer[str(l)] = {
            "n_features": int(sel.sum()),
            "n_distinct_programs": int(len(set(memb[sel].tolist()))),
        }
    summary["per_layer_programs_at_reference_resolution"] = per_layer
    summary["reference_resolution"] = ref_res

    # within-layer packing geometry
    packing = {}
    for l in args.layers:
        sae, _, run_dir = common.load_sae(l)
        W = sae.W_dec.weight.detach().numpy()
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
        results = json.load(open(run_dir / "results.json"))["results"]
        alive = int(results["alive_features"])
        rng = np.random.RandomState(args.seed + l)
        idx = rng.choice(W.shape[1], size=min(1500, W.shape[1]), replace=False)
        G = np.abs(W[:, idx].T @ W[:, idx])
        np.fill_diagonal(G, 0.0)
        packing[str(l)] = {
            "d_model": int(W.shape[0]),
            "n_atoms": int(W.shape[1]),
            "n_alive": alive,
            "atoms_per_dimension": float(W.shape[1] / W.shape[0]),
            "mean_abs_coherence": float(G[np.triu_indices_from(G, 1)].mean()),
            "max_abs_coherence_sampled": float(G.max()),
            "welch_bound": welch_bound(W.shape[1], W.shape[0]),
        }
    summary["within_layer_packing"] = packing

    common.write_json(OUT / "summary.json", summary, seed=args.seed)
    np.savez_compressed(OUT / "graph.npz", ei=ei, ej=ej, ew=ew, layers=layers_arr)
    common.write_json(OUT / "feature_index.json",
                      {"meta": [{"layer": int(a), "feature_idx": int(b)}
                                for a, b in meta]}, seed=args.seed)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
