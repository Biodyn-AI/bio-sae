"""Phase B2 — PMI memory logging and Leiden resolution sweep.

Reviewer 2 (major #3) asked for (a) explicit graph dimensions and memory
footprint for the co-activation analysis, and (b) justification of the Leiden
resolution = 1.0 choice. We rebuild the co-activation graph from a 500K-sample
of positions at Geneformer layers 0 and 11 (enough to stabilize PMI estimates
for a memory-bounded sweep), log peak RSS and edge counts, then sweep Leiden
resolution over {0.5, 0.75, 1.0, 1.5, 2.0} and report module counts and
Adjusted Rand Index against the gamma=1.0 baseline.

Outputs:
  experiments/revision/b2_pmi_leiden/pmi_memory.json
  experiments/revision/b2_pmi_leiden/leiden_sweep.json
  experiments/revision/b2_pmi_leiden/leiden_sweep.txt
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

PROJECT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map")
sys.path.insert(0, str(PROJECT / "src"))
from sae_model import TopKSAE  # noqa: E402

PHASE1 = PROJECT / "experiments" / "phase1_k562"
OUT = PROJECT / "experiments" / "revision" / "b2_pmi_leiden"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 11]
N_SUBSAMPLE = 500_000
BATCH_SIZE = 8192
PMI_THRESHOLD = 2.0
MIN_COACT = 20
RESOLUTIONS = [0.5, 0.75, 1.0, 1.5, 2.0]
BASELINE_RES = 1.0
SEED = 42


def encode_positions(layer: int, n_sub: int):
    run_dir = PHASE1 / "sae_models" / f"layer{layer:02d}_x4_k32"
    sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device="cpu")
    sae.eval()
    mean = np.load(run_dir / "activation_mean.npy")
    mean_t = torch.tensor(mean, dtype=torch.float32)

    act = np.lib.format.open_memmap(
        str(PHASE1 / f"layer_{layer:02d}_activations.npy"), mode="r"
    )
    n_total = act.shape[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=min(n_sub, n_total), replace=False))
    n = idx.shape[0]

    n_features = sae.n_features
    k = sae.k
    top_idx = np.zeros((n, k), dtype=np.int32)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        sl = act[idx[start:end]]
        batch = torch.tensor(np.ascontiguousarray(sl), dtype=torch.float32)
        batch = batch - mean_t
        with torch.no_grad():
            _, top_i = sae.encode(batch)
        top_idx[start:end] = top_i.numpy()

    return top_idx, n_features, k


def compute_pmi(top_idx, n_features, n_total, k):
    """Accumulate per-feature and pairwise co-activation counts, return PMI edges."""
    feat_count = np.zeros(n_features, dtype=np.int64)
    for fi in top_idx.ravel():
        feat_count[fi] += 1

    coact = np.zeros((n_features, n_features), dtype=np.int32)
    ii_tpl, jj_tpl = np.triu_indices(k, k=1)
    for row in top_idx:
        r = np.sort(row)
        coact[r[ii_tpl], r[jj_tpl]] += 1

    rows, cols = np.where(coact >= MIN_COACT)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    edges = []
    for i, j in zip(rows, cols):
        count = int(coact[i, j])
        p_ij = count / n_total
        p_i = feat_count[i] / n_total
        p_j = feat_count[j] / n_total
        if p_i == 0 or p_j == 0:
            continue
        pmi = float(np.log2(p_ij / (p_i * p_j)))
        if pmi >= PMI_THRESHOLD:
            edges.append((int(i), int(j), pmi))
    return edges, int((coact > 0).sum() // 2), coact.nbytes


def leiden_sweep(edges, n_features, resolutions):
    import igraph as ig
    import leidenalg

    nodes_in = set()
    for i, j, _ in edges:
        nodes_in.add(i)
        nodes_in.add(j)
    nodes = sorted(nodes_in)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), directed=False)
    g.add_edges([(node_to_idx[i], node_to_idx[j]) for i, j, _ in edges])
    g.es["weight"] = [p for _, _, p in edges]

    results = {}
    memberships = {}
    for gamma in resolutions:
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=gamma,
            n_iterations=-1,
            seed=SEED,
        )
        labels = np.array(part.membership)
        sizes = np.bincount(labels)
        big = sizes[sizes >= 3]
        results[gamma] = {
            "n_communities_total": int(len(sizes)),
            "n_modules_geq3": int(len(big)),
            "mean_module_size": float(np.mean(big)) if len(big) else 0.0,
            "max_module_size": int(big.max()) if len(big) else 0,
            "coverage_geq3": float(big.sum() / max(len(nodes), 1)),
            "coverage_geq3_vs_alive": float(big.sum() / n_features),
        }
        memberships[gamma] = labels
    return results, memberships, len(nodes), len(edges)


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def main():
    proc = psutil.Process(os.getpid())

    pmi_memory = []
    leiden_out = []

    for L in LAYERS:
        print(f"[layer {L:02d}] encoding positions")
        t0 = time.time()
        rss_before = proc.memory_info().rss
        top_idx, n_features, k = encode_positions(L, N_SUBSAMPLE)
        encode_rss = proc.memory_info().rss
        t_enc = time.time() - t0

        print(f"[layer {L:02d}] computing PMI")
        t0 = time.time()
        edges, n_nonzero_pairs, coact_bytes = compute_pmi(
            top_idx, n_features, top_idx.shape[0], k
        )
        pmi_rss = proc.memory_info().rss
        t_pmi = time.time() - t0

        pmi_memory.append({
            "layer": L,
            "n_subsampled_positions": int(top_idx.shape[0]),
            "n_features": int(n_features),
            "topk": int(k),
            "n_nonzero_pairs_total": int(n_nonzero_pairs),
            "n_edges_after_thresholds": int(len(edges)),
            "pmi_threshold": PMI_THRESHOLD,
            "min_coactivation": MIN_COACT,
            "coact_matrix_bytes": int(coact_bytes),
            "coact_matrix_mb": round(coact_bytes / 1e6, 1),
            "rss_before_mb": round(rss_before / 1e6, 1),
            "rss_after_encode_mb": round(encode_rss / 1e6, 1),
            "rss_after_pmi_mb": round(pmi_rss / 1e6, 1),
            "peak_rss_mb": round(pmi_rss / 1e6, 1),
            "encode_seconds": round(t_enc, 1),
            "pmi_seconds": round(t_pmi, 1),
        })

        print(f"[layer {L:02d}] leiden sweep")
        t0 = time.time()
        sweep, memberships, n_nodes_in_g, n_edges_in_g = leiden_sweep(
            edges, n_features, RESOLUTIONS
        )
        baseline = memberships[BASELINE_RES]
        aris = {
            g: adjusted_rand(baseline, memberships[g]) for g in RESOLUTIONS
        }
        leiden_out.append({
            "layer": L,
            "n_nodes_in_graph": int(n_nodes_in_g),
            "n_edges_in_graph": int(n_edges_in_g),
            "n_features_alive": int(n_features),
            "baseline_resolution": BASELINE_RES,
            "sweep": {
                str(g): {
                    **sweep[g],
                    "ari_vs_baseline": aris[g],
                }
                for g in RESOLUTIONS
            },
            "leiden_seconds": round(time.time() - t0, 1),
        })

        del top_idx, edges

    with (OUT / "pmi_memory.json").open("w") as f:
        json.dump(pmi_memory, f, indent=2)
    with (OUT / "leiden_sweep.json").open("w") as f:
        json.dump(leiden_out, f, indent=2)

    lines = ["Layer | resolution | n_modules (>=3) | mean_size | coverage | ARI vs gamma=1.0"]
    for r in leiden_out:
        for g in RESOLUTIONS:
            s = r["sweep"][str(g)]
            lines.append(
                f"{r['layer']:>5d} | {g:>9.2f} | {s['n_modules_geq3']:>15d} | "
                f"{s['mean_module_size']:>9.1f} | {s['coverage_geq3']:>8.2%} | "
                f"{s['ari_vs_baseline']:>+.3f}"
            )
    txt = "\n".join(lines) + "\n"
    (OUT / "leiden_sweep.txt").write_text(txt)
    print(txt)

    print("\n--- PMI memory summary ---")
    for r in pmi_memory:
        print(
            f"L{r['layer']:02d}: {r['n_subsampled_positions']:,} positions, "
            f"{r['n_edges_after_thresholds']:,} edges, "
            f"peak RSS {r['peak_rss_mb']} MB, "
            f"encode {r['encode_seconds']}s, pmi {r['pmi_seconds']}s"
        )


if __name__ == "__main__":
    main()
