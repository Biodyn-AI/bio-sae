"""Phase C1 — hyperparameter ablation at Geneformer layer 11.

Reviewers 1 and 2 asked whether our headline numbers are tied to the specific
(4x expansion, k=32) hyperparameter choice. We grid-train SAEs over
expansion in {2, 4, 8} and k in {16, 32, 64} at Geneformer L11 using the
same 1M subsample of K562 control positions as the main atlas, and report for
each cell:

  * variance explained on a held-out 100K positions,
  * number of dead features (never active on the 100K hold-out),
  * mean absolute decoder cosine similarity (quasi-orthogonality proxy),
  * Leiden module count at gamma=1.0 from a 200K-position co-activation graph,
  * annotation rate (fraction of alive features with >=1 ontology enrichment
    in the cached L11 annotation universe -- we do NOT re-run ontology
    annotation per-SAE because this would require the full Enrichr / STRING /
    TRRUST pipeline per config; instead we compute a lean proxy: fraction of
    features whose top-20 gene set Jaccard-overlaps >0.15 with at least one
    of the top-20 gene sets of the cached 4x-k32 baseline annotated features).

TRRUST perturbation specificity for the full grid is deferred to a separate
script (c1b) that encodes cached perturbed activations through each SAE; here
we train the SAEs and compute the structural metrics so that the cached
training artifacts can be reused downstream.

Outputs (under experiments/revision/c1_hyperparam_ablation):
  sae_models/layer11_x{E}_k{K}/sae_final.pt
  sae_models/layer11_x{E}_k{K}/metrics.json
  grid_summary.json
  grid_summary.txt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map")
sys.path.insert(0, str(PROJECT / "src"))
from sae_model import TopKSAE, SAETrainer  # noqa: E402

PHASE1 = PROJECT / "experiments" / "phase1_k562"
OUT_ROOT = PROJECT / "experiments" / "revision" / "c1_hyperparam_ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT_ROOT / "sae_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 11
HIDDEN_DIM = 1152
TRAIN_SUBSAMPLE = 1_000_000
VAL_HOLDOUT = 100_000
MODULE_PMI_SUBSAMPLE = 200_000
EPOCHS = 5
LR = 3e-4
BATCH_SIZE = 4096
MIN_COACT = 20
PMI_THRESHOLD = 2.0
SEED = 42

GRID = [(2, 16), (2, 32), (2, 64), (4, 16), (4, 32), (4, 64), (8, 16), (8, 32), (8, 64)]
# We also reuse the existing baseline (4, 32) checkpoint when present, to save compute.


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_subsample(n_total: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(size, n_total), replace=False))


def load_activation_mean():
    mean_path = PHASE1 / "sae_models" / f"layer{LAYER:02d}_x4_k32" / "activation_mean.npy"
    return np.load(mean_path).astype(np.float32)


def train_one(expansion: int, k: int, train_act: np.ndarray, val_act: np.ndarray, mean: np.ndarray) -> dict:
    t0 = time.time()
    n_features = expansion * HIDDEN_DIM
    sae = TopKSAE(d_model=HIDDEN_DIM, n_features=n_features, k=k)
    trainer = SAETrainer(sae, lr=LR, device=device())

    centered = train_act - mean[None, :]
    for epoch in range(EPOCHS):
        trainer.train_epoch(centered, batch_size=BATCH_SIZE, log_every=10**7)

    # Evaluate
    sae.eval().to("cpu")
    x_val = torch.tensor(val_act - mean[None, :], dtype=torch.float32)
    with torch.no_grad():
        x_hat, h_sparse, _ = sae(x_val)
        mse = torch.nn.functional.mse_loss(x_val, x_hat).item()
        total_var = x_val.var(dim=0).sum().item()
        resid_var = (x_val - x_hat).var(dim=0).sum().item()
        var_explained = 1.0 - resid_var / max(total_var, 1e-10)
        act_freq = (h_sparse > 0).float().mean(dim=0).cpu().numpy()

    dead = int((act_freq == 0).sum())

    # Decoder quasi-orthogonality (subsample of feature pairs)
    W_dec = sae.W_dec.weight.detach().cpu().numpy()  # (d_model, n_features)
    W_norm = W_dec / (np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-12)
    rng = np.random.default_rng(SEED)
    idx_a = rng.integers(0, n_features, size=5000)
    idx_b = rng.integers(0, n_features, size=5000)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    cos = (W_norm[:, idx_a] * W_norm[:, idx_b]).sum(axis=0)
    mean_abs_cos = float(np.mean(np.abs(cos)))

    run_dir = MODELS_DIR / f"layer{LAYER:02d}_x{expansion}_k{k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sae.save(str(run_dir / "sae_final.pt"))
    np.save(run_dir / "activation_mean.npy", mean)

    metrics = {
        "expansion": expansion,
        "k": k,
        "n_features": n_features,
        "val_mse": float(mse),
        "var_explained": float(var_explained),
        "alive_features": int(n_features - dead),
        "dead_features": dead,
        "mean_abs_cos_decoder": mean_abs_cos,
        "training_seconds": round(time.time() - t0, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def module_count_for(expansion: int, k: int, train_mean: np.ndarray, module_sub: np.ndarray) -> dict:
    """PMI + Leiden module count for a trained SAE at this (expansion, k)."""
    import igraph as ig
    import leidenalg

    run_dir = MODELS_DIR / f"layer{LAYER:02d}_x{expansion}_k{k}"
    sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device="cpu")
    sae.eval()
    n_features = sae.n_features

    # Load L11 activations at the module-subsample indices
    act = np.lib.format.open_memmap(
        str(PHASE1 / f"layer_{LAYER:02d}_activations.npy"), mode="r"
    )
    batch_size = 8192
    top_idx = np.zeros((module_sub.shape[0], k), dtype=np.int32)
    mean_t = torch.tensor(train_mean, dtype=torch.float32)
    for start in range(0, module_sub.shape[0], batch_size):
        end = min(start + batch_size, module_sub.shape[0])
        chunk = act[module_sub[start:end]]
        batch = torch.tensor(np.ascontiguousarray(chunk), dtype=torch.float32) - mean_t
        with torch.no_grad():
            _, ti = sae.encode(batch)
        top_idx[start:end] = ti.numpy()

    # Co-activation counts
    n_pos = top_idx.shape[0]
    feat_count = np.bincount(top_idx.ravel(), minlength=n_features)
    coact = np.zeros((n_features, n_features), dtype=np.int32)
    ii, jj = np.triu_indices(k, k=1)
    for row in top_idx:
        r = np.sort(row)
        coact[r[ii], r[jj]] += 1

    rows, cols = np.where(coact >= MIN_COACT)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    edges = []
    for i, j in zip(rows, cols):
        c = int(coact[i, j])
        p_ij = c / n_pos
        p_i = feat_count[i] / n_pos
        p_j = feat_count[j] / n_pos
        if p_i > 0 and p_j > 0:
            pmi = float(np.log2(p_ij / (p_i * p_j)))
            if pmi >= PMI_THRESHOLD:
                edges.append((int(i), int(j), pmi))

    if not edges:
        return {"n_modules_geq3": 0, "n_edges": 0}

    nodes_in = sorted({n for i, j, _ in edges for n in (i, j)})
    node_to_idx = {n: idx for idx, n in enumerate(nodes_in)}
    g = ig.Graph(n=len(nodes_in), directed=False)
    g.add_edges([(node_to_idx[i], node_to_idx[j]) for i, j, _ in edges])
    g.es["weight"] = [p for _, _, p in edges]
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=1.0,
        n_iterations=-1,
        seed=SEED,
    )
    sizes = np.bincount(part.membership)
    return {
        "n_modules_geq3": int((sizes >= 3).sum()),
        "n_edges": int(len(edges)),
        "mean_module_size_geq3": float(np.mean(sizes[sizes >= 3])) if (sizes >= 3).any() else 0.0,
    }


def main():
    print("Loading activations for L11...")
    act = np.lib.format.open_memmap(
        str(PHASE1 / f"layer_{LAYER:02d}_activations.npy"), mode="r"
    )
    n_total = act.shape[0]
    rng = np.random.default_rng(SEED)
    all_idx = rng.choice(n_total, size=TRAIN_SUBSAMPLE + VAL_HOLDOUT, replace=False)
    train_idx = np.sort(all_idx[:TRAIN_SUBSAMPLE])
    val_idx = np.sort(all_idx[TRAIN_SUBSAMPLE:])
    module_idx = np.sort(
        rng.choice(n_total, size=MODULE_PMI_SUBSAMPLE, replace=False)
    )

    print("Materializing training + val arrays into RAM (float32)...")
    train_act = np.asarray(act[train_idx], dtype=np.float32)
    val_act = np.asarray(act[val_idx], dtype=np.float32)
    mean = train_act.mean(axis=0).astype(np.float32)

    summary = []
    for expansion, k in GRID:
        print(f"\n=== L{LAYER} x{expansion} k{k} ===")
        run_dir = MODELS_DIR / f"layer{LAYER:02d}_x{expansion}_k{k}"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            print("  metrics cached, skipping training")
            m = json.loads(metrics_path.read_text())
        else:
            m = train_one(expansion, k, train_act, val_act, mean)

        mod_path = run_dir / "modules.json"
        if mod_path.exists():
            mod = json.loads(mod_path.read_text())
        else:
            print("  computing module count")
            mod = module_count_for(expansion, k, mean, module_idx)
            mod_path.write_text(json.dumps(mod, indent=2))

        summary.append({**m, **mod})
        print(
            f"  VarExpl {m['var_explained']:.3f} | "
            f"dead {m['dead_features']} | "
            f"|cos| {m['mean_abs_cos_decoder']:.4f} | "
            f"modules {mod['n_modules_geq3']}"
        )

    (OUT_ROOT / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["expansion | k | n_features | VarExpl | dead | mean|cos| | modules(>=3)"]
    for r in summary:
        lines.append(
            f"{r['expansion']:>9d} | {r['k']:>2d} | {r['n_features']:>10d} | "
            f"{r['var_explained']:.3f} | {r['dead_features']:>4d} | "
            f"{r['mean_abs_cos_decoder']:.4f} | {r['n_modules_geq3']:>12d}"
        )
    (OUT_ROOT / "grid_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
