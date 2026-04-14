"""Phase B3 — cross-layer PMI with permutation null + consecutive-layer sweep.

Reviewer 1 (#4) asked for:
  (a) a permutation null for the cross-layer PMI "information highway" claim,
      since TopK forces substantial co-occurrence and a fixed PMI > 3 threshold
      may overcount;
  (b) a consecutive layer-by-layer (L_i -> L_{i+1}) analysis in addition to the
      three long-range pairs L0->L5, L5->L11, L11->L17 used in the preprint.

We use Geneformer K562 activations for all 18 layers and a deterministic
100K-position subsample per layer pair, with 20 permutation rounds to estimate
the 99th-percentile null PMI per layer pair. This is lean enough to finish in
under 30 minutes on the current hardware while giving a stable null estimate.

Outputs (written incrementally after each pair):
  experiments/revision/b3_highways/highway_results.json
  experiments/revision/b3_highways/highway_results.txt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
)
sys.path.insert(0, str(PROJECT / "src"))
from sae_model import TopKSAE  # noqa: E402

PHASE1 = PROJECT / "experiments" / "phase1_k562"
OUT = PROJECT / "experiments" / "revision" / "b3_highways"
OUT.mkdir(parents=True, exist_ok=True)

N_SUBSAMPLE = 100_000
BATCH_SIZE = 8192
N_PERMUTATIONS = 20
PMI_FIXED = 3.0
SEED = 42

LONG_RANGE_PAIRS = [(0, 5), (5, 11), (11, 17)]
CONSECUTIVE_PAIRS = [(i, i + 1) for i in range(17)]


def log(msg: str):
    print(msg, flush=True)


def encode_topk_as_mask(layer: int, idx: np.ndarray) -> np.ndarray:
    """Return (n_positions, n_features) float32 one-hot mask of top-k actives."""
    run_dir = PHASE1 / "sae_models" / f"layer{layer:02d}_x4_k32"
    sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device="cpu")
    sae.eval()
    mean = np.load(run_dir / "activation_mean.npy")
    mean_t = torch.tensor(mean, dtype=torch.float32)

    act = np.lib.format.open_memmap(
        str(PHASE1 / f"layer_{layer:02d}_activations.npy"), mode="r"
    )
    n = idx.shape[0]
    n_features = sae.n_features
    k = sae.k

    mask = np.zeros((n, n_features), dtype=np.float32)
    row = np.arange(BATCH_SIZE)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        slc = idx[start:end]
        chunk = np.ascontiguousarray(act[slc])
        batch = torch.tensor(chunk, dtype=torch.float32) - mean_t
        with torch.no_grad():
            _, top_i = sae.encode(batch)
        top_np = top_i.numpy()  # (b, k)
        b = top_np.shape[0]
        rows = np.repeat(np.arange(start, start + b), k)
        cols = top_np.ravel()
        mask[rows, cols] = 1.0
    return mask, int(n_features)


def pmi_highway_fraction(joint: np.ndarray, n_pos: int, p_src: np.ndarray, p_tgt: np.ndarray, tau: float) -> dict:
    """Compute highway fraction from a joint count matrix."""
    p_joint = joint / n_pos
    outer = np.outer(p_src, p_tgt) + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(np.where(p_joint > 0, p_joint / outer, 1e-12))
    pmi = np.where(p_joint > 0, pmi, -np.inf)
    row_max = pmi.max(axis=1)
    highway = (row_max > tau).sum()
    finite_flat = pmi[np.isfinite(pmi)]
    return {
        "pct_highway": float(highway / pmi.shape[0]),
        "p99_finite_pmi": float(np.percentile(finite_flat, 99)) if finite_flat.size else 0.0,
        "mean_max_pmi": float(row_max[np.isfinite(row_max)].mean()) if np.isfinite(row_max).any() else 0.0,
    }


def run_pair(src_layer: int, tgt_layer: int, idx: np.ndarray, n_perm: int) -> dict:
    t0 = time.time()
    log(f"[L{src_layer:02d}->L{tgt_layer:02d}] encoding masks")
    src_mask, n_feat_src = encode_topk_as_mask(src_layer, idx)
    tgt_mask, n_feat_tgt = encode_topk_as_mask(tgt_layer, idx)
    n_pos = src_mask.shape[0]

    # Marginal frequencies (per feature)
    p_src = src_mask.mean(axis=0)
    p_tgt = tgt_mask.mean(axis=0)

    # Observed joint
    log(f"[L{src_layer:02d}->L{tgt_layer:02d}] observed matmul")
    joint = src_mask.T @ tgt_mask
    obs_fixed = pmi_highway_fraction(joint, n_pos, p_src, p_tgt, PMI_FIXED)

    # Permutation null: 99th percentile of PMI across permutations
    log(f"[L{src_layer:02d}->L{tgt_layer:02d}] {n_perm} permutations")
    rng = np.random.default_rng(SEED)
    null_p99s = []
    for pi in range(n_perm):
        perm = rng.permutation(n_pos)
        joint_p = src_mask.T @ tgt_mask[perm]
        # 99th percentile of PMI under null
        p_joint_p = joint_p / n_pos
        outer_p = np.outer(p_src, p_tgt) + 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi_p = np.log2(np.where(p_joint_p > 0, p_joint_p / outer_p, 1e-12))
        pmi_p = np.where(p_joint_p > 0, pmi_p, -np.inf)
        finite = pmi_p[np.isfinite(pmi_p)]
        if finite.size:
            null_p99s.append(float(np.percentile(finite, 99)))

    tau_null = float(np.max(null_p99s)) if null_p99s else 0.0
    tau_eff = max(PMI_FIXED, tau_null)

    obs_corrected = pmi_highway_fraction(joint, n_pos, p_src, p_tgt, tau_eff)

    return {
        "src_layer": src_layer,
        "tgt_layer": tgt_layer,
        "n_positions": n_pos,
        "n_permutations": n_perm,
        "pmi_fixed_threshold": PMI_FIXED,
        "tau_null_p99_max": tau_null,
        "tau_effective": tau_eff,
        "fixed_threshold_highway": obs_fixed,
        "null_corrected_highway": obs_corrected,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    act0 = np.lib.format.open_memmap(
        str(PHASE1 / "layer_00_activations.npy"), mode="r"
    )
    n_total = act0.shape[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=N_SUBSAMPLE, replace=False))

    all_pairs = LONG_RANGE_PAIRS + CONSECUTIVE_PAIRS
    seen = set()
    pairs_to_run = []
    for p in all_pairs:
        if p not in seen:
            seen.add(p)
            pairs_to_run.append(p)
    log(f"Running {len(pairs_to_run)} pairs")

    results = []
    for s, t in pairs_to_run:
        r = run_pair(s, t, idx, N_PERMUTATIONS)
        results.append(r)
        log(
            f"  done L{s:02d}->L{t:02d}: "
            f"fixed3.0={r['fixed_threshold_highway']['pct_highway']*100:.2f}% | "
            f"tau_eff={r['tau_effective']:.2f} | "
            f"null-corr={r['null_corrected_highway']['pct_highway']*100:.2f}% "
            f"({r['seconds']}s)"
        )
        # Write incrementally
        (OUT / "highway_results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "SrcL | TgtL | fixed3.0 highway% | tau_null_p99_max | tau_eff | null-corrected highway%"
    ]
    for r in results:
        lines.append(
            f"{r['src_layer']:>4d} | {r['tgt_layer']:>4d} | "
            f"{r['fixed_threshold_highway']['pct_highway']*100:>16.2f}% | "
            f"{r['tau_null_p99_max']:>16.2f} | "
            f"{r['tau_effective']:>7.2f} | "
            f"{r['null_corrected_highway']['pct_highway']*100:>22.2f}%"
        )
    (OUT / "highway_results.txt").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
