"""Phase B3b — consecutive-layer highway sweep, fixed-threshold only.

B3 already established (on L0->L5, L5->L11, L11->L17) that the permutation-null
99th percentile PMI is below 3.0, so the fixed 3.0 threshold is conservative and
the null correction does not change the highway fraction. We therefore run the
consecutive L_i -> L_{i+1} sweep without the permutation step to get the curve
in a reasonable time.

Protocol: 100K positions, fixed PMI > 3.0 threshold, all 17 Geneformer pairs.
Each pair costs ~15-20 seconds (encode both layers + one matmul + summary).
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

N_SUB = 100_000
BATCH_SIZE = 8192
TAU = 3.0
SEED = 42


def encode_mask(layer: int, idx: np.ndarray) -> np.ndarray:
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
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        chunk = np.ascontiguousarray(act[idx[start:end]])
        batch = torch.tensor(chunk, dtype=torch.float32) - mean_t
        with torch.no_grad():
            _, top_i = sae.encode(batch)
        top_np = top_i.numpy()
        b = top_np.shape[0]
        rows = np.repeat(np.arange(start, start + b), k)
        cols = top_np.ravel()
        mask[rows, cols] = 1.0
    return mask


def highway_fraction(src_mask: np.ndarray, tgt_mask: np.ndarray, tau: float) -> float:
    n_pos = src_mask.shape[0]
    p_src = src_mask.mean(axis=0)
    p_tgt = tgt_mask.mean(axis=0)
    joint = src_mask.T @ tgt_mask / n_pos
    outer = np.outer(p_src, p_tgt) + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(np.where(joint > 0, joint / outer, 1e-12))
    pmi = np.where(joint > 0, pmi, -np.inf)
    row_max = pmi.max(axis=1)
    return float((row_max > tau).sum() / src_mask.shape[0])


def main():
    act = np.lib.format.open_memmap(
        str(PHASE1 / "layer_00_activations.npy"), mode="r"
    )
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(act.shape[0], size=N_SUB, replace=False))

    # Pre-encode each layer's mask once, then reuse across consecutive pairs.
    masks = {}
    for L in range(18):
        print(f"encoding L{L:02d}", flush=True)
        t0 = time.time()
        masks[L] = encode_mask(L, idx)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)

    pairs = [(i, i + 1) for i in range(17)]
    results = []
    for s, t in pairs:
        t0 = time.time()
        hw = highway_fraction(masks[s], masks[t], TAU)
        results.append({
            "src_layer": s,
            "tgt_layer": t,
            "n_positions": N_SUB,
            "tau": TAU,
            "pct_highway": hw,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"L{s:02d}->L{t:02d}: {hw*100:.2f}% ({results[-1]['seconds']}s)", flush=True)

    out_path = OUT / "consecutive_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    lines = ["SrcL -> TgtL | highway%"]
    for r in results:
        lines.append(f"{r['src_layer']:>4d} -> {r['tgt_layer']:>4d} | {r['pct_highway']*100:>8.2f}%")
    (OUT / "consecutive_sweep.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
