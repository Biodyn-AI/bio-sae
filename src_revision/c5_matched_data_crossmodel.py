"""Phase C5 — matched-input Geneformer vs. scGPT comparison.

Reviewer 3 flagged that the cross-model comparison in main text Figure 3
(K562 Geneformer vs. Tabula Sapiens scGPT) conflates architecture with input
distribution. As a partial control we encode the same 3{,}000-cell Tabula
Sapiens subset through Geneformer and re-measure variance explained at layers
{0, 5, 11} using the *existing* K562-trained Geneformer SAEs (no retraining).
The Geneformer K562 activations of the same layers give the original numbers.

The activations are already cached at
    experiments/phase3_multitissue/ts_activations/layer_*_activations.npy
so this script only needs to load them, pass them through the K562 SAEs, and
compute variance explained + alive-feature count.

Outputs:
  experiments/revision/c5_matched/matched_comparison.json
  experiments/revision/c5_matched/matched_comparison.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
)
sys.path.insert(0, str(PROJECT / "src"))
from sae_model import TopKSAE  # noqa: E402

PHASE1 = PROJECT / "experiments" / "phase1_k562"
PHASE3 = PROJECT / "experiments" / "phase3_multitissue"
OUT = PROJECT / "experiments" / "revision" / "c5_matched"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 5, 11]
VAL_SUBSAMPLE = 200_000
BATCH_SIZE = 8192
SEED = 42


def eval_sae(sae, mean: np.ndarray, act_path: Path, idx: np.ndarray) -> dict:
    act = np.lib.format.open_memmap(str(act_path), mode="r")
    mean_t = torch.tensor(mean, dtype=torch.float32)

    total_var_accum = 0.0
    resid_var_accum = 0.0
    n_batches = 0
    alive = np.zeros(sae.n_features, dtype=np.int64)

    mse_accum = 0.0
    for start in range(0, idx.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, idx.shape[0])
        sl = idx[start:end]
        chunk = np.ascontiguousarray(act[sl])
        batch = torch.tensor(chunk, dtype=torch.float32) - mean_t
        with torch.no_grad():
            x_hat, h_sparse, _ = sae(batch)
            resid = batch - x_hat
            mse_accum += float(torch.mean(resid ** 2).item())
            total_var_accum += float(batch.var(dim=0).sum().item())
            resid_var_accum += float(resid.var(dim=0).sum().item())
            alive += (h_sparse > 0).sum(dim=0).cpu().numpy().astype(np.int64)
        n_batches += 1

    var_explained = 1.0 - (resid_var_accum / n_batches) / max(total_var_accum / n_batches, 1e-10)
    alive_count = int((alive > 0).sum())
    return {
        "n_positions": int(idx.shape[0]),
        "var_explained": float(var_explained),
        "alive_features": alive_count,
        "dead_features": int(sae.n_features - alive_count),
    }


def main():
    summary = []
    rng = np.random.default_rng(SEED)

    for L in LAYERS:
        print(f"[L{L:02d}]")
        run_dir = PHASE1 / "sae_models" / f"layer{L:02d}_x4_k32"
        sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device="cpu")
        sae.eval()
        mean = np.load(run_dir / "activation_mean.npy")

        # K562 baseline
        k562_path = PHASE1 / f"layer_{L:02d}_activations.npy"
        k562_act = np.lib.format.open_memmap(str(k562_path), mode="r")
        k562_idx = np.sort(
            rng.choice(k562_act.shape[0], size=VAL_SUBSAMPLE, replace=False)
        )
        k562_metrics = eval_sae(sae, mean, k562_path, k562_idx)

        # Tabula Sapiens matched input
        ts_path = PHASE3 / "ts_activations" / f"layer_{L:02d}_activations.npy"
        ts_act = np.lib.format.open_memmap(str(ts_path), mode="r")
        ts_idx = np.sort(rng.choice(ts_act.shape[0], size=VAL_SUBSAMPLE, replace=False))
        ts_metrics = eval_sae(sae, mean, ts_path, ts_idx)

        summary.append({
            "layer": L,
            "k562": k562_metrics,
            "tabula_sapiens_matched": ts_metrics,
        })
        print(
            f"  K562            var={k562_metrics['var_explained']:.3f}, "
            f"alive={k562_metrics['alive_features']}"
        )
        print(
            f"  Tabula Sapiens  var={ts_metrics['var_explained']:.3f}, "
            f"alive={ts_metrics['alive_features']}"
        )

    (OUT / "matched_comparison.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "Layer | K562 var | K562 alive | TS-matched var | TS-matched alive | delta_var"
    ]
    for r in summary:
        lines.append(
            f"{r['layer']:>5d} | "
            f"{r['k562']['var_explained']:>8.3f} | "
            f"{r['k562']['alive_features']:>10d} | "
            f"{r['tabula_sapiens_matched']['var_explained']:>14.3f} | "
            f"{r['tabula_sapiens_matched']['alive_features']:>16d} | "
            f"{r['tabula_sapiens_matched']['var_explained']-r['k562']['var_explained']:>+8.3f}"
        )
    (OUT / "matched_comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
