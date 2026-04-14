"""Phase C6 — batch/assay confound analysis for scGPT features.

Reviewer 3 asked whether scGPT SAE features recover batch- or assay-specific
signal, and whether features that look cell-type-selective are actually
tracking donor/assay metadata. The scGPT extraction metadata cached with the
atlas records only tissue and cell_type per cell, not donor or assay. As a
first-pass answer we compute, for each scGPT feature at layers {0, 4, 7, 11}:

  * Mutual information with tissue (3 values: immune / kidney / lung).
  * Mutual information with cell_type (56 values).
  * The ratio MI(tissue) / MI(cell_type).

Interpretation: tissue is a coarse "sample-level" variable. A feature whose
MI is dominated by tissue rather than cell_type is more likely to be capturing
a sampling/preparation effect than a biological cell-identity feature. This is
not a true donor-level batch test -- for that, the Tabula Sapiens source h5ad
with donor IDs would need to be joined with the cell_ids -- but it is a useful
first pass and is what we report in the main text as a caveat.

Outputs:
  experiments/revision/c6_batch/batch_analysis.json
  experiments/revision/c6_batch/batch_analysis.txt
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

SCGPT = PROJECT / "experiments" / "scgpt_atlas"
OUT = PROJECT / "experiments" / "revision" / "c6_batch"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 4, 7, 11]
SUBSAMPLE = 150_000
BATCH_SIZE = 8192
SEED = 42


def mutual_information(feature_active: np.ndarray, labels: np.ndarray) -> float:
    """Binary MI between feature-active and a discrete label."""
    from sklearn.metrics import mutual_info_score
    return float(mutual_info_score(labels, feature_active))


def main():
    meta_path = SCGPT / "activations" / "extraction_metadata.json"
    with meta_path.open() as f:
        meta = json.load(f)
    cell_data = meta["cell_data"]  # one entry per cell_idx (3000)
    cell_tissue = np.array([c["tissue"] for c in cell_data])
    cell_type = np.array([c["cell_type"] for c in cell_data])

    per_layer = []
    for L in LAYERS:
        print(f"[L{L:02d}]")
        run_dir = SCGPT / "sae_models" / f"layer{L:02d}_x4_k32"
        if not (run_dir / "sae_final.pt").exists():
            print(f"  SAE checkpoint missing: {run_dir}")
            continue
        sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device="cpu")
        sae.eval()
        mean = np.load(run_dir / "activation_mean.npy")
        mean_t = torch.tensor(mean, dtype=torch.float32)

        act_path = SCGPT / "activations" / f"layer_{L:02d}_activations.npy"
        cid_path = SCGPT / "activations" / f"layer_{L:02d}_cell_ids.npy"
        act = np.lib.format.open_memmap(str(act_path), mode="r")
        cids = np.load(cid_path)

        rng = np.random.default_rng(SEED + L)
        idx = np.sort(rng.choice(act.shape[0], size=min(SUBSAMPLE, act.shape[0]), replace=False))
        pos_cids = cids[idx]
        pos_tissue = cell_tissue[pos_cids]
        pos_ctype = cell_type[pos_cids]

        n_features = sae.n_features
        # Compute activation indicator for each position
        feat_active = np.zeros((idx.shape[0], n_features), dtype=np.int8)
        for start in range(0, idx.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, idx.shape[0])
            chunk = np.ascontiguousarray(act[idx[start:end]])
            batch = torch.tensor(chunk, dtype=torch.float32) - mean_t
            with torch.no_grad():
                h_sparse, _ = sae.encode(batch)
            feat_active[start:end] = (h_sparse.numpy() > 0).astype(np.int8)

        # Vectorized mutual information between a binary feature and a discrete label.
        # MI(F; L) = sum_{f in {0,1}, l} p(f, l) log( p(f, l) / (p(f) p(l)) )
        # We compute this for all features at once using matrix operations.
        def vectorized_mi(feat_bin: np.ndarray, labels: np.ndarray) -> np.ndarray:
            n_pos, nf = feat_bin.shape
            unique_labels, label_codes = np.unique(labels, return_inverse=True)
            n_labels = len(unique_labels)
            # p(l)
            p_l = np.bincount(label_codes, minlength=n_labels) / n_pos
            # p(f=1): shape (nf,)
            p_f1 = feat_bin.sum(axis=0) / n_pos
            p_f0 = 1.0 - p_f1
            # Joint p(f=1, l): shape (nf, n_labels)
            # One-hot label matrix: (n_pos, n_labels) int8
            lab_onehot = np.zeros((n_pos, n_labels), dtype=np.float32)
            lab_onehot[np.arange(n_pos), label_codes] = 1.0
            # joint_1 = feat_bin (f32) @ lab_onehot = (nf, n_labels) sum of f=1 & l=k
            joint_1 = feat_bin.astype(np.float32).T @ lab_onehot / n_pos  # (nf, n_labels)
            joint_0 = p_l[None, :] - joint_1  # p(f=0, l)

            eps = 1e-12
            # term1: p(f=1, l) * log( p(f=1, l) / (p(f=1) * p(l)) )
            denom_1 = (p_f1[:, None] * p_l[None, :]) + eps
            denom_0 = (p_f0[:, None] * p_l[None, :]) + eps
            t1 = np.where(joint_1 > 0, joint_1 * np.log(joint_1 / denom_1 + eps), 0.0)
            t0 = np.where(joint_0 > 0, joint_0 * np.log(joint_0 / denom_0 + eps), 0.0)
            mi = (t1 + t0).sum(axis=1)
            return mi  # nats

        active_count = feat_active.sum(axis=0)
        mi_tissue = vectorized_mi(feat_active, pos_tissue)
        mi_ctype = vectorized_mi(feat_active, pos_ctype)
        # Zero out features that are all-0 or all-1
        dead_mask = (active_count == 0) | (active_count == idx.shape[0])
        mi_tissue[dead_mask] = 0.0
        mi_ctype[dead_mask] = 0.0

        # Compare
        alive = active_count > 0
        ratio = np.divide(
            mi_tissue, mi_ctype + 1e-12, out=np.zeros_like(mi_tissue), where=mi_ctype > 0
        )
        tissue_dominant = (mi_tissue > mi_ctype) & alive

        per_layer.append({
            "layer": L,
            "n_features": int(n_features),
            "n_alive": int(alive.sum()),
            "mean_mi_tissue": float(mi_tissue[alive].mean()) if alive.any() else 0.0,
            "mean_mi_cell_type": float(mi_ctype[alive].mean()) if alive.any() else 0.0,
            "n_tissue_dominant": int(tissue_dominant.sum()),
            "frac_tissue_dominant": float(tissue_dominant.sum() / max(alive.sum(), 1)),
            "median_ratio_tissue_over_ctype": float(np.median(ratio[alive])) if alive.any() else 0.0,
        })
        r = per_layer[-1]
        print(
            f"  alive={r['n_alive']}, "
            f"mean MI(tissue)={r['mean_mi_tissue']:.4f}, "
            f"mean MI(cell_type)={r['mean_mi_cell_type']:.4f}, "
            f"tissue-dominant={r['n_tissue_dominant']} ({r['frac_tissue_dominant']*100:.1f}%)"
        )

    (OUT / "batch_analysis.json").write_text(json.dumps(per_layer, indent=2))
    lines = [
        "Layer | alive | mean MI(tissue) | mean MI(cell_type) | tissue-dominant | frac tissue-dominant"
    ]
    for r in per_layer:
        lines.append(
            f"{r['layer']:>5d} | {r['n_alive']:>5d} | "
            f"{r['mean_mi_tissue']:>15.4f} | "
            f"{r['mean_mi_cell_type']:>18.4f} | "
            f"{r['n_tissue_dominant']:>15d} | "
            f"{r['frac_tissue_dominant']*100:>19.2f}%"
        )
    (OUT / "batch_analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
