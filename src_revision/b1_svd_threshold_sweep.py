"""Phase B1 — SVD threshold sensitivity sweep.

Reviewer 2 (major #2) asked us to justify the cosine > 0.7 threshold used to
classify an SAE feature as ``SVD-aligned''. This script re-runs the comparison
at {0.3, 0.5, 0.7, 0.9} and reports:

  * fraction of alive features classified as SVD-aligned,
  * fraction of ontology enrichments that live exclusively in novel (non-aligned)
    features,
  * annotation rate among aligned vs. non-aligned features.

We sweep all 18 Geneformer layers and emit JSON + a plain-text LaTeX table for
Additional file 1 (Table S-SVD).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map")
sys.path.insert(0, str(PROJECT / "src"))
from sae_model import TopKSAE  # noqa: E402

PHASE1 = PROJECT / "experiments" / "phase1_k562"
OUT = PROJECT / "experiments" / "revision" / "b1_svd_sweep"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
LAYERS = list(range(18))


def per_layer(layer: int) -> dict:
    sae_dir = PHASE1 / "sae_models" / f"layer{layer:02d}_x4_k32"
    svd = np.load(sae_dir / "svd_axes.npy")          # (50, 1152)
    sae = TopKSAE.load(str(sae_dir / "sae_final.pt"), device="cpu")

    # Decoder columns: W_dec.weight has shape (d_model, n_features).
    # Feature direction for feature i is W_dec.weight[:, i].
    W_dec = sae.W_dec.weight.detach().cpu().numpy()   # (d_model, n_features)
    feats = W_dec / (np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-12)
    feats = feats.T                                    # (n_features, d_model)

    svd_norm = svd / (np.linalg.norm(svd, axis=1, keepdims=True) + 1e-12)

    # (n_features, 50) abs cosine
    cos = np.abs(feats @ svd_norm.T)
    max_cos = cos.max(axis=1)

    # Annotation cache
    with (sae_dir / "feature_annotations.json").open() as f:
        ann = json.load(f)
    feat_ann = ann["feature_annotations"]   # dict id -> list of terms
    n_features = W_dec.shape[1]

    # alive mask: features that appear in feat_ann have at least one hit; we
    # also need the full alive set from results.json
    with (sae_dir / "results.json").open() as f:
        results = json.load(f)
    n_alive = results.get("alive_features", results.get("n_alive"))

    # feature id -> number of unique (ontology, term) enrichments
    def count_enrichments(fid: int) -> int:
        terms = feat_ann.get(str(fid), [])
        # unique on (ontology, term) to match Table 1 of main text
        seen = set()
        for t in terms:
            seen.add((t.get("ontology"), t.get("term")))
        return len(seen)

    feat_n_enrich = np.array([count_enrichments(i) for i in range(n_features)])
    feat_annotated = feat_n_enrich > 0

    per_tau = []
    for tau in THRESHOLDS:
        aligned = max_cos > tau
        novel = ~aligned

        n_aligned = int(aligned.sum())
        n_novel = int(novel.sum())
        aligned_ann = int((aligned & feat_annotated).sum())
        novel_ann = int((novel & feat_annotated).sum())

        aligned_enrich = int(feat_n_enrich[aligned].sum())
        novel_enrich = int(feat_n_enrich[novel].sum())
        total_enrich = aligned_enrich + novel_enrich

        per_tau.append({
            "threshold": tau,
            "n_aligned": n_aligned,
            "pct_aligned": n_aligned / n_features,
            "n_novel": n_novel,
            "aligned_annotation_rate": aligned_ann / max(n_aligned, 1),
            "novel_annotation_rate": novel_ann / max(n_novel, 1),
            "aligned_enrichments": aligned_enrich,
            "novel_enrichments": novel_enrich,
            "pct_enrichments_in_novel": novel_enrich / max(total_enrich, 1),
        })

    return {
        "layer": layer,
        "n_features": n_features,
        "n_alive": n_alive,
        "max_cos_quantiles": {
            "p50": float(np.percentile(max_cos, 50)),
            "p90": float(np.percentile(max_cos, 90)),
            "p99": float(np.percentile(max_cos, 99)),
            "max": float(max_cos.max()),
        },
        "per_threshold": per_tau,
    }


def main():
    all_results = []
    for L in LAYERS:
        print(f"[layer {L:02d}] running SVD threshold sweep")
        try:
            all_results.append(per_layer(L))
        except FileNotFoundError as e:
            print(f"  skipped: {e}")

    out_json = OUT / "svd_threshold_sweep.json"
    with out_json.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {out_json}")

    # Summary table (text)
    lines = [
        "Layer | tau=0.3 aligned% | tau=0.5 aligned% | tau=0.7 aligned% | tau=0.9 aligned% | "
        "tau=0.3 %enrich novel | tau=0.5 %enrich novel | tau=0.7 %enrich novel | tau=0.9 %enrich novel"
    ]
    for r in all_results:
        by_tau = {pt["threshold"]: pt for pt in r["per_threshold"]}
        lines.append(
            f"{r['layer']:>5d} | "
            f"{by_tau[0.3]['pct_aligned']*100:6.2f}% | "
            f"{by_tau[0.5]['pct_aligned']*100:6.2f}% | "
            f"{by_tau[0.7]['pct_aligned']*100:6.2f}% | "
            f"{by_tau[0.9]['pct_aligned']*100:6.2f}% | "
            f"{by_tau[0.3]['pct_enrichments_in_novel']*100:6.2f}% | "
            f"{by_tau[0.5]['pct_enrichments_in_novel']*100:6.2f}% | "
            f"{by_tau[0.7]['pct_enrichments_in_novel']*100:6.2f}% | "
            f"{by_tau[0.9]['pct_enrichments_in_novel']*100:6.2f}%"
        )
    (OUT / "svd_threshold_sweep.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
