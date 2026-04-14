"""Phase B4 — unannotated-feature characterization.

Reviewer 1 (#5) correctly flagged that guilt-by-association against modules that
cover 96--99.5% of all features is nearly tautological. We replace that test
with two non-tautological ones at Geneformer layers {0, 5, 11, 17}:

  (i)  Cell-type specificity: fraction of unannotated features that are
       significantly enriched for >=1 Tabula Sapiens cell type (Fisher FDR<0.05).
       Cached output from phase3_multitissue/celltype_enrichments.

  (ii) Perturbation responsiveness: fraction of unannotated features with a
       significant response (|effect| > 0.5) to >=1 of the 100 Replogle CRISPRi
       targets. Cached output from perturbation_response/*.json, which records
       the top-20 responding features per target.

Each observed fraction is compared against a matched random-permutation null in
which feature identities are shuffled: we draw random feature subsets of the
same size and compute the fraction that pass the same test. This is intended
as a sanity check, not as a precise p-value.

Outputs:
  experiments/revision/b4_unannotated/unannotated_tests.json
  experiments/revision/b4_unannotated/unannotated_tests.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map")
PHASE1 = PROJECT / "experiments" / "phase1_k562"
PHASE3 = PROJECT / "experiments" / "phase3_multitissue"
OUT = PROJECT / "experiments" / "revision" / "b4_unannotated"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 5, 11, 17]
EFFECT_THRESHOLD = 0.5
SEED = 42


def load_annotated_ids(layer: int) -> set:
    p = PHASE1 / "sae_models" / f"layer{layer:02d}_x4_k32" / "feature_annotations.json"
    with p.open() as f:
        d = json.load(f)
    fa = d["feature_annotations"]
    return {int(k) for k, v in fa.items() if v}


def load_n_features(layer: int) -> int:
    p = PHASE1 / "sae_models" / f"layer{layer:02d}_x4_k32" / "results.json"
    with p.open() as f:
        d = json.load(f)
    return int(d.get("n_features", d.get("alive_features", 4608)))


def load_celltype_features(layer: int) -> set:
    """Set of features that are enriched for >=1 cell type in Tabula Sapiens."""
    p = PHASE3 / "celltype_enrichments" / f"celltype_enrichment_layer{layer:02d}.json"
    if not p.exists():
        return set()
    with p.open() as f:
        d = json.load(f)
    feats = d.get("features", {})
    hits = set()
    for fid_str, fv in feats.items():
        if fv.get("cell_types"):
            hits.add(int(fid_str))
    return hits


def load_perturbation_features(layer: int) -> set:
    """Set of features that respond to >=1 perturbation target.

    We use the cached target_results[*].top_changed_features lists. This is a
    conservative definition: it only counts features that made the per-target
    top-20 responding list.
    """
    p = PHASE1 / "perturbation_response" / f"perturbation_response_layer{layer:02d}.json"
    if not p.exists():
        p = PHASE3 / "perturbation_response" / f"perturbation_response_layer{layer:02d}.json"
    if not p.exists():
        return set()
    with p.open() as f:
        d = json.load(f)
    targets = d.get("target_results", [])
    hits = set()
    for t in targets:
        for tc in t.get("top_changed_features", []):
            if abs(tc.get("effect_size", 0.0)) >= EFFECT_THRESHOLD:
                hits.add(int(tc["feature_idx"]))
    return hits


def matched_null_fraction(
    all_ids: np.ndarray,
    hit_ids: set,
    sample_size: int,
    n_iter: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Expected fraction of a random sample of size `sample_size` drawn from
    all_ids that lands in `hit_ids`. Reports the mean over n_iter draws."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    if sample_size == 0:
        return 0.0
    hit_arr = np.array(list(hit_ids), dtype=np.int64)
    hit_mask = np.zeros(all_ids.max() + 1, dtype=bool)
    hit_mask[hit_arr] = True
    draws = []
    for _ in range(n_iter):
        draw = rng.choice(all_ids, size=sample_size, replace=False)
        draws.append(float(hit_mask[draw].mean()))
    return float(np.mean(draws))


def main():
    rng = np.random.default_rng(SEED)
    per_layer = []
    for L in LAYERS:
        n_features = load_n_features(L)
        all_ids = np.arange(n_features, dtype=np.int64)
        ann_ids = load_annotated_ids(L)
        unann_ids = set(all_ids.tolist()) - ann_ids
        ct_hits = load_celltype_features(L)
        pert_hits = load_perturbation_features(L)

        unann_ct = len(unann_ids & ct_hits)
        ann_ct = len(ann_ids & ct_hits)
        unann_pert = len(unann_ids & pert_hits)
        ann_pert = len(ann_ids & pert_hits)

        ct_unann_frac = unann_ct / max(len(unann_ids), 1)
        ct_ann_frac = ann_ct / max(len(ann_ids), 1)
        pert_unann_frac = unann_pert / max(len(unann_ids), 1)
        pert_ann_frac = ann_pert / max(len(ann_ids), 1)

        null_ct = matched_null_fraction(
            all_ids, ct_hits, sample_size=len(unann_ids), rng=rng
        )
        null_pert = matched_null_fraction(
            all_ids, pert_hits, sample_size=len(unann_ids), rng=rng
        )

        per_layer.append({
            "layer": L,
            "n_features": int(n_features),
            "n_annotated": len(ann_ids),
            "n_unannotated": len(unann_ids),
            "celltype_hits_total": len(ct_hits),
            "celltype_hits_unannotated": unann_ct,
            "celltype_hits_annotated": ann_ct,
            "celltype_frac_unannotated": ct_unann_frac,
            "celltype_frac_annotated": ct_ann_frac,
            "celltype_null_frac_matched": null_ct,
            "pert_hits_total": len(pert_hits),
            "pert_hits_unannotated": unann_pert,
            "pert_hits_annotated": ann_pert,
            "pert_frac_unannotated": pert_unann_frac,
            "pert_frac_annotated": pert_ann_frac,
            "pert_null_frac_matched": null_pert,
        })

    with (OUT / "unannotated_tests.json").open("w") as f:
        json.dump(per_layer, f, indent=2)

    lines = [
        "Layer | #unann | ct% unann | ct% ann | ct% null | pert% unann | pert% ann | pert% null"
    ]
    for r in per_layer:
        lines.append(
            f"{r['layer']:>5d} | {r['n_unannotated']:>6d} | "
            f"{r['celltype_frac_unannotated']*100:>8.2f}% | "
            f"{r['celltype_frac_annotated']*100:>6.2f}% | "
            f"{r['celltype_null_frac_matched']*100:>7.2f}% | "
            f"{r['pert_frac_unannotated']*100:>10.2f}% | "
            f"{r['pert_frac_annotated']*100:>8.2f}% | "
            f"{r['pert_null_frac_matched']*100:>9.2f}%"
        )
    (OUT / "unannotated_tests.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
