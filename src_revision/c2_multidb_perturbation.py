"""Phase C2 — multi-database perturbation specificity.

Reviewer 1 (#1) asked whether expanding beyond TRRUST (literature-mined, 48 TFs
in our Replogle panel) would change the 6.2% specificity number. We re-compute
the specificity test using:

  * TRRUST   (baseline, 9,396 edges)
  * DoRothEA (all confidence levels, 276,562 edges)
  * DoRothEA A+B+C only (high-confidence subset)

No new Geneformer forwards are needed: we reuse the cached
perturbation_response_layer11.json, which records the top-20 most-changed SAE
features for each of the 100 CRISPRi targets, along with the cached
feature_catalog.json which gives the top-20 genes per feature. For each
regulatory-database TF with >= min_known_targets known targets and present in
the perturbation panel, we call the feature response "specific" if any
responding feature's top-20 gene set overlaps the TF's known-target set.

Outputs:
  experiments/revision/c2_multidb/specificity_per_db.json
  experiments/revision/c2_multidb/specificity_per_db.txt
"""
import json
import sys
from pathlib import Path
from typing import Dict, Set

PROJECT = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
)
PHASE1 = PROJECT / "experiments" / "phase1_k562"
NETWORKS = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks"
)
OUT = PROJECT / "experiments" / "revision" / "c2_multidb"
OUT.mkdir(parents=True, exist_ok=True)

TRRUST_PATH = NETWORKS / "trrust_human.tsv"
DOROTHEA_PATH = NETWORKS / "dorothea_human.tsv"

EFFECT_THRESHOLD = 0.5
MIN_KNOWN_TARGETS = 5


def load_trrust() -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with TRRUST_PATH.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                tf, target = parts[0], parts[1]
                out.setdefault(tf, set()).add(target)
    return out


def load_dorothea(confidences: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with DOROTHEA_PATH.open() as f:
        header = f.readline().strip().split("\t")
        assert header[:3] == ["source", "target", "confidence"]
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                tf, target, conf = parts[0], parts[1], parts[2]
                if conf in confidences:
                    out.setdefault(tf, set()).add(target)
    return out


def load_feature_top_genes() -> Dict[int, Set[str]]:
    path = PHASE1 / "sae_models" / "layer11_x4_k32" / "feature_catalog.json"
    with path.open() as f:
        cat = json.load(f)
    out: Dict[int, Set[str]] = {}
    for feat in cat["features"]:
        fid = int(feat["feature_idx"])
        top = feat.get("top_genes") or []
        out[fid] = {g["gene_name"] for g in top[:20]}
    return out


UNIVERSE_SIZE = 20_000  # approx. human protein-coding genes


def compute_specificity(
    pert_path: Path,
    targets_db: Dict[str, Set[str]],
    feat_genes: Dict[int, Set[str]],
    universe: Set[str],
) -> dict:
    """Test whether responding features' top-20 genes are significantly enriched
    for each TF's known targets.

    For each TF with enough known targets in the panel:
      1. Collect the union of top-20 genes across all significantly responding
         features (|effect|>EFFECT_THRESHOLD).
      2. Run a hypergeometric test: given universe of size N, K targets in the
         DB, n union-gene-list size, k intersection → p-value.
      3. BH-correct across the panel of TFs.
      4. Call the TF "specific" if FDR<0.05 after correction.

    This is stricter than the original >=1-overlap test used in the main-text
    6.2% number and controls for different database target-set sizes.
    """
    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    with pert_path.open() as f:
        pert = json.load(f)
    results = pert["target_results"]

    db_tfs = set(targets_db.keys())
    N = UNIVERSE_SIZE  # approx. human protein-coding universe

    per_tf = []
    raw_pvals = []
    for r in results:
        tf = r["target_gene"]
        if tf not in db_tfs:
            continue
        known = targets_db.get(tf, set())  # do NOT intersect with observable universe
        if len(known) < MIN_KNOWN_TARGETS:
            continue

        # Union top-20 gene set of responding features (no universe intersection)
        union_genes: Set[str] = set()
        any_overlap = 0
        for tc in r.get("top_changed_features", []):
            if abs(tc.get("effect_size", 0.0)) < EFFECT_THRESHOLD:
                continue
            fid = int(tc["feature_idx"])
            union_genes |= feat_genes.get(fid, set())
            any_overlap += 1  # counts responding features, not overlap

        n_union = len(union_genes)
        intersection = len(union_genes & known)

        # Hypergeometric sf: P(X >= intersection) under no-enrichment null
        if n_union == 0 or len(known) == 0:
            pval = 1.0
        else:
            pval = float(hypergeom.sf(intersection - 1, N, len(known), n_union))

        per_tf.append({
            "tf": tf,
            "n_known_targets": len(known),
            "n_responding_features": any_overlap,
            "n_union_genes": n_union,
            "intersection": intersection,
            "pvalue_hypergeom": pval,
        })
        raw_pvals.append(pval)

    if raw_pvals:
        reject, qvals, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
        for tf_entry, rej, q in zip(per_tf, reject, qvals):
            tf_entry["qvalue"] = float(q)
            tf_entry["is_specific_fdr"] = bool(rej)
            # Also keep the loose >=1-overlap definition (main-text style) for comparison
            tf_entry["is_specific_loose"] = tf_entry["intersection"] > 0

    n_specific_fdr = sum(1 for t in per_tf if t.get("is_specific_fdr"))
    n_specific_loose = sum(1 for t in per_tf if t.get("is_specific_loose"))

    return {
        "n_db_tfs": len(db_tfs),
        "n_tf_in_panel": len(per_tf),
        "n_specific_loose": n_specific_loose,
        "rate_loose": n_specific_loose / max(len(per_tf), 1),
        "n_specific_fdr": n_specific_fdr,
        "rate_fdr": n_specific_fdr / max(len(per_tf), 1),
        "per_tf": per_tf,
    }


def build_gene_universe(feat_genes: Dict[int, Set[str]]) -> Set[str]:
    """Universe of genes that can ever appear in an SAE feature's top-20 list.

    This is the right denominator for the hypergeometric test: we are asking
    whether the TF's known targets are overrepresented among the genes the SAE
    features preferentially activate on.
    """
    universe: Set[str] = set()
    for s in feat_genes.values():
        universe |= s
    return universe


def main():
    trrust = load_trrust()
    dorothea_abc = load_dorothea({"A", "B", "C"})
    dorothea_all = load_dorothea({"A", "B", "C", "D", "E"})

    feat_genes = load_feature_top_genes()
    universe = build_gene_universe(feat_genes)
    print(f"SAE-feature gene universe size: {len(universe)}")

    pert_path = PHASE1 / "perturbation_response" / "perturbation_response_layer11.json"

    out = {}
    for name, db in [
        ("TRRUST", trrust),
        ("DoRothEA (A+B+C)", dorothea_abc),
        ("DoRothEA (all)", dorothea_all),
    ]:
        spec = compute_specificity(pert_path, db, feat_genes, universe)
        out[name] = {
            "n_db_tfs": spec["n_db_tfs"],
            "n_tf_in_panel": spec["n_tf_in_panel"],
            "n_specific_loose": spec["n_specific_loose"],
            "rate_loose": spec["rate_loose"],
            "n_specific_fdr": spec["n_specific_fdr"],
            "rate_fdr": spec["rate_fdr"],
        }
        (OUT / f"specificity_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')}.json").write_text(
            json.dumps(spec, indent=2)
        )
        print(
            f"{name:>20s}: TFs in panel = {spec['n_tf_in_panel']:>3d}, "
            f"loose>=1 = {spec['n_specific_loose']:>3d} ({spec['rate_loose']*100:>5.2f}%), "
            f"FDR<0.05 = {spec['n_specific_fdr']:>3d} ({spec['rate_fdr']*100:>5.2f}%)"
        )

    (OUT / "specificity_per_db.json").write_text(json.dumps(out, indent=2))
    lines = [
        f"{'Database':>20s} | {'DB TFs':>7s} | {'in panel':>9s} | {'loose':>8s} | {'loose%':>7s} | {'FDR<.05':>8s} | {'FDR%':>6s}"
    ]
    for name, spec in out.items():
        lines.append(
            f"{name:>20s} | {spec['n_db_tfs']:>7d} | {spec['n_tf_in_panel']:>9d} | "
            f"{spec['n_specific_loose']:>8d} | {spec['rate_loose']*100:>6.2f}% | "
            f"{spec['n_specific_fdr']:>8d} | {spec['rate_fdr']*100:>5.2f}%"
        )
    (OUT / "specificity_per_db.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
