"""Phase C4 — scGPT perturbation response mapping on Replogle.

Reviewer 2 (major #4) asked us to run the same perturbation-specificity test
on scGPT as on Geneformer, to verify whether scGPT exhibits similarly low
regulatory specificity. We:

  1. Pick the same 100 Replogle K562 CRISPRi targets used in the Geneformer
     test in `experiments/phase1_k562/perturbation_response/perturbation_response_layer11.json`.
  2. Re-tokenize each perturbed cell (and a control baseline of ~100 non-
     targeting K562 cells) for scGPT input: gene-ID tokens + per-cell binned
     expression values.
  3. Forward through scGPT, capture the layer-7 residual-stream hidden states
     for each gene position, encode through the scGPT L7 SAE.
  4. Compute mean feature activation under perturbation vs control, flag
     features with |effect size| > 0.5.
  5. Test TF specificity against TRRUST (the 48-TF panel already used in the
     Geneformer test) and DoRothEA A+B+C using the same hypergeometric FDR
     definition as `c2_multidb_perturbation.py`.

To keep wall-clock tractable this lean version uses 50 perturbation targets
(not 100) and 10 cells per target. The numbers are reported explicitly as a
first-pass cross-validation of the Geneformer finding.

Outputs:
  experiments/revision/c4_scgpt_pert/summary.json
  experiments/revision/c4_scgpt_pert/summary.txt
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Set

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import h5py
import numpy as np
import torch

PROJECT = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
)
MECHINTERP = Path(
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp"
)
NMI = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper")
SCGPT_REPO = MECHINTERP / "external" / "scGPT"
SCGPT_CKPT = MECHINTERP / "external" / "scGPT_checkpoints" / "whole-human" / "best_model.pt"
SCGPT_VOCAB = MECHINTERP / "external" / "scGPT_checkpoints" / "whole-human" / "vocab.json"

REPLOGLE = NMI / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
TRRUST_PATH = MECHINTERP / "external/networks/trrust_human.tsv"
DOROTHEA_PATH = MECHINTERP / "external/networks/dorothea_human.tsv"

sys.path.insert(0, str(PROJECT / "src"))
from sae_model import TopKSAE  # noqa: E402

SCGPT_ATLAS = PROJECT / "experiments" / "scgpt_atlas"
OUT = PROJECT / "experiments" / "revision" / "c4_scgpt_pert"
OUT.mkdir(parents=True, exist_ok=True)

LAYER = 7
D_MODEL = 512
N_TARGETS = 50
CELLS_PER_TARGET = 10
N_CONTROL_CELLS = 100
MAX_SEQ_LEN = 1200
SEED = 42
EFFECT_THRESHOLD = 0.5
MIN_KNOWN_TARGETS = 5
UNIVERSE_SIZE = 20_000


def log(msg):
    print(msg, flush=True)


def load_categorical(h5group, col_name):
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col["categories"][:]
        codes = col["codes"][:]
        if categories.dtype.kind in ("O", "S"):
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    data = col[:]
    if data.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
    return data


def load_trrust() -> Dict[str, Set[str]]:
    out = {}
    with TRRUST_PATH.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out.setdefault(parts[0], set()).add(parts[1])
    return out


def load_dorothea_abc() -> Dict[str, Set[str]]:
    out = {}
    with DOROTHEA_PATH.open() as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2] in ("A", "B", "C"):
                out.setdefault(parts[0], set()).add(parts[1])
    return out


def tokenize_cell_scgpt(expression_vector, gene_names, vocab, pad_token_id, max_seq_len=1200, pad_value=-2):
    nonzero_mask = expression_vector > 0
    nonzero_indices = np.where(nonzero_mask)[0]
    if len(nonzero_indices) == 0:
        return None
    valid_tok, valid_expr, valid_names = [], [], []
    for idx in nonzero_indices:
        gname = gene_names[idx]
        if gname in vocab:
            valid_tok.append(vocab[gname])
            valid_expr.append(expression_vector[idx])
            valid_names.append(gname)
    if not valid_tok:
        return None
    valid_tok = np.array(valid_tok, dtype=np.int64)
    valid_expr = np.array(valid_expr, dtype=np.float32)
    order = np.argsort(-valid_expr)
    valid_tok = valid_tok[order]
    valid_expr = valid_expr[order]
    valid_names = [valid_names[i] for i in order]
    if len(valid_tok) > max_seq_len:
        valid_tok = valid_tok[:max_seq_len]
        valid_expr = valid_expr[:max_seq_len]
        valid_names = valid_names[:max_seq_len]
    n_genes = len(valid_tok)
    pad_len = max_seq_len - n_genes
    gene_ids = np.pad(valid_tok, (0, pad_len), mode="constant", constant_values=pad_token_id)
    gene_values = np.pad(valid_expr, (0, pad_len), mode="constant", constant_values=pad_value)
    padding_mask = np.zeros(max_seq_len, dtype=bool)
    padding_mask[n_genes:] = True
    return {
        "gene_ids": gene_ids,
        "gene_values": gene_values,
        "src_key_padding_mask": padding_mask,
        "n_genes": n_genes,
        "gene_names": valid_names,
    }


def load_scgpt_model(vocab, device):
    sys.path.insert(0, str(SCGPT_REPO))
    import scgpt  # noqa
    from scgpt.model.model import TransformerModel

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=D_MODEL,
        nhead=8,
        d_hid=D_MODEL,
        nlayers=12,
        vocab=vocab,
        dropout=0.2,
        pad_token="<pad>",
        pad_value=-2,
        input_emb_style="continuous",
        use_fast_transformer=False,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        cell_emb_style="avg-pool",
        n_cls=1,
    )
    ckpt = torch.load(str(SCGPT_CKPT), map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt
    converted = {k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict.items()}
    model.load_state_dict(converted, strict=False)
    return model.to(device).eval()


def encode_scgpt_feature_stats(tokenized_cells, model, vocab, device, sae, mean_np, layer: int):
    """Forward tokenized cells through scGPT, capture layer hidden states, encode
    through SAE, return cumulative feature-activation stats."""
    n_features = sae.n_features
    feat_sum = np.zeros(n_features, dtype=np.float64)
    feat_sq = np.zeros(n_features, dtype=np.float64)
    feat_active = np.zeros(n_features, dtype=np.float64)
    mean_t = torch.tensor(mean_np, dtype=torch.float32)

    hidden_buf = {}

    def hook_fn(module, inp, out):
        hidden_buf["out"] = out.detach()

    handle = model.transformer_encoder.layers[layer].register_forward_hook(hook_fn)
    n_positions = 0
    try:
        for tok in tokenized_cells:
            n_genes = tok["n_genes"]
            if n_genes < 10:
                continue
            gene_ids_t = torch.tensor(tok["gene_ids"], dtype=torch.long).unsqueeze(0).to(device)
            gene_values_t = torch.tensor(tok["gene_values"], dtype=torch.float32).unsqueeze(0).to(device)
            padding_mask_t = torch.tensor(tok["src_key_padding_mask"], dtype=torch.bool).unsqueeze(0).to(device)

            hidden_buf.clear()
            with torch.no_grad():
                model._encode(src=gene_ids_t, values=gene_values_t, src_key_padding_mask=padding_mask_t)
            hidden = hidden_buf["out"][0, :n_genes].detach().cpu() - mean_t
            with torch.no_grad():
                h_sparse, _ = sae.encode(hidden)
            h_np = h_sparse.numpy()
            feat_sum += h_np.sum(axis=0)
            feat_sq += (h_np ** 2).sum(axis=0)
            feat_active += (h_np > 0).sum(axis=0)
            n_positions += n_genes
    finally:
        handle.remove()

    if n_positions == 0:
        return None
    mean_act = feat_sum / n_positions
    var_act = feat_sq / n_positions - mean_act ** 2
    freq = feat_active / n_positions
    return {
        "n_positions": int(n_positions),
        "mean": mean_act,
        "var": var_act,
        "freq": freq,
    }


def compute_specificity(per_tf_union_genes, targets_db, db_name: str):
    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    db_tfs = set(targets_db.keys())
    per_tf = []
    raw_pvals = []
    for tf, union_genes in per_tf_union_genes.items():
        if tf not in db_tfs:
            continue
        known = targets_db[tf]
        if len(known) < MIN_KNOWN_TARGETS:
            continue
        n_union = len(union_genes)
        intersection = len(union_genes & known)
        if n_union == 0 or len(known) == 0:
            pval = 1.0
        else:
            pval = float(hypergeom.sf(intersection - 1, UNIVERSE_SIZE, len(known), n_union))
        per_tf.append({
            "tf": tf,
            "n_known_targets": len(known),
            "n_union_genes": n_union,
            "intersection": intersection,
            "pvalue": pval,
        })
        raw_pvals.append(pval)
    if raw_pvals:
        reject, qvals, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
        for e, r, q in zip(per_tf, reject, qvals):
            e["qvalue"] = float(q)
            e["is_specific_fdr"] = bool(r)
            e["is_specific_loose"] = e["intersection"] > 0
    loose_n = sum(1 for e in per_tf if e.get("is_specific_loose"))
    fdr_n = sum(1 for e in per_tf if e.get("is_specific_fdr"))
    return {
        "database": db_name,
        "n_tf_in_panel": len(per_tf),
        "n_specific_loose": loose_n,
        "rate_loose": loose_n / max(len(per_tf), 1),
        "n_specific_fdr": fdr_n,
        "rate_fdr": fdr_n / max(len(per_tf), 1),
        "per_tf": per_tf,
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log(f"Device: {device}")

    with open(SCGPT_VOCAB) as f:
        vocab = json.load(f)
    id_to_name = {v: k for k, v in vocab.items()}

    # Load Replogle obs + gene names
    log("Loading Replogle metadata...")
    with h5py.File(str(REPLOGLE), "r") as f:
        cell_genes = load_categorical(f["obs"], "gene")
        cell_lines = load_categorical(f["obs"], "cell_line")
        var_genes = f["var"]["gene_name_index"][:]
        var_genes = np.array([g.decode() if isinstance(g, bytes) else g for g in var_genes])
    log(f"Replogle var gene count: {len(var_genes)}")
    k562_mask = cell_lines == "k562"
    log(f"K562 cells: {int(k562_mask.sum())}")

    # Reuse the same target list as the Geneformer test if available
    gf_pert_path = PROJECT / "experiments/phase1_k562/perturbation_response/perturbation_response_layer11.json"
    if gf_pert_path.exists():
        with gf_pert_path.open() as f:
            gf_pert = json.load(f)
        gf_targets = [r["target_gene"] for r in gf_pert["target_results"]]
    else:
        gf_targets = []
    # Use the first N_TARGETS
    chosen = gf_targets[:N_TARGETS] if gf_targets else None

    # Build per-target cell lists
    rng = np.random.default_rng(SEED)
    target_cells = {}
    for tgt in chosen:
        m = (cell_genes == tgt) & k562_mask
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > CELLS_PER_TARGET:
            idxs = rng.choice(idxs, size=CELLS_PER_TARGET, replace=False)
        target_cells[tgt] = np.sort(idxs)

    # Control cells: non-targeting K562
    ctrl_mask = (cell_genes == "non-targeting") & k562_mask
    ctrl_idxs = np.where(ctrl_mask)[0]
    if len(ctrl_idxs) > N_CONTROL_CELLS:
        ctrl_idxs = rng.choice(ctrl_idxs, size=N_CONTROL_CELLS, replace=False)
    ctrl_idxs = np.sort(ctrl_idxs)
    log(f"Control cells: {len(ctrl_idxs)}; target cells total: {sum(len(v) for v in target_cells.values())}")

    # Tokenize all needed cells once from Replogle
    log("Tokenizing cells for scGPT...")
    t0 = time.time()
    cell_tokens = {}  # cell_idx -> tokenized dict

    def tokenize_batch(idxs):
        with h5py.File(str(REPLOGLE), "r") as f:
            X = f["X"]
            for ci in idxs:
                ci_int = int(ci)
                row = X[ci_int, :].astype(np.float32)
                tok = tokenize_cell_scgpt(row, var_genes, vocab, vocab["<pad>"], max_seq_len=MAX_SEQ_LEN)
                if tok is not None:
                    cell_tokens[ci_int] = tok

    all_idxs = list(ctrl_idxs) + [ci for v in target_cells.values() for ci in v]
    tokenize_batch(all_idxs)
    log(f"Tokenized {len(cell_tokens)} cells in {time.time() - t0:.1f}s")

    # Load SAE
    run_dir = SCGPT_ATLAS / "sae_models" / f"layer{LAYER:02d}_x4_k32"
    sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device="cpu")
    sae.eval()
    mean_np = np.load(run_dir / "activation_mean.npy")

    # Load scGPT model
    log("Loading scGPT model...")
    model = load_scgpt_model(vocab, device)

    # Control baseline
    log("Computing control baseline...")
    ctrl_tok = [cell_tokens[int(ci)] for ci in ctrl_idxs if int(ci) in cell_tokens]
    ctrl_stats = encode_scgpt_feature_stats(ctrl_tok, model, vocab, device, sae, mean_np, LAYER)
    if ctrl_stats is None:
        log("FATAL: no control positions")
        return
    ctrl_std = np.sqrt(np.maximum(ctrl_stats["var"], 1e-10))

    # Load feature gene sets for hypergeometric test
    catalog_path = run_dir / "feature_catalog.json"
    feat_genes = {}
    if catalog_path.exists():
        with catalog_path.open() as f:
            cat = json.load(f)
        for feat in cat.get("features", []):
            fid = int(feat["feature_idx"])
            top = feat.get("top_genes") or []
            feat_genes[fid] = {g["gene_name"].upper() for g in top[:20]}

    # For each target, compute effect sizes and union gene set of responding features
    target_union_genes = {}
    per_target_results = []
    for ti, (tgt, idxs) in enumerate(target_cells.items(), start=1):
        toks = [cell_tokens[int(ci)] for ci in idxs if int(ci) in cell_tokens]
        if not toks:
            continue
        stats = encode_scgpt_feature_stats(toks, model, vocab, device, sae, mean_np, LAYER)
        if stats is None:
            continue
        effect = (stats["mean"] - ctrl_stats["mean"]) / ctrl_std
        responding_ids = np.where(np.abs(effect) >= EFFECT_THRESHOLD)[0]

        # Union gene set across responding features
        union_genes: Set[str] = set()
        for fid in responding_ids:
            union_genes |= feat_genes.get(int(fid), set())
        target_union_genes[tgt] = union_genes

        per_target_results.append({
            "target": tgt,
            "n_cells": len(toks),
            "n_responding": int(len(responding_ids)),
            "n_union_genes": int(len(union_genes)),
            "max_abs_effect": float(np.max(np.abs(effect))),
        })
        if ti % 5 == 0:
            log(f"  target {ti}/{len(target_cells)}: {tgt}, responding={len(responding_ids)}")

    # Compute specificity for TRRUST and DoRothEA A+B+C
    trrust = load_trrust()
    dorothea = load_dorothea_abc()
    spec_trrust = compute_specificity(target_union_genes, trrust, "TRRUST")
    spec_dor = compute_specificity(target_union_genes, dorothea, "DoRothEA (A+B+C)")

    summary = {
        "layer": LAYER,
        "n_targets_processed": len(per_target_results),
        "n_control_cells": int(ctrl_stats["n_positions"]),
        "per_target_results": per_target_results,
        "specificity": {"TRRUST": spec_trrust, "DoRothEA_ABC": spec_dor},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        f"scGPT perturbation test @ L{LAYER} on Replogle",
        f"Targets processed: {len(per_target_results)}",
        "",
        f"{'Database':>18s} | {'TFs in panel':>12s} | {'loose':>6s} | {'loose%':>7s} | {'FDR':>4s} | {'FDR%':>6s}",
    ]
    for s in (spec_trrust, spec_dor):
        lines.append(
            f"{s['database']:>18s} | {s['n_tf_in_panel']:>12d} | "
            f"{s['n_specific_loose']:>6d} | {s['rate_loose']*100:>6.2f}% | "
            f"{s['n_specific_fdr']:>4d} | {s['rate_fdr']*100:>5.2f}%"
        )
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
