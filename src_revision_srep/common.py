"""Shared infrastructure for the Scientific Reports revision experiments.

Provides a streaming Geneformer -> layer-hook -> SAE-encode -> per-cell-aggregate
pipeline. Aggregating at the cell level (rather than pooling gene positions across
cells) is what makes the cell the unit of statistical analysis, and keeps the disk
cost at tens of MB instead of tens of GB.

All randomness is seeded explicitly; every writer records its seed in the output JSON.
"""

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path("/Volumes/Crucial X6/MacBook/biomechinterp")
PAPER_DIR = BASE / "biodyn-nmi-paper"
PROJ = BASE / "biodyn-work/subproject_42_sparse_autoencoder_biological_map"
PHASE1 = PROJ / "experiments/phase1_k562"
SAE_BASE = PHASE1 / "sae_models"
TOKEN_DICTS = PAPER_DIR / "src/02_cssi_method/crispri_validation/data"
REPLOGLE = TOKEN_DICTS / "replogle_concat.h5ad"
NETWORKS = BASE / "biodyn-work/single_cell_mechinterp/external/networks"
TRRUST_PATH = NETWORKS / "trrust_human.tsv"
DOROTHEA_PATH = NETWORKS / "dorothea_human.tsv"
OUT_ROOT = PROJ / "experiments/revision_srep"

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"
HIDDEN_DIM = 1152
MAX_SEQ_LEN = 2048
EXPANSION = 4
K_VAL = 32
CELL_LINES = ("k562", "rpe1", "jurkat", "hepg2")

sys.path.insert(0, str(PROJ / "src"))


# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_json(path, payload, seed=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if seed is not None and isinstance(payload, dict):
        payload = {"seed": seed, **payload}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default)
    print(f"  wrote {path}")
    return path


# ----------------------------------------------------------------------------
# AnnData (h5py) helpers
# ----------------------------------------------------------------------------

def load_categorical_column(h5group, col_name):
    import h5py
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col["categories"][:]
        codes = col["codes"][:]
        if categories.dtype.kind in ("O", "S"):
            categories = np.array(
                [x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    data = col[:]
    if data.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
    return data


def load_replogle_obs():
    """Return (cell_line, perturbed_gene, var_gene_names) for the whole file."""
    import h5py
    with h5py.File(REPLOGLE, "r") as f:
        cell_line = load_categorical_column(f["obs"], "cell_line")
        gene = load_categorical_column(f["obs"], "gene")
        var_genes = load_categorical_column(f["var"], "gene_name_index")
    return cell_line, gene, var_genes


def read_expression(cell_indices, batch=256):
    """Read raw counts for the given absolute row indices, log1p-CP10K normalised."""
    import h5py
    cell_indices = np.asarray(sorted(cell_indices), dtype=np.int64)
    with h5py.File(REPLOGLE, "r") as f:
        n_genes = f["X"].shape[1]
        out = np.empty((len(cell_indices), n_genes), dtype=np.float32)
        for start in range(0, len(cell_indices), batch):
            idx = cell_indices[start:start + batch]
            out[start:start + len(idx), :] = f["X"][idx, :]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    np.log1p(out / row_sums * 1e4, out=out)
    return out, cell_indices


# ----------------------------------------------------------------------------
# Regulatory references
# ----------------------------------------------------------------------------

def load_trrust():
    """TF -> set(targets) from TRRUST v2."""
    targets = {}
    with open(TRRUST_PATH) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                targets.setdefault(parts[0], set()).add(parts[1])
    return targets


def load_dorothea(levels=("A", "B", "C")):
    """TF -> set(targets) from DoRothEA, restricted to the given confidence levels."""
    targets = {}
    if not DOROTHEA_PATH.exists():
        return targets
    with open(DOROTHEA_PATH) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        def _col(*names, default=None):
            for n in names:
                if n in header:
                    return header.index(n)
            return default
        i_tf = _col("source", "tf", default=0)
        i_tg = _col("target", default=1)
        i_cf = _col("confidence", default=2)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_tf, i_tg, i_cf):
                continue
            if levels is None or parts[i_cf] in levels:
                targets.setdefault(parts[i_tf], set()).add(parts[i_tg])
    return targets


# ----------------------------------------------------------------------------
# Geneformer tokenisation
# ----------------------------------------------------------------------------

class GeneformerTokenizer:
    """Rank-value tokenizer matching src/09_perturbation_response.py exactly."""

    def __init__(self, var_genes):
        with open(TOKEN_DICTS / "token_dictionary_gc104M.pkl", "rb") as f:
            token_dict = pickle.load(f)
        with open(TOKEN_DICTS / "gene_median_dictionary_gc104M.pkl", "rb") as f:
            gene_median_dict = pickle.load(f)
        with open(TOKEN_DICTS / "gene_name_id_dict_gc104M.pkl", "rb") as f:
            gene_name_id_dict = pickle.load(f)

        var_idx, token_ids, medians = [], [], []
        for i, gene_name in enumerate(var_genes):
            ensembl = gene_name_id_dict.get(gene_name)
            if ensembl and ensembl in token_dict:
                var_idx.append(i)
                token_ids.append(token_dict[ensembl])
                medians.append(gene_median_dict.get(ensembl, 1.0))
        self.var_indices = np.array(var_idx)
        self.token_ids = np.array(token_ids)
        self.medians = np.array(medians)

        id_to_name = {v: k for k, v in gene_name_id_dict.items()}
        token_to_ens = {v: k for k, v in token_dict.items()}
        self.token_to_gene = {}
        for tid in set(int(t) for t in token_ids):
            ens = token_to_ens.get(tid)
            if ens:
                self.token_to_gene[tid] = id_to_name.get(ens, ens)

    def encode(self, expression_vector, max_len=MAX_SEQ_LEN):
        expr = expression_vector[self.var_indices]
        nonzero = expr > 0
        if nonzero.sum() == 0:
            return None
        expr_nz = expr[nonzero]
        tokens_nz = self.token_ids[nonzero]
        medians_nz = self.medians[nonzero]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = expr_nz / medians_nz
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0)
        order = np.argsort(-normalized)
        ranked = tokens_nz[order][:max_len - 2]
        return np.concatenate([[2], ranked, [3]]).astype(np.int64)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def get_device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_geneformer(device=None):
    import torch
    from transformers import BertForMaskedLM
    device = device or get_device()
    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_hidden_states=False, output_attentions=False,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, device


def load_sae(layer, sae_base=None, device="cpu"):
    """Load a trained TopK SAE plus its activation mean."""
    import torch
    from sae_model import TopKSAE
    sae_base = Path(sae_base or SAE_BASE)
    run_dir = sae_base / f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    sae = TopKSAE.load(str(run_dir / "sae_final.pt"), device=device)
    sae.eval()
    act_mean = np.load(run_dir / "activation_mean.npy")
    return sae, torch.tensor(act_mean, dtype=torch.float32), run_dir


def load_feature_catalog(run_dir):
    """feature_idx -> list of top-20 gene names (ordered by mean activation)."""
    with open(Path(run_dir) / "feature_catalog.json") as fh:
        catalog = json.load(fh)
    out = {}
    for feat in catalog["features"]:
        genes = feat.get("top_genes") or []
        out[int(feat["feature_idx"])] = [g["gene_name"] for g in genes[:20]]
    return out


def load_feature_annotations(run_dir):
    with open(Path(run_dir) / "feature_annotations.json") as fh:
        data = json.load(fh)
    return {int(k): v for k, v in data.get("feature_annotations", {}).items()}


# ----------------------------------------------------------------------------
# Streaming encode: per-cell feature aggregates
# ----------------------------------------------------------------------------

def encode_cells(model, device, saes, act_means, tokenizer, expression, layer,
                 progress_every=100, tag="", batch_size=8):
    """Forward-pass each cell, capture the residual stream at `layer` with a module hook,
    SAE-encode on device, and aggregate per cell.

    Only the per-cell aggregates leave the accelerator, so the transfer per cell is a few
    kilobytes rather than the full (positions x d_model) hidden state.

    Returns dict with
        mean / freq / msq : (n_cells, n_features) per-cell mean, firing rate, and mean
            square of feature activation over that cell's gene positions
        n_pos  : (n_cells,) gene positions per cell
        pooled_sum / pooled_sqsum / pooled_active / pooled_n : position-level pooled
            statistics, kept so the position-pooled (pseudoreplicated) test can be
            reproduced side by side with the cell-level one.
    """
    import torch

    if not isinstance(saes, (list, tuple)):
        saes, act_means = [saes], [act_means]
    n_dict = len(saes)
    n_features = [s_.n_features for s_ in saes]
    n_cells = expression.shape[0]
    cell_mean = [np.zeros((n_cells, nf), dtype=np.float32) for nf in n_features]
    cell_freq = [np.zeros((n_cells, nf), dtype=np.float32) for nf in n_features]
    cell_msq = [np.zeros((n_cells, nf), dtype=np.float32) for nf in n_features]
    n_pos = np.zeros(n_cells, dtype=np.int32)

    pooled_sum = [np.zeros(nf, dtype=np.float64) for nf in n_features]
    pooled_sqsum = [np.zeros(nf, dtype=np.float64) for nf in n_features]
    pooled_active = [np.zeros(nf, dtype=np.float64) for nf in n_features]
    pooled_n = 0

    sae_dev = [s_.to(device) for s_ in saes]
    mean_dev = [m.to(device) for m in act_means]

    captured = {}

    def hook(_module, _inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.bert.encoder.layer[layer].register_forward_hook(hook)

    # Pre-tokenize so cells of similar length can be batched together.
    token_list, order = [], []
    for ci in range(n_cells):
        tokens = tokenizer.encode(expression[ci])
        if tokens is None:
            continue
        if ((tokens != 2) & (tokens != 3)).sum() == 0:
            continue
        token_list.append(tokens)
        order.append(ci)
    if not token_list:
        handle.remove()
        return [{"mean": cell_mean[di][:0], "freq": cell_freq[di][:0],
                 "msq": cell_msq[di][:0], "n_pos": n_pos[:0],
                 "kept": np.array([], dtype=np.int64),
                 "pooled_sum": pooled_sum[di], "pooled_sqsum": pooled_sqsum[di],
                 "pooled_active": pooled_active[di], "pooled_n": 0}
                for di in range(n_dict)]

    by_len = sorted(range(len(token_list)), key=lambda i: len(token_list[i]))
    kept = []
    t0 = time.time()
    done = 0
    try:
        for start in range(0, len(by_len), batch_size):
            group = by_len[start:start + batch_size]
            lens = [len(token_list[i]) for i in group]
            max_len = max(lens)
            ids = np.zeros((len(group), max_len), dtype=np.int64)
            attn = np.zeros((len(group), max_len), dtype=np.int64)
            for r, i in enumerate(group):
                tk = token_list[i]
                ids[r, :len(tk)] = tk
                attn[r, :len(tk)] = 1
            input_ids = torch.tensor(ids, device=device)
            attention_mask = torch.tensor(attn, device=device)

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = captured["h"]                       # (B, max_len, d_model)
                # tokens are [CLS] + genes + [SEP], so gene positions are 1 .. len-2
                segs = [hidden[r, 1:lens[r] - 1] for r in range(len(group))]
                counts = torch.tensor([s_.shape[0] for s_ in segs], device=device)
                raw = torch.cat(segs, dim=0)
                seg_id = torch.repeat_interleave(
                    torch.arange(len(group), device=device), counts)
                denom = counts.unsqueeze(1).float()
                c_all = counts.cpu().numpy()
                for di in range(n_dict):
                    h_sparse, _ = sae_dev[di].encode(raw - mean_dev[di])
                    nf = n_features[di]
                    acc = torch.zeros((len(group), nf), device=device)
                    accf = torch.zeros((len(group), nf), device=device)
                    accs = torch.zeros((len(group), nf), device=device)
                    acc.index_add_(0, seg_id, h_sparse)
                    accf.index_add_(0, seg_id, (h_sparse > 0).float())
                    accs.index_add_(0, seg_id, h_sparse ** 2)
                    m_all = (acc / denom).cpu().numpy()
                    f_all = (accf / denom).cpu().numpy()
                    s_all = (accs / denom).cpu().numpy()
                    for r, i in enumerate(group):
                        ci = order[i]
                        cell_mean[di][ci] = m_all[r]
                        cell_freq[di][ci] = f_all[r]
                        cell_msq[di][ci] = s_all[r]
                        pooled_sum[di] += m_all[r].astype(np.float64) * c_all[r]
                        pooled_sqsum[di] += s_all[r].astype(np.float64) * c_all[r]
                        pooled_active[di] += f_all[r].astype(np.float64) * c_all[r]
                for r, i in enumerate(group):
                    ci = order[i]
                    n_pos[ci] = int(c_all[r])
                    pooled_n += int(c_all[r])
                    kept.append(ci)
                    done += 1
            captured.clear()
            if device.type == "mps":
                torch.mps.empty_cache()
            if progress_every and done % progress_every < batch_size:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    {tag}{done}/{len(token_list)} cells ({rate:.2f} cells/s)",
                      flush=True)
    finally:
        handle.remove()

    kept = np.array(sorted(set(kept)), dtype=np.int64)
    return [{
        "mean": cell_mean[di][kept],
        "freq": cell_freq[di][kept],
        "msq": cell_msq[di][kept],
        "n_pos": n_pos[kept],
        "kept": kept,
        "pooled_sum": pooled_sum[di],
        "pooled_sqsum": pooled_sqsum[di],
        "pooled_active": pooled_active[di],
        "pooled_n": pooled_n,
    } for di in range(n_dict)]


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def hypergeom_enrichment(predicted, known, universe_size=20000):
    """One-sided hypergeometric p-value for overlap of `predicted` with `known`."""
    from scipy.stats import hypergeom
    predicted = set(predicted)
    known = set(known)
    overlap = len(predicted & known)
    if overlap == 0 or not predicted or not known:
        return 1.0, overlap, 0.0
    p = float(hypergeom.sf(overlap - 1, universe_size, len(known), len(predicted)))
    expected = len(predicted) * len(known) / universe_size
    fold = overlap / expected if expected > 0 else 0.0
    return p, overlap, fold


def intraclass_correlation(values, groups):
    """One-way random-effects ICC(1): between-group variance / total variance."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k = len(uniq)
    if k < 2:
        return float("nan")
    grand = values.mean()
    ns, ssb, ssw = [], 0.0, 0.0
    for g in uniq:
        v = values[groups == g]
        ns.append(len(v))
        ssb += len(v) * (v.mean() - grand) ** 2
        ssw += ((v - v.mean()) ** 2).sum()
    n_total = len(values)
    if n_total - k <= 0:
        return float("nan")
    msb = ssb / (k - 1)
    msw = ssw / (n_total - k)
    n0 = (n_total - sum(n ** 2 for n in ns) / n_total) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        return float("nan")
    return float((msb - msw) / denom)


def seed_everything(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed
