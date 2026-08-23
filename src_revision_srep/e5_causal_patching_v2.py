#!/usr/bin/env python3
"""E5 — is a feature causally necessary for the biology it is annotated with?

Ablating an SAE feature and watching the model's gene logits fall is only evidence of
*learned biology* if the genes that fall are not the very genes the feature's label was
read off. A feature is labelled by enriching its own top-20 activating genes against
ontology terms; if disruption is then scored on those same top-20 genes, the measurement
is circular — zeroing a direction necessarily hurts the tokens that direction fires on,
whether or not the direction encodes a coherent biological program. This script measures
causal necessity on gene sets that are disjoint from the labelling evidence, and against
selection- and expression-matched nulls.

Intervention (identical at every arm): forward pass on a control cell, capture the
residual stream at the output of encoder layer L, SAE-encode the gene positions
(mean-centred), zero one feature, write the decoded difference back into the residual
stream, and continue the forward pass from layer L+1. Because the TopK decoder is affine,
decode(h) - decode(h with feature f zeroed) = W_dec[:, f] * h[:, f], so the patch is
applied exactly, at the positions where f fires, without a second decoder pass. The
readout is the change in the output logit of each gene position's own token.

Three feature-selection arms are run side by side so that "specificity" can be read
against the selection procedure that produced it:

  top_annotated   the richest-annotated features: among features carrying >= 1 ontology
                  annotation, with >= 10 catalogued top genes and activation frequency
                  >= 0.01, rank by 10 * (number of distinct ontologies) + (number of
                  significant annotations) - log10(smallest BH-adjusted p), and take the
                  top n. This is the atlas's own "best-characterised features" rule.
  random_annotated  a uniform random sample of features carrying >= 1 annotation.
  random_any        a uniform random sample of all alive features, annotated or not.

For every feature the gene positions of each cell are partitioned into disjoint sets:

  annotation_topk  in the feature's top-20 genes AND in its annotated term
                   (the set that both defines and would confirm the label)
  heldout_term     in the annotated term but NOT in the top-20 genes
                   (the independent evaluation set — the key measurement)
  topk_not_term    in the top-20 genes but outside the term
  matched_random   a random gene set the size of heldout_term, drawn outside the term
                   and the top-20 and stratified on expression rank and occurrence
                   frequency, so the held-out ratio can be read against its own null
  matched_topk     the same stratified draw at the size of the top-20 set, the null for
                   the top-20-versus-rest ratio
  other            every remaining gene position

Reported per feature and per set: mean delta-logit, and the specificity ratio
|mean delta-logit(set)| / |mean delta-logit(other)|, which is the form the atlas uses, so
the arms are directly comparable with it. Because signed deltas partly cancel over the
thousands of positions in `other`, that denominator can approach zero and inflate the
ratio; a magnitude ratio mean|delta-logit(set)| / mean|delta-logit(other)| is therefore
reported alongside it, which measures the same specificity with a stable denominator.
Both are computed twice: over all gene positions, and restricted to positions where the
feature actually fires. The top-20-versus-everything-else ratio is reported in the exact
atlas form (top-20 genes against all non-top-20 positions).

Outputs: experiments/revision_srep/E5_causal_v2/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.OUT_ROOT / "E5_causal_v2"
ONTOLOGY_DIR = common.BASE / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets"
TERM_ONTOLOGIES = ("GO_BP", "KEGG", "Reactome")
MIN_TERM_SIZE, MAX_TERM_SIZE = 5, 500
N_CTRL_POOL = 2000
TOP_GENES = 20

CATEGORIES = ("annotation_topk", "heldout_term", "topk_not_term",
              "matched_random", "matched_topk", "other")
CAT = {c: i for i, c in enumerate(CATEGORIES)}
N_CAT = len(CATEGORIES)
ARMS = ("top_annotated", "random_annotated", "random_any")


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def load_ontologies():
    """term -> gene set, for the ontologies the feature labels are drawn from."""
    onts = {}
    for name, fn in (("GO_BP", "go_bp_gene_sets.json"),
                     ("KEGG", "kegg_gene_sets.json"),
                     ("Reactome", "reactome_gene_sets.json")):
        p = ONTOLOGY_DIR / fn
        if p.exists():
            raw = json.load(open(p))
            onts[name] = {k: set(g.upper() for g in v) for k, v in raw.items()
                          if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
    return onts


def load_catalog_meta(run_dir):
    """feature_idx -> {top_genes, activation_freq, is_dead}."""
    with open(Path(run_dir) / "feature_catalog.json") as fh:
        catalog = json.load(fh)
    meta = {}
    for feat in catalog["features"]:
        fi = int(feat["feature_idx"])
        meta[fi] = {
            "top_genes": [g["gene_name"] for g in (feat.get("top_genes") or [])[:TOP_GENES]],
            "activation_freq": float(feat.get("activation_freq", 0.0)),
            "is_dead": bool(feat.get("is_dead", False)),
        }
    return meta


# ---------------------------------------------------------------------------
# Feature selection arms
# ---------------------------------------------------------------------------

def annotation_score(anns):
    n_ont = len({a["ontology"] for a in anns})
    min_p = min((a.get("p_adjusted", 1.0) for a in anns), default=1.0)
    return n_ont * 10 + len(anns) - np.log10(max(min_p, 1e-30))


def src_rule_label(anns):
    """The atlas label: first annotation from a named-term ontology, in file order."""
    for a in anns:
        if a["ontology"] in TERM_ONTOLOGIES:
            return a["term"]
    return "unannotated"


def select_arms(ann_map, meta, n_features, seed):
    """Return arm -> ordered list of feature indices."""
    ranked = []
    for fi, anns in ann_map.items():
        m = meta.get(fi)
        if m is None or len(m["top_genes"]) < 10:
            continue
        if m["activation_freq"] < 0.01:
            continue
        ranked.append((annotation_score(anns), fi))
    ranked.sort(key=lambda x: -x[0])
    top_annotated = [fi for _, fi in ranked[:n_features]]

    rng = np.random.RandomState(seed)
    annotated_pool = np.array(sorted(fi for fi in ann_map if fi in meta))
    alive_pool = np.array(sorted(fi for fi, m in meta.items() if not m["is_dead"]))

    def draw(pool):
        k = min(n_features, len(pool))
        return sorted(int(x) for x in rng.choice(pool, size=k, replace=False))

    return {
        "top_annotated": top_annotated,
        "random_annotated": draw(annotated_pool),
        "random_any": draw(alive_pool),
    }


def choose_term(anns, ontologies):
    """The feature's annotated term: the most significant named term whose full gene set
    is available, so that term membership is defined for genes the feature never fires on."""
    best = None
    for a in anns:
        ont = a["ontology"]
        if ont not in ontologies:
            continue
        genes = ontologies[ont].get(a["term"])
        if not genes:
            continue
        key = (float(a.get("p_adjusted", 1.0)), -int(a.get("n_overlap", 0)), a["term"])
        if best is None or key < best[0]:
            best = (key, {"ontology": ont, "term": a["term"],
                          "p_adjusted": float(a.get("p_adjusted", 1.0)),
                          "n_overlap": int(a.get("n_overlap", 0)),
                          "genes": genes})
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def select_cells(cell_line, n_cells, seed):
    """Non-targeting control cells of the requested line, drawn from the same control
    population the dictionary was fitted on."""
    lines, perturbed, var_genes = common.load_replogle_obs()
    ctrl = np.zeros(len(perturbed), dtype=bool)
    for name in ("non-targeting", "Non-targeting", "non_targeting"):
        ctrl |= (perturbed == name)
    if cell_line != "all":
        ctrl &= (lines == cell_line)
    idx = np.where(ctrl)[0]
    rng = np.random.RandomState(seed)
    if len(idx) > N_CTRL_POOL:
        idx = np.sort(rng.choice(idx, N_CTRL_POOL, replace=False))
    if len(idx) > n_cells:
        idx = np.sort(rng.choice(idx, n_cells, replace=False))
    return idx, var_genes


def tokenize(tokenizer, expression):
    out = []
    for ci in range(expression.shape[0]):
        tokens = tokenizer.encode(expression[ci])
        if tokens is None:
            continue
        gene_pos = np.where((tokens != 2) & (tokens != 3))[0]
        if len(gene_pos) == 0:
            continue
        out.append((tokens, gene_pos, tokens[gene_pos].astype(np.int64)))
    return out


# ---------------------------------------------------------------------------
# Per-gene statistics used to match the random control sets
# ---------------------------------------------------------------------------

def token_statistics(cells, vocab_size):
    """Expression-rank proxy per gene token, computed on the cells that are patched.

    Geneformer's input order *is* the expression rank, so the mean relative position of a
    token across cells is its expression rank, and the number of cells containing it is
    its detection frequency. Both drive how strongly a position responds to any
    perturbation, which is exactly what the random control must be matched on.
    """
    rank_sum = np.zeros(vocab_size, dtype=np.float64)
    occ = np.zeros(vocab_size, dtype=np.int64)
    for _, gene_pos, gene_tokens in cells:
        n = len(gene_pos)
        frac = np.arange(n, dtype=np.float64) / max(n - 1, 1)
        np.add.at(rank_sum, gene_tokens, frac)
        np.add.at(occ, gene_tokens, 1)
    present = np.where(occ > 0)[0]
    mean_rank = rank_sum[present] / occ[present]
    return present, mean_rank, occ[present]


def make_strata(mean_rank, occ, n_rank_bins, n_freq_bins):
    """Equal-count bins via ranks, so ties and degenerate quantiles cannot collapse them."""
    n = len(mean_rank)
    def bins(values, k):
        b = np.empty(n, dtype=np.int64)
        b[np.argsort(values, kind="stable")] = (np.arange(n) * k) // max(n, 1)
        return np.minimum(b, k - 1)
    return bins(mean_rank, n_rank_bins) * n_freq_bins + bins(occ, n_freq_bins), \
        bins(mean_rank, n_rank_bins)


def draw_matched(targets, forbidden, present, stratum, rank_bin, rng):
    """One stratified draw per target token: same (rank, frequency) stratum where possible,
    then the same rank bin, then anywhere in the eligible pool."""
    used = set()
    by_stratum, by_rank = {}, {}
    for pos, tok in enumerate(present):
        if int(tok) in forbidden:
            continue
        by_stratum.setdefault(int(stratum[pos]), []).append(int(tok))
        by_rank.setdefault(int(rank_bin[pos]), []).append(int(tok))
    flat = [int(t) for t in present if int(t) not in forbidden]

    tok_pos = {int(t): i for i, t in enumerate(present)}
    picked = []
    for tok in targets:
        p = tok_pos.get(int(tok))
        pools = []
        if p is not None:
            pools = [by_stratum.get(int(stratum[p]), []), by_rank.get(int(rank_bin[p]), [])]
        pools.append(flat)
        choice = None
        for pool in pools:
            candidates = [t for t in pool if t not in used]
            if candidates:
                choice = int(rng.choice(candidates))
                break
        if choice is None:
            continue
        used.add(choice)
        picked.append(choice)
    return picked


# ---------------------------------------------------------------------------
# Gene-set construction per feature
# ---------------------------------------------------------------------------

def build_feature_sets(fi, meta, anns, ontologies, gene_to_tokens, present,
                       stratum, rank_bin, vocab_size, seed):
    """Category vector over the token vocabulary plus the bookkeeping of set sizes."""
    top_genes = meta["top_genes"]
    topk_tokens = set()
    for g in top_genes:
        topk_tokens.update(gene_to_tokens.get(g.upper(), []))

    term = choose_term(anns, ontologies) if anns else None
    term_tokens, n_term_genes_mapped = set(), 0
    if term is not None:
        for g in term["genes"]:
            tids = gene_to_tokens.get(g, [])
            if tids:
                n_term_genes_mapped += 1
                term_tokens.update(tids)

    ann_topk = term_tokens & topk_tokens
    heldout = term_tokens - topk_tokens
    topk_only = topk_tokens - term_tokens

    present_set = set(int(t) for t in present)
    heldout_present = sorted(heldout & present_set)
    topk_present = sorted(topk_tokens & present_set)

    rng = np.random.RandomState(seed + int(fi))
    forbidden = set(topk_tokens) | set(term_tokens)
    matched_random = draw_matched(heldout_present, forbidden, present, stratum, rank_bin, rng)
    forbidden_2 = forbidden | set(matched_random)
    matched_topk = draw_matched(topk_present, forbidden_2, present, stratum, rank_bin, rng)

    catvec = np.full(vocab_size, CAT["other"], dtype=np.int8)
    for tok in topk_only:
        catvec[tok] = CAT["topk_not_term"]
    for tok in ann_topk:
        catvec[tok] = CAT["annotation_topk"]
    for tok in heldout:
        catvec[tok] = CAT["heldout_term"]
    for tok in matched_random:
        catvec[tok] = CAT["matched_random"]
    for tok in matched_topk:
        catvec[tok] = CAT["matched_topk"]

    info = {
        "term": None if term is None else {k: v for k, v in term.items() if k != "genes"},
        "n_term_genes": 0 if term is None else len(term["genes"]),
        "n_term_genes_in_vocab": n_term_genes_mapped,
        "set_sizes": {
            "top_genes": len(top_genes),
            "topk_tokens": len(topk_tokens),
            "annotation_topk": len(ann_topk),
            "heldout_term": len(heldout),
            "heldout_term_present": len(heldout_present),
            "topk_not_term": len(topk_only),
            "topk_present": len(topk_present),
            "matched_random": len(matched_random),
            "matched_topk": len(matched_topk),
        },
        "heldout_available": len(heldout_present) > 0,
    }
    return catvec, info


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def make_logit_reader(model):
    """Read the logit of each position's own token without materialising the full
    (positions x vocabulary) logit matrix.

    The masked-LM head is a transform followed by a tied linear decoder, so the logit of
    one token at one position is a single dot product. Evaluating only the tokens that are
    actually scored keeps peak memory per forward pass at a few megabytes instead of
    hundreds, which matters over the tens of thousands of forward passes this run makes.
    """
    import torch

    head = getattr(getattr(model, "cls", None), "predictions", None)
    if head is None or not hasattr(head, "transform") or not hasattr(head, "decoder"):
        return None
    bias = getattr(head.decoder, "bias", None)
    if bias is None:
        bias = getattr(head, "bias", None)

    def read(seq_out, pos_t, tok_t):
        h = head.transform(seq_out[0, pos_t])
        vals = (h * head.decoder.weight[tok_t]).sum(-1)
        if bias is not None:
            vals = vals + bias[tok_t]
        return vals

    def verify(input_ids, attn, pos_t, tok_t):
        with torch.no_grad():
            full = model(input_ids=input_ids, attention_mask=attn).logits[0, pos_t, tok_t]
            fast = read(model.bert(input_ids=input_ids,
                                   attention_mask=attn).last_hidden_state, pos_t, tok_t)
            return float((full - fast).abs().max().item())

    return read, verify


def run_patching(model, device, sae, act_mean_t, layer, cells, feature_ids, catvecs,
                 save_cb=None, save_every=10):
    import torch

    sae = sae.to(device)
    mean_dev = act_mean_t.to(device)
    w_dec = sae.W_dec.weight.detach().to(device)          # (d_model, n_features)

    acc = {fi: {"sum_all": np.zeros(N_CAT), "abs_all": np.zeros(N_CAT),
                "cnt_all": np.zeros(N_CAT, dtype=np.int64),
                "sum_act": np.zeros(N_CAT), "abs_act": np.zeros(N_CAT),
                "cnt_act": np.zeros(N_CAT, dtype=np.int64),
                "n_cells": 0, "act_sum": 0.0, "act_n": 0, "cos_sum": 0.0, "cos_n": 0}
           for fi in feature_ids}

    state = {"replace": None, "capture": None}

    def hook(_module, _inputs, output):
        if state["replace"] is not None:
            if isinstance(output, tuple):
                return (state["replace"],) + tuple(output[1:])
            return state["replace"]
        state["capture"] = output[0] if isinstance(output, tuple) else output
        return output

    handle = model.bert.encoder.layer[layer].register_forward_hook(hook)
    reader = make_logit_reader(model)
    t0 = time.time()
    verified = None
    try:
        for ci, (tokens, gene_pos, gene_tokens) in enumerate(cells):
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            attn = torch.ones(1, len(tokens), dtype=torch.long, device=device)
            pos_t = torch.tensor(gene_pos, dtype=torch.long, device=device)
            tok_t = torch.tensor(gene_tokens, dtype=torch.long, device=device)

            if reader is not None and verified is None:
                verified = reader[1](input_ids, attn, pos_t, tok_t)
                print(f"    logit readout check: max |full - fast| = {verified:.2e}",
                      flush=True)
                if not np.isfinite(verified) or verified > 1e-2:
                    reader = None
                state["capture"] = None

            with torch.no_grad():
                if reader is None:
                    out = model(input_ids=input_ids, attention_mask=attn)
                    base_logits = out.logits[0, pos_t, tok_t].clone()
                else:
                    out = model.bert(input_ids=input_ids, attention_mask=attn)
                    base_logits = reader[0](out.last_hidden_state, pos_t, tok_t).clone()
            hidden = state["capture"]
            del out
            state["capture"] = None

            with torch.no_grad():
                h_sparse, _ = sae.encode(hidden[0, pos_t] - mean_dev)

            for fi in feature_ids:
                act = h_sparse[:, fi]
                mask = act > 0
                n_act = int(mask.sum().item())
                if n_act == 0:
                    continue
                idx = pos_t[mask]
                vals = act[mask]
                with torch.no_grad():
                    delta = -vals.unsqueeze(1) * w_dec[:, fi].unsqueeze(0)
                    modified = hidden.clone()
                    modified[0, idx] += delta
                    cos = torch.nn.functional.cosine_similarity(
                        hidden[0, idx], modified[0, idx], dim=1)
                    state["replace"] = modified
                    if reader is None:
                        out = model(input_ids=input_ids, attention_mask=attn)
                        ab = out.logits[0, pos_t, tok_t]
                    else:
                        out = model.bert(input_ids=input_ids, attention_mask=attn)
                        ab = reader[0](out.last_hidden_state, pos_t, tok_t)
                    state["replace"] = None
                    dl = (ab - base_logits).cpu().numpy()
                del out, ab, modified, delta

                codes = catvecs[fi][gene_tokens].astype(np.intp)
                mask_np = mask.cpu().numpy()

                adl = np.abs(dl)
                a = acc[fi]
                a["sum_all"] += np.bincount(codes, weights=dl, minlength=N_CAT)
                a["abs_all"] += np.bincount(codes, weights=adl, minlength=N_CAT)
                a["cnt_all"] += np.bincount(codes, minlength=N_CAT)
                a["sum_act"] += np.bincount(codes[mask_np], weights=dl[mask_np],
                                            minlength=N_CAT)
                a["abs_act"] += np.bincount(codes[mask_np], weights=adl[mask_np],
                                            minlength=N_CAT)
                a["cnt_act"] += np.bincount(codes[mask_np], minlength=N_CAT)
                a["n_cells"] += 1
                a["act_sum"] += float(vals.sum().item())
                a["act_n"] += n_act
                a["cos_sum"] += float((1.0 - cos).sum().item())
                a["cos_n"] += n_act

            del hidden, h_sparse, base_logits
            if device.type == "mps":
                torch.mps.empty_cache()
            rate = (ci + 1) / max(time.time() - t0, 1e-9)
            print(f"    cell {ci + 1}/{len(cells)} ({len(gene_pos)} gene positions, "
                  f"{rate * 3600:.0f} cells/h)", flush=True)
            if save_cb is not None and (ci + 1) % save_every == 0:
                save_cb(acc, ci + 1)
    finally:
        handle.remove()
        state["replace"] = None
    return acc


# ---------------------------------------------------------------------------
# Ratios and aggregation
# ---------------------------------------------------------------------------

def _mean(sums, cnts, keys):
    s = float(sum(sums[CAT[k]] for k in keys))
    n = int(sum(cnts[CAT[k]] for k in keys))
    return (s / n) if n > 0 else float("nan")


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return None
    return float(abs(num) / abs(den))


def _ratio_block(sums, cnts):
    means = {c: _mean(sums, cnts, [c]) for c in CATEGORIES}
    other = means["other"]
    topk_all = _mean(sums, cnts, ["annotation_topk", "topk_not_term"])
    atlas_other = _mean(sums, cnts,
                        ["heldout_term", "matched_random", "matched_topk", "other"])
    ratios = {
        "annotation_topk": _ratio(means["annotation_topk"], other),
        "heldout_term": _ratio(means["heldout_term"], other),
        "topk_not_term": _ratio(means["topk_not_term"], other),
        "matched_random": _ratio(means["matched_random"], other),
        "matched_topk": _ratio(means["matched_topk"], other),
        "topk_all": _ratio(topk_all, other),
        "topk_vs_all_other": _ratio(topk_all, atlas_other),
    }
    means_out = {c: (None if not np.isfinite(means[c]) else means[c]) for c in CATEGORIES}
    return means_out, (None if not np.isfinite(topk_all) else topk_all), ratios


def summarise_variant(sums, abs_sums, cnts):
    signed_means, signed_topk, signed_ratios = _ratio_block(sums, cnts)
    abs_means, abs_topk, abs_ratios = _ratio_block(abs_sums, cnts)
    return {
        "mean_delta_logit": signed_means,
        "mean_delta_logit_topk_all": signed_topk,
        "mean_abs_delta_logit": abs_means,
        "mean_abs_delta_logit_topk_all": abs_topk,
        "n_measurements": {c: int(cnts[CAT[c]]) for c in CATEGORIES},
        "specificity_ratio": signed_ratios,
        "specificity_ratio_abs": abs_ratios,
    }


RATIO_NAMES = ("annotation_topk", "heldout_term", "topk_not_term", "matched_random",
               "matched_topk", "topk_all", "topk_vs_all_other")
METRICS = (("ratios", "specificity_ratio"), ("ratios_abs", "specificity_ratio_abs"))


def aggregate(features, arm, variant, n_selected, n_selected_no_heldout):
    rows = [f for f in features if arm in f["arms"] and f["n_cells_patched"] > 0]
    with_heldout = [f for f in rows if f["heldout_available"]]
    out = {
        "n_features_selected": int(n_selected),
        "n_features_patched": len(rows),
        "n_features_with_heldout": len(with_heldout),
        "n_skipped_no_heldout": int(n_selected_no_heldout),
        "n_never_active": sum(1 for f in features
                              if arm in f["arms"] and f["n_cells_patched"] == 0),
    }
    for out_key, ratio_key in METRICS:
        out[out_key] = {}
        for name in RATIO_NAMES:
            pool = with_heldout if name in ("heldout_term", "matched_random") else rows
            vals = [f[variant][ratio_key][name] for f in pool]
            vals = np.array([v for v in vals if v is not None], dtype=float)
            out[out_key][name] = {
                "n": int(len(vals)),
                "median": float(np.median(vals)) if len(vals) else None,
                "mean": float(np.mean(vals)) if len(vals) else None,
                "frac_above_2x": float((vals > 2.0).mean()) if len(vals) else None,
                "frac_above_10x": float((vals > 10.0).mean()) if len(vals) else None,
            }

        paired = [(f[variant][ratio_key]["heldout_term"],
                   f[variant][ratio_key]["matched_random"]) for f in with_heldout]
        paired = [(a, b) for a, b in paired if a is not None and b is not None]
        test = {"n_pairs": len(paired), "statistic": None, "p_value": None,
                "median_difference": None}
        if paired:
            a = np.array([p[0] for p in paired])
            b = np.array([p[1] for p in paired])
            test["median_difference"] = float(np.median(a - b))
            if len(paired) >= 6 and np.any(a != b):
                from scipy.stats import wilcoxon
                try:
                    st, p = wilcoxon(a, b, alternative="two-sided")
                    test["statistic"] = float(st)
                    test["p_value"] = float(p)
                except ValueError:
                    pass
        suffix = "" if out_key == "ratios" else "_abs"
        out[f"wilcoxon_heldout_vs_matched_random{suffix}"] = test
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--n-features", type=int, default=50,
                    help="features per selection arm")
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--cell-line", default="k562", choices=list(common.CELL_LINES) + ["all"])
    ap.add_argument("--n-rank-bins", type=int, default=10)
    ap.add_argument("--n-freq-bins", type=int, default=3)
    ap.add_argument("--skip-no-term", action="store_true",
                    help="do not patch features without a held-out gene set at all")
    ap.add_argument("--save-every", type=int, default=10, help="checkpoint every N cells")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.seed_everything(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"E5 causal patching v2 — layer {args.layer}, {args.n_features} features/arm, "
          f"{args.n_cells} {args.cell_line} control cells, seed {args.seed}", flush=True)

    # --- references -------------------------------------------------------
    sae, act_mean_t, run_dir = common.load_sae(args.layer)
    meta = load_catalog_meta(run_dir)
    ann_map = common.load_feature_annotations(run_dir)
    ontologies = load_ontologies()
    print(f"  catalog: {len(meta)} features, {sum(1 for m in meta.values() if not m['is_dead'])} alive, "
          f"{len(ann_map)} annotated; ontologies "
          f"{[(k, len(v)) for k, v in ontologies.items()]}", flush=True)

    arms = select_arms(ann_map, meta, args.n_features, args.seed)
    feature_arms = {}
    for arm in ARMS:
        for fi in arms[arm]:
            feature_arms.setdefault(int(fi), []).append(arm)
    feature_ids = sorted(feature_arms)
    for arm in ARMS:
        print(f"  arm {arm}: {len(arms[arm])} features", flush=True)
    print(f"  {len(feature_ids)} distinct features to patch", flush=True)

    # --- cells ------------------------------------------------------------
    cell_idx, var_genes = select_cells(args.cell_line, args.n_cells, args.seed)
    tokenizer = common.GeneformerTokenizer(var_genes)
    expression, _ = common.read_expression(cell_idx)
    cells = tokenize(tokenizer, expression)
    del expression
    print(f"  tokenised {len(cells)}/{len(cell_idx)} cells", flush=True)

    gene_to_tokens = {}
    for tid, gname in tokenizer.token_to_gene.items():
        if gname:
            gene_to_tokens.setdefault(gname.upper(), []).append(int(tid))
    vocab_size = int(max(int(tokenizer.token_ids.max()), 3)) + 1

    present, mean_rank, occ = token_statistics(cells, vocab_size)
    stratum, rank_bin = make_strata(mean_rank, occ, args.n_rank_bins, args.n_freq_bins)
    print(f"  {len(present)} gene tokens present in the sampled cells; "
          f"{len(set(stratum.tolist()))} matching strata", flush=True)

    # --- gene sets --------------------------------------------------------
    catvecs, infos = {}, {}
    for fi in feature_ids:
        catvecs[fi], infos[fi] = build_feature_sets(
            fi, meta[fi], ann_map.get(fi, []), ontologies, gene_to_tokens,
            present, stratum, rank_bin, vocab_size, args.seed)
    n_no_heldout = sum(1 for fi in feature_ids if not infos[fi]["heldout_available"])
    print(f"  {len(feature_ids) - n_no_heldout} features have a non-empty held-out set, "
          f"{n_no_heldout} do not", flush=True)

    patch_ids = [fi for fi in feature_ids
                 if infos[fi]["heldout_available"] or not args.skip_no_term]
    print(f"  patching {len(patch_ids)} features x {len(cells)} cells", flush=True)

    # --- model ------------------------------------------------------------
    model, device = common.load_geneformer()
    print(f"  model on {device}", flush=True)

    def build_payload(acc, n_cells_done):
        features = []
        for fi in patch_ids:
            a = acc[fi]
            entry = {
                "feature_idx": int(fi),
                "arms": feature_arms[fi],
                "atlas_label": src_rule_label(ann_map.get(fi, [])),
                "annotation_score": (float(annotation_score(ann_map[fi]))
                                     if fi in ann_map else None),
                "n_annotations": len(ann_map.get(fi, [])),
                "n_ontologies": len({x["ontology"] for x in ann_map.get(fi, [])}),
                "activation_freq": meta[fi]["activation_freq"],
                "top_genes": meta[fi]["top_genes"][:10],
                "n_cells_patched": a["n_cells"],
                "n_active_positions": a["act_n"],
                "mean_activation": (a["act_sum"] / a["act_n"]) if a["act_n"] else 0.0,
                "mean_cos_distance": (a["cos_sum"] / a["cos_n"]) if a["cos_n"] else 0.0,
                "all_positions": summarise_variant(a["sum_all"], a["abs_all"],
                                                   a["cnt_all"]),
                "active_positions": summarise_variant(a["sum_act"], a["abs_act"],
                                                      a["cnt_act"]),
            }
            entry.update(infos[fi])
            features.append(entry)

        aggregates = {}
        for arm in ARMS:
            n_sel = len(arms[arm])
            n_sel_no_heldout = sum(1 for fi in arms[arm]
                                   if not infos[int(fi)]["heldout_available"])
            aggregates[arm] = {
                v: aggregate(features, arm, v, n_sel, n_sel_no_heldout)
                for v in ("all_positions", "active_positions")}
        return {
            "layer": args.layer,
            "config": {
                "n_features_per_arm": args.n_features,
                "n_cells_requested": args.n_cells,
                "n_cells_tokenised": len(cells),
                "n_cells_done": n_cells_done,
                "cell_line": args.cell_line,
                "n_rank_bins": args.n_rank_bins,
                "n_freq_bins": args.n_freq_bins,
                "skip_no_term": args.skip_no_term,
                "sae_run_dir": str(run_dir),
                "n_features_patched": len(patch_ids),
                "n_features_no_heldout": n_no_heldout,
            },
            "arms": {arm: arms[arm] for arm in ARMS},
            "aggregates": aggregates,
            "features": features,
            "elapsed_s": time.time() - t_start,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def checkpoint(acc, n_done):
        common.write_json(OUT / "results_partial.json", build_payload(acc, n_done),
                          seed=args.seed)

    acc = run_patching(model, device, sae, act_mean_t, args.layer, cells, patch_ids,
                       catvecs, save_cb=checkpoint, save_every=args.save_every)

    payload = build_payload(acc, len(cells))
    common.write_json(OUT / "results.json", payload, seed=args.seed)

    arrays = {}
    for arm in ARMS:
        rows = [f for f in payload["features"] if arm in f["arms"]]
        for variant in ("all_positions", "active_positions"):
            for _, ratio_key in METRICS:
                for name in RATIO_NAMES:
                    vals = [f[variant][ratio_key][name] for f in rows]
                    arrays[f"{arm}__{variant}__{ratio_key}__{name}"] = np.array(
                        [np.nan if v is None else v for v in vals], dtype=float)
        arrays[f"{arm}__feature_idx"] = np.array([f["feature_idx"] for f in rows])
    np.savez_compressed(OUT / "specificity_ratios.npz", **arrays)
    print(f"  wrote {OUT / 'specificity_ratios.npz'}", flush=True)

    for arm in ARMS:
        ag = payload["aggregates"][arm]["all_positions"]
        print(f"\n{arm}: {ag['n_features_patched']} patched, "
              f"{ag['n_features_with_heldout']} with held-out genes", flush=True)
        for out_key, _ in METRICS:
            r = ag[out_key]
            print(f"  [{out_key}]", flush=True)
            for name in ("topk_vs_all_other", "annotation_topk", "heldout_term",
                         "matched_random", "matched_topk"):
                print(f"    {name:>18}: median={r[name]['median']}, "
                      f">2x={r[name]['frac_above_2x']}, n={r[name]['n']}", flush=True)
        print(f"    wilcoxon heldout vs matched-random (signed): "
              f"{ag['wilcoxon_heldout_vs_matched_random']}", flush=True)
        print(f"    wilcoxon heldout vs matched-random (abs):    "
              f"{ag['wilcoxon_heldout_vs_matched_random_abs']}", flush=True)

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
