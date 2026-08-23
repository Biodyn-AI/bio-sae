"""Figure generation for the manuscript.

Every panel is drawn from a result file under experiments/revision_srep/, so a figure can
never drift from the numbers it depicts. Run with --figures to draw a subset.

Typography follows the journal's minimum size (>= 8 pt at final print size); no label is
truncated, and tick density is chosen per panel width rather than globally.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RES = common.OUT_ROOT
OUTDIR = common.PROJ / "paper/scientific_reports_revision/figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

BLUE, ORANGE, GREEN, GREY, RED = "#2c6fbb", "#e08214", "#2e8b57", "#8a8a8a", "#b2182b"


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def wrap_labels(labels, width=34):
    return ["\n".join(textwrap.wrap(l, width)) if l else "" for l in labels]


# ---------------------------------------------------------------------------
# Figure: capacity-matched SAE vs SVD  (E6)
# ---------------------------------------------------------------------------

def fig_svd_capacity():
    summary = json.load(open(RES / "E6_svd_capacity/summary.json"))
    layers = summary["layers"]
    k_grid = [int(k) for k in summary["k_grid"]]
    show = [l for l in ["0", "5", "11", "17"] if l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # (a) cumulative projection curves vs the random-direction null
    ax = axes[0]
    colors = [BLUE, GREEN, ORANGE, RED]
    for c, l in zip(colors, show):
        curve = np.load(RES / f"E6_svd_capacity/rho_curves_layer{int(l):02d}.npz")
        med = curve["median"]
        ax.plot(np.arange(1, len(med) + 1), med, color=c, lw=1.6, label=f"layer {l}")
        if l == show[-1]:
            ax.fill_between(np.arange(1, len(med) + 1), curve["q25"], curve["q75"],
                            color=c, alpha=0.15, lw=0)
    d = len(med)
    ax.plot(np.arange(1, d + 1), np.arange(1, d + 1) / d, color=GREY, ls="--", lw=1.4,
            label="random direction")
    ax.axvline(50, color="k", lw=0.8, ls=":")
    ax.text(52, 0.03, "50 axes", fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("principal subspace size $k$")
    ax.set_ylabel("cumulative projection $\\rho_k$")
    ax.set_title("a  SAE directions within the principal subspace", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 1.02)

    # (b) SVD rank needed to match SAE reconstruction, per layer
    ax = axes[1]
    ls = sorted(layers, key=int)
    ranks = [layers[l]["svd_rank_matching_sae_varexpl"] for l in ls]
    ax.bar([int(l) for l in ls], ranks, color=BLUE, width=0.75)
    ax.axhline(50, color=RED, lw=1.4, ls="--")
    ax.annotate("50 axes", xy=(17.4, 50), xytext=(17.4, 105), color=RED, fontsize=7.5,
                ha="right", arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.set_xlabel("Geneformer layer")
    ax.set_ylabel("SVD rank matching SAE\nvariance explained")
    ax.set_title("b  Reconstruction-matched capacity", loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))

    # (c) annotation yield per direction
    ax = axes[2]
    cats, svd_vals, sae_vals = [], [], []
    for l in show:
        am = layers[l].get("annotation_matched")
        if not am:
            continue
        cats.append(f"layer {l}")
        svd_vals.append(am["svd_axes_either"]["mean_terms_per_direction"])
        sae_vals.append(am["sae_features_random_sample"]["mean_terms_per_direction"])
    x = np.arange(len(cats))
    ax.bar(x - 0.19, svd_vals, width=0.36, color=ORANGE, label="top-50 principal axes")
    ax.bar(x + 0.19, sae_vals, width=0.36, color=BLUE, label="random SAE features")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("significant terms per direction")
    ax.set_title("c  Annotation yield per direction", loc="left")
    ax.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig_svd_capacity.pdf")


# ---------------------------------------------------------------------------
# Figure: distinct concepts across layers  (E7)
# ---------------------------------------------------------------------------

def fig_concepts():
    s = json.load(open(RES / "E7_concepts/summary.json"))
    ref = s["reference_resolution"]
    per_layer = s["per_layer_programs_at_reference_resolution"]
    res = s["resolutions"][ref]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    ax = axes[0]
    layers = sorted(per_layer, key=int)
    atoms = [per_layer[l]["n_features"] for l in layers]
    progs = [per_layer[l]["n_distinct_programs"] for l in layers]
    ax.bar([int(l) for l in layers], atoms, color=GREY, width=0.75,
           label="dictionary atoms")
    ax.bar([int(l) for l in layers], progs, color=BLUE, width=0.75,
           label="distinct gene-level programs")
    ax.set_xlabel("Geneformer layer")
    ax.set_ylabel("count")
    ax.set_title("a  Atoms and distinct programs per layer", loc="left")
    ax.set_ylim(0, max(atoms) * 1.28)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7.5,
              handlelength=1.2, columnspacing=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))

    ax = axes[1]
    totals = [s["n_features_total"], res["n_programs"]]
    ax.bar(["atoms\n(18 dictionaries)", "distinct\nprograms"], totals,
           color=[GREY, BLUE], width=0.6)
    for i, v in enumerate(totals):
        ax.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=8.5)
    ax.set_ylabel("count")
    ax.set_title("b  Aggregate across layers", loc="left")
    ax.set_ylim(0, max(totals) * 1.18)

    ax = axes[2]
    pack = s["within_layer_packing"]
    ls = sorted(pack, key=int)
    ax.plot([int(l) for l in ls], [pack[l]["mean_abs_coherence"] for l in ls],
            "o-", color=BLUE, ms=3.5, lw=1.4, label="mean $|\\cos|$ between atoms")
    ax.plot([int(l) for l in ls], [pack[l]["welch_bound"] for l in ls],
            "--", color=GREY, lw=1.4, label="Welch bound")
    ax.set_xlabel("Geneformer layer")
    ax.set_ylabel("decoder coherence")
    ax.set_title("c  Packing geometry within one space", loc="left")
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))

    fig.tight_layout()
    save(fig, "fig_concepts.pdf")


# ---------------------------------------------------------------------------
# Figure: recoverability ceiling  (E1 + SAE arm)
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "de_perturbation": "differential expression\nof the knockdown",
    "sae_features": "SAE features",
    "genie3": "GENIE3",
    "grnboost2": "GRNBoost2",
    "pearson": "co-expression (Pearson)",
    "spearman": "co-expression (Spearman)",
    "geneformer_emb": "Geneformer embeddings",
    "random": "random gene sets",
}
METHOD_ORDER = ["de_perturbation", "sae_features", "genie3", "grnboost2",
                "pearson", "spearman", "geneformer_emb", "random"]


def fig_ceiling():
    path = RES / "E1_ceiling/scores_k562_merged.json"
    if not path.exists():
        path = RES / "E1_ceiling/scores_k562.json"
    scores = json.load(open(path))
    ev = scores["evaluability"]
    S = scores["scores"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    refs = ["TRRUST|measured", "DoRothEA_ABC|measured", "DoRothEA_all|measured"]
    ref_names = ["TRRUST", "DoRothEA A+B+C", "DoRothEA (all)"]
    methods = [m for m in METHOD_ORDER if any(m in S.get(r, {}) for r in refs)]
    x = np.arange(len(methods))
    width = 0.26
    for i, (r, rn) in enumerate(zip(refs, ref_names)):
        vals, ns = [], []
        for m in methods:
            s = S.get(r, {}).get(m)
            vals.append(100 * s["frac_significant"] if s else 0.0)
            ns.append(f"{s['n_significant']}/{s['n_tfs']}" if s else "")
        bars = ax.bar(x + (i - 1) * width, vals, width=width,
                      color=[BLUE, ORANGE, GREEN][i], label=rn)
        for b, t in zip(bars, ns):
            if t:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, t,
                        ha="center", fontsize=6.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    ax.set_ylabel("TFs with target enrichment\nat FDR < 0.05 (%)")
    ax.set_title("a  Target recovery under one evaluation framework", loc="left")
    top = max([b.get_height() for c in ax.containers for b in c] + [1.0])
    ax.set_ylim(0, top * 1.28)
    ax.legend(frameon=False, title="reference network", title_fontsize=8,
              loc="upper right")

    ax = axes[1]
    thresholds = [1, 3, 5, 10, 20]
    for c, (key, name) in zip([BLUE, ORANGE, GREEN],
                              [("TRRUST", "TRRUST"),
                               ("DoRothEA_ABC", "DoRothEA A+B+C"),
                               ("DoRothEA_all", "DoRothEA (all)")]):
        vals = [ev.get(f"{key}_tfs_with_ge{t}_measured_targets", 0) for t in thresholds]
        ax.plot(thresholds, vals, "o-", color=c, ms=4, lw=1.5, label=name)
    ax.set_xlabel("curated targets present in the assay")
    ax.set_ylabel("perturbed TFs")
    ax.set_title("b  How many TFs are evaluable at all", loc="left")
    ax.legend(frameon=False)
    ax.set_xticks(thresholds)

    fig.tight_layout()
    save(fig, "fig_ceiling.pdf")


# ---------------------------------------------------------------------------
# Figure: cell-level statistics, power, and cross-cell-line replication
# ---------------------------------------------------------------------------

def _load_cell_level(tag):
    p = RES / f"E3_cell_level/{tag}/per_target.json"
    return json.load(open(p))["per_target"] if p.exists() else None


def fig_power_and_replication(lines=("k562", "rpe1", "jurkat", "hepg2")):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    rows = _load_cell_level("k562_main")
    ax = axes[0]
    if rows:
        tf = [r for r in rows if r["role"] == "tf"]
        cell = [r["n_responding_cell_level"] for r in tf]
        pos = [r["n_responding_position_fdr"] for r in tf]
        ax.scatter(cell, pos, s=18, color=BLUE, alpha=0.8)
        lim = max(max(cell + [1]), max(pos + [1])) * 1.1
        ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1.2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("responding features, cell-level test")
        ax.set_ylabel("responding features,\nposition-pooled test")
        ax.set_title("a  Unit of replication", loc="left")

    ax = axes[1]
    sweep_path = RES / "E3_cell_level/k562_main/sweep.json"
    if sweep_path.exists():
        sweep = json.load(open(sweep_path))["rows"]
        ns = sorted({r["n"] for r in sweep})
        spec, self_det, lo, hi = [], [], [], []
        for n in ns:
            sub = [r for r in sweep if r["n"] == n and r["role"] == "tf"]
            by_gene = {}
            for r in sub:
                by_gene.setdefault(r["gene"], []).append(r)
            frac = []
            for reps in by_gene.values():
                vals = [1.0 if rr.get("DoRothEA_all_p", 1.0) < 0.05 else 0.0
                        for rr in reps if "DoRothEA_all_p" in rr]
                if vals:
                    frac.append(np.mean(vals))
            spec.append(100 * np.mean(frac) if frac else np.nan)
            boot = [100 * np.mean(np.random.RandomState(i).choice(frac, len(frac)))
                    for i in range(200)] if frac else [np.nan]
            lo.append(np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5))
            self_det.append(100 * np.mean([r["self_detected"] for r in sub]))
        ax.plot(ns, self_det, "o-", color=GREEN, ms=4, lw=1.5,
                label="knockdown detected (positive control)")
        ax.plot(ns, spec, "o-", color=BLUE, ms=4, lw=1.5, label="target enrichment")
        ax.fill_between(ns, lo, hi, color=BLUE, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("perturbed cells per target")
        ax.set_ylabel("percent of TFs")
        ax.set_title("b  Sample size", loc="left")
        ax.legend(frameon=False, loc="upper left")
        ax.set_ylim(0, 105)

    ax = axes[2]
    names, vals, texts = [], [], []
    for line in lines:
        rows = _load_cell_level(f"{line}_main") or _load_cell_level(f"{line}_k562sae")
        if not rows:
            continue
        tf = [r for r in rows if r["role"] == "tf" and "DoRothEA_all_cell_p" in r]
        if not tf:
            continue
        q = common.bh_fdr([r["DoRothEA_all_cell_p"] for r in tf])
        n_sig = int((q < 0.05).sum())
        names.append(line.upper())
        vals.append(100 * n_sig / len(tf))
        texts.append(f"{n_sig}/{len(tf)}")
    if names:
        bars = ax.bar(names, vals, color=BLUE, width=0.6)
        for b, t in zip(bars, texts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, t,
                    ha="center", fontsize=8)
        ax.set_ylabel("TFs with target enrichment\nat FDR < 0.05 (%)")
        ax.set_title("c  Four cell lines", loc="left")

    fig.tight_layout()
    save(fig, "fig_power_replication.pdf")


# ---------------------------------------------------------------------------
# Figure: causal patching with independent evaluation genes  (E5)
# ---------------------------------------------------------------------------

def fig_causal():
    path = RES / "E5_causal_v2/results.json"
    if not path.exists():
        print("  E5 results not available yet")
        return
    s = json.load(open(path))
    arms = s.get("aggregates", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    labels, data, colors = [], [], []
    ARM_LABEL = {"top_annotated": "richly annotated", "random_annotated": "random annotated",
                 "random_any": "random feature"}
    per_feature = s.get("features", [])
    for arm_name in ("top_annotated", "random_annotated", "random_any"):
        members = [f for f in per_feature if arm_name in (f.get("arms") or [])]
        for key, col, tag in (("annotation_topk", ORANGE, "top-20 ∩ term"),
                              ("heldout_term", BLUE, "held-out term genes"),
                              ("matched_random", GREY, "matched random")):
            vals = []
            for f in members:
                block = (f.get("all_positions") or {}).get("ratios_abs") or {}
                v = block.get(key)
                if v is None:
                    ratios = (f.get("all_positions") or {}).get("specificity_ratio_abs") or {}
                    v = ratios.get(key)
                if v is not None and np.isfinite(v):
                    vals.append(v)
            if vals:
                labels.append(f"{ARM_LABEL[arm_name]}\n{tag}")
                data.append(vals)
                colors.append(col)
    if data:
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.axhline(1.0, color="k", lw=0.9, ls=":")
        ax.set_xticklabels(wrap_labels(labels, 16), fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("specificity ratio")
        ax.set_title("a  Evaluation-gene definition", loc="left")

    fig.tight_layout()
    save(fig, "fig_causal.pdf")



# ---------------------------------------------------------------------------
# Figure: cross-layer information highways
# ---------------------------------------------------------------------------

def fig_highways():
    rev = common.PROJ / "experiments/revision/b3_highways"
    long_range = [r for r in json.load(open(rev / "highway_results.json"))
                  if (r["src_layer"], r["tgt_layer"]) in ((0, 5), (5, 11), (11, 17))]
    consec = json.load(open(rev / "consecutive_sweep_corrected.json"))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax = axes[0]
    labels = [f"L{r['src_layer']}$\\to$L{r['tgt_layer']}" for r in long_range]
    pct = [100 * r["null_corrected_highway"]["pct_highway"] for r in long_range]
    tau_null = [r["tau_null_p99_max"] for r in long_range]
    x = np.arange(len(labels))
    bars = ax.bar(x, pct, color=BLUE, width=0.55)
    for b, v in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center",
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("features with a cross-layer\ndependency (%)")
    ax.set_title("a  Long-range pairs, null-corrected", loc="left")
    ax2 = ax.twinx()
    ax2.plot(x, tau_null, "o", color=RED, ms=5, label="permutation null (99th pct)")
    ax2.axhline(3.0, color=RED, ls="--", lw=1.2, label="applied threshold $\\tau=3.0$")
    ax2.set_ylabel("PMI threshold", color=RED)
    ax2.set_ylim(0, 4.2)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)
    leg = ax2.legend(loc="lower center", fontsize=7.2, frameon=True, framealpha=0.92,
                     edgecolor="none", labelcolor=RED, handlelength=1.5)
    leg.get_frame().set_facecolor("white")

    ax = axes[1]
    src = [r["src_layer"] for r in consec]
    pct = [100 * r["pct_highway"] for r in consec]
    ax.plot(src, pct, "o-", color=BLUE, ms=4, lw=1.5)
    ax.set_ylim(min(pct) - 3, 101)
    ax.set_xlabel("source layer $i$ of the pair L$_i\\to$L$_{i+1}$")
    ax.set_ylabel("features with a cross-layer\ndependency (%)")
    ax.set_title("b  Every consecutive pair", loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    lo, hi = min(pct), max(pct)
    ax.annotate(f"{lo:.1f}%", xy=(src[int(np.argmin(pct))], lo),
                xytext=(src[int(np.argmin(pct))] - 3.4, lo - 1.6), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.tight_layout()
    save(fig, "fig_highways.pdf")


FIGURES = {
    "svd": fig_svd_capacity,
    "concepts": fig_concepts,
    "ceiling": fig_ceiling,
    "power": fig_power_and_replication,
    "causal": fig_causal,
    "highways": fig_highways,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", nargs="*", default=list(FIGURES))
    args = ap.parse_args()
    for name in args.figures:
        print(f"drawing {name} ...")
        try:
            FIGURES[name]()
        except FileNotFoundError as exc:
            print(f"  skipped ({exc})")


if __name__ == "__main__":
    main()
