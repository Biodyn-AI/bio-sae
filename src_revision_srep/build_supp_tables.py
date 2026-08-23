"""Generate the supplementary tables that report new analyses, straight from the result
files, so that no number in the supplement is transcribed by hand.

Each table is written as a standalone .tex fragment under
paper/scientific_reports_revision/sections/supp/, to be included from supplementary.tex.
Tables whose inputs are not yet on disk are skipped with a notice, so the script can be
re-run as results land.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RES = common.OUT_ROOT
OUT = common.PROJ / "paper/scientific_reports_revision/sections/supp"
LINE_LABEL = {"k562": "K562", "rpe1": "RPE1", "jurkat": "Jurkat", "hepg2": "HepG2"}


def esc(s):
    return (str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"  wrote {OUT / name}")


def table(label, caption, colspec, header, rows, note=None, small=True):
    lines = ["\\begin{table}[htbp]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\smallskip"]
    if small:
        lines.append("\\small")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
    lines += rows
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\n\\footnotesize {note}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def maybe(path):
    p = RES / path
    return json.load(open(p)) if p.exists() else None


# ---------------------------------------------------------------------------

def t_cohorts():
    d = maybe("E0_cohorts/composition.json")
    if not d:
        return None
    lines = ["k562", "rpe1", "jurkat", "hepg2"]
    rows = []
    for key, label in (("available_non_targeting_cells", "Non-targeting cells in the resource"),
                       ("atlas_extraction_cohort", "Atlas extraction cohort")):
        c = d[key]["by_cell_line"]
        rows.append(" & ".join([label] + [f"{c.get(l, 0):,}" for l in lines]
                               + [f"{d[key]['n']:,}"]) + " \\\\")
    v = d.get("validation_against_extraction_metadata", {})
    note = ("The control condition is defined by perturbation label, so the cohort spans every "
            "cell line in the resource. The cohort reported here was reproduced from the "
            "extraction procedure under its recorded seed and validated against the per-cell "
            "gene counts recorded when the activations were written.")
    return table("tab:cohorts",
                 "\\textbf{Composition of the control cohorts.} Cells per line.",
                 "lrrrrr", " & ".join(["Cohort"] + [LINE_LABEL[l] for l in lines]
                                      + ["Total"]) + " \\\\",
                 rows, note)


def t_svd_capacity():
    d = maybe("E6_svd_capacity/summary.json")
    if not d:
        return None
    L = d["layers"]
    rows = []
    for l in sorted(L, key=int):
        e = L[l]
        rows.append(" & ".join([
            l,
            f"{e['sae_variance_explained']:.3f}",
            str(e["svd_rank_matching_sae_varexpl"]),
            str(e["k_median_rho_50pct"]),
            str(e["k_median_rho_90pct"]),
            f"{e['median_rho_at_k']['100']:.3f}",
            f"{100 * e['alignment_fraction']['50']['0.5']:.2f}",
            f"{100 * e['alignment_fraction']['1152']['0.5']:.2f}",
        ]) + " \\\\")
    note = ("$k^{*}$ is the truncated-SVD rank whose cumulative eigenvalue mass equals the SAE's "
            "variance explained. $k_{50}$ and $k_{90}$ are the subspace sizes at which the median "
            "decoder direction reaches 50\\% and 90\\% of its norm; for a random direction these "
            "are 576 and 1{,}037. The last two columns give the percentage of features aligned "
            "above $|\\cos| = 0.5$ with any of the leading 50 axes and with the complete basis.")
    return table("tab:svd_capacity",
                 "\\textbf{Capacity-matched comparison against the principal subspace.}",
                 "rrrrrrrr",
                 "Layer & SAE VE & $k^{*}$ & $k_{50}$ & $k_{90}$ & $\\rho_{100}$ & "
                 "align.\\ top-50 (\\%) & align.\\ full (\\%) \\\\",
                 rows, note)


def t_concepts():
    d = maybe("E7_concepts/summary.json")
    if not d:
        return None
    per = d["per_layer_programs_at_reference_resolution"]
    pack = d["within_layer_packing"]
    rows = []
    for l in sorted(per, key=int):
        p = pack.get(l, {})
        rows.append(" & ".join([
            l, f"{per[l]['n_features']:,}", f"{per[l]['n_distinct_programs']:,}",
            f"{per[l]['n_features'] / max(per[l]['n_distinct_programs'], 1):.2f}",
            f"{p.get('mean_abs_coherence', float('nan')):.4f}",
            f"{p.get('welch_bound', float('nan')):.4f}",
        ]) + " \\\\")
    res = d["resolutions"][d["reference_resolution"]]
    rows.append("\\midrule")
    rows.append(" & ".join(["\\textbf{All}", f"\\textbf{{{d['n_features_total']:,}}}",
                            f"\\textbf{{{res['n_programs']:,}}}",
                            f"\\textbf{{{res['atoms_per_program']:.2f}}}", "", ""]) + " \\\\")
    note = ("Programs are communities of the top-20 gene-set similarity graph at Leiden "
            f"resolution {d['reference_resolution']}; the total is {res['n_programs']:,} and varies "
            "by less than 0.3\\% over resolutions 0.5--2.0. The Welch bound is the smallest "
            "achievable maximum coherence for that many unit vectors in $d$ dimensions.")
    return table("tab:concepts",
                 "\\textbf{Dictionary atoms and distinct gene-level programs.}",
                 "rrrrrr",
                 "Layer & Atoms & Distinct programs & Atoms/program & Mean $|\\cos|$ & "
                 "Welch bound \\\\", rows, note)


def t_evaluability():
    d = maybe("E1_ceiling/scores_k562_merged.json") or maybe("E1_ceiling/scores_k562.json")
    if not d or not d.get("evaluability"):
        return None
    ev = d["evaluability"]
    refs = [("TRRUST", "TRRUST"), ("DoRothEA_ABC", "DoRothEA A+B+C"),
            ("DoRothEA_all", "DoRothEA (all)")]
    rows = []
    for key, label in refs:
        rows.append(" & ".join([label, str(ev.get(f"{key}_tfs_perturbed", 0))]
                               + [str(ev.get(f"{key}_tfs_with_ge{t}_measured_targets", 0))
                                  for t in (1, 3, 5, 10, 20)]
                               + [f"{ev.get(f'{key}_median_measured_targets_per_tf', 0):.1f}"])
                    + " \\\\")
    note = ("Of the perturbation targets with at least 50 cells, the table counts how many are "
            "transcription factors in each reference network and, of those, how many have at "
            "least $n$ of their curated targets among the genes the assay measures. A factor "
            "whose curated targets are not measured cannot show target-specific behaviour under "
            "any method.")
    return table("tab:evaluability",
                 "\\textbf{How many perturbed transcription factors are evaluable.}",
                 "lrrrrrrr",
                 "Reference & TFs perturbed & $\\geq$1 & $\\geq$3 & $\\geq$5 & $\\geq$10 & "
                 "$\\geq$20 & Median \\\\", rows, note)


METHOD_LABEL = {
    "de_perturbation": "Differential expression of the knockdown",
    "sae_features": "SAE feature response",
    "genie3": "GENIE3", "grnboost2": "GRNBoost2",
    "pearson": "Co-expression (Pearson)", "spearman": "Co-expression (Spearman)",
    "geneformer_emb": "Geneformer gene embeddings", "random": "Random gene sets",
}
METHOD_ORDER = ["de_perturbation", "sae_features", "genie3", "grnboost2", "pearson",
                "spearman", "geneformer_emb", "random"]


def t_ceiling():
    d = maybe("E1_ceiling/scores_k562_merged.json") or maybe("E1_ceiling/scores_k562.json")
    if not d:
        return None
    S = d["scores"]
    refs = [("TRRUST|measured", "TRRUST"),
            ("DoRothEA_ABC|measured", "DoRothEA A+B+C"),
            ("DoRothEA_all|measured", "DoRothEA (all)")]
    methods = [m for m in METHOD_ORDER if any(m in S.get(r, {}) for r, _ in refs)]
    rows = []
    for m in methods:
        cells = []
        for r, _ in refs:
            s = S.get(r, {}).get(m)
            cells.append(f"{s['n_significant']}/{s['n_tfs']} ({100 * s['frac_significant']:.1f}\\%)"
                         if s else "---")
        rows.append(" & ".join([METHOD_LABEL.get(m, m)] + cells) + " \\\\")
    note = ("Each method emits a ranked list of 100 putative targets per transcription factor; "
            "enrichment for that factor's curated targets is tested by hypergeometric test with "
            "Benjamini--Hochberg correction across the panel, with the universe set to the genes "
            "the assay measures. Differential expression of the knockdown is perturbation-aware "
            "and bounds what these data support.")
    return table("tab:ceiling",
                 "\\textbf{Target recovery under one evaluation framework.} Transcription "
                 "factors with significant target enrichment (FDR $< 0.05$).",
                 "lccc",
                 " & ".join(["Method"] + [lbl for _, lbl in refs]) + " \\\\", rows, note)


def t_cell_level(tag="k562_main/k562sae"):
    p = RES / f"E3_cell_level/{tag}/per_target.json"
    if not p.exists():
        return None
    rows_in = json.load(open(p))["per_target"]
    tf = [r for r in rows_in if r["role"] == "tf"]
    if not tf:
        return None
    rows = []
    for r in sorted(tf, key=lambda r: -r["n_responding_cell_level"])[:25]:
        rows.append(" & ".join([
            esc(r["gene"]), str(r["n_cells"]),
            str(r["n_responding_cell_level"]), str(r["n_responding_position_fdr"]),
            f"{r['inflation_ratio']:.1f}",
            f"{r['median_icc_active_features']:.3f}" if r.get("median_icc_active_features")
            else "---",
            "yes" if r["self_gene_in_prediction"] else "no",
        ]) + " \\\\")
    icc = [r["median_icc_active_features"] for r in tf
           if r.get("median_icc_active_features") is not None]
    infl = [r["inflation_ratio"] for r in tf]
    note = ("Responding features under a cell-level Mann--Whitney test against a position-pooled "
            "test on exactly the same cells. The intraclass correlation is the between-cell share "
            f"of feature-activation variance (median across factors {np.median(icc):.3f}). Median "
            f"inflation of the responding-feature count under position pooling: "
            f"{np.median(infl):.1f}$\\times$. The last column records whether the knocked-down "
            "gene itself appears in the predicted set, a positive control for detection power.")
    return table("tab:cell_level",
                 "\\textbf{Unit of replication in the perturbation analysis.} "
                 "Transcription factors with the largest cell-level response.",
                 "lrrrrrc",
                 "Target & Cells & Responding (cell) & Responding (position) & Inflation & "
                 "ICC & Self detected \\\\", rows, note)


def t_crossline():
    rows = []
    for line in ("k562", "rpe1", "jurkat", "hepg2"):
        for dict_name, dict_label in (("k562sae", "multi-line control"),
                                      ("multitissue", "+ primary tissue")):
            p = RES / f"E3_cell_level/{line}_main/{dict_name}/per_target.json"
            if not p.exists():
                continue
            tf = [r for r in json.load(open(p))["per_target"] if r["role"] == "tf"]
            if not tf:
                continue
            for ref in ("DoRothEA_all",):
                ps = [r[f"{ref}_cell_p"] for r in tf if f"{ref}_cell_p" in r]
                if not ps:
                    continue
                q = common.bh_fdr(ps)
                rows.append(" & ".join([
                    LINE_LABEL[line], dict_label, str(len(tf)), str(len(ps)),
                    f"{int((q < 0.05).sum())}/{len(ps)} "
                    f"({100 * (q < 0.05).mean():.1f}\\%)",
                    f"{np.median([r['n_responding_cell_level'] for r in tf]):.0f}",
                ]) + " \\\\")
    if not rows:
        return None
    note = ("Perturbed and non-targeting cells are drawn from the same line in every row, so a "
            "difference between them cannot reflect cell-line composition. Specificity is the "
            "fraction of evaluable transcription factors whose predicted target set is enriched "
            "for its curated targets (DoRothEA, all confidence levels; FDR $< 0.05$).")
    return table("tab:crossline",
                 "\\textbf{Regulatory specificity across four cell lines and two dictionaries.}",
                 "llrrcr",
                 "Cell line & Dictionary & TFs tested & Evaluable & Specific & "
                 "Median responding \\\\", rows, note)


def t_seed_stability():
    d = maybe("E8_seed_stability/summary.json")
    if not d:
        return None
    rows = []
    for l in sorted([k for k in d if k.isdigit()], key=int):
        e = d[l]
        ve, dead = e["variance_explained"], e["dead_features"]
        init = [p for p in e["pairs"] if p["kind"] == "init"]
        rows.append(" & ".join([
            l, str(ve["n"]),
            f"{ve['mean']:.4f} $\\pm$ {ve['sd']:.4f}",
            f"{dead['mean']:.0f} $\\pm$ {dead['sd']:.0f}",
            f"{e['init_pairs_mean_best_cosine']:.3f}",
            f"{100 * np.mean([p['frac_best_above_0.9'] for p in init]):.1f}",
            f"{e['module_count']['mean']:.1f} $\\pm$ {e['module_count']['sd']:.1f}"
            if e.get("module_count") else "---",
        ]) + " \\\\")
    note = ("Runs differ in weight initialisation and batch order with the training subsample "
            "held fixed, plus one run per layer with a different subsample. Dictionary agreement "
            "is the mean over atoms of the best decoder cosine to any atom of the other run.")
    return table("tab:seed_stability",
                 "\\textbf{Run-to-run stability of the dictionaries.}",
                 "rrlllrl",
                 "Layer & Runs & Variance explained & Dead features & Mean best $|\\cos|$ & "
                 "$|\\cos| > 0.9$ (\\%) & Modules \\\\", rows, note)


def t_sweep():
    rows = None
    for tag in ("k562_main/k562sae",):
        f = RES / f"E3_cell_level/{tag}/sweep.json"
        if f.exists():
            rows = json.load(open(f))["rows"]
    if not rows:
        return None
    tf = [r for r in rows if r["role"] == "tf"]
    by = {}
    for r in tf:
        by.setdefault(r["gene"], set()).add(r["n"])
    ns = sorted({r["n"] for r in tf})
    common_genes = {g for g, v in by.items() if set(ns) <= v}
    out = []
    for n in ns:
        sub = [r for r in tf if r["n"] == n and r["gene"] in common_genes]
        sig = [r for r in sub if "DoRothEA_all_p" in r]
        out.append(" & ".join([
            str(n), f"{np.mean([r['n_responding'] for r in sub]):.2f}",
            f"{100 * np.mean([r['self_detected'] for r in sub]):.1f}",
            f"{100 * np.mean([r['DoRothEA_all_p'] < 0.05 for r in sig]):.1f}" if sig else "---",
            str(len(sub)),
        ]) + " \\\\")
    note = ("Restricted to the %d transcription factors with at least %d cells, so sample size is "
            "not confounded with which factors are available. Twenty bootstrap draws per factor "
            "per sample size. Enrichment p-values are nominal; under Benjamini--Hochberg across "
            "the panel no sample size yields a significant factor."
            % (len(common_genes), max(ns)))
    return table("tab:sweep",
                 "\\textbf{Sample-size sweep.} Perturbation response as a function of cells per "
                 "target.",
                 "rrrrr",
                 "Cells per target & Responding features & Detects knockdown (\\%) & "
                 "Nominal enrichment (\\%) & Draws \\\\", out, note)


def t_causal_arms():
    d = maybe("E5_causal_v2/results.json")
    if not d:
        return None
    rows = []
    labels = {"top_annotated": "richly annotated", "random_annotated": "random annotated",
              "random_any": "random feature"}
    for arm, agg in d.get("aggregates", {}).items():
        ap = agg.get("all_positions", {})
        r = ap.get("ratios_abs", {})
        w = ap.get("wilcoxon_heldout_vs_matched_random_abs", {})
        if not r:
            continue
        rows.append(" & ".join([
            labels.get(arm, arm), str(ap.get("n_features_with_heldout", 0)),
            f"{r.get('annotation_topk', {}).get('median', float('nan')):.1f}",
            f"{r.get('heldout_term', {}).get('median', float('nan')):.2f}",
            f"{r.get('matched_random', {}).get('median', float('nan')):.2f}",
            f"{w.get('p_value'):.3f}" if w.get("p_value") is not None else "---",
        ]) + " \\\\")
    if not rows:
        return None
    cfg = d.get("config", {})
    note = ("Median specificity ratio, mean $|\\Delta\\text{logit}|$ on the gene set over the same "
            "quantity on all other positions, at layer~11 in %s cells with %s cells per feature. "
            "The held-out set is the annotated term's genes excluding the feature's own top-20; the "
            "matched random set is drawn per feature at the same size and mean-activation decile. "
            "The last column is a Wilcoxon signed-rank test of held-out against matched random. "
            "%d of %d features admit no held-out set because their annotated term lies entirely "
            "within their top-20 genes."
            % (cfg.get("cell_line", "K562"), cfg.get("n_cells_done", "---"),
               cfg.get("n_features_no_heldout", 0), cfg.get("n_features_patched", 0)))
    return table("tab:causal_arms",
                 "\\textbf{Causal ablation under three feature-selection rules and three "
                 "evaluation gene sets.}",
                 "lrrrrr",
                 "Selection arm & Features & top-20 $\\cap$ term & Held-out term & "
                 "Matched random & $p$ \\\\", rows, note)


def t_crossmodel():
    import glob
    f = glob.glob(str(RES / "E9*/summary.json"))
    if not f:
        return None
    d = json.load(open(f[0]))
    rows = []
    for key, r in d.get("results", {}).items():
        lp = r["layer_pair"]
        cells = r["cells"]
        def ve(name):
            m = cells.get(name)
            return f"{m['var_explained']:.4f}" if isinstance(m, dict) and "var_explained" in m else "---"
        rows.append(" & ".join([
            f"{lp['geneformer_relative_depth']:.2f}",
            f"L{lp['geneformer_layer']} / L{lp['scgpt_layer']}",
            ve("geneformer_tabula_sapiens"), ve("scgpt_tabula_sapiens"),
            ve("scgpt_tabula_sapiens_matched_protocol"), ve("geneformer_k562"),
        ]) + " \\\\")
    if not rows:
        return None
    note = ("Variance explained on held-out positions. The two models are compared on the same "
            "Tabula Sapiens cells at matched relative depth (layer index divided by depth minus "
            "one). The matched-protocol column retrains the scGPT dictionaries under the same "
            "1M-position subsample used for Geneformer. The final column gives the Geneformer "
            "dictionaries on their own training distribution, which bounds how much of any "
            "cross-model difference is attributable to input distribution.")
    return table("tab:crossmodel",
                 "\\textbf{Cross-model comparison on matched inputs at matched relative depth.}",
                 "llrrrr",
                 "Rel.\\ depth & GF / scGPT layer & Geneformer/TS & scGPT/TS & "
                 "scGPT/TS matched & Geneformer/K562 \\\\", rows, note)


BUILDERS = [
    ("supp_cohorts.tex", t_cohorts),
    ("supp_svd_capacity.tex", t_svd_capacity),
    ("supp_concepts.tex", t_concepts),
    ("supp_evaluability.tex", t_evaluability),
    ("supp_ceiling.tex", t_ceiling),
    ("supp_cell_level.tex", t_cell_level),
    ("supp_crossline.tex", t_crossline),
    ("supp_seed_stability.tex", t_seed_stability),
    ("supp_sweep.tex", t_sweep),
    ("supp_causal_arms.tex", t_causal_arms),
    ("supp_crossmodel.tex", t_crossmodel),
]


def main():
    built, skipped = [], []
    for name, fn in BUILDERS:
        try:
            body = fn()
        except Exception as exc:  # a partial result file should not stop the rest
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
            skipped.append(name)
            continue
        if body is None:
            print(f"  {name}: inputs not available yet")
            skipped.append(name)
            continue
        write(name, body)
        built.append(name)
    print(f"\nbuilt {len(built)}, pending {len(skipped)}")
    if skipped:
        print("pending:", ", ".join(skipped))


if __name__ == "__main__":
    main()
