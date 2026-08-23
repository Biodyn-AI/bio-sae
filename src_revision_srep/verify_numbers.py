"""Check that headline numbers in the manuscript match the result files they come from.

Each entry names a claim, the value the manuscript states, and the path to read the true
value from. Anything that disagrees is printed and the script exits non-zero, so a number
cannot drift out of sync with its source between edits.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

RES = common.OUT_ROOT
TEX = common.PROJ / "paper/scientific_reports_revision/main.tex"
text = TEX.read_text(encoding="utf-8")

def jget(rel):
    return json.load(open(RES / rel))

checks, problems = [], []

def check(name, stated, actual, tol=0.0):
    ok = abs(stated - actual) <= tol if isinstance(actual, float) else stated == actual
    checks.append((name, stated, actual, ok))
    if not ok:
        problems.append(f"{name}: manuscript says {stated}, result file says {actual}")

def in_text(s):
    if s not in text:
        problems.append(f"string missing from manuscript: {s!r}")

# cohort composition
c = jget("E0_cohorts/composition.json")["atlas_extraction_cohort"]["by_cell_line"]
for line, n in (("Jurkat", 591), ("K562", 581), ("RPE1", 567), ("HepG2", 261)):
    check(f"cohort {line}", n, c[line.lower()])
in_text("591\nJurkat, 581 K562, 567 RPE1 and 261 HepG2")

# SVD capacity
L = jget("E6_svd_capacity/summary.json")["layers"]
check("svd rank L0", 403, L["0"]["svd_rank_matching_sae_varexpl"])
check("svd rank L17", 250, L["17"]["svd_rank_matching_sae_varexpl"])
check("k50 L11", 178, L["11"]["k_median_rho_50pct"])

# concepts
E7 = jget("E7_concepts/summary.json")
check("distinct programs", 18711, E7["resolutions"]["1.0"]["n_programs"])
check("total atoms", 82525, E7["n_features_total"])

# seed stability
E8 = jget("E8_seed_stability/summary.json")
check("L0 mean best cosine", 0.282, round(E8["0"]["init_pairs_mean_best_cosine"], 3), 0.001)
check("L11 mean best cosine", 0.294, round(E8["11"]["init_pairs_mean_best_cosine"], 3), 0.001)

# causal arms
E5 = jget("E5_causal_v2/results.json")["aggregates"]
for arm, stated in (("top_annotated", 1.04), ("random_annotated", 0.94), ("random_any", 0.97)):
    actual = E5[arm]["all_positions"]["ratios_abs"]["heldout_term"]["median"]
    check(f"heldout median {arm}", stated, round(actual, 2), 0.005)

# cross-line total
import numpy as np
tot_tf = tot_sig = 0
for line in ("k562", "rpe1", "jurkat", "hepg2"):
    p = RES / f"E3_cell_level/{line}_main/k562sae/per_target.json"
    if not p.exists():
        continue
    tf = [r for r in json.load(open(p))["per_target"] if r["role"] == "tf"]
    ps = [r["DoRothEA_all_cell_p"] for r in tf if "DoRothEA_all_cell_p" in r]
    q = common.bh_fdr(ps)
    tot_tf += len(ps); tot_sig += int((q < 0.05).sum())
check("cross-line significant", 1, tot_sig)
check("cross-line panel", 66, tot_tf)

# external datasets: the control comparison the paper's positive claim rests on
from scipy.stats import fisher_exact
norman = jget("E10_external/norman/results.json")["per_target"]
tf = [r for r in norman if r.get("role", "tf") == "tf"]
nt = [r for r in norman if r.get("role") == "non_tf_control"]
check("norman TF panel", 16, len(tf))
check("norman control panel", 10, len(nt))
in_text("3 of 13 cases (23\\%)")
in_text("$p = 0.62$")
in_text("8 of the 9 control perturbations")
# the paper must not still claim the superseded separation
for banned in ("15\\% of factors against 1\\%", "fifteenfold separation",
               "genuinely factor-specific target signal"):
    if banned in text:
        problems.append(f"superseded claim still present: {banned!r}")

print(f"{sum(1 for c in checks if c[3])}/{len(checks)} numeric checks passed")
for n, s, a, ok in checks:
    if not ok:
        print(f"  MISMATCH {n}: text {s} vs data {a}")
if problems:
    print("\nproblems:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("all manuscript numbers match their result files")
