"""E9 — Geneformer and scGPT dictionaries compared on the same cells at matched depth.

A dictionary fitted to Geneformer activations from a CRISPRi screen of one leukemia line
and a dictionary fitted to scGPT activations from a multi-tissue human atlas differ in two
things at once: the model that produced the activations, and the distribution of cells that
was fed to it. Any difference in variance explained, dead-feature count or dictionary
geometry is then unattributable. This script removes the second factor by fitting
Geneformer dictionaries to the *same* Tabula Sapiens cells the scGPT atlas was built from,
under the identical training recipe (4x expansion, k=32, 5 epochs, lr 3e-4, batch 4096, a
1M-position training subsample and a disjoint 100K-position hold-out), so the two models are
read out on matched data.

The two encoders also have different depths (18 vs 12 blocks), so "layer 11" does not denote
the same stage of computation in each. Layers are therefore paired by *relative* depth,
layer index / (n_layers - 1), and every reported comparison names the pair explicitly.

The design is a model x input-distribution grid:

    Geneformer / K562              published dictionaries, re-evaluated out of sample
    Geneformer / Tabula Sapiens    fitted here on the cached Tabula Sapiens activations
    scGPT      / Tabula Sapiens    published dictionaries, plus a protocol-matched refit
    scGPT      / K562              unavailable (no scGPT activations exist for that panel)

Per cell we report, on held-out positions: variance explained, alive and dead features,
mean absolute decoder coherence, atoms per input dimension, and the Welch bound for that
many unit vectors in that dimension. The Welch bound is what makes coherence comparable
across models: 4,608 atoms in 1,152 dimensions and 2,048 atoms in 512 dimensions are the
same overcompleteness but not the same packing problem, and only the ratio of measured
coherence to the bound is dimension-free.

Within the Geneformer arm we additionally apply the K562-fitted dictionary to the Tabula
Sapiens hold-out, with and without correcting the centering mean. This separates two
explanations of a cross-distribution drop that are otherwise conflated: a dictionary fitted
to the wrong distribution, versus the model genuinely encoding the new data along different
directions.

Outputs: experiments/revision_srep/E9_matched_crossmodel/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from sae_model import TopKSAE, SAETrainer  # noqa: E402  (common puts PROJ/src on sys.path)

OUT = common.OUT_ROOT / "E9_matched_crossmodel"
SAE_CACHE = OUT / "sae_models"

# ---------------------------------------------------------------------------
# Cached activation stores and published dictionaries
# ---------------------------------------------------------------------------
GF_TS_ACT = common.PROJ / "experiments/phase3_multitissue/ts_activations"
GF_K562_ACT = common.PHASE1
GF_K562_SAE = common.SAE_BASE
SCGPT_ACT = common.PROJ / "experiments/scgpt_atlas/activations"
SCGPT_SAE = common.PROJ / "experiments/scgpt_atlas/sae_models"

GF_N_LAYERS = 18
GF_D_MODEL = common.HIDDEN_DIM          # 1152
SCGPT_N_LAYERS = 12
SCGPT_D_MODEL = 512
SCGPT_LAYERS = tuple(range(SCGPT_N_LAYERS))

# The published K562 dictionaries were fitted on a 1,000,000-position subsample drawn from
# the layer memmap with numpy's legacy global generator seeded at 42 (src/02_train_sae.py,
# driven for every layer by src/02b_train_all_layers.py, so the draw is layer-independent).
# Reproducing that draw is what lets us score those dictionaries on positions they never saw.
K562_TRAIN_SUBSAMPLE = 1_000_000
K562_TRAIN_SEED = 42

# The published scGPT dictionaries were fitted on *all* extracted positions
# (scgpt_src/02_train_sae.py takes no subsample), so no hold-out exists for them.
SCGPT_PUBLISHED_USED_ALL_POSITIONS = True


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def welch_bound(n_vectors, d):
    """Minimum achievable maximum coherence for n unit vectors in R^d."""
    if n_vectors <= d:
        return 0.0
    return float(np.sqrt((n_vectors - d) / (d * (n_vectors - 1))))


def pick_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    return common.get_device()


def open_layer(act_dir, layer):
    path = Path(act_dir) / f"layer_{layer:02d}_activations.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.lib.format.open_memmap(str(path), mode="r"), path


def available_ts_layers():
    """Geneformer layers for which Tabula Sapiens activations were extracted."""
    layers = []
    for p in GF_TS_ACT.glob("layer_*_activations.npy"):
        try:
            layers.append(int(p.name.split("_")[1]))
        except ValueError:
            continue
    return sorted(set(layers))


def gather_rows(mm, idx, chunk=100_000, tag=""):
    """Materialise the given (sorted) row indices of a memmap as float32."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty((len(idx), mm.shape[1]), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(idx), chunk):
        end = min(start + chunk, len(idx))
        out[start:end] = mm[idx[start:end]]
        print(f"      {tag}read {end:,}/{len(idx):,} rows ({time.time() - t0:.0f}s)",
              flush=True)
    return out


def streaming_mean(mm, idx, chunk=100_000):
    """Mean over the given rows without materialising them."""
    idx = np.asarray(idx, dtype=np.int64)
    total = np.zeros(mm.shape[1], dtype=np.float64)
    for start in range(0, len(idx), chunk):
        end = min(start + chunk, len(idx))
        total += np.asarray(mm[idx[start:end]], dtype=np.float64).sum(axis=0)
    return (total / len(idx)).astype(np.float32)


def k562_training_indices(n_total, subsample=K562_TRAIN_SUBSAMPLE):
    """Reproduce the position subsample the published K562 dictionaries were fitted on."""
    np.random.seed(K562_TRAIN_SEED)
    idx = np.random.choice(n_total, subsample, replace=False)
    idx.sort()
    return idx


def split_train_val(n_total, n_train, n_val, seed):
    """Disjoint sorted training / hold-out position indices."""
    if n_train + n_val > n_total:
        raise ValueError(f"need {n_train + n_val} positions, only {n_total} available")
    rng = np.random.default_rng(seed)
    draw = rng.choice(n_total, size=n_train + n_val, replace=False)
    return np.sort(draw[:n_train]), np.sort(draw[n_train:])


def holdout_outside(n_total, used_idx, n_val, seed):
    """Hold-out positions drawn from the complement of an already-used index set."""
    mask = np.ones(n_total, dtype=bool)
    mask[used_idx] = False
    pool = np.flatnonzero(mask)
    if len(pool) < n_val:
        raise ValueError(f"only {len(pool)} unused positions, need {n_val}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n_val, replace=False))


def cells_covered(act_dir, layer, idx):
    """How many distinct source cells the held-out positions come from."""
    path = Path(act_dir) / f"layer_{layer:02d}_cell_ids.npy"
    if not path.exists():
        return None
    cell_ids = np.load(path, mmap_mode="r")
    return int(np.unique(np.asarray(cell_ids)[np.asarray(idx)]).size)


# ---------------------------------------------------------------------------
# Depth matching
# ---------------------------------------------------------------------------

def matched_scgpt_layer(gf_layer):
    """The scGPT block whose relative depth is closest to this Geneformer block's."""
    rel_gf = gf_layer / (GF_N_LAYERS - 1)
    best = min(SCGPT_LAYERS, key=lambda m: abs(m / (SCGPT_N_LAYERS - 1) - rel_gf))
    rel_sc = best / (SCGPT_N_LAYERS - 1)
    return {
        "geneformer_layer": int(gf_layer),
        "geneformer_n_layers": GF_N_LAYERS,
        "geneformer_relative_depth": round(float(rel_gf), 4),
        "scgpt_layer": int(best),
        "scgpt_n_layers": SCGPT_N_LAYERS,
        "scgpt_relative_depth": round(float(rel_sc), 4),
        "relative_depth_gap": round(float(abs(rel_sc - rel_gf)), 4),
        "description": (f"Geneformer block {gf_layer}/{GF_N_LAYERS - 1} "
                        f"(depth {rel_gf:.3f}) paired with scGPT block "
                        f"{best}/{SCGPT_N_LAYERS - 1} (depth {rel_sc:.3f})"),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(sae, act_mean, val_raw, device, batch=8192):
    """Score a dictionary on held-out positions.

    `act_mean` is the centering vector the dictionary is deployed with; passing a mean from
    a different distribution is what the transfer analysis does, so the residual statistics
    are accumulated in a way that keeps the systematic offset visible:

      var_explained             = 1 - sum_j Var(r_j) / sum_j Var(x_j)   (the quantity the
                                  published results.json files report; blind to a constant
                                  reconstruction offset because Var removes it)
      var_explained_incl_offset = 1 - sum_j E[r_j^2] / sum_j Var(x_j)   (charges the offset)
    """
    sae = sae.to(device).eval()
    mu = torch.as_tensor(np.asarray(act_mean, dtype=np.float32), device=device)
    n, d = val_raw.shape
    nf = sae.n_features

    sum_x = np.zeros(d, dtype=np.float64)
    sumsq_x = np.zeros(d, dtype=np.float64)
    sum_r = np.zeros(d, dtype=np.float64)
    sumsq_r = np.zeros(d, dtype=np.float64)
    active = np.zeros(nf, dtype=np.float64)
    l0_total = 0.0

    # Accumulators are float64 on the host; the per-batch reductions stay float32 because
    # MPS has no double precision. Batches are small enough that this is exact to ~1e-7.
    for start in range(0, n, batch):
        end = min(start + batch, n)
        x = torch.as_tensor(val_raw[start:end], device=device) - mu
        x_hat, h_sparse, _ = sae(x)
        r = x - x_hat
        sum_x += x.sum(dim=0).cpu().numpy().astype(np.float64)
        sumsq_x += (x ** 2).sum(dim=0).cpu().numpy().astype(np.float64)
        sum_r += r.sum(dim=0).cpu().numpy().astype(np.float64)
        sumsq_r += (r ** 2).sum(dim=0).cpu().numpy().astype(np.float64)
        fired = (h_sparse > 0).to(torch.float32)
        active += fired.sum(dim=0).cpu().numpy().astype(np.float64)
        l0_total += float(fired.sum().item())
        if device.type == "mps":
            torch.mps.empty_cache()

    var_x = sumsq_x / n - (sum_x / n) ** 2
    var_r = sumsq_r / n - (sum_r / n) ** 2
    msq_r = sumsq_r / n
    total_var = float(var_x.sum())
    act_freq = active / n
    dead = int((act_freq == 0).sum())

    return {
        "n_eval_positions": int(n),
        "d_model": int(d),
        "n_features": int(nf),
        "var_explained": float(1.0 - var_r.sum() / max(total_var, 1e-10)),
        "var_explained_incl_offset": float(1.0 - msq_r.sum() / max(total_var, 1e-10)),
        "mse": float(msq_r.sum() / d),
        "total_variance": total_var,
        "residual_offset_norm": float(np.linalg.norm(sum_r / n)),
        "l0_norm": float(l0_total / n),
        "alive_features": int(nf - dead),
        "dead_features": dead,
        "dead_feature_pct": float(100.0 * dead / nf),
        "_act_freq": act_freq,
    }


def decoder_geometry(sae, act_freq, seed, max_atoms=0):
    """Coherence and packing statistics of the dictionary, independent of the data.

    Decoder columns are re-normalised to unit norm before the Gram matrix is formed, so
    "coherence" means the cosine between dictionary directions rather than between
    unnormalised columns.
    """
    W = sae.W_dec.weight.detach().cpu().numpy().astype(np.float32)   # (d_model, n_features)
    d, nf = W.shape
    W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)

    def coherence(cols):
        if len(cols) < 2:
            return None, None
        A = W[:, cols]
        G = np.abs(A.T @ A)
        iu = np.triu_indices(G.shape[0], 1)
        v = G[iu]
        return float(v.mean()), float(v.max())

    rng = np.random.RandomState(seed)
    cols = np.arange(nf)
    if max_atoms and nf > max_atoms:
        cols = np.sort(rng.choice(nf, size=max_atoms, replace=False))
    mean_all, max_all = coherence(cols)

    alive_cols = np.flatnonzero(np.asarray(act_freq) > 0)
    n_alive = int(len(alive_cols))
    if max_atoms and n_alive > max_atoms:
        alive_cols = np.sort(rng.choice(alive_cols, size=max_atoms, replace=False))
    mean_alive, max_alive = coherence(alive_cols)

    wb_all = welch_bound(nf, d)
    wb_alive = welch_bound(n_alive, d)
    return {
        "d_model": int(d),
        "n_atoms": int(nf),
        "n_alive_atoms": n_alive,
        "atoms_per_dimension": float(nf / d),
        "alive_atoms_per_dimension": float(n_alive / d),
        "mean_abs_coherence": mean_all,
        "max_abs_coherence": max_all,
        "mean_abs_coherence_alive": mean_alive,
        "max_abs_coherence_alive": max_alive,
        "welch_bound": wb_all,
        "welch_bound_alive": wb_alive,
        "coherence_over_welch_bound": (float(mean_all / wb_all)
                                       if wb_all > 0 and mean_all is not None else None),
        "n_atoms_scored": int(len(cols)),
    }


def cell_metrics(sae, act_mean, val_raw, device, seed, batch, max_atoms):
    ev = evaluate(sae, act_mean, val_raw, device, batch=batch)
    act_freq = ev.pop("_act_freq")
    geom = decoder_geometry(sae, act_freq, seed, max_atoms=max_atoms)
    return {**ev, "decoder_geometry": geom}, act_freq


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sae_cached(run_dir, mm, train_idx, k, expansion, args, device, tag):
    """Fit a TopK dictionary on the given positions, or reuse a matching cached fit."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "sae_final.pt"
    mean_path = run_dir / "activation_mean.npy"
    cfg_path = run_dir / "train_config.json"

    d_model = int(mm.shape[1])
    n_features = expansion * d_model
    cfg = {
        "d_model": d_model,
        "n_features": n_features,
        "expansion": expansion,
        "k": k,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_train_positions": int(len(train_idx)),
        "train_index_checksum": int(np.sum(np.asarray(train_idx, dtype=np.int64) % 1_000_003)),
        "seed": args.seed,
    }

    if ckpt.exists() and mean_path.exists() and cfg_path.exists():
        cached = json.loads(cfg_path.read_text())
        if cached == cfg:
            print(f"    {tag}: cached fit reused ({ckpt})", flush=True)
            sae = TopKSAE.load(str(ckpt), device="cpu")
            sae.eval()
            return sae, np.load(mean_path), json.loads((run_dir / "training.json").read_text())
        print(f"    {tag}: cached fit has a different configuration, refitting", flush=True)

    print(f"    {tag}: reading {len(train_idx):,} training positions", flush=True)
    train_act = gather_rows(mm, train_idx, chunk=args.read_chunk, tag=f"{tag} ")
    act_mean = train_act.mean(axis=0).astype(np.float32)
    train_act -= act_mean[None, :]                      # centre in place: one copy in RAM

    common.seed_everything(args.seed)
    sae = TopKSAE(d_model=d_model, n_features=n_features, k=k)
    trainer = SAETrainer(sae, lr=args.lr, device=device)
    print(f"    {tag}: fitting {n_features} atoms in {d_model} dims for {args.epochs} "
          f"epoch(s) on {device}", flush=True)

    t0 = time.time()
    epoch_losses = []
    for epoch in range(args.epochs):
        e0 = time.time()
        loss = trainer.train_epoch(train_act, batch_size=args.batch_size, log_every=10 ** 7)
        epoch_losses.append(float(loss))
        print(f"      epoch {epoch + 1}/{args.epochs}: loss {loss:.6f} "
              f"({time.time() - e0:.0f}s)", flush=True)
    seconds = time.time() - t0

    sae.eval().to("cpu")
    sae.save(str(ckpt))
    np.save(mean_path, act_mean)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    record = {"epoch_losses": epoch_losses, "training_seconds": round(seconds, 1),
              "device": str(device), **cfg}
    (run_dir / "training.json").write_text(json.dumps(record, indent=2))

    del train_act
    return sae, act_mean, record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="Geneformer layers to analyse (default: all with cached "
                         "Tabula Sapiens activations)")
    ap.add_argument("--expansion", type=int, default=common.EXPANSION)
    ap.add_argument("--k", type=int, default=common.K_VAL)
    ap.add_argument("--train-subsample", type=int, default=1_000_000)
    ap.add_argument("--val-holdout", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--eval-batch", type=int, default=8192)
    ap.add_argument("--read-chunk", type=int, default=100_000)
    ap.add_argument("--coherence-max-atoms", type=int, default=0,
                    help="0 = use every atom when forming the Gram matrix")
    ap.add_argument("--skip-scgpt-refit", action="store_true",
                    help="report only the published scGPT dictionaries, which were fitted "
                         "on every extracted position and so have no hold-out")
    ap.add_argument("--verify-k562-split", type=int, default=1,
                    help="check the reproduced K562 training subsample against the stored "
                         "activation mean (done once, not per layer)")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SAE_CACHE.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    common.seed_everything(args.seed)
    t_start = time.time()

    ts_layers = available_ts_layers()
    layers = sorted(args.layers) if args.layers else ts_layers
    missing = [l for l in layers if l not in ts_layers]
    if missing:
        raise SystemExit(f"no cached Tabula Sapiens activations for Geneformer layer(s) "
                         f"{missing}; available: {ts_layers}")

    print("=" * 78, flush=True)
    print("E9 — matched-data, matched-relative-depth model comparison", flush=True)
    print(f"  device {device} | seed {args.seed} | Geneformer layers {layers}", flush=True)
    print("=" * 78, flush=True)

    pairs = [matched_scgpt_layer(l) for l in layers]
    for p in pairs:
        print("  " + p["description"], flush=True)

    # ---- position splits: layer-independent, so every layer is scored on the same cells --
    ts_mm0, ts_path0 = open_layer(GF_TS_ACT, layers[0])
    k562_mm0, k562_path0 = open_layer(GF_K562_ACT, layers[0])
    sc_mm0, sc_path0 = open_layer(SCGPT_ACT, pairs[0]["scgpt_layer"])
    n_ts, n_k562, n_sc = ts_mm0.shape[0], k562_mm0.shape[0], sc_mm0.shape[0]

    ts_train, ts_val = split_train_val(n_ts, args.train_subsample, args.val_holdout,
                                       args.seed)
    sc_train, sc_val = split_train_val(n_sc, args.train_subsample, args.val_holdout,
                                       args.seed + 1)
    k562_train_published = k562_training_indices(n_k562)
    k562_val = holdout_outside(n_k562, k562_train_published, args.val_holdout,
                               args.seed + 2)

    print(f"\n  Tabula Sapiens (Geneformer): {n_ts:,} positions -> "
          f"{len(ts_train):,} train / {len(ts_val):,} held out", flush=True)
    print(f"  Tabula Sapiens (scGPT):      {n_sc:,} positions -> "
          f"{len(sc_train):,} train / {len(sc_val):,} held out", flush=True)
    print(f"  K562 (Geneformer):           {n_k562:,} positions, published fit used "
          f"{len(k562_train_published):,}; {len(k562_val):,} held out from the complement",
          flush=True)

    provenance = {
        "geneformer_tabula_sapiens_activations": {
            "path": str(ts_path0), "shape": [int(x) for x in ts_mm0.shape],
            "dtype": str(ts_mm0.dtype), "layers_extracted": ts_layers},
        "geneformer_k562_activations": {
            "path": str(k562_path0), "shape": [int(x) for x in k562_mm0.shape],
            "dtype": str(k562_mm0.dtype)},
        "scgpt_tabula_sapiens_activations": {
            "path": str(sc_path0), "shape": [int(x) for x in sc_mm0.shape],
            "dtype": str(sc_mm0.dtype)},
        "geneformer_k562_dictionaries": str(GF_K562_SAE),
        "scgpt_dictionaries": str(SCGPT_SAE),
        "geneformer_tabula_sapiens_dictionaries": str(SAE_CACHE),
        "holdout_unit": ("gene position; cells are shared between training and held-out "
                         "positions, so this measures generalisation across positions, "
                         "not across cells"),
        "holdout_cells_covered": {
            "geneformer_tabula_sapiens": cells_covered(GF_TS_ACT, layers[0], ts_val),
            "geneformer_k562": cells_covered(GF_K562_ACT, layers[0], k562_val),
            "scgpt_tabula_sapiens": cells_covered(SCGPT_ACT, pairs[0]["scgpt_layer"], sc_val),
        },
        "k562_split_verification": None,
    }

    if args.verify_k562_split:
        print("\n  verifying the reproduced K562 training subsample ...", flush=True)
        stored = np.load(GF_K562_SAE / f"layer{layers[0]:02d}_x{args.expansion}_k{args.k}"
                         / "activation_mean.npy")
        recomputed = streaming_mean(k562_mm0, k562_train_published, chunk=args.read_chunk)
        diff = float(np.abs(stored - recomputed).max())
        cos = float(stored @ recomputed /
                    max(np.linalg.norm(stored) * np.linalg.norm(recomputed), 1e-12))
        provenance["k562_split_verification"] = {
            "layer": int(layers[0]),
            "max_abs_difference_to_stored_activation_mean": diff,
            "cosine_to_stored_activation_mean": cos,
            "reproduced": bool(diff < 1e-4),
        }
        print(f"    max |Δmean| {diff:.3e}, cosine {cos:.8f}", flush=True)

    del ts_mm0, k562_mm0, sc_mm0

    results = {}
    freq_store = {}

    for pair in pairs:
        gf_layer = pair["geneformer_layer"]
        sc_layer = pair["scgpt_layer"]
        print("\n" + "-" * 78, flush=True)
        print(pair["description"], flush=True)
        print("-" * 78, flush=True)
        cells = {}

        # --- Geneformer / Tabula Sapiens : fitted here on matched data -------------------
        ts_mm, _ = open_layer(GF_TS_ACT, gf_layer)
        run_dir = SAE_CACHE / f"geneformer_ts_layer{gf_layer:02d}_x{args.expansion}_k{args.k}"
        ts_sae, ts_mean, ts_train_record = train_sae_cached(
            run_dir, ts_mm, ts_train, args.k, args.expansion, args, device,
            tag=f"geneformer/TS L{gf_layer}")

        print(f"    reading {len(ts_val):,} held-out Tabula Sapiens positions", flush=True)
        ts_val_act = gather_rows(ts_mm, ts_val, chunk=args.read_chunk, tag="ts-val ")
        m, freq = cell_metrics(ts_sae, ts_mean, ts_val_act, device, args.seed,
                               args.eval_batch, args.coherence_max_atoms)
        m.update({"model": "Geneformer", "input_distribution": "Tabula Sapiens",
                  "layer": gf_layer, "dictionary_provenance": "fitted in this experiment",
                  "evaluation_positions_seen_in_training": False,
                  "training_seconds": ts_train_record["training_seconds"]})
        cells["geneformer_tabula_sapiens"] = m
        freq_store[f"L{gf_layer}_geneformer_tabula_sapiens"] = freq.astype(np.float32)
        print(f"    geneformer/TS  VE {m['var_explained']:.4f} | alive "
              f"{m['alive_features']}/{m['n_features']} | |coh| "
              f"{m['decoder_geometry']['mean_abs_coherence']:.4f}", flush=True)

        # --- Geneformer / K562 : published dictionary, scored out of sample --------------
        k562_sae, k562_mean_t, k562_run = common.load_sae(gf_layer, sae_base=GF_K562_SAE,
                                                          device="cpu")
        k562_mean = k562_mean_t.numpy()
        k562_mm, _ = open_layer(GF_K562_ACT, gf_layer)
        print(f"    reading {len(k562_val):,} held-out K562 positions", flush=True)
        k562_val_act = gather_rows(k562_mm, k562_val, chunk=args.read_chunk, tag="k562-val ")
        m, freq = cell_metrics(k562_sae, k562_mean, k562_val_act, device, args.seed,
                               args.eval_batch, args.coherence_max_atoms)
        published = json.loads((k562_run / "results.json").read_text())["results"]
        m.update({"model": "Geneformer", "input_distribution": "K562 (Replogle control)",
                  "layer": gf_layer,
                  "dictionary_provenance": f"published, {k562_run}",
                  "evaluation_positions_seen_in_training": False,
                  "published_in_sample_var_explained": float(published["variance_explained"]),
                  "published_alive_features": int(published["alive_features"])})
        cells["geneformer_k562"] = m
        freq_store[f"L{gf_layer}_geneformer_k562"] = freq.astype(np.float32)
        print(f"    geneformer/K562 VE {m['var_explained']:.4f} | alive "
              f"{m['alive_features']}/{m['n_features']} | |coh| "
              f"{m['decoder_geometry']['mean_abs_coherence']:.4f}", flush=True)

        # --- Which distribution does the Geneformer arm's drop come from? ---------------
        transfer = {"pair": pair["description"], "layer": gf_layer}
        ev = evaluate(k562_sae, k562_mean, ts_val_act, device, batch=args.eval_batch)
        ev.pop("_act_freq")
        transfer["k562_dictionary_on_tabula_sapiens_k562_mean"] = ev
        ev2 = evaluate(k562_sae, ts_mean, ts_val_act, device, batch=args.eval_batch)
        ev2.pop("_act_freq")
        transfer["k562_dictionary_on_tabula_sapiens_ts_mean"] = ev2
        ev3 = evaluate(ts_sae, k562_mean, k562_val_act, device, batch=args.eval_batch)
        ev3.pop("_act_freq")
        transfer["tabula_sapiens_dictionary_on_k562_k562_mean"] = ev3
        transfer["summary"] = {
            "ts_dictionary_on_ts": cells["geneformer_tabula_sapiens"]["var_explained"],
            "k562_dictionary_on_ts": ev["var_explained"],
            "k562_dictionary_on_ts_after_recentring": ev2["var_explained"],
            "k562_dictionary_on_k562": cells["geneformer_k562"]["var_explained"],
            "delta_wrong_dictionary": (cells["geneformer_tabula_sapiens"]["var_explained"]
                                       - ev2["var_explained"]),
            "delta_wrong_centering_mean": ev2["var_explained"] - ev["var_explained"],
            "k562_dictionary_on_ts_incl_offset": ev["var_explained_incl_offset"],
            "k562_dictionary_on_ts_after_recentring_incl_offset":
                ev2["var_explained_incl_offset"],
            "delta_wrong_centering_mean_incl_offset": (ev2["var_explained_incl_offset"]
                                                       - ev["var_explained_incl_offset"]),
            "delta_ts_is_harder_for_the_model": (cells["geneformer_k562"]["var_explained"]
                                                 - cells["geneformer_tabula_sapiens"]
                                                 ["var_explained"]),
            "reading": ("delta_wrong_dictionary is the loss from using a dictionary fitted "
                        "to the wrong input distribution at the same layer of the same "
                        "model; delta_ts_is_harder_for_the_model is the loss that remains "
                        "once each distribution has its own dictionary, i.e. the part that "
                        "is a property of the data rather than of the fit"),
        }
        print(f"    transfer: TS-fit {transfer['summary']['ts_dictionary_on_ts']:.4f} vs "
              f"K562-fit-on-TS {ev['var_explained']:.4f} "
              f"(recentred {ev2['var_explained']:.4f})", flush=True)

        del k562_val_act
        del ts_val_act

        # --- scGPT / Tabula Sapiens : published dictionary at matched relative depth -----
        sc_mm, _ = open_layer(SCGPT_ACT, sc_layer)
        print(f"    reading {len(sc_val):,} scGPT positions", flush=True)
        sc_val_act = gather_rows(sc_mm, sc_val, chunk=args.read_chunk, tag="scgpt-val ")

        sc_sae, sc_mean_t, sc_run = common.load_sae(sc_layer, sae_base=SCGPT_SAE,
                                                    device="cpu")
        sc_mean = sc_mean_t.numpy()
        m, freq = cell_metrics(sc_sae, sc_mean, sc_val_act, device, args.seed,
                               args.eval_batch, args.coherence_max_atoms)
        sc_published = json.loads((sc_run / "results.json").read_text())
        m.update({"model": "scGPT", "input_distribution": "Tabula Sapiens",
                  "layer": sc_layer,
                  "dictionary_provenance": f"published, {sc_run}",
                  "evaluation_positions_seen_in_training": SCGPT_PUBLISHED_USED_ALL_POSITIONS,
                  "evaluation_caveat": ("the published scGPT dictionaries were fitted on "
                                        "every extracted position, so no hold-out exists "
                                        "for them; this figure is in sample"),
                  "published_in_sample_var_explained":
                      float(sc_published["results"]["variance_explained"]),
                  "published_n_training_positions": int(sc_published["config"]["n_samples"])})
        cells["scgpt_tabula_sapiens"] = m
        freq_store[f"L{gf_layer}_scgpt_tabula_sapiens"] = freq.astype(np.float32)
        print(f"    scgpt/TS       VE {m['var_explained']:.4f} (in sample) | alive "
              f"{m['alive_features']}/{m['n_features']} | |coh| "
              f"{m['decoder_geometry']['mean_abs_coherence']:.4f}", flush=True)

        # --- scGPT / Tabula Sapiens : refitted under the identical protocol --------------
        if args.skip_scgpt_refit:
            cells["scgpt_tabula_sapiens_matched_protocol"] = None
        else:
            sc_run_dir = (SAE_CACHE /
                          f"scgpt_ts_layer{sc_layer:02d}_x{args.expansion}_k{args.k}")
            sc_fit_sae, sc_fit_mean, sc_fit_record = train_sae_cached(
                sc_run_dir, sc_mm, sc_train, args.k, args.expansion, args, device,
                tag=f"scgpt/TS L{sc_layer}")
            m, freq = cell_metrics(sc_fit_sae, sc_fit_mean, sc_val_act, device, args.seed,
                                   args.eval_batch, args.coherence_max_atoms)
            m.update({"model": "scGPT", "input_distribution": "Tabula Sapiens",
                      "layer": sc_layer,
                      "dictionary_provenance": "fitted in this experiment",
                      "evaluation_positions_seen_in_training": False,
                      "purpose": ("matches the training subsample size, epoch count and "
                                  "hold-out protocol used for the Geneformer arm, so the "
                                  "two models are also matched on evaluation protocol"),
                      "training_seconds": sc_fit_record["training_seconds"]})
            cells["scgpt_tabula_sapiens_matched_protocol"] = m
            freq_store[f"L{gf_layer}_scgpt_ts_matched"] = freq.astype(np.float32)
            print(f"    scgpt/TS-refit VE {m['var_explained']:.4f} | alive "
                  f"{m['alive_features']}/{m['n_features']} | |coh| "
                  f"{m['decoder_geometry']['mean_abs_coherence']:.4f}", flush=True)

        del sc_val_act

        cells["scgpt_k562"] = None
        results[str(gf_layer)] = {"layer_pair": pair, "cells": cells,
                                  "distribution_transfer": transfer}

    # ---- summary -----------------------------------------------------------------------
    summary = {
        "experiment": "E9_matched_crossmodel",
        "question": ("Do Geneformer and scGPT dictionaries differ because the models differ, "
                     "or because they were read out on different cells at different depths?"),
        "config": {
            "geneformer_layers": layers,
            "expansion": args.expansion,
            "k": args.k,
            "train_subsample": args.train_subsample,
            "val_holdout": args.val_holdout,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "device": str(device),
            "coherence_max_atoms": args.coherence_max_atoms,
            "scgpt_refit": not args.skip_scgpt_refit,
        },
        "design": {
            "factors": {
                "model": ["Geneformer (18 blocks, d=1152)", "scGPT (12 blocks, d=512)"],
                "input_distribution": ["K562 (Replogle control cells)",
                                       "Tabula Sapiens (immune, kidney, lung)"],
            },
            "cells": {
                "geneformer_k562": {
                    "model": "Geneformer", "input_distribution": "K562",
                    "dictionary": "published (experiments/phase1_k562/sae_models)",
                    "evaluation": "held out from the reproduced 1M-position training draw",
                },
                "geneformer_tabula_sapiens": {
                    "model": "Geneformer", "input_distribution": "Tabula Sapiens",
                    "dictionary": "fitted in this experiment on the cached Geneformer "
                                  "Tabula Sapiens activations",
                    "evaluation": "held out by construction",
                },
                "scgpt_tabula_sapiens": {
                    "model": "scGPT", "input_distribution": "Tabula Sapiens",
                    "dictionary": "published (experiments/scgpt_atlas/sae_models)",
                    "evaluation": "in sample — the published fit used every position",
                },
                "scgpt_tabula_sapiens_matched_protocol": {
                    "model": "scGPT", "input_distribution": "Tabula Sapiens",
                    "dictionary": "refitted here under the Geneformer arm's protocol",
                    "evaluation": "held out by construction",
                },
                "scgpt_k562": {
                    "model": "scGPT", "input_distribution": "K562",
                    "dictionary": None,
                    "evaluation": None,
                    "status": "unavailable — no scGPT activations were extracted for the "
                              "Replogle K562 panel, and scGPT cannot be run in this "
                              "environment",
                },
            },
            "depth_matching": (
                "Geneformer has 18 transformer blocks and scGPT has 12, so absolute layer "
                "indices are not comparable. Blocks are paired by relative depth "
                "index/(n_layers-1) and every comparison names the pair it uses."),
            "controlled": ["input cells", "expansion factor", "k", "epochs", "learning "
                           "rate", "batch size", "training subsample size",
                           "hold-out protocol (for the matched-protocol cells)"],
            "not_controlled": ["d_model (1152 vs 512)", "tokenisation and sequence length",
                               "pre-training corpus of each model"],
        },
        "layer_pairs": pairs,
        "results": results,
        "provenance": provenance,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    common.write_json(OUT / "summary.json", summary, seed=args.seed)
    if freq_store:
        np.savez_compressed(OUT / "feature_activation_frequencies.npz", **freq_store)
        print(f"  wrote {OUT / 'feature_activation_frequencies.npz'}", flush=True)

    # ---- printed table -----------------------------------------------------------------
    header = (f"{'pair':<16}{'cell':<40}{'VE':>8}  {'alive/atoms':>13}  {'|coh|':>7}"
              f"  {'Welch':>7}  {'|coh|/Welch':>11}  {'atoms/dim':>9}")
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for gf_layer in layers:
        block = results[str(gf_layer)]
        tag = f"GF L{gf_layer}/sc L{block['layer_pair']['scgpt_layer']}"
        for name, m in block["cells"].items():
            if m is None:
                print(f"{tag:<16}{name:<40}{'n/a':>8}  {'unavailable':>13}", flush=True)
                continue
            g = m["decoder_geometry"]
            alive = f"{m['alive_features']}/{m['n_features']}"
            print(f"{tag:<16}{name:<40}{m['var_explained']:>8.4f}  {alive:>13}  "
                  f"{g['mean_abs_coherence']:>7.4f}  {g['welch_bound']:>7.4f}  "
                  f"{g['coherence_over_welch_bound']:>11.2f}  "
                  f"{g['atoms_per_dimension']:>9.1f}", flush=True)

    print(f"\ndone in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
