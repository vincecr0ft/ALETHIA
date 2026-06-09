r"""Cross-campaign comparison: the ManifoldInformer against sibling event-set
foundation-model backbones, on the capability matrix.

Research question:
  Across a battery of capability-matched campaigns (coefficient recovery, label
  efficiency, OOD robustness, calibrated inference, unsupervised anomaly), how
  does the ManifoldInformer's closed-form in-context aggregator rank against
  mean-pool and attention aggregators trained IDENTICALLY — is its in-context
  structure a cross-campaign generalist advantage, or campaign-specific?

This is the multi-campaign, held-out extension of the prior single-gate null
(ALETHEIA_manifold_informer_landscape_practice.md §3/§4): there the Intention
ridge, mean-pool, and attention aggregators were indistinguishable on one
in-sample P3/P4 gate. Here all three share the SAME JEPA pretraining (same
per-event encoder, EMA target, VICReg, RS3L view term, d_emb=16) and differ
ONLY in the aggregator; each frozen representation is then scored on every
campaign. A non-learned hand-feature summary is included as the floor every FM
must beat to justify itself.

Backbones (event-set FMs, manifold_coord -> per-scenario vector of width 16):
  - ManifoldInformer  : closed-form Intention ridge aggregator (the repo model)
  - MeanPool          : DeepSets aggregator
  - Attention         : Set-Transformer PMA aggregator
  - RawSummary        : non-FM baseline (mean/std/quantiles of raw events)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "manifold-informer"))

import substrate as sub  # noqa: E402
import probes  # noqa: E402
from manifold_informer import ManifoldInformer  # noqa: E402
from manifold_informer_variants import MeanPoolInformer, AttnInformer  # noqa: E402

OUT = HERE / "output_matrix"
D_EMB = 16
N_CTX = 120          # context events per set; remainder are JEPA query events
PRETRAIN_STEPS = 500
BATCH = 32


def make_backbones():
    return {
        "ManifoldInformer": lambda: ManifoldInformer(
            d_event=2, d_emb=D_EMB, hidden=32, density_anchor_weight=0.0),
        "MeanPool": lambda: MeanPoolInformer(
            d_event=2, d_emb=D_EMB, hidden=32, density_anchor_weight=0.0),
        "Attention": lambda: AttnInformer(
            d_event=2, d_emb=D_EMB, hidden=32, density_anchor_weight=0.0),
    }


def pretrain(model, X1, X2, *, steps=PRETRAIN_STEPS, bs=BATCH, lr=2e-3, seed=0):
    """Identical JEPA pretraining for every backbone. Context = first N_CTX
    events; query = the rest; second view (X2) drives the RS3L view-invariance
    term. The Wilson coefficient never enters."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    n = X1.shape[0]
    model.train()
    for step in range(steps):
        idx = rng.choice(n, bs, replace=False)
        xb1 = X1[idx]
        ctx, qry = xb1[:, :N_CTX], xb1[:, N_CTX:]
        view2 = X2[idx][:, :N_CTX]
        out = model(ctx, qry, X_ctx_view2=view2, return_losses=True)
        opt.zero_grad(); out["total"].backward(); opt.step()
        model.update_target()
    return model


@torch.no_grad()
def rep_of(model, X):
    model.eval()
    return model.manifold_coord(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()


# --------------------------------------------------------------------------
# Campaign scorers — identical for every backbone's frozen representation.
# --------------------------------------------------------------------------


def campaign_recovery(Z, C, raw, seed=0):
    """Held-out, capacity-controlled coefficient-recovery probe."""
    r = probes.probe_with_floor(Z["train"], C["train"], Z["test"], C["test"],
                                raw["train"], raw["test"], seed=seed)
    return {"r2": r.r2, "floor_r2": r.floor_r2, "margin": r.margin}


def campaign_label_efficiency(Z, C, budgets=(25, 50, 100, 200, 450), seed=0):
    r2s = []
    for b in budgets:
        b = min(b, Z["train"].shape[0])
        r2s.append(probes.held_out_probe(
            Z["train"][:b], C["train"][:b], Z["test"], C["test"], seed=seed).r2)
    return {"auc": float(np.mean(r2s)), "r2_by_budget": [float(x) for x in r2s],
            "budgets": list(budgets)}


def campaign_ood(Z, C, r2_box, seed=0):
    r2_ood = probes.held_out_probe(Z["train"], C["train"], Z["ood"], C["ood"],
                                   seed=seed).r2
    return {"r2_ood": r2_ood, "r2_box": r2_box, "drop": r2_box - r2_ood}


def _train_gauss_head(Ztr, Ctr, *, steps=800, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    d, k = Ztr.shape[1], Ctr.shape[1]
    mean = torch.nn.Linear(d, k)
    logstd = torch.nn.Linear(d, k)
    opt = torch.optim.Adam(list(mean.parameters()) + list(logstd.parameters()), lr=lr)
    Z = torch.as_tensor(Ztr, dtype=torch.float32)
    Y = torch.as_tensor(Ctr, dtype=torch.float32)
    for _ in range(steps):
        m, ls = mean(Z), logstd(Z).clamp(-6, 3)
        var = torch.exp(2 * ls)
        loss = (0.5 * ((Y - m) ** 2 / var) + ls).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return mean, logstd


def campaign_calibration(Z, C, seed=0):
    """Fit a shared Gaussian head on the frozen rep, score held-out calibration."""
    mean, logstd = _train_gauss_head(Z["train"], C["train"], seed=seed)
    with torch.no_grad():
        Zt = torch.as_tensor(Z["test"], dtype=torch.float32)
        m = mean(Zt).numpy()
        s = np.exp(logstd(Zt).clamp(-6, 3).numpy())
    cov = probes.gaussian_coverage(C["test"], m, s)
    return {"calibration_error": cov["calibration_error"],
            "coverage": cov["coverage"], "levels": cov["levels"]}


def campaign_anomaly(Z, seed=0):
    """Unsupervised rep-space anomaly: fit a Gaussian on SM-only representations,
    score by Mahalanobis distance, separate held-out SM from BSM (box)."""
    Zsm = Z["sm"]
    n_fit = int(0.7 * len(Zsm))
    fit, bg = Zsm[:n_fit], Zsm[n_fit:]
    mu = fit.mean(0)
    cov = np.cov(fit.T) + 1e-3 * np.eye(fit.shape[1])
    inv = np.linalg.inv(cov)

    def maha(A):
        d = A - mu
        return np.einsum("ni,ij,nj->n", d, inv, d)

    bg_s, sig_s = maha(bg), maha(Z["test"])
    auc = probes.roc_auc(bg_s, sig_s)
    sig = probes.significance_at_background(bg_s, sig_s, bg_eff=0.05)
    return {"auc": auc, "significance": sig["significance"],
            "n_bg": int(len(bg)), "n_sig": int(len(sig_s))}


# Campaign metric directions (+1 = higher better, -1 = lower better) for ranking.
CAMPAIGN_KEYS = {
    "recovery": ("margin", +1),
    "label_efficiency": ("auc", +1),
    "ood_robustness": ("r2_ood", +1),
    "calibration": ("calibration_error", -1),
    "anomaly": ("auc", +1),
}


def run(seed: int = 0, write: bool = True) -> dict:
    t0 = time.time()
    data = sub.load_cache()
    splits = ["train", "test", "ood", "sm"]
    Xnp = {s: data[f"{s}_X1"] for s in splits}
    C = {s: data[f"{s}_c"] for s in ["train", "test", "ood"]}
    raw = {s: probes.raw_event_summary(data[f"{s}_X1"]) for s in ["train", "test"]}

    X1 = torch.as_tensor(data["train_X1"], dtype=torch.float32)
    X2 = torch.as_tensor(data["train_X2"], dtype=torch.float32)

    # Representations for every backbone (FMs pretrained; RawSummary is fixed).
    reps, meta = {}, {}
    for name, factory in make_backbones().items():
        print(f"pretraining {name} ...", flush=True)
        model = factory()
        pretrain(model, X1, X2, seed=seed)
        reps[name] = {s: rep_of(model, Xnp[s]) for s in splits}
        meta[name] = {"n_params": int(model.n_params), "d_rep": D_EMB}
        print(f"  done {name} ({time.time() - t0:.0f}s)", flush=True)

    reps["RawSummary"] = {s: probes.raw_event_summary(Xnp[s]) for s in splits}
    meta["RawSummary"] = {"n_params": 0, "d_rep": reps["RawSummary"]["train"].shape[1]}

    # Score every campaign for every backbone.
    results = {}
    for name, Z in reps.items():
        rec = campaign_recovery(Z, C, raw, seed=seed)
        le = campaign_label_efficiency(Z, C, seed=seed)
        ood = campaign_ood(Z, C, rec["r2"], seed=seed)
        cal = campaign_calibration(Z, C, seed=seed)
        ano = campaign_anomaly(Z, seed=seed)
        results[name] = {"recovery": rec, "label_efficiency": le,
                         "ood_robustness": ood, "calibration": cal,
                         "anomaly": ano}
        print(f"scored {name}", flush=True)

    # Per-campaign ranks (1 = best) and mean rank across campaigns.
    ranks = {n: {} for n in results}
    for camp, (key, direction) in CAMPAIGN_KEYS.items():
        vals = {n: results[n][camp][key] for n in results}
        order = sorted(vals, key=lambda n: vals[n], reverse=(direction > 0))
        for rank, n in enumerate(order, 1):
            ranks[n][camp] = rank
    mean_rank = {n: float(np.mean(list(ranks[n].values()))) for n in ranks}

    out = {
        "research_question": (
            "Across capability-matched campaigns, how does the ManifoldInformer's "
            "closed-form in-context aggregator rank against mean-pool / attention "
            "aggregators trained identically, and a non-FM hand-feature baseline?"),
        "campaign_keys": {k: {"metric": v[0], "direction": v[1]}
                          for k, v in CAMPAIGN_KEYS.items()},
        "results": results, "ranks": ranks, "mean_rank": mean_rank, "meta": meta,
        "config": {"d_emb": D_EMB, "n_ctx": N_CTX, "pretrain_steps": PRETRAIN_STEPS,
                   "n_train": int(X1.shape[0])},
        "wall_seconds": time.time() - t0,
    }
    if write:
        OUT.mkdir(exist_ok=True)
        (OUT / "informer_campaigns.json").write_text(json.dumps(out, indent=2))
    return out


def run_multiseed(seeds=(0, 1, 2, 3, 4)) -> dict:
    """Run every backbone × campaign across seeds; rank on the per-campaign
    MEANS (robust to the near-ties in calibration / anomaly), and report the
    spread so the ranking's stability is visible."""
    runs = [run(seed=s, write=False) for s in seeds]
    backbones = list(runs[0]["results"])

    # Aggregate each campaign metric: mean & std across seeds.
    agg = {n: {} for n in backbones}
    for n in backbones:
        for camp, (key, _) in CAMPAIGN_KEYS.items():
            xs = [r["results"][n][camp][key] for r in runs]
            agg[n][camp] = {"mean": float(np.mean(xs)), "std": float(np.std(xs)),
                            "metric": key}

    # Rank on the means.
    ranks = {n: {} for n in backbones}
    for camp, (key, direction) in CAMPAIGN_KEYS.items():
        means = {n: agg[n][camp]["mean"] for n in backbones}
        order = sorted(means, key=lambda n: means[n], reverse=(direction > 0))
        for rank, n in enumerate(order, 1):
            ranks[n][camp] = rank
    mean_rank = {n: float(np.mean(list(ranks[n].values()))) for n in backbones}

    # Per-seed mean-rank spread (stability of the headline ordering).
    seed_mean_rank = {n: [r["mean_rank"][n] for r in runs] for n in backbones}

    out = {
        "research_question": runs[0]["research_question"],
        "seeds": list(seeds),
        "campaign_keys": runs[0]["campaign_keys"],
        "agg": agg, "ranks": ranks, "mean_rank": mean_rank,
        "seed_mean_rank": seed_mean_rank, "meta": runs[0]["meta"],
        "config": runs[0]["config"],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "informer_campaigns_multiseed.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed", action="store_true")
    args = ap.parse_args()
    if args.multiseed:
        r = run_multiseed()
        print(f"\nmulti-seed (n={len(r['seeds'])}) mean rank across campaigns (1=best):")
        for n, mr in sorted(r["mean_rank"].items(), key=lambda kv: kv[1]):
            sp = r["seed_mean_rank"][n]
            print(f"  {n:18s} rank={mr:.2f}  per-seed=[{min(sp):.1f}-{max(sp):.1f}]  "
                  f"ranks={r['ranks'][n]}")
    else:
        r = run()
        print("\nmean rank across campaigns (1=best):")
        for n, mr in sorted(r["mean_rank"].items(), key=lambda kv: kv[1]):
            print(f"  {n:18s} {mr:.2f}   ranks={r['ranks'][n]}")
