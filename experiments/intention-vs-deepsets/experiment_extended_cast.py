"""Extended-cast FM comparison on the SMEFT morphing manifold.

Runs the four landscape entrants in architectures_extended.py through the
*same* battery and the *same* splits as the existing cast (Intention,
DeepSets, JEPA, regressor), so the rows merge directly into the Claim-A
architecture comparison. This realises A.1 item 1 of
manifoldinformer_FM_landscape_and_novelty.md: a real head-to-head among
architectures that could plausibly disclose the morphing manifold.

Splits: reuses output_representation_analysis/splits.pt (built by
representation_analysis.py). If absent, builds the matched splits with the
same sampler/seeds. Per model it reports:

  * predictive R² (median, p5) on held-out Y_query, in-box and out-of-box;
  * c-recoverability — joint R² of a linear and an MLP probe from the
    per-scenario representation to the 4 Wilson coefficients (the disclosure
    metric, in/out);
  * y-shuffle invariance — cosine of the representation under per-scenario
    label permutation (low = uses the m-y joint structure; ~1 = rate-blind);
  * effective rank of the representation.

Crash-resilient: per-model checkpoints under output_extended_cast/, and
summary.json is rewritten after each model so a kill loses at most one model.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import (make_oracle, sample_c, sample_c_shell, make_dataset,
                  scenarios_to_tensors, N_WC)
from representation_analysis import fit_c_probe, representation_diagnostics
from architectures_extended import EXTENDED_CAST

OUT_DIR = HERE / "output_extended_cast"
OUT_DIR.mkdir(exist_ok=True)
SUM_PATH = OUT_DIR / "summary.json"
SPLITS = HERE / "output_representation_analysis" / "splits.pt"

# Matched to representation_analysis.py so the rows are comparable.
K_CTX = 12
Q_QUERY = 32
N_TRAIN = 2000
N_EVAL = 200
N_META_STEPS = 3000
LR = 1e-3
BATCH_S = 32
SEED = 0


# ---- data ----------------------------------------------------------------
def build_data():
    if SPLITS.exists():
        print(f"# reuse splits {SPLITS.relative_to(HERE.parent.parent)}")
        return torch.load(SPLITS, weights_only=False)
    print("# splits.pt absent — building matched splits")
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train = make_dataset(N_TRAIN, lambda r: sample_c(r, 1)[0],
                         oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=1)
    eval_in = make_dataset(N_EVAL, lambda r: sample_c(r, 1)[0],
                           oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=42)
    eval_out = make_dataset(N_EVAL, lambda r: sample_c_shell(r, 1)[0],
                            oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=43)
    return {"train_scen": train, "eval_in_scen": eval_in,
            "eval_out_scen": eval_out,
            "train_t": scenarios_to_tensors(train),
            "eval_in_t": scenarios_to_tensors(eval_in),
            "eval_out_t": scenarios_to_tensors(eval_out)}


# ---- metrics -------------------------------------------------------------
def r2_per_scenario(y_pred, y_true):
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def summarise(y_pred, y_true):
    r2 = r2_per_scenario(y_pred, y_true)
    return {"r2_median": float(np.median(r2)),
            "r2_p5": float(np.percentile(r2, 5)),
            "r2_mean": float(np.mean(r2))}


# ---- training (matched to representation_analysis) -----------------------
def train(model, train_t, n_steps=N_META_STEPS, lr=LR, batch_s=BATCH_S, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    S = train_t["M_ctx"].size(0)
    for _ in range(n_steps):
        idx = rng.choice(S, size=min(batch_s, S), replace=False)
        yp = model(train_t["M_ctx"][idx], train_t["Y_ctx"][idx],
                   train_t["M_query"][idx])
        loss = ((yp - train_t["Y_query"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_fm(model, t):
    model.eval()
    return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


@torch.no_grad()
def extract(model, t):
    model.eval()
    return model.representation(t["M_ctx"], t["Y_ctx"]).cpu().numpy()


def y_shuffle_cosine(model, t, n_perms=5, seed=7):
    rng = np.random.default_rng(seed)
    R0 = extract(model, t)
    R0n = R0 / (np.linalg.norm(R0, axis=1, keepdims=True) + 1e-12)
    sims = []
    for _ in range(n_perms):
        Y = t["Y_ctx"].clone()
        for s_i in range(Y.size(0)):
            Y[s_i] = Y[s_i, rng.permutation(Y.size(1))]
        Rp = extract(model, {**t, "Y_ctx": Y})
        Rpn = Rp / (np.linalg.norm(Rp, axis=1, keepdims=True) + 1e-12)
        sims.append(float(np.mean(np.sum(R0n * Rpn, axis=1))))
    return float(np.mean(sims))


# ---- main ----------------------------------------------------------------
def main():
    bundle = build_data()
    train_t = bundle["train_t"]
    eval_in_t, eval_out_t = bundle["eval_in_t"], bundle["eval_out_t"]
    c_train = np.stack([s["c"] for s in bundle["train_scen"]])
    c_in = np.stack([s["c"] for s in bundle["eval_in_scen"]])
    c_out = np.stack([s["c"] for s in bundle["eval_out_scen"]])

    summary = {}
    if SUM_PATH.exists():
        summary = json.loads(SUM_PATH.read_text())
    summary.setdefault("config", {
        "N_TRAIN": N_TRAIN, "N_EVAL": N_EVAL, "K_CTX": K_CTX,
        "Q_QUERY": Q_QUERY, "N_META_STEPS": N_META_STEPS,
        "BATCH_S": BATCH_S, "LR": LR, "SEED": SEED,
        "splits_source": str(SPLITS.relative_to(HERE.parent.parent)),
    })
    summary.setdefault("rows", {})

    for name, factory in EXTENDED_CAST.items():
        if name in summary["rows"]:
            print(f"# [skip] {name} already in summary.json")
            continue
        print(f"\n# {name}")
        model = factory()
        ckpt = OUT_DIR / f"{name}.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            print(f"  [load] {ckpt.name}")
        else:
            t0 = time.time()
            train(model, train_t)
            torch.save(model.state_dict(), ckpt)
            print(f"  [train] {time.time()-t0:.1f}s  n_params={model.n_params}")

        res_in = summarise(eval_fm(model, eval_in_t),
                           eval_in_t["Y_query"].numpy())
        res_out = summarise(eval_fm(model, eval_out_t),
                            eval_out_t["Y_query"].numpy())
        R_tr = extract(model, train_t)
        R_in, R_out = extract(model, eval_in_t), extract(model, eval_out_t)
        crec = {
            "linear_in": fit_c_probe(R_tr, c_train, R_in, c_in, hidden=None),
            "linear_out": fit_c_probe(R_tr, c_train, R_out, c_out, hidden=None),
            "mlp_in": fit_c_probe(R_tr, c_train, R_in, c_in, hidden=64),
            "mlp_out": fit_c_probe(R_tr, c_train, R_out, c_out, hidden=64),
        }
        diag = representation_diagnostics(R_in)
        ysh = y_shuffle_cosine(model, eval_in_t)

        summary["rows"][name] = {
            "n_params": int(model.n_params),
            "r2_median_in": res_in["r2_median"], "r2_p5_in": res_in["r2_p5"],
            "r2_median_out": res_out["r2_median"], "r2_p5_out": res_out["r2_p5"],
            "c_recoverability_in_linear": crec["linear_in"]["r2_joint"],
            "c_recoverability_out_linear": crec["linear_out"]["r2_joint"],
            "c_recoverability_in_mlp": crec["mlp_in"]["r2_joint"],
            "c_recoverability_out_mlp": crec["mlp_out"]["r2_joint"],
            "y_shuffle_cosine": ysh,
            "alignment_retained": 1.0 - ysh,
            "effective_rank": diag["effective_rank_1pct"],
            "participation_ratio": diag["participation_ratio"],
        }
        SUM_PATH.write_text(json.dumps(summary, indent=2))
        r = summary["rows"][name]
        print(f"  R²_in={r['r2_median_in']:+.4f}  R²_out={r['r2_median_out']:+.4f}  "
              f"c_lin_in={r['c_recoverability_in_linear']:+.3f}  "
              f"align={r['alignment_retained']:+.3f}  eff_rank={r['effective_rank']}")

    # ---- comparison table to stdout --------------------------------------
    print(f"\n# wrote {SUM_PATH.relative_to(HERE.parent.parent)}\n")
    print(f"{'architecture':<24}{'params':>8}{'R²_in':>9}{'R²_out':>9}"
          f"{'c_lin_in':>10}{'c_mlp_in':>10}{'align':>8}{'rank':>6}")
    for name, r in summary["rows"].items():
        print(f"{name:<24}{r['n_params']:>8}{r['r2_median_in']:>9.4f}"
              f"{r['r2_median_out']:>9.4f}{r['c_recoverability_in_linear']:>10.3f}"
              f"{r['c_recoverability_in_mlp']:>10.3f}"
              f"{r['alignment_retained']:>8.3f}{r['effective_rank']:>6}")
    return summary


if __name__ == "__main__":
    main()
