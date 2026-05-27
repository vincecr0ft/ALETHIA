"""ALETHIA full-chain demonstration run.

ONE active scenario: a target Wilson configuration in the withheld band
|c_lq^(3)| in [0.6, 1.0]. The Intention FM has been pretrained on the
analytic SMEFT oracle with that band excluded. The loop watches the FM
predict in this scenario from a thin seed context; drift detectors fire
because the FM's pretrained psi_theta basis is weak in the band and the
seed context's m-support is narrow; EPIG selects more oracle queries in
the high-leverage m-regions; the context grows; predictions and coverage
recover.

Run with:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/full-chain-run/run.py
"""
from __future__ import annotations

import os, sys, time, json, logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from phoenix.otel import register

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import (
    IntentionFM, M_REF,
    IntentionConformal,
    epig_acquire_m, target_set_entropy,
    DASCUSUMState, das_cusum_update,
    coverage_bh_test, kappa_drift, aggregate_action,
)

TRACER_PROVIDER = register(project_name="alethia", auto_instrument=False,
                           protocol="http/protobuf")
tracer = TRACER_PROVIDER.get_tracer("alethia.full_chain")

# ----- Configuration -----
SEED = 2026
N_WC = 4
WITHHOLD_DIM = 2               # clq3
WITHHOLD_BAND = (0.6, 1.0)
C_TRAIN_BOX = 0.7
M_RANGE = (0.3, 2.3)
M_REF_TEV = 1.0

K_CTX = 12
Q_CTX = 32
N_PRETRAIN_SCEN = 1500
PRETRAIN_BATCH = 24
PRETRAIN_STEPS = 1500
LR = 1e-3

# ----- Loop configuration -----
N_LOOP_CYCLES = 400
PROBES_PER_CYCLE = 24
N_SEED_CTX = 8                 # thin seed: FM has narrow m-support
SEED_M_RANGE = (0.5, 1.0)      # seed context clustered in low-m only
N_CAL_POINTS = 400             # cal set drawn over full M_RANGE
ORACLE_BUDGET = 500
K_EPIG = 5

# ----- Thresholds (calibrated in agent c second-pass empirical) -----
CUSUM_H = 6.0
CUSUM_W = 30
CUSUM_K = 0.5
BH_ALPHA = 0.05
KAPPA_THRESHOLD = 1e3          # lowered from 1e4 to be sensitive to thin contexts
PROJ_RATIO_THRESHOLD = 2.5
PERSISTENCE_N = 3
CAL_PERSISTENCE_ESCALATE = 4   # consecutive cal fires that escalate to local_retrain
COOLDOWN_CYCLES = 3

# ----- Logging -----
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=OUT / "run.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
)
log = logging.getLogger("full_chain")
log.info("=== ALETHIA full-chain run starting ===")


def sample_c_training(n: int, rng: np.random.Generator) -> np.ndarray:
    """c ~ U([-0.7, 0.7]^4) minus |c_lq^(3)| in [0.6, 1.0]."""
    out = np.empty((n, N_WC))
    i = 0
    while i < n:
        c = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=N_WC)
        if abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c
            i += 1
    return out


def sample_c_broad(n: int, rng: np.random.Generator) -> np.ndarray:
    """c sampled from the full probe region (including withheld band)."""
    return rng.uniform(-1.0, 1.0, size=(n, N_WC))


def mu_at(oracle, c: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    C = np.tile(c, (len(m_values), 1))
    return oracle.truth(C, m_values)


# --------------------- Pretraining ---------------------

def pretrain_intention(oracle, rng) -> tuple:
    log.info("Pretraining: steps=%d batch=%d K=%d Q=%d",
             PRETRAIN_STEPS, PRETRAIN_BATCH, K_CTX, Q_CTX)
    torch.manual_seed(SEED)
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    opt = Adam(model.parameters(), lr=LR)

    pretrain_wall = 0.0
    losses = []
    with tracer.start_as_current_span("chain.pretrain") as span:
        span.set_attribute("aletheia.pretrain.n_steps", PRETRAIN_STEPS)
        span.set_attribute("aletheia.pretrain.batch", PRETRAIN_BATCH)
        span.set_attribute("aletheia.pretrain.d_psi", 16)
        span.set_attribute("aletheia.pretrain.withhold_dim_name", "c_lq^(3)")
        span.set_attribute("aletheia.pretrain.withhold_band_lo",
                           WITHHOLD_BAND[0])
        span.set_attribute("aletheia.pretrain.withhold_band_hi",
                           WITHHOLD_BAND[1])

        c_pool = sample_c_training(N_PRETRAIN_SCEN, rng)
        t0 = time.time()
        for step in range(PRETRAIN_STEPS):
            idx = rng.choice(N_PRETRAIN_SCEN, size=PRETRAIN_BATCH,
                             replace=False)
            cs = c_pool[idx]
            B, K, Q = PRETRAIN_BATCH, K_CTX, Q_CTX
            M_ctx_np = rng.uniform(*M_RANGE, size=(B, K))
            M_q_np = rng.uniform(*M_RANGE, size=(B, Q))
            C_ctx = np.repeat(cs, K, axis=0)
            C_q = np.repeat(cs, Q, axis=0)
            Y_ctx_np = oracle.truth(C_ctx, M_ctx_np.flatten()).reshape(B, K)
            Y_q_np = oracle.truth(C_q, M_q_np.flatten()).reshape(B, Q)

            M_ctx = torch.from_numpy(M_ctx_np).float()
            Y_ctx = torch.from_numpy(Y_ctx_np).float()
            M_q = torch.from_numpy(M_q_np).float()
            Y_q = torch.from_numpy(Y_q_np).float()

            y_pred = model(M_ctx, Y_ctx, M_q)
            loss = F.mse_loss(y_pred, Y_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

            if step % 100 == 0 or step == PRETRAIN_STEPS - 1:
                log.info("  step %4d/%d loss=%.6f", step, PRETRAIN_STEPS,
                         loss.item())
        pretrain_wall = time.time() - t0
        span.set_attribute("aletheia.pretrain.wall_seconds", pretrain_wall)
        span.set_attribute("aletheia.pretrain.final_loss", losses[-1])
        log.info("Pretrain done in %.1fs, final loss=%.6f",
                 pretrain_wall, losses[-1])
    return model, pretrain_wall, np.array(losses)


# --------------------- Loop ---------------------

class RegionTagger:
    """Tag m-values into 5 buckets over M_RANGE."""

    def __init__(self):
        self.edges = np.linspace(M_RANGE[0], M_RANGE[1], 6)

    def tag(self, m: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges[1:-1], m), 0,
                       len(self.edges) - 2)


def run_loop(oracle, model: IntentionFM, rng) -> dict:
    log.info("=== Loop starting ===")

    # ONE target scenario in the withheld band.
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = 0.8   # clq3 = +0.8 in the withheld band
    log.info("Target scenario c=%s", target_c.tolist())

    # Thin seed context: K=8 points clustered in [0.5, 1.0] only.
    M_ctx = rng.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
    Y_ctx = mu_at(oracle, target_c, M_ctx)
    log.info("Seed context: K=%d, m in %s", N_SEED_CTX, SEED_M_RANGE)

    # Calibration set: broader probe region (m in full M_RANGE for target c).
    # Per the invariant: cal set covers the distribution the model will be
    # queried on. For a single-scenario loop, the cal set is m-values across
    # the full M_RANGE for the same target c.
    M_cal = rng.uniform(*M_RANGE, size=N_CAL_POINTS)
    Y_cal = mu_at(oracle, target_c, M_cal)

    cc = IntentionConformal(n_strata=5, noise_frac=0.05)
    cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)

    # Target set for H_T (the headline monitor): fine m-grid.
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    Y_target_truth = mu_at(oracle, target_c, M_target)

    # Region tagger and baseline projection variance.
    tagger = RegionTagger()
    n_regions = 5
    train_M_sample = rng.uniform(*M_RANGE, size=300)
    Psi_train = model.psi_np(train_M_sample)
    eigs_train, vecs_train = np.linalg.eigh(
        Psi_train.T @ Psi_train + model.alpha * np.eye(model.d_psi))
    v_min_train = vecs_train[:, 0]
    train_proj_var = float(np.var(Psi_train @ v_min_train))

    # Streaming state.
    cusum_state = DASCUSUMState(
        buf=DASCUSUMState().buf.__class__(maxlen=CUSUM_W))
    coverage_counts_68 = np.zeros((n_regions, 2), dtype=int)
    coverage_counts_95 = np.zeros((n_regions, 2), dtype=int)
    history_flags = []

    # Trajectories.
    traj = dict(
        H_T=[], kappa=[], cov_68_by_region=[], cov_95_by_region=[],
        context_size=[], oracle_calls=[], drift_flags=[], action=[],
        cusum_S=[], cal_pvalue_min=[], proj_ratio=[],
    )
    oracle_calls = 0
    consec_recovered = 0
    cycles_done = 0
    last_action_cycle = -100
    consec_cal_fires = 0
    rmse_band_trace = []

    t0 = time.time()
    for cycle in range(N_LOOP_CYCLES):
        with tracer.start_as_current_span("chain.cycle") as cyc_span:
            cyc_span.set_attribute("aletheia.cycle.index", cycle)
            cyc_span.set_attribute("aletheia.cycle.context_size", len(M_ctx))

            # Sample probe m-values across full M_RANGE. Bias 50% to the
            # tails [1.5, 2.3] where seed context didn't reach.
            n_tail = int(0.5 * PROBES_PER_CYCLE)
            probe_m_tail = rng.uniform(1.5, M_RANGE[1], n_tail)
            probe_m_bulk = rng.uniform(*M_RANGE, PROBES_PER_CYCLE - n_tail)
            probe_m = np.concatenate([probe_m_tail, probe_m_bulk])
            rng.shuffle(probe_m)
            probe_y_truth = mu_at(oracle, target_c, probe_m)

            # FM predict.
            with tracer.start_as_current_span("tool.surrogate.predict") as sp:
                mu_pred = model.predict_np(M_ctx, Y_ctx, probe_m)
                lev = model.leverage(M_ctx, probe_m)
                sd_68 = cc.coverage_sigma(
                    model, M_ctx, Y_ctx, probe_m, 0.683)
                sd_95 = cc.coverage_sigma(
                    model, M_ctx, Y_ctx, probe_m, 0.954)
                sp.set_attribute("aletheia.fm.n_query", PROBES_PER_CYCLE)
                sp.set_attribute("aletheia.fm.leverage_mean",
                                 float(lev.mean()))
                sp.set_attribute("aletheia.fm.leverage_max",
                                 float(lev.max()))
                sp.set_attribute("aletheia.fm.mu_mean", float(mu_pred.mean()))
                sp.set_attribute("aletheia.fm.sigma_mean", float(sd_68.mean()))

            # Standardised residuals; per-region coverage tally.
            std_resid = (probe_y_truth - mu_pred) / np.maximum(sd_68, 1e-12)
            tags = tagger.tag(probe_m)
            cov68_hit = np.abs(probe_y_truth - mu_pred) < sd_68
            cov95_hit = np.abs(probe_y_truth - mu_pred) < sd_95
            for s in range(n_regions):
                mask = (tags == s)
                if mask.any():
                    coverage_counts_68[s, 0] += int(cov68_hit[mask].sum())
                    coverage_counts_68[s, 1] += int(mask.sum())
                    coverage_counts_95[s, 0] += int(cov95_hit[mask].sum())
                    coverage_counts_95[s, 1] += int(mask.sum())

            # Drift detection.
            with tracer.start_as_current_span("chain.drift.evaluate"):
                with tracer.start_as_current_span(
                        "tool.drift.accuracy.das_cusum") as ds:
                    z_mean = float(np.median(std_resid))
                    cusum_state, acc_fired, S_t = das_cusum_update(
                        cusum_state, z_mean, w=CUSUM_W, h=CUSUM_H, k=CUSUM_K)
                    ds.set_attribute("aletheia.drift.acc.S_t", S_t)
                    ds.set_attribute("aletheia.drift.acc.fired", acc_fired)

                with tracer.start_as_current_span(
                        "tool.drift.calibration.bh") as cs_:
                    cal_fired, pvals, fail_mask, bh_thr = coverage_bh_test(
                        coverage_counts_68[:, 0], coverage_counts_68[:, 1],
                        target_coverage=0.683, alpha=BH_ALPHA)
                    cs_.set_attribute("aletheia.drift.cal.global_pvalue",
                                      float(pvals.min()))
                    cs_.set_attribute("aletheia.drift.cal.failing_regions",
                                      int(fail_mask.sum()))
                    cs_.set_attribute("aletheia.drift.cal.fired", cal_fired)

                with tracer.start_as_current_span(
                        "tool.drift.coverage.kappa") as ks:
                    cov_fired, kappa, proj_ratio = kappa_drift(
                        model, M_ctx, probe_m, train_proj_var,
                        kappa_threshold=KAPPA_THRESHOLD,
                        proj_ratio_threshold=PROJ_RATIO_THRESHOLD)
                    ks.set_attribute("aletheia.drift.cov.kappa", kappa)
                    ks.set_attribute(
                        "aletheia.drift.cov.vmin_projection_ratio",
                        proj_ratio)
                    ks.set_attribute("aletheia.drift.cov.fired", cov_fired)

            # Persistence escalation: 4+ consecutive cal fires => escalate.
            consec_cal_fires = consec_cal_fires + 1 if cal_fired else 0
            forced_local = consec_cal_fires >= CAL_PERSISTENCE_ESCALATE

            with tracer.start_as_current_span("tool.drift.aggregate") as ag:
                action, target_signal = aggregate_action(
                    acc_fired, cal_fired, cov_fired,
                    history_flags, persistence_N=PERSISTENCE_N)
                history_flags.append((acc_fired, cal_fired, cov_fired))
                if forced_local and action in ("recal", "watch", "noop"):
                    action = "local_retrain"
                    target_signal = "cal_persist"
                flag_str = ("1" if acc_fired else "0") + \
                           ("1" if cal_fired else "0") + \
                           ("1" if cov_fired else "0")
                ag.set_attribute("aletheia.drift.combined_flag", flag_str)
                ag.set_attribute("aletheia.drift.action", action)
                if target_signal:
                    ag.set_attribute("aletheia.drift.target_signal",
                                     target_signal)

            # Act.
            if action in ("local_retrain", "global_retrain"):
                if cycle - last_action_cycle < COOLDOWN_CYCLES:
                    action = "cooled"
                elif oracle_calls + K_EPIG > ORACLE_BUDGET:
                    action = "budget_exhausted"
                else:
                    last_action_cycle = cycle
                    consec_cal_fires = 0
                    with tracer.start_as_current_span(
                            "tool.epig.select") as es:
                        # Pool of candidates: emphasise high-leverage regions.
                        M_pool = rng.uniform(*M_RANGE, size=300)
                        picked = epig_acquire_m(
                            model, M_ctx, Y_ctx, M_pool, M_target,
                            k=K_EPIG)
                        picked_m = M_pool[picked]
                        es.set_attribute("aletheia.epig.k", K_EPIG)
                        es.set_attribute("aletheia.epig.pool_size", 300)
                        es.set_attribute("aletheia.epig.picked_m_values",
                                         picked_m.round(3).tolist())

                    with tracer.start_as_current_span(
                            "tool.oracle.query") as oq:
                        picked_y = mu_at(oracle, target_c, picked_m)
                        oq.set_attribute("aletheia.oracle.fidelity_tier",
                                         "T1")
                        oq.set_attribute("aletheia.oracle.n_points", K_EPIG)
                        oq.set_attribute("aletheia.oracle.wilson_norm",
                                         float(np.linalg.norm(target_c)))

                    with tracer.start_as_current_span(
                            "tool.fm.update") as fu:
                        H_T_pre = target_set_entropy(
                            model, M_ctx, Y_ctx, M_target)
                        ctx_in = len(M_ctx)
                        M_ctx = np.concatenate([M_ctx, picked_m])
                        Y_ctx = np.concatenate([Y_ctx, picked_y])
                        H_T_post = target_set_entropy(
                            model, M_ctx, Y_ctx, M_target)
                        fu.set_attribute("aletheia.fm.context_size_in",
                                         ctx_in)
                        fu.set_attribute("aletheia.fm.context_size_out",
                                         len(M_ctx))
                        fu.set_attribute(
                            "aletheia.fm.target_entropy_H_T_pre", H_T_pre)
                        fu.set_attribute(
                            "aletheia.fm.target_entropy_H_T_post",
                            H_T_post)
                        fu.set_attribute(
                            "aletheia.fm.condition_number_kappa",
                            model.kappa_A(M_ctx))

                    cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)
                    oracle_calls += K_EPIG
            elif action in ("recal", "recal_then_check"):
                cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)

            # Track.
            H_T_cur = target_set_entropy(model, M_ctx, Y_ctx, M_target)
            kappa_cur = model.kappa_A(M_ctx)
            cov_68_per = np.where(
                coverage_counts_68[:, 1] > 0,
                coverage_counts_68[:, 0] /
                np.maximum(coverage_counts_68[:, 1], 1),
                np.nan)
            cov_95_per = np.where(
                coverage_counts_95[:, 1] > 0,
                coverage_counts_95[:, 0] /
                np.maximum(coverage_counts_95[:, 1], 1),
                np.nan)
            # RMSE on the target set (the band the FM should be predicting):
            mu_band = model.predict_np(M_ctx, Y_ctx, M_target)
            rmse_band = float(np.sqrt(np.mean((mu_band - Y_target_truth) ** 2)))
            rmse_band_trace.append(rmse_band)

            traj["H_T"].append(H_T_cur)
            traj["kappa"].append(kappa_cur)
            traj["cov_68_by_region"].append(cov_68_per.copy())
            traj["cov_95_by_region"].append(cov_95_per.copy())
            traj["context_size"].append(len(M_ctx))
            traj["oracle_calls"].append(oracle_calls)
            traj["drift_flags"].append(flag_str)
            traj["action"].append(action)
            traj["cusum_S"].append(S_t)
            traj["cal_pvalue_min"].append(float(pvals.min()))
            traj["proj_ratio"].append(proj_ratio)

            cyc_span.set_attribute("aletheia.cycle.H_T", H_T_cur)
            cyc_span.set_attribute("aletheia.cycle.kappa", kappa_cur)
            cyc_span.set_attribute("aletheia.cycle.oracle_calls",
                                   oracle_calls)
            cyc_span.set_attribute("aletheia.cycle.rmse_target_band",
                                   rmse_band)

            # Rolling window cov check: every 30 cycles inspect aggregate
            # cov_68 over [0.65, 0.72] band.
            if cycle % 30 == 29:
                tot_cov = coverage_counts_68[:, 0].sum()
                tot_n = max(coverage_counts_68[:, 1].sum(), 1)
                rolling_68 = tot_cov / tot_n
                if abs(rolling_68 - 0.683) < 0.03:
                    consec_recovered += 1
                else:
                    consec_recovered = 0
                # Reset for the next window.
                coverage_counts_68 = np.zeros((n_regions, 2), dtype=int)
                coverage_counts_95 = np.zeros((n_regions, 2), dtype=int)

            cycles_done = cycle + 1

            if cycle % 25 == 0:
                log.info(
                    "[c%4d] H_T=%.2f kappa=%.1f oracle=%d ctx=%d flag=%s "
                    "act=%-15s S=%.2f p=%.3f proj=%.2f rmse=%.3f",
                    cycle, H_T_cur, kappa_cur, oracle_calls, len(M_ctx),
                    flag_str, action, S_t, float(pvals.min()), proj_ratio,
                    rmse_band)

            if (oracle_calls >= ORACLE_BUDGET
                    and consec_recovered >= 1):
                log.info("[c%d] budget reached and recovered", cycle)
                break
            if consec_recovered >= 2 and oracle_calls > 50:
                log.info("[c%d] coverage recovered for 2 windows", cycle)
                break

    wall = time.time() - t0
    log.info("Loop done: %d cycles in %.1fs, oracle=%d, ctx=%d",
             cycles_done, wall, oracle_calls, len(M_ctx))

    final_mu_band = model.predict_np(M_ctx, Y_ctx, M_target)
    final_lev_band = model.leverage(M_ctx, M_target)
    return dict(
        traj=traj, M_ctx=M_ctx, Y_ctx=Y_ctx,
        M_cal=M_cal, Y_cal=Y_cal,
        target_c=target_c, M_target=M_target,
        Y_target_truth=Y_target_truth,
        final_mu_band=final_mu_band, final_lev_band=final_lev_band,
        cycles_done=cycles_done, oracle_calls=oracle_calls,
        wall_seconds=wall, rmse_band_trace=rmse_band_trace,
    )


def main():
    rng = np.random.default_rng(SEED)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    t_total = time.time()

    model, pretrain_wall, pretrain_losses = pretrain_intention(oracle, rng)

    # BEFORE snapshot: FM's prediction with only the thin seed context.
    rng_b = np.random.default_rng(SEED + 1)
    target_c = np.zeros(N_WC)
    target_c[WITHHOLD_DIM] = 0.8
    M_band_eval = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 100)
    Y_band_truth = mu_at(oracle, target_c, M_band_eval)
    M_seed_b = rng_b.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
    Y_seed_b = mu_at(oracle, target_c, M_seed_b)
    mu_before = model.predict_np(M_seed_b, Y_seed_b, M_band_eval)

    out = run_loop(oracle, model, rng)

    mu_after = model.predict_np(out["M_ctx"], out["Y_ctx"], M_band_eval)

    # Persist everything.
    np.savez(
        OUT / "trajectory.npz",
        **{k: np.array(v) for k, v in out["traj"].items()},
        rmse_band_trace=np.array(out["rmse_band_trace"]),
        M_ctx_final=out["M_ctx"], Y_ctx_final=out["Y_ctx"],
        M_band_eval=M_band_eval, Y_band_truth=Y_band_truth,
        mu_before=mu_before, mu_after=mu_after,
        M_target=out["M_target"], Y_target_truth=out["Y_target_truth"],
        final_mu_band=out["final_mu_band"],
        final_lev_band=out["final_lev_band"],
        target_c=target_c,
        pretrain_losses=pretrain_losses,
    )

    torch.save(model.state_dict(), OUT / "intention_fm.pt")

    # Summary.
    final_cov_68 = float(np.nanmean(out["traj"]["cov_68_by_region"][-1])
                         if len(out["traj"]["cov_68_by_region"]) > 0
                         else float("nan"))
    final_kappa = out["traj"]["kappa"][-1]
    final_H_T = out["traj"]["H_T"][-1]
    summary = dict(
        cycles_done=out["cycles_done"],
        oracle_calls=out["oracle_calls"],
        loop_wall_seconds=out["wall_seconds"],
        pretrain_wall_seconds=pretrain_wall,
        total_wall_seconds=time.time() - t_total,
        final_cov_68_avg_over_regions=final_cov_68,
        final_kappa=final_kappa,
        final_H_T=final_H_T,
        final_context_size=int(out["traj"]["context_size"][-1]),
        before_band_rmse=float(np.sqrt(
            np.mean((mu_before - Y_band_truth) ** 2))),
        after_band_rmse=float(np.sqrt(
            np.mean((mu_after - Y_band_truth) ** 2))),
        n_drift_events=int(sum(
            1 for a in out["traj"]["action"]
            if a in ("local_retrain", "global_retrain", "recal",
                     "recal_then_check"))),
        n_local_retrain=int(sum(1 for a in out["traj"]["action"]
                                if a == "local_retrain")),
        n_recal=int(sum(1 for a in out["traj"]["action"] if a == "recal")),
        engineered_band=list(WITHHOLD_BAND),
        engineered_dim_name="c_lq^(3)",
        engineered_target_c=target_c.tolist(),
        d_psi=16, pretrain_steps=PRETRAIN_STEPS,
        oracle_budget=ORACLE_BUDGET, K_epig=K_EPIG,
        cusum_h=CUSUM_H, bh_alpha=BH_ALPHA,
        kappa_threshold=KAPPA_THRESHOLD,
        proj_ratio_threshold=PROJ_RATIO_THRESHOLD,
    )
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=== Run complete ===")
    log.info(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
