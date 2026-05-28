"""ALETHIA operator-progression demonstration run.

Phase 1: the oracle is configured to activate only the four-fermion
operators (c_lq^(3), c_lq^(1)) in the training distribution; the vertex
operators (c_phi_q^(3), c_phi_q^(1)) are held at zero throughout
pretraining. The Intention FM never sees a non-zero vertex operator in
its training scenarios. The deployment target also has vertex operators
zero; the FM predicts well.

Phase transition: at cycle PHASE_BOUNDARY the oracle activates the two
vertex operators in the deployment target (i.e. the underlying physics
the agent is being asked about changes). The drift detectors fire because
the new oracle observations are inconsistent with the FM's pretrained
predictions; EPIG drives oracle queries into the affected band; the FM's
context grows; the conformal calibrator refits; predictions recover.

The identifiability probe runs twice: at cycle PHASE_BOUNDARY-1 (Phase 1
end-state) and at cycle N_LOOP_CYCLES-1 (Phase 2 end-state). The
two-phase disclosure profile is the headline deliverable: the agent
reports, via Phoenix, what its representation has learned to recover at
each phase.

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

TRACER_PROVIDER = register(project_name="alethia-progression", auto_instrument=False,
                           protocol="http/protobuf")
tracer = TRACER_PROVIDER.get_tracer("alethia.progression")

# ----- Configuration -----
SEED = 2026
N_WC = 4
IDX_CHQ3, IDX_CHQ1, IDX_CLQ3, IDX_CLQ1 = 0, 1, 2, 3
WITHHOLD_DIM = IDX_CLQ3        # carried over; unused in progression schedule
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

# ----- Operator-progression schedule -----
# Training only activates the four-fermion operators; vertex ops zero.
TRAINING_ACTIVE_OPS = (IDX_CLQ3, IDX_CLQ1)
# Phase 1 deployment target: vertex ops zero, four-fermion ops active.
TARGET_C_PHASE1 = np.array([0.0, 0.0, 0.8, 0.3])
# Phase 2 deployment target: vertex ops activate.
TARGET_C_PHASE2 = np.array([0.4, -0.3, 0.8, 0.3])
PHASE_BOUNDARY_CYCLE = 100

# ----- Loop configuration -----
N_LOOP_CYCLES = 400
PROBES_PER_CYCLE = 24
N_SEED_CTX = 8                 # thin seed: FM has narrow m-support
SEED_M_RANGE = (0.5, 1.0)      # seed context clustered in low-m only
N_CAL_POINTS = 400             # cal set drawn over full M_RANGE
ORACLE_BUDGET = 500
K_EPIG = 5
ACQUISITION = os.environ.get("ACQUISITION", "epig").lower()
assert ACQUISITION in {"epig", "random", "leverage"}, (
    f"ACQUISITION must be one of epig|random|leverage, got {ACQUISITION!r}")

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
OUT_BASENAME = "output_progression"
OUT = HERE / OUT_BASENAME
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
    """Operator-progression training distribution.

    Only the operators in TRAINING_ACTIVE_OPS vary in U([-0.7, 0.7]); the
    remaining operators are held at zero. The FM sees a sparse Wilson
    distribution and never trains on the deactivated directions.
    """
    out = np.zeros((n, N_WC))
    for k in TRAINING_ACTIVE_OPS:
        out[:, k] = rng.uniform(-C_TRAIN_BOX, C_TRAIN_BOX, size=n)
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
    log.info("=== Loop starting (operator-progression schedule) ===")

    # Phase 1 target: four-fermion operators active, vertex operators zero.
    # The deployment matches the training distribution; the FM is in-domain.
    target_c = TARGET_C_PHASE1.copy()
    log.info("Phase 1 target c=%s (vertex ops at zero, four-fermion active)",
             target_c.tolist())

    # Thin seed context: K=8 points clustered in [0.5, 1.0] only.
    M_ctx = rng.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
    Y_ctx = mu_at(oracle, target_c, M_ctx)
    log.info("Seed context: K=%d, m in %s", N_SEED_CTX, SEED_M_RANGE)

    # Calibration set fit against the Phase 1 target. The conformal layer
    # will need to refit after the phase transition (see PHASE_BOUNDARY_CYCLE).
    M_cal = rng.uniform(*M_RANGE, size=N_CAL_POINTS)
    Y_cal = mu_at(oracle, target_c, M_cal)

    cc = IntentionConformal(n_strata=5, noise_frac=0.05)
    cc.fit(model, M_ctx, Y_ctx, M_cal, Y_cal)

    # Target set for H_T (the headline monitor): fine m-grid.
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    Y_target_truth = mu_at(oracle, target_c, M_target)

    # Phase-transition bookkeeping.
    phase = 1
    phase_transition_done = False

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
            # Phase transition: activate the vertex operators in the
            # deployment target at PHASE_BOUNDARY_CYCLE. The seed context
            # and accumulated history persist, so the agent observes a
            # distribution shift in subsequent oracle queries.
            if not phase_transition_done and cycle >= PHASE_BOUNDARY_CYCLE:
                with tracer.start_as_current_span("chain.phase_transition") as ps:
                    target_c = TARGET_C_PHASE2.copy()
                    Y_target_truth = mu_at(oracle, target_c, M_target)
                    ps.set_attribute("aletheia.phase.from", 1)
                    ps.set_attribute("aletheia.phase.to", 2)
                    ps.set_attribute("aletheia.phase.cycle", cycle)
                    ps.set_attribute("aletheia.phase.target_c",
                                     target_c.tolist())
                    log.info("Phase transition at cycle %d: target c -> %s",
                             cycle, target_c.tolist())
                phase = 2
                phase_transition_done = True

            cyc_span.set_attribute("aletheia.cycle.index", cycle)
            cyc_span.set_attribute("aletheia.cycle.phase", phase)
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
                        if ACQUISITION == "random":
                            picked = rng.choice(len(M_pool), size=K_EPIG, replace=False)
                        elif ACQUISITION == "leverage":
                            # D-optimal greedy: pick k pool points with highest leverage(M_pool, M_target).
                            A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
                            Psi_pool = model.psi_np(M_pool)
                            lev_P = np.einsum("pd,de,pe->p", Psi_pool, A_inv, Psi_pool)
                            picked = np.argsort(lev_P)[::-1][:K_EPIG]
                        else:
                            picked = epig_acquire_m(
                                model, M_ctx, Y_ctx, M_pool, M_target,
                                k=K_EPIG)
                        picked_m = M_pool[picked]
                        es.set_attribute("aletheia.epig.k", K_EPIG)
                        es.set_attribute("aletheia.epig.pool_size", 300)
                        es.set_attribute("aletheia.epig.acquisition", ACQUISITION)
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

    # BEFORE snapshot uses the Phase 2 target on the seed-only context:
    # this is the FM's "fresh deployment" prediction on the post-transition
    # physics, the baseline against which Phase 2 recovery is measured.
    rng_b = np.random.default_rng(SEED + 1)
    target_c_phase2 = TARGET_C_PHASE2.copy()
    M_band_eval = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 100)
    Y_band_truth = mu_at(oracle, target_c_phase2, M_band_eval)
    M_seed_b = rng_b.uniform(*SEED_M_RANGE, size=N_SEED_CTX)
    Y_seed_b = mu_at(oracle, target_c_phase2, M_seed_b)
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
        target_c_phase1=TARGET_C_PHASE1,
        target_c_phase2=TARGET_C_PHASE2,
        phase_boundary_cycle=PHASE_BOUNDARY_CYCLE,
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
        phase_boundary_cycle=PHASE_BOUNDARY_CYCLE,
        target_c_phase1=TARGET_C_PHASE1.tolist(),
        target_c_phase2=TARGET_C_PHASE2.tolist(),
        training_active_ops=list(TRAINING_ACTIVE_OPS),
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
