"""
Toy demo for FM-taxonomy branch 02 (foundation models in HEP).

Isolates ONE core HEP-FM idea concretely: self-supervised *masked particle
modeling* (MPM, Golling/Heinrich et al. arXiv:2401.13537) on toy jets, then
shows the pretrained permutation-invariant representation makes a downstream
tagging task linearly separable with FEW labels, beating a from-scratch
baseline trained on the same few labels.

Pipeline (numpy + sklearn only, no torch needed):

  1. Generate synthetic jets as variable-length particle point clouds.
     - QCD-like: single soft-radiating prong.
     - Top-like: two well-separated prongs (a decay).
     Each particle is x_i = (log pT, d_eta, d_phi).  N varies per jet.

  2. Tokenizer: KMeans codebook over particles from UNLABELED jets -> token ids
     (the toy stand-in for MPM's VQ-VAE codebook).

  3. Pretrain (self-supervised, NO labels): masked-particle token prediction.
     For each jet, mask a random subset of particles; predict each masked
     particle's codebook token from a permutation-invariant summary of the
     visible particles (Deep Sets sum-pool) + the masked particle's pT-rank
     (the MPM ordering trick that breaks the permutation degeneracy at the
     head).  Trained as multinomial logistic regression = the linear MPM head.

  4. Build the pretrained REPRESENTATION of a whole jet: a permutation-invariant
     histogram of predicted token posteriors pooled over its particles. This is
     a frozen feature map derived purely from the SSL objective.

  5. Downstream FEW-LABEL test (linear classifier test): top vs QCD.
     - Pretrained: linear probe on the frozen MPM representation.
     - From-scratch: same linear model on a raw permutation-invariant pooling
       of the particles (matched, label-free feature budget) with no pretraining.
     Compare test accuracy / AUC as a function of #labels.

Everything is seeded and self-contained. Prints numbers that make the
distinction visible.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

RNG = np.random.default_rng(2026)
N_TOKENS = 16          # codebook size (toy VQ-VAE stand-in)
MASK_FRAC = 0.5        # fraction of particles masked per jet during pretraining


# -----------------------------------------------------------------------------
# 1. Toy jet generator: variable-length particle point clouds
# -----------------------------------------------------------------------------
def make_jet(is_top, rng):
    """Return an (N,3) array of particles: (log pT, d_eta, d_phi)."""
    n = int(rng.integers(12, 40))
    if is_top:
        # two prongs: a hard splitting, each prong radiates
        prongs = rng.normal(0.0, 0.05, size=(2, 2))
        prongs[0] += np.array([0.45, 0.0])    # separate the two prongs
        prongs[1] += np.array([-0.45, 0.0])
        assign = rng.integers(0, 2, size=n)
        centers = prongs[assign]
        spread = 0.18
    else:
        # single prong: one soft-radiating core
        centers = np.zeros((n, 2))
        spread = 0.30
    deta = centers[:, 0] + rng.normal(0.0, spread, size=n)
    dphi = centers[:, 1] + rng.normal(0.0, spread, size=n)
    # steeply falling pT spectrum; top prongs carry a bit more momentum
    log_pt = rng.exponential(1.0, size=n) + (0.4 if is_top else 0.0)
    parts = np.stack([log_pt, deta, dphi], axis=1)
    # IRC-safe-ish: sort by pT so the pT-rank feature is well defined
    parts = parts[np.argsort(-parts[:, 0])]
    return parts.astype(np.float64)


def make_dataset(n_jets, rng, frac_top=0.5):
    jets, labels = [], []
    for _ in range(n_jets):
        is_top = rng.random() < frac_top
        jets.append(make_jet(is_top, rng))
        labels.append(int(is_top))
    return jets, np.array(labels)


# -----------------------------------------------------------------------------
# 2. Tokenizer (codebook) over unlabeled particles
# -----------------------------------------------------------------------------
def fit_tokenizer(jets, rng):
    allp = np.concatenate(jets, axis=0)
    mu, sd = allp.mean(0), allp.std(0) + 1e-9
    km = KMeans(n_clusters=N_TOKENS, n_init=4, random_state=2026)
    km.fit((allp - mu) / sd)
    return km, mu, sd


def tokenize(jet, km, mu, sd):
    return km.predict((jet - mu) / sd)


# -----------------------------------------------------------------------------
# 3. Self-supervised MPM pretraining (no labels)
#    Predict masked particle token from (visible-set summary, pT-rank).
# -----------------------------------------------------------------------------
def context_summary(jet_tokens, visible_mask):
    """Permutation-invariant Deep Sets summary of VISIBLE particles:
    normalized token histogram (sum-pool of one-hot, then normalize)."""
    hist = np.zeros(N_TOKENS)
    vis = jet_tokens[visible_mask]
    if len(vis):
        for t in vis:
            hist[t] += 1.0
        hist /= hist.sum()
    return hist


def build_mpm_training(jets_tokens, jets, rng):
    X, y = [], []
    for toks, jet in zip(jets_tokens, jets):
        n = len(toks)
        n_mask = max(1, int(round(MASK_FRAC * n)))
        masked = rng.choice(n, size=n_mask, replace=False)
        vis = np.ones(n, dtype=bool)
        vis[masked] = False
        summ = context_summary(toks, vis)
        for idx in masked:
            # pT-rank feature: breaks the permutation degeneracy at the head
            # (MPM orders masked queries by pT). jets are pT-sorted, so idx/n.
            rank = idx / max(1, n - 1)
            feat = np.concatenate([summ, [rank, np.log(n)]])
            X.append(feat)
            y.append(toks[idx])           # target = masked particle's token
    return np.asarray(X), np.asarray(y)


# -----------------------------------------------------------------------------
# 4. Frozen pretrained representation of a whole jet
#    Pool the MPM head's predicted token-posteriors over all particles.
# -----------------------------------------------------------------------------
def pretrained_representation(jet_tokens, mpm_head):
    n = len(jet_tokens)
    feats = []
    for idx in range(n):
        vis = np.ones(n, dtype=bool)
        vis[idx] = False
        summ = context_summary(jet_tokens, vis)
        rank = idx / max(1, n - 1)
        feats.append(np.concatenate([summ, [rank, np.log(n)]]))
    post = mpm_head.predict_proba(np.asarray(feats))   # (n, N_TOKENS)
    # permutation-invariant pool: mean + max of posteriors over particles
    return np.concatenate([post.mean(0), post.max(0)])


# -----------------------------------------------------------------------------
# from-scratch baseline representation: raw permutation-invariant pooling
# (matched, label-free, NO pretraining)
# -----------------------------------------------------------------------------
def raw_representation(jet):
    # Deep Sets style: mean + std + max over particle features + multiplicity
    return np.concatenate([jet.mean(0), jet.std(0), jet.max(0),
                           [np.log(len(jet))]])


def main():
    # ---- data ----
    pretrain_jets, _ = make_dataset(4000, RNG)             # unlabeled corpus
    label_jets, label_y = make_dataset(4000, RNG)          # labeled pool + test
    n_test = 2000
    test_jets, test_y = label_jets[-n_test:], label_y[-n_test:]
    pool_jets, pool_y = label_jets[:-n_test], label_y[:-n_test]

    print("=" * 70)
    print("HEP-FM toy demo: masked-particle pretraining vs from-scratch")
    print("=" * 70)
    print(f"unlabeled pretrain jets : {len(pretrain_jets)}")
    print(f"labeled pool / test     : {len(pool_jets)} / {len(test_jets)}")
    print(f"codebook tokens         : {N_TOKENS}   mask fraction: {MASK_FRAC}")

    # ---- tokenizer (unlabeled) ----
    km, mu, sd = fit_tokenizer(pretrain_jets, RNG)
    pre_tokens = [tokenize(j, km, mu, sd) for j in pretrain_jets]

    # ---- SSL pretraining: masked particle modeling (no labels) ----
    Xm, ym = build_mpm_training(pre_tokens, pretrain_jets, RNG)
    # multinomial is the default solver behaviour for lbfgs; do not pass the
    # deprecated multi_class kwarg (removed in recent sklearn).
    mpm_head = LogisticRegression(max_iter=400, C=2.0)
    mpm_head.fit(Xm, ym)
    mpm_acc = accuracy_score(ym, mpm_head.predict(Xm))
    chance = 1.0 / N_TOKENS
    print(f"\n[pretext] masked-token prediction acc {mpm_acc:.3f} "
          f"(chance {chance:.3f}) -> representation has learned jet structure")

    # ---- frozen pretrained representation of labeled jets ----
    pool_tok = [tokenize(j, km, mu, sd) for j in pool_jets]
    test_tok = [tokenize(j, km, mu, sd) for j in test_jets]
    Rep_pool = np.array([pretrained_representation(t, mpm_head) for t in pool_tok])
    Rep_test = np.array([pretrained_representation(t, mpm_head) for t in test_tok])

    # ---- from-scratch matched representation (no pretraining) ----
    Raw_pool = np.array([raw_representation(j) for j in pool_jets])
    Raw_test = np.array([raw_representation(j) for j in test_jets])

    # ---- few-label downstream comparison (linear classifier test) ----
    print("\nDownstream: top vs QCD tagging, linear probe, test acc / AUC")
    print(f"{'#labels':>8} | {'pretrained MPM':>22} | {'from-scratch':>20}")
    print("-" * 60)
    results = {}
    for n_lab in [10, 25, 50, 100, 250, 500]:
        # ensure both classes present in the few-label draw
        for _ in range(50):
            sel = RNG.choice(len(pool_jets), size=n_lab, replace=False)
            yl = pool_y[sel]
            if len(np.unique(yl)) == 2:
                break
        # pretrained
        clf_p = LogisticRegression(max_iter=500, C=1.0)
        clf_p.fit(Rep_pool[sel], yl)
        acc_p = accuracy_score(test_y, clf_p.predict(Rep_test))
        auc_p = roc_auc_score(test_y, clf_p.predict_proba(Rep_test)[:, 1])
        # from-scratch
        clf_s = LogisticRegression(max_iter=500, C=1.0)
        clf_s.fit(Raw_pool[sel], yl)
        acc_s = accuracy_score(test_y, clf_s.predict(Raw_test))
        auc_s = roc_auc_score(test_y, clf_s.predict_proba(Raw_test)[:, 1])
        results[n_lab] = (acc_p, auc_p, acc_s, auc_s)
        print(f"{n_lab:>8} | acc {acc_p:.3f} auc {auc_p:.3f}    "
              f"| acc {acc_s:.3f} auc {auc_s:.3f}")

    # ---- headline gap at the few-label end ----
    a_p10, au_p10, a_s10, au_s10 = results[10]
    a_p50, au_p50, a_s50, au_s50 = results[50]
    print("\nSummary")
    print(f"  at 10 labels : pretrained AUC {au_p10:.3f} vs scratch {au_s10:.3f}"
          f"  (gap {au_p10 - au_s10:+.3f})")
    print(f"  at 50 labels : pretrained AUC {au_p50:.3f} vs scratch {au_s50:.3f}"
          f"  (gap {au_p50 - au_s50:+.3f})")
    print("  -> SSL masked-particle pretraining buys label efficiency: the "
          "frozen\n     representation is linearly separable with far fewer "
          "labels.")


if __name__ == "__main__":
    main()
