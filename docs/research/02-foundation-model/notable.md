# Foundation Model — empirical investigations (agent b)

Five numbered investigations, each with a one-sentence hypothesis and a one-sentence expected outcome.

1. **Stage 0 → stage 1 transfer benefits the held-out operator** —
   *Hypothesis*: pre-training on stage-0 oracle data (4 operators) gives the encoder enough inductive bias that, after stage 1 (8 operators) of continued pre-training, per-operator probe `R²` for a held-out operator is higher than the same operator's `R²` for an encoder pre-trained from scratch on stage-1 data only with the same wall-clock compute.
   *Expected outcome*: continued pre-training wins by at least 0.1 in `R²`, demonstrating that the embedding manifold refines productively across stages and does not require restart on each fidelity upgrade.

2. **Linear probe beats MLP probe on operators inside the pre-training `|c|` range; loses on extrapolation** —
   *Hypothesis*: inside `|c| ∈ [0, 0.7]`, a linear probe on `z` is within `0.05` `R²` of an MLP probe (representation is approximately linear in `c` near zero); outside `|c| ∈ (0.7, 1.0]` the MLP probe wins by at least `0.1` (the manifold curves).
   *Expected outcome*: the gap pattern confirms the embedding is locally linear in `c` and globally curved, which is the right qualitative behaviour for a representation of the SMEFT cross-section manifold.

3. **InfoNCE with `c`-pairing recovers a sharper representation than VICReg with the same pairing** —
   *Hypothesis*: at matched compute, an InfoNCE-trained encoder gives per-operator linear-probe `R²` at least 0.15 higher than a VICReg-trained encoder on held-out `c`. InfoNCE's hard-negatives provide a stronger separation signal in low-dimensional SMEFT space.
   *Expected outcome*: InfoNCE wins, and the margin gives an empirical signature for which contrastive variant the brief's "the representation is the model" tagline reads best on.

4. **Event-set encoder (A) beats histogram encoder (B) at matched parameter count** —
   *Hypothesis*: a DeepSets event-set encoder with parameter count `P` matches or exceeds the histogram-AE encoder (also `P` parameters) on held-out probe `R²` by at least `0.1`, because the histogram's `B = 50` bins discard event-level kinematic correlations relevant at sub-bin resolution.
   *Expected outcome*: A wins, validating the architectural recommendation in `summary.md` § 4. If B wins or ties, that's a finding too — it means binned distributions retain enough information for the current oracle fidelity, and B becomes the cheap production choice.

5. **EPIG over `phi(z)` outperforms uniform-random acquisition for filling the probe's `c`-space coverage** —
   *Hypothesis*: starting from a small seed set, EPIG-driven oracle calls (using `phi(z)` as the feature map for the existing closed-form EPIG in `acquisition.py`) reach a fixed per-operator probe `R²` target with at least 30 % fewer oracle calls than uniform-random acquisition.
   *Expected outcome*: EPIG wins by this margin or better, demonstrating that the existing closed-form acquisition machinery — once liberated from the forbidden `phi_joint(c, m)` feature map — is still the right active-learning engine, and that the FM redesign does not throw away the most useful piece of the old design.
