# Results — EFT/SMEFT quadratic-oracle survey

`oracle_survey.py` is self-contained: numpy required; scikit-learn (GP + MLP)
and matplotlib optional and guarded (rows print "skipped" without sklearn). It
is seeded (`SEED = 2026`) and writes `oracle_survey.csv` (+ optional
`sample_efficiency.png`).

Run:

```
python3 oracle_survey.py
# or, in the ALETHIA env:
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
uv run python experiments/fm-taxonomy/phase2/03-eft-oracle/oracle_survey.py
```

## Run output

Executed in the main session (numpy 2.4.4 + scikit-learn 1.8.0; matplotlib
present). 28 CSV rows + sample-efficiency plot written.

```
[1] Learner panel — interpolation vs extrapolation relative RMSE
    (MC budget n_events=1e4 -> ~1% per-point stat error; n_train=64; hull |c|<=1)

  --- n_coef = 1 (basis N = 3) ---     interp_relRMSE  extrap_relRMSE
     morphing(exact)                       3.802e-17       9.647e-17   cond(M)=3.23
     poly_deg2(correct)                     1.945e-03       4.523e-03
     poly_deg1(under)                       2.917e-02       1.404e-01
     poly_deg3(over)                        2.136e-03       2.213e-02
     gaussian_process                       2.353e-03       3.027e-02
     mlp_64x64                              3.159e-03       7.994e-02

  --- n_coef = 2 (basis N = 6) ---
     morphing(exact)                        6.860e-17       1.264e-16   cond(M)=8.07
     poly_deg2(correct)                     3.384e-03       1.294e-02
     poly_deg1(under)                       4.370e-02       1.706e-01
     poly_deg3(over)                        4.998e-03       5.919e-02
     gaussian_process                       4.573e-03       4.683e-02
     mlp_64x64                              7.777e-03       1.092e-01

  --- n_coef = 3 (basis N = 10) ---
     morphing(exact)                        1.217e-16       2.211e-16   cond(M)=14.11
     poly_deg2(correct)                     4.931e-03       1.074e-02
     poly_deg1(under)                       5.089e-02       2.224e-01
     poly_deg3(over)                        8.288e-03       5.403e-02
     gaussian_process                       6.046e-03       3.075e-02
     mlp_64x64                              6.435e-03       1.114e-01

[3] Degenerate / flat-Fisher direction (n=2), working point (0.6,0.6)
     degen   F_cond(c=0)   F_cond_wp  lift_min_eig   poly2_flat    gp_flat
      0.00     1.513e+02   1.692e+02     1.635e+00    9.321e-04  1.745e-03
      0.70     1.969e+03   2.157e+03     1.707e+00    2.662e-03  1.920e-03
      0.95     7.524e+04   8.191e+04     1.732e+00    5.981e-03  7.692e-03
```

Verified blocks:
- `[0]` minimal basis `n=1→3, n=2→6, n=3→10` = `C(n+2,2)`, as predicted.
- `[1]` morphing is exact to machine precision (~1e-16) in **both** columns at
  every n; correct-degree poly sits at the ~1% MC noise floor; underfit deg-1 is
  biased (up to 22% off-support at n=3); GP/MLP track the noise floor *inside*
  the hull but degrade 3–11% *outside* it — the learned-manifold off-support
  failure.
- `[2]` morphing is exact at the minimal `N` noiseless points; GP/MLP error falls
  with `n_train` but trails the structure-aware learners and never closes the
  extrapolation gap.
- `[3]` Fisher conditioning explodes (1.5e2 → 7.5e4) as the degenerate direction
  is approached; the working point lifts the small eigenvalue (`lift_min_eig` ≈
  1.6–1.7) — the flat-vertex mechanism in closed form.

## Interpretation

A simple learner can stand in for the MC program exactly when it carries the
right structure (morphing, correct-degree polynomial): minimal data, exact
extrapolation, the Fisher geometry available in closed form. A degree-agnostic
GP/MLP is a local interpolator — acceptable inside the sampled hull, unreliable
off-support, and data-hungry. This is axis III made quantitative on one object:
structured (morphing) vs learned (generic) provenance of the same prediction
manifold, the structured pole of ALETHIA's ManifoldInformer tension.
