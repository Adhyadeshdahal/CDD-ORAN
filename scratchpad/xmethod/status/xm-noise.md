# READY-TO-MERGE -- Status: xm-noise (updated 2026-10-02)
Branch xm/noise from feat/v2 1191678; commit 0914e95 (+ this status). Rulings R-24, R-26 noted (all work local).

## HAND-BACK
Deliverables
- `generate_dataset(world, regime, n, seed, lam=1.0, kappa=None)` (worlds/generate.py): observed KPI series =
  noiseless + eps_t,k ~ N(0, (kappa sigma_k)^2), i.i.d. per env time and KPI, ONE draw shared by X_kpi_lag and Y
  (Y[r] == X_kpi_lag[r+1] kept); actions, designs, context, truth unchanged. RNG SeedSequence([7800, seed, world,
  regime, 5 "obs_noise"]) = E5's existing stream; standard normals independent of kappa (common random numbers).
  kappa=None = KAPPA_DEFAULT (E1-E4 0, E5 .3 = old J7), so every world is byte-identical to before (Q1, approved);
  any explicit kappa (0 incl.) applies to every world, E5 included: one noise layer. meta["obs_noise"] =
  {kappa, sigma} iff kappa > 0. kappa < 0 / NaN / inf -> ValueError. GENERATOR_VERSION unchanged (default output identical).
- NOISE_SIGMA (frozen): E1-E4 = pooled sd (ddof 0) of noiseless R1 Y over DEV seeds 3_000_000-019 at n 24_000
  (480k rows/world; E4 at lam 1.0, which R1 ignores); `scratchpad/xmethod/noise_sigma.py` ->
  `results/noise_sigma.json`. Agreement with the env reference moments (rel. diff): E1 analytic sqrt(1/12),
  sqrt(1.25/12): <= .10%; E2 frozen KPI_MEAN_STD (n 1e6): <= .42%; E3 analytic steady state: <= .17%; E4 analytic
  sqrt(.085 + 2.5^2): .05%. E5 = E5_KPI_MOMENTS (unchanged).
  E1 (.288709, .288894, .322411, .322784); E2 (27.672172, 34.66717, 44.7979, 32.491503, 40.552988, 4.791992);
  E3 (.288581, .288582, .407729, .407559, .288733); E4 (2.515616,). One sigma per world for all regimes.
- Runner: spec key "kappas" REQUIRED (missing / empty / negative / NaN -> ValueError); job keys always carry
  "k<kappa>|"; record job.kappa; configs looked up default -> kappa-free cell key -> cell key with kappa;
  merge cells keyed with kappa. `scripts/xm_classic_integration.py` (xm-classic's driver) got a required
  --kappa (passed to its tuning data and spec): it no longer runs with world defaults. Smoke spec has "kappas": [0.3].

Checks (truth-free)
- Byte-identity: `scratchpad/xmethod/noise_identity.py` vs the generator at 1191678: 32/32 dataset hashes identical
  (all 12 cells, E4 lam 0 / 1.5, n 300 / 1000).
- Realised noise sd / (kappa sigma_k) on R1, 5 DEV seeds x n 4000 (targets and lags): E1 .998-1.001, E2 .994-1.013,
  E3 .995-1.003, E4 1.011, E5 .994-1.002 (identical across kappa: CRN).
- Residual variance share of Y_k under the linear Z_eq fit (1 + all actions + design_covariates), mean of 5 seeds,
  n 4000 (`results/noise_check.json`), per KPI at kappa 0 / .1 / .3 / .5:
  E1 R1 0 / .0096-.0099 / .080-.082 / .195-.200; E1 R2 0 / .032-.040 / .23-.27 / .45-.51
  E3 R1 0 / .0096-.0100 / .080-.083 / .193-.199; E3 R2 0 / .035-.038 / .24-.26 / .46-.49
  E4 R3 (lam 1.5) 0 / .014 / .111 / .258;  E5 K_mid (R1) 0 / .0099 / .082 / .200 (exact fit too when noiseless)
  So the exact fits (R-22) end for every kappa > 0. In R1 the share is kappa^2 / (1 + kappa^2), i.e. R-24's
  u <-> kappa map holds exactly for linear worlds; in R2 it is larger (smaller noiseless spread than R1).
  Nonlinear / latent cells are far from exact already at kappa 0 (E2 R1 .64-.99, E4 R1 .98, E4 R4 .36, E5 K_harm .99).
Tests: `tests/test_xmethod_noise.py` (28: default == world default; explicit kappa uniform incl. E5; only the KPI
series change; realised sd; CRN; one E5 noise layer; exact fits broken, R1 share = kappa^2/(1+kappa^2) +/- 20%;
invalid kappa; runner requires kappas, keys / configs / records / merge). All xmethod tests: 220 passed; ruff clean.
No dependencies added.

## QUESTIONS
(none open; Q1 answered: None = world default, explicit kappa uniform.)
Note: sigma_k is the R1 spread for every regime, so at one kappa the noise share differs by regime (R2 higher).
