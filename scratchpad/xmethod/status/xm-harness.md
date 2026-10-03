# Status: xm-harness  (updated 2026-10-02) -- READY-TO-MERGE
Rulings applied: Q1 (R1 sign correct, R4 / R3-marginal wrong at lam 1.5); (a) R3 redrawable categorical table;
(b) R3 actions independent given Z; (c) R-6 primary = action->KPI, secondary = KPI->KPI; R-10 placebo split.

## HAND-BACK
Deliverables (branch xm/xm-harness, commits beef962, a0960a2, 4b52bc8, + this status): `cdd_oran/xmethod/worlds/
{generate,e4_logged}.py` (`generate_dataset(world, regime, n, seed, lam=1.0)`, `truth_for`, `candidates_for`,
`dataset_hash`; generator "xm-worlds/2"), `cdd_oran/xmethod/score.py` (`score`, `placebo_tau`, `apply_threshold`),
`cdd_oran/xmethod/runner.py` (list / run --part i/P / merge / kaggle / colab; `dummy`; registry takes "module:Class"),
`tests/test_xmethod_harness.py`, `scratchpad/xmethod/{specs/smoke_dummy.json,gen_timing.py}`.
Tests: 76 passed; ruff clean; e2slice + e4 gate tests still pass. Kaggle smoke `xm-harness-smoke1` (dummy, 48 jobs,
2 parts): exit 0, 48/48 merged, 0 errors. Dependencies added: none (`uv sync --group baselines` for kaggle_job.py).

Shapes (actions incl. placebos) | primary cand. (true/null; incl. P_placebo) | secondary | signed | gen n=24000 *
E1 A[n,5] K[n,4] | 20 (4/16) | 16 (2/14) | 6           | 0.25 s, 56-60 MB
E2 A[n,9] K[n,6] | 54 (16/38) | 36 (0/36) | 1 (P6->K5 -) | 0.43 s, 59-67 MB
E3 A[n,5] K[n,5] | 25 (4/21) | 25 (3/22) | 7           | 0.26 s, 56-61 MB
E4 R1/R2 A[n,2] K[n,1] | 2 (1/1) | 1 (0/1) | 1 (P0->K0 -) | 1.0 s, 54-56 MB
E4 R3/R4 A[n,3] | same + diagnostic 1 (P_placebo_conf->K0, null) | R3 4.5-4.8 s, 324 MB; R4 2.7 s, 54 MB
E5 A[n,5] K[n,4] | 20 (6/14) | 16 (1/15) | 3           | 1.35 s, 56-61 MB
* Kaggle CPU, 1 thread, fresh process per cell, 2 reps (job xm-harness-gentime1); per-job peak RSS. Laptop
  timings were contaminated by other workers' load (2-7x noise) and are not reported.

Checks: truth == SCM by intervention (source replaced -> target moves iff true edge; signed edges move only in their
sign); roller == e2slice.generate_rows byte-for-byte; E4 R1 slope in (-1.3,-0.7) at every lam and lam-inert; E4 R4 and
R3-marginal slope > +.5 at lam 1.5; R3 OLS adjusted for Z = -1 exactly; R3 action == rounded frozen R4 action; R3
tables sum to 1, > 0, match realised; R2 column == fixed + random, 20 constant-setpoint blocks, lag-1 autocorr > .5,
dither i.i.d.; placebos leave all other arrays byte-identical; P_placebo |corr| < 4.5/sqrt(n) with every column (R1;
R2 dither; R3/R4 incl. Z and P_placebo_conf), same marginal as P0 (R3 table = Z-average of the policy);
P_placebo_conf corr with K > .1 (lam .5, 1.5); E5 noise sd = kappa*sigma.
Gate visits (10 DEV seeds, n 4000): E2 rows K5<-5: R1 4.5%, R2 8.4% (min 5.5%); E5 in-gate: R1 5.0%, R2 9.4% (min 4.8%).

JUDGEMENT CALLS (for the protocol review)
J1 Rows: one trajectory per (world, seed), env_seed = seed, episode 0, 4 priming advances, row = committed state at t
   -> KPIs at t+1 (e1slice/e2slice alignment). R1 redraws ALL actions every row (e2slice style).
J2 R1 dists: uniform on ID ranges; E4 uniform on the frozen do-grid V (101 points), dist "categorical".
J3 R2 = Stage 0 F-dither primary: delta .10, 20 blocks of n/20 rows, setpoints U(mid +/- .25 range) once per corpus
   (fixed_part, conditioned on), dither = random_part; never clips.
J4 R3: frozen behaviour draw W = .5 + lam Z + eta (same tape slots) clipped and rounded to V; context = Z exactly;
   Design.propensity = [n,101] P(A=V[k]|Z_i), dist "categorical_rows"; realised probs in meta["realised_propensity"].
J5 R4 = frozen E4 mode "obs". E4 R1/R2 ignore lam (lam acts only via the behaviour policy).
J6 (R-10) P_placebo = tuning placebo, own stream, independent of everything: R1/R2 P0's design; R3/R4 i.i.d. from
   P0's marginal (W ~ N(.5, lam^2+.25); R3 rounded to V, Design "iid" categorical with the exact marginal p; R4
   clipped, Design "iid" {"name": "clipped_normal"}). P_placebo_conf (E4 R3/R4, last column) = Z policy with its own
   eta (R3 "logged" table, R4 "none"); null in truth; meta["diagnostic_candidates"], NOT in primary_candidates;
   scored as placebo_conf_declared / placebo_conf_fpr; placebo_tau reads P_placebo only.
J7 E5 = fully composed (subdom .2, chain .2, theta -2.5, mode "do"), process noise off, observation noise
   N(0,(0.3 sigma_k)^2) per time and KPI (E5_NOISE_DETECT model), shared by lag and target.
J8 Signs only where monotone over the ID range (E2: only P6->K5).
J9 Score top level = primary set minus both placebos; precision NaN if R=0, FDP = V/max(R,1); missing = not declared;
   placebo_tau = (k+1)-th largest P_placebo score, declare score > tau. `secondary` and `all` blocks included.
J10 RNG: SeedSequence([7800, seed, world, regime, purpose]) (7800 to be registered by orchestrator); dummy uses 7899;
   Design.redraw_seed_tag left None.

CAVEATS
D1 Byte-identity holds per platform: Linux vs Windows identical for E1, E3, E4 R1/R2/R4; E2, E5, E4 R3 differ at ulp
   level (libm exp in frozen SCMs, erfc in R3). Compare dataset_sha256 within one platform only.
D2 peak_rss_mb is per job on Linux, process-lifetime on Windows. R3 table build (np.vectorize erfc) dominates R3 cost.
D3 LF line endings normalised in 4b52bc8. OPEN QUESTIONS: none. api.py doc idea: "logged" propensity may be [n, n_values].
