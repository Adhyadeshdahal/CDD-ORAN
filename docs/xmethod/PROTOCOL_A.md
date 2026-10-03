# Study A protocol: design-based vs design-blind edge tests on E1-E5 (EVAL pre-registration)

FROZEN: no (DRAFT v4, worker `protocol`, 2026-10-03; branch xm/protocol8, feat/v2 95663ec; rulings R-1..R-51)

Freeze procedure. Before any EVAL unit is generated: (1) every T-item of section 13 is filled by its rule and the
value is written here; (2) the line above becomes "FROZEN: yes (<date>)"; (3) `specs/eval/full.json` gets
`protocol_sha256` = the LF sha256 of this file (the campaign EVAL guard and `eval_analysis.py` check it) and its
`pkgs_lock`; (4) the freeze commit message records the LF sha256 of this file, `scratchpad/xmethod/eval_analysis.py`,
`specs/eval/full.json`, `cdd_oran/xmethod/campaign.py` and `results/fmax_sim/dependence.json`; (5) the orchestrator
registers the EVAL block 3_100_000-3_100_299 in `docs/benchmark/SEED_REGISTRY.json`. After the freeze this file is
never edited: amendments go to `docs/xmethod/PROTOCOL_A_AMENDMENTS.md` + `specs/eval/amendments.json` (section 14).
Sources: CONTRACT.md, PROTOCOL_NOTES.md, PMRT_CORE.md, FIDELITY_CLASSIC / CITESTS, xm/dev-runs, reviews A-C.

## 0. What was seen before the freeze (R-36)

Only DEV seeds 3_000_000-3_000_199 were generated or read; no EVAL seed (3_100_000+). Seen, bearing on the verdicts:
(a) fidelity gates F1-F7 of every adapter; (b) the kappa calibration (R-27); (c) the dev-runs pilot: corr and
granger_native declare the R2 placebo 2-4x nominal, so C1's direction for them was known; (d) the full DEV run (tune
3_000_000-019, measure 3_000_100-159) incl. every arm's DEV validity (T5); (e) three protocol reviews of the draft and
pilot numbers; (f) the CI-test audit's smoke (n 1000, 5 DEV seeds: mscr invalid in R2 in both arms, pdcor eq E2 R2
null .22); (g) F2-GPU and the cmi_knn GPU cost on one DEV dataset (E5 R1, arm eq; Kaggle T4 wall 172 s at n 1000,
~2.4k-6.1k s at n 4000, ~9k s extrapolated at n 8000); (h) the DEV power read (R-42): pmrt_eq recall .28-.46 in E2 /
E5 vs SHAP-DAG .67-.97 (pmrt_eq at raw p <= .05: .38-.54), attributed (not shown) to pmrt_core's linear statistic;
PMRT admits any statistic, its validity coming from the redraw. Choices made after DEV numbers, reviews or the
audit: no Huber clip in PMRT (R-14, E2 R2 DEV truth-null rates); eq arms primary (R-17, before DEV validity);
NOTEARS kept despite F3 19/20; the R2 design (20 setpoint blocks, +/-.10 dither, Stage 0 on E6) supplies PMRT's randomisation, so favours it
by construction; after the reviews (R-29..R-39): conformal tau, three-way validity, simulated F_max (dependence from
DEV seeds 3_000_100-199), C2a / C2b with fallback wording, eq arms in E4 R3, the HAC baseline pcorr_hac (added after
reviewers predicted dependence-robust design-blind tests might hold level; its F4 was seen: over-rejection at n <=
1000 under strong persistence) and its fixed-b variant (set-D choice on synthetic F4 only, T8), eq_min, S_E4 by
simulation (T9), P_placebo_conf never a conditioner (R-37), the draft's "n 500 PARTIAL" clause dropped; after the
audit (R-40): mscr_eq out of C2b (not equal information), pdcor labelled a dependence test and pdcor_native out of D;
after the GPU cost (R-41): cmi_knn on the torch backend in every arm with a 2 h GPU wall budget; after the DEV power
read (R-42): a nonlinear PMRT statistic picked by a selection rule fixed in R-42 before any candidate was built
(T10; DEV seeds 3_000_100-159 and synthetic data only, never 3_000_160-189, reserved for the pmrt-diag check),
pmrt_nl_eq the primary PMRT arm and the linear pmrt_eq secondary, PMRT power in R4 reported as not applicable
(never recall 0), and the like-for-like table V0 (every p arm at raw p <= .05 and with the conformal tau, next to
the score-only arms) promoted to a headline; after the CI-test audit and the DEV cost estimate (R-47..R-50): the
CI DEV plan C (R-47: full DEV grid for pcorr and rcot2, mscr at n <= 1000 only, cost probes only for pdcor and
cmi_knn; the full grid was ~5900 core-h + 372-1482 GPU-h), pdcor dropped (R-48) and cmi_knn dropped (R-49 revised,
cost; section 4), and the authors' conference method CDL added as a
score-only arm (R-50). Estimand check: PASS, 18 cells.

## 1. Question and claim

Question. On one-step action -> KPI data logged under a known randomisation design, which edge tests hold their
nominal false-positive level, and how much power do the valid ones have?
Claim under test (CONTRACT section 8, R-36 wording): "tests that ignore the randomisation design are invalid on the
setpoint + dither data of the R2 design tested (C1); using the design restores validity (C2a: the design-based
test; C2b: design-covariate adjustment of model-based tests); PMRT does so by construction, its validity does not
depend on the outcome model, it admits any statistic, and it stays valid under logged confounded policies (C3)".
"RIC-style" means only this R2 design. The outcome-model and any-statistic properties are argued from the
construction (PMRT_CORE.md), not tested; fallback wordings in section 10; power descriptive, no superiority claim.

## 2. Worlds and regimes (controlled diagnostic environments)

Each world isolates one mechanism; none models a deployed RAN. SCMs read-only (`cdd_oran/envs/v2/e*.py`), generator
`cdd_oran/xmethod/worlds/generate.py` (`xm-worlds/3`, RNG streams keyed by (seed, world, regime, purpose), not by n:
smaller n are prefixes of the same draws). Candidates true / null (P_placebo -> KPI among the nulls): E1 4/16, E2
16/38, E3 4/21, E4 1/1, E5 6/14; E4 has no real-action null (read-outs: P_placebo, and P_placebo_conf in R3 / R4).

| world | mechanism isolated | regimes |
|---|---|---|
| E1 | clean identifiable linear graph (K2 = P2 + .5 K0, K3 = P3 + .5 K1) | R1, R2 |
| E2 | gated edge, active only in a rare regime | R1, R2 |
| E3 | delayed harm: effects through lagged KPI cascades (depth 3) | R1, R2 |
| E4 | latent confounding; associational sign wrong at large lambda; lambda in {0, .5, 1, 1.5} | R1-R4 |
| E5 | E2-E4 mechanisms composed (subdom .2, chain .2, theta -2.5) | R1, R2 |

| regime | design (`api.Design.kind`) | role |
|---|---|---|
| R1 | `iid`: every action redrawn i.i.d. per row | reference: valid tests hold level |
| R2 | `dither`: 20 setpoints per corpus (+/-25 % envelope), held n/20 rows, + i.i.d. U(+/-.10 range) dither | C1 / C2 test bed |
| R3 | `logged`: E4 behaviour policy on an OBSERVED confounder, per-row probabilities logged (R-7) | C3 |
| R4 | `none`: E4 latent confounder, nothing logged | negative control |

E4 lambda: R1 / R2 at 1 only (inert, R-1); R3 / R4 at all four. Placebos (R-10): `P_placebo` (real actions' design in
R1 / R2, marginal design in R3 / R4) is in the primary BY family and the only tuning column; `P_placebo_conf`
(E4 R3 / R4, confounded policy) is outside every BY family, never tuned on, never a conditioner (R-37).

## 3. Observation noise (R-24, R-27)

y_k <- y_k + N(0, (kappa sigma_k)^2) in every world; PRIMARY kappa .25 (kappa_E6, CI [.229, .337], an upper bound on
measurement noise); sweep {.125, .5} in R2 at n 1000; kappa 0 is a DEV sanity check only.

## 4. Methods

| adapter (version) | family | authors' null / native rule | arms | fidelity (F1-F7) |
|---|---|---|---|---|
| pmrt_core (`pmrt-core-v1`: linear statistic; the T10 statistic's version string) | proposed, design-based CRT | redraw of the known random part, B <= 9999 Besag-Clifford, no clip; any statistic (R-42) | nl_eq (primary: statistic by T10), eq (linear, secondary), r3 | engine bit-exact vs E6 pmrt `plain`; F4 null .043-.058 (linear); nonlinear: F4 under T10 |
| mscr (`xm-mscr-v2-bc`) | CI test; eq arm: single-conditioner max statistic, cannot condition on the joint design set (R-40) | stratified random-partition bank, B 9999 BC | eq (not in C2b), eq_min, native | F3 exact on stored E2 |
| pcorr (`xm-pcorr-v1`) | CI test | partial-correlation t-test, df n-2-\|Z\| | eq, eq_min, native | F2 / F4 PASS |
| pcorr_hac (`PcorrHac` 1.0, R-32) | design-blind, serial-dependence robust | OLS t, Newey-West HAC SE, Andrews AR(1) bandwidth, time order, t(n-k) | native, eq_min | F2 == statsmodels; F4 iid .049-.054, AR(1) .8: .104 / .084 / .060 (n 500 / 1000 / 4000) |
| pcorr_hac_fb (`inference: fixed_b`, 1.1, R-38) | as pcorr_hac; set-D member (T8) | same statistic and bandwidth, Kiefer-Vogelsang (2005) Bartlett fixed-b p | native, eq_min | F3 vs KV Table I PASS; F4 iid .047-.053, AR(1) .8: .091 / .080 / .056 |
| rcot2 (`xm-rcot-v2-lpd4`) | CI test | Strobl et al. LPB4, num_f 100 | eq, eq_min, native | F4 null .084 / .065: authors' null LIBERAL, kept for faithfulness |
| pc (causal-learn 0.1.4.8) | classic discovery | Fisher-z .05; score = pMax | eq (tier-0 design nodes), native | PASS-WITH-NOTES |
| notears (xunzheng/notears@4a9ab19) | classic discovery | w_threshold .3 | native only | F3 19/20; included, disclosed |
| shap_dag (XGBoost 3.4.1 + shap) | O-RAN (Sharma et al.) | "most influential" | native only | PASS-WITH-NOTES |
| two_tower (E2 port, row-share score) | ADAPTATION inspired by arXiv:2601.13213 | relative gate .10 | native only | not the published supervised model |
| cdl (port of the authors' NaNA 2026 conference method, refactor/codebase `cdd_oran/models/cdl.py`; version TBD, R-50) | the authors' conference method: Causal Dynamics Learning (Wang et al. 2022), masked neural predictor, per-edge CMI from likelihood ratios, EMA | conference fixed threshold CMI >= .16 (secondary, V0); primary: conformal tau | native only | TBD (worker cdl: F1 / F3 vs the conference golden runs, F4) |
| corr (scipy pearsonr) | simple | Pearson t-test | native only | PASS |
| granger (own OLS F == statsmodels) | simple, E3 only | F-test, one lag | eq (VARX), native | PASS-WITH-NOTES |

pdcor (partial distance correlation, `xm-pdcor-v2-projperm`) was considered and excluded from every arm (R-48): pdCor
= 0 is not conditional independence, so it is not a valid CI test under conditioning (CI-test audit smoke: E2 R2 eq
truth-null rate .22, P_placebo_conf declared 5 / 5), and its cost is O(n^2). cmi_knn (tigramite CMIknn,
`xm-cmiknn-tigramite-5.2.10.1-bc`) was considered and excluded from every arm (R-49 revised) because of its cost
(Kaggle T4, torch neighbour search: ~2.4k-6.1k s per dataset at n 4000, ~9k s extrapolated at n 8000); pcorr and
rcot2 remain as CI tests.
Hyperparameters = authors' defaults; nothing tuned on DEV except R-14. CI-test adapters, pcorr_hac and cdl enter only
under T4. Conditioning-set sizes |Z| for focal P0 (native / eq_min / eq): R1 E1 8/8/18, E2 14/14/32, E3 9/9/19, E4
2/2/6, E5 8/8/18; R2 E1 8/9/23, E2 14/15/41, E3 9/10/24, E4 2/3/8, E5 8/9/23; E4 R3 4/4/10 (DEV seed 3_000_000, n 500).

## 5. Arms (R-17, R-18, R-19, R-25, R-28, R-33)

- PRIMARY block: `pmrt_nl_eq`, the PMRT arm (spec `focal`; pmrt_core with covariates eq and the statistic chosen by
  the R-42 rule, T10; it is "the PMRT arm" of C2a, C3, V3 and T1; if no candidate passes, pmrt_eq, T10 fallback),
  every `*_eq` arm (WHOLE conditioning set from
  `covariates.design_covariates(data, focal=<action>)`: R-3 set + setpoints of all designed actions + all actions at
  t-1, t-2 + context in R3; PC tier-0 nodes; Granger exogenous regressors; lagged-KPI sources per R-28) and the
  methods without a conditioning interface (corr, notears, shap_dag, two_tower, cdl; native only, disclosed, kept
  apart in every table by their rule).
- SECONDARY block: `pmrt_eq` (pmrt_core's linear statistic, `pmrt-core-v1`; secondary since R-42, labelled);
  `*_native` (R-3 set, design-blind, R-18); `*_eq_min` (R-3 set + `sp:<focal>` only, the minimal design
  information, R-33; = native in R1 and for lagged-KPI sources); pcorr_hac(_fb) (+ eq_min); pmrt_r3 (R-19).

## 6. Declaration rules (R-2, R-6, R-9, R-22, R-23, R-29)

- p arms (`declare: by`): BY at q .05 over the dataset's PRIMARY family (action -> KPI incl. P_placebo), by the
  adapter; secondary family (lagged KPI -> KPI) its own BY. pmrt_core counts not-applicable candidates (R4) in its m
  (`notes['by_family_m']`, conservative); the V11 BY re-check uses it.
- tau arms: declare iff score > tau, tau = the ceil((M+1)(1-.05))-th smallest of the (arm, cell)'s M tune-seed
  P_placebo scores (the largest if the index exceeds M): out-of-sample placebo level <= .05 under exchangeability
  (R-29, shared `score.placebo_tau`; E4: M 20, tau = max, level 1/21); the KPI -> KPI family reuses it. p arms (incl.
  the PMRT arms) are also scored with tau (V0, V3) and at raw p <= .05 per edge, no multiplicity (V0; R-42). cdl is
  also scored at the conference's fixed threshold (declare iff CMI >= .16, spec `fixed_threshold`; R-50): a
  secondary scoring (V0 rule `fixed`), not tuned, so all three of its rates enter its validity. Not testable / not
  applicable = NOT declared, counted (R-23). Signs R-4.

## 7. Grid, seeds, budget, platforms

- n grid {500, 1000, 4000, 8000, 24000} for every arm; feasibility per (arm, n) fixed from DEV cost (T3); a
  T3-infeasible cell is shown as "infeasible (DEV cost X)" and is not part of the arm's pre-registered cell set.
- Measurement seeds from the EVAL block: S seeds 3_100_000.. (T1) for every cell, except that the C3 readers
  (pmrt_nl_eq, pmrt_eq, pmrt_r3, every eq arm; R-39) use S_E4 = 300 seeds (T9) in E4 R3 / R4; sweep cells the first
  ceil(S/2). Tune seeds = DEV tune seeds 3_000_000-019 re-run at the freeze commit (R-34; sha and declarations vs DEV
  reported, V11). `specs/eval/full.json`: 25 arms; S 40: 207 560 units / 16 200 datasets (46 480 tune); S 100:
  323 180 / 19 500. The EVAL n grid of the mscr arms at large n is open (T11).
- Budget (R-13, R-35): 7200 CPU-s per (method, dataset) decides feasibility (T3, Kaggle reference host). In EVAL
  RLIMIT_CPU = 14 400 (2x safety cap): a unit over it is recorded infeasible with its CPU-s, never re-run, and its
  cell is infeasible; an OOM kill is an error (re-run). RNG tags 7800 data, 7801 pmrt, 7802 citests, 7803 classic.
- GPU budget (R-41, the R-13 analogue): an arm run on a GPU backend (spec `budget_wall_s`; none since cmi_knn was
  dropped; cdl's device is fixed from its cost, spec `tbd.cdl`) has a budget of 2 h wall (7200 s) per (method,
  dataset) on a Kaggle T4 (`budget_cpu_s` none); over it = "infeasible at this n (measured cost X)", X in GPU wall-s
  (T3). In EVAL the same 2x safety cap applies (14 400 wall-s), same rule as the CPU cap. GPU wall time is reported
  apart from CPU-s (V10, Experiment C), never summed with it. Arms on CPU backends report any `gpu_s` (V10), not
  budgeted.
- Platforms (R-26, R-35, R-41a, R-46): packages pinned to uv.lock everywhere (`pkgs_lock`); shards split across
  Kaggle, Lightning and Colab in parallel (R-46); a dataset's arms on one platform, in one shard, except that GPU
  arms may run in a separate GPU shard on the same platform (R-41a). Every record stamps the dataset hash (`dataset_sha256`, sha256 over every array and name of the generated
  dataset), and all arms of a dataset must carry the same hash (eval_analysis V11: a mismatch, or an ok record without
  a hash, makes the run PROVISIONAL; a unit not run has no dataset and is only counted). EVAL shards in their own
  directory (tune keys equal DEV keys).

## 8. Metrics (per (arm, cell) on measurement seeds; cell = (world, regime, lambda, kappa, n))

- Rates: truth-null (primary truth-null candidates), placebo (P_placebo), confounded placebo (P_placebo_conf); for p
  arms raw (p <= .05) and declared (BY), for tau arms declared. Not-testable candidates leave every denominator;
  not-applicable ones (no p) leave the raw denominators and count as not declared (V7).
- 95 % CI: seed-cluster percentile bootstrap (2000 reps, rng 20261002, seeds sorted), Wilson with design effect
  alongside. Pooled rates cluster on (world, seed) (separate RNG streams per world).
- Validity per rate (R-30): INVALID if the CI lower bound > .05 (takes precedence); VALID if the upper bound <= .075
  (Bradley); else INCONCLUSIVE. Cell validity over its rates (p arms: all raw and declared; tau arms: truth-null and
  confounded-placebo declarations; P_placebo is their tuning column, reported only): INVALID if any rate is, VALID
  if all are, else INCONCLUSIVE ("no read" if none, e.g. tau arms in E4 R1 / R2); < 10 seeds: "few seeds".
- Power (R-39: cells not INVALID; VALID-only is the sensitivity read, since at S 40 an exactly nominal cell is VALID
  w.p. only .25-.45): recall per seed (mean, sd), FDP (mean = FDR, CI), pooled sign accuracy. Also counted: not
  testable / not applicable; cost (CPU-s; wall-s for the GPU arms, R-41; peak RSS, GPU-s, infeasible and error units).
- PMRT in R4 (R-42): the real actions have design kind `none`, so a PMRT arm tests no true edge there. Its power
  (recall, sign) is "not applicable" (NA), never recall 0, and is left out of every recall table, mean and paired
  difference (V0, V2, V3, V6). Its placebo rates are still read (P_placebo has an i.i.d. design in R4).

## 9. Analyses

HEADLINE (R-42, descriptive, no verdict): V0 like-for-like table, printed first. Every p arm is scored twice: at raw
p <= .05 per edge (no multiplicity) and with the conformal placebo tau of section 6, like the score-only arms. It
sits next to the score-only (tau) arms. Per cell and scoring: mean per-seed recall, truth-null rate and placebo
rate (declaration rates, CI, three-way validity; a tau scoring's placebo is its tuning column, reported only).
Recall is shown for every counted cell next to its rates. PMRT in R4 is NA (section 8). cdl also has a row at the
conference's fixed threshold (rule `fixed`, secondary; section 6).
PRIMARY (kappa .25): A1 validity (V1), verdicts C1, C2a, C2b, C3 (V4); A2 recall among cells not INVALID (V2), paired
recall differences PMRT arm - arm under the SAME rule (V3: BY arms vs the PMRT arm's BY, tau arms vs the PMRT arm
scored with tau; per-cell paired-t CI, descriptive; E4 rows descriptive). SECONDARY (no verdict): information levels
in R2 with |Z| (V5), KPI -> KPI (V8), E4 by lambda (V6), kappa sweep (V9), not testable (V7), cost (V10), the C2a /
C3 rules for the other PMRT arms (pmrt_eq, pmrt_r3; V4, no effect).

## 10. Support rules (computed by `eval_analysis.py`, kappa .25)

An arm's pre-registered cell set = its grid cells minus T3 cells. A cell is counted if its status is ok or
under-seeded (some planned seeds missing) with >= 10 seeds. Caps (R-35): a component is NOT EVALUABLE if < 90 % of
its planned cells are counted, and SUPPORTED is capped at PARTIAL if any planned cell is not counted (missing,
EVAL-infeasible, untuned, few seeds) or is under-seeded; such cells never drop out silently (listed in V4 / V1).
F_max (R-30) = smallest f with P(#INVALID <= f) >= .95 for an arm whose tests have level exactly .05, by
`eval_analysis.fmax_simulate` (2000 runs, seed 20261003): S seeds shared by all cells; null statistics Brownian in n
(DEV check on E1: corr .68 / .51 / .34 vs sqrt(n1/n2) .71 / .50 / .35); E4 lambdas (.69-.95) and same-dataset
candidates (|r| <= .12) correlated as estimated in `results/fmax_sim/dependence.json` (`fmax_calib.py`, DEV seeds
3_000_100-199, pcorr stand-in; sha in the spec); worlds, regimes independent; the analysis' own bootstrap per cell.
Fixed values (`results/fmax_sim/fmax_table.json`; other cell sets by the same function):

| cell set (rates null_raw, plac_raw) | S 40 | S 50 | S 60 | S 70 | S 80 | S 90 | S 100 |
|---|---|---|---|---|---|---|---|
| R1 + R2, 50 cells (C2a, C2b): F_max (P pass at exact level) | 3 (.93) | 3 (.92) | 3 (.92) | 3 (.91) | 3 (.92) | 4 (.94) | 4 (.94) |
| E3 R1 + R2, 10 cells (granger_eq): F_max (P pass) | 1 (.91) | 2 (.95) | 1 (.93) | 1 (.92) | 1 (.92) | 2 (.95) | 2 (.96) |
| R1, 25 cells (C1 leg iii); E3 R1, 5 cells (granger_native) | 2; 1 | 2; 1 | 2; 1 | 2; 1 | 2; 1 | 2; 1 | 2; 1 |
| R2, 25 cells: P(pooled rate INVALID) at exact level (C1 leg ii false alarm) | .049 | .048 | .036 | .042 | .048 | .042 | .052 |

C3 set (E4 R3, 20 cells, P_placebo and P_placebo_conf raw): F_max 2 at S_E4 200 / 300, P(pass) .82 / .93 (T9;
one candidate per dataset, shared seeds). Chance INVALID per cell at an exact level .02-.026; non-monotone entries
are Monte Carlo steps of an integer quantile (P(#INVALID <= F_max) >= .95 holds for each).

- C1 (design-blind tests invalid on R2). D = the native p arms of the frozen spec with `set_D: true`: corr,
  granger_native, the HAC variant chosen under T8 and the native CI tests admitted by T4 (a dependence test that is
  not CI would make C1 a strawman; R-40, Q11). The HAC variant not chosen gets the same legs reported
  descriptively, no effect. Arm a is a FAILURE iff (i) INVALID in >= half of its planned R2 cells
  (a cell not counted is not INVALID), (ii) its pooled R2 truth-null or placebo raw rate is INVALID and (iii) #INVALID
  R1 cells <= F_max(R1 set); if (iii) fails, "INVALID IN R1" (invalid regardless of the design, e.g. a liberal
  authors' null; not a failure); else NOT A FAILURE. C1 needs >= 3 assessable arms (>= 90 % of R1 and R2 cells
  counted), else NOT EVALUABLE; SUPPORTED if >= half fail, NOT SUPPORTED if none, else PARTIAL; capped; a verdict per
  arm. If the HAC member is NOT A FAILURE, C1 reads "design-blind tests with i.i.d. / exchangeable nulls are invalid
  on the R2 design tested; the serial-dependence-robust design-blind test is not".
- C2a (the design-based test is valid). The PMRT arm (pmrt_nl_eq, spec `focal`) over its R1 + R2 cells: (a)
  #INVALID <= F_max; (b) the pooled truth-null and placebo raw rates are VALID. SUPPORTED iff (a) and (b); capped.
  The same rule for the other PMRT arms (pmrt_eq, linear statistic; pmrt_r3) is reported, no effect (R-42).
- C2b (design covariates restore validity). Every eq p arm except mscr_eq whose native partner is a C1 FAILURE, same
  rule on its R1 + R2 cells, a verdict per arm; SUPPORTED if >= half pass, NOT SUPPORTED if none, else PARTIAL;
  capped. Wording: C2a and C2b SUPPORTED -> "using the design restores validity (design-based inference and
  design-covariate adjustment)"; C2a SUPPORTED, C2b not -> "design-based inference restores validity; adding design
  covariates does not suffice" (+ the arms for which it does). R-40: mscr_eq (spec `c2b: false`) is not equal
  information, so it is excluded; its same-rule R1 + R2 result is reported, no effect, labelled "single-conditioner
  max statistic; cannot condition on the joint design set" (also on mscr_eq_min).
- C3 (valid under the logged confounded policy). The PMRT arm over E4 R3 (4 lambda x 5 n, S_E4 300): (a) #INVALID
  <= F_max on P_placebo and P_placebo_conf raw; (b) both pooled raw rates VALID. Reported with the same rule for every
  eq arm (tau arms: P_placebo_conf declarations), no claim effect; if any also passes, the wording adds "so do
  <arms>; the property is not specific to PMRT", and PMRT's property is stated as "by construction" only. The other
  PMRT arms get the same rule reported, no effect, never among those <arms>.
- Claim SUPPORTED iff C1, C2a, C2b and C3 are SUPPORTED; otherwise V4 prints the wording the components license.
  A re-read without n 500 is descriptive only. V0 (headline), the R4 negative control (V6) and V3 sit next to the
  verdicts. The estimand check (`estimand_check.py`, section 0) is re-run at the freeze commit; a failing candidate
  is reclassified.

## 11. Execution and integrity

- Driver `cdd_oran/xmethod/campaign.py` (xm/dev-runs, built and audited for the freeze, R-35): EVAL seeds only if the
  spec's `protocol_sha256` equals this file's LF sha256 and it says "FROZEN: yes"; records stamp code commit,
  `run_mode.protocol_sha256`, `run_mode.spec_sha256`, pkgs, host. Shards on Kaggle, Lightning and Colab (R-46);
`campaign merge`.
- `eval_analysis.py` uses only records whose key is an expected unit with the expected role. FINAL (exit 0) needs:
  protocol frozen with the spec's sha; every record from the freeze commit or an amendment listing its key; dirty
  exactly false; stamps equal the spec's shas; no missing, unexpected, wrong-role or duplicate key; full candidate set
  per record; every error unit listed as persistent; dataset sha stamped on every ok record and equal across all arms
  and shards of a dataset (R-41a); one platform per dataset; one pkgs set equal to `pkgs_lock`; F_max inputs' sha as
  in the spec. Else PROVISIONAL (exit 2), failed checks listed. Error units are re-run until none remain; one failing
  every re-run is listed in amendments.json (key, reason) and its cell is under-seeded or few-seeds, which caps the
  verdicts.

## 12. Outputs (`eval_tables.json` + `EVAL_TABLES.md`)

V0 like-for-like headline, printed first (R-42; section 9); V1 validity per (arm, cell) with status; V2 recall / FDP
/ sign (not INVALID; NA = not applicable); V3 same-rule differences; V4 verdicts, F_max, wording (+ the other PMRT
arms, no effect); V5 information levels; V6 E4; V7 not testable; V8 KPI -> KPI; V9 kappa sweep; V10 cost (GPU arms in
wall-s, a separate table; R-41); V11 integrity. Figures (not drawn): validity map, recall vs n, information levels,
E4 by lambda, cost vs n.

## 13. Register of values fixed at the freeze (rules fixed now)

- T1 seed count S: `eval_analysis.py t1 --spec <DEV specs> --merged <files> --cap <T6>` on the full DEV runs
  (dev-runs `results/dev/full/merged.jsonl.gz` + the citests DEV merged file; paths and LF sha256 in the freeze
  commit). Kappa .25 cells at n in {500, 1000, 4000}, E4 excluded (one true edge); every pair PMRT arm vs a
  primary-block arm under the same declaration rule, neither INVALID on DEV, gives the paired-t seed count for a
  recall gap .15 (alpha .05, power .8); S = min(cap, max(40, max count rounded up to a multiple of 10)); minimum
  detectable gap at S per pair reported; the max over noisy DEV sd's biases S upward (disclosed). The PMRT arm of the
  pairs is the primary one (`t1 --focal pmrt_nl_eq`): dev-runs runs the T10 winner on the T1 cells (DEV measure
  seeds) and that DEV run is in the T1 input; under the T10 fallback, pmrt_eq. CI DEV plan C (R-47): mscr's DEV run
  covers n <= 1000 only, so its pairs use n 500 / 1000. cdl (R-51): dev-runs runs it on the full DEV grid before the
  freeze, and that run is in the T1 input (T4 (iii) applies, no waiver). VALUE: TBD.
- T2 tau: from the EVAL tune records (section 6); not a DEV quantity.
- T3 infeasible (arm, n): max over worlds, regimes, lambdas of DEV `cpu_s` per unit, in Kaggle reference seconds,
  > 7200 makes n and every larger n infeasible (spec `max_n`, `t3_cost_cpu_s`). DEV costs from any platform count
  (R-51): a unit run on another host type (Lightning, Colab) is converted as cost x f, f the measured speed factor of
  that (arm, host type) = sum of Kaggle cost / sum of host cost over the same calibration units run on both hosts
  (DEV seeds 3_000_000-002 in the arm's 3 costliest (world, regime) at its largest measured n). The units, both
  costs and f are recorded in the freeze commit. Without a DEV cost at that n: a cost pilot on those calibration
  units, on Kaggle or on another host with its f; its max decides. Pilot: none infeasible (max 144 s). GPU arms
  (R-41): the same rule on DEV wall-s per unit (child wall clock when isolated), referenced to a Kaggle T4 > 7200
  (spec `t3_cost_wall_s`), f measured against a Kaggle T4. VALUE: TBD.
- T4 admitted CI-test arms, HAC arms and cdl: iff (i) the F1-F6 line in `status/xm-citests.md` (HAC: `status/hac.md`;
  cdl: its fidelity line in `status/cdl.md`, R-50) and
  (ii) the verdict in `status/audit-citests.md` are PASS or PASS-WITH-NOTES (audit table: OK, or a doc / protocol FIX
  whose fix is merged; mscr's M1 protocol fix is R-40, so mscr, a faithful port, stays admissible; Q11), and (iii) its
  DEV run is in the T1 input. rcot2's liberal authors' null is a disclosed note, not a failure. Others removed, with
  reason. VALUE: TBD.
- T5 DEV validity (R-21): computed by `eval_analysis.py` on the DEV records, reported in the freeze commit; not used.
- T6 S cap: 100, unless `campaign project` at S 100 with the DEV costs (incl. citests) exceeds the compute the user
  states at the freeze; then the largest multiple of 10 that fits, not below 40 (R-34). VALUE: TBD.
- T7 freeze-commit contents: campaign EVAL mode (audited), all adapter branches, `uv.lock`, filled `pkgs_lock`, this
  file FROZEN, the estimand-check re-run. VALUE: TBD.
- T8 set-D HAC variant (R-38): pcorr_hac vs pcorr_hac_fb on synthetic F4 only (AR(1) rho .5 / .8 and iid, n 500 /
  1000 / 4000): the smaller max |rate - .05| gets `set_D: true`, the other false (secondary). VALUE: pcorr_hac_fb
  (fixed-b .0405 vs t .0535; calib F4, xm/hac2 `results/pcorr_hac_f4b.json`); pcorr_hac (t) secondary.
- T9 S_E4 (R-39): the smallest of {200, 300, 400, 600} at which an exactly nominal PMRT arm passes C3 with
  probability >= .90 in `fmax_simulate` (`results/fmax_sim/s_e4_rule.json`; the simulation does not depend on the
  statistic). VALUE: 300 (P .93; 200: .82).
- T10 PMRT statistic of pmrt_nl_eq (R-42): statistic chosen by the R-42 rule. Among the candidate nonlinear
  statistics (each a function of the redrawn random part and quantities fixed under the redraw, so the CRT stays
  exact) that pass exactness (synthetic F4 + DEV truth-null and placebo rates not INVALID in any cell, R-30) and the
  R-13 budget at n 4000, pick the highest mean per-edge recall at raw p <= .05 over DEV E2 / E5 R1 / R2 n 1000 + the
  synthetic nonlinear scenarios; ties -> the cheaper. DEV seeds 3_000_100-159 only (3_000_160-189 reserved). Worker
  pmrt-nl applies it (status `pmrt-nl.md`). The winner's option and version string go into pmrt_nl_eq's spec config
  (placeholder `statistic: TBD-R-42`). The linear pmrt_eq stays as the secondary arm. Fallback: if no candidate
  passes the gates, pmrt_nl_eq is removed from the spec and the linear pmrt_eq is the primary PMRT arm (spec
  `focal: pmrt_eq`, analysis primary, its R-42 label dropped) in every rule above that names the PMRT arm. VALUE: TBD.
- T11 EVAL grid of the mscr arms at large n (R-47): OPEN. Under CI DEV plan C mscr has DEV records at n <= 1000
  only, and the EVAL grid at larger n is decided with the user at the freeze (the R-47 question also named pdcor and
  cmi_knn, both dropped since: R-48, R-49 revised). Until then the spec keeps the full n grid (spec
  `tbd.large_n_citests`); T3 still decides feasibility for any n kept. Rule: TBD with the user. VALUE: TBD.

## 14. Deviations

After the freeze this file is not edited. Amendments (date, what, why, records touched) go to
`docs/xmethod/PROTOCOL_A_AMENDMENTS.md` and, if they touch records, `specs/eval/amendments.json` (id, commit, keys;
persistent errors: key, reason), before the affected records are analysed where possible. A bug found after EVAL:
fix, re-run every affected unit under a new commit listed there, report original and corrected results. EVAL never
feeds back into a method, tau or this protocol; a second EVAL needs a fresh seed block and a new protocol version.
