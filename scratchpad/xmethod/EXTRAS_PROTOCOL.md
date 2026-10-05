# Study A supplementary experiments X2 / X3: declaration (R-60)

DECLARED 2026-10-05, before launch and before any EVAL output was read. The author (worker exp-b) opened no EVAL
result, shard, merge or table. Ruling: CONTRACT R-60 (feat/v2 2f01f1e). The commit that adds this file is the
declaration. Later changes go only into the dated "Amendments" section at the end, and the report lists them.

**Status: EXPLORATORY / supplementary.** These runs never change the C1-C3 verdicts, the Study A claim or the E6
headline (PROTOCOL_A is untouched).

## 1. Rules (R-60)

- **Own files.** Specs live in `scratchpad/xmethod/specs/extras/`, with their own output directories. Never
  `specs/eval`, never EVAL shards.
- **Frozen methods and procedures.** Every arm's ref, config and declaration are copied from the frozen
  `specs/eval/full.json` (a test checks this). tau is the frozen R-29 conformal cutoff. Scoring uses the frozen
  `eval_analysis.py`.
- **Wrapper only, no frozen file edited.** `cdd_oran/xmethod/extras.py` overrides two generator constants (X2) or the
  design told to a method (X3) in its own process only. The freeze manifest
  (`scratchpad/xmethod/freeze/FREEZE_MANIFEST.sha256`, LF rule) stays valid, and a test checks it.
- **Seeds.** A fresh block, 3_200_000-3_200_199, claimed as XMETHOD_EXTRAS for E1-E4 in
  `docs/benchmark/SEED_REGISTRY.json` (section 2). Nothing touches seeds 3_100_000-3_100_299.
- **Launch timing.** Start only after EVAL has launched all 96 parts (Kaggle) and its last VPS job has been pulled.
  EVAL requeues keep priority: keep one Kaggle slot free until the EVAL merge checks pass.
- **Order.** X1 > X2 > X3. X3 runs only if capacity remains. X1 (paper timing, Exp C) is declared by its own
  pre-registered block, not here.
- **Every declared run is reported**, whatever the outcome. Cells that are infeasible, in error or missing are
  printed as such, never dropped. Error records are re-run (campaign resume). A unit over budget is infeasible.

## 2. Seeds (XMETHOD_EXTRAS)

| range | use |
|---|---|
| 3_200_000-3_200_019 | X2 tune (20 seeds; tau of the score-only arms, per design) |
| 3_200_020-3_200_059 | X2 measure (40 seeds per design) |
| 3_200_060-3_200_119 | X3 measure (60 seeds) |
| 3_200_120-3_200_199 | unassigned; any use needs an amendment before its launch |

- **X2 designs share the seed numbers** (common random numbers). The dither of a seed is the same uniform draws
  scaled by delta, so the delta comparison is paired.
- **The wrapper refuses** any other seed for an extras spec. Campaign's own guard still applies to every other spec.

## 3. X2: dither dose-response (C1 mechanism)

**Question.** How do the truth-null and P_placebo rejection rates, and recall, of these arms change with two design
settings?
- The arms are: the design-blind arms corr, pc_eq, notears, shap_dag and pcorr_native; the design-covariate arm
  pcorr_eq; and the design-based arms pmrt_eq and pmrt_nl_eq.
- The settings are: the dither amplitude delta at the frozen 20 setpoint blocks, and the number of setpoint blocks at
  the frozen delta .10.
- No directional hypothesis is tested. The result is descriptive.

**Cells.**
- E1, E2 and E3, regime R2, n 1000 and 4000, kappa .25.
- 6 designs (delta, blocks): (.02, 20), (.05, 20), (.10, 20), (.20, 20), (.10, 5), (.10, 80). One spec each:
  `x2_d02`, `x2_d05`, `x2_d10`, `x2_d20`, `x2_b05`, `x2_b80`.
- Only R2 reads the two constants. The envelope (.25) and everything else stay frozen.

**Reproduction cell.** (.10, 20) is the frozen EVAL R2 design. At equal seed its datasets equal the frozen
generator's bit-for-bit:
- tested in `tests/test_xmethod_extras.py`;
- the analysis regenerates every dataset of `x2_d10` with the frozen generator and requires equal hashes;
- for the other designs it requires every R2 dataset to differ.

**Seeds and declaration.**
- Tune 3_200_000-019 for the score-only arms pc_eq, notears and shap_dag only. Their tau is re-derived per design and
  cell by R-29 on that design's own tune records.
- Measure 3_200_020-059 for all 8 arms.
- p arms declare by pooled BY (q .05), as frozen. No conformal tau is computed for p arms.

**Metrics.** As in section 5.

## 4. X3: design-misspecification stress test (C2a assumption)

**Question.** Where does the known-design assumption break? The data are generated with the frozen design; only the
design TOLD to the method is wrong. The result is reported as a degradation curve, not as evidence against C2a
(which assumes the design is known).

**Cells.**
- E1 R2 and E2 R2 at n 1000.
- E4 R3, lambda 1, at n 1000 (propensity distortion; the code is small, so it is included).
- kappa .25, measure seeds 3_200_060-119 (60). One spec: `x3_told`. The data are generated once per dataset and
  shared by every variant.

**Arms.** pmrt_nl_eq, pmrt_eq, pmrt_r3, and pcorr_eq as the contrast. Each runs with the exact design (arm name
unchanged) and with every variant (arm name suffix).

**Told variants.** Each variant changes every applicable design of the dataset, including P_placebo's dither:
- **Dither width** x0.5, x0.8, x1.25, x2 (`.w050`, `.w080`, `.w125`, `.w200`; E1 / E2 R2): each dither design's law
  U(-w, w) is told as U(-c w, c w). The realised random and fixed parts are unchanged.
- **Switch times late by 2 % / 5 % of a block** (`.s02`, `.s05`; E1 / E2 R2):
  - the told setpoint is the true one shifted s rows later, s = floor(f L + .5) with L = n / blocks = 50 rows, so
    s = 1 and 3 rows (effective 2 % and 6 %; the rounding is declared here);
  - told random part = action - told setpoint, so it is exact wherever the told setpoint is right.
- **Logging lambda** x0.5, x0.8, x1.25, x2 (`.l050`, `.l080`, `.l125`, `.l200`; E4 R3):
  - every logged design's propensity table is recomputed with the told lambda from the observed Z;
  - the i.i.d. P_placebo design is not touched, so its rate is a control.

**Expected invariances** (verified on one dataset before this declaration; any difference is a harness bug, reported
as such):
- pcorr_eq is identical under the width and lambda variants, because it never reads the design law or the
  propensity;
- pcorr_eq changes under the switch-time variants, because its eq set uses the told setpoints as covariates.

**Declaration.** All X3 arms are p arms (BY). There are no tune seeds.

## 5. Metrics (as the frozen eval_analysis defines them)

Per cell (spec / design, arm, world, regime, lambda, n, kappa), from `eval_analysis.screen` and `build_cells` on
the spec's merged records. Only expected units with the expected role enter.

- **Truth-null rate.**
  - Applies to the primary-family (action -> KPI) candidates with zero true effect.
  - "decl" = declared under the arm's primary declaration (BY for p arms; score > tau for tau arms).
  - "raw" = p <= .05 (p arms).
  - Denominator: the testable candidates. Not-testable candidates are left out of every denominator.
- **P_placebo rate** (and P_placebo_conf in E4 R3): the same, over the placebo -> KPI candidates. For tau arms
  P_placebo is the tuning column (reported, not a validity rate).
- **Recall:** the mean over measurement seeds of per-seed recall (true primary edges declared / true primary edges).
- **CIs:** seed-cluster 95 % percentile bootstrap (2000 reps, seed 20261002).
  - A cell's clusters are its seeds.
  - Pooled rates per (spec, arm, regime) use `eval_analysis._pooled`, with clusters = (world, seed), over all cells
    of that arm.
- **The R-30 class** (INVALID if CI low > .05, VALID if CI high <= .075, else INCONCLUSIVE) is printed as a
  descriptive label, not a verdict.
- **Status and provenance:** the cell status (ok / few_seeds / under_seeded / infeasible / missing / untuned), the
  record counts and the provenance check (section 3; X3 requires every record's generated-data hash to equal the
  frozen generator's).

Output: `uv run python -m cdd_oran.xmethod.extras analyse --spec <specs> --merged <merged> --out
scratchpad/xmethod/results/extras/` writes `extras_tables.json` and `EXTRAS_TABLES.md` (cells, pooled, X2 trend, X3
curve).

## 6. Report wording (exact; the placeholders are filled from extras_tables.json)

**Lead (always, verbatim):** "These supplementary experiments are exploratory: they were declared before launch
(scratchpad/xmethod/EXTRAS_PROTOCOL.md) on a fresh seed block, use the frozen methods and procedures, and do not
change C1-C3." Followed by "(declaration commit <sha>)".

**X2, one sentence per arm and rate** (truth-null decl; truth-null raw and P_placebo raw for p arms):

> "X2 (dither dose-response; R2 of E1-E3, kappa .25, n 1000 and 4000, 40 measurement seeds per design): the pooled
> <rate> of <arm> was <r> [<lo>, <hi>], <r> [...], <r> [...] and <r> [...] at dither delta .02, .05, .10 and .20 (20
> setpoint blocks), so it <trend> with delta, and <r> [...] / <r> [...] with 5 / 80 setpoint blocks at delta .10."

- **`<trend>`** is the declared rule on the four point estimates in increasing delta:
  - "rises" if they never decrease and at least one increases;
  - "falls" if they never increase and at least one decreases;
  - "does not change" if all are equal;
  - otherwise "is not monotone";
  - "not reported" if a design has no rate.
- **Appended sentence:** "The delta .10 / 20-block design is the EVAL R2 design (datasets bit-identical to the frozen
  generator at equal seed); its seeds are disjoint from EVAL."

**X3, one sentence per base arm, regime and family** (width narrower x0.8 then x0.5; width wider x1.25 then x2;
switch late 2 % then 5 %; lambda lower x0.8 then x0.5; lambda higher x1.25 then x2):

> "X3 (design misspecification; data generated with the frozen design, the design told to the method wrong; 60
> measurement seeds): with the <family> the pooled truth-null raw rate of <arm> in <regime> was <r> [...] (exact),
> <r> [...] and <r> [...], and its P_placebo raw rate <r> [...], <r> [...] and <r> [...]; the first variant with an
> INVALID pooled rate is <variant | none>."

- In E4 R3 the sentence also gives the P_placebo_conf raw rate.
- **Closing sentence (verbatim):** "This is a degradation curve showing where the known-design assumption breaks; it
  is not evidence against C2a, which assumes the design is known."
- **Recall** of every cell and pooled arm is in the tables. The text gives recall only alongside the rates above, in
  the same format.

## 7. Cost and launch (DEV cost table; Kaggle-reference CPU-h, an upper estimate)

`scratchpad/xmethod/specs/extras/PROJECTION.md`:
- each X2 spec: 2280 units, 360 datasets, 9.6 CPU-h (pmrt_nl_eq 7.1);
- x3_told: 4560 units, 180 datasets, 19.6 CPU-h (pmrt_nl_eq E4 R3 extrapolated: DEV had no pmrt_nl E4 cost);
- total 77.2 CPU-h.

The launch commands are in `scratchpad/xmethod/status/extras.md`.

## Amendments

### 2026-10-05: X4, random accept / defer referee (Study 3 / E6-P certsafe baseline)

**POST HOC, DESCRIPTIVE.** X4 was added on 2026-10-05, after the Study 3 results were known: the certsafe EVAL
(`docs/benchmark/E6P_CERTSAFE_PROTOCOL.md`; verdict PARTIAL; CS:PMRT identical to never_sleep on all 160 seeds) and
the report `reports/2026-10-02-v4-pmrt-confounded-certsafe/`. Requested by the orchestrator (user GO); author worker
exp-c. X4 tests no hypothesis and has no criterion. It changes no frozen artifact, verdict, protocol, driver or
analyzer of Study 1-3, and no X2 / X3 rule above. The commit that adds this entry is the declaration; it precedes
every X4 episode (the 1-2 seed local plumbing test below is not a result).

**Question.** Where does a referee that defers at random, blind to the request and to any map, land on the Study 3
scale (R, retention, guard ratios, eligibility)? It is a reference point for MG:PMRT's deferral volume.

**Policy** (`X4:rand@<p>`):
- Each xApp request reaching the referee is deferred (decision "reject") independently with probability p and
  accepted otherwise. Every family and direction is treated alike. No writes, no rollbacks.
- Active from t = 0, warm-up included (as MG:PMRT and never_sleep).
- RNG: one stream per episode, `default_rng([seed, 6640])`, one uniform u per request in arrival order
  (obs["requests"] order within a step). Defer iff u < p. All three p share the stream (common random numbers), so the
  three arms make nested decisions until their request streams diverge.
- Tag 6640 is new. It was checked free against `SEED_REGISTRY.json` rng_stream_tags. The registry is not edited:
  its sha256 is pinned in the E6-P record headers.

**p, three arms:**
- **p_PMRT = .352.** This is MG:PMRT's request-level deferral rate in the certsafe EVAL: deferred requests / requests
  reaching the referee = 37 890 / 107 525 = .35238 (records `scratchpad/e6_dev/runs/e6p-cs-eval-1/*/res_*.jsonl`,
  160 seeds, all t), rounded to .352.
- For reference only: the certsafe composition of MG:PMRT counts directional units, not requests. It deferred
  carrier 71, sleep 869, ptx 8313 and prot_min 0 units, and accepted 25 805, a unit-level rate of
  9 253 / 35 058 = .264. X4 defers requests, so the request-level rate is the one matched.
- **.25 and .50.**

**Episodes.**
- Seeds: the Study 3 EVAL seeds, 191100-191259 (160), cfg seed = seed. These are no new seeds; X4 claims none.
- Plant: identical to Study 3: `e6p_screen.make_cfg(e6p_conf.PAIR, e6p_conf.STRATUM, seed, lf)`, lf = L40 from
  `e6p_state.json`, 120 s warm-up unscored + 600 s scored.
- Jobs: 160 seeds x 3 arms = 480. The record schema is that of the certsafe EVAL with sub "x4". Every record carries
  p, realised deferrals per family / phase, requests, cpu_s and the platform.
- Code: one new file, `scratchpad/e6_dev/x4_random_defer.py`, plus its test. It imports the frozen driver modules
  and edits none of them.

**Platform.**
- Runs on the VPS (`scratchpad/e6_dev/vps_run.py`, `/opt/cdd-xm`, 7 processes in the capped systemd scope).
  Python 3.12.14, numpy 2.4.2 / scipy 1.18.1 (uv.lock pins).
- The stored Study 3 arms ran on Kaggle (Linux, Python 3.12.13, same numpy / scipy).
- **Provenance check, run first in the same VPS job.** noarb and never_sleep on seeds 191100-191101, made with
  the frozen driver's arbiters. Every outcome field is compared with the stored Kaggle record.
  - If all are equal, the plant is bit-identical across the two platforms.
  - Otherwise the differences are reported, and the X4 comparison is labelled cross-platform (still reported).
- **Cost.** X4 cost is VPS CPU-h, converted to Kaggle-reference CPU-h:
  - with `scratchpad/xmethod/results/exp_c/calib/factors.json` (pooled "*" factor, VPS Python 3.12.14);
  - and with the direct factor of this workload: stored Kaggle cpu_s / VPS cpu_s of the four provenance jobs.

**Scoring** (the Study 3 code, unchanged):
- Inputs: the stored EVAL records (20 shards; the aliased CS arms filled from their targets as
  `e6p_certsafe_analyze.cmd_eval` does), plus the X4 records.
- Call: `e6p_conf_analyze.arm_stats_multi` (N_BOOT 10 000, `default_rng([6624, 20, 160])`: the same bootstrap
  index matrix as `cs_eval.json`).
- Hence V_AA, V_ref, the reference arm and the denominator are the stored ones. Every stored arm must reproduce
  `cs_eval.json`; any mismatch is a harness bug, reported.
- **Per X4 arm:** V, R, R*, retention, guard ratios (svr, nonprot_embb_viol, ll_viol, rlf), eligible (the point
  rule, unchanged), and 90 % CIs (paired seed bootstrap). Also the realised deferral rate (deferred / requests) and
  deferrals per family.
- **Comparisons:** paired dR (90 % CI) to noarb (accept-all), never_sleep, MG:PMRT, CS:PMRT and CS:GT, all
  descriptive.
- MG:GT is not a Study 3 arm and has no record on these seeds, so the table shows CS:GT and states this.

**Output.**
- `scratchpad/xmethod/results/extras/x4/` (records mirror, `x4_tables.json`) and
  `scratchpad/xmethod/results/extras/X4_REPORT.md`.
- The report has one table (X4 arms next to never_sleep, MG:PMRT, CS:PMRT, CS:GT, accept-all) and a three-line
  descriptive reading.
- **Never:** a verdict, a claim, or a change to Study 3's E / D1-D3 / S1.
- Missing or failed jobs are reported as such, never dropped. A failed job is re-run (resume by key).
