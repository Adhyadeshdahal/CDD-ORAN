# E6-P discovery protocol v2: MSCR-CRT v2 re-test on fresh seeds (step 1, amended)

FROZEN: yes (2026-09-30)

**Status: FROZEN 2026-09-30 01:25 NST, before any record of eval_v2 / gt_ext / placebo_ext was opened.** This is a disclosed, outcome-informed amendment written AFTER the v1 verdict. It must
be frozen (this line set to "FROZEN: yes" and the file's LF-normalised sha256 recorded in the commit message and in
`.tmp/PLAN.md`) BEFORE any record of the fresh stages below is opened. The fresh records may be downloaded before the
freeze, but not parsed.

## 0. Why a v2 exists (disclosure)

**v1** (`E6P_DISCOVERY_PROTOCOL.md`, frozen f9d87740, commit 9164518) ran on EVAL 183320-183379 and GT
183380-183399. The result was **KILL**:
- K0 PASS (placebo rate .050, 0 BY), K1 PASS (228 sleep units, 62 rejects), G PASS (GT sleep → nbr pv TRUE(+),
  mean 15.4, CI [8.0, 23.7]).
- MSCR-CRT v1 declared 13/60 on pooled EVAL. Indirect recall was **3/8** (sleep nbr v, sleep nbr load, ptx nbr
  load), below the 2/3 bar.
- One sign error: prot_min → own pv was declared +1; the GT says −1.

The user authorised, for a KILL, a diagnosis by independent agents and an improved MSCR re-tested on fresh seeds
under a new, disclosed protocol version. Four agents (reports in `.tmp/diag/`, summarised in `.tmp/PLAN.md`) found:
1. **Power, not the method, bound recall.** A perfect unit-level test on 60 episodes is expected to find about 3-4 of
   the 8 GT-TRUE indirect edges. MSCR found exactly the three with an oracle z > 3. At 60 episodes, sleep → nbr pv
   has an expected z of about 1.2-2.5 depending on the statistic.
2. **The sign rule is a bug.** `crt_units._beta` regresses on x = level × sgn, but the request direction sgn is not
   randomized and predicts the KPI trend (Simpson-type mixing). Design-centred slopes give the GT sign.
3. **"half" is mis-dosed.** It is applied 88-100 % of the time because of the slew/alternation rule, but was coded
   0.5.
4. **far is not a null.** Sleep has TRUE 2-hop effects on far cells in the GT, so far cannot serve as a negative
   control, and the v1 "≤ 1 far declaration" rule and the baselines' far-FPR τ rule assume a false null.
5. **The associational baselines are not calibrated.** On the PLACEBO data (sharp null) corr declared 18, INT 10
   and Granger 11 hypotheses with their DEV τ; MSCR-CRT declared 0.
6. **The estimand mismatch (π0 vs AA continuation) is not the cause.** π0 accept-vs-reject contrasts on EVAL match the
   GT within about 1.4 SE.

**Forking-paths statement.** Every change below was chosen after the v1 KILL. Items 1-3 of section 2 are mechanism
or design corrections that the DEV data alone supports (the sign bug shows on DEV). Item 4 (sample size) follows a
DEV power analysis. The primary criterion's edge set (section 4) is the plan's a-priori "Expected" list
(`STEP1_MSCR_PLAN.md`, written before any data), not a list chosen from v1 outcomes. Nothing was tuned on v1 EVAL; v1
EVAL, GT, DEV and PLACEBO are development data for v2 and are never scored in v2.

## 1. Question

Does the amended MSCR-CRT v2, with an adequate sample, recover the indirect conflict chain of the E6-P plant (pico
sleep → neighbour load → neighbour protected violations, and PowerES ptx → neighbour load) from randomized logs, on
fresh seeds, with a fresh ground truth? And does it match or beat the baselines on the same data?

## 2. Method: MSCR-CRT v2 (`cdd_oran/decision/crt_units_v2.py`, version `mscr-crt-units-v2`)

Unchanged from v1: the unit table (`crt_units.build_unit_data`: H = 90 s, pre-window 90 s, relations own / nbr /
far with N(c) from `units_p.exposure_sets`, KPIs pv / v / e / rlf / load), the 60 hypotheses, the "dir" orientation,
the logged-π0 conditional redraw of one family's modes (other families fixed), B = 9999, BY at q = 0.05 over the
tested hypotheses, and the support rule (≥ 30 units, ≥ 5 accepted, ≥ 5 rejected, ≥ 3 episodes).

Changed:
1. **Dose.** level(accept) = level(half) = level(accept+rb) = 1, level(reject) = 0.
2. **Design-centred treatment.** v_u = sgn_u × (level(mode_u) − E_π0,u[level]), with the expectation over the unit's
   logged π0 table. Only the randomized part of the assignment enters.
3. **Residualised outcome.** Per family and hypothesis, r = the OLS residual of y(rel, kpi) on episode × sgn fixed
   effects plus the same target's pre-window value. r uses only pre-assignment quantities, so it is fixed under the
   sharp null and the test stays exact.
4. **Statistic.** T = |Σ_u v_u r_u|, with p = (1 + #{T_b ≥ T_obs}) / (B + 1) over logged-π0 redraws. RNG
   `default_rng([seed, 6616, 2, family_idx, lag, split])`, analysis seed 0.
5. **Sign and effect.** β = Σ v r / Σ v². sign = sign(β). Descriptive z_approx = S_obs / sd(S under the null
   draws), with S = Σ v r signed.
6. **Sample size.** EVAL v2 has 480 episodes (section 3). The DEV power analysis projects sleep → nbr pv at an
   expected z of about 1.4-1.7 per 60 episodes, so 480 episodes give an expected z of about 4-4.8.

The v1 statistic S* is not used for declarations in v2. It may be reported descriptively.

## 3. Data (fresh; all in the registered block 180000-183999, P3 surge-L40, same plant, π0 and driver as v1)

| stage (driver) | seeds | records | use |
|---|---|---|---|
| eval_v2 | 183460-183519 + 183560-183979 (480 eps) | stage "eval", sub "v2", fold = j // 120 (4 folds) | the discovery test |
| gt_ext | 183520-183539 (20 eps) | stage "gt", sub "ext" | fresh knockout ground truth |
| placebo_ext | 183540-183559 (20 eps) | stage "placebo", sub "ext" | fresh K0 |
| dev (v1) | 183300-183319 | stage "dev" | baseline τ tuning, as v1 |

Collection code: `scratchpad/e6_dev/e6p_discovery.py` at d453caf / f654dac (stages added; the episode code is
unchanged from v1).
- Runs: Kaggle e6p-disc-gtx-3 (3 kernels, gt_ext) and e6p-disc-ev2-2 (2 kernels, eval_v2), plus Colab
  e6p-disc-plx-c2 (placebo_ext).
- Earlier attempts (gtx-1/-2, ev2-1, plx-c1) produced no records: the cloud bundle mis-handled a repo-relative
  script path (fixed in b7d5326).
- The gtx-3 / ev2-2 bundle manifests say e6_dirty = true. The only cause is the then-untracked analysis module
  `cdd_oran/decision/crt_units_v2.py`, which the collection code never imports.
- Kaggle and Colab records can be pooled. Local Windows records are never used for EVAL, GT or PLACEBO.

**Analysis run.** The analysis may run in two passes on any machine:
1. `--no-baselines` gives K0, K1, G and P1v2;
2. the full run adds the baselines for P2v2.

The verdict uses the full run.

## 4. Criteria and verdict (`scratchpad/e6_dev/e6p_disc_analyze_v2.py`)

| criterion | rule | data |
|---|---|---|
| K0 validity | v1 rule on the MSCR-CRT v2 statistic: P(Binom(m, 0.05) ≥ n_reject) ≥ 0.01 AND ≤ 1 BY declaration | placebo_ext |
| K1 support | ≥ 60 sleep units AND ≥ 15 logged reject (as v1) | eval_v2 |
| G premise | fresh GT (dir, frozen v1 GT rule) sleep → nbr pv TRUE(+) | gt_ext |
| **P1v2** | see below | eval_v2 pooled vs gt_ext |
| P2v2 | MSCR-CRT v2 indirect (nbr) F1 ≥ the best baseline's (v1 baseline code and DEV-τ rule, unchanged), pooled AND in ≥ 3 of 4 folds | eval_v2 |

**P1v2.** The chain set is C = {sleep → nbr load (+), sleep → nbr pv (+), sleep → nbr v (+), ptx → nbr load (−)}.
These are the plan's a-priori expected indirect edges, in "dir" signs. C* = the members of C that are TRUE with that
sign in the fresh GT. P1v2 passes iff all of these hold:
- sleep → nbr pv is declared with sign +1 (G already requires it to be GT-TRUE(+));
- at least min(3, |C*|) members of C* are declared with the correct sign;
- overall precision over GT TRUE ∪ NULL is ≥ 0.80;
- overall sign accuracy is ≥ 0.90.

Far declarations are reported but are no longer a criterion.

**Baseline caveat (disclosed).** The baselines keep the v1 code and DEV far-FPR τ rule unchanged, for
comparability. Because sleep → far is TRUE, that rule sets τ somewhat higher than a sharp-null rule for corr, INT
and QACM (agent D: corr .129 vs .120, INT .251 vs .185, QACM .262 vs .108), which is conservative for them. The
physics-τ sensitivity scores and the placebo_ext calibration column are reported next to P2v2. A P2v2 PASS is
stated with this caveat.

**Verdict precedence** (as v1): INVALID > NO-CHAIN > UNDERPOWERED > PASS (P1v2 ∧ P2v2) / PARTIAL (P1v2 only) / KILL.

**Consequences.**
- PASS or PARTIAL: step 2 may use the MSCR v2 map. With PARTIAL, step 3 must include the SHAP-map arm.
- KILL: randomized unit-level logs cannot recover the chain at this budget. Step 2 then uses the paired-knockout
  (digital-twin) instrument or physics, as a declared privilege.

**Descriptive only:**
- recall over all GT-TRUE nbr edges;
- per-hypothesis β, approximate z and p;
- each baseline's declarations on placebo_ext with its DEV τ (a calibration column);
- the v1 MSCR-CRT on the same data (`--with-v1`).

## 5. Claims wording

- **Declared.** "Sharp null of no assigned-mode effect rejected (mscr-crt-units-v2, design-centred score); BY across
  the tested hypotheses."
- **Not detected.** This never means "no edge".
- **Scope.** Any positive result is a re-test after a disclosed amendment, on fresh seeds. It is not a confirmation of
  v1.
