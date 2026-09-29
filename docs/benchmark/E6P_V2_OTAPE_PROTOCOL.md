# E6-P v2 O_tape falsifier (E6P-v2)

Status: **FROZEN 2026-09-29.** The sha256 of this file (LF line endings) is recorded in `scratchpad/e6_dev/e6p_v2.py`
(`FROZEN_SHA256`). The driver refuses a non-smoke run when this file, `E6P_SCREEN_PROTOCOL.md`, `E6P_SPEC.md` or
`SEED_REGISTRY.json` differ from the recorded hashes. The user approved exactly one follow-up to the E6-P Gate A v2
screen: rerun the clairvoyant oracle on fresh seeds, but forbid it from breaking guardrails. This is that
follow-up. It is a DEV screen and cannot confirm an edge. Running stages 1 and 2 needs one explicit go from the user,
and stage 3 needs a second go.

## 0. Inheritance
Everything is inherited from `E6P_SCREEN_PROTOCOL.md` (sha256 `9705653c8e797af159bf3b74424b49d85818dd7dc38bf1727797142399be90d0`)
and `E6P_SPEC.md` (sha256 `5db7ef17b6a5fa3f53b49dac3e28e58fecca25190a2248f4cca6f34f6b4b21a9`), both frozen at commit
`2afbc1c8da381375891c438efea6500f7f5e5467`, as run by `scratchpad/e6_dev/e6p_screen.py` (E6P-v1). Only §1-§3 below
differ. The inherited items are:
- the plant and its code;
- §1 common configuration, the arms and their definitions (§5), and the oracle mechanics and constants;
- the energy admissibility bound, which stays **one-sided**, and the churn cap;
- the v1 stage-0a `load_factor` values (L10 0.7875, L40 1.75546875);
- the v1 stage-0b mechanism results (M1-M6 pass; P2 not screenable);
- the QACM fit seeds and mapping;
- all §7 quantities, criteria 1-4, thresholds, the paired-seed bootstrap (`default_rng([6611, pair_idx, stratum])`,
  10 000 resamples) and the verdict labels.

No stage 0 is rerun, because the plant and the calibration are unchanged.

## 1. Change (a): oracle guardrail admissibility
- **Privileged guardrail traces.** The per-episode accept-all (AA) reference run, which v1 already ran for the churn
  cap, records the per-second cumulative value from episode start of each guardrail key, on the same tape. It records
  them exactly as the freeze and A-alone runs record the energy traces `E_freeze(t)` and `E_A(t)`: index t = the
  value after t control seconds. The keys are:
  - `svr` = 3600 · `viol_ue_s` / `ue_s`, over the scored seconds so far;
  - `nonprot_embb_viol` = `embb_viol` − `prot_viol`;
  - `ll_viol`;
  - `rlf`. This is the plant's `sla["rlf"]` counter, which counts from t = 0, warm-up included. The warm-up is
    identical in every arm.
- **Rule.** A trial plan is admissible only if all of the following hold at the end of its H-second rollout (second t):
  - for every guardrail key k, the plan's cumulative value from episode start is ≤ 1.10 · AA_k(t). The factor 1.10 is
    the §7 guardrail margin (`E6P_SPEC.md` §5).
  - **Zero AA value:** if AA_k(t) = 0, the bound is 1 in k's own unit: 1 RLF, 1 UE-s, or 1 UE-s per UE-h for `svr`.
    This is the +1 slack.
  - the existing one-sided energy bound holds, unchanged;
  - the existing churn cap holds, unchanged.
  - The plan's value includes the oracle's own history before the decision point, exactly as for the energy bound.
- Everything else about the oracle is unchanged:
  - the search (stage 1 = accept-all, freeze, incumbent and n_glob random plans; stage 2 = one coordinate-descent pass);
  - the objective (protected violated UE-s, then all-UE violated UE-s, lexicographic);
  - the fallback: if no plan is admissible, accept-all is used.
- **Declaration.** This change moves the gate's own eligibility rule (§7: guardrails vs AA at margin 1.10) into the
  optimiser's admissibility test. It changes no threshold. Oracle eligibility is still judged by §7 on the pooled
  8-seed stratum sums. A per-episode, per-horizon constraint does not guarantee pooled eligibility, and an oracle that
  still fails §7 has `R_or = 0`, as in v1. The v2 oracle is called **O_tape**: it keeps full lookahead on the true tape.

## 2. Change (b): fresh seeds
- Block 155000-155399 in E6 DEV, registered in `SEED_REGISTRY.json` (E6 → `dev_reserved`) before any use.
- **Screen:** 155000-155199. **Confirmation (reserved, untouched here):** 155200-155399.
- **Mapping:** v1's `screen_seed` with base 155000, `seed = 155000 + 10·stratum + j`, j = 0..7 (N = 8 per stratum).
  The strata in scope are 1 and 3, so the seeds used are 155010-155017 and 155030-155037. The rest of the screen block
  stays unused.
- Paired arms reuse the episode seed, so every arm of a cell sees the same plant tape.

## 3. Change (c): scope
Only the three cells where v1 criterion 1 passed:
- P1 stratum 3 (surge-L40);
- P3 stratum 1 (base-L40);
- P3 stratum 3 (surge-L40).

The stages:
- **Stage 1 (fresh).** The shared freeze arm plus every stage-1 arm of the cell's pair (§5 arms 2-8: singles,
  accept-all, static subsets, per-knob priority, cell-priority lock, knob-lock and QACM), on the fresh seeds. It yields
  criterion 1 and `R_static` (arms 4-8, plus arm 9 when stage 3 runs).
- **Stage 2 (O_tape).** Arm 10 with change (a), in all three cells on the same fresh seeds, run **concurrently** with
  stage 1 to save wall time. It is not gated by criterion 1. This costs compute only: the criteria are conjunctive and
  c1 comes from stage 1 alone, which the oracle cannot influence. In a cell that fails the fresh c1, the oracle's
  `R_or` and `ρ_sign` are reported as descriptive only.
- **Stage 3.** Arm 9 (per-region hindsight static), only in cells that pass c1 and c2 on the fresh seeds.

## 4. Change (d): nothing in the decision rule
- Criteria 1-4, their thresholds (15 %, 36 UE-s per UE-h, 50 %, 10 pp, ρ_sign 0.70), the single-redraw `ρ_sign`
  definition and the verdict labels (PASS / NOISE STOP / DEAD (no loss) / DEAD (not recoverable) / NO-EDGE STOP) are
  **unchanged**. The verdict is given per pair over that pair's v2 cells.
- **Additionally reported, descriptive only (it can never change a verdict):**
  - the summed re-drawn advantage Σ (AA_redraw − plan_redraw) of protected violated UE-s over the H-second rollouts,
    summed over all oracle deviations in the cell, with a one-sided 90 % seed-bootstrap lower bound. The bootstrap uses
    10 000 resamples of the 8 seeds, the per-seed sums and `default_rng([6612, pair_idx, stratum])`; 6612 is a new tag.
  - the same sum on the true tape;
  - the oracle's per-key guardrail ratios vs AA;
  - how often trial plans were rejected by the guardrail bound.

## 5. Rule K4
There is **no v3 of E6-P under any outcome**. The v1 result (DEAD) and this v2 result are published together, each as
is.

## 6. Disclosure (forking paths)
This change was chosen **after** the v1 outcome was seen. In the three v1 c1 cells, the budgeted oracle raw-recovered
R = 0.38 / 0.88 / 0.65. It was ineligible because it broke guardrails: RLF 1.22-1.39 × AA and LL 1.107 × AA. Its
objective saw only (protected, all-UE) violations plus the one-sided energy bound. Its `ρ_sign` was 0.55 / 0.61 / 0.66,
all below the 0.70 criterion.

Choosing the fix and the cells after seeing outcomes is a forking-paths risk. It is mitigated by:
- fresh seeds;
- unchanged thresholds and criteria;
- a scope capped at the three cells, which must re-pass c1 on the fresh seeds;
- a single follow-up, with rule K4 forbidding any v3.

## 7. Explicitly NOT changed
- the energy rule (it stays one-sided);
- `RB_WINDOW` (60 s);
- H (90 s) and D (20 s);
- the search budget (n_glob 6, n_loc 2);
- the "half" slew rule and the churn cap;
- the plant (`E6P_SPEC.md`, `cdd_oran/envs/e6`);
- the QACM fit and the stage-1 arm set.

## 8. Implementation and checks
- **Driver:** `scratchpad/e6_dev/e6p_v2.py`, with the wrappers `e6p_v2_stage{1,2,3}.py`. It imports `e6p_screen.py` and
  overrides only the seed mapping, the unit lists, the oracle job (the AA trace run and the admissibility test) and the
  state files (`e6p_v2_state.json` plus its `.py` mirror; `load_factor` is copied from the v1 state).
- **Pre-run mechanism checks** (`e6p_v2.py selftest`; `tests/test_e6p_v2_oracle.py`):
  1. a trial plan whose rollout RLF exceeds the bound is rejected;
  2. with the guardrail constraint disabled, the v2 oracle reproduces v1 decisions exactly on a DEV smoke.
- **Numerics:** results are pooled only within one numeric platform class. Kaggle and Colab qualify when the
  `cloud.py` fingerprint trajectory matches.
