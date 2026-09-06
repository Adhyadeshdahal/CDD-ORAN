# FROZEN pre-registration — E2 discovery via RCoT-v2 (block_perm, `block_perm_reps = 299`)

> **STATUS: FROZEN.** This is the frozen pre-registration for RCoT-**v2**, human-approved. **Committing
> this document is the freeze point:** its commit SHA is recorded as `protocol_commit` inside every
> `discovery_rcot_v2.json` artifact, and `PROTOCOL_COMMIT_V2` in `cdd_oran/e2slice/discovery_rcot_v2.py` is
> set to that SHA (config re-derived fail-closed on load). From this commit forward the anti-p-hacking
> ordering is in force: no constant may change after any E2 result is inspected.
>
> **What v2 changes vs v1 (the whole story): exactly ONE constant — `block_perm_reps` 99 → 299.** Nothing
> else. Every other §11 constant, the RCoT numerics, the null, the guard, the BH-FDR selection, the
> conditioning set, the data, and the firewall are **identical to v1** and are *imported/reused* from
> `discovery_rcot.py`, not re-implemented.

This contract inherits the v1 freeze discipline verbatim: **frozen before it is executed.** Its commit
SHA (set at freeze) is recorded as `protocol_commit` inside every `discovery_rcot_v2.json` artifact, and
the executable `discovery_rcot_v2.py` is locked to the §11 constants (its `PROTOCOL_COMMIT_V2` = this
document's SHA, config re-derived fail-closed on load). E2 ground truth is read **only after** the v2 mask
is persisted and hashed. Nothing here may be revised to improve a recovery number after any v2 recovery
result is inspected. A partial or null recovery is a valid recorded result.

## Revision history

- **2026-09-06, initial FROZEN version (authored pre-execution; human-approved).** First authored version
  of the E2 **RCoT-v2** contract, frozen by the commit that introduces it. It **supersedes** `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md`
  (`protocol_commit eba381a`, the v1 RCoT freeze) for the sole reason stated in §3: v1's
  `block_perm_reps = 99` carries a **provable-a-priori BH × permutation-resolution bug** that discards
  true edges the test already detects. v2 changes only `block_perm_reps` (99 → 299) and is otherwise a
  byte-for-byte reuse of the v1 method. v1's frozen result **stands as the recorded first-freeze outcome**
  (§14); v2 is a fresh, separately pre-registered iteration, not a re-tuning of v1.

---

## 1. Freeze preamble (inherited from v1)

- **FROZEN.** This contract is frozen by the commit that introduces it; that commit's SHA is the
  freeze point, recorded as `protocol_commit` in every produced artifact. At the same freeze
  `cdd_oran/e2slice/discovery_rcot_v2.py` is locked to the §11 constants (`PROTOCOL_COMMIT_V2` set to this
  SHA; config re-derived fail-closed on load) with the primary null pinned to
  `block_perm` and `block_perm_reps = 299`.
- Discovery reads **no environment truth** and imports no true-adjacency symbol. Dimensions come from the
  persisted row shapes. (The v2 module is a thin sibling of `discovery_rcot.py` and inherits this: it
  conditions on `np.delete(xs, i, axis=1)` and touches no truth.)
- E2 ground truth is read only after the v2 mask is persisted and hashed. Recovery scoring lives in a
  separate record (`recovery_rcot_v2.json`, §9) that can never feed back into selection.
- Imperfect recovery is a recorded result, not a defect to tune away.
- **All numerical constants in §11 become FROZEN at the freeze** and MUST NOT change after any E2 result
  is inspected.

## 2. What this plan is — and is NOT (the firewall, unchanged from v1)

The v1 firewall carries over **unchanged and in force**:

1. **E2 RCoT discovery is METHOD validation** — a corrected, better-calibrated discovery test for the
   nonlinear E2 mechanism class. A green v2 discovery would extend the label-free pipeline's validated
   operating range to nonlinear mechanisms with a conditional-independence test. It says **nothing** about
   whether E2's decision gap is real.
2. **E2's preregistered DECISION gate is a recorded NULL and STAYS one** (`GATE_CONTRACT_E2.md`, committed
   `8d70a4f`, adversarially reviewed **FAIL**, `xfail(strict=True)`). This protocol does **not** touch,
   reopen, improve, or re-tune that gate. E2 discovery is methodological/exploratory — **NOT** evidence
   for the E2 decision-value hypothesis. Presenting v2 recovery as decision-value evidence, or
   reopening/re-tuning the recorded E2 decision null, is a STOP (§15).

## 3. Why v1 is superseded — the BH × permutation-resolution bug (provable a priori)

v1 (`eba381a`, `block_perm_reps = 99`) fixed the lagged KPI→KPI over-selection but recovered only ~2–3 of
the 16 true NCP→KPI edges per seed (seed-mean recall 0.163, F1 0.251). The diagnosis
(`reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md`) established that this is **not** low test power and
**not** over-conditioning, but a purely arithmetical **selection artifact**:

- A block-conditional permutation p-value with `B` permutations has a hard **floor** of `1/(B+1)`.
  At `B = 99` the floor is `1/100 = 0.010`.
- Per-target Benjamini–Hochberg FDR at `q = 0.05` with `m = 14` candidates has a **leading threshold**
  (the smallest p-value it can accept) of `q/m = 0.05/14 = 0.00357`.
- **`0.010 > 0.00357`**, so at `B = 99` the strongest true-edge p-value is **structurally unreachable** by
  BH: even a true edge the conditional test detects (raw `p < 0.05` for ~80% of true edges; minimum
  achievable p pinned at exactly 0.010 on all 10 frozen seeds) is **discarded at the selection step**. The
  edge can only be selected on a lucky ≥3-way tie at the floor that lifts the BH staircase — hence the
  collapse to ~2–3 edges.

**The fix is arithmetic and a-priori: lower the floor below the BH leading threshold by raising `B`.**
At `B = 299` the floor is `1/300 = 0.00333`, and **`0.00333 < 0.00357`**, so the strongest true edge
becomes selectable **on its own**. This is derived from the fixed floor formula and the fixed BH threshold
— no E2 recovery number is used to choose `299`; the value is the smallest round `B` whose floor clears
the (fixed) BH threshold. (`B = 199` gives floor `0.005 > 0.00357`, which does **not** clear it per the
formula, yet already recovers most power empirically because BH accepts more than just its leading
candidate; `B = 299` is the clean choice whose *leading* threshold is satisfied.)

This changes **only the permutation count**, not the test statistic, the null construction, the
conditioning set, or the selection rule. Everything structural about why RCoT (not a linear score, not a
marginal-permutation null) is the right method carries over unchanged from v1 §3 and is not restated here.

## 4. Data generation (frozen — identical to v1 §4)

Unchanged from v1: the **same frozen E2 dataset** (E2V2Env TRUE SCM, **decoy OFF**, `obs_noise_scale = 0`,
`N_rows_per_seed = 4000`, eight exogenous params sampled mutually independently over their registered ID
ranges, input `X = [P0..P7(t) | K0..K5(t)]` 14 columns, target `Y = [K0..K5(t+1)]` 6 columns). The dataset
content-hash binds the v2 artifact fail-closed.

## 5. Candidate graph (frozen — identical to v1 §5)

Unchanged from v1: score-matrix shape **`(6, 14)`** = 48 NCP→KPI + 36 lagged KPI→KPI = 84 candidates per
seed; direction from the frozen temporal ordering `t → t+1`; ground truth has **0 true KPI→KPI edges**
(all 36 are true-negatives the method must reject). Read only post-persistence for recovery (§9).

## 6. Edge test — RCoT (frozen — **identical to v1 §6**, imported not re-implemented)

**Every RCoT numeric is reused verbatim from `cdd_oran/e2slice/discovery_rcot.py`** — the per-column
z-standardization (§6.1 of v1, a deliberate non-affine-no-op preconditioning of the joint-`Z` bandwidth),
the median-heuristic bandwidth, the RFF construction (`dxy = 5`, `dz = 25`), the ridge residualization
(`ridge = 1e-6`), the statistic `T = n·‖Cxy‖²_F`, and the relative residual-collapse guard
(`residual_epsilon = 1e-12`). `dz = 25` remains a **locked constant, NOT a tunable fix** (higher `dz`
over-rejects at small n). The v2 module imports all of this; it re-implements none of it.

## 7. Null variant and selection (frozen)

- **Primary null = `block_perm`** (block-conditional permutation), **identical construction to v1 §7.1**:
  greedy nearest-neighbour `Z` tour split into contiguous blocks of `block_size = 25`, candidate permuted
  **within** blocks, RCoT statistic recomputed with a **FIXED RFF basis**, p-value
  `(1 + #{stat_perm ≥ stat_0}) / (block_perm_reps + 1)`. **The ONLY change vs v1: `block_perm_reps`
  99 → 299.** `analytic_hbe` is retained as a fast, clearly-labelled **sensitivity** variant only, **never
  the selector** — and is explicitly **NOT adopted** here because its false-positive rate **grows with n**
  (diagnosis §B3: KPI→KPI FP 0.037 → 0.120 as n: 1000 → 4000), re-opening the over-selection v1 closed.
- **Selection = per-target BH-FDR at `q = 0.05`, `m = 14`** (reusing the exact frozen `bh_fdr_reject`),
  **identical to v1 §7.2**. Guarded cells forced to 0. Persisted `binary_mask` is the row-wise stack of BH
  decisions, `(6, 14)`.
- **Guard-fraction HALT** at `≥ ceil(0.05·84) = 5` of 84 guarded candidates — **identical to v1**.

## 8. Persistence (before any ground truth is read)

Identical mechanism to v1 §8, with **distinct filenames** so v1 artifacts are never overwritten:

- The v2 discovery artifact is **`discovery_rcot_v2.json`** (never `discovery_rcot.json`), written
  atomically, recording `protocol_commit = PROTOCOL_COMMIT_V2` (set to this document's commit SHA at the
  freeze), `frozen: true`, `block_perm_reps: 299`, and every other field exactly as v1's record (`schema_version`,
  `dataset_hash`, `n_rows`, `seed`, standardization moments, `score_method`
  `"rcot_conditional_independence"`, `null_method` `"block_perm"`, `dz`, `dxy`, `ridge`, `q`, `block_size`,
  `residual_epsilon`, `rng_seed`, `statistic`, `pvalues`, `guarded_mask`, `binary_mask`,
  `candidate_shape`, `git_sha`/`git_dirty`, `content_hash`).
- `load_discovery_rcot_v2` re-derives **every** frozen §11 constant and the `protocol_commit` against the
  v2 frozen config fail-closed (so a v1 mask, `block_perm_reps = 99`/`eba381a`, will **not** load as v2 and
  vice versa), verifies the content hash, checks the `(6, 14)` shapes, and refuses any record where a
  guarded cell is selected.

## 9. Recovery scoring (post-persistence ONLY)

Identical machinery to v1 §9, into the **distinct** `recovery_rcot_v2.json` (never `recovery_rcot.json`):
map the `(6, 14)` mask into the `(14, 14)` graph, read `E2V2Env().true_adj_matrix()` for the first time,
and report precision/recall/F1 overall + NCP→KPI, the KPI→KPI FP count (of 36) and rejection rate
`1 − FP/36`, per-target metrics, and `protocol_commit = PROTOCOL_COMMIT_V2` stamped in. Reuses the shared
metric helpers from `evaluate_rcot.py` / the pdCor evaluator; **no new metric code, no free parameter.**

## 10. Reporting format (frozen — identical to v1 §10)

Per (target, candidate): RCoT `statistic`, p-value, BH decision, selected bit, guarded flag. Per target:
BH `q`, selected parents, guarded count. Aggregate: P/R/F1 overall + NCP→KPI, KPI→KPI FP count (of 36) and
rejection rate, per seed and across the 10-seed envelope, **no seed dropped**, `null_method` = `block_perm`
and `block_perm_reps = 299`.

## 11. Frozen constant table (ONLY `block_perm_reps` differs from v1)

| Constant                | v1 value            | **v2 value**            | Same as v1? | Section |
|-------------------------|---------------------|-------------------------|-------------|---------|
| `decoy`                 | OFF                 | OFF                     | ✅          | §4      |
| `N_rows_per_seed`       | `4000`              | `4000`                  | ✅          | §4      |
| `obs_noise_scale`       | `0.0`               | `0.0`                   | ✅          | §4      |
| candidate shape         | `(6, 14)` = 84      | `(6, 14)` = 84          | ✅          | §5      |
| conditioning set `Z`    | other 13 candidates | other 13 candidates     | ✅          | §6      |
| standardization         | per-column z, ddof=0| per-column z, ddof=0    | ✅          | §6      |
| bandwidth               | median-heuristic    | median-heuristic        | ✅          | §6      |
| `dxy`                   | `5`                 | `5`                     | ✅          | §6      |
| `dz`                    | `25`                | `25`                    | ✅          | §6      |
| `ridge`                 | `1e-6`              | `1e-6`                  | ✅          | §6      |
| statistic               | `T = n·‖Cxy‖²_F`    | `T = n·‖Cxy‖²_F`        | ✅          | §6      |
| `residual_epsilon`      | `1e-12`             | `1e-12`                 | ✅          | §6      |
| **`block_perm_reps`**   | **`99`**            | **`299`** ⬅ THE CHANGE  | ❌          | §7      |
| `block_size`            | `25`                | `25`                    | ✅          | §7      |
| BH-FDR `q`              | `0.05` (m=14)       | `0.05` (m=14)           | ✅          | §7      |
| `rng_seed`              | `0`                 | `0`                     | ✅          | §6/§7   |
| `max_guard_fraction`    | `0.05` → HALT ≥5/84 | `0.05` → HALT ≥5/84     | ✅          | §7      |
| seeds                   | `r = 0..9`          | `r = 0..9`              | ✅          | §13     |
| `sampling_seed`         | `0`                 | `0`                     | ✅          | §13     |
| `null_method`           | `block_perm`        | `block_perm`            | ✅          | §7      |

**Exactly one row (`block_perm_reps`) differs.** All constants are truth-free and are pinned before any
real-E2 RCoT-v2 recovery is examined. Arithmetic justification of the change: floor `1/(299+1) = 0.00333`
`<` BH leading threshold `0.05/14 = 0.00357` (v1's floor `1/100 = 0.010` `>` `0.00357`).

## 12. Truth-free evidence for the fix (pre-registered de-risking)

Two independent truth-free lines, both measured **before** any real-E2 v2 recovery, confirm the fix
restores power without re-opening the KPI→KPI over-selection. Real E2 has **ZERO** true KPI→KPI edges, so
the KPI→KPI numbers are the false-positive rate the fix must keep low.

**(a) Diagnosis B-sweep** (`reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md` §B2, byte-faithful testbed,
n=1000, Z_full): the block_perm permutation-resolution sweep, floor `1/(B+1)` vs BH threshold `0.00357`:

| `block_perm_reps` | floor  | raw power | **BH power** | **BH KPI→KPI FP** |
|-------------------|--------|-----------|--------------|-------------------|
| 99 (v1)           | 0.0100 | 0.781     | **0.086**    | 0.003             |
| 199               | 0.0050 | 0.789     | **0.695**    | 0.010             |
| **299 (v2)**      | 0.0033 | 0.797     | **0.742**    | 0.010             |
| 499               | 0.0020 | 0.797     | **0.734**    | 0.010             |

Raising 99 → 299 lifts BH power **0.086 → 0.742** while KPI→KPI FP stays flat at **~0.010** (an order of
magnitude below `analytic_hbe`'s n-growing FP). Diagnosis §B3 shows `analytic_hbe` is **not** a safe
alternative (KPI→KPI FP grows to 0.120 at n=4000) — hence v2 keeps the exact-nominal permutation null and
fixes only the resolution.

**(b) v2-code-path re-verification** (this task; the ACTUAL `discover_graph_rcot` + `frozen_config_v2`
shipping pipeline, not the diagnosis script; byte-faithful testbed `calib_study4_correct.kpi_fn`, n=1000):

Measured through the shipping v2 pipeline (`discover_graph_rcot(rows, frozen_config_v2())`, B=299) on the
byte-faithful testbed, n=1000, 8 reps, single-core: **BH-power on the 16 true param→KPI edges = `0.656`**
(84/128 selected across reps); **KPI→KPI FP rate = `0.003`** (1/288 across reps). Same regime as the
diagnosis B=299 row (BH-power ~0.742, FP ~0.010) and an order of magnitude above v1's collapsed BH-power
(`B=99` → 0.086) — the fix is confirmed through the ACTUAL v2 code path, KPI→KPI FP still block_perm-grade
(~0.01). The ~0.09 shortfall vs 0.742 is seed/rep variance at 8 reps plus the hard-to-detect
heteroscedastic width-param edges in the 16-edge denominator; it does not affect the direction of the fix.

## 13. Freeze / lock mechanism (identical to v1 §13, with v2 filenames)

The freeze executes in order:

1. **Commit this document.** Its git commit SHA is the freeze point.
2. **Record the SHA in code.** Set `PROTOCOL_COMMIT_V2 = "<this document's full sha>"` in
   `cdd_oran/e2slice/discovery_rcot_v2.py`, replacing the placeholder. Every `discovery_rcot_v2.json`
   records this `protocol_commit` and sets `frozen: true`.
3. **Lock the config.** `frozen_config_v2()` returns the §11 values with `block_perm_reps = 299`;
   `write_discovery_rcot_v2` uses it, and `load_discovery_rcot_v2` re-derives every §11 constant and the
   `protocol_commit` fail-closed. (Already implemented; the only freeze action is step 2's SHA.)

Committing the code in steps 2–3 (with `PROTOCOL_COMMIT_V2` set to the step-1 SHA) follows the document
commit, so the executable provably references a contract that predates it, and no run precedes the lock.

## 14. Retention of v1 (the superseded first freeze) and the retired pdCor baseline

- **RCoT-v1 (`eba381a`, `block_perm_reps = 99`) stays untouched on disk** — its module
  (`discovery_rcot.py`), its frozen artifacts (`discovery_rcot.json` / `recovery_rcot.json` in
  `runs/e2slice-recovery/replicate-*/`), and its protocol doc all remain valid and loadable. **v1's frozen
  result stands as the recorded first-freeze outcome** (seed-mean NCP→KPI recall 0.163, F1 0.251, KPI→KPI
  6 FP/360): a real, honestly-recorded result that motivated — but is not overwritten by — v2. v2 does not
  re-run, re-tune, or re-score v1.
- The **retired pdCor baseline** (`E2_DISCOVERY_PROTOCOL.md` / `discovery.py`, `protocol_commit 828e345`,
  seed-0 P 0.929 / R 0.813 / F1 0.867, KPI→KPI 17 FP/36, 8.65 h) likewise stays untouched as the recorded
  naive baseline.

## 15. Run + recovery-scoring plan (post-freeze) and STOP conditions

**Run plan** (post-freeze, ordering identical to v1 §15):

1. **Truth-free smoke** (tiny n, few reps, no truth) — throwaway; §11 constants never reduced.
2. **Discovery, all 10 seeds** (`r = 0..9`) at the frozen v2 constants (`block_perm`, `B = 299`). Cost:
   `B = 299` is ~3× the v1 `B = 99` block_perm compute (~3.2 min/seed at n=4000 → **~10 min/seed**;
   **~1.7 h for all 10 seeds** single-core). Persist + content-hash every `discovery_rcot_v2.json` mask
   FIRST; verify all 10 fail-closed-loadable.
3. **Numerical HALTs** (broken-run guards, not scientific nulls): NaN/inf outside the guard path → HALT;
   guard fires on ≥5/84 → HALT that seed. Record guarded counts; never silently drop edges.
4. **ONLY THEN** run recovery scoring (§9) vs `E2V2Env().true_adj_matrix()` into `recovery_rcot_v2.json`;
   report per-seed + aggregate P/R/F1 overall + NCP→KPI, KPI→KPI FP-count/rejection rate; **no seed
   dropped**. Compare against v1's recorded result (§14) and the retired pdCor baseline.
5. Write the as-run result doc + dated delta report **only after** results exist.

**STOP conditions (scientific — inherited from v1 §16):** the method would have to see ground truth to
select edges; any constant (incl. `dz` **or `block_perm_reps`**) is adjusted to hit specific E2 edges or
revised after a v2 recovery number is inspected; recovery is inconsistent across seeds and the response is
to drop/reweight seeds; presenting v2 discovery as E2 decision-value evidence, or reopening/re-tuning the
recorded E2 decision null (`8d70a4f`); adding noise/decoy/any SCM change to manufacture a cleaner
recovery; adding a training/planning arm without registering a new E2 benchmark first.

If a reviewer rejects this contract **before** its first execution, revise it and record why in the
Revision history. **After** first execution, do not revise it to improve recovery.
