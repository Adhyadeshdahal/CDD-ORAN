# Plan 008: E2 nonlinear label-free discovery via RCoT (a fresh, better-calibrated freeze)

> **Executor instructions**: Use your own worktree/branch, read this fully, run every gate, and STOP
> rather than silently changing a preregistered scientific contract. The freeze ordering below is the
> anti-p-hacking guarantee — do NOT run the method against E2 truth until the protocol doc is committed
> and its SHA is recorded in the artifacts.
>
> **STATUS: PROTOCOL FROZEN (human-approved, primary null `block_perm`); IMPL BUILT + ADVERSARIALLY
> REVIEWED + LOCKED; RUN = pending execution.**
> The frozen pre-registration is `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md` (this plan's sibling).
> The method is implemented and hardened in `cdd_oran/e2slice/discovery_rcot.py` (16 unit tests green;
> reads/scores NO truth); at the freeze its `PROTOCOL_COMMIT` is set to the protocol-doc commit SHA and
> the config is locked to the §11 constants. **Nothing about real-E2 RCoT recovery has been examined at
> the freeze.** This is a **FRESH** pre-registration at a clean pre-truth point.

## Status

- **Priority**: P1 (corrected discovery method for the E2 frontier; sibling to Plan 007)
- **Effort**: M (impl already built; remaining = freeze + run + score)
- **Risk**: HIGH (scientific-integrity / anti-p-hacking; PLUS the claim-scope firewall, see §"firewall")
- **Depends on**: Plan 007 (frozen pdCor slice + dataset + `evaluate.py` recovery machinery + candidate
  layout), `E2V2Env` (committed `8d70a4f`), the verified calibration study `runs/calib-study/`
- **Category**: scientific method + nonlinear causal discovery
- **Registered at**: 2026-09-06 (this doc); frozen by the introducing commit (its SHA = `protocol_commit`)

## Why RCoT (root cause, a-priori + structural)

Plan 007's frozen pdCor method scores `|pdCor|` and tests it with a **marginal permutation** of the
candidate column. For a lagged-KPI candidate that shares generating params with sibling lagged KPIs in
the conditioning set `Z`, a marginal permutation destroys the candidate|Z coupling, the null collapses
too tight, and a tiny statistic rejects — the mechanism behind the observed **~9× over-selection of
lagged KPI→KPI edges** (retired pdCor seed-0: **17 FP / 36** KPI→KPI). A verified calibration study on a
testbed byte-faithful to the real E2 KPI mechanism traced this to a marginal-permutation MISCALIBRATION.

**RCoT (Randomized Conditional Correlation Test)** is a genuine **conditional-independence** test: it
residualizes **both** the candidate and target on random Fourier features of `Z`, then tests the
residuals for dependence, so the candidate|Z coupling is preserved and the null is (near-)nominal and
**flat in n**. Full structural justification: `E2_RCOT_DISCOVERY_PROTOCOL.md` §3. The nonlinear-score
argument (Gaussian-bump mechanism defeats any linear score) carries over unchanged from Plan 007.

## What this plan is — and is NOT (the firewall)

Identical firewall to Plan 007, restated:

1. **E2 RCoT discovery is METHOD validation** — a corrected, better-calibrated discovery test. A green
   result extends the label-free pipeline's operating range to nonlinear mechanisms with a CI test. It
   says **nothing** about whether E2's decision gap is real.
2. **E2's preregistered DECISION gate is a recorded NULL and STAYS one** (`GATE_CONTRACT_E2.md`,
   committed `8d70a4f`, adversarially reviewed FAIL, `xfail(strict=True)`). This plan does **not** touch,
   reopen, improve, or re-tune it. Presenting E2 discovery as decision-value evidence, or reopening the
   decision null, is a STOP.

## Frozen method (transcribed from `discovery_rcot.py`; pinned in the protocol doc)

1. **Test = RCoT**, per (target KPI `j`, candidate `i`) pair: `cand_i ⟂ target_j | Z`, `Z` = the other
   13 candidates (identical conditioning to pdCor).
2. **Per-column z-standardization** (ddof=0) of all 14 candidate + 6 target columns up front — rebalances
   the joint multivariate-`Z` median-heuristic bandwidth across mixed-scale columns (params ±100 vs KPIs
   ±40); **NOT an affine no-op** (adversarial review corrected this; measured `Δp` up to 0.307; keeps/
   improves calibration).
3. **Numerics**: median-heuristic bandwidth; RFF `dxy = 5` (cand/target), `dz = 25` (Z); ridge `1e-6`;
   statistic `T = n * ||Cxy||_F^2`. `dz = 25` is deliberate — higher `dz` OVER-rejects at small n; it is
   NOT a tunable "fix".
4. **Null (FROZEN primary = `block_perm`)**: `block_perm` (block-conditional permutation, `block_perm_reps = 99`,
   `block_size = 25`) = EXACT-nominal, the LIVE selector; `analytic_hbe` (HBE weighted-χ²) = fast
   sensitivity variant, never the selector.
5. **Selection**: per-target BH-FDR, `m = 14`, `q = 0.05` (reuses the frozen `bh_fdr_reject`).
6. **Guard + HALT**: relative RFF residual-collapse guard → `p = 1`, never selected; guard-fraction HALT
   at `>= ceil(0.05*84) = 5` of 84 guarded (parity with pdCor §11).
7. **Data**: same frozen E2 dataset (N=4000, noiseless, decoy OFF), candidate shape `(6,14)` = 48
   NCP→KPI + 36 lagged KPI→KPI.

## Truth-free calibration evidence (pre-registered de-risking)

Measured on the SHIPPING standardized path against a synthetic faithful testbed (byte-identical to
`E2V2Env._update_kpis`, max abs diff `1.4e-14`); NO real-E2 truth touched. Full tables in the protocol
§12. Headlines:
- **FP@.05**: `block_perm` at/below nominal and flat in n (ANCHOR 0.047/0.040/0.033); `analytic_hbe`
  liberality grows with n (NULL_k0 0.078/0.130/0.164).
- **Power@.05 (n=1000)**: NCP→KPI parents ~0.97–0.99 for both; injected KPI→KPI-edge power is a robustness
  probe only (real E2 has ZERO true KPI→KPI edges).
- **Cost (n=4000, single-core)**: `analytic_hbe` 0.76 s/seed; `block_perm` **194.6 s/seed (~3.2 min)** →
  **~32 min all 10 seeds**. Both trivial vs pdCor's ~8 h/seed.
- Module self-checks: analytic FP 0.065, block_perm FP 0.037, power 0.99; 16 unit tests pass.

## Human decisions (RESOLVED at freeze)

- **Primary `null_method`** — **`block_perm`** (EXACT-nominal, flat in n, ~32 min/10 seeds), human-approved
  2026-09-06 as the LIVE selector; `analytic_hbe` retained as a labelled sensitivity variant.
- **The freeze itself** — protocol doc committed (its SHA = `protocol_commit`), `PROTOCOL_COMMIT` set in
  `discovery_rcot.py`, config locked. See §"Freeze ordering".

## Freeze ordering (mandatory)

1. Commit `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md` with the primary `null_method` pinned to
   `block_perm`. **This commit's SHA is the freeze.**
2. Set `PROTOCOL_COMMIT = "<that sha>"` in `discovery_rcot.py`; every `discovery_rcot.json` records it and
   `frozen: true`. Lock `RCoTDiscoveryConfig` via `frozen_config()` + `load_discovery_rcot` re-derivation so
   any non-frozen config or `protocol_commit` mismatch is rejected, mirroring `discovery.py`. Commit the code.
3. Generate/reuse the frozen dataset; run discovery for ALL 10 seeds; persist + content-hash every
   `discovery_rcot.json` mask FIRST; verify all 10 fail-closed-loadable.
4. Only after masks are persisted and hashed, score recovery vs `E2V2Env().true_adj_matrix()` into a
   separate record (§9 of the protocol).

## Scope

**In scope**: the fresh frozen protocol `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md`; the freeze commit
+ `PROTOCOL_COMMIT` + config lock in `discovery_rcot.py`; a run on the frozen dataset across the 10-seed
envelope; recovery scoring via the existing `evaluate.py`; an as-run result doc + dated delta report
after results exist.

**Out of scope**: any training/planning arm; the decoy device; reopening/re-tuning the E2 decision null
(`8d70a4f`); changing the E2 SCM/truth; any `E2V2Env`/true-adjacency import inside discovery; retuning
`dz` or any constant after seeing a recovery number; touching the retired pdCor method on disk.

## Gates

```
uv run ruff check .
uv run ty check
uv run pytest
```
plus: the RCoT discovery artifact recomputes its own `content_hash` and (post-freeze) refuses to load on
any content-hash / shape / constant / `protocol_commit` mismatch; guarded cells never selected.

## RUN STATUS — pending execution (freeze done; run is trivial, ~32 min)

- **Freeze**: DONE — protocol committed (SHA → `protocol_commit`), `PROTOCOL_COMMIT` set + config locked,
  primary `null_method = block_perm` human-approved. The run may now proceed; masks persist BEFORE truth.
- **Entry point (post-freeze)**: reuse `scripts/e2_slice_recovery_sweep.py`-style driver /
  `cdd_oran/e2slice/discovery_rcot.py::write_discovery_rcot` per seed, then `evaluate.py` recovery.
- **Frozen constants (do NOT change)**: N=4000, decoy OFF, noiseless, `(6,14)` graph, `dz=25`, `dxy=5`,
  `ridge=1e-6`, `q=0.05`, `residual_epsilon=1e-12`, guard HALT `>=5/84`, seeds r=0..9; primary
  `null_method = block_perm` (human-approved), `block_perm_reps=99`, `block_size=25`.
- **Anti-p-hacking ordering (hard)**: persist + content-hash all 10 `discovery_rcot.json` masks FIRST;
  verify fail-closed-loadable; ONLY THEN run recovery scoring.
- **Resource profile**: single-core `block_perm` ~3.2 min/seed → **~32 min for all 10 seeds**;
  `analytic_hbe` ~7.6 s total. Trivial vs pdCor's ~8 h/seed — this box runs it comfortably.
- **Discipline**: preserve every replicate; on a guard-fraction HALT (≥5/84) or NaN/inf outside the guard
  path, RECORD and STOP — never silently drop/retune. Watch guard rate + KPI→KPI rejection.
- **Output**: recovery records + a committed digest + as-run result doc + delta report; compare vs the
  retired pdCor baseline (seed-0 P 0.929 / R 0.813 / F1 0.867, KPI→KPI 17 FP/36, 8.65 h).

## Done criteria

- [ ] `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md` frozen (committed); SHA recorded as `PROTOCOL_COMMIT`
      in every artifact; config locked.
- [ ] Primary `null_method` decided by the human and pinned at the freeze.
- [ ] Method run on all 10 seeds; masks persisted + hashed BEFORE any truth read; per-seed recovery + the
      KPI→KPI FP count / rejection rate reported; no seed dropped.
- [ ] Gate outcome stated honestly — green OR partial/null is a valid recorded boundary result. No tuning
      to hit specific edges; `dz` not raised as a "fix".
- [ ] Result doc restates the firewall: E2 discovery is method validation, NOT E2 decision-value evidence;
      the E2 decision null (`8d70a4f`) stands untouched; pdCor retained as retired baseline.
- [ ] Full verification passes; artifacts fail-closed; adversarial review verdict recorded (impl already
      reviewed GENUINE).

## STOP conditions (scientific)

- The method would have to see ground truth (or a truth-informed threshold) to select edges.
- The score/null/threshold (incl. `dz`) is chosen or adjusted to hit specific E2 edges, or revised after a
  recovery number is inspected.
- Recovery is inconsistent across seeds and the response is to drop/reweight seeds.
- Any attempt to present E2 discovery as evidence for the E2 decision-value hypothesis, or to reopen /
  re-tune / "improve" the recorded E2 decision null.
- Adding noise, the decoy, or any SCM change to manufacture a cleaner recovery.
- A training/planning arm is added without the project intentionally registering a *new* E2 benchmark first.

(Numerical halt conditions — broken-run guards, not scientific nulls — are the guard-fraction HALT ≥5/84
and NaN/inf outside the guard path; see the protocol §15.)
