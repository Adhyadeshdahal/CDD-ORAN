# Plan 007: E2 nonlinear label-free discovery under a fresh freeze

> **Executor instructions**: Use your own worktree/branch, read this fully, run every gate,
> and STOP rather than silently changing a preregistered scientific contract. The freeze
> ordering below is the anti-p-hacking guarantee — do not run the method against E2 truth until
> the protocol doc is committed and its SHA is recorded in the artifacts.
>
> **STATUS: PROTOCOL FROZEN + IMPL COMPLETE; RUN = TODO (parked, ready).** Protocol frozen at
> `protocol_commit = 828e345`. Implementation banked on `feat/v2 @ c0101e6` (`cdd_oran/e2slice/` +
> scripts + tests; 440 tests green; adversarial review verdict GENUINE; firewall verified twice).
> The frozen run has NOT been executed: at the frozen §14 constants it is memory-bandwidth-bound and
> takes ~5–18 days on the current 16 GiB/~3 GiB-free machine (the smoke extrapolation missed a CPU
> cache cliff). **The run is a standing TODO — run it whenever a high-RAM/high-bandwidth machine is
> free (see "## RUN STATUS" below).** Nothing about E2 truth has been examined; the freeze is intact.

## Status

- **Priority**: P1 (next discovery frontier after E1 green)
- **Effort**: L
- **Risk**: HIGH (scientific-integrity risk — anti-p-hacking; PLUS a claim-scope risk, see the firewall)
- **Depends on**: Plan 006 (E1 discovery-method prerequisite cleared), `E2V2Env` (committed `8d70a4f`)
- **Category**: scientific method + nonlinear causal discovery
- **Registered at**: `e290255` (draft), amended (this commit), 2026-09-04

## What this plan is — and is NOT (read first)

Two separate things must never be conflated:

1. **E1 cleared the discovery-method PREREQUISITE.** The label-free pipeline recovers structure
   when structure is recoverable (linear noiseless control). That is a statement about the
   *method*, not about any environment's decision-value hypothesis. Plan 007 **extends** that
   prerequisite to nonlinear, non-monotonic mechanisms. Clearing it is still only a method
   result.

2. **E2's preregistered DECISION gate is a recorded NULL and STAYS one.** `GATE_CONTRACT_E2.md`
   tested a shared-control decision-gap; the recorded result is **FAIL** (mean positive
   `gap_norm = 0.016203` vs `tau_E2 = 0.10`; separation `0.0162` vs required `>= 0.09`; committed
   `8d70a4f`, adversarially reviewed as a REAL null, encoded `xfail(strict=True)`). Plan 007 does
   **not** touch, reopen, improve, or re-tune that gate. **E2 discovery is
   methodological/exploratory — it is NOT evidence for the E2 decision-value hypothesis.** A green
   E2 discovery would extend the discovery pipeline's validated operating range to nonlinear
   mechanisms; it says nothing about whether E2's decision gap is real. A *failed* E2 discovery
   extension means "this method did not establish nonlinear recovery here," NOT "the E2 decision
   hypothesis is disproven." That firewall is a STOP condition below.

## Root cause: why E1's partial-correlation rule must NOT be carried over (a-priori, structural)

Every E2 KPI is a Gaussian bump (`cdd_oran/envs/v2/e2.py`, Env II verbatim), schematically:

```
K_j = A_j * exp( -( sum(shift_parents) - c_j )^2 / ( 2 * safe_exp(width_parent_j)^2 ) )
```

Two structural properties defeat any linear residual score (partial correlation, OLS |beta|,
Pearson/Spearman):

- **Non-monotonic, near-even dependence in the shift parents.** The bump rises to a peak and
  falls; where a parent's support straddles the center `c_j`, the induced dependence is
  approximately symmetric, so **linear and rank correlation are ~0 while the functional
  dependence is total.** Example: `K0 = 80*exp(-P0^2 / (2*safe_exp(P1)^2))` depends on `P0`
  through `P0^2` (even) — partial correlation of `P0->K0` is ~0.
- **Width parents enter the denominator (scale channel).** `P1, P3, P4, P6` modulate bump width,
  not the conditional mean linearly. `P1` is a true parent of `K0,K1,K3,K4` largely through this
  channel; a linear score sees almost nothing.

Therefore the E2 edge score must be a **general statistical dependence measure** (sensitive to
arbitrary non-monotonic association), not a linear one. This is the a-priori justification,
written in structural terms only — no reference to any recovered edge or magnitude.

## Frozen method (decisions confirmed 2026-09-04; exact spec pinned in the protocol doc)

1. **Edge score = magnitude of the U-centered partial distance correlation, `|pdCor|`.**
   - Estimator: the **U-centered (unbiased) estimator of squared distance covariance**
     (Szekely-Rizzo 2013/2014); partial distance correlation via the projection construction of
     Szekely-Rizzo 2014 on the U-centered distance matrices. The sample partial statistic is
     **signed and may be negative**; the **edge score is `abs(pdCor)`** and the **signed value is
     retained as a diagnostic** (reported, never the selector).
   - **This is a nonlinear conditional-ASSOCIATION score, NOT a conditional-independence
     certificate.** Ordinary distance correlation's "zero iff independence" does **not** transfer
     to the partial statistic: `pdCor = 0` is *not* equivalent to conditional independence
     (Szekely-Rizzo 2014, explicit). The protocol states this limit.
   - **Not "bias-corrected dCor" (ambiguous) and not "parameter-free".** The unbiased object is
     squared distance *covariance*. "Bandwidth-free" removes a kernel bandwidth, but the
     **distance geometry is a frozen choice**: per-column standardization (z-score on the frozen
     dataset), Euclidean distance, distance exponent `alpha = 1`, and the explicit construction of
     the **13-dimensional conditioning vector** (the other 13 candidates, each standardized,
     stacked into one Euclidean space). All pinned in the protocol.
   - **Denominator guard.** When the normalization denominator (product of the U-centered
     self-distance variances) falls below a frozen epsilon, `pdCor` is undefined → report `NaN`,
     the edge is **not selected** (fail-closed), and the candidate is flagged. Degenerate-candidate
     guard, analogous to E1's collinearity STOP.

2. **Threshold = per-candidate permutation-null test, per-target, never pooled. (RESOLVED.)**
   - For each (candidate `i` -> target `j`): hold the conditioning vector fixed, permute the
     candidate column over `B_perm` frozen permutations, recompute `|pdCor|`, and form a one-sided
     null. Select the edge iff observed `|pdCor|` exceeds the null at a frozen level with
     **per-target multiplicity control (BH-FDR at a frozen `q`)**. `B_perm`, `q`, and the
     permutation RNG seed are frozen constants; **none is truth-informed** and the permutation
     uses no ground truth.
   - Chosen over a raw per-target largest-gap because the U-centered statistic scatters around 0
     (incl. negative) and true width-channel edges may be weak, so a gap on a noisy signed
     statistic is unstable. The per-target largest-gap value is retained as a **reported
     diagnostic** only.
   - The permutation tests association-given-the-fixed-conditioning-set — a frozen benchmark
     heuristic, consistent with the association (not CI-certificate) reading above.

3. **Candidate graph = temporal 14-wide layout. (RESOLVED; wording corrected.)**
   - input `X = [P0..P7, K0_t..K5_t]` (8 params + 6 **lagged** KPIs); target
     `Y = [K0_{t+1}..K5_{t+1}]` (6). **Score matrix shape `(6, 14)`.**
   - The 6 KPI candidates are the **lagged** `K_t`, NOT contemporaneous KPIs — there are **no
     same-time KPI->KPI edges**; direction comes from the frozen temporal ordering (`t -> t+1`),
     since dCor itself is nondirectional.
   - This yields **48 NCP->KPI candidates** (6x8) and **36 lagged KPI->KPI candidates** (6x6);
     ground truth has **16 true NCP->KPI edges and zero true KPI->KPI edges** — all 36 KPI->KPI
     candidates are true-negatives the method must reject. **Report the KPI->KPI false-positive
     count (out of 36) and rejection rate = 1 - FP/36** explicitly (per seed + aggregate).

4. **Full-conditioning caveat (causal-interpretation limit).** Conditioning each candidate on all
   13 others is an E1-compatible stress test, but it can **overcondition** or **open collider
   paths** when lagged-KPI candidates are descendants. The score is therefore a **benchmark
   heuristic**: it certifies neither conditional independence nor causal direction (direction is
   supplied only by the frozen temporal ordering). The protocol states this limit explicitly.

5. **Cross-check (reported, not the selector): stability selection over the seed envelope** — as
   E1. Per-edge selection frequency across seeds; coarse CI at B=10; never alters the persisted
   mask.

6. **Rejected as primary**: any linear score (root cause above); marginal/unconditional
   dependence (shared-parent confounding hallucinates lagged KPI->KPI edges); pooled thresholding
   (the `largest_gap` straddling trap). Diagnostics only, never altering the persisted mask.

## Honest difficulty stance (locked)

Unlike E1, E2 recovery is **not expected to be a clean 0/1 by construction.** Conditioning a
nonlinear dependence measure on 13 other candidates with finite samples is the genuine
methodological challenge — this is *why E2 is where discovery becomes non-trivial.* **A partial or
non-green E2 recovery is a VALID recorded boundary result, not a defect to tune away**, provided it
is reported as-run and does not trigger threshold changes, seed dropping, retraining, or reopening
the decision gate. We do NOT add noise, alter the E2 SCM, or restrict the candidate graph to
manufacture a cleaner number. E2 is noiseless (like E1); its difficulty comes from nonlinearity and
conditioning, not noise.

## Reporting format and numerical halt conditions (frozen)

**Reporting format** (written into the persisted artifacts + result doc):
- Per (target, candidate): **signed `pdCor`**, **`|pdCor|`** (edge score), permutation p-value,
  selected bit.
- Per target: applied threshold decision, selected parents, the BH-FDR `q` used.
- **Negative unbiased estimates** are reported as-is in the signed diagnostic and folded by
  `abs()` for the score — **never silently clipped** to 0.
- **Denominator-guard hits** (undefined `pdCor`) are reported as `NaN`, flagged, counted, and
  fail-closed to not-selected.
- Aggregate: precision / recall / F1 overall + `NCP->KPI`; **KPI->KPI FP count (of 36) and
  rejection rate**; per seed and across the 10-seed envelope; no seed dropped.

**Numerical halt conditions** (distinct from the scientific STOPs — these mean "the run is broken;
fix the code/geometry, do NOT report a result"):
- Score matrix contains `NaN`/`inf` **outside** the declared denominator-guard path → HALT.
- The denominator guard fires on more than a frozen fraction of candidates → HALT and diagnose the
  distance geometry / standardization (do not silently drop edges).
- The permutation null is degenerate (zero variance) for a selected candidate → HALT.

## The freeze manifest (all seven frozen BEFORE any E2 truth is examined)

`docs/benchmark/E2_DISCOVERY_PROTOCOL.md` must fix, in structural terms:

1. **Data generation** — observational dataset from `E2V2Env` **TRUE SCM (decoy OFF)**; exogenous
   params `P0..P7` sampled **mutually independently** over their registered ID ranges (so
   non-parents have exact-zero population dependence); `N` rows per seed (frozen); one-step KPI
   outputs **aligned across the actuation-latency split** (`X` at `t`, `Y = K_{t+1}`); noiseless.
   Generation code committed; dataset content-hashed.
2. **Candidate graph** — the temporal 14-wide `(6, 14)` layout (method item 3).
3. **Nonlinear dependence score** — U-centered partial distance correlation, `|pdCor|`, with the
   frozen distance geometry and denominator guard (method item 1). Fully specified, no free
   parameter.
4. **Label-free threshold** — per-candidate permutation null, per-target BH-FDR at frozen `q`,
   frozen `B_perm` and permutation RNG seed (method item 2). Truth-free.
5. **Recovery metrics + reporting** — precision / recall / F1 overall + `NCP->KPI` +
   `KPI->KPI`-rejection, per-target, plus the reporting format above; vs
   `E2V2Env().true_adj_matrix()`, read **only after** the mask is persisted and hashed.
6. **Seed envelope** — 10 seeds (0..9), `env_seed = weight_seed = r`, sampling_seed frozen;
   per-seed recovery reported, no seed dropped.
7. **STOP criteria** (scientific) + **numerical halt conditions** — see below and above.

## Scope

**In scope**: a new label-free E2 discovery module (`cdd_oran/e2slice/` or a shared discovery core
reused from `e1slice`); the fresh frozen protocol `docs/benchmark/E2_DISCOVERY_PROTOCOL.md`; new
tests; the frozen protocol commit; a run on the frozen dataset + across the 10-seed envelope; a
digest `docs/benchmark/plan007_recovery_digest/`; an as-run result doc
`docs/benchmark/E2_DISCOVERY_RESULT.md`; a dated delta report under `reports/` only after results
exist.

**Out of scope**:
- Any **training/planning arm** — no MLP arms, no oracle/dense/discovered MSE envelope. Discovery
  recovery only. (Training re-enters only if the project *intentionally registers a new E2
  benchmark*, a separate decision, not this plan.)
- The **decoy** device (`decoy_omit_p0_k5`, `P0->K5`) — it belongs to the decision gate; discovery
  runs against the true SCM.
- **Reopening / re-tuning the recorded E2 decision null** (`GATE_CONTRACT_E2.md`, commit `8d70a4f`).
- Changing the E2 SCM / truth; any `E2V2Env`/true-adjacency import inside discovery; E3-E5; legacy
  training; retuning after seeing a recovery number.

## Freeze ordering (mandatory)

1. Decisions confirmed (done). Author `docs/benchmark/E2_DISCOVERY_PROTOCOL.md` as the executable
   spec of the frozen method (structural terms only — no reference to any specific E2 edge or
   magnitude). Review, then **commit it. This commit's SHA is the freeze.**
2. Implement the discovery module + tests to record that SHA as `protocol_commit` in every
   artifact; discovery reads NO ground truth and does not import `E2V2Env`.
3. Generate the frozen dataset; run discovery once, then across the 10-seed envelope. Persist
   masks/artifacts (fail-closed, Plan 003 pattern).
4. Only after masks are persisted and hashed, score recovery vs `E2V2Env().true_adj_matrix()` into
   a separate record.

## Gates

```
uv run ruff check .
uv run ty check
uv run pytest
```
plus: the discovery artifact recomputes its own `content_hash` and refuses to load on any
parent-hash / shape / numeric / `protocol_commit` mismatch (Plan 003 fail-closed pattern).

## Done criteria

- [ ] `docs/benchmark/E2_DISCOVERY_PROTOCOL.md` frozen (committed); its SHA recorded in every
      produced artifact. Estimator, distance geometry, denominator guard, threshold rule, and
      reporting format all pinned as an executable spec.
- [ ] Selection rule justified on the declared mechanism structure alone; pre-registration
      committed before first run against truth.
- [ ] Method run on the frozen dataset AND across the 10-seed envelope; per-seed recovery + the
      KPI->KPI FP count / rejection rate reported; no seed dropped. Reproducible from a committed
      sweep driver + `docs/benchmark/plan007_recovery_digest/`.
- [ ] Gate outcome stated **honestly** — green OR partial/null, whichever it is, is a valid
      recorded boundary result. No tuning to hit specific edges.
- [ ] Result doc explicitly restates the firewall: E2 discovery is method validation, NOT E2
      decision-value evidence; the E2 decision null (`8d70a4f`) stands untouched.
- [ ] Full verification passes; artifacts fail-closed; adversarial review verdict recorded.

## RUN STATUS — TODO (parked, ready to execute on adequate hardware)

The frozen discovery run is banked and ready. **Orchestrator: run it whenever a machine (or a bigger
machine) is free; it may run as long as needed.** Runbook:

- **Entry point**: `scripts/e2_slice_recovery_sweep.py` (10-seed envelope driver), or the
  `scripts/e2_slice.py` `generate`/`discover`/`recover` subcommands for manual phase control.
- **Frozen constants (do NOT change)**: N=4000, B_perm=999, q=0.05, decoy OFF, noiseless, temporal
  (6,14) graph, seeds r=0..9, `protocol_commit 828e345`. The code REJECTS non-frozen configs.
- **Anti-p-hacking ordering (hard)**: run discovery for ALL 10 seeds and persist + content-hash every
  `discovery.json` mask FIRST; verify all 10 fail-closed-loadable; ONLY THEN run recovery scoring
  (`evaluate.py` §9 reads `true_adj_matrix()`).
- **Discipline**: preserve every replicate; on a §11 numerical HALT (≥5/84 guarded, NaN/inf outside
  guard, degenerate null on a selected candidate) RECORD and STOP — never silently drop/retune. Watch
  the guard rate + KPI→KPI rejection.
- **Resource profile**: ~2.3 GiB RAM per concurrent seed; the work is memory-BANDWIDTH-bound so it
  wants high aggregate bandwidth (many memory channels) and enough free RAM to avoid paging.
  Wall-clock: hours-to-~1-day on a high-RAM/high-bandwidth box; ~5–18 days on the current machine.
  Parallelism across seeds only helps if bandwidth (not just cores/RAM) scales.
- **Output**: recovery records + a committed digest `docs/benchmark/plan007_recovery_digest/`
  (summary.json + replicates.jsonl + run_provenance.json + README.md), bulk runs gitignored; then the
  as-run result doc `docs/benchmark/E2_DISCOVERY_RESULT.md` + delta report; then merge to `feat/v2`.
- **Pre-run polish (nice-to-have, truth-free)**: harden the firewall test
  `test_discovery_imports_no_env_truth` to also assert NO truth on the transitive path through
  `dataset.py` (adversarial review obs #3; currently checks only `discovery.py`'s direct imports —
  the transitive path was verified clean by hand).

## STOP conditions (scientific)

- The method would have to see ground truth (or a truth-informed threshold) to select edges.
- The score/threshold is chosen or adjusted to hit specific E2 edges, or revised after its
  recovery number is inspected.
- Recovery is inconsistent across seeds and the response is to drop/reweight seeds.
- **Any attempt to present E2 discovery as evidence for the E2 decision-value hypothesis, or to
  reopen / re-tune / "improve" the recorded E2 decision null.** (Claim-scope firewall.)
- Adding noise, the decoy, or any SCM change to manufacture a cleaner recovery.
- A training/planning arm is added without the project intentionally registering a *new* E2
  benchmark first.

(Numerical halt conditions — broken-run guards, not scientific nulls — are listed under "Reporting
format and numerical halt conditions" above.)
