# Plan 007: E2 nonlinear label-free discovery under a fresh freeze

> **Executor instructions**: Use your own worktree/branch, read this fully, run every gate,
> and STOP rather than silently changing a preregistered scientific contract. The freeze
> ordering below is the anti-p-hacking guarantee — do not run the method against E2 truth until
> the protocol doc is committed and its SHA is recorded in the artifacts.
>
> **STATUS: DRAFT (registered, not frozen).** The method choices marked **[PROPOSED — confirm
> before freeze]** are the orchestrator's recommendation and are NOT yet locked. Nothing about
> E2 truth has been examined to write this plan; it is derived only from the declared E2
> generative mechanism. The freeze (committing `E2_DISCOVERY_PROTOCOL.md`) happens only after
> the marked decisions are confirmed.

## Status

- **Priority**: P1 (next discovery frontier after E1 green)
- **Effort**: L
- **Risk**: HIGH (scientific-integrity risk — anti-p-hacking; PLUS a claim-scope risk, see the firewall)
- **Depends on**: Plan 006 (E1 discovery-method prerequisite cleared), `E2V2Env` (committed `8d70a4f`)
- **Category**: scientific method + nonlinear causal discovery
- **Registered at**: (this commit), 2026-09-04

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
   mechanisms; it would say nothing about whether E2's decision gap is real. That firewall is a
   STOP condition below.

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

Therefore the E2 edge score must be a **general statistical dependence measure** (zero iff
independence; sensitive to arbitrary non-monotonic association), not a linear one. This is the
a-priori justification, written in structural terms only — no reference to any recovered edge
or magnitude.

## Frozen method (PROPOSED — confirm before freeze)

1. **Edge score = bias-corrected PARTIAL (conditional) distance correlation** per (target,
   candidate). **[PROPOSED — confirm before freeze]**
   - *Distance correlation* (Szekely-Rizzo-Bakirov 2007) is 0 **iff** independence and detects
     arbitrary non-monotonic/even dependence — it fixes the blind spot above. It is
     bandwidth-free (an advantage over HSIC, whose kernel bandwidth would itself need a frozen,
     structurally-justified choice).
   - The *partial* form (partial distance correlation, Szekely-Rizzo 2014) conditions on the
     other candidate parents. This is the **direct nonlinear generalization of E1's partial
     correlation** and is required to remove **shared-parent confounding**: two KPIs that share a
     param (e.g. `K0,K1` both use `P0,P1`) are marginally dependent, so a *marginal* score would
     hallucinate a `KPI->KPI` edge. Conditioning on the shared param removes that.
   - Use the **bias-corrected / unbiased dCov estimator** (dCor*, Szekely-Rizzo 2013): empirical
     distance correlation of independent variables has positive finite-sample bias, so a naive
     near-zero threshold is unsafe. The bias-corrected statistic centers non-edges near 0 (can go
     slightly negative), keeping the label-free gap honest.
2. **Threshold applied per-target (per-row), never pooled** — same structural reason as E1
   (heterogeneous targets have different score scales; pooling straddles them). Rule: a
   largest-gap analogue on each target's score vector with a frozen floor, OR a permutation-null
   calibration per candidate. **[PROPOSED — confirm before freeze; exact rule pinned in the
   protocol doc.]** The choice must be justified structurally and frozen before running.
3. **Candidate graph = full 14-wide `(K, P+K)` layout.** Each of the 6 KPI targets searches all 8
   params + all 6 other KPIs as candidate parents. `KPI->KPI` candidates are **included** so the
   method must actively *reject* them (E2's true graph is NCP->KPI only). This is the stronger
   test — restricting to NCP-only would make "no false KPI->KPI edges" trivially true. **[PROPOSED
   — confirm before freeze.]**
4. **Cross-check (reported, not the selector): stability selection over the seed envelope** —
   per-edge selection frequency across seeds, as a reported heuristic (coarse CI at B=10), never
   altering the persisted mask. Same stance as E1.
5. **Rejected as primary**: any linear score (partial correlation, |beta|, Pearson/Spearman) —
   fails by the root cause above; marginal (unconditional) dependence — confounded by shared
   parents; pooled thresholding — the `largest_gap` straddling trap. May appear only as
   clearly-labelled diagnostics that never alter the persisted mask.

## Honest difficulty stance (locked)

Unlike E1, E2 recovery is **not expected to be a clean 0/1 by construction.** Conditioning a
nonlinear dependence measure on ~13 other candidates with finite samples is the genuine
methodological challenge — this is *why E2 is where discovery becomes non-trivial.* **A partial
or non-green E2 recovery is a VALID recorded boundary result, not a defect to tune away.** We do
NOT add noise, alter the E2 SCM, or restrict the candidate graph to manufacture a cleaner number.
E2 is noiseless (like E1); its difficulty comes from nonlinearity and conditioning, not noise.

## The freeze manifest (all seven frozen BEFORE any E2 truth is examined)

The protocol doc `docs/benchmark/E2_DISCOVERY_PROTOCOL.md` must fix, in structural terms:

1. **Data generation** — observational dataset from `E2V2Env` **TRUE SCM (decoy OFF)**; exogenous
   params `P0..P7` sampled **mutually independently** over their registered ID ranges (so
   non-parents have exact-zero population dependence); `N` rows per seed (frozen constant); one-step
   KPI outputs aligned across the actuation-latency split; noiseless. Generation code committed;
   dataset content-hashed.
2. **Candidate graph** — the full 14-wide `(K, P+K)` layout (item 3 above).
3. **Nonlinear dependence score** — bias-corrected partial distance correlation (item 1 above);
   estimator fully specified, no free bandwidth.
4. **Label-free threshold** — per-target rule (item 2 above); frozen, truth-free.
5. **Recovery metrics** — precision / recall / F1 overall + `NCP->KPI` + `KPI->KPI`-rejection,
   plus per-target, vs `E2V2Env().true_adj_matrix()`, read **only after** the mask is persisted and
   hashed. Written to a separate `recovery.json`-style record.
6. **Seed envelope** — 10 seeds (0..9), `env_seed = weight_seed = r`, sampling_seed frozen;
   per-seed recovery reported, **no seed dropped.**
7. **STOP criteria** — see below.

## Scope

**In scope**: a new label-free E2 discovery module (`cdd_oran/e2slice/` or a shared discovery core
reused from `e1slice`); a fresh frozen protocol `docs/benchmark/E2_DISCOVERY_PROTOCOL.md`; new
tests; the frozen protocol commit; a run on the frozen dataset + across the 10-seed envelope; a
digest `docs/benchmark/plan007_recovery_digest/`; an as-run result doc
`docs/benchmark/E2_DISCOVERY_RESULT.md`; a dated delta report under `reports/` only after results
exist.

**Out of scope**:
- Any **training/planning arm** — no MLP arms, no oracle/dense/discovered MSE envelope. Discovery
  recovery only. (Training re-enters only if the project *intentionally registers a new E2
  benchmark*, which is a separate decision, not this plan.)
- The **decoy** device (`decoy_omit_p0_k5`, `P0->K5`) — it belongs to the decision gate, not
  discovery; discovery runs against the true SCM.
- **Reopening / re-tuning the recorded E2 decision null** (`GATE_CONTRACT_E2.md`, commit `8d70a4f`).
- Changing the E2 SCM / truth; any `E2V2Env`/true-adjacency import inside discovery; E3-E5; legacy
  training; retuning after seeing a recovery number.

## Freeze ordering (mandatory)

1. Confirm the **[PROPOSED]** decisions (orchestrator + user). Write
   `docs/benchmark/E2_DISCOVERY_PROTOCOL.md` capturing the frozen method in structural terms (no
   reference to any specific E2 edge or magnitude). Commit it. **This commit's SHA is the freeze.**
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

- [ ] `[PROPOSED]` decisions confirmed with the user; protocol frozen and committed; its SHA
      recorded in every produced artifact.
- [ ] Selection rule justified on the declared mechanism structure alone; pre-registration
      committed before first run against truth.
- [ ] Method run on the frozen dataset AND across the 10-seed envelope; per-seed recovery
      reported (overall / `NCP->KPI` / `KPI->KPI`-rejection); no seed dropped. Reproducible from a
      committed sweep driver + `docs/benchmark/plan007_recovery_digest/`.
- [ ] Gate outcome stated **honestly** — green OR partial/null, whichever it is, is a valid
      recorded boundary result. No tuning to hit specific edges.
- [ ] Result doc explicitly restates the firewall: E2 discovery is method validation, NOT E2
      decision-value evidence; the E2 decision null (`8d70a4f`) stands untouched.
- [ ] Full verification passes; artifacts fail-closed; adversarial review verdict recorded.

## STOP conditions

- The method would have to see ground truth (or a truth-informed threshold) to select edges.
- The score/threshold is chosen or adjusted to hit specific E2 edges, or revised after its
  recovery number is inspected.
- Recovery is inconsistent across seeds and the response is to drop/reweight seeds.
- **Any attempt to present E2 discovery as evidence for the E2 decision-value hypothesis, or to
  reopen / re-tune / "improve" the recorded E2 decision null.** (Claim-scope firewall.)
- Adding noise, the decoy, or any SCM change to manufacture a cleaner recovery.
- A training/planning arm is added without the project intentionally registering a *new* E2
  benchmark first.
