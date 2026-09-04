# Plan 006: KPI->KPI-capable E1 discovery under a fresh freeze

> **Executor instructions**: Use your own worktree/branch, read this fully, run every gate,
> and STOP rather than silently changing a preregistered scientific contract. The freeze
> ordering below is the anti-p-hacking guarantee — do not run the method against E1 truth until
> the protocol doc is committed and its SHA is recorded in the artifacts.

## Status

- **Priority**: P1 (this is the E1 gate; everything downstream is blocked on it)
- **Effort**: L
- **Risk**: MEDIUM (scientific-integrity risk — see the central constraint)
- **Depends on**: Plan 003 (frozen artifacts), Plan 004 (multi-seed envelope)
- **Category**: scientific method + causal discovery
- **Registered at**: `13036cb`, 2026-09-04

## Why this matters

The learned E1 discovery arm (Plan 003, frozen) recovers all four NCP->KPI edges but misses
both KPI->KPI edges (K2<-K0, K1<-K3). Per `docs/benchmark/E1_DISCOVERY_RESULT.md` the failure
is **not** identifiability: the true KPI->KPI standardized coefficients (0.439, 0.456) are
present and correctly placed. They fall **below** the single global `largest_gap` cut (~0.674),
which lands between the KPI->KPI band (~0.44) and the NCP->KPI band (~0.9). A method that
thresholds with awareness of that structure recovers them.

By the locked benchmark, E2-E5 decision results are uninterpretable until a discovery method
clears the E1 recovery gate. This plan is that method.

## THE CENTRAL CONSTRAINT (read before designing anything)

We have **already inspected** the coefficients (0.439, 0.456) in the frozen 003 result. Any
method designed now is informed by truth we have seen — exactly what the freeze discipline
exists to prevent. A method hand-tuned to the values 0.439/0.456 is p-hacking and the result is
worthless. The method is only defensible if all three hold:

1. **A-priori principle, not the numbers.** The selection rule is justified by the *structure*
   of the problem (standardization compresses multi-parent edges; heterogeneous parent counts),
   never by the specific values we saw. The justification is written in the frozen protocol in
   structural terms only.
2. **Frozen before running against truth**, exactly like Plan 003: the protocol commit SHA is
   recorded inside every artifact; ground truth is read only after the mask is persisted and
   hashed. No revision after seeing the recovery number.
3. **Validated across seeds**, not on the one vector we peeked at: clears the gate across the
   Plan 004 10-seed envelope (env_seed=weight_seed=r, split_seed=0); per-seed recovery reported,
   no seed dropped. Overfitting to one score vector is the primary failure mode.

If the method does not clear the gate honestly across seeds, that is a **valid recorded result**,
not a defect to tune away.

## Root cause (derivable a-priori, before any result)

The standardized coefficient is `beta = b * sd(x)/sd(y)`. For `K2 = P2 + 0.5*K0` with
unit-variance independent parents, `sd(K2) = sqrt(1 + 0.25) = 1.118`, so `beta_P2 = 0.894` and
`beta_K0 = 0.447`. This *predicts* the observed ~0.9 / ~0.44 from the SCM alone. A downstream
KPI's `sd(y)` is inflated by every parent, pushing all its incoming edge scores down together;
a global cut then straddles targets of different `sd(y)`/R^2 and buries weak-but-real edges.

## Frozen method (locked 2026-09-04, orchestrator + user)

1. **Edge score = per-target partial-correlation magnitude** (equivalently partial R^2 /
   squared semi-partial). Partial correlation residualizes both feature and target on the other
   candidate parents, normalizing by *residual* variance instead of total `sd(y)` — removing the
   inflation above. In a noiseless SCM this drives every true edge to ~1 and every non-edge to
   ~0. Use the partial-correlation *magnitude*, NOT a t-statistic (noiseless data makes SE -> 0
   and t-scores degenerate). Guard near-collinear parents (unstable partial correlations).
2. **Threshold applied per-target (per-row), never pooled.** Pooling across heterogeneous
   targets is structurally wrong. Chosen over a per-target cut on the existing standardized
   `|beta|`, which is ambiguous: in K2's row `P2=0.894, K0=0.447, rest~0` gives a tie
   (0.894->0.447 and 0.447->0 are both 0.447). Partial correlation lifts K0 to ~1.0, making the
   per-target cut unambiguous. The score change, not just the grouping, is what resolves it.
3. **Cross-check (reported, not the selector): stability selection over the 10 seeds.** Report
   per-edge selection frequency across the envelope (Meinshausen-Buhlmann 2010; CPSS
   Shah-Samworth 2013). With B=10 the frequency CI is coarse (~+/-0.3), so it is a reported
   heuristic confirmation that the per-target rule is stable, not the magnitude rule itself.
4. **Rejected as primary:** pooled GMM / Otsu / Jenks / Kneedle on the score distribution — each
   cuts the pooled bimodal shape and is pulled to the dominant valley (the `largest_gap` trap).
   May appear only as clearly-labelled diagnostics that never alter the persisted mask.

## Interpretation stance (locked)

Accept near-perfect noiseless recovery as **E1's intended role**. E1 is the recoverable control;
green E1 means "the label-free pipeline recovers structure when it is recoverable," which
unblocks interpreting E2+. The 006 report states the near-perfect separation plainly as a
property of the noiseless linear control — NOT dressed up as a hard-won result. **Do NOT add
noise to E1 or otherwise modify the locked benchmark to manufacture difficulty.** Noise and
nonlinearity belong to E2+, where discovery is non-trivial.

## Scope

**In scope**: a new label-free discovery selection rule (`cdd_oran/e1slice/` or a sibling
module) behind a fresh frozen protocol doc `docs/benchmark/E1_DISCOVERY_PROTOCOL_V2.md`; new
tests; the fresh frozen protocol commit; a run against the 003 rows and across the 004 envelope;
a dated delta report only after results exist.

**Out of scope**: changing the 003 frozen method or its artifacts; changing E1 SCM/truth; any
`E1V2Env`/true-adjacency import inside discovery; E2-E4 experiments or legacy training; retuning
after seeing a recovery number; any truth-informed threshold.

## Freeze ordering (mandatory)

1. Write `docs/benchmark/E1_DISCOVERY_PROTOCOL_V2.md` capturing the frozen method above in
   structural terms (no reference to 0.44/0.9). Commit it. **This commit's SHA is the freeze.**
2. Implement the discovery module + tests to record that SHA as `protocol_commit` in every
   artifact; discovery reads NO ground truth and does not import `E1V2Env`.
3. Run once on the 003 rows, then across the 004 10-seed envelope. Persist masks/artifacts.
4. Only after masks are persisted and hashed, score recovery vs `E1V2Env().true_adj_matrix()`
   into a separate `recovery.json`-style record.

## Gates

```
uv run ruff check .
uv run ty check
uv run pytest
```
plus: the discovery artifact recomputes its own `content_hash` and refuses to load on any
parent-hash / shape / numeric / `protocol_commit` mismatch (Plan 003 fail-closed pattern).

## Done criteria

- [x] `docs/benchmark/E1_DISCOVERY_PROTOCOL_V2.md` frozen (`c66b81d`); its commit SHA recorded in
      every produced artifact.
- [x] Selection rule justified on structure alone; pre-registration committed before first run.
- [x] Method run once against 003 rows AND across the 004 10-seed envelope; per-seed recovery
      reported (overall / NCP->KPI / KPI->KPI = 1.000 all seeds); no seed dropped. Reproducible
      from committed driver `scripts/e1_slice_recovery_v2_sweep.py` + digest
      `docs/benchmark/plan006_recovery_digest/`.
- [x] Gate outcome stated honestly: E1 recovery GREEN (both KPI->KPI edges recovered across seeds);
      near-perfect separation is a property of the noiseless control, not a hard-won result.
- [x] Full verification passes; artifacts fail-closed. (Merged `8a88829`; adversarial review verdict
      GENUINE — no leakage/triviality/deviation.)
- [ ] Delta report + `E1_DISCOVERY_RESULT_V2.md` + trained three-arm prediction envelope — **phase 3**.

Recovery deliverable merged to `feat/v2` at `8a88829` (2026-09-04). Remaining: phase 3 (trained
three-arm envelope for the prediction payoff, the as-run result doc, and the delta report).

## STOP conditions

- The method would have to see ground truth (or a truth-informed threshold) to select edges.
- The selection rule is chosen or adjusted to hit the values 0.439/0.456.
- The method is revised after its recovery number is inspected.
- Recovery is inconsistent across seeds and the response is to drop/reweight seeds.
