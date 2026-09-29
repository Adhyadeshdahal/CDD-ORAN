# Freeze review: `E6-xapps-v2` and `e6-probe/3`

**Verdict: neither contract is freeze-ready.** V1 Gate A remains killed; this is not an efficacy verdict.

## `E6-xapps-v2` — BLOCKING

1. **Paired MLB writes are not atomic.** `MLB._pair` emits separate `cio[s,n]` and `cio[n,s]` requests; `_meta`/`result` tracks only the forward key. WG3 can decide them separately across regions, or one can fail feasibility. A surviving reverse write breaks the spec's “ONE antisymmetric decision,” and its ownership/restore accounting. Either implement paired disposition under the claimed arbiter, or specify non-atomic semantics, track *both* applied deltas, and test partial ACK/NACK/veto/rollback. This directly affects conflict mitigation, not just Gate A.

2. **No executable v2 Gate A path is frozen.** `gate_a.py` hardcodes `mix="M4"` and `E6Env(cfg)`; v2's KPM/full-pair knobs require `make_env_v2`. Freeze a hashed v1/v2 runner; verify builder per arm, identical tapes and freeze/noarb/static identities before confirmation. Resolve optional `env.py` hooks versus the factory.

3. **Decide calm-base MRO behavior.** `late_eff = too_late + lowq_dwell/T310` compares dwell seconds to HO-count denominator, each second one near-RLF. A disclosed *no-SLA* check lowers TTT on ~20/24 base cells; the precursor dominates ordinary-network actions. One UE dwelling cannot be treated as independent failures without justification. Predeclare mechanism-based normalization/minimum independent UEs or a base action/churn guardrail; verify on mechanics-only tapes. If retained, explicitly accept widespread base changes. Never choose the weight from SVR.

4. **Freeze artifacts and authority.** Contract DRAFT, spec hash placeholder, seeds/tag 7101 unregistered, `[V]` citations unchecked, confirmation n and hooks undecided. Resolve and hash runner/code before v2 outcomes. Preserve whole-episode 0.85/0.35 and v1 disclosure; event-window scores cannot replace the gate. Tuned static uses old neighbours while v2 accesses all CIO pairs: retain continuity, but predeclare equal-rights static/goal-keeping sensitivity before superiority claims.

**Non-blocking:** NRT misses ≥95% too-late coverage on base/surge (14–19 events); disclose uncertainty and first-discovery misses. ≈17/23 relations/cell weaken selective-ANR framing. Document `mr` occupancy/L3 measurement latency before live-RIC claims. Physics predicts failure; do not move thresholds.

## `e6-probe/3` — BLOCKING

1. **Freeze the complete probe contract.** `/2` remains labelled pending; `/3` is an addendum. Pin `ProbeConfig` (`randomization="block"`), `ABORT_THETA` table/hash and derivation seeds, v1 `M4`, 1800 s, arms, CRT, splits, outcomes and whole-trajectory claim; identify grid/audit revisions. Reference-calibrated thresholds do not guarantee ≤2% future false aborts.

2. **Verify harm monitoring before 180 jobs.** `/2` aborted 33% of shams; `/3` reports 0/46 offline, 1/9 fresh-seed sham aborts, and only 147/174 offline block decisions reproduced. Resolve 27 discordances (delivery, baseline, restore, scoring). Predeclare arm/stratum pause thresholds. q99.5 thresholds are large (LL ratios 9–59; RLF excess 5–38): inject harmful trajectories to verify timely detection and set absolute harm/cost limits. Reference rarity alone is insufficient. Keep CRT intention-to-treat; never select rows by post-assignment effectiveness/aborts.

**Non-blocking:** CIO/carrier carryover allows only the *whole-trajectory* sharp-null claim, not a local edge. Supersession and strict-unit power leave many hypotheses undetermined. Keep blinded tranche and full cost reporting; placebo calibration does not validate locality.
