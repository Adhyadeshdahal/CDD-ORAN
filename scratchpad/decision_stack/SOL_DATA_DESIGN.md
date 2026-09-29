# E6 DEV collection and MSCR null: decision

**Use both, for different estimands.** Randomized WG3 arbitration trains the **policy-effect world model**. A separate bounded, operator-run DEV experiment identifies knob-family→KPI links; it is neither WG3 policy data nor a deployable controller. Start after gate B.

## (a) Variation and its price

**Policy data:** Keep `collect.py`'s logged assignments/propensities. Replace uniform 512-code exploration as the main regime: many codes cause no change. Predeclare a small mixture of accept-all and *requested-step* fractions 0 (reject), 0.5 (modify), 1 (accept), with sparse lock/rollback blocks. Draw before outcomes at each 20 s region epoch; log eligibility, request, assigned fraction, realised delta, ACK and neighbouring assignments. Do not condition inference on successful actuation. Randomization identifies **policy intention-to-treat** effects including xApp reactions, not arbitrary knob `do()` effects: request occurrence/base value still depend on load.

**Graph data:** In separate `mix=M4` DEV episodes, operator-authorized direct one-step, quantised writes to prelisted CIO, Hys, TTT, LL ratio and carrier knobs; randomize sites, sign/level and sham **before contemporaneous load**. Respect `ric.feasible`, dwell, a changes/hour cap, limited simultaneous treated regions, and predeclared per-slice/RLF/energy abort-and-restore rules. Carrier probes require observation/washout beyond 120 s dwell. No sleep probing if outside active M4 actions; unsupported edges remain unknown. Predeclare values/schedule; no free writes in WG3 policy runs.

Report *collection cost*: runs and changed-knob-seconds; paired excess SVR/per-slice violations, RLF, severe events, energy, goals and churn versus same-seed accept-all/no-probe, including tails and aborts. Paired runs account for cost, **not** the inferential null. Keep probes outside TEST.

## (b) Null and graph claim

**Frozen MSCR-v2's row-permutation p-values/BY declarations are invalid here.** Thinning, `n_eff`, or one row per episode×cell do not remove feedback or spatial interference. Whole-episode permutation works only for episode-randomized treatments within strata; its sample size is independent episodes, not cells.

Minimal test: a **sharp null of no assigned-probe effect** on predeclared future KPI windows. Assignment unit = region plus interference neighbourhood and carryover washout; keep its cell/time outcomes together. Resample the *logged assignment mechanism* conditional on pre-assignment history, scenario/load, eligibility and site type, holding outcome blocks and other assignments fixed (design-based CRT). Do **not** shuffle applied values/KPI rows. Adaptive reassignment, overlapping 90 s windows or treated neighbours invalidate naive swaps: use disjoint washed-out blocks, or episode-randomized schedules and whole-episode randomization inference. For average effects use episode-cluster inference. Recompute the full statistic/conditioner search per draw; calibrate type-I error and power on simulated null/injected effects.

**Implementation:** Preserve `mscr.py`; a versioned wrapper reuses its **observed statistic** but computes a new assignment-based null/p-values. If necessary, copy/version the statistic outside frozen MSCR. Do not inherit its FDR claim. WG3 assignment yields *policy→KPI*, not knob→KPI, edges. Graph weights remain soft priors.

## (c) Episodes

Reserved DEV seeds 16–30 are a **pilot**, not 15 replicates per stratum. Predeclare additional non-TEST seeds: **30 independent seeds per load×scenario** (six strata): **180 policy episodes + 180 independently seeded probe episodes**, plus paired accept-all/no-probe references. Use 20/stratum for fitting, 10 untouched for diagnostics; never recycle gate or TEST seeds. The 30 epochs in a 600 s episode are dependent, and rare Hys/TTT requests may still be underpowered. Report first-stage counts/effective assignment blocks by knob family; label unsupported edges **undetermined**, not absent. Expand only under a predeclared power/support rule.
