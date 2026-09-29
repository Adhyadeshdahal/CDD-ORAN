# Adversarial scenario-freeze review (S1/S2)

**Verdict:** The *use cases* are legitimate 3GPP SON cases; numerical scenarios are mostly E6 assumptions, not standardized traces. **Do not freeze this version yet.** Fix the following before any arbitration/xApp outcome. `mix="none"` checks establish mechanics, not controllability.

## BLOCKING before freeze

1. **S1 source-to-intervention leap.** Shafiq et al.'s ~3× *aggregate downlink volume and user count* does not imply 3× **per-existing-UE arrivals in a 250 m disk**. Only 13–83 UEs are exposed across six sampled seeds; overlap with existing hotspots/picos dominates realized stress. Calling this concentration “conservative” is unsupported. Tag the per-UE transfer **[A], informed by an [S] aggregate ratio**; disclose the spatial/population mismatch. Freeze unfiltered seed/location draws and duration. Do not select successful seeds or tune radius after gate A. Call 250 m a venue-scale assumption, not “one site's worth of area.”

2. **S2 favours intervention against freeze.** The post-warm-up OAM push affects 8–10/24 cells and bypasses churn; freeze cannot reject it. This tests exogenous misconfiguration recovery, **not just conflict mitigation**. Include simple alarm-triggered SMO restore and tuned-static references, or limit claims explicitly to WG3-only RICs. WG3 rollback's inability to undo OAM is declared (§9), but omitting SMO recovery overstates a systems win. Specify whether OAM resets actuator dwell/`last_change`; direct `plant.hys/ttt` writes (`sim.py:461–465`) leave RIC history stale. Test first post-push MRO write and rollback.

3. **The “120 km/h mistune” attribution is too strong.** TR 36.839 gives plausible settings/speeds, not a documented erroneous rollout on this road. The E6 A3-offset→Hys mapping changes repairability: MRO **never lowers Hys**, reducing TTT only after CIO saturates (`xapps.py:98–110`). Tag the mapping and applying it to all best-server road cells **[A]**; retain the numeric-pair citation and call it a *too-late HO hypothesis*. H-dir's `−1 dB` offset clipped to `Hys=0` is **not** Set 5; tag the result [A]/approximation.

4. **Freeze the full contract.** `SCENARIO_HOLDOUT` declares variants but not immutable seed lists, configuration matrix or hashes. Record DEV/untouched TEST seeds, `mix`, mobility/KPM/update, scored duration (600 s checks versus 1800 s default), reporting of all failures, and doc/code hashes before gate A. Version later edits; a prose “before outcomes” assertion is insufficient provenance.

5. **Enforce parameter validity and timing.** Constructors accept negative radius/speed, out-of-range fractions, subunit multiplier, event ending after scoring, invalid corridor length/share, and unsupported Hys/TTT. Validate scenario construction and event ordering. Add a default-length test of onset/end, one-time OAM push, persistence after RIC action, and action-independent tape *through* onset. Current tape test runs 4 s, **before S2's push**.

## Non-blocking, disclose rather than retune

- S1 is a *static* venue on DEV; moving hotspot is HOLDOUT. Do not describe the primary DEV scenario as a moving event. A 3× BE/eMBB file-rate multiplier with unchanged LL arrivals and UE count is not the observed mix of event arrivals; label this simplified offered-load experiment.
- S2's reflected U-turns are artificial: at 120 km/h a UE turns every ≤30 s, generating repeated traversals and far more HO attempts than a one-way passing car. Show corridor-only/no-rollout event counts alongside the mis-set effect (already in §7), and call it a recurrent-road stress, not a representative drive-through.
- The `base` branch has no scenario object and a reported seed-11 equality, but the test compares **two current implementations** for 100 ticks rather than comparing a pinned pre-scenario artifact. Preserve a full-score pinned baseline/hash for base reproducibility. Do not infer bit identity for all configurations from one seed.

**Then run gate A as written.** If S1/S2 fail, preserve the negative result; changing the intensity, road placement or repairability starts a new version and a new holdout contract.
