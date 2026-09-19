# "Going causal" for E-series discovery — 3-lane research synthesis (2026-09-06)

Synthesis of three parallel research lanes (methods / systems-GPU / theory-fit) on the user's question:
should we upgrade from "bland RCoT (conditional correlation)" toward genuine causality? Lane reports:
`2026-09-06-causal-upgrade-{methods,systems,theory}.md`. This is a decision brief; changes no frozen code.

## Headline (all three lanes converge)

**For E2, the "genuine causality" machinery — PC/FCI orientation, colliders, v-structures, Markov-
equivalence classes, FCM orientation (ANM/PNL/LiNGAM/IGCI) — is largely REDUNDANT.** Two properties E2
already has for free remove the need for it:

1. **Direction is known** — the temporal order `t→t+1` orients every edge. Orientation is the one thing
   every FCM method exists to provide, and we don't need it.
2. **Causal sufficiency holds** — params are independent exogenous roots, no latent confounders → FCI's
   latent machinery (and the missing-data FCI of arXiv:1705.09031) is overkill.

So the fashionable upgrade ("wrap it in PC/FCI, get a causal graph") would be **mostly wasted effort on
E2.** Both arXiv papers the search surfaced are off-target for our crux (1705.09031 = MNAR missing-data
FCI, we have none; 1502.02454 = PC scalability to thousands of nodes, we have 14). Neither touches
determinism.

## The honest correction to the premise

The exploration was partly motivated by "RCoT under-recovers because E2 is noiseless." **The theory lane
confirms our own diagnosis: that was NOT determinism** — it was the BH × permutation-resolution arithmetic
artifact, already fixed by **RCoT-v2** (`block_perm_reps 99→299`, ~0.7 recall / ~0.01 KPI→KPI FP, pending
the running confirmation). Determinism actually bites as **co-parent false positives** (sibling KPIs share
params), not as lost power. And **noiselessness sharpens the task** (edges are exactly identifiable) — it
only caps FCM *orientation*, which we don't use. **Do NOT add noise to the benchmark** (counter-productive,
and risks a gameable benchmark). Change the method, not the benchmark.

## The two genuinely valuable upgrades that emerged

Not orientation — these:

**(A) Conditioning STRATEGY — tiered low-order skeleton search (methods + systems lanes).**
Instead of testing every candidate against the full 13-D conditioning set (where RCoT strains), a
PC-stable *skeleton* search uses minimal, low-order conditioning:
- kills the 36 lagged KPI→KPI true-negatives at **order 0 (marginal)** — where RCoT is trivially
  calibrated; sidesteps the very block_perm×BH artifact that forced v1→v2;
- removes most false param→KPI edges at low order; the 16 true edges survive (no separating set);
- keeps decisions in the 1-D/low-D regime where RCoT has power AND calibration.
Keep our **per-target BH-FDR** on top (plain PC loses graph-level error control — our FDR is the *cleaner*
story). Reference: **causal-learn** (MIT, pluggable custom CI, tiered background knowledge, Windows/NumPy-
clean) — but the skeleton loop is ~40 lines; adopt the pattern, not a whole library. `tigramite`/PCMCI is
the time-series twin but GPLv3.

**(B) Determinism-NATIVE functional discovery (theory lane) — the deeper, more novel bet.**
Instead of fighting noiselessness with a CI test, *exploit* it: `H(Y|S)=0` (zero conditional entropy /
functional support / predictability / sensitivity screening) is a razor-sharp parent signal that subsumes
RCoT. Turns our benchmark's "weakness" (determinism) into the discriminator. Non-starters ruled out with
the exact violated assumption: ANM/PNL (need noise), LiNGAM (needs linearity), IGCI (bivariate orientation
+ invertibility we lack), plain PC (harmed by determinism-faithfulness).

## GPU (systems lane) — and the critical staging rule

The user's two links (Tencent fast-causal-inference, NVIDIA RAPIDS) are **effect estimation** (treatment
effects at SQL scale), NOT structure discovery — wrong tool family. Off-the-shelf GPU-PC (cuPC, gpucsl)
bakes in a Gaussian CI in CUDA with no plug-point for RCoT and is Linux-only. The target architecture —
**batched GPU-RCoT as the CI oracle driving a CPU PC skeleton** — exists as a torch reference (Bloomberg
`causal-ts`, GPL-3: read for design, don't vendor). Fits our 4 GB card easily (RFF tensors ~4000×25).

**Staging rule (matches our freeze discipline):**
- **Stage (a) — GPU-accelerate the EXISTING frozen RCoT-v2** (batched torch, `device=` flag, NumPy oracle,
  equivalence gate: GPU reproduces CPU masks). Behavior-identical → **no re-freeze**, pure systems.
- **Stage (b) — the method upgrades (A conditioning, B functional)** are NEW science → fresh pre-
  registration + adversarial review, truth-free prototypes first.
- **Do (a) first — it de-risks (b)** and is the right use of the offered overnight GPU time.

## Recommended sequence

1. **Now / overnight (pure systems, GPU, no science change):** batched GPU-RCoT-v2 + equivalence gate.
   Reusable acceleration, no re-registration; the corrected version of the GPU work.
2. **Truth-free prototypes (inform a fresh pre-reg), bar = beat v2's ~0.7 recall / ~0.01 FP on the
   byte-faithful test-bed with strictly lower-dim conditioning / a determinism-native signal:**
   - Prototype **A**: tiered PC-stable skeleton + frozen RCoT oracle (conditioning order ≤2–3).
   - Prototype **B**: functional-support / Markov-boundary (zero-conditional-entropy) screening.
   - Winner → fresh pre-registration → freeze → score. Keep per-target BH-FDR.
3. **Do not add benchmark noise. Re-audit assumptions per env:** E3–E5 may flip the verdict — genuine
   noise → ANM/PNL apply; latent confounders → FCI/causal-sufficiency relevant; real KPI→KPI edges →
   determinism-faithfulness harder and (B) matters more.

## One-line answer to "is the move to causality the obvious upgrade?"
Partly. Orientation machinery is redundant on E2 (time + sufficiency give it free). The real upgrades are
**smarter conditioning (A)** and **exploiting determinism (B)** — better than bland RCoT for the *right*
reasons, not fashion — plus GPU-accelerating the frozen test as safe, reusable systems work.
