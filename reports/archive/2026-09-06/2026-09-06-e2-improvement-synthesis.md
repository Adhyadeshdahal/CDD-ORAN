# E2 Discovery — Improvement Synthesis (4-agent analysis, 2026-09-06)

Four parallel read-only analyses (performance, statistical quality, replacement method, systems/process)
of why the E2 label-free discovery run costs ~8h/seed and over-selects edges. Nothing here changes code
or the freeze. This is the decision brief. Prior research: `reports/2026-09-05-discovery-method-scout.md`.

## TL;DR

The analyses converge on one story: **the frozen pdCor-with-permutation method is BOTH irreducibly slow
AND statistically miscalibrated — two independent problems.** Making it faster (parallelize or cloud)
just finishes a method we now know is unsound. The principled long-term fix is to **retire the
permutation-based pdCor as the LIVE E2 method and move to a genuine conditional-independence test with an
analytic null (RCoT), done as a clean pre-truth re-freeze** — which fixes the calibration root cause and
the speed at the same time. But **de-risk it with one cheap, truth-free calibration study first**,
because there's a deeper risk (E2's noiselessness) that would defeat *any* CI test and redirect the effort
to the benchmark design instead.

## The two independent problems

### Problem 1 — Speed (8h/seed; multi-day envelope)
- The permutation loop is **99.7% of runtime**: ~84 candidates × 999 permutations, each rebuilding a
  4000×4000 (128 MiB) U-centered distance matrix — single-threaded, memory-bandwidth-bound.
- **No bit-identical caching win remains** — the code already hoists every genuine cross-permutation
  invariant (`cond_u`, `cond_self`, `pyz`, `vb_proj` computed once per cell). The 999× rebuild is inherent
  to the frozen null definition. Confirmed by profiling.
- Ways to make it fast, and their cost:
  - **Parallelize across the 84 independent candidates** — byte-identical output, ~12× on 16 cores →
    ~35–45 min/seed (order-of-magnitude), but **full-core thermal load** (the thing that overheated the
    laptop). Effort M.
  - **Cloud burst** — run the unchanged frozen code on a bandwidth-optimized box; all 10 seeds in
    **~3–8h wall for <$50** (spot ~$5–15), no laptop heat, freeze preserved exactly. Effort S.
  - Faster still (bilinear reduction ~2×, GPU 10–100×) all **change float rounding → not bit-identical**
    → would require a re-freeze anyway.

### Problem 2 — Quality (9× over-selection) — the pivotal finding
- The implementation is **faithful to the spec (no bug)**; the over-selection is a property of the method.
- 17/36 spurious KPI→KPI edges is **~9× the nominal FDR** — genuinely anti-conservative.
- **It is NOT the collider mechanism the protocol pre-registered a worry about** — E2's exogenous-
  independence design (`P_{t-1} ⊥ P_t`) rules colliders out. The real cause:
  **marginal-permutation miscalibration.** Permuting the candidate column destroys its dependence on the
  conditioning set, so the null samples *joint* independence, not *conditional* independence. The lagged
  KPIs are all smooth functions of one shared parent vector → strongly conditioning-set-dependent → the
  null is too tight → true-negatives land in its upper tail. Params (drawn independent) calibrate fine.
  That asymmetry is the signature and it predicts the FPs exactly where observed.
- **Deeper limit:** `|pdCor|` is a conditional-*association* score, not a CI certificate, and its rank-1
  partial cannot residualize a target that is a rich deterministic function of a 13-dim Z (this also
  explains why the denominator guard fired 0 times — the failure is invisible to it).
- **Fixes:** *in-family* — replace marginal permutation with **conditional randomization** preserving X|Z
  (closes the calibration gap; feasible since the exogenous law is known), restrict the conditioning set,
  use BY not BH. *Fundamental* — structure recovery wants a genuine **CI test** (RCoT/KCIT).
- **Inherent hard point (hits every method):** E2 is **noiseless** — `Y = f(Z)` deterministically — which
  makes *any* CI test near-degenerate. This is a property of the benchmark, not the estimator.
- **Empirical confirmation (seed-0 mask scored, 2026-09-06):** NCP→KPI recovery P **0.929** / R 0.812 /
  F1 0.867 (13 of 16 true edges, **1 FP**); KPI→KPI **17 FP / 36** (all true-negatives), rejection 0.528.
  The failure is **isolated to the Z-dependent lagged-KPI candidates** exactly as the miscalibration
  diagnosis predicts — the independent params calibrate cleanly, the Z-dependent KPIs don't. The
  estimator's power on real edges is fine; the defect is null-calibration on Z-dependent nulls. (n=1 seed;
  documented baseline for the retired method — nothing tuned.)

## Why speed and quality interact

Cloud-bursting or parallelizing pdCor solves speed but finishes a **known-miscalibrated** method. That
yields an honest-but-weak recovery number — a valid *documented boundary baseline*, but **not a good E2
discovery method going forward.** Given the goal is a principled long-term fix (not a hack), accelerating
the flawed method is the wrong primary investment.

## The replacement (RCoT) fixes both at the root

Agent 3's build plan (freeze-ready): a randomized-Fourier-feature kernel CI test with an **analytic null**.
- The analytic null **eliminates the permutation-miscalibration root cause entirely** (no permutation → no
  marginal-vs-conditional gap).
- It is a **genuine CI test** (addresses the association-not-CI limit).
- It is **linear-N, no O(N²), no permutations → minutes for all 10 seeds** (speed solved as a byproduct).
- We are at a **clean pre-truth point** (0 recovery runs), so switching is scientifically free *iff* done
  as a fresh pre-registration (new protocol doc + new commit; pdCor stays on disk untouched).
- **Residual risk:** RFF Monte-Carlo variance, null-approximation tail accuracy, and — crucially — the
  noiseless-determinism degeneracy above, which RCoT does **not** escape.

## Recommended path (sequenced, de-risked)

- **Step 0 — cheap, truth-free calibration study (do FIRST, ~hours, no commitment).** On synthetic
  E2-structured data (X⊥Y|Z with the shared-parent, deterministic structure), empirically compare the
  calibration of (a) marginal permutation, (b) conditional randomization, (c) RCoT's analytic null — and
  test whether any of them actually separates parents from non-parents **under E2's noiselessness**. This
  (i) confirms the Problem-2 mechanism (currently medium-high confidence), (ii) validates RCoT calibrates
  where pdCor doesn't *before* building it, and (iii) reveals whether the deeper blocker is the method or
  the benchmark's determinism. This single study decides the branch below.
- **Step 1a — if RCoT calibrates well:** build → adversarial review → freeze → run (minutes) → score, per
  Agent 3's plan. This becomes the LIVE E2 method.
- **Step 1b — if noiselessness defeats CI tests too:** the fix is a **benchmark-design decision** (add
  bounded observation noise to E2, or accept determinism and use a determinism-aware recovery criterion),
  not a method swap. Escalate that as its own scientific call.
- **Optional / parallel — document pdCor as a recorded baseline:** score the already-computed seed-0 mask
  (examines truth for a *retired* method, nothing to tune) to put an honest number on its ~53% KPI→KPI
  rejection, then close it out. Cheap, integrity-clean, good for the writeup.
- **Deprioritize:** cloud-burst / laptop-parallelize the frozen pdCor run — only worth it if you
  specifically want the pdCor *result* as a documented baseline; it is not the path forward.
- **Adopt regardless (process, cheap):** a **real-N pre-commit cost gate** (time one full-N cell before
  freezing a heavy run — would have caught "5–18 days" up front), and pin `OMP_NUM_THREADS=1` for
  cross-machine hash reproducibility.

## Decisions needed from you

1. **Direction:** endorse retiring permutation-pdCor as the LIVE E2 method and moving to a genuine CI test
   (RCoT), keeping pdCor on disk as a recorded protocol? (yes / adjust)
2. **Sequencing:** run the cheap Step-0 calibration study first (recommended), or go straight to building
   RCoT?
3. **Benchmark noise:** are you open to adding bounded observation noise to E2 if Step 0 shows determinism
   defeats CI tests? (This is a benchmark-design call, not a hack — E2 being noiseless is itself a design
   choice.)
4. **Baseline:** document pdCor's failure by scoring the existing seed-0 mask (examines truth for the
   retired method), or skip and just retire it?

## Appendix — one-line per agent
- **Performance:** perm loop = 99.7%; no caching win left; only bit-identical speedup is 84-way candidate
  parallelism (~12×, full-core heat) → ~40 min/seed; anything faster needs a re-freeze.
- **Quality:** faithful impl; 9× over-selection = marginal-vs-conditional permutation miscalibration (not
  collider); fixable in-family via conditional randomization; pdCor is association-not-CI (fundamental);
  noiselessness degenerates any CI test.
- **Replacement:** RCoT (+WGCM cross-check), analytic null, minutes not days, fresh pre-truth freeze;
  human knobs = RFF feature counts, null approximation, deps (scipy/sklearn).
- **Systems:** cloud burst <$50/<1 day preserves freeze exactly; add real-N cost gate; per-cell
  checkpointing + K=1 on laptop; pin BLAS threads for reproducible hashes.
