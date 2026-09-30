# Morning brief 3: overnight 2026-09-29/30

**Status: FINAL for the night (06:05 NST),** except one pending number (marked ⏳). Everything is committed locally
on feat/v2; **nothing is pushed** (you said: push in the morning). No Colab machine is left running, and no Kaggle
job is running.

## The short version
1. **Step 1 (can our method find the cause-effect map?): yes, with enough data. Verdict PARTIAL** (PASS is still
   possible, ⏳).
   - On 1,200 brand-new episodes, MSCR found the whole conflict chain with the right signs: when a pico cell goes to
     sleep, its neighbours get more load, more violations, and more *protected-user* violations.
   - It found 7 of the 8 true neighbour links and made no wrong neighbour claims.
   - It is also the only method that stays honest on fake data where nothing happens. Every other method (SHAP,
     Granger, correlation, QACM, ...) "finds" 6-16 links there.
   - **Caveat: this was the third attempt.** The first failed on too little data plus a bug; the second just missed
     on too little data. Each attempt had its rulebook frozen beforehand and used brand-new seeds. Any write-up must
     show all three.
2. **Step 2 (a referee that uses the map to beat the others): on this scenario it most likely cannot. This needs
   your decision.**
   - The winning moves here are simple fixed rules, mainly "stop PowerES turning power back up", and they don't need
     a causal map.
   - The safe version of such a rule recovers only ~10 % of the loss. The version that recovers ~30 % breaks a safety
     guardrail (+27 % radio-link failures).
   - So I did **not** spend the ~110 CPU-hours on a full step-2 evaluation.
3. **What I suggest we discuss** (details at the end): move the referee test to a setting where the *map* is what
   makes the difference, or make the discovery result itself the contribution.

## Step 1 in detail: three attempts

| attempt | data (fresh each time) | what changed | result |
|---|---|---|---|
| v1 (frozen yesterday) | 60 episodes | — | **KILL.** Found 3 of 8 neighbour links (bar: 6). One sign backwards. |
| v2 | 480 episodes | fixed the sign bug; noise reduction; "half" counted as acting | **KILL, just.** Everything passed except the key link (p = .007 vs a bar of about .004). |
| v3 | 1,200 episodes | only the amount of data | **Main criterion PASS.** Key link found at z 6.1. |

**Why v1 missed.** Four agents independently diagnosed it.
- Mainly **too little data**: with 60 episodes, only about 3 of the 8 links are strong enough for *any* method of
  this kind to see, and MSCR found exactly those.
- Plus **one real bug.** MSCR mixed "raise" and "lower" requests when reading an effect's direction.
- Also: the "far cells" we used as a no-effect control are actually affected (sleeping picos push users two hops
  away), so that control was wrong.

**Why v2 just missed.** Its sample size came from v1's estimate of the key effect (+15). The fresh ground truth says
the effect is smaller (+9), the same "first estimates are too optimistic" effect as before. v3 was sized from v2's
real result.

**v3 numbers** (1,200 episodes, fresh ground truth, fresh fake data):
- Pico sleep → neighbour protected violations: estimated **+10.3** (truth +9.1), z 6.1.
- Pico sleep → neighbour load: z 29.5. Pico sleep → neighbour violations: z 14.8. PowerES → neighbour load: z −21.3.
- Neighbour links: 7 of 8 found, 0 wrong. Overall precision 0.92, signs 100 % right. 0 false links on fake data.
- ⏳ The comparison against the other methods (it decides PASS vs PARTIAL) was still running at 06:05. It is written
  to `.tmp/disc_v3/full.log` and then to `STEP1_V3_RESULT.md`.

**v2 comparison with the other methods** (480 episodes; v3's is pending):

| method | overall score (F1) | precision | false links on fake data |
|---|---|---|---|
| **MSCR v2** | **0.82** | **0.95** | **0** |
| Granger | 0.69 | 0.64 | 13 |
| correlation | 0.54 | 0.85 | 16 |
| SHAP | 0.47 | 0.90 | 6 |
| QACM / two-tower / PACIFISTA-style | 0.28-0.51 | | 6-11 |

On the one pre-registered head-to-head (neighbour links only), Granger edged MSCR in v2 (0.71 vs 0.67). It did so by
claiming many links (36), a lot of them false, with signs right only 71 % of the time.

## Step 2 in detail: why I stopped before the big run
1. **The cheapest go/no-go check failed.** After the 2-minute warm-up there is little left to referee (about 2.6
   carrier-offs, 2.7 power-downs and 0.4 pico sleeps per episode). Predicted recovery was 3 %; the bar is 35 %.
2. **Three agents argued it out** (forensics, advocate, skeptic). They agreed:
   - The "cheating referee" that recovered 135 % mostly blocks PowerES power-restores, which is in effect a fixed
     rule, plus luck: about half of its gain comes from seeing the future.
   - The one signal a learner could use (protected load on the cell itself) never changes a decision. So a referee
     with our map, without a map, or with a SHAP map would behave the same. The planned step-3 comparison could not
     show that the map matters here.
3. **A 40-seed development test on Kaggle** (7.4 CPU-h) settled it:

   | rule | share of the loss recovered | safety guardrails |
   |---|---|---|
   | block all PowerES power-restores | 30 % (CI −2 to 62 %) | **fail** (radio-link failures +27 %) |
   | same plus block sleeps and carrier-offs | 36 % (CI 4 to 69 %) | **fail** (+27 %) |
   | map-guided: block only in full cells | 10 % | pass |
   | map-guided plus latency guard | 10 % (CI 2 to 19 %) | pass |

   Nothing safe gets near the 35 % bar.

## Decisions for you
1. **Push?** About 20 local commits since d819c5a (the last pushed commit) are waiting. Say "push" and I'll push
   feat/v2.
2. **Where should the referee (step 2/3) live?** My suggestions, from the skeptic agent plus the results:
   - **(a) A test where map quality decides the outcome.** Use logs that are *not* randomised, as in real networks.
     The association-based methods then pick wrong links (they already invent 6-16 on fake data), and a referee built
     on their map makes wrong calls. MSCR's calibrated map should not. This keeps the causal claim central.
   - **(b) A scenario where the neighbour chain is the only fix,** for example a load dip followed by a surge, so
     there are many pico sleeps after warm-up.
   - **(c) Make discovery the contribution:** "which xApp harms which KPI, with error control", validated against
     simulator ground truth, with the honest-on-fake-data result.
   - These can be combined; (c) is nearly ready now.
3. **The "far cells" control** turned out not to be a clean control. Future protocols should use the fake-data
   (placebo) check instead, as v2/v3 do.

## Where things are
- **Results:** `scratchpad/e6_dev/decision/`:
  - STEP1_V1_RESULT.md, STEP1_V2_RESULT.md, STEP1_V3_RESULT.md (plus their JSON files);
  - STEP2_DEV_RESULT.md, STEP2_REFEREE_PLAN.md.
- **Rulebooks:** `docs/benchmark/E6P_DISCOVERY_PROTOCOL.md`, `_V2.md` and `_V3.md` (each frozen before its data).
- **New method:** `cdd_oran/decision/crt_units_v2.py` (12 tests); analyzer `scratchpad/e6_dev/e6p_disc_analyze_v2.py`.
- **Agent reports** (diagnosis A-D, step-2 debate F/G/H): summarised in `.tmp/PLAN.md`; scripts in `.tmp/diag/` and
  `.tmp/step2debate/`.
- **Night hiccups:** Colab took a machine back once (re-run on the laptop). Kaggle's upload service rejected new
  bundles for about 20 minutes. My own path mistake broke one round of cloud jobs; fixed in `cloud.py`.
