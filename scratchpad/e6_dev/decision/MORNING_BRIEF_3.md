# Morning brief 3: overnight 2026-09-29/30 (step 1: can our method find the cause-effect map?)

**Status: IN PROGRESS (last update 00:55 NST).** Everything is committed locally only; nothing is pushed.

## The short version
1. **Step 1, first try: KILL.**
   - Our discovery method (MSCR) looked at randomized logs to find which xApp actions hurt which cells.
   - The **ground truth** comes from the simulator's "what if we had said no" replays. It shows that the conflict
     chain is real: when a small (pico) cell goes to sleep, its neighbours get more load and more protected-user
     violations.
   - MSCR found 3 of the 8 true "neighbour" links. The bar was 6 of 8.
2. **Why it missed.** Four independent agents looked into it. Their verdict: mostly **not enough data, not a broken
   method.**
   - With 60 episodes, only about 3 of the 8 links are strong enough for *any* method of this kind to see.
   - MSCR found exactly those 3. The key link (sleep → neighbour protected violations) is real but faint: a signal
     of about 1.5 against a noise bar of 3.
   - There was also **one real bug.** MSCR mixed "raise" and "lower" requests when reading the direction of an
     effect, so it got one sign backwards.
   - Two smaller issues: the "half" action was treated as half, but it actually acts almost fully; and the "far
     cells" we used as a no-effect control are in fact affected.
   - **The comparison methods are not honest either.** On fake data where nothing has any effect, they still
     "found" 10-18 links. MSCR found 0 there.
3. **What I did, as you authorised.**
   - Wrote an improved **MSCR v2**: the sign bug fixed, correct dose for "half", and noise reduced using each
     outcome's own before-window.
   - Wrote a new, disclosed rulebook: `docs/benchmark/E6P_DISCOVERY_PROTOCOL_V2.md`.
   - Started a **fresh test on brand-new seeds**: 480 episodes (8× more, sized by the power analysis), plus a fresh
     ground truth and fresh fake data. The old data is used only for development, never for the final score.
4. **Result of the fresh test:** pending (see below).

## Step 2 (the referee): an early warning, which needs your decision
I did not wait for step 1 to finish before checking whether step 2 can win at all. I ran the plan's cheapest
go/no-go check, then had three agents argue it out: a forensics agent, an advocate and a skeptic.
- **The go/no-go check failed badly.** After the 2-minute warm-up there is little left for a referee to decide:
  about 2.6 carrier-offs, 2.7 power-downs and 0.4 pico sleeps per episode. Predicted recovery is about 3 % of the
  loss, against the 35 % needed.
- **The advocate could not make the case either.** It tried hard, including 60 small simulations on development
  seeds.
  - The best referee it found recovers about 23 %, very noisily, and breaks the latency guardrail.
  - Its one working lever is a simple fixed rule, "don't let PowerES raise power in already-full cells", and that
    rule needs no causal map.
- **The skeptic's numbers.**
  - About half of the "cheating referee's" 1.35 comes from seeing the future. Without that it is worth about 0.64,
    and even that needs a full simulator for planning ahead.
  - The one learnable signal (protected-user load on the cell itself) never changes a decision.
  - So "referee with our map" vs "no map" vs "SHAP map" would likely come out identical. The step-3 comparison could
    not show that the map matters on this scenario.
- **The forensics agent found what the "cheating referee" actually does.** It re-ran the oracle and recorded its
  choices.
  - Mostly it **blocks PowerES from turning power back up**, about 149 of 156 times per episode, so in effect it is
    a fixed rule. It also damps the slice-protection xApp a little.
  - A plain fixed rule on development seeds, "after warm-up, block power-restores, pico sleeps and carrier-offs",
    already recovers about 30 % of the loss with the guardrails passing. That is very noisy (CI 1-55 %), but it
    beats the best fixed rule we had tested (26 %), and it uses no causal map at all.
- **My reading.** On P3 surge-L40, a causal-map referee is unlikely to beat a well-chosen fixed rule. Even if it
  did, the map would not be the reason. The discovery half (step 1) can still stand on its own.
- **Options for you, from the skeptic, which I think are worth discussing:**
  1. A test world where map quality decides the outcome: logs that are *not* randomised, so the correlation-based
     methods pick the wrong links (they already invent 10-18 links on fake data) and a referee built on their map
     makes wrong calls.
  2. A scenario where the neighbour chain is the only fix, for example a load dip then a surge, so there are many
     sleep decisions after warm-up.
  3. A diagnosis/attribution claim: "which xApp harms which KPI, with error control". MSCR's precision 1.00, correct
     signs and honest placebo behaviour support this; the baselines don't.
- I did **not** spend the ~110 CPU-h step-2 evaluation. The evidence says it would most likely fail, and a pivot is
  your call.

## Timeline
- 23:30 The full step-1 analysis started on Colab. Colab took the machine back halfway, so I re-ran it on the laptop
  as a separate, light process.
- 23:55 Ground truth computed. The key chain is real (the premise check passes).
- 00:00 Four diagnosis agents launched. All four agreed: mainly a data-size limit, plus the sign bug.
- 00:30 Fresh data collection launched: Kaggle (ground truth, 3 jobs; test set, 2 jobs) and Colab (fake data).
  Kaggle's upload service was flaky, which cost about 20 minutes.
- 00:50 A builder agent is writing MSCR v2. The v2 rulebook is drafted, and will be frozen before any fresh data is
  opened.
