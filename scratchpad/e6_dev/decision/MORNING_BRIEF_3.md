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

## Timeline
- 23:30 The full step-1 analysis started on Colab. Colab took the machine back halfway, so I re-ran it on the laptop
  as a separate, light process.
- 23:55 Ground truth computed. The key chain is real (the premise check passes).
- 00:00 Four diagnosis agents launched. All four agreed: mainly a data-size limit, plus the sign bug.
- 00:30 Fresh data collection launched: Kaggle (ground truth, 3 jobs; test set, 2 jobs) and Colab (fake data).
  Kaggle's upload service was flaky, which cost about 20 minutes.
- 00:50 A builder agent is writing MSCR v2. The v2 rulebook is drafted, and will be frozen before any fresh data is
  opened.
