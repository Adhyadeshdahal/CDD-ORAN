# F7 audit (aud2): Experiment D, the E6 validity audit of PMRT (ruling R-15)

Auditor aud2, 2026-10-02, branch xm/audit-aud2 (from feat/v2 b7456c2). Read only. I did not write e6_audit.py.

**Verdict: PASS-WITH-NOTES.** The recomputation is sound: 1020 hypothesis tables reproduced with 0 mismatches
(`results/e6_audit/e6_audit.json`, reproduction_check). GT-NULL set A is the frozen v4 GT label (21 NULL; v4 counts
TRUE 29 / NULL 21 / INDET 10 reproduced). Set B is a GT-only, outcome-independent subset. The hypothesis-level
numbers below match the status file. But the stated CIs are too narrow, and set C is selected on PMRT's own
outcome. The reading "no clear evidence of invalidity" stands. The precision claim ("upper limits ~.12") does not.

Sources read: `scratchpad/xmethod/e6_audit.py`, `e6_audit_breakdown.py`, `results/e6_audit/*.json`,
`scratchpad/e6_dev/e6p_disc_analyze_v4.py` (slice_plan), `cdd_oran/decision/gt_p.py` (classify), status xm-pmrt.md
and PROTOCOL_NOTES.md (Experiment D lines).

## Findings (most severe first)

1. **MEDIUM: the cluster CIs hold the hypothesis set fixed.** `e6_audit.py:160-178` (`rates`) and
   `e6_audit_breakdown.py:33-47` bootstrap only over episode slices. The same 12 / 19 / 21 hypotheses recur in every
   slice, and their rejection propensities differ: `prot_min|nbr|e` (in B) rejects 2/10 s60 and 2/5 s120, always
   with the GT sign; most others reject 0. So the stated interval is conditional on these hypotheses. It does not
   cover the rate of a "GT-null hypothesis" in general, which is how PROTOCOL_NOTES reads it ("CIs cover .05",
   "upper limits ~.12"). I recomputed from the stored per-slice p-values (loadsp_c, p2, .05; 4000 reps):

   | set | slices | reported (slice bootstrap) | crossed (hyp x slice) bootstrap | slice t-interval |
   |---|---|---|---|---|
   | A 21 | s60 | .152 [.110, .195] | [.048, .286] | [.100, .205] |
   | B 12 | s60 | .075 [.033, .117] | [.017, .158] | [.023, .127] |
   | B 12 | s120 | .083 [.033, .133] | [.000, .217] | [.010, .156] |
   | C 19 | s60 | .074 [.032, .121] | [.021, .153] | [.020, .128] |
   | C 19 | s120 | .063 [.032, .095] | [.000, .147] | [.008, .118] |

   With 5 clusters (s120), a percentile bootstrap over slices undercovers anyway: only 126 distinct resamples exist.
   Fix: report the crossed (or the hypothesis-level) interval next to the slice one. State the precision as "upper
   limits .15-.22: the data cannot rule out a rate of 3-4x nominal on GT-null hypotheses", and say that no
   exceedance is detected.
2. **MEDIUM: s60, s120 and pooled are re-partitions of the SAME 600 episodes** (`e6p_disc_analyze_v4.py:298-314`:
   s60_i = seeds with j // 60 == i, s120_k = j // 120 == k). The s60 and s120 rates are therefore not two
   independent confirmations. The status and notes list them side by side as if they were. Fix: present one slice
   kind as primary (s60: more clusters) and the others as the same data at another granularity.
3. **LOW: set C is selected on the outcome.** `e6_audit_breakdown.py:76-77` drops a hypothesis when its GT CI
   excludes 0 AND PMRT rejects it in >= half of the s60 slices. Any hypothesis that PMRT rejects often is removed,
   so C's rate is biased down by construction. It cannot serve as a false-positive rate. The two removed
   hypotheses do look like real sub-delta effects: sleep|own|e (GT -187 [-192, -183], 10/10 rejections) and
   carrier|own|load (GT -26 [-32, -20], 8/10), both with PMRT's sign equal to the GT sign. So the explanation is
   plausible. The rate, however, belongs to B, which uses GT information only: .075 / .083. Seven hypotheses in C
   also have GT CIs that exclude 0, so C is not a "null" set either. Fix: B is the headline proxy, and C is
   described as "A minus two hypotheses identified post hoc from PMRT's own rejections".
4. **LOW: the placebo-log interval assumes independent hypotheses.** The Wilson CI on the 60 placebo hypotheses is
   over hypotheses from ONE set of 40 episodes; the 15 hypotheses of a family share the redraws and the units. Per
   family (loadsp_c): sleep 3/15, carrier 2/15, ptx 1/15, prot_min 0/15. Without the clip: sleep 3/15, carrier 0/15,
   ptx 1/15, prot_min 0/15. So the clip-related difference is the 2 carrier rejections. The true uncertainty is
   larger than [.047, .201]. The status already says "one cluster, inconclusive". Fix: show the per-family split
   and drop the Wilson interval, or call it a lower bound on the uncertainty.
5. **NOTE: B's membership in rare-event KPIs is borderline.** `prot_min|far|rlf` and `prot_min|own|rlf` have GT CI
   [-.001, 0.0], so 0 lies exactly on the upper limit. delta for rlf is .0075. Harmless to the conclusion: neither
   hypothesis rejects in more than 1 slice.
6. **NOTE: a wrong-sign rejection.** carrier|far|e (GT +153 [108, 198]) is rejected in 1 / 10 s60 slices with
   sign -1. That is a sign error on a real (sub-delta) effect, not a null false positive. Worth one line wherever
   "sign = GT" is claimed for the excess.

## What checks out
- The statistic, RNG stream (`default_rng([0, 6616, 3, family_idx, split])`), exceedance counting and p_used rule
  reproduce the frozen v4 tables: 1020 tables, 0 mismatches.
- A = GT status "NULL" exactly. B = A with GT 95% CI containing 0 (GT episodes are disjoint from eval episodes, so
  B does not depend on the PMRT outcome).
- Per-hypothesis counts, signs, and the A / B / C rates in PROTOCOL_NOTES and xm-pmrt.md match the stored JSON.
- The loadsp vs loadsp_c comparison on A / B / C is like-for-like (same draws, same hypotheses). The clip changes
  little on GT-NULL, as stated.
