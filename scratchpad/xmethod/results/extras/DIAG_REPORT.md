# R-60 diagnostics X5 / X6: report

**POST HOC, EXPLORATORY (R-60).** Declared in `scratchpad/xmethod/EXTRAS_PROTOCOL.md` (Amendments, 2026-10-05)
before any X5 / X6 run. Neither changes a frozen file, a frozen verdict (C1-C3) or an EVAL output. Branch `xm/diag`.
Tables: `results/extras/diag/diag_tables.json` and `DIAG_TABLES.md` (built by `python -m cdd_oran.xmethod.diag`).
Raw merged records stay local, outside git (repo / disk rule).

Rates are raw p <= .05 over testable candidates. "boot CI" = the frozen seed-cluster bootstrap (eval_analysis,
2000 reps); "CP" = Clopper-Pearson. R-30 labels (INVALID: CI low > .05; VALID: CI high <= .075) are descriptive.

**Notes that apply to both.**
- `tests/test_xmethod_extras.py::test_freeze_manifest_still_valid` fails only for `scratchpad/e6_dev/xm_dispatch.py`,
  the post-freeze change covered by amendment A-1 (`docs/xmethod/PROTOCOL_A_AMENDMENTS.md`). The frozen file and the
  manifest were not edited.
- `diag.py` (the analysis module) was written after the declaration. It implements the declared metrics. Two
  additions are marked below as not declared.

## X6: PMRT-GBM under a wider told dither (advisor question C)

**What was run.**
- E1 R2, n 1000, kappa .25; seeds 3_302_000-3_302_199 (200 fresh datasets); 15 arms, 3000 units, all ok.
- The arms:
  - the 2 x 2 at told width x2 (centring law x redraw law, each told / true);
  - the told-width grid x .5, .8, 1 (exact), 1.25, 1.5, 2 for pmrt_nl_eq (GBM) and pmrt_eq (Lin);
  - the bias log on every GBM arm.
- Platform: Colab.
  - The first session (`xm-diag-x6`, TPU v5e-1 host, 4 parts, commit 3628f57) was job-lost at 14:44:47Z.
  - Kept under the campaign merge rule: 1470 ok records, 96 complete datasets plus 4 partial ones.
  - Only the missing units were relaunched (`--skip-complete-from`; R-35 re-runs every arm of a partial dataset).
    TPU assignment returned "Service Unavailable" twice, so the relaunch used a Colab CPU runtime (`xm-diag-x6d`,
    2 parts, commit 4348ab4). That commit differs from 3628f57 only by `diag.py`; the run code is unchanged.
  - x6d exited 0 at 20:28:30Z with 1560 records and clean pins. The merge gives 3000 / 3000 ok, 0 missing, 0 errors
    and 0 conflicts. The 30 duplicates are the 4 re-run partial datasets; the later record wins.
- Provenance: all 200 dataset hashes equal the frozen generator's. There are 0 role mismatches and 0 unexpected
  records.

**Numbers** (truth-null raw rate, 12 candidates per seed, n = 2400; boot 95 % CI).

| cell (told width x2) | centring | redraw | truth-null raw [CI] | R-30 | mean z_bias [CI] |
|---|---|---|---|---|---|
| TT = frozen told arm | told | told | .079 (190/2400) [.068, .090] | INVALID | .090 [.051, .130] |
| Tt | told | true | .046 (110/2400) [.038, .054] | VALID | .090 [.051, .130] |
| tT | true | told | .038 (92/2400) [.031, .047] | VALID | -.448 [-.506, -.390] |
| tt | true | true | .045 (109/2400) [.038, .053] | VALID | -.448 [-.506, -.390] |
| exact (width x1) | exact | exact | .043 (103/2400) [.035, .051] | VALID | 0 |

- TT's rejection rate is .004 in the bottom z_bias tercile and .210 in the top. Top minus bottom is .206
  [.177, .234].
- tT's top minus bottom is .105 [.083, .128].
- P_placebo raw rates (n = 800) agree: TT .077 [.059, .095], tT .036 [.022, .051].
- TT reproduces X3. It is lower than X3's E1 n 1000 told-x2 rate, .100 [.082, .118] (60 seeds, different datasets),
  but the CIs overlap.

**Width grid** (truth-null raw rate; GBM mean z_bias):

| told width | x .5 | x .8 | x 1 (exact) | x 1.25 | x 1.5 | x 2 |
|---|---|---|---|---|---|---|
| pmrt_eq (Lin) | .322 INVALID | .112 INVALID | .052 VALID | .012 | .003 | .000 |
| pmrt_nl_eq (GBM) | .156 INVALID | .083 INVALID | .043 VALID | .038 | .048 | .079 INVALID |
| GBM mean z_bias | .237 | .099 | 0 | -.036 | -.012 | .090 |

- A told law narrower than the truth inflates both methods.
- Lin is monotone: it becomes conservative as the told law widens.
- GBM is U-shaped: it is valid from x1 to x1.5 and inflates again at x2.

**Declared outcome: PARTIAL (failed: P4, sub-prediction P4c).**
- P1 (CRT sanity): holds. Tt and tt are not INVALID.
- P2 (reproduction): holds. TT is INVALID.
- P3 (mechanism): holds. tT's CI high is .047 <= .075, so true centring removes the inflation.
- P4a: holds. TT's mean z_bias is .090 > 0, and its CI excludes 0.
- P4b: holds. TT's top-minus-bottom tercile difference is .206, and its CI excludes 0.
- P4c: fails. |tT mean z_bias| is .448, which is 5.0 times TT's (the requirement was < 1/4).
- **Reading (not declared).** c_t is the same in both centring arms. tT's w_t is TT's minus c_t, so the difference
  of the two means estimates sum c_t^2 / sd, about .54 (taking sd as unchanged).
  - That .54 is the shift the hypothesis predicted for TT. TT shows only .09.
  - So told-law centring is the cause (P3), but the declared first-order term z_bias does not measure the size of the
    effect: it is too small in TT and not near 0 in tT.
  - Within tT, rejections still concentrate in high-z_bias candidates, though the overall rate is valid.

**Suggested Discussion sentence.** "A post hoc ablation on 200 fresh E1 datasets traced PMRT-GBM's size inflation
under a twice-too-wide told dither to how its statistic is centred.
- With the told law used for both centring and redraws, the truth-null rejection rate was 0.079 (95 % CI
  0.068-0.090).
- Centring on the true law alone brought it to 0.038 (0.031-0.047).
- PMRT-Lin, whose linear statistic has the same centre under both laws, only became conservative.
- The first-order bias term we derived did not account for the size of the inflation, so the exact pathway remains
  open."

Confidence:
- **High** for "centring, not the randomization null": P2 and P3 have CIs clear of both bounds, and the redraw
  sanity check holds.
- **Low** for any quantitative bias mechanism: P4c failed.
- Scope is one cell (E1 R2 n 1000).

## X5: PMRT-Lin's E4 R3 C3 failures: chance or real excess? (advisor question B)

**What was run.**
- **Step 1** (EVAL records, read only; `diag_x5_step1.json`).
  - Every E4 R3 cell, pmrt_eq and pmrt_r3, P_placebo and P_placebo_conf -> K0.
  - Reported: uniformity (KS, QQ), tail o/e, seed overlap, and the label count by chance.
- **Step 2** (fresh datasets; seeds 3_300_000-3_301_999, common random numbers; VPS, frozen arms):
  - x5_fail: the 3 failing cells x 2 arms x 2000 seeds = 12 000 units. Job `xm-diag-x5f`, commit 70468c6,
    16:35:40Z-20:40Z. All 7 processes returned 0. 12 000 / 12 000 ok; 0 missing, errors, duplicates or conflicts.
  - x5_adj: the 5 adjacent cells x 1000 seeds (3_300_000-3_300_999). It was trimmed by the orchestrator's pre-run
    amendment, which was declared before any X5 run. Job `xm-diag-x5a`, commit 264d351, started 20:42:23Z
    (projected about 4.3 h, ETA about 01:00Z). Results: see below.
- **Step 3**: pmrt_r3 on the same datasets, exact McNemar, read only if a primary pair is EXCESS.

**Step 1 numbers.**
- The EVAL rates reproduce: .080 / .083 / .083.
- pmrt_eq has 3 INVALID labels among its 40 E4 R3 C3 rates.
  - The declared approximation (rates independent; P(INVALID | rate exactly .05) = .0099) gives P(>= 3) = .007. It
    is liberal because the lambdas share seeds.
  - Not declared: the frozen dependence-aware F_max null (fmax_simulate, results/fmax_sim/dependence.json) gives
    F_max 2 and P(>= 3) = .037.
- pmrt_r3 has 1 INVALID label (P = .33).
- The same seeds reject across lambdas. At n 24000, P_placebo's extreme tail is heavy at every lambda (rejection o/e
  at .01: 2 to 3.3).
- Step 1 decides nothing.

**Step 2 numbers: primary** (pmrt_eq; one-sided exact binomial vs .05; Holm over 3).

| cell | rate | hits/n | rate | CP 95 % | p one-sided | p Holm | outcome |
|---|---|---|---|---|---|---|---|
| lambda 0, n 8000 | P_placebo_conf | 105/2000 | .052 | [.043, .063] | .318 | .657 | CHANCE |
| lambda .5, n 8000 | P_placebo_conf | 108/2000 | .054 | [.045, .065] | .219 | .657 | CHANCE |
| lambda 1, n 24000 | P_placebo | 103/2000 | .051 | [.042, .062] | .393 | .657 | CHANCE |

- The other rates in these cells (CP 95 %) are all R-30 VALID:
  - pmrt_eq: P_placebo .041 / .038 at n 8000; P_placebo_conf .051 at n 24000.
  - pmrt_r3: .039, .051, .040, .051, .051, .050.
- Step 3 is not triggered, because nothing is EXCESS. For the record, the discordant pairs (eq only / r3 only) are
  7 / 3, 13 / 7 and 6 / 5, with McNemar p .34, .26 and 1.0.

**Declared outcome: all three CHANCE, so the answer to B is "chance".**
- Each CP upper bound is <= .065, under the .075 VALID bound. No Holm p is < .05.
- The fresh data rule out an excess above about 1.5 points at these settings.
- The frozen C3 verdict stands.

**x5_adj (secondary, descriptive) and Linux provenance: PENDING.**
- The dataset-hash provenance check run on Windows differs from the records. The records were generated on the VPS
  (Linux), so the check must run on Linux: on the VPS after x5_adj, since one cdd-xm job runs at a time.
- X6's E1 hashes matched on Windows. Until the Linux check passes, step 2's provenance is not confirmed.

**Suggested Discussion sentence.** "Re-running, on 2000 fresh datasets each, the three E4 R3 cells in which
PMRT-Lin's measured size had been labelled INVALID gave rates of 0.052, 0.054 and 0.051 (95 % CIs within
0.042-0.065; Holm-adjusted p = 0.66). So those labels are consistent with chance among the 40 E4 R3 rates, not with a
size distortion."

Confidence:
- **High** that there is no excess above about .015 in these three cells.
- **Moderate** for "chance": step 1's dependence-aware P(>= 3 labels) was .037, and the heavy extreme tail at n 24000
  is unexplained.
- Provenance confirmation is pending.
