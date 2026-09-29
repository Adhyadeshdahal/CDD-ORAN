# Morning brief 2: overnight 2026-09-29 (E6-P Gate A v2 screen)

**Status: FINAL for the night (07:15 NST).** All results are DEV only. Only the xTRUCE screen protocol was frozen (f08747e, as you authorised); no
confirmatory or TEST runs have been made.

## TL;DR (07:15 NST)
1. **E6-P: DEAD, preregistered.**
   - The loss is real: three L40 cells have Λ 0.72-1.55.
   - No static rule and no QACM recovers any of it (R_static ≈ 0).
   - The oracle recovers 38-88 %, but only by breaking guardrails.
   - Its per-decision advantage keeps its sign on a redrawn future only 55-66 % of the time.
2. **xTRUCE XTS-v1: NOT SCREENABLE.** QoS and LB fail M4. I built the plant, froze the protocol and ran stage 0.
   v2 needs you.
3. **Three agents argued the strategy** (`DEBATE_2026-09-29.md`):
   - **A:** noise is fundamental, so pivot to regime-level envelopes and attribution.
   - **B:** the oracle and estimator are flawed, so run E6-P v2 with a guardrail-constrained, expectation-selecting
     oracle.
   - **C:** publish the negative results and Gate A v2 as a benchmark paper.

   **B's own numbers argue against B.** Scaling raw R by the share of advantage that survives a redraw gives about
   0.18 / 0.19 / 0.38 < 0.50. Even a perfect world model likely cannot pass c2 at per-decision scale.
4. **My recommendation:**
   - **Now: C.** Write the benchmark/negative-result paper. Nearly all tables are derivable from runs already on
     disk.
   - **Optionally**, if you want a stronger C2 claim or a last shot at the edge, run B's cheap falsifier: O_tape only,
     guardrail-constrained, fresh seeds, about 36 CPU-h, preregistered as v2.
   - **Treat A** (regime envelope) as the *only* remaining edge hypothesis. First check cheaply whether any eligible
     static envelope exists: A's own counter is that it may reduce to "PACIFISTA with better knobs".
   - **Do not** build the learned per-decision arbiter. Its L0 infrastructure is committed (c30f893), but it is
     parked.

**Decisions only you can make:**
- (a) whether to write XTS-v2 (M4 on h0 strata only, which screens X1 and X2), costing < 5 CPU-h;
- (b) whether to write E6-P v2 as O_tape only (~36 CPU-h) or in full (~170 CPU-h);
- (c) whether to pivot to A or C;
- (d) how to reconcile the Discover journal draft, whose thesis this contradicts.

## VERDICT: E6-P Gate A v2 = DEAD for P1 and P3 (P2 not screenable). Stage 3 was not run.
- **Stage 2:** in all three cells where the loss criterion passed, the oracle is **ineligible** because it breaks a
  guardrail versus accept-all:

  | cell | failed guardrail | vs accept-all (limit 1.10×) |
  |---|---|---|
  | P1 s3 | LL violations | 1.107× |
  | P3 s1 | RLF | 1.22× |
  | P3 s3 | RLF | 1.39× |

  So R_or = 0 and c2 fails.
- **ρ_sign is also below 0.70 everywhere** (0.55 / 0.61 / 0.66). c4 would have failed even with an eligible oracle:
  the oracle's advantage is still largely tape-specific, the same failure mode as E6.
- **Diagnostic only:** raw oracle recovery was 0.38 / 0.88 / 0.65, at energy retention 1.14 / 1.36 / 1.23. The
  oracle over-sleeps because its frozen objective ignores the guardrails. I did not amend anything after seeing the
  outcome.
- **Next, per your instruction:** the xTRUCE screen, which ran and is reported below. The one lesson carried over,
  before any xTRUCE outcome and disclosed in its protocol, is that the oracle's admissibility must include the
  guardrails.

## VERDICT: xTRUCE screen (XTS-v1, frozen f08747e) = NOT SCREENABLE. No pair can be screened.
- **Stage 0, mechanism checks.** M1-M3 and M5-M8 pass. M4 (each xApp alone improves its own KPI vs freeze) fails
  for two xApps:
  - **QoS** fails at L6-h50 (hallucination 0.5): pooled V improves 984 → 816, but only 2 of 5 seeds improve, and
    the rule needs 4 of 5. It passes at L6-h0 (5/5), L1-h0 (4/5) and L1-h50 (4/5).
  - **LB** makes its own KPI (max demand load) worse at L6: 1 of 5 and 0 of 5 seeds improve, and at L1-h0 only 3 of 5. Its rule is one of our
    assumptions, because the paper publishes none.
- **Consequence.** QoS is in every pair (X1-X4), so all pairs are "not screenable under XTS-v1" by the frozen rule.
  Changing a value means a new version, XTS-v2, with fresh seeds.
- **Why I stopped.** That amendment is outcome-informed, like the E6-P M4 amendment you approved, so I did **not**
  run it without you.
- **The natural v2, for your decision.** Evaluate M4 only in the h0 strata, where hallucination is off (a
  hallucinating xApp is *meant* to be unreliable). LB stays failed, so X3 and X4 drop out and X1 = QoS+ES and
  X2 = +IC get screened.
- **Cost.** Very cheap: the oracle takes 82 s per episode on this plant, and the whole screen needs less than
  ~5 CPU-h.
- **Caveat.** The plant's xApp rules are our assumptions, because the paper does not publish them. So its evidential
  value against the published xTRUCE is limited either way.

## Headline of stage 1
- **Stage 1: criterion 1 (real co-deployment loss at matched energy) PASSES in 3 of 8 pair-strata.** All three are
  at L40:

  | pair | stratum | V_AA | V_ref (SG alone) | Λ | LB90(V_AA−V_ref) | R_static |
  |---|---|---|---|---|---|---|
  | P1 ES×SG | s3 surge-L40 | 92.2 | 53.5 | 0.72 | 25.2 | 0.00 |
  | P3 ES×PowerES×SG | s1 base-L40 | 149.2 | 79.1 | 0.89 | 30.0 | 0.09 |
  | P3 ES×PowerES×SG | s3 surge-L40 | 135.8 | 53.2 | 1.55 | 55.9 | 0.01 |

- **No static arm recovers the loss.** Every static priority order and the lock arm are *identical* to accept-all
  (same V, same churn). Each knob has exactly one writer, so there is no direct same-knob conflict to arbitrate and
  the loss is purely **indirect**. This is the regime the scouts said an edge needs. QACM ≈ AA too (R 0.008).
- The one static option that helps is dropping PowerES (P3 sub:ES+SG, R ≈ 0.53-0.58), but it gives up the energy
  saving (retention 0.49-0.84 < 0.90), so it is not eligible.
- **L10 strata fail c1.** The loss is real in relative terms (Λ 0.45-2.6) but small in absolute terms (V < 33
  protected UE-s/h): below the 36 materiality floor (Λ ≥ 0.34 and LB > 0 everywhere; only the absolute floor fails; corrected by the report).
- **Stage 2** (oracle, 24 episodes: 8 seeds × 3 cells) runs on Kaggle; each episode is about 1.26 CPU-h. See below.

## Stage log
- **0a:** load_factor L10 = 0.7875, L40 = 1.755.
- **0b:** ALL PASS after the M4 amendment you approved. P2 is not screenable because Coverage is inert.
- **0c:** the oracle pilot takes 4528 s per episode (947 rollouts, 24 deviations).
- **1:** 992 jobs on 4 Kaggle kernels. The results are the headline above.
- **2:** e6p-s2 ran 24 oracle episodes on 5 kernels, from 01:55 to 06:25. The verdict is above.
- **3:** not run, because no cell passed c1 and c2.

## Driver fixes made overnight (plumbing only, no protocol value changed)
- `units_for` stage 1 now skips `not_screenable_pairs`. Before, it would have run P2 anyway.
- `summary`'s stage1_done check ignores inert pairs. Before, it never wrote `stage1_crit1`.

## Built in parallel (hedges, all DEV)
- **xTRUCE plant**, commit 4dae03a: `cdd_oran/envs/xtruce/`, 19 tests, ~2.5k epochs/s.
  - Our notes were partly wrong. The paper's xApps post KPI *targets*, not knob writes. Path loss is TR 36.814. Its
    baselines are Direct and Clipping, and there is no static priority. The xApp rules are unpublished, so 13 values
    are unsourced and qos_margin was tuned on seeds 0-2.
  - Spec: `docs/benchmark/XTRUCE_SIM_SPEC.md`.
- **xTRUCE Gate A v2 protocol and driver**, NOT frozen: `docs/benchmark/XTRUCE_SCREEN_PROTOCOL.md` and
  `scratchpad/e6_dev/xtruce_screen.py`.
  - Estimated cost 4-17 CPU-h. It runs only if E6-P fails, per your instruction.
  - I changed the draft's V_ref to E6-P's best-single definition before freeze. The draft's version would have
    required R_or ≥ 1.10.
- **Learned-arbiter design:** `scratchpad/e6_dev/decision/ARBITER_DESIGN.md`. It is a structural world model
  (known capacity and energy mechanisms plus a learned monotone hurdle model), with a DR bandit ablation, a CRT
  exposure map and MSCR features.
- **Arbiter L0 infrastructure**, commit c30f893: units, randomized π0 collection with propensities, CRN rollout
  labels, and IPS/SNIPS/DR estimators, with 25 tests. Driver: `scratchpad/e6_dev/e6p_arbiter_pilot.py`.

## Things you should know
- **The memory directory was found emptied at session start.** Only the colab-api-access and next-session-handoff
  notes exist now, and I rewrote them. Anything else you expected to find there is gone.
- **The oracle's "rollback" does nothing on ES.** The rollback window is 60 s but the carrier/sleep dwell is 120 s, so
  the oracle is somewhat conservative.
- **`half` equals a toggle on single-quantum knobs.** Only PowerES has a true half step.
- **Decisions waiting for you:**
  - freezing the arbiter protocol (π0, thresholds, β, seed block 160000-179999 and tags 6612-6615);
  - freezing the xTRUCE screen;
  - any confirmatory run on 150200-150399.
