# X7: are the associational referees the static subset rule sub:ES+PowerES? (POST HOC, DESCRIPTIVE)

Advisor question D. Declared in `scratchpad/xmethod/EXTRAS_PROTOCOL.md`, Amendments 2026-10-05 X7 (declaration commit
9a03f9c), before any X7 episode. X7 was added after the Study 3 results were known.
- It tests no hypothesis and has no criterion. No verdict.
- Study 3's verdict and every frozen file are unchanged.
- Code: `scratchpad/e6_dev/x7_subset_check.py` (runs at c2b1d0c; analysis at c332cec), a new file that imports the
  frozen certsafe driver and edits nothing. Test: `tests/test_x7_subset_check.py`.
- Numbers: `x7/x7_tables.json`. Records: `x7/records/`. Provenance: `x7/provenance_frozen_vps.json`,
  `x7/provenance/`.

**Arms.**
- Referees: the four frozen certsafe-artifact arms CS:granger@dev, CS:granger_by, CS:two_tower@dev and
  CS:shap_gbdt@dev.
- Reference: the Gate A anchor sub:ES+PowerES.
- Seeds and plant: the 160 Study 3 EVAL seeds 191100-191259 and the Study 3 plant (P3, L40, 120 s warm-up + 600 s
  scored).

**Run.**
- VPS job xm-x7-v1 (AMD EPYC-Rome; capped systemd scope with 7 processes), 2026-10-05 13:49-16:30Z.
- Python 3.12.14, numpy 2.4.2, scipy 1.18.1, glibc 2.39. Pin check: 0 mismatches.
- 2080 / 2080 jobs, 2080 unique keys, 0 missing, 0 bad lines; rc 0 on all 7 parts.

**Scoring.**
- The Study 3 code, unchanged: `e6p_conf_analyze.arm_stats_multi` on the 26 stored arms plus the X7 arms.
- One paired seed bootstrap: N 10 000, `default_rng([6624, 20, 160])`. dV uses the same index matrix.
- All stored arms reproduce `cs_eval.json`.
- Agreement CIs: seed-cluster percentile bootstrap, 2000 reps, `default_rng([20261005, 7])`.

**Cost.**
- 18.53 VPS CPU-h, plus 0.04 for the provenance check.
- That is **47.5 [35.5, 52.1]** Kaggle-reference CPU-h with `factors.json` "*" (2.56 [1.91, 2.81]).

## Deviation: the re-runs are not bit-identical to the stored records on every seed

The declaration required the reference-stream run and the own-stream re-runs to be bit-identical to the stored
(Kaggle) records. That holds on most seeds but not all.

| VPS re-run vs stored Kaggle record | identical seeds | differing fields (max abs difference) |
|---|---|---|
| X7:sub\|shadow vs sub:ES+PowerES | 136 / 160 | embb_viol (11), prot_viol (6), psvr (1.5), st_req (4), energy_j (6e-9) |
| X7:granger@dev\|own vs CS:granger@dev | 139 / 160 | embb_viol (7), prot_viol (5), psvr (0.97), st_req (3) |
| X7:granger_by\|own vs CS:granger_by | 138 / 160 | embb_viol (20), prot_viol (6), psvr (0.97), energy_j (2.3) |
| X7:two_tower@dev\|own vs CS:two_tower@dev | 139 / 160 | embb_viol (7), prot_viol (5), psvr (0.97), st_req (3) |
| X7:shap_gbdt@dev\|own vs CS:shap_gbdt@dev | 130 / 160 | embb_viol (12), prot_viol (5), psvr (0.97), st_req (4) |

The cause is platform numerics, not the X7 wrapper:
1. **The frozen driver alone reproduces the VPS values.** Five episodes on differing seeds were re-run on the VPS with
   the frozen `e6p_certsafe.arm_job`, with no X7 code in the loop (job xm-x7-prov):
   - sub:ES+PowerES on 191104, 191106 and 191145;
   - CS:granger@dev on 191104;
   - CS:shap_gbdt@dev on 191108.

   All 5 are bit-identical to the X7 VPS re-runs. All 5 differ from Kaggle in the same fields (e.g. psvr on 191104:
   646.645 on the VPS vs 647.032 on Kaggle).
2. **Differences cluster by seed, across arms.** Seeds 191104, 191106, 191126, 191139 and 191148 differ for all arms
   that share a trajectory.
3. **The relation to the reference is platform-independent.** "Referee record == sub:ES+PowerES record" holds on
   exactly the same seeds on both platforms (160 / 160 per referee): granger@dev 117, two_tower@dev 109,
   granger_by 65, shap_gbdt@dev 11.
4. **The effect on the scores is negligible.** VPS minus stored: dR within [-0.0004, +0.0003] and dV within
   [-0.02, +0.02] (90 % CI) for all five pairs.

**Handling.**
- Same-platform comparisons are reported alongside the declared ones. The A3 replays are compared with the VPS
  re-runs (`diff_same_platform`), so no paired difference mixes platforms.
- **X4 note:** X4's 4 provenance pairs were identical, but this shows VPS ≠ Kaggle on about 15 % of seeds by a few
  violations. So X4's "not cross-platform" sentence overstates. The size (|dR| about 1e-4) changes no X4 number at
  the reported precision.

## A1. Code check

**Same request semantics.**
- Both the static rule and the referees emit decision "reject"; neither uses "defer" (st_def 0 in every record).
- The env NACKs the xApp (`x.result(r, False, ...)`) and drops the request (`cdd_oran/envs/e6/env.py:249-254`).
- Only "defer" re-queues, for up to 10 s (`env.py:240-248`, re-offered at `env.py:193`).
- Re-submission is the xApp's own per-seed NACK behaviour, identical for both (`cdd_oran/envs/e6/xapps.py:53-60`):
  retry (re-propose next cycle), escalate (keep integrating the target) or hold (skip one cadence,
  `xapps.py:39-43`).

**Static rule.**
- sub:ES+PowerES = `baselines.subset({"ES", "PowerES"})` (`cdd_oran/envs/e6/baselines.py:185-197`, via
  `scratchpad/e6_dev/e6p_screen.py:281-282` and `e6p_step2_dev.py:237`).
- It rejects every request of the xApp not kept, SliceGuarantee: stateless, per request, from t = 0.
- In P3, SliceGuarantee is the only prot_min proposer (`cdd_oran/envs/e6/xapps_p.py:162-195`). ES proposes sleep and
  carrier, PowerES ptx (`xapps_p.py:52-125`).
- **So the rule is "reject every prot_min request".**

**Referees.**
- `DirectionalUnitArbiter(CertSafeMapGateV2)` (`cdd_oran/decision/certsafe.py:149-155`;
  `cdd_oran/decision/mapgate.py:200-219`; `cdd_oran/decision/units_p.py:221-265`).
- Active from t = 0.
- A (cell, xApp) unit opens on the first request and fixes accept or reject, from the map's (family, direction)
  class and the pressure state at opening.
- The decision holds for T = 60 s, for same-direction requests only; the opposite direction passes
  (`mapgate.py:208-218`). A dwell-blocked request is accepted and opens no unit (`units_p.py:236-238`).
- Frozen decision tables:
  - granger@dev: prot_min+ defer.
  - granger_by: prot_min+ defer iff own / neighbour pressure > theta.
  - two_tower@dev, shap_gbdt@dev: prot_min+ defer iff own pressure > theta.
  - All four: prot_min- accept. Conditional carrier- deferrals; granger_by also conditional sleep+.

**Difference in mechanism.** The static rule decides per request, by xApp identity. The referees decide per
60-s directional unit, by (family, direction) and pressure. In P3 both reduce to "reject prot_min increases" almost
always, because SliceGuarantee proposes almost only prot_min increases while they are being rejected.

## A2. Per-request agreement (accept vs reject; 90 % CI; n = requests)

### (i) Reference stream (primary)

sub:ES+PowerES is applied and each referee is queried in shadow on the same requests. Scored window (t >= 120 s):

| referee | overall | prot_min + | ptx | carrier | sleep |
|---|---|---|---|---|---|
| granger@dev | 0.9999 [0.9999, 1.0000] (n 113 817) | 1.0000 [1.0000, 1.0000] (n 99 785) | 1.0000 (n 5459) | 0.9989 [0.9983, 0.9995] (n 6476) | 1.0000 (n 2097) |
| granger_by | 0.9994 [0.9993, 0.9996] | 1.0000 [0.9999, 1.0000] | 1.0000 | 0.9989 [0.9983, 0.9995] | 0.9742 [0.9689, 0.9792] |
| two_tower@dev | 0.9994 [0.9990, 0.9997] | 0.9993 [0.9989, 0.9997] | 1.0000 | 0.9989 [0.9983, 0.9995] | 1.0000 |
| shap_gbdt@dev | 0.9964 [0.9958, 0.9968] | 0.9993 [0.9989, 0.9997] | 1.0000 | 0.9463 [0.9409, 0.9512] | 1.0000 |

- No prot_min- request occurs in this stream: SliceGuarantee proposes only increases while it is rejected.
- **All t** (n 130 328): overall agreement 0.9995 (granger@dev), 0.9983 (granger_by), 0.9990 (two_tower@dev),
  0.9914 (shap_gbdt@dev). prot_min+ is 0.9993-1.0000 for all four.

Where they disagree, scored window (applied / shadow; a = accept, r = reject):

| referee | prot_min+ (sub r, referee a) | carrier- (sub a, referee r) | sleep+ (sub a, referee r) | ptx, carrier+, sleep- |
|---|---|---|---|---|
| granger@dev | 0 | 7 | 0 | 0 |
| granger_by | 2 | 7 | 54 | 0 |
| two_tower@dev | 66 | 7 | 0 | 0 |
| shap_gbdt@dev | 66 | 348 | 0 | 0 |

### (ii) Own stream (secondary)

Each referee is applied, with the subset rule in shadow. Scored window:

| referee | overall | prot_min + | carrier | sleep |
|---|---|---|---|---|
| granger@dev | 0.9996 [0.9994, 0.9998] (n 113 590) | 1.0000 [1.0000, 1.0000] | 0.9933 [0.9890, 0.9971] | 1.0000 |
| granger_by | 0.9950 [0.9938, 0.9960] (n 110 927) | 1.0000 [1.0000, 1.0000] | 0.9897 [0.9844, 0.9942] | 0.7478 [0.7017, 0.7926] |
| two_tower@dev | 0.9991 [0.9986, 0.9995] (n 113 393) | 0.9994 [0.9990, 0.9997] | 0.9942 [0.9904, 0.9976] | 1.0000 |
| shap_gbdt@dev | 0.9361 [0.9257, 0.9456] (n 115 312) | 0.9992 [0.9988, 0.9996] | 0.3844 [0.3416, 0.4334] | 1.0000 |

- ptx agreement is 1.0000 for all four.
- On its own stream, shap_gbdt@dev rejects 7283 carrier- requests the subset rule accepts; granger_by rejects 493
  sleep+ requests.
- Rejecting carrier-down requests makes ES re-propose them, which inflates their count: shap's own stream has
  11 831 carrier requests, against 6476 in the reference stream.

## A3. Attribution replays and A4. Paired differences vs sub:ES+PowerES

- `|pm` keeps only the referee's prot_min deferrals; every other deferral becomes accept.
- `|other` keeps every deferral except prot_min.
- Referee and reference rows are the stored (Kaggle) records; replays are VPS.
- Guard ratios are vs accept-all. Eligible = retention >= .90 and every guard ratio <= 1.10.

| arm | V | R [90 % CI] | retention [90 % CI] | svr | nonprot eMBB | LL | RLF | eligible |
|---|---|---|---|---|---|---|---|---|
| sub:ES+PowerES | 824.80 | -11.106 [-12.779, -9.815] | 1.000 [1.000, 1.000] | 1.03 | 0.89 | 0.51 | 0.99 | yes |
| noarb (accept-all) | 175.30 | +0.000 | 0.995 [0.990, 1.000] | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| CS:granger@dev | 821.95 | -11.057 [-12.726, -9.780] | 1.003 [0.999, 1.008] | 1.03 | 0.88 | 0.50 | 0.99 | yes |
| X7:granger@dev\|pm | 824.80 | -11.106 [-12.779, -9.815] | 1.000 [1.000, 1.000] | 1.03 | 0.89 | 0.51 | 0.99 | yes |
| X7:granger@dev\|other | 175.85 | -0.009 [-0.049, +0.024] | 0.996 [0.990, 1.003] | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| CS:granger_by | 792.33 | -10.551 [-12.176, -9.286] | 1.032 [1.022, 1.043] | 1.00 | 0.86 | 0.49 | 0.97 | yes |
| X7:granger_by\|pm | 824.56 | -11.102 [-12.773, -9.814] | 1.000 [1.000, 1.000] | 1.03 | 0.89 | 0.51 | 0.99 | yes |
| X7:granger_by\|other | 171.16 | +0.071 [+0.012, +0.133] | 1.028 [1.017, 1.040] | 0.97 | 0.97 | 0.97 | 0.97 | yes |
| CS:two_tower@dev | 819.96 | -11.023 [-12.683, -9.745] | 1.003 [0.999, 1.008] | 1.03 | 0.88 | 0.50 | 0.99 | yes |
| X7:two_tower@dev\|pm | 822.81 | -11.072 [-12.735, -9.786] | 1.000 [1.000, 1.000] | 1.03 | 0.89 | 0.51 | 0.99 | yes |
| X7:two_tower@dev\|other | 175.85 | -0.009 [-0.049, +0.024] | 0.996 [0.990, 1.003] | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| CS:shap_gbdt@dev | 803.49 | -10.742 [-12.407, -9.474] | 0.891 [0.874, 0.908] | 1.01 | 0.87 | 0.50 | 0.99 | **no** |
| X7:shap_gbdt@dev\|pm | 822.81 | -11.072 [-12.735, -9.786] | 1.000 [1.000, 1.000] | 1.03 | 0.89 | 0.51 | 0.99 | yes |
| X7:shap_gbdt@dev\|other | 172.10 | +0.055 [-0.009, +0.113] | 0.930 [0.916, 0.944] | 0.98 | 0.98 | 0.98 | 0.99 | yes |

R* equals R for every row. Two pairs of replays coincide because their tables share the deciding rule:
- two_tower@dev|pm and shap_gbdt@dev|pm: the same prot_min rule;
- granger@dev|other and two_tower@dev|other: the same carrier- rule.

**Paired differences** (arm minus sub:ES+PowerES, 90 % CI). "Identical seeds" = the arm's record equals the
reference in every outcome field.

| arm | dR | dV | identical seeds | per-seed dpsvr: mean (sd); q05 / q50 / q95 |
|---|---|---|---|---|
| CS:granger@dev | +0.049 [-0.016, +0.127] | -2.85 [-7.34, +0.91] | 117 | -2.5 (26.9); -23.2 / 0.0 / +23.4 |
| CS:two_tower@dev | +0.083 [+0.011, +0.166] | -4.84 [-9.64, -0.62] | 109 | -4.7 (31.1); -31.0 / 0.0 / +23.4 |
| CS:shap_gbdt@dev | +0.364 [+0.179, +0.560] | -21.31 [-32.57, -10.31] | 11 | -21.3 (81.4); -162.0 / -5.3 / +79.0 |
| CS:granger_by | +0.555 [+0.389, +0.740] | -32.47 [-43.30, -22.64] | 65 | -32.0 (75.7); -147.3 / 0.0 / +22.8 |

Same-platform replays (VPS) vs the VPS re-run of sub:ES+PowerES (X7:sub|shadow):

| arm | dR | dV | identical seeds |
|---|---|---|---|
| X7:granger@dev\|pm | +0.000 [+0.000, +0.000] | 0.00 [0.00, 0.00] | 160 |
| X7:granger_by\|pm | +0.004 [+0.000, +0.012] | -0.24 [-0.72, 0.00] | 159 |
| X7:two_tower@dev\|pm, X7:shap_gbdt@dev\|pm | +0.034 [+0.007, +0.069] | -1.99 [-3.94, -0.43] | 151 |
| X7:granger@dev\|other, X7:two_tower@dev\|other | +11.096 [+9.807, +12.762] | -648.9 [-680.1, -617.4] | 0 |
| X7:granger_by\|other | +11.177 [+9.894, +12.845] | -653.6 [-684.8, -622.0] | 0 |
| X7:shap_gbdt@dev\|other | +11.161 [+9.872, +12.824] | -652.7 [-684.2, -620.9] | 0 |

The declared comparisons (replays vs the stored sub:ES+PowerES) give the same numbers to the third decimal
(`diff_vs_ref` in the JSON).

Replay minus the referee's own VPS run:
- |pm minus own:
  - -0.049 [-0.127, +0.016] for granger@dev and two_tower@dev;
  - -0.330 [-0.521, -0.147] for shap_gbdt@dev;
  - -0.551 [-0.734, -0.385] for granger_by.
- |other minus own: +10.6 to +11.0.

## Reading (descriptive)

1. **On requests.** On the reference's request stream, the four referees make the same accept / reject decision as
   sub:ES+PowerES on 99.64-99.99 % of scored-window requests, and on 99.93-100 % of prot_min increases (which are
   88 % of all requests).
   - The rest are carrier-down and, for granger_by, sleep-up deferrals that the subset rule never makes, plus 66
     pressure-gated prot_min accepts (two_tower, shap).
   - On their own streams, agreement stays >= 99.5 % except for shap_gbdt@dev (93.6 %). Its carrier-down deferrals
     make ES re-propose, and inflate the disagreeing requests.
2. **On outcomes.** Keeping only the referees' prot_min deferrals reproduces the static rule:
   - bit-identical on 160 / 160 seeds (granger@dev), 159 / 160 (granger_by), 151 / 160 (two_tower, shap);
   - dR <= +0.034 on R about -11.1.

   Dropping the prot_min deferrals gives R near accept-all (-0.009 to +0.071). So virtually all of the referees'
   deficit vs accept-all (R -10.55 to -11.06) is the static rule's prot_min deferral.
3. **On what differs.** The referees' small advantage over sub:ES+PowerES comes from their non-prot_min deferrals,
   which are the same ones that separate them from the rule:
   - two_tower@dev: dR +0.08 [+0.01, +0.17];
   - shap_gbdt@dev: +0.36 [+0.18, +0.56];
   - granger_by: +0.56 [+0.39, +0.74];
   - granger@dev: +0.05 [-0.02, +0.13], not separated from 0; 117 of 160 seeds bit-identical to the rule.

   The advantage is within 5 % of the deficit. shap_gbdt@dev's carrier-down deferrals also cost energy: retention
   0.891, ineligible.

## Suggested Discussion sentence

"In the E6-P plant the four associational referees behaved, descriptively, as the static Gate A anchor that rejects
every SliceGuarantee (prot_min) request: they matched its decision on 99.6-99.99 % of scored requests, and
replaying only their prot_min deferrals reproduced its outcome exactly on 151-160 of 160 seeds (dR <= 0.03 on
R ≈ -11), so their large negative R reflects that one deferral rule rather than a distinct learned policy."

## Confidence

- **High** that the four referees are, in effect, the subset rule sub:ES+PowerES in this plant. The agreement is
  exact or near exact, with tight CIs, and the attribution replays reproduce the rule bit for bit on most seeds,
  within one platform.
- **Moderate** on the size of the small residual advantages (dR +0.05 to +0.56): these are one bootstrap on 160
  seeds.
- **Scope.** Only Study 3's P3 / L40 plant and these four frozen tables. In a plant where another xApp also
  proposes prot_min, the two rules would no longer coincide.
- The cross-platform deviation above does not affect any conclusion (|dR| <= 4e-4).
