# Request-level effect pilot: predeclared design (2026-09-28, written before any label was computed)

Question: do REQUEST-LEVEL features carry predictable signal about the effect of rejecting ONE xApp request, where
the region-policy features (RANK_DIAG_RESULT.md: CV R2 <= 0) did not?

Code pin: cdd_oran at commit b00920b (tracked files unchanged; E6Env imports envs/e6/xapps.py, NOT the untracked
xapps_v2.py). numpy/sklearn versions recorded in every output row's run header.

## Episodes (DEV diagnostic policy seeds only)
`collect.dev_seed("policy", scn, load, j)` with j in 20..29 (split "diag", the rank_grid seeds; no calib/eval/TEST).
Order: j = 20, 21, 22, ... ; within j, strata in (base, surge, mistune) x (medium, high) order. Fixed stop: the first
12 episodes (j = 20, 21 x 6 strata). E6Config(seed, load, mobility="mixed", mix="M4", warmup_s=120, scored_s=600,
scenario), E6Env(wg3=True, log=False). Main trajectory = accept-all.

## Decision seconds and requests
Anchors t_k = 150, 250, 350 (k = 0, 1, 2). Decision second = first second s in [t_k, t_k + 10) whose pending requests
include one from an xApp other than SLICE; if none, the first second >= t_k with any pending request.
Requests labelled: all pending if <= 8, else a uniform sample of 8 without replacement,
rng = default_rng([seed, 4401, k]).

## Label
For request r at decision second t: dJ_H(r) = J_H(reject r, accept all else at t, then accept-all) - J_H(accept-all),
both from env.copy() between step_propose and step_apply (common random numbers: same tape), J = world_model.objective
with lam_e = 1e4, w_ll = 5 (rank_eval LAM_E/W_LL), window = the decision second's apply + the following H seconds
(TrueSimWM accept_all semantics with D irrelevant: only one second carries a non-accept decision). One 90 s rollout
per plan; the cumulative SLA counters are snapshotted after H = 10, 30, 90 s. Sign: dJ > 0 = rejecting HURTS.
"Exactly zero": |dJ| < 1e-6.

## Replication (label reliability)
At the k = 0 decision second of every episode, all labelled requests (and the accept-all base) are re-labelled on a
copy whose UE positions are nudged by +1e-6 m (as diag_rank/i_perturb.py). Per H: reliability = 1 - (var(a-b)/2) /
var([a;b]) pooled over pairs (j_reliab.py formula); also Pearson r and sign agreement on pairs non-zero in both.
Known caveat (stated before running): RANK_DIAG found the nudge replication optimistic (r 0.96-0.99) vs a
numpy-version replication (0.56); a high nudge reliability is necessary, not sufficient.

## Features (observable at decision time: obs + RIC-side state)
Request: xApp (one-hot), knob type (one-hot), direction sign(prop - cur), |prop - cur| in actuator quanta, cur value
(normalised per type: ll_ratio as is, cio/10, hys/5, ttt/1000, sleep, carrier/n_trx-ish raw), is_reverse_cio.
Target cell c = knob[1] (source cell for cio; n = knob[2] for cio else c): latest delivered "fast" report
prb_util[c], prb_util_slice LL share[c], prb_rsv_idle[c], ll_delay_p95[c] / LL_DELAY_TARGET_S (nan -> -1 + flag),
act_ue[c], carriers[c]; ll_ratio[c] in force; latest "thp" embb_thp_p5[c] / EMBB_THP_TARGET_BPS; for cio the same
prb_util at n and util difference c - n; mean prb_util over c's neighbours; is_macro[c]; asleep[c].
Time since the knob's last applied change (env.last_change; capped at 600 s); number of pending requests this second
on the same cell c; on the same knob; total pending; direction x ll_delay ratio and direction x prb_util interactions.
Reports tracked by the script from obs["new_reports"] (latest per granularity).

## Analysis (per H in {10, 30, 90})
1. Label distribution: n, fraction exactly 0, sign split, |dJ| quantiles, by xApp x knob type x direction.
2. Reliability (above).
3. Predictability, GroupKFold(5) by episode:
   - sign on non-zero labels: logistic (StandardScaler, C=1) and HGB classifier (max_depth 3, max_iter 100,
     learning_rate 0.05, min_samples_leaf 10); pooled out-of-fold AUC + accuracy; 95% CI by episode bootstrap (1000).
   - BASELINE: logistic on one-hot(xApp x knob type x direction) only.
   - zero vs non-zero AUC (same models), secondary.
   - magnitude: ridge (RidgeCV alphas 1e-2..1e4) and HGB regressor (same hyper-params) on all labels, CV R2;
     baseline ridge on the one-hot cell means.
No hyper-parameter is tuned on outcomes.

## Decision rule
Request-level route PROMISING iff at some H (the same H for both): max(logistic, HGB) CV sign AUC on non-zero labels
>= 0.70 AND nudge reliability >= 0.70 AND >= 30 non-zero labels at that H. Otherwise DEAD for E6.
Reported alongside (not part of the rule): whether full features beat the xApp+direction baseline (if the baseline
alone reaches 0.70, the signal is "direction", which the region-policy unit could not express).

## Execution note (added 2026-09-28, before any label existed; design above unchanged)
Local RAM stayed < 3 GB (other agents), so execution moved to Kaggle: labeller
scratchpad/decision_stack/req_pilot_label.py (same logic as the drafted req_pilot/label.py, which was removed; no
local rollout was ever completed), shard driver scratchpad/decision_stack/req_pilot_grid.py (`run --part i/k`,
episodes e with e % k == i, one jsonl line per (episode, k)), bundled flat into e6dev/ by scratchpad/e6_dev/cloud.py.
The numpy version is recorded per line (Kaggle numpy != local 2.4.2; RANK_DIAG found trajectories are not
bit-reproducible across numpy versions, so every label and its nudge replicate come from the same machine).
analyze.py takes `--in a.jsonl,b.jsonl,...`.
