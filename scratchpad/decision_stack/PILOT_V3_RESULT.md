# v3 pilot result (e6-collect-v3; fit seeds j=0,1 × 6 strata; 12 v3 + 12 paired refs; local, 2026-09-28)
Efficacy NOT inspected. Contract checks (E6_COLLECTION_CONTRACT.md F1–F6):
- F1 trace reconstruction: labels_match_score, changes_match, requests_match, policy_requests_match, config_rebuild
  12/12 (share 1.0). PASS
- F2 propensity_exact 12/12. PASS (pooled branch-frequency binomial check not separately computed)
- F3 first stage (pooled): CIO f=0 → 0.000 realised (n 587), f=.5 → 0.504 (n 285), f=1 → 0.977 (n 2160);
  LL ratio f=0 → 0.000 (579), f=.5 → 0.508 (380), f=1 → 0.995 (2597). PASS
- F4 continuation_ok 12/12 (TrueSimWM accept_all == realised, exact). PASS
- F5 washout_accept_only, rollback_only_at_slot_start, cap_respected, churn_blocked_accounted 12/12; no exceptions. PASS
- F6 refs: non-default share 0 in all; no-arbiter reproduction checked in-job. PASS
- Cost trigger: median paired excess SVR per stratum −0.8 % … +1.2 %; severe within ±4 of refs → no review.
- Eligibility (share of region-slots with requests in the 20 s window): TS .46–.81, SLICE .10–.66, ES .06–.10,
  MRO 0–.03 → MRO/ES expected "unidentified" at the planned budget (predeclared support rule).
- F2 (added per sol sign-off): 840 draws; branch accept/frac/lock/rb 409/341/44/46 (z −0.76/+0.35/+0.32/+0.63);
  per-xApp mode marginals max |z| = 1.22 (< 4). PASS → contract frozen.
