READY-TO-MERGE
# freeze status (2026-10-04): Study A EVAL freeze, S = 40 (PROTOCOL_A FROZEN; R-59 fixes in; re-audit PASS-WITH-NOTES)

Branch xm/freeze = feat/v2 5a4ef0f + xm/vps-factors + xm/anchor-note + R-59 F1 (853392e, 581317a) + the freeze
commit "xmethod: Study A EVAL freeze, re-hashed after R-59" (its tip is in the orchestrator report). Local only, not
pushed. EVAL NOT launched; no EVAL unit run.

## Freeze (FREEZE_CHECKLIST C)
- C1: xm/dev-runs on feat/v2 (41d6c37); calib block 12748be; VPS smoke 9e91a4c; R-59 0542a99; re-audit 5a4ef0f.
- C2 spec:
  - Seeds: S 40 -> [3100000, 3100039], half [.., 3100019]; E4 R3 readers 300.
  - pkgs_lock = uv.lock versions of campaign.PKGS, torch '2.10.0+cpu' (confirmed); status FROZEN.
  - tbd notes FILLED; `tbd.driver` now says integrity.spec_file_sha256 (re-audit N4).
- C3 PROTOCOL_A:
  - FROZEN: yes (2026-10-04).
  - T1 40; T3 f / e by host type (uncalibrated hosts provisional, accepted); T4 (iii); T5 final REPORT; T6; s.7
    counts.
  - s.11 stamp wording integrity.spec_file_sha256 (R-59 F1, confirmed).
- C4: spec protocol_sha256 = c5f7a4fe...fa22 = LF sha of PROTOCOL_A; campaign.eval_authorised(spec) True.
- C5:
  - Full xmethod test set (14 files, `uv run --group citests`): all pass. Skips: campaign 2 (Linux-only fork tests),
    eval_analysis 1 (the [eval] stamp case needs fork; the re-audit ran it on Linux: pass).
  - `campaign list` on the frozen spec: mode eval, 150 960 units, 11 000 datasets, 40 520 tune.
  - Estimand check (R-36): all 18 cells PASS; its json sha is unchanged (SCM code is unchanged, see the manifest).
- C6 FREEZE_NOTE.md:
  - Every LF sha256 recomputed. Changed: PROTOCOL_A, spec, eval_analysis, eval_report, campaign.py. These equal
    the re-audit's N5 except the spec (N4 text).
  - Every T-register input unchanged. FREEZE_MANIFEST.sha256 = 301 code files (`_ref/` excluded); FREEZE_CALIB.sha256
    = 36 calibration files, unchanged.
  - Both audits with path, LF sha256 and verdict (FAIL -> R-59; re-audit PASS-WITH-NOTES).
  - Re-audit N1 operational rule: every EVAL relaunch passes --skip-complete-from over ALL prior EVAL shards. Before
    eval_analysis, list keys with both an infeasible and an ok record across raw shards (report; none expected).
  - Re-audit N6 noted (test hygiene).
- C7: SEED_REGISTRY: 3100000-3100299 used + claimed_by XMETHOD_EVAL for E1-E5.
- Projection at S 40 with the R-55 factors: 509 CPU-h; ~14 h on all platforms / ~17 h on Kaggle + VPS / ~32 h on
  Kaggle only.

## Next (orchestrator)
- C8: merge (one squash = the freeze commit on feat/v2), then the launch GO. EVAL launches from a clean checkout of
  that commit; records must stamp it, dirty false.
- Never after the freeze: edit PROTOCOL_A (amendments file only), the spec, eval_analysis.py, eval_report.py,
  campaign.py or any adapter.

## Questions
- None open. Earlier ones were answered by the orchestrator: calib block, audits, lane fallback, torch pin,
  uncalibrated hosts, `_ref/`, s.11 wording.
