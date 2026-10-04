READY-TO-MERGE

# audit-campaign status (2026-10-04): DONE. Verdict FAIL (1 blocker, 1 ruling needed)

Independent audit of `cdd_oran/xmethod/campaign.py` EVAL paths (+ `scratchpad/e6_dev/vps_run.py`), freeze
checklist B6 / PROTOCOL_A s.11 T7. Worker exp-b (branch xm/exp-b, worktree xm-classic), after merging feat/v2
9e91a4c (merge ceeb6bd, not pushed). campaign.py not edited.
Report: scratchpad/xmethod/results/audit/campaign_eval_audit.md.

## Result
- Audited file = the frozen one: LF sha256 b6fc4042... = FREEZE_NOTE (e50c2d7); identical at 9e91a4c.
- Passed:
  - `campaign list` on the frozen spec = 150 960 units / 11 000 datasets / 40 520 tune (PROTOCOL_A s.7);
  - EVAL guard bound to the frozen protocol sha (ae5e4bb5...);
  - Python 3.12.14 and the lock pins are checked before any data;
  - integrity stamps, the R-55 gate and cgroup quota;
  - one platform per dataset at merge;
  - campaign tests pass.
- F1 HIGH (blocker), campaign.py:717 vs eval_analysis.py:1160 / 1784:
  - campaign stamps the spec sha under `integrity` (`spec_file_sha256`), but eval_analysis requires
    `run_mode.spec_sha256`;
  - so every real EVAL record fails V11 "stamps" and the run can never be FINAL;
  - eval_analysis tests build the field by hand, so they miss it;
  - suggested amendment: eval_analysis reads `integrity.spec_file_sha256` (records and campaign sha unchanged).
- F2 MEDIUM-HIGH (ruling):
  - campaign records a 2x safety-cap breach as **error**, which every resume re-runs;
  - PROTOCOL_A s.7 says it is **infeasible, never re-run**;
  - amend s.7 or the code.
- F3-F8, all detected at merge as missing, never accepted wrongly:
  - F3 skip-from not filtered by run mode;
  - F4 part membership depends on an unstamped cost table;
  - F5 session exit 0 even when every process refused;
  - F6 `--budget` overrides EVAL and is unstamped;
  - F7 `--no-isolate` allowed in EVAL;
  - F8 VPS `--procs` not enforced.
- F9-F10: info only.

## Questions: all ANSWERED
- (none open; the F1 / F2 decisions are for the orchestrator, listed in the report)
