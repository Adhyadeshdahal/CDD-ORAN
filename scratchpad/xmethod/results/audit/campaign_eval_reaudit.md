# campaign.py EVAL re-audit (R-59 fixes)

**Verdict: PASS-WITH-NOTES.** F1-F9 are fixed as R-59 says, and each fix has a real test. The fixes add no path that
skips or duplicates units, accepts a wrong record, or changes method outputs. DEV behaviour changes only where noted
below. The PROTOCOL_A s.11 edit only renames the stamp field. The notes below are low or info; none blocks EVAL.

- Auditor: worker exp-b, 2026-10-04. The audited code was not edited.
- Target: xm/freeze tip 581317a.
  - Fix commits: 853392e (F1) and 0542a99 (F2-F9, merged into freeze as c91de2c), plus 581317a (F1 test isolated).
  - Baseline: the version I audited, 9e91a4c (campaign.py identical to the freeze e50c2d7).
  - Rulings: CONTRACT R-59 at feat/v2 1524dc2.

## Tests on 581317a (temporary worktree, since removed)

- **Windows:** `tests/test_xmethod_campaign.py` + `tests/test_xmethod_eval_analysis.py` gave 96 passed, 3 skipped.
  The skipped three are Linux-only: the fork / RLIMIT tests and F1's EVAL case.
  - In a fresh checkout `test_vps_launch_steps` and `test_lightning_and_colab_gpu_launch_commands` first failed with
    FileNotFoundError. The cause is the gitignored, generated `scratchpad/xmethod/_bundle/requirements.lock.txt`.
  - After `python -m cdd_oran.xmethod.campaign lock` in the temp worktree, both pass (note N6).
- **Linux:** throwaway Docker container (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, Python 3.12.12, worktree
  mounted read-only, hash-pinned `requirements.cloud.txt`). Both files: 98 passed, 1 skipped (a git-dependent test
  that ran on Windows).
- Every test ran on at least one OS and all passed. This includes
  `test_v11_stamps_pass_on_campaign_written_records[eval]`, whose records come from `campaign.run_units`.

## Findings F1-F9 (fix, test, verdict)

| # | Fix (581317a file:line) | Test | Verdict |
|---|---|---|---|
| F1 | eval_analysis.py:1159-1161 compares `integrity.spec_file_sha256` (LF rule of `file_sha256`); eval_report.py `slim` keeps it; campaign CLI stamps `_sha_lf(spec_bytes)` (campaign.py:1557) | `test_v11_stamps_pass_on_campaign_written_records` [dev, eval]: records from `run_units`, V11 stamps pass, another spec sha fails. Hand-built records in the older tests now mirror campaign's layout | fixed |
| F2 | `_over` (campaign.py:827-) EVAL = `infeasible` with its CPU-s. `done` includes infeasible keys, so it is never re-run. No propagation in EVAL (`over` only read when `not ev`, registry None). OOM / other signals stay errors | `test_r59_f2_eval_cap_breach_is_infeasible_never_rerun_oom_is_error`, `test_eval_rules_cap_is_infeasible_no_propagation_and_t3` | fixed |
| F3 | `accept_rule` (887-): EVAL = mode eval + protocol sha + canonical spec sha; DEV = dev. Used by `complete_dataset_keys` (858-, via `iter_jsonl`, which skips broken lines, so a cut shard is safe) and by `merge` (1026). The dispatcher's slim done files now carry `run_mode` + spec sha (xm_dispatch.py:99-102) | `test_r59_f3_eval_skip_from_applies_the_merge_accept_rule` | fixed |
| F4 | `run_part` stamps `cost_table_sha256` (canonical sha of the table actually used), `part`, `parts` (881-884). `merge` refuses an EVAL merge with >1 cost-table sha or part count, after removing `out` and keeping the summary (1053-1062) | `test_r59_f4_f9_partition_stamps_merge_sets_and_mixed_refusal` | fixed |
| F5 | `_cloud_cmd` (1397-1430): each part in `( ...; echo $? > rc_<i>.txt ) &`; the final check exits 1 if any part is non-zero or has no rc file; vps_run `status` shows rc files | `test_r59_f5_session_exit_nonzero_if_any_part_fails` [gate on/off]: real bash run, a failing part gives non-zero, rc files 0 / 3 / 0 | fixed |
| F6 | EVAL refuses a budget != spec `budget_cpu_s` (campaign.py:727-730); `budget_cpu_s`, `cap_factor` per run and `limits` per record stamped (739-745, 817) | `test_r59_f6_f7_eval_refuses_cli_budget_and_no_isolation` | fixed |
| F7 | EVAL refuses `isolate` false (731-733); `isolation` stamped | same test (+ the F1 eval case runs isolated, Linux) | fixed |
| F8 | vps_run.py `scope_limits(procs)` (56-) gives CPUQuota = procs x 100 %, used by `launch` (297) and recorded in launch.json | `test_r59_f8_vps_scope_cpuquota_follows_procs` | fixed |
| F9 | `MERGE_SETS` (1017) gives python, lock_sha256, spec_file_sha256, isolation, budget_cpu_s, cost_table_sha256 and parts in the merge summary | F4 / F9 test | fixed |
| F10 | no change (R-59) | none | as ruled |

## No new risky paths, and DEV behaviour

- **Unit selection:**
  - F3 only removes records from the skip set, so the change can only cause extra runs, never skips.
  - F4 adds stamps and a merge refusal; partitioning itself is unchanged.
  - F5 changes the shell wrapper only; the semaphore still counts the subshells, and `campaign_procs` (keyed by
    `--out`) is unchanged.
- **Record acceptance:** the EVAL merge is stricter (it now also needs the canonical spec sha). Nothing is accepted
  that was refused before.
- **Method outputs:** untouched. The config, `R.run_one`, dataset generation and isolation (now mandatory in EVAL)
  are the same.
- **DEV behaviour.** The only differences are the ones aud1 noted:
  - extra `integrity` fields and `limits` per record;
  - skip-from ignores EVAL records;
  - the session exit code is non-zero when a part fails (e.g. a DEV `timeout` 124);
  - the merge summary gains the integrity sets.

  `_over` for DEV, the registry and over-n propagation, and the DEV merge accept rule are unchanged. The F6 / F7
  refusals sit inside `if is_eval(spec)`.
- **PROTOCOL_A** (853392e; `git diff e50c2d7 581317a -- docs/xmethod/PROTOCOL_A.md` has one hunk): s.11
  `run_mode.spec_sha256` becomes `integrity.spec_file_sha256 (the spec file's LF sha256; R-59)`, re-wrapped. Nothing
  else changed. `specs/eval/full.json` `protocol_sha256` = c5f7a4fe..., which equals the LF sha of PROTOCOL_A at
  581317a.

## Notes

- **N1 (LOW, pre-existing ranking now relevant to F2).**
  - What happens: `merge_index` ranks ok > infeasible. "Never re-run" is enforced within an output file and through
    `--skip-complete-from`, but not across sessions that run the same dataset independently.
  - Failure scenario: a unit is cap-infeasible on Kaggle (15 000 CPU-s). A relaunch on the VPS without skip-from
    finishes it in ~6 000 CPU-s, because the VPS is faster. The merge selects the dataset's VPS records, so the cell
    becomes feasible.
  - The one-platform rule keeps the dataset consistent, and a CPU-s cap is host-dependent anyway.
  - Suggestion: list keys that have both an infeasible and an ok record in the merge summary.
- **N2 (INFO).** The EVAL merge drops (as "foreign") every record whose canonical spec sha differs. Any edit to
  `full.json` after launch therefore turns earlier records into "missing", which is loud, not silent. Amendments go to
  `amendments.json`, so this should not happen.
- **N3 (INFO).** EVAL `run_units` does not refuse `spec_file_sha256=None`. Only the CLI path (1557) passes it, so a
  direct caller would write records that fail V11, which is also loud.
- **N4 (INFO).** `full.json` `tbd.driver` still says records stamp `run_mode.spec_sha256`. It is descriptive text,
  but inside the frozen spec, and does not match PROTOCOL_A s.11 any more.
- **N5 (INFO; freeze step, not code).** FREEZE_NOTE at 581317a is marked STALE (R-59 re-hash pending). Its campaign,
  eval_analysis, eval_report and full.json values are the pre-fix ones. Current LF sha256 at 581317a:
  - PROTOCOL_A c5f7a4fe4dff4bfdeef3ccebac2dcc3748dfa9c2d0c3d17f2ac9d6246379fa22
  - campaign.py ab87a268ffbce0dc60c87b479ffec9e48dfa6604f5706695313b745189837450
  - eval_analysis.py 73685abb3e539a3125ead531ab63555c0292c6ef04bfd969ac9182035b8278df
  - eval_report.py 8099ca4a43afaa2791c275a45093f5c771124fdcbcb66c7cd3393b9fdd6cb185
  - full.json 5021188acb3fbff01578f45b08b3044dbe240554e4088852979f204eb403e065
  - dev_power.py unchanged, equal to the note.
- **N6 (LOW, test hygiene).** The two launch-command tests need the generated, gitignored lock export, so they fail
  in a fresh checkout until `campaign lock` runs. Generate it in a fixture, or skip with a reason.
- **Test interpreter.** The Linux run used Python 3.12.12 (the container image), not 3.12.14. That is enough for these
  tests, since the EVAL case monkeypatches the interpreter / pin preconditions, which other tests cover.
