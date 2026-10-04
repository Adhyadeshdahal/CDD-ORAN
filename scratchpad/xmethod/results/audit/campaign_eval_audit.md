# campaign.py EVAL audit (freeze checklist B6 / PROTOCOL_A s.11 T7)

**Verdict: FAIL.** There is one blocker (F1): records as campaign writes them can never pass `eval_analysis`'s stamp
check, so every EVAL run would end PROVISIONAL. It is a one-line fix, but both files are frozen, so it needs an
amendment. F2 is a protocol / code disagreement that needs a ruling. Everything else is PASS-WITH-NOTES.

- Auditor: worker exp-b (did not write campaign.py), 2026-10-04.
- Code: branch xm/exp-b after `git merge feat/v2` (9e91a4c, merge ceeb6bd). Not edited.
- Audited files:
  - `cdd_oran/xmethod/campaign.py`, LF sha256 b6fc4042...802c. This equals the freeze note's (e50c2d7), and the
    file is identical at e50c2d7 and 9e91a4c.
  - The VPS launch path `scratchpad/e6_dev/vps_run.py`.
  - Where records are consumed: `runner.run_one` / `_done_keys`, and `eval_analysis.py` at e50c2d7.
- References: PROTOCOL_A as frozen (xm/freeze e50c2d7, worktree xm-harness), whose LF sha256 ae5e4bb5... equals the
  spec's `protocol_sha256`; CONTRACT R-35, R-55, R-57, R-58.

## Checks that passed (evidence)

- **Unit expansion.** `campaign list` on the frozen `specs/eval/full.json` (run read-only in xm-harness) gives 150 960
  units, 11 000 datasets and 40 520 tune units, which equals PROTOCOL_A s.7.
  - The EVAL guard authorised that run against the frozen text.
  - No key collisions: E4 lambdas are 0 / .5 / 1 / 1.5 and kappas .125 / .25 / .5, so the `:g` key format is
    injective here.
  - Tune and measure seeds are disjoint and checked (`validate_spec` 230-240).
  - T3 / R-58 gaps come from block `arms` / `ns` (no `max_n` / `infeasible_n` in the spec), so they are expected
    keys only where planned.
- **EVAL guard** (`eval_authorised` 151-178).
  - Refuses without a 64-hex `protocol_sha256`, and refuses unless the live file says FROZEN.
  - Accepts only a text whose LF sha256 equals the constant and which says FROZEN (live file, freeze-commit blob or
    bundled copy).
  - The launcher's bundled copy (1537-1541) falls back to the live file, which is still sha-checked, so there is no
    bypass.
  - EVAL seeds are refused without authorisation (`_seed_list` 191-194).
  - DEV and EVAL records cannot share an output file (722-725).
  - The merge accepts only `mode == eval` records with the spec's protocol sha (986-990).
- **Interpreter and packages before data (R-57, R-35).** `check_eval_preconditions` (659-673) runs before the first
  `generate_dataset`. It requires:
  - exactly `platform.python_version() == "3.12.14"`;
  - a known, clean commit;
  - a lock export with no pin mismatch (pin_check, python included).

  So no `mode: eval` record can come from another interpreter or package set. The launchers install 3.12.14 in a
  fresh venv (Kaggle / Colab / Lightning: `_setup_cmd` 1399-1404; VPS: `vps_run.venv` 233-253 exits 4 / 5 on another
  patch or a floating home).
- **Integrity stamps per record.** Code commit and dirty flag, protocol sha, spec sha (canonical) and spec file sha,
  python, lock sha, pin mismatches, torch build, pkgs, host (platform + /proc/cpuinfo model), load at start and end,
  campaign version (726-730, 801-803). The dataset sha256 comes from `runner.run_one`, computed before `method.run`.
- **R-55.**
  - EVAL sessions gate on NP = `cpu_quota()` (bash semaphore, 1375-1378).
  - Each process refuses if the campaign processes, counted once per `--out` (527-545), exceed the cgroup quota
    (720).
  - `cpu_quota` follows the process's own cgroup path, so the VPS scope's 700 % gives 7.
  - Threads are pinned to 1.
- **T3 budget / 2x cap.** EVAL ignores the DEV registry and over-n propagation (711-712, 753). The RLIMIT_CPU soft
  limit is 2 x 7200 = 14 400 (744, 616). An OOM / other signal is an error (585-597).
- **One platform per dataset.** `complete_dataset_keys` (841-851) re-runs every arm of an incomplete dataset. The
  merge (943-959) moves a split dataset to one platform when every key has a non-error record there, else it lists
  it as mixed.
- **Method outputs.** The campaign passes the arm `config` unchanged. Forked isolation (default on Linux) gives each
  unit a copy-on-write copy of the dataset and method object, so there is no cross-unit state.
- **Tests.** `tests/test_xmethod_campaign.py`: all pass locally (2 Linux-only skips).

## Findings

| # | Severity | Where | Finding |
|---|---|---|---|
| F1 | **HIGH (blocker)** | campaign.py:717, 726-730; eval_analysis.py:1159-1160, 1784 | spec sha stamp is not where eval_analysis reads it |
| F2 | MEDIUM-HIGH (ruling) | campaign.py:811-818 (and 707-708 doc) vs PROTOCOL_A s.7 | 2x-cap breach recorded as **error**, protocol says **infeasible, never re-run** |
| F3 | MEDIUM | campaign.py:841-851, 859, 1506 | `--skip-complete-from` accepts records of any run mode / protocol / spec |
| F4 | MEDIUM | campaign.py:280-294, 1493; records | part membership depends on the cost table, which is not stamped or checked |
| F5 | LOW-MEDIUM | campaign.py:1382; vps_run.py:277-278 | session exit code 0 even when every EVAL process refused or crashed |
| F6 | LOW | campaign.py:1447, 860, 744 | `--budget` overrides the spec budget in EVAL, and the effective cap is not stamped |
| F7 | LOW | campaign.py:1449, 713-714, 788 | `--no-isolate` allowed in EVAL (no RLIMIT, shared dataset / method objects), and isolation is not stamped |
| F8 | LOW | campaign.py:1584; vps_run.py:53 | VPS `--procs` is not enforced in EVAL: the scope is always CPUQuota=700 %, so NP = 7 |
| F9 | INFO | campaign.py:1012-1016 | merge summary has no python / lock_sha256 / spec_file_sha256 sets |
| F10 | INFO | campaign.py:186, 138-139, 527-545 | seed-pair range rule, any-line FROZEN mark, restricted /proc |

**F1 (blocker).**
- What campaign writes: `run_mode = {"mode", "protocol_sha256"}`. The spec hashes are under `integrity`:
  - `spec_sha256` = canonical-JSON sha;
  - `spec_file_sha256` = LF file sha (`_sha_lf(spec_bytes)`, CLI 1509).
- What eval_analysis checks: `main` passes `spec_sha = file_sha256(spec)` (LF), and `v11` puts a record in
  `bad_stamp` if `run_mode.spec_sha256 != spec_sha`.
- Why tests miss it: `tests/test_xmethod_eval_analysis.py` (80, 325) builds synthetic records that carry
  `run_mode.spec_sha256`. A real record does not; checked on a real campaign record (Exp B):
  `run_mode {'mode': 'dev', 'protocol_sha256': None}`.
- Failure scenario: the full EVAL completes. Every record lacks `run_mode.spec_sha256`, so V11 "stamps" fails for
  all 150 960 keys and the run is PROVISIONAL (exit 2) with no way to reach FINAL by re-running.
- Fix (needs an amendment; both files frozen):
  - Preferred, so records and campaign.py stay unchanged: eval_analysis compares `integrity.spec_file_sha256`, the
    same LF rule as `file_sha256`.
  - Alternative: campaign also writes `run_mode.spec_sha256 = spec_file_sha256`.
  - Either way, add a test that builds the record with `campaign.run_units`, not by hand.

**F2 (ruling needed).**
- PROTOCOL_A s.7: "In EVAL RLIMIT_CPU = 14 400 (2x safety cap): a unit over it is recorded infeasible with its CPU-s,
  never re-run, and its cell is infeasible; an OOM kill is an error (re-run)."
- `_over(..., ev=True)` instead writes `status: error` ("EVAL safety cap ... exceeded"). `_done_keys` excludes errors,
  so every resume or relaunch re-runs it, up to 4 h CPU each time.
- In eval_analysis it is an error that must be listed in `amendments.json` as persistent. Its cell then counts as
  under-seeded / missing, not "infeasible (cost X)".
- Failure scenario: a pmrt_nl_eq or notears unit at n 24 000 (E4 R3, 300 seeds) needs 15 000 CPU-s. It is recorded
  error and re-run on each relaunch. The run stays PROVISIONAL until an amendment lists it, and the cell is reported
  differently from what s.7 pre-registers.
- CONTRACT R-35 says only "safety cap (2x)", and the campaign docstring says "breach is an ERROR". Either amend s.7
  (amendments file), or make the EVAL breach `infeasible` with its cost and keep OOM as error.

**F3.**
- `run_part` builds `skip` from `--skip-complete-from` files by key alone (`R._done_keys | _infeasible_keys`), with
  no run_mode / protocol / spec filter. Tune keys equal DEV keys (s.7).
- Failure scenario: a relaunch passes a glob that also matches DEV shards (or an EVAL shard of a superseded spec).
  The EVAL tune units of datasets complete there are silently not run. The merge then drops the foreign records
  (`accept`), so those keys are missing (PROVISIONAL) and need another session.
- Detected late, never accepted wrongly. Fix: apply the merge's `accept` rule to the skip-from records.

**F4.**
- `partition` assigns datasets to parts by LPT on `--cost-table`, which is unstamped. Two sessions of one EVAL run
  using different (or no) cost tables give different part memberships.
- Failure scenario: Kaggle runs parts 0-3 with the pilot cost table and the VPS runs 4-7 without one. Some datasets
  fall in neither part set (missing at merge), others in both (duplicates; the merge keeps one platform).
- Fix: stamp the cost-table sha and the part index / count in `integrity`, and refuse mixed values at merge. Or fix
  the partition table once in a bundled file.

**F5.** `_cloud_cmd` ends `...& wait; tail -n 3 log_*`, so the shell exit code is `tail`'s, and vps_run's
`exit_code` records it.
- Failure scenario: a pin mismatch or a non-3.12.14 interpreter makes every EVAL process refuse at start. The session
  still reports exit 0 ("DONE rc 0"), and the dispatcher treats the parts as finished until a merge shows them
  missing.
- Fix: `wait` each PID and exit non-zero if any part's process failed; the per-part logs already hold the reason.

**F6.** `campaign run --budget B` replaces the spec's 7200 in EVAL (limit = 2B), and neither B nor the limit is
stamped.
- Failure scenario: a copied DEV command line with `--budget 2784` (VPS-scaled DEV budget) caps EVAL units at 5568
  CPU-s. Units over it become errors (F2), and nothing in the records shows why the cap differed.
- Fix: refuse `--budget` != spec in EVAL, and stamp the limit.

**F7.** Non-isolated EVAL runs in-process.
- No RLIMIT; only a post-hoc CPU check, so a runaway unit is never stopped.
- Arms of a dataset share one dataset object, so an arm that mutates it in place changes later arms' inputs. Each
  arm hashes the dataset before it runs, so V11's equal-hash check would catch this, but only after the fact.
- Isolation mode is not stamped: only `child_cpu_s` hints at it.
- Fix: refuse `--no-isolate` in EVAL.

**F8.** `campaign vps --procs 4` passes `--procs 4` to `vps_run launch`, but the scope LIMITS are fixed at
CPUQuota=700 %. The EVAL semaphore uses `cpu_quota()` = 7, so 7 parts run concurrently.
- Still within R-55 (VPS 7), so not a validity issue. The flag does not do what it says, and timing runs that intend
  fewer processes get 7.

**F9.** The merge summary reports commits, dirty, platforms, protocol sha and canonical spec sha. It has no python,
lock_sha256 or spec_file_sha256 sets, which are what an operator needs to see a mixed run before eval_analysis.
Campaign's pre-data refusal makes a wrong interpreter impossible in `mode: eval` records, so this is visibility only.

**F10.**
- (a) `_seed_list`: a 2-element list `[a, b]` with b > a + 1 is a range, so an intended pair of discrete seeds would
  silently become b - a + 1 seeds. Every frozen block is a range, so there is no effect now.
- (b) `_is_frozen` accepts "FROZEN: yes" on any line. Harmless, because the sha must match too.
- (c) `campaign_procs` undercounts under a restricted /proc (logged as `proc_visible`), and `cpu_quota` without a
  cgroup quota returns the visible count (8 on the VPS outside a scope; vps_run always uses the scope).

## What was not found

- No path accepts a `mode: eval` record from another interpreter, package set or protocol sha.
- No path resumes a dataset across platforms without the merge putting it back on one platform or listing it.
- Nothing changes method outputs: config is passed through, isolation is per unit, and threads are pinned.
- No silent duplicates: the merge counts them, and conflicting declarations on the same dataset hash are listed.

Seeds or units can be skipped only through the run-time selection (F3, F4). Both surface as `missing` at merge,
never as wrongly accepted records.
