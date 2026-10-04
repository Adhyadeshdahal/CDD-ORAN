# VPS lane (5th platform): root@vps.brihatech.com (worker exp-c, branch xm/exp-c; no push)

Host: Ubuntu 24.04.3, kernel 6.8, 8 vCPU "AMD EPYC-Rome Processor" (AVX2, no AVX-512), 15.2 GB RAM, no swap,
97 GB free on /, cgroup v2, systemd 255. Runs PRODUCTION docker services (dockerd ~3 % CPU, idle).
Host type for R-55 = "vps|AMD EPYC-Rome Processor" (record host.platform = vps via XM_PLATFORM).

## Safety rules (user) and how the lane enforces them (`scratchpad/e6_dev/vps_run.py`)
- Nothing outside /opt/cdd-xm: every remote command exports HOME, XDG_*, UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR,
  UV_PYTHON_BIN_DIR, UV_TOOL_DIR, TMPDIR into /opt/cdd-xm (UV_NO_CONFIG, UV_NO_MODIFY_PATH). Verified after setup:
  no uv / python files in /root/.cache, /root/.local, /root/.config, /tmp. No docker / services / packages /
  firewall / users / cron touched; nothing killed but our own scope.
- uv 0.12.1 = the local version, official release binary, sha256 checked, in /opt/cdd-xm/bin; Python 3.12.13 via
  uv in /opt/cdd-xm/python. Venv per lock (/opt/cdd-xm/venvs/<sha12>, 1.9 GB): hash-pinned --no-deps install of
  the uv.lock export + torch 2.10.0+cpu; pin check 0 mismatches.
- Compute only as `systemd-run --scope --unit cdd-xm-<job> -p CPUQuota=700% -p MemoryMax=11G -p MemorySwapMax=0
  nice -n 19 ionice -c3 run.sh` (limits raised by the user from 600 % / 9G: docker services unused), <= 7
  single-threaded processes, one cdd-xm job at a time (launch refused while a cdd-xm scope is active).
- Pre-launch refusal: MemAvailable < 12 GB, or 1-min load minus our processes > 1.
- Launch refused unless the pushed bundle = the clean local HEAD (records stamp the commit, R-35).
- `stop NAME` = `systemctl stop cdd-xm-NAME.scope` (our unit only). Never reboot.
- exp_c_timing.cpu_quota now reads cpu.max on the process's own cgroup path (the scope), so P = 7 there.

## Lane commands
probe (read-only) | setup | push NAME --paths .. (git-tracked files + MANIFEST) | venv NAME --lock-install
--lock --pincheck | launch NAME --cmd "... {OUT} {PROCS}" [--procs 7] [--dry-run] | status | pull (append-only,
never shrinks a local file) | stop. Local mirror scratchpad/e6_dev/runs/NAME/. Generic: any bundle + command,
so campaign.py parts (aud1's lane) can run as `--cmd "python -m cdd_oran.xmethod.campaign run --spec .. --part
i/P --out {OUT}/res_i.jsonl"` per part, or one command that starts <= 7 parts.

## VPS slot (serialization)
- 2026-10-04 ~08:35Z: exp-c work on the VPS is DONE (Py 3.12.14 + cal-v2; factors sent to xm-harness, in its
  EVAL projection at xm/vps-factors 229decf). VPS idle; handed to the dev-runs / aud1 lane smoke (xm-pmrt-7f got
  "go"); exp-c launches nothing there until they report done. CLI v1 unchanged (venv fix 1b33ca1 only).
- 13:00Z: their smoke xm-vps-smoke-v1-041259 (4 DEV units, 2 procs) exit 0, pulled; venv 1358c306489e-py3.12.14
  (their lock), pin_check 0 mismatches, cpu_quota 7. Probe 13:0xZ: idle (load .05, 14.1 GB free, /opt/cdd-xm
  2.3 GB). VPS FREE; exp-c has nothing queued there (next: none until EVAL / paper timing).

## Jobs
- xm-expc-cal-v1 (R-55 calib, 51 units, commit 4cf8291, 7 procs): DONE 03:46-03:57Z, 635 s wall, 51 / 51 ok,
  wall / CPU 1.00 (uncontended), cpu_quota read 7 from the scope, peak RSS 570 MB, dataset sha = Kaggle for all 51.
  Paired factors f = Kaggle CPU-s / VPS CPU-s (results/exp_c/calib/factors.json; VPS faster): cdl 2.37 (n 500 /
  1000 / 4000: 1.85 / 1.73 / 2.59), mscr_eq_min 3.01, pmrt_nl_eq 2.33, shap_dag 2.34, rcot2_eq 1.99, pcorr_eq
  1.98; pooled "*" 2.69 [1.98, 3.01]. Converted table results/exp_c/calib/vps/.
- xm-expb-p1-v1 (Exp B P1 parts 8-15 of 16, agreed with the exp-b worker; their commit 219bf0b, clean; bundle
  read-only from xm-classic + their 2 npz tables (sha256 checked) + my lock export at their XM_LOCK_FILE path;
  pin_check 75 / 0 mismatches; Python 3.12.13): DONE 04:01-07:54Z, exit 0, xargs -P 7, --budget 2784 VPS CPU-s
  = 7200 / 2.586 (cdl f_hi). 656 records (8:70 9:68 10:87 11:68 12:106 13:87 14:85 15:85), 0 dup / bad / errors;
  652 ok + 4 infeasible (cdl E6 n ~9.8k-10.2k, stopped at ~2783 CPU-s). Scope memory never near 10 GiB. Final
  pull scratchpad/e6_dev/runs/xm-expb-p1-v1/out/; exp-b (xm-classic-e0) told 07:57Z, merges.
- For the orchestrator (exp-b's note): the VPS ruling (5th platform, limits, Exp B parts 8-15 on the VPS) is not
  in CONTRACT.md yet.

## Python 3.12.14 switch (user request + R-57, 2026-10-04): DONE 07:56-08:00Z, after Exp B ended (exit 0)
- uv 0.12.1 -> 0.12.23 (official release binary, sha256 OK) in /opt/cdd-xm/bin; `uv python install 3.12.14` in
  /opt/cdd-xm/python. Venv /opt/cdd-xm/venvs/52b8e75602a7-py3.12.14 (same lock): hash-pinned install + torch
  2.10.0+cpu; `python --version` = Python 3.12.14; pin_check 0 mismatches, python 3.12.14.
- Caught + fixed (1b33ca1): `uv venv --python <exact patch dir>` still wrote home = the floating cpython-3.12
  link (a later patch install would silently retarget the venv). The lane now asks `--managed-python --python
  3.12.14` (uv keeps home on the patch dir) and `venv` exits 5 unless pyvenv.cfg home is the 3.12.14 dir.
- Removed inside /opt/cdd-xm: old venv 52b8e75602a7 (3.12.13) and Python 3.12.13 (uv python uninstall). System
  python untouched (/usr/bin/python3 = 3.12.3). /opt/cdd-xm 2.1 GB. Exp B parts 8-15 stay wholly on 3.12.13.
- xm-expc-cal-v2 (R-55 calib on 3.12.14, commit 1b33ca1, 7 procs): DONE 08:00-08:10Z, 652 s, 51 / 51 ok, wall /
  CPU 1.00, pins ok; datasets + scores identical to cal-v1; CPU-s 4.6 % above cal-v1 (v1 / v2 .93-1.00 by arm).
  factors.json now uses cal-v2 for the VPS: cdl 2.31 [1.72, 2.51], mscr 2.81, pmrt_nl 2.28, shap 2.30, rcot2
  1.99, pcorr 1.92, pooled 2.56 [1.92, 2.81]. The 3.12.13 factors (cal-v1) are kept apart in
  calib/factors_vps_py3.12.13.json for 3.12.13 VPS records (Exp B parts 8-15). Table calib/vps/ rebuilt (v2).
- Drift warning: python-build-standalone ships 3.12.15 (20261001+): `uv python install 3.12` with a fresh uv gets
  3.12.15; my Kaggle / Colab driver pins 3.12.14 (a658e5d); aud1 pins EVAL_PYTHON (R-57).

## Questions
- Q1 ANSWERED (orchestrator, 2026-10-04): accept Python 3.12.13 (R-35b = 3.12; patch stamped per record; the DEV
  10-arm run was 3.12.13 too).
- Q2 ANSWERED (orchestrator): I run parts through vps_run directly now (calibration, then Exp B P1 parts,
  coordinated with the cdl worker who owns xm/exp-b specs / part numbers: no part runs twice). For EVAL aud1 adds
  a dispatcher / campaign lane calling vps_run.py; its CLI is frozen as "vps-run CLI v1" (docstring).
- Q4 ANSWERED (orchestrator, 2026-10-04): Exp B parts 8-15 (Python 3.12.13) are converted with the 3.12.13 VPS
  factor, calib/factors_vps_py3.12.13.json (cdl f_hi 2.586): same interpreter, and it set --budget 2784, so the cdl
  infeasible cut is exactly 7200 ref s. Exp B is descriptive. exp-b (xm-classic-e0) told to name the file in its
  Exp B records / report (their files). EVAL uses the 3.12.14 factors (factors.json); sent to xm-harness-4e for
  the EVAL projection (it assumed VPS = Kaggle speed).
- Q3 ANSWERED (orchestrator) = R-57: EVAL interpreter pinned to 3.12.14 everywhere; aud1 pins EVAL_PYTHON in
  campaign.py; the VPS venv is upgraded to 3.12.14 (after Exp B parts 8-15, see above).
