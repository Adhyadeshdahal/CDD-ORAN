"""Colab runner for e6dev grid scripts (DEV scratch): the Colab counterpart of kaggle_run.py.

  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py launch NAME SCRIPT P [--parts 0,1,...] [--smoke]
                     [--no-schedule] [--every MIN=10] [--no-auto-resume] [--max-resumes N=3]
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py status NAME
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py pull NAME        # incremental merge -> runs/NAME/all.jsonl
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py stop NAME        # stop + verify 0 active assignments
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py tick NAME        # one-shot: keep-alive, pull, finish/resume
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py resume NAME      # new session, re-seed partials, relaunch
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py schedule NAME [--every MIN=10] | unschedule NAME
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py config NAME [auto_resume=0|1] [max_resumes=N]
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py smoke NAME [SCRIPT=e6p_v2_stage2.py] [MINUTES=8]
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py probe NAME [NS=1,4,8,12,16,24]   # CPU scaling only
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py job NAME --paths P1 P2 ... --cmd "SHELL CMD" [--out-dir DIR]
                     [--no-schedule] [--every MIN=10] [--threads N=cgroup quota] [--pins numpy,scipy,...]
                     [--imports mod1,mod2,...] [--force]

ONE-SHOT JOBS (2026-09-29; for analysis runs too heavy for the laptop). `job` zips the given repo-relative paths (files,
dirs -- __pycache__ skipped -- or globs; + JOB_MANIFEST.json: git head, dirty status, per-file sha256, local versions of
the --pins packages) into a temp zip, opens a NEW session NAME, unpacks it to /content/e6run/b (the repo root), pip-pins
every --pins package (default JOB_PINS) whose version differs from the local .venv (torch from the CPU wheel index),
re-reads the versions in a fresh process, import-checks --imports (a failed import stops the runtime and aborts: e.g.
a missing shap would silently switch baselines_disc to its fallback) and writes startup.json (versions before / after,
pip log, cpu quota, threads, env). Then `bash job.sh` (= set -o pipefail + CMD) runs under nohup with cwd = repo root,
PYTHONPATH=., `python` = the pinned interpreter, OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads = --threads (default: the cgroup
CPU quota, 4 on the TPU v5e-1 VM: 24 visible vCPUs would oversubscribe it), stdout+stderr -> job.log; when it returns,
bash `times` -> times.txt (children user / sys CPU) and the exit code -> exit_code (written atomically). --out-dir DIR
(repo-relative) is created before the run. Everything is recorded in runs/NAME/colab.json ("kind": "job").
  tick (same scheduled task as launch): keep-alive ping + one kernel exec (job alive?, exit code, log tail, session CPU /
        RSS) that packs job.log + the out dir + startup/job/times files; downloaded to runs/NAME/ (out dir mirrored at
        runs/NAME/DIR). exit_code present (or the job died without one) -> final pull, stop runtime, verify 0
        assignments, unschedule. Session gone -> NOT resumable: logged "job-lost", colab.json "lost_utc", unscheduled,
        never relaunched (re-run `job` under a new NAME). 3 consecutive failed execs -> stop runtime, lost.
  status / pull / stop NAME work for jobs (status = the tick's exec without download).

UNATTENDED RUNS (2026-09-29, after run e6p-v2-s2c was lost: its laptop poller was OOM-killed, the idle runtime was
reclaimed, nothing had been pulled). `launch` registers a Windows Task Scheduler task "CDD-ORAN-colab-NAME" (current
user, no admin) that runs `tick NAME` every MIN minutes through runs/NAME/tick.cmd (sets PYTHONPATH / COLAB_STUB /
MSYS_NO_PATHCONV, cd to the repo, conhost --headless = no console flash). No long-lived laptop process.
  tick: session alive -> keep-alive (GET /tun/m/<endpoint>/keep-alive/ + one kernel exec, see KEEP-ALIVE), part
        status, incremental pull: remote res_i.jsonl are MERGED into runs/NAME/<i>/res_<i>.jsonl by job key (append
        only: a local file never shrinks, partial last lines are ignored), runs/NAME/all.jsonl is rebuilt.
        All parts closed (or dead with a traceback) -> final pull, stop runtime, verify 0 assignments, unschedule.
        Session gone -> `resume` (unless auto_resume is off or max_resumes are used up). One line per tick ->
        runs/NAME/tick.log. A lock file stops overlapping ticks.
  resume: new session, the SAME bundle (runs/NAME/bundle.zip, saved by launch), setup + fingerprint, uploads the
        local merged res_<i>.jsonl into /content/e6run/, verifies the remote job-key counts, relaunches only the
        unfinished parts (the drivers skip finished keys). Recorded in colab.json "resumes".
KEEP-ALIVE (investigated 2026-09-29, google-colab-cli 0.7.4): the CLI has no keep-alive, heartbeat or idle-timeout
option. colab-vscode kept servers alive with GET /tun/m/<endpoint>/keep-alive/ (header X-Colab-Tunnel: Google) every
5 min, but only while a kernel was busy or had activity in the last hour; it deleted that code on 2026-09-25
(PR #718: "with cl/982796273 rolled out to Production, the client-side ServerKeepAliveController is effectively a
no-op now"). So the backend now judges liveness itself, from kernel activity. Background (nohup) processes are NOT
kernel activity: e6p-v2-s2c had no kernel call after its launch exec (05:46:58Z) and was gone by 08:18Z. A tick
therefore always runs one kernel exec (the status + pack cell, which moves the kernel's last_activity) and also
sends the legacy keep-alive ping (logged HTTP status). `colab status` / `sessions` / `usage` never touch the runtime.
Measured 2026-09-29 (colab_e2e_test.py, runtime Jupyter /api/kernels + /api/status): kernel last_activity moved ONLY
on an exec (unchanged after usage + status + sessions, and after the keep-alive ping, which still answers HTTP 200);
the server-level last_activity also moves on any REST call to the runtime (the probe's own). Not measured: the idle
timeout itself (the lost run died 0-152 min after its last exec). Ticks every 10 min keep kernel activity far inside
colab-vscode's 1 h window; if the laptop sleeps and the runtime is reclaimed anyway, the next tick resumes.
E2E TEST 2026-09-29 (colab_e2e_test.py, 2 parts of colab_selftest.py, ticks fired through the real scheduled task):
tick pulled 12/24 jobs -> `colab stop` mid-run -> tick: gone -> resume (new VM, 6+6 keys re-seeded and verified,
parts relaunched, 54 s) -> ticks -> final: 24/24 keys, 0 duplicates, 0 missing, runtime stopped, 0 assignments
(usage lags the unassign by 3-5 min, hence stop's 600 s wait), task deleted by the tick itself.

Runtime: free-tier TPU v5e-1 host (`colab new --tpu v5e1`), used only for its CPUs (24 vCPU AMD EPYC, 47 GB).
Bundle: the SAME zip cloud.py builds for Kaggle (cloud.main: cdd_oran e6/xtruce/decision + e6dev/*.py + MANIFEST with
local_env / local_fp). Setup pip-installs the MANIFEST's numpy / scipy when they differ and runs `cloud.py fingerprint`.
startup.json records the per-layer equality against the newest Kaggle startup.json with the same numpy version found
under runs/ (the Linux reference that Colab results are pooled with) and against the local Windows fingerprint
(expected to differ: platform libm, cloud.py docstring). launch starts P parts (`SCRIPT run --part i/P --out
res_i.jsonl`) with nohup as background processes, one single-threaded process per part, so `exec` returns.
CPU QUOTA (measured 2026-09-29, `probe`): the TPU v5e-1 VM shows 24 vCPUs (12 cores x 2 threads) but its cgroup
cpu.max is 400000/100000 = 4 CPUs. Throughput vs one process: 2 -> 2.0x, 4 -> 3.94x, 8 -> 3.85x, 12 -> 3.74x,
24 -> 2.2x (each process 10.9x slower). Use P = 4 parts per runtime; one process runs ~2.9x faster than a Kaggle
kernel CPU (v1 stage-1 episode 25.5 s vs 74 s, bit-identical outcome).
`smoke` = launch 2 smoke parts + one full-length timing episode and a 24-process contention probe, poll, pull, and
ALWAYS stop the runtime (try/finally).

colab CLI (uv tool google-colab-cli) on Windows needs a termios stub on PYTHONPATH (written to
%TEMP%/colab_stub if COLAB_STUB is unset), `--auth adc`, and MSYS_NO_PATHCONV=1 (set here for every call).
The CLI is run through the uv tool's own python with google-auth's per-request HTTP timeout raised from 120 s to
COLAB_HTTP_TIMEOUT (default 900 s): the TPU v5e-1 assignment POST takes ~130 s, so the stock `colab new --tpu v5e1`
times out client-side (measured 2026-09-29). Falls back to the plain `colab` executable if the tool python is absent.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W = "/content/e6run"                                              # remote work dir
COLAB = shutil.which("colab") or os.path.expanduser("~/.local/bin/colab.exe")
TOOL_PY = os.environ.get("COLAB_TOOL_PY") or os.path.join(os.environ.get("APPDATA", ""), "uv", "tools",
                                                          "google-colab-cli", "Scripts", "python.exe")
HTTP_TIMEOUT = int(os.environ.get("COLAB_HTTP_TIMEOUT", "900"))
WRAP = ("import sys\n"
        "import google.auth.transport.requests as R\n"
        "_o = R.AuthorizedSession.request\n"
        "def _req(self, method, url, *a, **k):\n"
        f"    k.setdefault('timeout', {HTTP_TIMEOUT})\n"
        "    return _o(self, method, url, *a, **k)\n"
        "R.AuthorizedSession.request = _req\n"
        "from colab_cli.cli import main\n"
        "sys.argv = ['colab'] + sys.argv[1:]\n"
        "main()\n")
TIMING_KEY = ["1", "P3", 1, 150110, "noarb"]                      # a v1 stage-1 job already run on Kaggle (re-run only
#                                                                   as a speed + bit-reproducibility reference)


# ---------------------------------------------------------------------------------------------- colab CLI
def _stub():
    d = os.environ.get("COLAB_STUB") or os.path.join(tempfile.gettempdir(), "colab_stub")
    if not os.path.exists(os.path.join(d, "termios.py")):
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "termios.py"), "w").write(
            "TCSANOW=0\nTCSADRAIN=1\nTCSAFLUSH=2\n"
            "def tcgetattr(fd):\n    raise OSError('termios unavailable on Windows')\n"
            "def tcsetattr(*a):\n    raise OSError('termios unavailable on Windows')\n")
    return d


def colab(*args, timeout=600, check=True):
    env = dict(os.environ, PYTHONPATH=_stub(), MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8", COLUMNS="200")
    exe = [TOOL_PY, "-c", WRAP] if os.path.exists(TOOL_PY) else [COLAB]
    r = subprocess.run([*exe, "--auth", "adc", *args], capture_output=True, text=True, timeout=timeout, env=env,
                       encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        raise RuntimeError(f"colab {' '.join(args[:3])} rc={r.returncode}: {out[-1500:]}")
    return out


def remote(name, code, timeout=600, tag=None):
    """Execute python ``code`` in the session's kernel; returns the full output, or the JSON after ``tag``."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="colab_exec_")
    with os.fdopen(fd, "w", newline="\n") as f:
        f.write(code)
    try:
        out = colab("exec", "-s", name, "-f", path, "--timeout", str(timeout), timeout=timeout + 120)
    finally:
        os.remove(path)
    if tag is None:
        return out
    for ln in reversed(out.splitlines()):
        if ln.startswith(tag + " "):
            return json.loads(ln[len(tag) + 1:])
    raise RuntimeError(f"no {tag} line in remote output:\n{out[-3000:]}")


def new_session(name, tries=3):
    """`colab new -s NAME --tpu v5e1` with retries; after a failed attempt, refuse to continue if the server holds an
    assignment the CLI does not know (orphan: stop it by hand). Env COLAB_ACCEL picks the runtime (2026-10-02,
    xm-citests): "tpu:v5e1" (default, unchanged), "gpu:T4" (GPU ports), "cpu" (plain CPU runtime)."""
    accel = os.environ.get("COLAB_ACCEL", "tpu:v5e1")
    kind, _, variant = accel.partition(":")
    flags = {"tpu": ["--tpu", variant], "gpu": ["--gpu", variant], "cpu": []}[kind]
    for k in range(tries):
        try:
            out = colab("new", "-s", name, *flags, timeout=HTTP_TIMEOUT + 300)
            print(out[-300:].encode("ascii", "replace").decode(), flush=True)
            if "READY" in out:
                return
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"new attempt {k + 1} failed: {str(e)[-300:]}".encode("ascii", "replace").decode(), flush=True)
        n, _ = active_assignments()
        if n:
            ses = colab("sessions", timeout=120, check=False)
            if f"[{name}]" in ses:
                return
            known = sum(1 for ln in ses.splitlines() if ln.lstrip().startswith("["))
            if n > known:      # an assignment no CLI session accounts for: orphan, stop it by hand
                raise SystemExit(f"{n} active assignment(s), only {known} known sessions (not {name!r}): {ses}")
            # every assignment is another known session (concurrent jobs): just retry
        time.sleep(60)
    raise SystemExit(f"could not create a {accel} runtime after {tries} attempts")


def run_dir(name):
    return os.path.join(HERE, "runs", name)


def meta_path(name):
    return os.path.join(run_dir(name), "colab.json")


def load_meta(name):
    return json.load(open(meta_path(name)))


def save_meta(name, meta):
    os.makedirs(run_dir(name), exist_ok=True)
    tmp = meta_path(name) + ".tmp"
    json.dump(meta, open(tmp, "w"), indent=1)
    os.replace(tmp, meta_path(name))


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tick_log(name, event, info=None):
    os.makedirs(run_dir(name), exist_ok=True)
    line = f"{_utc()} {event} {json.dumps(info or {}, default=str)}"
    with open(os.path.join(run_dir(name), "tick.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def session_probe(name, ping=True):
    """Assignment state of session NAME (alive / orphans) + keep-alive ping + kernels' last_activity. One tool-python
    process; no kernel traffic."""
    env = dict(os.environ, PYTHONPATH=_stub(), PYTHONIOENCODING="utf-8")
    r = subprocess.run([TOOL_PY, "-c", SPROBE, name, "1" if ping else "0"], capture_output=True, text=True,
                       timeout=400, env=env, encoding="utf-8", errors="replace")
    for ln in reversed((r.stdout or "").splitlines()):
        if ln.startswith("SPROBE "):
            return json.loads(ln[7:])
    raise RuntimeError(f"session probe rc={r.returncode}: {(r.stdout + r.stderr)[-1500:]}")


def active_assignments():
    out = colab("usage", timeout=120)
    for ln in out.splitlines():
        if "Active assignments" in ln:
            return int(ln.split(":")[1].strip()), out
    raise RuntimeError(f"cannot parse `colab usage`: {out}")


# ---------------------------------------------------------------------------------------------- references
def kaggle_reference(numpy_version):
    """Newest Kaggle startup.json under runs/ with a fingerprint on Linux with the given numpy version."""
    best = None
    for f in glob.glob(os.path.join(HERE, "runs", "*", "*", "startup.json")):
        try:
            s = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        fp = s.get("fingerprint") or {}
        env = fp.get("env") or {}
        if "traj" in fp and env.get("numpy") == numpy_version and str(env.get("platform", "")).startswith("Linux"):
            if best is None or os.path.getmtime(f) > os.path.getmtime(best[0]):
                best = (f, fp)
    return best


def kaggle_timing_record():
    for f in glob.glob(os.path.join(HERE, "runs", "e6p-s1", "all.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("kind") == "job" and r.get("key") == TIMING_KEY:
                return {k: r[k] for k in ("secs", "psvr", "svr", "rlf", "energy_j", "prot_viol", "st_changes",
                                          "viol_ue_s", "ll_viol")}
    return None


# ---------------------------------------------------------------------------------------------- remote scripts
SETUP = r'''
import glob, json, os, shutil, subprocess, sys, zipfile
import importlib.metadata as md
W, BUNDLE, KREF, KREF_SRC = __W__, __BUNDLE__, __KREF__, __KREF_SRC__
os.makedirs(W, exist_ok=True)
shutil.move(BUNDLE, os.path.join(W, "bundle.zip"))
root = os.path.join(W, "b")
shutil.rmtree(root, ignore_errors=True)
zipfile.ZipFile(os.path.join(W, "bundle.zip")).extractall(root)
man = json.load(open(os.path.join(root, "MANIFEST.json")))
want = man.get("local_env", {})
have = {p: md.version(p) for p in ("numpy", "scipy")}
inst = []
if want and any(have[p] != want.get(p) for p in have):
    for pkgs in ([f"{p}=={want[p]}" for p in have], [f"numpy=={want['numpy']}"]):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
        inst.append({"pkgs": pkgs, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-600:]})
        if r.returncode == 0:
            break
env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
fp = subprocess.run([sys.executable, os.path.join(root, "e6dev", "cloud.py"), "fingerprint"], env=env, cwd=root,
                    capture_output=True, text=True)
try:
    fpj = json.loads(fp.stdout.strip().splitlines()[-1])
except Exception:
    fpj = {"error": (fp.stdout + fp.stderr)[-1500:]}
L = ("rng", "ufunc", "scipy_fp", "traj")
lfp = man.get("local_fp", {})
eq_local = {k: fpj.get(k) == lfp.get(k) for k in L} if lfp and "env" in fpj else {}
eq_k = {k: fpj.get(k) == KREF.get(k) for k in L} if KREF and "env" in fpj else {}
start = {"manifest": man["git_head"], "e6_dirty": man.get("e6_dirty"), "cpus": os.cpu_count(), "want": want,
         "install": inst, "env": fpj.get("env"), "kaggle_ref": KREF_SRC, "fp_layers_equal_kaggle": eq_k,
         "fp_match": bool(eq_k) and all(eq_k.values()), "fp_traj_match_kaggle": eq_k.get("traj"),
         "ufunc_diff_vs_kaggle": [x for x in (KREF or {}).get("ufunc", {})
                                  if fpj.get("ufunc", {}).get(x) != KREF["ufunc"][x]],
         "fp_layers_equal_local_windows": eq_local, "fp_error": fpj.get("error")}
try:
    mem = subprocess.run(["bash", "-c", "free -g | sed -n 2p; lscpu | grep -E 'Model name|Thread|Core|Socket'"],
                         capture_output=True, text=True).stdout
except Exception as e:
    mem = repr(e)
start["host"] = mem
try:
    q, per = open("/sys/fs/cgroup/cpu.max").read().split()
    start["cpu_quota"] = None if q == "max" else int(q) / int(per)
except Exception:
    start["cpu_quota"] = None
json.dump(dict(start, fingerprint=fpj), open(os.path.join(W, "startup.json"), "w"), indent=1)
print("STARTUP " + json.dumps(start), flush=True)
'''

LAUNCH = r'''
import json, os, subprocess, sys
W, SCRIPT, P, PARTS, SMOKE = __W__, __SCRIPT__, __P__, __PARTS__, __SMOKE__
root = os.path.join(W, "b")
env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
lp = os.path.join(W, "launch.json")
old = json.load(open(lp)) if os.path.exists(lp) else {}
pids = {int(k): v for k, v in old.get("pids", {}).items()}          # relaunch in place keeps the other parts' pids
PARTS_ALL = sorted(set(old.get("parts", [])) | set(PARTS))
for i in PARTS:
    cmd = ["nohup", sys.executable, "-u", os.path.join(root, "e6dev", SCRIPT), "run", "--part", f"{i}/{P}",
           "--out", os.path.join(W, f"res_{i}.jsonl")] + (["--smoke"] if SMOKE else [])
    log = open(os.path.join(W, f"log_{i}.txt"), "a")
    pids[i] = subprocess.Popen(cmd, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True).pid
json.dump({"script": SCRIPT, "P": P, "parts": PARTS_ALL, "smoke": SMOKE, "pids": pids}, open(lp, "w"))
print("LAUNCHED " + json.dumps({i: pids[i] for i in PARTS}), flush=True)
'''

TIMING = r'''
import json, os, subprocess, sys
W, KEY = __W__, __KEY__
root = os.path.join(W, "b")
env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
code = ("import sys, json, time; sys.path.insert(0, 'e6dev'); import e6p_screen as S; st = S.load_state(); "
        f"t = time.time(); r = S.run_job(tuple({KEY!r}), st, False); r['wall'] = time.time() - t; "
        f"json.dump(r, open({os.path.join(W, 'timing.json')!r}, 'w'))")
pid = subprocess.Popen(["nohup", sys.executable, "-c", code], env=env, cwd=root,
                       stdout=open(os.path.join(W, "log_timing.txt"), "a"), stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, start_new_session=True).pid
print("TIMING " + json.dumps({"pid": pid}), flush=True)
'''

PROBE = r'''
import json, os, subprocess, sys, time
W, NS = __W__, __N__
root = os.path.join(W, "b")
env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
code = ("import sys, time, dataclasses; sys.path.insert(0, 'e6dev'); import e6p_screen as S; "
        "st = S.load_state(); lf = S.lf_of(st, 1); "
        "cfg = dataclasses.replace(S.make_cfg('P3', 1, 3, lf, smoke=True), warmup_s=120.0, scored_s=60.0); "
        "t = time.time(); S.run_env(cfg, None); print(time.time() - t)")
def cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[7] if len(v) > 7 else 0                                  # total jiffies, steal
def batch(n):
    t, (c0, s0) = time.time(), cpu()
    ps = [subprocess.Popen([sys.executable, "-c", code], env=env, cwd=root, stdout=subprocess.PIPE, text=True)
          for _ in range(n)]
    per = [float(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
    c1, s1 = cpu()
    return {"n": n, "wall": time.time() - t, "per_mean": sum(per) / n, "per_max": max(per),
            "steal_frac": (s1 - s0) / max(c1 - c0, 1)}
def rd(f):
    try:
        return open(f).read().strip()
    except Exception as e:
        return repr(e)
res = {"episode": "P3 base-L40 DEV seed 3, 180 s sim (warmup 120 + 60), accept-all",
       "cgroup_cpu_max": rd("/sys/fs/cgroup/cpu.max"), "nproc": os.cpu_count(),
       "affinity": len(os.sched_getaffinity(0)), "single": batch(1)}
res["batches"] = [batch(n) for n in NS]
for b in res["batches"]:
    b["slowdown"] = b["per_mean"] / res["single"]["per_mean"]
    b["throughput_x_single"] = b["n"] / b["slowdown"]                      # episodes/s relative to one process
res["concurrent"] = res["batches"][-1]
res["slowdown"] = res["concurrent"]["slowdown"]
json.dump(res, open(os.path.join(W, "probe.json"), "w"))
print("PROBE " + json.dumps(res), flush=True)
'''


STATUS = r'''
import glob, json, os
W = __W__
L = json.load(open(os.path.join(W, "launch.json"))) if os.path.exists(os.path.join(W, "launch.json")) else {}
def alive(pid):
    try:
        st = open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
        return st != "Z"
    except Exception:
        return False
parts = {}
for i in L.get("parts", []):
    f = os.path.join(W, f"res_{i}.jsonl")
    recs = [json.loads(x) for x in open(f) if x.strip()] if os.path.exists(f) else []
    log = os.path.join(W, f"log_{i}.txt")
    tail = open(log).read()[-300:] if os.path.exists(log) else ""
    full = open(log).read() if os.path.exists(log) else ""
    parts[i] = {"jobs": sum(r.get("kind") == "job" for r in recs),
                "closed": any(r.get("kind") == "close" for r in recs), "alive": alive(L["pids"][str(i)]),
                "traceback": "Traceback" in full, "tail": tail.strip().splitlines()[-1:] if tail.strip() else []}
out = {"script": L.get("script"), "P": L.get("P"), "n_parts": len(parts),
       "finished": sum(p["closed"] for p in parts.values()), "alive": sum(p["alive"] for p in parts.values()),
       "failed": [i for i, p in parts.items() if not p["alive"] and not p["closed"]],
       "jobs_done": sum(p["jobs"] for p in parts.values()), "parts": parts,
       "timing_done": os.path.exists(os.path.join(W, "timing.json")),
       "load": open("/proc/loadavg").read().split()[:3]}
print("STATUS " + json.dumps(out), flush=True)
'''

PACK = r'''
import glob, json, os, tarfile
W = __W__
dst = "/content/e6run_pull.tar.gz"
with tarfile.open(dst, "w:gz") as t:
    for f in sorted(glob.glob(os.path.join(W, "res_*.jsonl")) + glob.glob(os.path.join(W, "log_*.txt")) +
                    [os.path.join(W, x) for x in ("startup.json", "timing.json", "probe.json", "launch.json")]):
        if os.path.exists(f):
            t.add(f, arcname=os.path.basename(f))
print("PACKED " + json.dumps({"bytes": os.path.getsize(dst)}), flush=True)
'''

# one kernel exec per tick: part status (from the remote files + pids) and the pull tarball. Also the keep-alive.
TICKR = r'''
import json, os, tarfile, time
W, PARTS = __W__, __PARTS__
lp = os.path.join(W, "launch.json")
L = json.load(open(lp)) if os.path.exists(lp) else {}
pids = {int(k): v for k, v in L.get("pids", {}).items()}
def alive(pid):
    try:
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except Exception:
        return False
parts = {}
for i in PARTS:
    f, lg = os.path.join(W, f"res_{i}.jsonl"), os.path.join(W, f"log_{i}.txt")
    jobs, closed = 0, False
    if os.path.exists(f):
        for x in open(f):
            try:
                r = json.loads(x)
            except ValueError:
                continue
            jobs += r.get("kind") == "job"
            closed = closed or r.get("kind") == "close"
    log = open(lg, errors="replace").read() if os.path.exists(lg) else ""
    parts[i] = {"jobs": jobs, "closed": closed, "alive": i in pids and alive(pids[i]), "launched": i in pids,
                "traceback": "Traceback" in log, "tail": log.strip().splitlines()[-1:] if log.strip() else []}
dst = "/content/e6run_pull.tar.gz"
with tarfile.open(dst, "w:gz") as t:
    for n in sorted(os.listdir(W)) if os.path.isdir(W) else []:
        if (n.startswith("res_") and n.endswith(".jsonl")) or (n.startswith("log_") and n.endswith(".txt")) or \
                n in ("startup.json", "timing.json", "probe.json", "launch.json"):
            t.add(os.path.join(W, n), arcname=n)
print("TICK " + json.dumps({"parts": parts, "bytes": os.path.getsize(dst), "t": time.time(),
                            "uptime_s": float(open("/proc/uptime").read().split()[0]),
                            "load": open("/proc/loadavg").read().split()[:3]}), flush=True)
'''

# run with the colab tool's python: is the session's assignment alive + legacy keep-alive ping + the kernels'
# server-side activity as the runtime's Jupyter API reports it (the evidence for what counts as activity)
SPROBE = r'''
import json, sys
import google.auth.transport.requests as R
_o = R.AuthorizedSession.request
def _req(self, method, url, *a, **k):
    k.setdefault("timeout", 120)
    return _o(self, method, url, *a, **k)
R.AuthorizedSession.request = _req
import requests
from colab_cli.auth import AuthProvider
from colab_cli.common import state
state.auth_provider = AuthProvider("adc")
name, ping = sys.argv[1], sys.argv[2] == "1"
s = state.store.get(name)
asg = state.client.list_assignments()
by = {a.endpoint: a for a in asg}
out = {"known": s is not None, "n_assignments": len(asg), "endpoint": s.endpoint if s else None,
       "alive": bool(s) and s.endpoint in by, "orphans": [e for e in by if not s or e != s.endpoint]}
if out["alive"] and ping:
    try:
        r = state.client.session.request(
            "GET", f"https://colab.research.google.com/tun/m/{s.endpoint}/keep-alive/", params={"authuser": "0"},
            headers={"X-Colab-Tunnel": "Google", "X-Colab-Client-Agent": "colab-cli", "Accept": "application/json"})
        out["keepalive_http"] = r.status_code
    except Exception as e:
        out["keepalive_err"] = repr(e)[:300]
if out["alive"]:
    try:
        ri = by[s.endpoint].runtime_proxy_info
        h = {"X-Colab-Runtime-Proxy-Token": ri.token, "X-Colab-Client-Agent": "colab-cli"}
        r = requests.get(ri.url.rstrip("/") + "/api/status", headers=h, timeout=60)   # before /api/kernels
        if r.ok:
            out["server"] = r.json()
        r = requests.get(ri.url.rstrip("/") + "/api/kernels", headers=h, timeout=60)
        out["kernels_http"] = r.status_code
        if r.ok:
            out["kernels"] = [{k: x.get(k) for k in ("id", "last_activity", "execution_state", "connections")}
                              for x in r.json()]
    except Exception as e:
        out["kernels_err"] = repr(e)[:300]
print("SPROBE " + json.dumps(out), flush=True)
'''


def _fill(src, **kw):
    for k, v in kw.items():
        src = src.replace(f"__{k}__", repr(v))
    return src


# ---------------------------------------------------------------------------------------------- commands
def _setup(name, meta):
    """Upload meta's bundle into the (new) session, unpack, pin numpy/scipy, fingerprint -> startup dict."""
    import zipfile
    man = json.loads(zipfile.ZipFile(meta["bundle"]).read("MANIFEST.json"))
    ref = kaggle_reference(man["local_env"]["numpy"])
    rb = f"/content/{name}_bundle.zip"
    colab("upload", "-s", name, meta["bundle"], rb, timeout=900)
    return remote(name, _fill(SETUP, W=W, BUNDLE=rb, KREF=ref[1] if ref else None,
                              KREF_SRC=os.path.relpath(ref[0], HERE) if ref else None), timeout=1200, tag="STARTUP")


def launch(name, script, P, parts=None, smoke=False, timing=False, probe=(), reuse=False, sched=False, every=10,
           auto_resume=True, max_resumes=3):
    P = int(P)
    parts = list(range(P)) if parts is None else [int(x) for x in parts]
    assert all(0 <= i < P for i in parts), parts
    out = os.path.join(tempfile.gettempdir(), "e6_colab", name)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    import cloud
    cloud.main(out, name, script, "1", "match")                     # the SAME bundle as the Kaggle path
    import zipfile
    os.makedirs(run_dir(name), exist_ok=True)
    bundle = os.path.join(run_dir(name), "bundle.zip")              # durable copy (resume re-uploads it; gitignored)
    shutil.copyfile(os.path.join(out, "dataset", f"{name}_bundle.zip"), bundle)
    man = json.loads(zipfile.ZipFile(bundle).read("MANIFEST.json"))
    ref = kaggle_reference(man["local_env"]["numpy"])
    meta = {"session": name, "script": script, "P": P, "parts": parts, "smoke": smoke, "bundle": bundle,
            "git_head": man["git_head"], "e6_dirty": man["e6_dirty"], "kaggle_ref": ref[0] if ref else None,
            "created_utc": _utc(), "auto_resume": bool(auto_resume), "max_resumes": int(max_resumes),
            "session_no": 0, "resumes": []}
    save_meta(name, meta)
    t0 = time.time()
    if not reuse:
        new_session(name)
    meta["t_new"] = round(time.time() - t0, 1)
    t1 = time.time()
    start = _setup(name, meta)
    meta.update(t_setup=round(time.time() - t1, 1), startup=start)
    print("startup:", json.dumps({k: start.get(k) for k in ("cpus", "cpu_quota", "env", "fp_match",
                                                            "fp_layers_equal_kaggle", "ufunc_diff_vs_kaggle",
                                                            "fp_layers_equal_local_windows", "kaggle_ref",
                                                            "install", "host")}), flush=True)
    quota = start.get("cpu_quota")
    if quota and len(parts) > quota:
        print(f"WARNING: {len(parts)} parts > cgroup CPU quota {quota:g}: throughput drops (probe: 24 parts on 4 CPUs"
              f" = 2.2x one process, 4 parts = 3.9x)", flush=True)
    if probe:
        meta["probe"] = remote(name, _fill(PROBE, W=W, N=[int(x) for x in probe]), timeout=1800, tag="PROBE")
        print("probe:", json.dumps(meta["probe"]), flush=True)
    meta["launched"] = remote(name, _fill(LAUNCH, W=W, SCRIPT=script, P=P, PARTS=parts, SMOKE=bool(smoke)),
                              timeout=300, tag="LAUNCHED")
    if timing:
        meta["timing_pid"] = remote(name, _fill(TIMING, W=W, KEY=TIMING_KEY), timeout=300, tag="TIMING")
    meta["launched_utc"] = _utc()
    save_meta(name, meta)
    print("launched parts", parts, "of", P, "->", meta["launched"], flush=True)
    tick_log(name, "launch", {"parts": parts, "P": P, "script": script, "pids": meta["launched"]})
    if sched and parts:
        schedule(name, every)
    return meta


def status(name, quiet=False):
    if _is_job(name):
        return job_status(name, quiet)
    st = remote(name, _fill(STATUS, W=W), timeout=300, tag="STATUS")
    if not quiet:
        print(json.dumps({k: st[k] for k in ("script", "P", "n_parts", "finished", "alive", "failed", "jobs_done",
                                             "timing_done", "load")}))
        for i, p in list(st["parts"].items())[:30]:
            print(f"  part {i}: jobs={p['jobs']} closed={p['closed']} alive={p['alive']} tb={p['traceback']} "
                  f"{p['tail']}")
    return st


# ---------------------------------------------------------------------------------------------- incremental results
def part_file(name, i):
    return os.path.join(run_dir(name), str(i), f"res_{i}.jsonl")


def _read_lines(path):
    """Complete JSON records of a jsonl file as [(line, rec)] + the number of skipped lines (unparsable, or a partial
    last line that was still being written when the file was packed)."""
    if not os.path.exists(path):
        return [], 0
    chunks = open(path, "rb").read().decode("utf-8", "replace").split("\n")
    out, bad = [], int(bool(chunks[-1].strip()))
    for c in chunks[:-1]:
        c = c.rstrip("\r")
        if c.strip():
            try:
                out.append((c, json.loads(c)))
            except ValueError:
                bad += 1
    return out, bad


def _ident(line, rec):
    """Job records (anything with a key) are identified by (kind, key, smoke) -- the drivers' resume key; every
    other record (header, close) by its exact text."""
    if isinstance(rec, dict) and "key" in rec:
        return ("K", rec.get("kind"), json.dumps(rec["key"]), rec.get("smoke"))
    return ("L", line)


def merge_file(local, remote_file):
    """Append to ``local`` the records of ``remote_file`` it does not have yet. Append-only: the local file never
    shrinks and is never replaced by the remote one. Returns (added, skipped_remote_lines, duplicates_ignored)."""
    loc, _ = _read_lines(local)
    rem, bad = _read_lines(remote_file)
    seen = {_ident(ln, r) for ln, r in loc}
    new, dup = [], 0
    for ln, r in rem:
        k = _ident(ln, r)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        new.append(ln)
    if new:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        size0 = os.path.getsize(local) if os.path.exists(local) else 0
        with open(local, "ab") as f:
            if size0 and open(local, "rb").read()[-1:] != b"\n":
                f.write(b"\n")
            f.write("".join(x + "\n" for x in new).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        assert os.path.getsize(local) > size0
    return len(new), bad, dup


def local_parts(name, meta):
    """Per part: unique job keys, closed, record count -- from the local merged files (the source of truth)."""
    out = {}
    for i in meta["parts"]:
        recs, _ = _read_lines(part_file(name, i))
        keys = {_ident(ln, r) for ln, r in recs if isinstance(r, dict) and r.get("kind") == "job"}
        out[i] = {"jobs": len(keys), "job_lines": sum(isinstance(r, dict) and r.get("kind") == "job" for _, r in recs),
                  "closed": any(isinstance(r, dict) and r.get("kind") == "close" for _, r in recs)}
    return out


def rebuild_all(name, meta):
    """runs/NAME/all.jsonl = concatenation of the local part files (never written smaller than the existing one)."""
    lines = []
    for i in sorted(meta["parts"]):
        lines += [ln + "\n" for ln, _ in _read_lines(part_file(name, i))[0]]
    dst = os.path.join(run_dir(name), "all.jsonl")
    old = sum(1 for _ in open(dst, encoding="utf-8", errors="replace")) if os.path.exists(dst) else 0
    if len(lines) < old:
        dst += ".new"
        tick_log(name, "warn-all-smaller", {"old": old, "new": len(lines), "written": dst})
    tmp = dst + ".tmp"
    open(tmp, "w", newline="\n", encoding="utf-8").writelines(lines)
    os.replace(tmp, dst)
    return len(lines)


def _download_merge(name, meta):
    """Download the tarball the TICK cell packed and merge it into runs/NAME/<i>/ (res by key; logs / startup per
    session number)."""
    d, k = run_dir(name), meta.get("session_no", 0)
    tgz, tmp = os.path.join(d, "pull.tar.gz"), os.path.join(d, ".pull")
    colab("download", "-s", name, "/content/e6run_pull.tar.gz", tgz, timeout=1200)
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tgz) as t:
        t.extractall(tmp, filter="data")
    os.remove(tgz)
    added, skipped, dups = {}, 0, 0
    for f in sorted(os.listdir(tmp)):
        src = os.path.join(tmp, f)
        if f.startswith("res_") and f.endswith(".jsonl"):
            i = int(f[4:-6])
            a, b, c = merge_file(part_file(name, i), src)
            added[i], skipped, dups = a, skipped + b, dups + c
        elif f.startswith("log_") and f.endswith(".txt") and f[4:-4].isdigit():
            i = int(f[4:-4])
            os.makedirs(os.path.join(d, str(i)), exist_ok=True)
            shutil.copyfile(src, os.path.join(d, str(i), f"log_{i}.s{k}.txt"))
        elif f.endswith(".json"):
            shutil.copyfile(src, os.path.join(d, f))
            if f == "startup.json":
                shutil.copyfile(src, os.path.join(d, f"startup.s{k}.json"))
    shutil.rmtree(tmp, ignore_errors=True)
    n_all = rebuild_all(name, meta)
    return {"added": added, "skipped_partial": skipped, "all": n_all}


def pull(name):
    """Incremental pull (same as a tick's pull, without keep-alive / finish / resume logic)."""
    if _is_job(name):
        return job_pull(name)
    meta = load_meta(name)
    st = remote(name, _fill(TICKR, W=W, PARTS=meta["parts"]), timeout=600, tag="TICK")
    res = _download_merge(name, meta)
    dst = run_dir(name)
    print(f"{res['all']} records from {len(meta['parts'])} parts -> {os.path.join(dst, 'all.jsonl')}; "
          f"added {res['added']}", flush=True)
    sp = os.path.join(dst, "startup.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        print("startup", {k: s.get(k) for k in ("fp_match", "fp_layers_equal_kaggle", "ufunc_diff_vs_kaggle",
                                                "kaggle_ref")},
              {k: (s.get("env") or {}).get(k) for k in ("numpy", "scipy", "platform", "simd")})
    return dst, st, res


# ---------------------------------------------------------------------------------------------- tick / resume
_LOCKED = False


def _pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))                     # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259                                          # STILL_ACTIVE


class _Lock:
    """runs/NAME/tick.lock: one tick / resume at a time (stale if its pid is gone or it is older than 3 h)."""

    def __init__(self, name):
        self.p, self.own = os.path.join(run_dir(name), "tick.lock"), False

    def __enter__(self):
        global _LOCKED
        if _LOCKED:
            return True
        for _ in range(2):
            try:
                fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    info = json.load(open(self.p))
                    stale = not _pid_alive(info["pid"]) or time.time() - info["t"] > 3 * 3600
                except Exception:  # noqa: BLE001
                    stale = True
                if not stale:
                    return False
                os.remove(self.p)
                continue
            os.write(fd, json.dumps({"pid": os.getpid(), "t": time.time()}).encode())
            os.close(fd)
            self.own = _LOCKED = True
            return True
        return False

    def __exit__(self, *exc):
        global _LOCKED
        if self.own:
            os.remove(self.p)
            self.own = _LOCKED = False


def _finish(name, meta, info, alive=True):
    """Final pull already merged: stop the runtime (if still assigned), verify 0 assignments, mark finished,
    unschedule."""
    tick_log(name, "final", info)
    try:
        if alive:
            stop(name)
        else:
            t0, n, out = time.time(), None, ""
            while time.time() - t0 < 600:
                n, out = active_assignments()
                if n == 0:
                    break
                time.sleep(20)
            if n:
                raise SystemExit(f"STILL {n} ACTIVE COLAB ASSIGNMENT(S)")
    except SystemExit as e:
        tick_log(name, "stop-failed", {"err": str(e)})               # stays scheduled: the next tick retries
        return
    meta["finished_utc"] = _utc()
    meta["final"] = info
    save_meta(name, meta)
    tick_log(name, "done", {"all": info.get("all"), "finished_utc": meta["finished_utc"]})
    unschedule(name)


def tick(name):
    with _Lock(name) as got:
        if not got:
            tick_log(name, "skip", {"reason": "another tick/resume holds runs/NAME/tick.lock"})
            return
        meta = load_meta(name)
        if meta.get("finished_utc") or meta.get("gave_up_utc") or meta.get("lost_utc") or meta.get("failed_utc"):
            tick_log(name, "noop", {k: meta.get(k + "_utc") for k in ("finished", "gave_up", "lost", "failed")
                                    if meta.get(k + "_utc")})
            unschedule(name)
            return
        if meta.get("kind") == "job":
            return _job_tick(name, meta)
        try:
            pr = session_probe(name)
        except Exception as e:  # noqa: BLE001  (network / auth hiccup: try again next tick)
            tick_log(name, "probe-error", {"err": str(e)[-400:]})
            return
        ka = {k: pr.get(k) for k in ("keepalive_http", "keepalive_err", "kernels", "kernels_err")}
        if pr["alive"]:
            try:
                st = remote(name, _fill(TICKR, W=W, PARTS=meta["parts"]), timeout=600, tag="TICK")
                res = _download_merge(name, meta)
            except Exception as e:  # noqa: BLE001
                meta["exec_failures"] = meta.get("exec_failures", 0) + 1
                save_meta(name, meta)
                tick_log(name, "exec-error", {"n": meta["exec_failures"], "err": str(e)[-400:], **ka})
                if meta["exec_failures"] < 3:
                    return
                tick_log(name, "unhealthy", {"action": "stop runtime, then resume"})
                try:
                    stop(name)
                except SystemExit as e2:
                    tick_log(name, "stop-failed", {"err": str(e2)})
                    return
                pr = {"alive": False, "n_assignments": 0, "orphans": []}
            else:
                meta["exec_failures"] = 0
                loc = local_parts(name, meta)
                rp = {int(i): p for i, p in st["parts"].items()}
                done = [i for i in meta["parts"] if loc[i]["closed"]]
                running = [i for i in meta["parts"] if rp[i]["alive"]]
                dead = [i for i in meta["parts"] if i not in done and i not in running]
                failed = [i for i in dead if rp[i]["traceback"]]
                info = {"session_no": meta.get("session_no", 0), "jobs": {i: loc[i]["jobs"] for i in meta["parts"]},
                        "done": done, "running": running, "dead": dead, "failed": failed, **res,
                        "uptime_min": round(st["uptime_s"] / 60, 1), "load": st["load"], **ka}
                if not running and len(done) + len(failed) == len(meta["parts"]):
                    save_meta(name, meta)
                    _finish(name, meta, info)
                    return
                rel = meta.setdefault("relaunched", {})
                again = [i for i in dead if i not in failed and rel.get(str(i), 0) < 2]
                if again:                                             # died without a traceback (OOM kill?)
                    info["relaunch_in_place"] = remote(
                        name, _fill(LAUNCH, W=W, SCRIPT=meta["script"], P=meta["P"], PARTS=again,
                                    SMOKE=bool(meta["smoke"])), timeout=300, tag="LAUNCHED")
                    for i in again:
                        rel[str(i)] = rel.get(str(i), 0) + 1
                save_meta(name, meta)
                tick_log(name, "tick", info)
                return
        # ---- the session is gone
        loc = local_parts(name, meta)
        if all(p["closed"] for p in loc.values()):                   # e.g. a previous final tick's stop was slow
            _finish(name, meta, {"jobs": {i: p["jobs"] for i, p in loc.items()}, "note": "already complete locally",
                                 "all": rebuild_all(name, meta)}, alive=False)
            return
        if pr.get("n_assignments"):
            tick_log(name, "gone-but-assignments", {"orphans": pr.get("orphans"), "known": pr.get("known"),
                                                    "action": "none (stop them by hand: colab --auth adc sessions)"})
            return
        used, mx = len(meta.get("resumes", [])), meta.get("max_resumes", 3)
        if not meta.get("auto_resume", True) or used >= mx:
            meta["gave_up_utc"] = _utc()
            save_meta(name, meta)
            tick_log(name, "gone-no-resume", {"auto_resume": meta.get("auto_resume", True), "resumes": used,
                                              "max_resumes": mx, "local": local_parts(name, meta)})
            unschedule(name)
            return
        tick_log(name, "gone", {"resumes_used": used, "max_resumes": mx, "local": local_parts(name, meta)})
        resume(name, reason="session gone (tick)")


SEED = r'''
import json, os, tarfile
W, TGZ, PARTS = __W__, __TGZ__, __PARTS__
os.makedirs(W, exist_ok=True)
with tarfile.open(TGZ) as t:
    t.extractall(W, filter="data")
os.remove(TGZ)
out = {}
for i in PARTS:
    f = os.path.join(W, f"res_{i}.jsonl")
    ks = set()
    if os.path.exists(f):
        for x in open(f):
            try:
                r = json.loads(x)
            except ValueError:
                continue
            if r.get("kind") == "job":
                ks.add(json.dumps([r.get("kind"), r.get("key"), r.get("smoke")]))
    out[i] = len(ks)
print("SEEDED " + json.dumps(out), flush=True)
'''


def resume(name, reason="manual"):
    """New session + same bundle + setup, re-seed the local merged res files, relaunch the unfinished parts."""
    with _Lock(name) as got:
        if not got:
            raise SystemExit("another tick/resume holds runs/NAME/tick.lock")
        meta = load_meta(name)
        loc = local_parts(name, meta)
        todo = [i for i in meta["parts"] if not loc[i]["closed"]]
        if not todo:
            tick_log(name, "resume-nothing", {"local": loc})
            return
        pr = session_probe(name, ping=False)
        if pr["alive"] or pr["n_assignments"]:
            raise SystemExit(f"refusing to resume: assignment(s) still active {pr} -- `colab_run.py stop {name}` first")
        if not os.path.exists(meta["bundle"]):
            raise SystemExit(f"bundle missing: {meta['bundle']}")
        k, t0 = meta.get("session_no", 0) + 1, time.time()
        tick_log(name, "resume-start", {"reason": reason, "session_no": k, "todo": todo, "local": loc})
        new_session(name)
        t_new = round(time.time() - t0, 1)
        start = _setup(name, meta)
        json.dump(start, open(os.path.join(run_dir(name), f"startup.s{k}.local.json"), "w"), indent=1)
        seed = os.path.join(run_dir(name), "seed.tar.gz")
        with tarfile.open(seed, "w:gz") as t:
            for i in meta["parts"]:
                if os.path.exists(part_file(name, i)):
                    t.add(part_file(name, i), arcname=f"res_{i}.jsonl")
        rs = f"/content/{name}_seed.tar.gz"
        colab("upload", "-s", name, seed, rs, timeout=900)
        os.remove(seed)
        cnt = {int(i): n for i, n in remote(name, _fill(SEED, W=W, TGZ=rs, PARTS=meta["parts"]), timeout=300,
                                             tag="SEEDED").items()}
        bad = {i: (cnt.get(i), loc[i]["jobs"]) for i in meta["parts"] if cnt.get(i) != loc[i]["jobs"]}
        if bad:
            tick_log(name, "resume-seed-mismatch", {"remote_vs_local": bad})
            stop(name)
            raise SystemExit(f"re-seeded job counts differ from local: {bad}; runtime stopped")
        launched = remote(name, _fill(LAUNCH, W=W, SCRIPT=meta["script"], P=meta["P"], PARTS=todo,
                                      SMOKE=bool(meta["smoke"])), timeout=300, tag="LAUNCHED")
        meta["session_no"] = k
        meta["exec_failures"] = 0
        meta["relaunched"] = {}
        meta.setdefault("resumes", []).append({
            "utc": _utc(), "reason": reason, "session_no": k, "relaunched": todo, "pids": launched,
            "seeded_jobs": cnt, "t_new": t_new, "t_total": round(time.time() - t0, 1),
            "fp_match": start.get("fp_match"), "env": start.get("env"), "cpu_quota": start.get("cpu_quota")})
        save_meta(name, meta)
        tick_log(name, "resumed", meta["resumes"][-1])


# ---------------------------------------------------------------------------------------------- one-shot jobs
JOB_PINS = ("numpy", "scipy", "scikit-learn", "shap", "numba", "llvmlite", "pandas", "torch")
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
JOB_PULL = "/content/e6job_pull.tar.gz"
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
_JOB_FILES = ("job.log", "job.json", "job.sh", "startup.json", "exit_code", "times.txt", "ended", "wrapper.log")

# run in a FRESH process after the pip installs: the versions the job will really see + the import check
JOB_CHECK = r'''
import importlib, importlib.metadata as md, json, os, sys, time
P, I = json.loads(sys.argv[1]), json.loads(sys.argv[2])
out = {"python": sys.version.split()[0], "executable": sys.executable, "versions": {}, "imports": {}}
for p in P:
    try:
        out["versions"][p] = md.version(p)
    except Exception:
        out["versions"][p] = None
for m in I:
    t = time.time()
    try:
        mod = importlib.import_module(m)
        out["imports"][m] = {"ok": True, "s": round(time.time() - t, 2), "file": getattr(mod, "__file__", None)}
    except BaseException as e:
        out["imports"][m] = {"ok": False, "err": repr(e)[-600:]}
if "torch" in sys.modules:
    out["torch_threads"] = sys.modules["torch"].get_num_threads()
print("CHECK " + json.dumps(out), flush=True)
'''

JOB_SETUP = r'''
import json, os, shutil, subprocess, sys, time, zipfile
import importlib.metadata as md
W, BUNDLE, PINS, IMPORTS, THREADS, TORCH_INDEX, CHECK = __W__, __BUNDLE__, __PINS__, __IMPORTS__, __THREADS__, __TORCH_INDEX__, __CHECK__
t0 = time.time()
os.makedirs(W, exist_ok=True)
root = os.path.join(W, "b")
shutil.rmtree(root, ignore_errors=True)
for f in ("exit_code", "exit_code.tmp", "ended", "times.txt", "job.json", "job.log", "job.sh", "wrapper.log",
          "startup.json"):
    try:
        os.remove(os.path.join(W, f))
    except FileNotFoundError:
        pass
shutil.move(BUNDLE, os.path.join(W, "bundle.zip"))
zipfile.ZipFile(os.path.join(W, "bundle.zip")).extractall(root)
man = json.load(open(os.path.join(root, "JOB_MANIFEST.json")))
want = {p: v for p, v in man.get("local_env", {}).items() if p in PINS and v}
def base(v):
    return v.split("+")[0] if v else v
def ver(p):
    try:
        return md.version(p)
    except Exception:
        return None
before = {p: ver(p) for p in PINS}
need = {p: base(v) for p, v in want.items() if base(before.get(p)) != base(v)}
inst = []
def pip(*args):
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "--disable-pip-version-check",
                        *args], capture_output=True, text=True)
    inst.append({"args": list(args), "rc": r.returncode, "s": round(time.time() - t, 1),
                 "tail": (r.stdout + r.stderr)[-800:]})
    return r.returncode == 0
plain = [f"{p}=={v}" for p, v in need.items() if p != "torch"]
if plain and not pip(*plain):
    for s in plain:
        pip(s)
if "torch" in need:                                       # CPU wheel: the PyPI one drags in GBs of CUDA libraries
    pip(f"torch=={need['torch']}", "--index-url", TORCH_INDEX)
try:
    q, per = open("/sys/fs/cgroup/cpu.max").read().split()
    quota = None if q == "max" else int(q) / int(per)
except Exception:
    quota = None
threads = int(THREADS or max(1, int(quota or os.cpu_count() or 1)))
bindir = os.path.join(W, "bin")
os.makedirs(bindir, exist_ok=True)
for n in ("python", "python3"):
    p = os.path.join(bindir, n)
    if os.path.lexists(p):
        os.remove(p)
    os.symlink(sys.executable, p)
jenv = {"PYTHONPATH": ".", "PATH": bindir + os.pathsep + os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg", **{k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                          "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")}}
r = subprocess.run([sys.executable, "-c", CHECK, json.dumps(list(PINS)), json.dumps(list(IMPORTS))],
                   env=dict(os.environ, **jenv), cwd=root, capture_output=True, text=True, timeout=1200)
chk = None
for ln in reversed(r.stdout.splitlines()):
    if ln.startswith("CHECK "):
        chk = json.loads(ln[6:])
        break
if chk is None:
    chk = {"error": (r.stdout + r.stderr)[-1500:], "versions": {}, "imports": {}}
after = chk["versions"]
diff = {p: {"local": v, "colab": after.get(p)} for p, v in want.items() if base(after.get(p)) != base(v)}
try:
    host = subprocess.run(["bash", "-c", "free -g | sed -n 2p; lscpu | grep -E 'Model name|Thread|Core|Socket'"],
                          capture_output=True, text=True).stdout
except Exception as e:
    host = repr(e)
start = {"manifest": {k: man.get(k) for k in ("name", "git_head", "git_branch", "created_utc", "dirty", "paths")},
         "n_files": len(man.get("files", [])), "local_python": man.get("local_python"), "python": chk.get("python"),
         "executable": sys.executable, "cpus": os.cpu_count(), "cpu_quota": quota, "threads": threads,
         "torch_threads": chk.get("torch_threads"), "want": want, "before": before, "after": after,
         "pins_ok": not diff, "pins_diff": diff, "install": inst, "imports": chk.get("imports"),
         "imports_ok": "error" not in chk and all(v.get("ok") for v in chk.get("imports", {}).values()),
         "check_error": chk.get("error"), "host": host, "job_env": jenv, "t_setup_s": round(time.time() - t0, 1)}
json.dump(start, open(os.path.join(W, "startup.json"), "w"), indent=1)
print("JOBSTART " + json.dumps(start), flush=True)
'''

JOB_LAUNCH = r'''
import json, os, shlex, subprocess, sys, time
W, CMD, OUT = __W__, __CMD__, __OUT__
root = os.path.join(W, "b")
st = json.load(open(os.path.join(W, "startup.json")))
env = dict(os.environ, **st["job_env"])
if OUT:
    os.makedirs(os.path.join(root, OUT), exist_ok=True)
sh = os.path.join(W, "job.sh")
open(sh, "w").write("set -o pipefail\n" + CMD + "\n")
q = lambda n: shlex.quote(os.path.join(W, n))
wrap = (f"bash {q('job.sh')} > {q('job.log')} 2>&1 < /dev/null; rc=$?; times > {q('times.txt')}; "
        f"date +%s > {q('ended')}; echo $rc > {q('exit_code.tmp')}; mv {q('exit_code.tmp')} {q('exit_code')}")
p = subprocess.Popen(["nohup", "bash", "-c", wrap], env=env, cwd=root, stdin=subprocess.DEVNULL,
                     stdout=open(os.path.join(W, "wrapper.log"), "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
J = {"pid": p.pid, "cmd": CMD, "out": OUT, "cwd": root, "started": time.time()}
json.dump(J, open(os.path.join(W, "job.json"), "w"))
time.sleep(5)
J["alive_5s"] = p.poll() is None
try:
    J["log_head"] = open(os.path.join(W, "job.log"), errors="replace").read()[:600]
except Exception:
    J["log_head"] = None
print("JOBLAUNCHED " + json.dumps(J), flush=True)
'''

# one kernel exec per tick (= the keep-alive): job state + the pull tarball
JOB_TICK = r'''
import json, os, tarfile, time
W, OUT, DST, FILES, PACK = __W__, __OUT__, __DST__, __FILES__, __PACK__
root = os.path.join(W, "b")
jp = os.path.join(W, "job.json")
J = json.load(open(jp)) if os.path.exists(jp) else {}
def stat(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    except Exception:
        return None
def alive(pid):
    s = stat(pid)
    return bool(s) and s[0] != "Z"
def rd(n):
    try:
        return open(os.path.join(W, n), errors="replace").read().strip()
    except Exception:
        return None
tck = os.sysconf("SC_CLK_TCK")
cpu_s, rss, nproc = 0.0, 0, 0
if J:                                     # live processes of the job's session (sid = the nohup wrapper's pid)
    for d in os.listdir("/proc"):
        s = stat(d) if d.isdigit() else None
        if s and int(s[3]) == J["pid"] and s[0] != "Z":
            nproc += 1
            cpu_s += (int(s[11]) + int(s[12])) / tck
            rss += int(s[21]) * os.sysconf("SC_PAGE_SIZE")
rc, ended, log = rd("exit_code"), rd("ended"), rd("job.log") or ""
lines = log.splitlines()
od = os.path.join(root, OUT) if OUT else None
outs = []
if od and os.path.isdir(od):
    for dp, _, fns in os.walk(od):
        for fn in sorted(fns):
            f = os.path.join(dp, fn)
            outs.append([os.path.relpath(f, od), os.path.getsize(f)])
size = None
if PACK:
    with tarfile.open(DST, "w:gz") as t:
        for n in FILES:
            if os.path.exists(os.path.join(W, n)):
                t.add(os.path.join(W, n), arcname=n)
        if od and os.path.isdir(od):
            t.add(od, arcname="outdir")
    size = os.path.getsize(DST)
now = time.time()
mem = {ln.split(":")[0]: int(ln.split()[1]) // 1024 for ln in open("/proc/meminfo") if ln.split(":")[0] in
       ("MemTotal", "MemAvailable")}
print("JOBTICK " + json.dumps({
    "launched": bool(J), "alive": bool(J) and alive(J["pid"]), "exit_code": int(rc) if rc else None,
    "elapsed_s": round((float(ended) if ended else now) - J.get("started", now), 1), "times": rd("times.txt"),
    "live_cpu_s": round(cpu_s, 1), "live_rss_mb": rss >> 20, "live_procs": nproc, "mem_mb": mem,
    "log_lines": len(lines), "log_tail": lines[-8:], "traceback": "Traceback" in log, "n_out": len(outs),
    "out_files": outs[:50], "bytes": size, "uptime_s": float(open("/proc/uptime").read().split()[0]),
    "load": open("/proc/loadavg").read().split()[:3]}), flush=True)
'''


def _is_job(name):
    try:
        return load_meta(name).get("kind") == "job"
    except (OSError, ValueError):
        return False


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60, errors="replace")
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _job_files(paths):
    """{repo-relative posix path: absolute path} for files, dirs (walked, caches skipped) and globs under ROOT."""
    out, rootn = {}, os.path.normcase(os.path.abspath(ROOT))
    for p in paths:
        p = p.replace("\\", "/").strip()
        full = os.path.normpath(os.path.join(ROOT, p))
        hits = sorted(glob.glob(full, recursive=True)) if any(c in p for c in "*?[") else \
            ([full] if os.path.exists(full) else [])
        if not hits:
            raise SystemExit(f"--paths: nothing at {p!r}")
        for h in hits:
            if os.path.commonpath([rootn, os.path.normcase(os.path.abspath(h))]) != rootn:
                raise SystemExit(f"--paths: {h} is outside the repo")
            walk = os.walk(h) if os.path.isdir(h) else [(os.path.dirname(h), [], [os.path.basename(h)])]
            for dp, dns, fns in walk:
                dns[:] = sorted(d for d in dns if d not in _SKIP_DIRS)
                for fn in sorted(fns):
                    if not fn.endswith((".pyc", ".pyo")):
                        f = os.path.join(dp, fn)
                        out[os.path.relpath(f, ROOT).replace("\\", "/")] = f
    return out


def _job_bundle(name, paths, pins):
    import importlib.metadata as md
    import zipfile
    files = _job_files(paths)
    local = {}
    for p in pins:
        try:
            local[p] = md.version(p)
        except md.PackageNotFoundError:
            local[p] = None
    man = {"name": name, "created_utc": _utc(), "git_head": _git("rev-parse", "HEAD"),
           "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
           "dirty": (_git("status", "--porcelain", "--", *paths) or "").splitlines(), "paths": list(paths),
           "local_env": local, "local_python": sys.version.split()[0],
           "files": [{"path": r, "bytes": os.path.getsize(f), "sha256": _sha256(f)} for r, f in sorted(files.items())]}
    d = os.path.join(tempfile.gettempdir(), "e6_colab")
    os.makedirs(d, exist_ok=True)
    zp = os.path.join(d, f"{name}_job.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r, f in sorted(files.items()):
            z.write(f, r)
        z.writestr("JOB_MANIFEST.json", json.dumps(man, indent=1))
    return zp, man


def job(name, paths, cmd, out_dir=None, sched=True, every=10, threads=None, pins=JOB_PINS, imports=(), force=False):
    """One-shot job: bundle PATHS, new session, pin + import check, run CMD under nohup (see the module docstring)."""
    if os.path.exists(meta_path(name)) and not force:
        raise SystemExit(f"{meta_path(name)} exists: pick a new NAME (or --force)")
    if out_dir:
        out_dir = out_dir.replace("\\", "/").strip("/")
        if os.path.isabs(out_dir) or ".." in out_dir.split("/") or out_dir in _JOB_FILES:
            raise SystemExit(f"--out-dir must be a repo-relative dir, got {out_dir!r}")
    n, _ = active_assignments()
    if n:
        print(f"WARNING: {n} active assignment(s) already; a new TPU v5e-1 may be refused", flush=True)
    zp, man = _job_bundle(name, paths, pins)
    meta = {"kind": "job", "session": name, "cmd": cmd, "paths": list(paths), "out_dir": out_dir, "threads": threads,
            "pins_local": man["local_env"], "imports": list(imports), "git_head": man["git_head"],
            "dirty": man["dirty"], "n_files": len(man["files"]), "bundle_bytes": os.path.getsize(zp),
            "bundle_sha256": _sha256(zp), "created_utc": _utc(), "session_no": 0}
    save_meta(name, meta)
    print(f"bundle {len(man['files'])} files, {meta['bundle_bytes'] / 1e6:.1f} MB, head {man['git_head']}, "
          f"dirty {len(man['dirty'])}, local pins {man['local_env']}", flush=True)
    t0 = time.time()
    new_session(name)
    meta["t_new"] = round(time.time() - t0, 1)
    save_meta(name, meta)
    try:
        t1, rb = time.time(), f"/content/{name}_job.zip"
        colab("upload", "-s", name, zp, rb, timeout=1800)
        meta["t_upload"] = round(time.time() - t1, 1)
        start = remote(name, _fill(JOB_SETUP, W=W, BUNDLE=rb, PINS=list(pins), IMPORTS=list(imports), THREADS=threads,
                                   TORCH_INDEX=TORCH_CPU_INDEX, CHECK=JOB_CHECK), timeout=2700, tag="JOBSTART")
        meta.update(t_setup=round(time.time() - t1, 1), startup=start)
        json.dump(start, open(os.path.join(run_dir(name), "startup.json"), "w"), indent=1)
        print("startup:", json.dumps({k: start.get(k) for k in ("python", "cpus", "cpu_quota", "threads",
                                                                "torch_threads", "after", "pins_ok", "pins_diff",
                                                                "imports_ok", "t_setup_s")}), flush=True)
        if not start["pins_ok"]:
            print(f"WARNING: versions differ from local: {start['pins_diff']}", flush=True)
        if not start["imports_ok"]:
            raise SystemExit(f"import check failed: {start.get('imports')} {start.get('check_error')}")
        meta["launched"] = remote(name, _fill(JOB_LAUNCH, W=W, CMD=cmd, OUT=out_dir), timeout=300, tag="JOBLAUNCHED")
    except BaseException as e:
        meta["failed_utc"], meta["error"] = _utc(), repr(e)[-1500:]
        save_meta(name, meta)
        tick_log(name, "job-setup-failed", {"err": repr(e)[-800:]})
        stop(name)
        raise
    finally:
        os.remove(zp)
    meta["launched_utc"] = _utc()
    save_meta(name, meta)
    tick_log(name, "job-launch", {"pid": meta["launched"]["pid"], "alive_5s": meta["launched"]["alive_5s"],
                                  "cmd": cmd, "out_dir": out_dir, "threads": meta["startup"]["threads"]})
    if not meta["launched"]["alive_5s"]:
        print("NOTE: the job had already exited 5 s after launch; log head:\n" + (meta["launched"]["log_head"] or ""),
              flush=True)
    if sched:
        schedule(name, every)
    return meta


def _job_remote(name, meta, pack=True):
    return remote(name, _fill(JOB_TICK, W=W, OUT=meta.get("out_dir"), DST=JOB_PULL, FILES=list(_JOB_FILES),
                              PACK=pack), timeout=600, tag="JOBTICK")


def _job_download(name, meta):
    """Download the tarball the JOB_TICK cell packed into runs/NAME/ (out dir -> runs/NAME/<out_dir>/)."""
    d = run_dir(name)
    tgz, tmp = os.path.join(d, "pull.tar.gz"), os.path.join(d, ".pull")
    colab("download", "-s", name, JOB_PULL, tgz, timeout=1800)
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tgz) as t:
        t.extractall(tmp, filter="data")
    os.remove(tgz)
    got = []
    for f in sorted(os.listdir(tmp)):
        src = os.path.join(tmp, f)
        if f == "outdir" and os.path.isdir(src) and meta.get("out_dir"):
            shutil.copytree(src, os.path.join(d, *meta["out_dir"].split("/")), dirs_exist_ok=True)
            got.append(meta["out_dir"] + "/")
        elif f in _JOB_FILES and os.path.isfile(src):
            shutil.copyfile(src, os.path.join(d, f))
            got.append(f)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"pulled": got}


def _job_brief(st):
    return {k: st.get(k) for k in ("alive", "exit_code", "elapsed_s", "live_cpu_s", "live_rss_mb", "live_procs",
                                   "mem_mb", "log_lines", "traceback", "n_out", "times", "load")}


def job_status(name, quiet=False):
    st = _job_remote(name, load_meta(name), pack=False)
    if not quiet:
        print(json.dumps(_job_brief(st)))
        for ln in st["log_tail"]:
            print("  |", ln)
        for f, b in st["out_files"]:
            print(f"  out: {f} {b} B")
    return st


def job_pull(name):
    meta = load_meta(name)
    st = _job_remote(name, meta)
    res = _job_download(name, meta)
    print(json.dumps(dict(_job_brief(st), **res)), flush=True)
    return run_dir(name), st, res


def _job_lost(name, meta, why, info=None):
    meta["lost_utc"], meta["lost_reason"] = _utc(), why
    save_meta(name, meta)
    tick_log(name, "job-lost", dict(info or {}, reason=why, action="not resumable: not relaunched; re-run `job` "
                                                                    "under a new NAME"))
    unschedule(name)


def _job_tick(name, meta):
    """tick for kind == "job" (called with the tick lock held)."""
    try:
        pr = session_probe(name)
    except Exception as e:  # noqa: BLE001
        tick_log(name, "probe-error", {"err": str(e)[-400:]})
        return
    ka = {k: pr.get(k) for k in ("keepalive_http", "keepalive_err", "kernels_err")}
    if pr["alive"]:
        try:
            st = _job_remote(name, meta)
            res = _job_download(name, meta)
        except Exception as e:  # noqa: BLE001
            meta["exec_failures"] = meta.get("exec_failures", 0) + 1
            save_meta(name, meta)
            tick_log(name, "exec-error", {"n": meta["exec_failures"], "err": str(e)[-400:], **ka})
            if meta["exec_failures"] < 3:
                return
            try:
                stop(name)
            except SystemExit as e2:
                tick_log(name, "stop-failed", {"err": str(e2)})
                return
            _job_lost(name, meta, "3 consecutive failed execs (runtime stopped)")
            return
        meta["exec_failures"] = 0
        info = dict(_job_brief(st), log_tail=st["log_tail"][-3:], uptime_min=round(st["uptime_s"] / 60, 1),
                    **res, **ka)
        if st["exit_code"] is not None or not st["alive"]:
            meta["exit_code"] = st["exit_code"]
            info["result"] = (f"exit {st['exit_code']}" if st["exit_code"] is not None
                              else "DIED without an exit code (wrapper killed?)")
            info["out_files"] = st["out_files"]
            save_meta(name, meta)
            _finish(name, meta, info)
            return
        save_meta(name, meta)
        tick_log(name, "tick", info)
        return
    # ---- the session is gone: a job is never resumed
    ec = os.path.join(run_dir(name), "exit_code")
    if os.path.exists(ec):                                          # finished + pulled; a previous stop was slow
        meta["exit_code"] = int(open(ec).read().strip() or -1)
        _finish(name, meta, {"exit_code": meta["exit_code"], "note": "already complete locally"}, alive=False)
        return
    _job_lost(name, meta, "session gone before the exit code was pulled",
              {"n_assignments": pr.get("n_assignments"), "orphans": pr.get("orphans"), "known": pr.get("known"),
               "warning": "orphan assignment(s): stop them by hand (colab --auth adc sessions)"
               if pr.get("n_assignments") else None})


# ---------------------------------------------------------------------------------------------- Task Scheduler
def task_name(name):
    return f"CDD-ORAN-colab-{name}"


def _schtasks(*args):
    r = subprocess.run(["schtasks", *args], capture_output=True, text=True, timeout=120, errors="replace")
    return r.returncode, (r.stdout + r.stderr).strip()


TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>CDD-ORAN colab_run.py tick {name} every {every} min</Description></RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition><Interval>PT{every}M</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>
      <StartBoundary>{start}</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{conhost}</Command>
      <Arguments>--headless cmd.exe /c "{cmd}"</Arguments>
      <WorkingDirectory>{root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def schedule(name, every=10):
    """Register the Task Scheduler task that runs `tick NAME` every EVERY minutes (current user, no admin)."""
    every = int(every)
    d = run_dir(name)
    os.makedirs(d, exist_ok=True)
    stub = _stub()
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    py = py if os.path.exists(py) else sys.executable
    cmd = os.path.join(d, "tick.cmd")
    lines = ["@echo off", f'set "PYTHONPATH={stub}"', f'set "COLAB_STUB={stub}"', 'set "MSYS_NO_PATHCONV=1"',
             'set "PYTHONIOENCODING=utf-8"', f'set "COLAB_TOOL_PY={TOOL_PY}"', f'cd /d "{ROOT}"',
             f'echo ==== %DATE% %TIME% >> "{os.path.join(d, "tick.out")}"',
             f'"{py}" "{os.path.join(HERE, "colab_run.py")}" tick {name} >> "{os.path.join(d, "tick.out")}" 2>&1']
    open(cmd, "w", newline="\r\n").write("\n".join(lines) + "\n")
    xml = os.path.join(d, "task.xml")
    user = f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}".lstrip("\\")
    conhost = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "conhost.exe")
    open(xml, "w", encoding="utf-16").write(TASK_XML.format(
        name=name, every=every, start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 60 * every)),
        user=user, conhost=conhost, cmd=cmd, root=ROOT))
    rc, out = _schtasks("/create", "/tn", task_name(name), "/xml", xml, "/f")
    if rc != 0:
        raise SystemExit(f"schtasks /create failed rc={rc}: {out}")
    rq, q = _schtasks("/query", "/tn", task_name(name), "/fo", "list")
    if os.path.exists(meta_path(name)):
        meta = load_meta(name)
        meta.update(task=task_name(name), every_min=every, scheduled_utc=_utc())
        save_meta(name, meta)
    tick_log(name, "scheduled", {"task": task_name(name), "every_min": every, "query_rc": rq, "cmd": cmd})
    return q


def unschedule(name):
    rc, out = _schtasks("/delete", "/tn", task_name(name), "/f")
    rq, _ = _schtasks("/query", "/tn", task_name(name))
    tick_log(name, "unscheduled", {"task": task_name(name), "delete_rc": rc, "gone": rq != 0})
    return rq != 0


def config(name, *kv):
    meta = load_meta(name)
    for x in kv:
        k, v = x.split("=", 1)
        if k == "auto_resume":
            meta[k] = v.lower() in ("1", "true", "yes", "on")
        elif k == "max_resumes":
            meta[k] = int(v)
        else:
            raise SystemExit(f"unknown key {k!r} (auto_resume, max_resumes)")
    save_meta(name, meta)
    print({k: meta.get(k) for k in ("auto_resume", "max_resumes", "resumes", "task", "every_min")})


def stop(name, wait_s=600):                                            # unassign shows in usage after ~3 min
    try:
        print(colab("stop", "-s", name, timeout=300, check=False)[-400:], flush=True)
    finally:
        t0, n, out = time.time(), None, ""
        while time.time() - t0 < wait_s:
            n, out = active_assignments()
            if n == 0:
                break
            time.sleep(15)
        print(out, flush=True)
        if n != 0:
            raise SystemExit(f"STILL {n} ACTIVE COLAB ASSIGNMENT(S): stop them by hand (colab --auth adc sessions)")
        print("verified: 0 active assignments", flush=True)


def probe(name, ns="1,4,8,12,16,24"):
    """CPU-throughput probe only: new runtime, bundle, fingerprint, concurrency batches of one 180 s-sim episode;
    ALWAYS stops the runtime. Launches no parts (P=0 parts)."""
    try:
        launch(name, "e6p_v2_stage2.py", 1, parts=[], probe=[int(x) for x in ns.split(",")])
    finally:
        stop(name)


def smoke(name, script="e6p_v2_stage2.py", minutes=8):
    """Real end-to-end smoke: new runtime, bundle upload, pin + fingerprint, 24-process contention probe, 2 smoke
    parts of SCRIPT (parts 0,1 of 24) + one full-length timing episode (TIMING_KEY), poll, pull; ALWAYS stop."""
    t0 = time.time()
    try:
        launch(name, script, 24, parts=[0, 1], smoke=True, timing=True, probe=[24])
        deadline = time.time() + 60 * float(minutes)
        while time.time() < deadline:
            time.sleep(30)
            st = status(name, quiet=True)
            print(f"{(time.time() - t0) / 60:.1f} min: finished {st['finished']}/2 alive {st['alive']} "
                  f"timing_done {st['timing_done']} load {st['load']}", flush=True)
            if st["finished"] + len(st["failed"]) >= 2 and st["timing_done"]:
                break
        status(name)
        dst = pull(name)[0]
        tp = os.path.join(dst, "timing.json")
        ref = kaggle_timing_record()
        if os.path.exists(tp):
            tr = json.load(open(tp))
            same = {k: tr.get(k) == v for k, v in (ref or {}).items() if k != "secs"}
            print("timing episode", TIMING_KEY, "colab secs", tr.get("secs"), "kaggle secs", (ref or {}).get("secs"),
                  "bit-identical outcome vs Kaggle:", all(same.values()) if same else None, same, flush=True)
    finally:
        stop(name)
        print(f"smoke total {(time.time() - t0) / 60:.1f} min", flush=True)


def _opt(rest, flag, default=None, cast=str):
    """Pop ``flag VALUE`` from rest."""
    if flag in rest:
        i = rest.index(flag)
        v = cast(rest[i + 1])
        del rest[i:i + 2]
        return v
    return default


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "launch":
        flags = {"--smoke", "--no-schedule", "--no-auto-resume"}
        rest = [x for x in a[1:] if x not in flags]
        parts = _opt(rest, "--parts", None, lambda v: v.split(","))
        every = _opt(rest, "--every", 10, int)
        mx = _opt(rest, "--max-resumes", 3, int)
        launch(rest[0], rest[1], rest[2], parts=parts, smoke="--smoke" in a, sched="--no-schedule" not in a,
               every=every, auto_resume="--no-auto-resume" not in a, max_resumes=mx)
    elif cmd in ("status", "pull", "stop", "tick", "resume", "unschedule"):
        {"status": status, "pull": pull, "stop": stop, "tick": tick, "resume": resume,
         "unschedule": unschedule}[cmd](a[1])
    elif cmd == "schedule":
        rest = a[1:]
        every = _opt(rest, "--every", 10, int)
        print(schedule(rest[0], every))
    elif cmd == "job":
        if len(a) < 2 or a[1].startswith("--"):
            raise SystemExit(__doc__)
        jname, rest, kw, paths = a[1], a[2:], {}, []
        single = {"--cmd": "cmd", "--out-dir": "out_dir", "--every": "every", "--threads": "threads",
                  "--pins": "pins", "--imports": "imports"}
        i = 0
        while i < len(rest):
            x = rest[i]
            if x == "--paths":
                i += 1
                while i < len(rest) and not rest[i].startswith("--"):
                    paths.append(rest[i])
                    i += 1
                continue
            if x in single:
                kw[single[x]] = rest[i + 1]
                i += 2
                continue
            if x in ("--no-schedule", "--force"):
                kw[x[2:].replace("-", "_")] = True
                i += 1
                continue
            raise SystemExit(f"unknown job argument {x!r}\n{__doc__}")
        if not paths or not kw.get("cmd"):
            raise SystemExit("job needs --paths ... and --cmd")
        job(jname, paths, kw["cmd"], out_dir=kw.get("out_dir"), sched=not kw.get("no_schedule"),
            every=int(kw.get("every", 10)), threads=int(kw["threads"]) if kw.get("threads") else None,
            pins=tuple(x for x in kw["pins"].split(",") if x) if "pins" in kw else JOB_PINS,
            imports=tuple(x for x in kw.get("imports", "").split(",") if x), force=bool(kw.get("force")))
    elif cmd == "config":
        config(*a[1:])
    elif cmd == "smoke":
        smoke(*a[1:])
    elif cmd == "probe":
        probe(*a[1:])
    else:
        raise SystemExit(__doc__)
