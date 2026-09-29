"""Colab runner for e6dev grid scripts (DEV scratch): the Colab counterpart of kaggle_run.py.

  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py launch NAME SCRIPT P [--parts 0,1,...] [--smoke]
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py status NAME
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py pull NAME        # -> scratchpad/e6_dev/runs/NAME/all.jsonl
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py stop NAME        # stop + verify 0 active assignments
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py smoke NAME [SCRIPT=e6p_v2_stage2.py] [MINUTES=8]
  .venv/Scripts/python.exe scratchpad/e6_dev/colab_run.py probe NAME [NS=1,4,8,12,16,24]   # CPU scaling only

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
    assignment the CLI does not know (orphan: stop it by hand)."""
    for k in range(tries):
        try:
            out = colab("new", "-s", name, "--tpu", "v5e1", timeout=HTTP_TIMEOUT + 300)
            print(out[-300:], flush=True)
            if "READY" in out:
                return
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"new attempt {k + 1} failed: {str(e)[-300:]}", flush=True)
        n, _ = active_assignments()
        if n:
            ses = colab("sessions", timeout=120, check=False)
            if f"[{name}]" in ses:
                return
            raise SystemExit(f"{n} active assignment(s) not known as session {name!r}: {ses}")
        time.sleep(60)
    raise SystemExit(f"could not create a TPU v5e-1 runtime after {tries} attempts")


def meta_path(name):
    return os.path.join(HERE, "runs", name, "colab.json")


def load_meta(name):
    return json.load(open(meta_path(name)))


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
pids = {}
for i in PARTS:
    cmd = ["nohup", sys.executable, "-u", os.path.join(root, "e6dev", SCRIPT), "run", "--part", f"{i}/{P}",
           "--out", os.path.join(W, f"res_{i}.jsonl")] + (["--smoke"] if SMOKE else [])
    log = open(os.path.join(W, f"log_{i}.txt"), "a")
    pids[i] = subprocess.Popen(cmd, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True).pid
json.dump({"script": SCRIPT, "P": P, "parts": PARTS, "smoke": SMOKE, "pids": pids},
          open(os.path.join(W, "launch.json"), "w"))
print("LAUNCHED " + json.dumps(pids), flush=True)
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


def _fill(src, **kw):
    for k, v in kw.items():
        src = src.replace(f"__{k}__", repr(v))
    return src


# ---------------------------------------------------------------------------------------------- commands
def launch(name, script, P, parts=None, smoke=False, timing=False, probe=(), reuse=False):
    P = int(P)
    parts = list(range(P)) if parts is None else [int(x) for x in parts]
    assert all(0 <= i < P for i in parts), parts
    out = os.path.join(tempfile.gettempdir(), "e6_colab", name)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    import cloud
    cloud.main(out, name, script, "1", "match")                     # the SAME bundle as the Kaggle path
    bundle = os.path.join(out, "dataset", f"{name}_bundle.zip")
    import zipfile
    man = json.loads(zipfile.ZipFile(bundle).read("MANIFEST.json"))
    ref = kaggle_reference(man["local_env"]["numpy"])
    os.makedirs(os.path.dirname(meta_path(name)), exist_ok=True)
    meta = {"session": name, "script": script, "P": P, "parts": parts, "smoke": smoke, "bundle": bundle,
            "git_head": man["git_head"], "e6_dirty": man["e6_dirty"], "kaggle_ref": ref[0] if ref else None,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(meta_path(name), "w"), indent=1)
    t0 = time.time()
    if not reuse:
        new_session(name)
    meta["t_new"] = round(time.time() - t0, 1)
    rb = f"/content/{name}_bundle.zip"
    colab("upload", "-s", name, bundle, rb, timeout=900)
    t1 = time.time()
    start = remote(name, _fill(SETUP, W=W, BUNDLE=rb, KREF=ref[1] if ref else None,
                               KREF_SRC=os.path.relpath(ref[0], HERE) if ref else None), timeout=1200, tag="STARTUP")
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
    meta["launched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(meta, open(meta_path(name), "w"), indent=1)
    print("launched parts", parts, "of", P, "->", meta["launched"], flush=True)
    return meta


def status(name, quiet=False):
    st = remote(name, _fill(STATUS, W=W), timeout=300, tag="STATUS")
    if not quiet:
        print(json.dumps({k: st[k] for k in ("script", "P", "n_parts", "finished", "alive", "failed", "jobs_done",
                                             "timing_done", "load")}))
        for i, p in list(st["parts"].items())[:30]:
            print(f"  part {i}: jobs={p['jobs']} closed={p['closed']} alive={p['alive']} tb={p['traceback']} "
                  f"{p['tail']}")
    return st


def pull(name):
    dst = os.path.join(HERE, "runs", name)
    os.makedirs(dst, exist_ok=True)
    remote(name, _fill(PACK, W=W), timeout=600, tag="PACKED")
    tgz = os.path.join(dst, "pull.tar.gz")
    colab("download", "-s", name, "/content/e6run_pull.tar.gz", tgz, timeout=1200)
    with tarfile.open(tgz) as t:
        t.extractall(dst, filter="data")
    os.remove(tgz)
    res = sorted(glob.glob(os.path.join(dst, "res_*.jsonl")), key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    lines = [ln if ln.endswith("\n") else ln + "\n" for f in res for ln in open(f) if ln.strip()]
    open(os.path.join(dst, "all.jsonl"), "w", newline="\n").writelines(lines)
    print(f"{len(lines)} records from {len(res)} shard files -> {os.path.join(dst, 'all.jsonl')}")
    sp = os.path.join(dst, "startup.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        print("startup", {k: s.get(k) for k in ("fp_match", "fp_layers_equal_kaggle", "ufunc_diff_vs_kaggle",
                                                "kaggle_ref")},
              {k: (s.get("env") or {}).get(k) for k in ("numpy", "scipy", "platform", "simd")})
    return dst


def stop(name, wait_s=180):
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
        dst = pull(name)
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


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "launch":
        rest = [x for x in a[1:] if x != "--smoke"]
        parts = None
        if "--parts" in rest:
            i = rest.index("--parts")
            parts = rest[i + 1].split(",")
            rest = rest[:i] + rest[i + 2:]
        launch(rest[0], rest[1], rest[2], parts=parts, smoke="--smoke" in a)
    elif cmd in ("status", "pull", "stop"):
        {"status": status, "pull": pull, "stop": stop}[cmd](a[1])
    elif cmd == "smoke":
        smoke(*a[1:])
    elif cmd == "probe":
        probe(*a[1:])
    else:
        raise SystemExit(__doc__)
