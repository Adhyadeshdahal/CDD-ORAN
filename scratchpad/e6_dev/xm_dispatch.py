"""Unattended R-46 dispatcher for a campaign spec split into P parts: each tick (Windows Task Scheduler, every 15 min)
launches the next chunk of parts on the first platform with room, at most one launch per tick. Lightning is never
used (credits: ask first).

  .venv/Scripts/python.exe scratchpad/e6_dev/xm_dispatch.py init NAME --spec S --cost-table T --parts P
                           [--kaggle-chunk 8] [--colab-chunk 4] [--skip-complete-from GLOB ...]
  .venv/Scripts/python.exe scratchpad/e6_dev/xm_dispatch.py tick NAME      # one launch at most
  .venv/Scripts/python.exe scratchpad/e6_dev/xm_dispatch.py schedule NAME [MIN] | unschedule NAME | show NAME

Room: Kaggle = fewer than 5 sessions RUNNING / QUEUED on the account AND fewer than 4 of them mine (xm-dev-*;
the 5th slot is for Exp B P1); Colab = fewer than 2 of my jobs (runs/*/colab.json without finished_utc /
lost_utc); VPS (init --vps) = no unpulled VPS job of this queue and no 30 min backoff after a vps_run safety
refusal. Kaggle first (long unattended), then the VPS (7 parts per job), then Colab (plain CPU runtime, 2 vCPU). Jobs are named
NAME-k<i> / NAME-c<i>; state + log in runs/NAME/ (dispatch.json, dispatch.log). All parts launched -> unschedule.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
KAGGLE = os.path.expanduser("~/.cloudtools/Scripts/kaggle.exe")
KAGGLE_MAX, KAGGLE_MINE_MAX, COLAB_MAX = 5, 4, 2   # 4 mine; 5th Kaggle slot = Exp B P1 (2026-10-04)
VPS_CHUNK = 7                                      # user VPS: one cdd-xm job at a time, 7 processes (vps_run.py)
VPS_RUN = os.path.join(HERE, "vps_run.py")


def _state_path(name):
    return os.path.join(HERE, "runs", name, "dispatch.json")


def _load(name):
    return json.load(open(_state_path(name), encoding="utf-8"))


def _save(name, st):
    os.makedirs(os.path.dirname(_state_path(name)), exist_ok=True)
    json.dump(st, open(_state_path(name), "w", encoding="utf-8", newline="\n"), indent=1)


def _log(name, event, info):
    with open(os.path.join(HERE, "runs", name, "dispatch.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {event} {json.dumps(info)}\n")


def kaggle_running() -> tuple[int, int]:
    out = subprocess.run([KAGGLE, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "20"],
                         capture_output=True, text=True, timeout=180, encoding="utf-8",
                         errors="replace").stdout
    refs = [ln.split()[0] for ln in out.splitlines()[2:] if ln.strip()]
    n = mine = 0
    for r in refs:
        s = subprocess.run([KAGGLE, "kernels", "status", r], capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace").stdout
        if "RUNNING" in s or "QUEUED" in s:
            n += 1
            mine += r.split("/")[-1].startswith("xm-dev-")
    return n, mine


def colab_active() -> int:
    n = 0
    for f in glob.glob(os.path.join(HERE, "runs", "*", "colab.json")):
        m = json.load(open(f, encoding="utf-8"))
        done = os.path.exists(os.path.join(os.path.dirname(f), "exit_code"))      # final pull landed
        if m.get("kind") == "job" and not m.get("finished_utc") and not m.get("lost_utc") and m.get("launched")                 and not done:
            n += 1
    return n


def _outputs(j) -> list[str]:
    sub = "xm_out" if j["platform"] == "colab" else "out"            # kaggle / lightning pulls: runs/JOB/out
    return sorted(glob.glob(os.path.join(HERE, "runs", j["job"], sub, "res_*.jsonl")))


def _done_dir(name) -> str:
    return os.path.join(HERE, "runs", name, "done")


def check_job(a, st, j) -> list[int]:
    """After a job is pulled (finished or lost): slim done-key copies of its records go to runs/NAME/done/ (the
    --skip-complete-from input of requeues), and every part of the job whose datasets are not all complete in the
    union of done files is requeued (resume from pulled partials; at most 3 attempts per part)."""
    sys.path.insert(0, ROOT)
    from cdd_oran.xmethod import campaign as C
    dd = _done_dir(a.name)
    os.makedirs(dd, exist_ok=True)
    for f in _outputs(j):
        with open(os.path.join(dd, f"{j['job']}__{os.path.basename(f)}"), "w", encoding="utf-8", newline="\n") as fo:
            for r in C.iter_jsonl(f):
                fo.write(json.dumps({"key": r.get("key"), "status": r.get("status"), "error": r.get("error")}) + "\n")
    if j.get("spec"):                                      # a priority job of another spec: report only
        sp = json.load(open(os.path.join(ROOT, j["spec"]), encoding="utf-8"))
        ups = C.partition(C.expand(sp), j["nparts"], None)
        got = {r.get("key") for f in _outputs(j) for r in C.iter_jsonl(f) if r.get("status") in ("ok", "infeasible")}
        j["incomplete_parts"] = [i for i in j["parts"] if {u.key for u in ups[i]} - got]
        _log(a.name, "checked", {"job": j["job"], "incomplete_parts": j["incomplete_parts"], "priority": True})
        return j["incomplete_parts"]
    spec = json.load(open(os.path.join(ROOT, st["spec"]), encoding="utf-8"))
    ct = json.load(open(os.path.join(ROOT, st["cost_table"]), encoding="utf-8"))
    parts = C.partition(C.expand(spec), st["parts"], ct)
    files = sorted(glob.glob(os.path.join(dd, "*.jsonl")))
    inc = []
    for i in j["parts"]:
        left = {u.key for u in parts[i]} - C.complete_dataset_keys(parts[i], files)
        if left:
            inc.append(i)
    tries = st.setdefault("part_attempts", {})
    queued = set(st.setdefault("requeue", []))
    for i in inc:
        tries[str(i)] = tries.get(str(i), 1) + 1
        if tries[str(i)] <= 3 and i not in queued:
            st["requeue"].append(i)
    j["incomplete_parts"] = inc
    _log(a.name, "checked", {"job": j["job"], "incomplete_parts": inc, "requeue": st["requeue"]})
    return inc


def _launch(st, platform, ids, job, skip_from=None, pri=None):
    """``pri``: a priority job {"spec", "nparts"} (own spec, no cost table, no skip)."""
    if pri:
        cmd = [PY, "-m", "cdd_oran.xmethod.campaign", platform, "--spec", pri["spec"], "--parts", str(pri["nparts"]),
               "--part-set", ",".join(map(str, ids)), "--name", job, "--venv-python", "3.12", "--selftest"]
    else:
        cmd = [PY, "-m", "cdd_oran.xmethod.campaign", platform, "--spec", st["spec"], "--cost-table",
               st["cost_table"], "--parts", str(st["parts"]), "--part-set", ",".join(map(str, ids)), "--name", job,
               "--venv-python", "3.12", "--selftest"]
    skip = [] if pri else list(st.get("skip_from") or []) + list(skip_from or [])
    if skip:
        rel = [os.path.relpath(g, ROOT).replace("\\", "/") if os.path.isabs(g) else g for g in skip]
        cmd += ["--paths", *sorted({os.path.dirname(g) for g in rel}), "--skip-complete-from", *rel]
    if platform == "colab":
        cmd += ["--wall-s", "39600"]
    if platform == "vps":
        cmd += ["--procs", str(min(VPS_CHUNK, len(ids)))]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **({"COLAB_ACCEL": "cpu"} if platform == "colab" else {})}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=2400, env=env, encoding="utf-8",
                       errors="replace")
    o = r.stdout + r.stderr
    if r.returncode == 0:
        return 0, o[-3000:]
    keep = [ln for ln in o.splitlines() if not ln.startswith("[xm-c]") and len(ln) < 600]   # drop the long cmd echo
    return r.returncode, "\n".join(keep)[-3000:]


def pull_finished(a, st) -> bool:
    """Pull every finished Kaggle job of this queue once (kaggle_job.py pull -> runs/JOB/out); Colab jobs pull
    themselves (their own ticks). True when every launched job is finished and pulled."""
    all_done = True
    for j in st["jobs"]:
        if j.get("pulled"):
            continue
        if j["platform"] == "vps":                         # pull every tick (append-only); done = exit_code pulled
            r = subprocess.run([PY, VPS_RUN, "pull", j["job"]], cwd=ROOT, capture_output=True, text=True,
                               timeout=1800, encoding="utf-8", errors="replace")
            ec = os.path.join(HERE, "runs", j["job"], "exit_code")
            if r.returncode == 0 and os.path.exists(ec):
                j["pulled"] = "vps exit " + open(ec).read().strip()
                _log(a.name, "pulled", {"job": j["job"], "state": j["pulled"]})
                check_job(a, st, j)
            else:
                all_done = False
            continue
        if j["platform"] == "lightning":                   # its own tick pulls + stops it (balance guard)
            m = os.path.join(HERE, "runs", j["job"], "lightning.json")
            mm = json.load(open(m, encoding="utf-8")) if os.path.exists(m) else {}
            if mm.get("stopped_utc"):
                j["pulled"] = "lightning stopped"
                _log(a.name, "lightning-done", {"job": j["job"], "balance_after": mm.get("balance_after")})
                check_job(a, st, j)
            else:
                all_done = False
            continue
        if j["platform"] == "colab":
            m = os.path.join(HERE, "runs", j["job"], "colab.json")
            mm = json.load(open(m, encoding="utf-8")) if os.path.exists(m) else {}
            done = mm.get("finished_utc") or mm.get("lost_utc") or os.path.exists(
                os.path.join(HERE, "runs", j["job"], "exit_code"))
            if done:
                j["pulled"] = "colab " + ("lost" if mm.get("lost_utc") else "finished")
                _log(a.name, "colab-done", {"job": j["job"], "state": j["pulled"]})
                check_job(a, st, j)
            else:
                all_done = False
            continue
        out = subprocess.run([KAGGLE, "kernels", "status", f"bishalpanta/{j['job']}"], capture_output=True, text=True,
                             timeout=120, encoding="utf-8", errors="replace").stdout or ""
        if "RUNNING" in out or "QUEUED" in out or "Status" not in out:
            all_done = False
            continue
        r = subprocess.run([PY, os.path.join(HERE, "kaggle_job.py"), "pull", j["job"]], cwd=ROOT, capture_output=True,
                           text=True, timeout=1800, encoding="utf-8", errors="replace")
        state = out.strip().split()[-1].strip('"')
        _log(a.name, "pulled", {"job": j["job"], "state": state, "rc": r.returncode, "tail": r.stdout[-200:]})
        if r.returncode == 0:
            j["pulled"] = state
            check_job(a, st, j)
        else:
            all_done = False
    _save(a.name, st)
    return all_done


class _TickLock:
    """One tick at a time per queue (a scheduled and a manual tick once launched the same job name twice, and the
    failing one's state save dropped the running job). Stale after 2 h."""

    def __init__(self, name):
        self.p = os.path.join(HERE, "runs", name, "dispatch.lock")
        self.own = False

    def __enter__(self):
        try:
            if os.path.exists(self.p) and time.time() - os.path.getmtime(self.p) > 7200:
                os.remove(self.p)
            fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self.own = True
        except FileExistsError:
            pass
        return self.own

    def __exit__(self, *exc):
        if self.own:
            os.remove(self.p)


def _job_name(a, st, kind):
    """A slug never used before: attempt counter + UTC day-hour-minute (Kaggle slugs and NAME-code datasets are
    unique per account, and a failed push can leave its dataset behind)."""
    att = st.setdefault("attempts", {})
    att[kind] = att.get(kind, 0) + 1
    return f"{a.name}-{kind[0]}{att[kind]}-{time.strftime('%d%H%M', time.gmtime())}"


def tick(a):
    with _TickLock(a.name) as mine:
        if not mine:
            _log(a.name, "tick-skipped", {"why": "another tick holds runs/NAME/dispatch.lock"})
            return
        _tick(a)


def _tick(a):
    st = _load(a.name)
    for j in st["jobs"]:                                   # jobs pulled before check_job existed
        if j.get("pulled") and "incomplete_parts" not in j:
            check_job(a, st, j)
    _save(a.name, st)
    done = pull_finished(a, st)
    rq = st.get("requeue") or []
    pq = st.get("priority") or []
    if st["next"] >= st.get("parts_end", st["parts"]) and not rq and not pq:
        _log(a.name, "all-launched", {"all_pulled": done})
        if done:
            unschedule(a)
        return
    kn, km = kaggle_running()
    ca = colab_active()
    info = {"kaggle_running": kn, "kaggle_mine": km, "colab_active": ca, "next": st["next"], "requeue": rq}
    if pq:                                                 # priority jobs (another spec) go first, Kaggle only
        if kn < KAGGLE_MAX and km < KAGGLE_MINE_MAX:
            pri = pq[0]
            job = _job_name(a, st, pri.get("kind", "pilot"))
            _save(a.name, st)
            ids = list(range(pri["nparts"]))
            rc, tail = _launch(st, "kaggle", ids, job, pri=pri)
            _log(a.name, "priority-launch", {**info, "job": job, "spec": pri["spec"], "rc": rc,
                                             "tail": tail[-300:] if rc == 0 else tail})
            if rc == 0:
                st["jobs"].append({"job": job, "platform": "kaggle", "parts": ids, "spec": pri["spec"],
                                   "nparts": pri["nparts"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                st["priority"] = pq[1:]
                _save(a.name, st)
        else:
            _log(a.name, "no-room-priority", info)
        return                                             # nothing else launches before it
    vps_free = st.get("vps", False) and not any(j["platform"] == "vps" and not j.get("pulled") for j in st["jobs"])         and time.time() >= st.get("vps_backoff_until", 0)
    if rq:                                                 # resume lost / cut parts first: Kaggle, else the VPS
        kag = kn < KAGGLE_MAX and km < KAGGLE_MINE_MAX     # (never Colab: it reclaims runtimes after ~7 h)
        if kag or vps_free:
            plat = "kaggle" if kag else "vps"
            ids = rq[:st["kaggle_chunk"] if kag else VPS_CHUNK]
            job = _job_name(a, st, "requeue")
            _save(a.name, st)
            rc, tail = _launch(st, plat, ids, job, skip_from=[os.path.join(_done_dir(a.name), "*.jsonl")])
            _log(a.name, "requeue-launch", {**info, "job": job, "parts": ids, "rc": rc,
                                            "tail": tail[-300:] if rc == 0 else tail})
            if rc == 0:
                st["jobs"].append({"job": job, "platform": plat, "parts": ids, "requeue": True,
                                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                st["requeue"] = [i for i in rq if i not in ids]
                _save(a.name, st)
            return                                         # one launch per tick
        if st["next"] >= st.get("parts_end", st["parts"]):
            _log(a.name, "no-room-requeue", info)
            return
    vps_ok = st.get("vps", False) and not any(j["platform"] == "vps" and not j.get("pulled") for j in st["jobs"])         and time.time() >= st.get("vps_backoff_until", 0)
    for platform, ok, chunk in (("kaggle", kn < KAGGLE_MAX and km < KAGGLE_MINE_MAX, st["kaggle_chunk"]),
                                ("vps", vps_ok, VPS_CHUNK),
                                ("colab", ca < COLAB_MAX and time.time() >= st.get("colab_backoff_until", 0),
                                 st["colab_chunk"])):
        if not ok:
            continue
        ids = list(range(st["next"], min(st["next"] + chunk, st.get("parts_end", st["parts"]))))
        job = _job_name(a, st, platform)
        _save(a.name, st)
        rc, tail = _launch(st, platform, ids, job)
        _log(a.name, "launch", {**info, "platform": platform, "job": job, "parts": [ids[0], ids[-1]], "rc": rc,
                                "tail": tail[-300:] if rc == 0 else tail})  # full tail on failure (cause)
        if rc != 0 and platform == "colab" and "Allocation refused" in tail:   # free-tier limit: back off 2 h
            st["colab_backoff_until"] = time.time() + 7200
            _save(a.name, st)
        if rc != 0 and platform == "vps" and "refused:" in tail:              # VPS safety refusal: back off 30 min
            st["vps_backoff_until"] = time.time() + 1800
            _save(a.name, st)
        if rc == 0:
            st["jobs"].append({"job": job, "platform": platform, "parts": ids,
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            st["next"] = ids[-1] + 1
            _save(a.name, st)
            return                                         # one launch per tick
    _log(a.name, "no-room", info)


TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>CDD-ORAN xm_dispatch.py tick {name}</Description></RegistrationInfo>
  <Triggers><TimeTrigger><Repetition><Interval>PT{every}M</Interval><StopAtDurationEnd>false</StopAtDurationEnd>
  </Repetition><StartBoundary>{start}</StartBoundary><Enabled>true</Enabled></TimeTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType>
  <RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
  <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
  <StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT1H</ExecutionTimeLimit><Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{conhost}</Command><Arguments>--headless cmd.exe /c "{cmd}"</Arguments>
  <WorkingDirectory>{root}</WorkingDirectory></Exec></Actions>
</Task>
"""


def schedule(a):
    d = os.path.join(HERE, "runs", a.name)
    cmd, out = os.path.join(d, "dispatch.cmd"), os.path.join(d, "dispatch.out")
    open(cmd, "w", newline="\r\n").write("\n".join([
        "@echo off", 'set "PYTHONIOENCODING=utf-8"', 'set "MSYS_NO_PATHCONV=1"', f'cd /d "{ROOT}"',
        f'echo ==== %DATE% %TIME% >> "{out}"', f'"{PY}" "{os.path.abspath(__file__)}" tick {a.name} >> "{out}" 2>&1'])
        + "\n")
    xml = os.path.join(d, "dispatch_task.xml")
    user = "\\".join(x for x in (os.environ.get("USERDOMAIN", ""), os.environ.get("USERNAME", "")) if x)
    every = int(a.every)
    open(xml, "w", encoding="utf-16").write(TASK_XML.format(
        name=a.name, every=every, start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 60)),
        user=user, conhost=os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "conhost.exe"),
        cmd=cmd, root=ROOT))
    r = subprocess.run(["schtasks", "/create", "/tn", f"CDD-ORAN-dispatch-{a.name}", "/xml", xml, "/f"],
                       capture_output=True, text=True, errors="replace")
    _log(a.name, "scheduled", {"rc": r.returncode, "every_min": every})
    print(r.returncode, (r.stdout + r.stderr).strip())


def unschedule(a):
    r = subprocess.run(["schtasks", "/delete", "/tn", f"CDD-ORAN-dispatch-{a.name}", "/f"], capture_output=True,
                       text=True, errors="replace")
    _log(a.name, "unscheduled", {"rc": r.returncode})


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("init")
    p.add_argument("name")
    p.add_argument("--spec", required=True)
    p.add_argument("--cost-table", required=True)
    p.add_argument("--parts", type=int, required=True)
    p.add_argument("--kaggle-chunk", type=int, default=8)
    p.add_argument("--colab-chunk", type=int, default=4)
    p.add_argument("--skip-complete-from", nargs="*", default=None)
    p.add_argument("--vps", action="store_true", help="also use the user VPS lane (one 7-part job at a time)")
    for c in ("tick", "unschedule", "show"):
        sub.add_parser(c).add_argument("name")
    p = sub.add_parser("schedule")
    p.add_argument("name")
    p.add_argument("every", nargs="?", default=15)
    a = ap.parse_args()
    if a.action == "init":
        _save(a.name, {"spec": a.spec, "cost_table": a.cost_table, "parts": a.parts, "kaggle_chunk": a.kaggle_chunk,
                       "colab_chunk": a.colab_chunk, "skip_from": a.skip_complete_from, "vps": a.vps, "next": 0,
                       "jobs": []})
        _log(a.name, "init", _load(a.name))
    elif a.action == "show":
        print(json.dumps(_load(a.name), indent=1))
    else:
        {"tick": tick, "schedule": schedule, "unschedule": unschedule}[a.action](a)   # noqa: E501
    return 0


if __name__ == "__main__":
    sys.exit(main())
