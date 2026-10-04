"""Lightning AI studio runner for xmethod campaign jobs (R-46). Called by `campaign lightning` (which builds the shell
command: lock install + pin check + the campaign parts, exactly as for Kaggle / Colab); run it with the interpreter
that has lightning_sdk (~/.cloudtools/Scripts/python.exe).

  launch NAME --machine CPU --out-dir xm_out --cmd "SHELL" --paths P1 P2 ...  # bundle, upload, start unattended
  status NAME          # studio state, live processes, records so far, log tails, credit balance
  pull NAME            # out dir (+ job.log) -> scratchpad/e6_dev/runs/NAME/out/ (incremental: whole dir each time)
  stop NAME            # stop the studio; logs the balance (runs/NAME/lightning.json)
  credits              # account balance
  tick NAME            # one-shot (Task Scheduler, every 15 min): pull; job exited -> final pull, stop, unschedule
  schedule NAME [MIN] | unschedule NAME

Rules (R-44 / R-46): CPU machines unless budgeted; never override auto-sleep; stop the studio after the final pull;
log credits per job. The balance is per account (shared with other workers), so a job's credit use is estimated as
studio-hours x the machine rate and the balance before / after is logged next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile

from lightning_sdk import Machine, Studio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEAMSPACE, USER = "default-project", "bishalpanta01"
RATE = {"CPU": 0.15, "CPU_SMALL": 0.075}            # credits per studio-hour (observed burn 2026-10-03, cpu-4)
SKIP = ("__pycache__", ".pytest_cache")


def studio(name: str) -> Studio:
    return Studio(name=name, teamspace=TEAMSPACE, user=USER, create_ok=True)


def balance() -> float | None:
    try:
        from lightning_sdk.lightning_cloud.rest_client import LightningClient
        return float(LightningClient(retry=False).billing_service_get_user_balance().balance)
    except Exception as e:                                       # noqa: BLE001 (logged, never fatal)
        print(f"balance unavailable: {e}")
        return None


def meta_path(name: str) -> str:
    return os.path.join(HERE, "runs", name, "lightning.json")


def load_meta(name: str) -> dict:
    p = meta_path(name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def save_meta(name: str, m: dict) -> None:
    os.makedirs(os.path.dirname(meta_path(name)), exist_ok=True)
    json.dump(m, open(meta_path(name), "w", encoding="utf-8", newline="\n"), indent=1)


def files_of(paths: list[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        a = os.path.abspath(os.path.join(ROOT, p))
        if not a.startswith(os.path.abspath(ROOT)):
            sys.exit(f"--paths must be inside the repo: {p}")
        if os.path.isdir(a):
            for dp, dns, fns in os.walk(a):
                dns[:] = sorted(d for d in dns if d not in SKIP)
                for fn in sorted(fns):
                    if not fn.endswith((".pyc", ".pyo")):
                        f = os.path.join(dp, fn)
                        out[os.path.relpath(f, ROOT).replace("\\", "/")] = f
        elif os.path.isfile(a):
            out[os.path.relpath(a, ROOT).replace("\\", "/")] = a
        else:
            sys.exit(f"--paths: nothing at {p!r}")
    return out


def bundle(name: str, paths: list[str]) -> str:
    """Zip of the repo-relative paths (CRLF -> LF for text) + MANIFEST.json (git head, dirty, sha256 per file)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    blobs = {}
    for rel, f in files_of(paths).items():
        b = open(f, "rb").read()
        blobs[rel] = b.replace(b"\r\n", b"\n") if rel.endswith((".py", ".json", ".txt", ".md", ".yaml", ".yml",
                                                                   ".sh", ".toml", ".cfg")) else b
    man = {"name": name, "git_head": git("rev-parse", "HEAD"),
           "dirty": bool(git("status", "--porcelain", "cdd_oran", "scratchpad/xmethod/specs")),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sha256": {r: hashlib.sha256(b).hexdigest() for r, b in blobs.items()}}
    z = os.path.join(HERE, "runs", name, f"{name}_bundle.zip")
    os.makedirs(os.path.dirname(z), exist_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, b in blobs.items():
            zf.writestr(rel, b)
        zf.writestr("MANIFEST.json", json.dumps(man, indent=1))
    return z


def launch(a) -> None:
    rd = f"xm_{a.name}"
    z = bundle(a.name, a.paths)
    job = os.path.join(HERE, "runs", a.name, "job.sh")
    open(job, "w", encoding="utf-8", newline="\n").write(
        f"#!/bin/bash\ncd \"$HOME/{rd}/b\" || exit 9\nexport PYTHONPATH=.\n{a.cmd}\necho $? > \"$HOME/{rd}/exit_code\"\n")
    s = studio(a.name)
    b0 = balance()
    if "running" not in str(s.status).lower():
        s.start(getattr(Machine, a.machine))
    s.upload_file(z, f"{rd}_bundle.zip")                    # flat remote names (Windows separators otherwise)
    s.upload_file(job, f"{rd}_job.sh")
    print(s.run(f"cd $HOME && rm -rf {rd} && mkdir -p {rd} && mv {rd}_bundle.zip {rd}/bundle.zip && "
                f"mv {rd}_job.sh {rd}/job.sh && cd {rd} && python3 -m zipfile -e bundle.zip b && nproc && "
                f"python3 --version").strip()[-300:])
    s.run(f"cd $HOME/{rd} && setsid nohup bash job.sh > job.log 2>&1 < /dev/null &")
    time.sleep(5)
    print(s.run(f"cd $HOME/{rd} && tail -n 3 job.log; ls b/{a.out_dir} 2>/dev/null | head").strip()[-600:])
    m = {"name": a.name, "machine": a.machine, "rate_credits_per_h": RATE.get(a.machine), "out_dir": a.out_dir,
         "remote_dir": rd, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "started_epoch": time.time(),
         "balance_before": b0, "min_balance": a.min_balance,
         "bundle_sha256": hashlib.sha256(open(z, "rb").read()).hexdigest()}
    save_meta(a.name, m)
    print(json.dumps(m, indent=1))


def status(a) -> None:
    m, s = load_meta(a.name), studio(a.name)
    rd, od = m.get("remote_dir", f"xm_{a.name}"), m.get("out_dir", "xm_out")
    print("studio", s.status)
    if "running" in str(s.status).lower():
        print(s.run(f"cd $HOME/{rd} && echo exit_code=$(cat exit_code 2>/dev/null) && echo procs=$(pgrep -fc "
                    f"'cdd_oran.xmethod.campaign run') && cat b/{od}/res_*.jsonl 2>/dev/null | wc -l && "
                    f"grep -ho ' ok \\| error \\| infeasible ' b/{od}/log_*.txt 2>/dev/null | sort | uniq -c; "
                    f"tail -qn 1 b/{od}/log_*.txt 2>/dev/null | tail -4; tail -n 2 job.log").strip()[-1500:])
    if m.get("started_epoch"):
        h = (time.time() - m["started_epoch"]) / 3600
        print(f"elapsed {h:.2f} h, est. credits {h * (m.get('rate_credits_per_h') or 0):.3f}, balance {balance()}")


def pull(a) -> None:
    m, s = load_meta(a.name), studio(a.name)
    rd, od = m.get("remote_dir", f"xm_{a.name}"), m.get("out_dir", "xm_out")
    s.run(f"cd $HOME/{rd} && cp job.log b/{od}/job.log 2>/dev/null; cp exit_code b/{od}/ 2>/dev/null; "
          f"tar czf $HOME/{rd}_out.tgz -C b/{od} .")
    dst = os.path.join(HERE, "runs", a.name)
    os.makedirs(dst, exist_ok=True)
    t = os.path.join(dst, "out.tgz")
    for k in range(8):                                   # the studio's files reach storage with a delay
        try:
            s.download_file(f"{rd}_out.tgz", t)
            break
        except RuntimeError:
            if k == 7:
                raise
            time.sleep(20)
    with tarfile.open(t) as tf:
        tf.extractall(os.path.join(dst, "out"))
    n = sum(sum(1 for _ in open(os.path.join(dst, "out", f), encoding="utf-8"))
            for f in os.listdir(os.path.join(dst, "out")) if f.endswith(".jsonl") and f.startswith(("res_", "eval_res_")))
    print(f"{n} records -> {os.path.join(dst, 'out')}")


def stop(a) -> None:
    m, s = load_meta(a.name), studio(a.name)
    s.stop()
    h = (time.time() - m["started_epoch"]) / 3600 if m.get("started_epoch") else None
    m.update(stopped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), studio_hours=h,
             credits_est=(h * (m.get("rate_credits_per_h") or 0)) if h else None, balance_after=balance())
    save_meta(a.name, m)
    print("stopped", s.status, json.dumps({k: m.get(k) for k in ("studio_hours", "credits_est", "balance_before",
                                                                 "balance_after")}))


def _log(name: str, event: str, info: dict) -> None:
    os.makedirs(os.path.join(HERE, "runs", name), exist_ok=True)
    with open(os.path.join(HERE, "runs", name, "tick.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {event} {json.dumps(info)}\n")


def tick(a) -> None:
    """Unattended: studio running + job alive -> pull; job exited (exit_code) or no campaign process left ->
    final pull, stop the studio, unschedule; studio not running -> log it, pull nothing, unschedule (look by hand)."""
    m, s = load_meta(a.name), studio(a.name)
    rd = m.get("remote_dir", f"xm_{a.name}")
    st = str(s.status)
    b = balance()
    m.setdefault("balance_log", []).append([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), b])
    save_meta(a.name, m)
    est = (time.time() - m.get("started_epoch", time.time())) / 3600 * (m.get("rate_credits_per_h") or 0)
    budget = (m["balance_before"] - m["min_balance"]) if (m.get("min_balance") is not None
                                                         and m.get("balance_before") is not None) else None
    low = b is not None and m.get("min_balance") is not None and b < m["min_balance"]
    spent = budget is not None and est >= budget          # the balance API can lag for hours: estimated spend too
    if "running" in st.lower() and (low or spent):
        _log(a.name, "low-balance", {"balance": b, "min_balance": m["min_balance"], "est_spent": round(est, 3),
                                     "budget": budget})   # final pull, then stop
        try:
            pull(a)
        except Exception as e:                           # noqa: BLE001
            _log(a.name, "pull-failed", {"error": str(e)[:200]})
        stop(a)
        unschedule(a)
        return
    if "running" not in st.lower():
        _log(a.name, "studio-not-running", {"status": st})
        unschedule(a)
        return
    out = s.run(f"cd $HOME/{rd} && echo \"ec=$(cat exit_code 2>/dev/null) procs=$(pgrep -fc "
                f"'cdd_oran.xmethod.campaign run')\"").strip().splitlines()[-1]
    kv = dict(x.split("=", 1) for x in out.split() if "=" in x)
    try:
        pull(a)
    except Exception as e:                               # noqa: BLE001 (next tick retries; never block the stop)
        _log(a.name, "pull-failed", {"error": str(e)[:200]})
    done = kv.get("ec", "") != "" or (kv.get("procs") == "0" and time.time() - m.get("started_epoch", 0) > 1800)
    _log(a.name, "tick", {**kv, "done": done})
    if done:
        stop(a)
        unschedule(a)
        _log(a.name, "finished", {k: load_meta(a.name).get(k) for k in ("studio_hours", "credits_est", "balance_after")})


TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>CDD-ORAN xm_lightning.py tick {name}</Description></RegistrationInfo>
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


def schedule(a) -> None:
    d = os.path.join(HERE, "runs", a.name)
    os.makedirs(d, exist_ok=True)
    cmd = os.path.join(d, "tick.cmd")
    out = os.path.join(d, "tick.out")
    open(cmd, "w", newline="\r\n").write("\n".join([
        "@echo off", 'set "PYTHONIOENCODING=utf-8"', f'cd /d "{ROOT}"', f'echo ==== %DATE% %TIME% >> "{out}"',
        f'"{sys.executable}" "{os.path.abspath(__file__)}" tick {a.name} >> "{out}" 2>&1']) + "\n")
    xml = os.path.join(d, "task.xml")
    user = "\\".join(x for x in (os.environ.get("USERDOMAIN", ""), os.environ.get("USERNAME", "")) if x)
    every = int(a.every)
    open(xml, "w", encoding="utf-16").write(TASK_XML.format(
        name=a.name, every=every, start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 60 * every)),
        user=user, conhost=os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "conhost.exe"),
        cmd=cmd, root=ROOT))
    r = subprocess.run(["schtasks", "/create", "/tn", f"CDD-ORAN-lightning-{a.name}", "/xml", xml, "/f"],
                       capture_output=True, text=True, errors="replace")
    _log(a.name, "scheduled", {"rc": r.returncode, "every_min": every, "out": (r.stdout + r.stderr).strip()[-200:]})
    print(r.returncode, (r.stdout + r.stderr).strip())


def unschedule(a) -> None:
    r = subprocess.run(["schtasks", "/delete", "/tn", f"CDD-ORAN-lightning-{a.name}", "/f"], capture_output=True,
                       text=True, errors="replace")
    _log(a.name, "unscheduled", {"rc": r.returncode})


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("launch")
    p.add_argument("name")
    p.add_argument("--machine", default="CPU")
    p.add_argument("--out-dir", default="xm_out")
    p.add_argument("--cmd", required=True)
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--min-balance", type=float, default=None,
                   help="tick: final pull + stop the studio once the account balance falls below this")
    for c in ("status", "pull", "stop", "tick", "unschedule"):
        sub.add_parser(c).add_argument("name")
    p = sub.add_parser("schedule")
    p.add_argument("name")
    p.add_argument("every", nargs="?", default=15)
    sub.add_parser("credits")
    a = ap.parse_args()
    if a.action == "credits":
        print(balance())
        return 0
    {"launch": launch, "status": status, "pull": pull, "stop": stop, "tick": tick, "schedule": schedule,
     "unschedule": unschedule}[a.action](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
