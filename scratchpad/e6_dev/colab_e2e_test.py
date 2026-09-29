"""Real end-to-end test of colab_run.py launch / tick / resume / schedule on a Colab TPU v5e-1 VM (DEV scratch).

  .venv/Scripts/python.exe scratchpad/e6_dev/colab_e2e_test.py [NAME=e6colab-selftest]

2 parts of colab_selftest.py (12 x 20 s jobs each). Ticks are fired through the REAL scheduled task
(`schtasks /run`), so the tick.cmd wrapper, env and headless console are exercised. Steps: launch (auto-schedule) ->
keep-alive evidence (kernel last_activity vs usage/status/ping/exec) -> tick (partial pull) -> deliberate `colab stop`
mid-run -> tick (detect loss + resume) -> ticks until done (stop + unschedule) -> verify keys, 0 assignments, task gone.
ALWAYS stops the runtime and removes the task (try/finally).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import colab_run as C  # noqa: E402
import colab_selftest as S  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "e6colab-selftest"
T0 = time.time()
TERMINAL = {"tick", "resumed", "done", "gone-no-resume", "exec-error", "probe-error", "skip", "noop",
            "gone-but-assignments", "stop-failed", "resume-seed-mismatch", "resume-nothing"}


def say(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def log_lines():
    p = os.path.join(C.run_dir(NAME), "tick.log")
    return open(p, encoding="utf-8").read().splitlines() if os.path.exists(p) else []


def fire_and_wait(timeout_s=1800):
    n0 = len(log_lines())
    rc, out = C._schtasks("/run", "/tn", C.task_name(NAME))
    say("schtasks /run ->", rc, out[-120:])
    t = time.time()
    while time.time() - t < timeout_s:
        time.sleep(10)
        new = log_lines()[n0:]
        ev = [ln.split(" ", 2)[1] for ln in new]
        if any(e in TERMINAL for e in ev) and not os.path.exists(os.path.join(C.run_dir(NAME), "tick.lock")):
            for ln in new:
                say("  tick.log:", ln[:900])
            return ev, new
    raise RuntimeError(f"no tick result within {timeout_s} s; new lines: {log_lines()[n0:]}")


def kernels(tag):
    pr = C.session_probe(NAME, ping=False)
    ks = [(k.get("last_activity"), k.get("execution_state"), k.get("connections")) for k in pr.get("kernels", [])]
    srv = pr.get("server") or {}
    r = {"utc": C._utc(), "kernels": ks, "server_last_activity": srv.get("last_activity"),
         "server_connections": srv.get("connections"), "err": pr.get("kernels_err")}
    say(f"activity [{tag}]:", r)
    return r


REP = {"name": NAME}


def main():
    rep = REP
    n, out = C.active_assignments()
    assert n == 0, f"refusing: {n} active assignment(s) before the test:\n{out}"
    C.launch(NAME, "colab_selftest.py", 2, sched=True, every=10)
    t_launch = time.time()
    rep["launch_min"] = round((t_launch - T0) / 60, 1)

    # --- what moves the kernel's server-side last_activity?
    ev = {}
    ev["0_after_launch"] = kernels("after launch execs")
    time.sleep(20)
    C.colab("usage", timeout=120)
    C.colab("status", "-s", NAME, timeout=120, check=False)
    C.colab("sessions", timeout=120, check=False)
    ev["1_after_usage_status_sessions"] = kernels("after usage+status+sessions")
    time.sleep(20)
    pr = C.session_probe(NAME, ping=True)
    rep["keepalive_http"] = pr.get("keepalive_http", pr.get("keepalive_err"))
    say("keep-alive ping ->", rep["keepalive_http"])
    ev["2_after_keepalive_ping"] = kernels("after keep-alive ping")
    time.sleep(20)
    C.remote(NAME, "print('KA 1')", timeout=60, tag=None)
    ev["3_after_exec"] = kernels("after one exec")
    rep["last_activity_evidence"] = ev

    # --- tick 1: partial pull
    while time.time() - t_launch < 100:
        time.sleep(5)
    evs, _ = fire_and_wait()
    loc = C.local_parts(NAME, C.load_meta(NAME))
    say("local after tick 1:", loc)
    assert "tick" in evs, evs
    tot = sum(p["jobs"] for p in loc.values())
    assert 0 < tot < 24 and not all(p["closed"] for p in loc.values()), loc
    rep["tick1_local_jobs"] = {i: p["jobs"] for i, p in loc.items()}

    # --- deliberate loss mid-run
    say("STOPPING the session mid-run (simulated loss)")
    C.stop(NAME)
    rep["stopped_after_min"] = round((time.time() - t_launch) / 60, 1)

    # --- tick 2: must detect the loss and resume
    evs, lines = fire_and_wait(timeout_s=2700)
    assert "gone" in evs or "resumed" in evs, evs
    meta = C.load_meta(NAME)
    assert len(meta.get("resumes", [])) == 1, meta.get("resumes")
    rep["resume"] = {k: meta["resumes"][0].get(k) for k in ("relaunched", "seeded_jobs", "t_new", "t_total",
                                                             "fp_match")}
    say("resume:", rep["resume"])

    # --- ticks until finished
    t = time.time()
    while time.time() - t < 1800:
        time.sleep(90)
        evs, _ = fire_and_wait()
        if "done" in evs:
            break
        if not any(e in ("tick", "skip") for e in evs):
            raise RuntimeError(f"unexpected tick events {evs}")
    meta = C.load_meta(NAME)
    assert meta.get("finished_utc"), "not finished"

    # --- verification
    recs = [json.loads(x) for x in open(os.path.join(C.run_dir(NAME), "all.jsonl")) if x.strip()]
    jobs = [tuple(r["key"]) for r in recs if r.get("kind") == "job"]
    want = {tuple(k) for k in S.keys(2)}
    dup = sorted({k for k in jobs if jobs.count(k) > 1})
    rep["verify"] = {"job_lines": len(jobs), "unique": len(set(jobs)), "expected": len(want), "duplicates": dup,
                     "missing": sorted(want - set(jobs)), "unexpected": sorted(set(jobs) - want),
                     "closes": sum(r.get("kind") == "close" for r in recs),
                     "headers": sum(r.get("kind") == "header" for r in recs),
                     "hosts": sorted({r.get("host") for r in recs if r.get("kind") == "job"})}
    say("verify:", rep["verify"])
    assert not dup and set(jobs) == want, rep["verify"]


if __name__ == "__main__":
    rep_ok = False
    try:
        main()
        rep_ok = True
    finally:
        try:
            n, out = C.active_assignments()
            if n:
                say("cleanup: stopping", n, "active assignment(s)")
                C.stop(NAME)
                n, out = C.active_assignments()
            say("FINAL colab usage:", out.replace("\n", " | "))
        finally:
            rc, q = C._schtasks("/query", "/tn", C.task_name(NAME))
            if rc == 0:
                say("cleanup: task still registered -> unschedule")
                C.unschedule(NAME)
                rc, q = C._schtasks("/query", "/tn", C.task_name(NAME))
            say("FINAL schtasks /query rc:", rc, "(nonzero = task gone)", q[-150:])
            r = subprocess.run(["schtasks", "/query", "/fo", "csv", "/nh"], capture_output=True, text=True)
            say("tasks matching CDD-ORAN:", [x for x in r.stdout.splitlines() if "CDD-ORAN" in x])
            REP["passed"] = rep_ok
            json.dump(REP, open(os.path.join(C.run_dir(NAME), "e2e_report.json"), "w"), indent=1, default=str)
            say("REPORT", json.dumps(REP, default=str))
            say("TEST", "PASSED" if rep_ok else "FAILED", f"total {(time.time() - T0) / 60:.1f} min")
