"""Throwaway resumable driver for testing colab_run.py tick / resume (DEV scratch; no simulation).

  python colab_selftest.py run --part i/P --out FILE.jsonl [--smoke]
  python colab_selftest.py keys P            # expected job keys of all parts (JSON), for the test's check

Same record contract as the real drivers (e6p_screen.run): one header per start, one {"kind": "job", "key": [...]}
line per finished job, a "close" record when the part is complete; a restart skips every job key already in --out.
N_UNITS units of JOB_S seconds each, unit u belongs to part u % P (P=2 -> 12 jobs of 20 s = 4 min per part).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

N_UNITS = int(os.environ.get("SELFTEST_UNITS", "24"))
JOB_S = float(os.environ.get("SELFTEST_JOB_S", "20"))


def _append(out, rec):
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _done(out, smoke):
    if not os.path.exists(out):
        return set()
    keys = set()
    for ln in open(out):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == "job" and r.get("smoke") == smoke:
            keys.add(tuple(r["key"]))
    return keys


def keys(P):
    return [["selftest", u] for u in range(N_UNITS)]


def run(part, out, smoke=False):
    i, P = map(int, part.split("/"))
    t0 = time.time()
    _append(out, {"kind": "header", "stage": "selftest", "part": part, "smoke": smoke, "host": socket.gethostname(),
                  "pid": os.getpid(), "t": time.time()})
    done, n = _done(out, smoke), 0
    for u in range(N_UNITS):
        if u % P != i or ("selftest", u) in done:
            continue
        t = time.time()
        while time.time() - t < JOB_S:                 # a little real CPU work, then sleep
            sum(k * k for k in range(20000))
            time.sleep(0.5)
        _append(out, {"kind": "job", "key": ["selftest", u], "smoke": smoke, "part": part,
                      "secs": round(time.time() - t, 1), "host": socket.gethostname(), "t": time.time()})
        n += 1
        print(json.dumps({"unit": u, "secs": round(time.time() - t, 1)}), flush=True)
    _append(out, {"kind": "close", "stage": "selftest", "part": part, "n_jobs": n, "secs": round(time.time() - t0, 1)})


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        run(a[a.index("--part") + 1], a[a.index("--out") + 1], smoke="--smoke" in a)
    elif a[:1] == ["keys"]:
        print(json.dumps(keys(int(a[1]))))
    else:
        raise SystemExit(__doc__)
