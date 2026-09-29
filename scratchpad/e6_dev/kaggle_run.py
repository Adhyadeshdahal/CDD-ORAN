"""One-command Kaggle launch / status / pull for e6dev grid scripts (DEV scratch).

  .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch NAME SCRIPT KERNELS [PIN=match]   # bundle, dataset, push
  .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py status NAME KERNELS
  .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py pull NAME KERNELS            # -> scratchpad/e6_dev/runs/NAME/
NAME must be new per launch (Kaggle dataset ids are unique). Bundles live in the session temp dir, results in runs/.
PIN (cloud.py docstring): match = kernel installs the local numpy/scipy (internet on; phone-verified account),
strict = also abort on a numpy mismatch, off = Kaggle stack, no internet. pull also prints each startup.json verdict.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KAGGLE = os.path.expanduser("~/.cloudtools/Scripts/kaggle.exe")
PY = sys.executable


def k(*args, cwd=None, timeout=600):
    r = subprocess.run([KAGGLE, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def tags(n):
    return "abcdefgh"[:int(n)]


def launch(name, script, kernels, pin="match"):
    out = os.path.join(tempfile.gettempdir(), "e6_kaggle", name)
    subprocess.run([PY, os.path.join(HERE, "cloud.py"), out, name, script, kernels, pin], check=True, cwd=ROOT)
    print(k("datasets", "create", "-p", "dataset", cwd=out).splitlines()[-1])     # relative -p: CLI mangles abs paths
    t0 = time.time()
    while "ready" not in k("datasets", "status", f"bishalpanta/{name}").lower():
        if time.time() - t0 > 600:
            sys.exit("dataset not ready after 10 min")
        time.sleep(10)
    for t in tags(kernels):
        print(k("kernels", "push", "-p", f"kernel_{t}", cwd=out).splitlines()[-1])


def status(name, kernels):
    for t in tags(kernels):
        print(k("kernels", "status", f"bishalpanta/{name}-{t}", timeout=60))


def pull(name, kernels):
    dst = os.path.join(HERE, "runs", name)
    for t in tags(kernels):
        d = os.path.join(dst, t)
        os.makedirs(d, exist_ok=True)
        k("kernels", "output", f"bishalpanta/{name}-{t}", "-p", os.path.relpath(d, HERE), cwd=HERE)
    res = sorted(glob.glob(os.path.join(dst, "*", "res_*.jsonl")))
    lines = [ln for f in res for ln in open(f) if ln.strip()]
    open(os.path.join(dst, "all.jsonl"), "w").writelines(lines)
    print(f"{len(lines)} records from {len(res)} shard files -> {os.path.join(dst, 'all.jsonl')}")
    for f in sorted(glob.glob(os.path.join(dst, "*", "startup.json"))):
        s = json.load(open(f))
        print(os.path.basename(os.path.dirname(f)), {k: s.get(k) for k in ("fp_match", "fp_layers_equal", "pin")},
              {k: (s.get("env") or {}).get(k) for k in ("numpy", "scipy", "platform", "simd")})


if __name__ == "__main__":
    {"launch": launch, "status": status, "pull": pull}[sys.argv[1]](*sys.argv[2:])
