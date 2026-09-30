"""One-command Kaggle ANALYSIS job (sibling of kaggle_run.py / cloud.py; DEV scratch).

  .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py launch NAME --cmd "python scratchpad/e6_dev/x.py ..."
        [--paths repo/relative/file ...] [--sources owner/kernel-a,owner/kernel-b] [--internet] [--gpu] [--pin match|off] [--wait MIN]
  .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py status NAME
  .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py pull NAME [--all | --pattern REGEX]   # -> runs/NAME/

NAME must be new per launch: code dataset bishalpanta/NAME-code (a kernel may not share a dataset's slug: 409), kernel
bishalpanta/NAME (attach it as a --sources entry of later jobs to read its /kaggle/working outputs). A push blocked by
the account's 5-concurrent-CPU-session limit is retried every 60 s for --wait minutes (run launch in the background).
Bundle (zip, dataset bishalpanta/NAME-code): cdd_oran/**/*.py, scripts/*.py, scratchpad/e6_dev/*.py (repo-relative, so
`python scratchpad/e6_dev/x.py` works from the bundle root), docs/benchmark/*.md + SEED_REGISTRY.json, the --paths files
(repo-relative; a directory adds its files recursively), MANIFEST.json (git HEAD, sha256, local numpy / scipy) and a
marker _job_<NAME>.json that the kernel uses to find the bundle (Kaggle mounts datasets AUTO-UNZIPPED under
/kaggle/input/datasets/<owner>/<name>/...; the kernel also accepts the raw zip).
Kernel (private script, bishalpanta/NAME; kernel_sources = --sources, mounted under /kaggle/input/...): copies the bundle
to /tmp/job_root, PIN=match (default) pip-installs the local numpy / scipy when they differ (internet forced on), then
runs CMD with bash from the bundle root, env PYTHONPATH=<root>, KAGGLE_SOURCES_DIR=/kaggle/input,
JOB_SRC=/tmp/src (a symlink /tmp/src/<slug> per --sources entry: USE THIS, the mount layout varies between runs:
/kaggle/input/<slug>/... or /kaggle/input/notebooks/<owner>/<slug>/..., code dataset /kaggle/input/<slug>-code/ or
/kaggle/input/datasets/<owner>/<slug>-code/), JOB_OUT=/kaggle/working/out (write outputs there). stdout / stderr -> /kaggle/working/job.log (also the kernel log);
/kaggle/working/job.json = rc, wall time, env, install log; /kaggle/working/job_inputs.txt = every file under
/kaggle/input (depth-limited listing + all *.jsonl / *.npz / *.json paths) = the real mount paths.
pull: default skips *.npz (big caches stay on Kaggle; attach the kernel as a source instead); --all pulls everything.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KAGGLE = os.path.expanduser("~/.cloudtools/Scripts/kaggle.exe")
OWNER = "bishalpanta"
DEFAULT_PULL = r"^(?!.*\.npz$)"
DOT = "_dot_"                                     # bundle encoding of a leading "." in a path component

KERNEL = r'''
import glob, json, os, shutil, subprocess, sys, time, zipfile
import importlib.metadata as md
NAME, CMD, PIN, SOURCES = __NAME__, __CMD__, __PIN__, __SOURCES__
W = "/kaggle/working"
OUT = os.path.join(W, "out")
os.makedirs(OUT, exist_ok=True)
logf = open(os.path.join(W, "job.log"), "a")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


t0 = time.time()
root = "/tmp/job_root"
mk = glob.glob(f"/kaggle/input/**/_job_{NAME}.json", recursive=True)
if mk:
    shutil.copytree(os.path.dirname(mk[0]), root, dirs_exist_ok=True)
    how = "unzipped dataset at " + os.path.dirname(mk[0])
else:
    z = glob.glob(f"/kaggle/input/**/{NAME}_bundle.zip", recursive=True)
    zipfile.ZipFile(z[0]).extractall(root)
    how = "zip " + z[0]
for dp, dn, fn in os.walk(root, topdown=False):             # decode dot-dirs / dot-files (kaggle_job.DOT)
    for x in dn + fn:
        if x.startswith("_dot_"):
            os.rename(os.path.join(dp, x), os.path.join(dp, "." + x[5:]))
log(f"[job] bundle: {how} -> {root}")
# mount listing (depth <= 5) + every data file path
lines = []
for dp, dn, fn in os.walk("/kaggle/input"):
    depth = dp.count("/") - 2
    if "/bundle" in dp or "__pycache__" in dp:
        continue
    if depth <= 5:
        lines.append(f"D {dp} ({len(fn)} files)")
    for f in fn:
        if f.endswith((".jsonl", ".npz", ".json", ".zip")) and "/bundle/" not in dp:
            p = os.path.join(dp, f)
            lines.append(f"F {p} {os.path.getsize(p)}")
open(os.path.join(W, "job_inputs.txt"), "w").write("\n".join(lines) + "\n")
log(f"[job] /kaggle/input: {len(lines)} listing lines -> job_inputs.txt")
# stable per-source paths: Kaggle's mount layout varies between runs (/kaggle/input/<slug>/ or
# /kaggle/input/notebooks/<owner>/<slug>/; datasets /kaggle/input/datasets/<owner>/<slug>/) -> /tmp/src/<slug>
SRC = "/tmp/src"
os.makedirs(SRC, exist_ok=True)
found = {}
for s in SOURCES:
    slug = s.split("/")[-1]
    hits = sorted((dp for dp, dn, fn in os.walk("/kaggle/input") if os.path.basename(dp) == slug
                   and dp.count("/") <= 6), key=len)
    if hits:
        found[slug] = hits[0]
        if not os.path.exists(os.path.join(SRC, slug)):
            os.symlink(hits[0], os.path.join(SRC, slug))
log(f"[job] sources -> {SRC}/<slug>: {found}; missing {[s for s in SOURCES if s.split('/')[-1] not in found]}")
man = json.load(open(os.path.join(root, "MANIFEST.json")))
want = man.get("local_env", {})
have = {p: md.version(p) for p in ("numpy", "scipy")}
inst = []
if PIN == "match" and want and any(have[p] != want.get(p) for p in have):
    for pkgs in ([f"{p}=={want[p]}" for p in have], [f"numpy=={want['numpy']}"]):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
        inst.append({"pkgs": pkgs, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]})
        if r.returncode == 0:
            break
env = dict(os.environ, PYTHONPATH=root, KAGGLE_SOURCES_DIR="/kaggle/input", JOB_SRC=SRC, JOB_OUT=OUT, JOB_NAME=NAME,
           PYTHONUNBUFFERED="1")
ver = subprocess.run([sys.executable, "-c", "import numpy, scipy, sys; print(numpy.__version__, scipy.__version__, "
                      "sys.version.split()[0])"], capture_output=True, text=True, env=env).stdout.strip()
log(f"[job] git {man.get('git_head')} dirty {man.get('dirty')}; pin {PIN}; kaggle had {have}; now {ver}; "
    f"want {want.get('numpy')}/{want.get('scipy')}; install {[(i['pkgs'], i['rc']) for i in inst]}")
log(f"[job] cpus {os.cpu_count()}; CMD: {CMD}")
p = subprocess.Popen(["bash", "-c", CMD], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
for line in p.stdout:
    log(line.rstrip("\n"))
rc = p.wait()
wall = round(time.time() - t0, 1)
log(f"[job] exit code {rc}; wall {wall} s")
json.dump({"name": NAME, "cmd": CMD, "rc": rc, "wall_s": wall, "pin": PIN, "have": have, "versions_now": ver,
           "install": inst, "git_head": man.get("git_head"), "dirty": man.get("dirty"), "cpus": os.cpu_count()},
          open(os.path.join(W, "job.json"), "w"), indent=1)
'''


def k(*args, cwd=None, timeout=900):
    r = subprocess.run([KAGGLE, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def _files(extra) -> dict:
    """{bundle path: local path}."""
    pats = ["cdd_oran/**/*.py", "scripts/*.py", "scratchpad/e6_dev/*.py", "docs/benchmark/*.md",
            "docs/benchmark/SEED_REGISTRY.json"]
    files = {}
    for p in pats:
        for f in glob.glob(os.path.join(ROOT, *p.split("/")), recursive=True):
            if "__pycache__" not in f:
                files[os.path.relpath(f, ROOT).replace("\\", "/")] = f
    for e in extra or []:
        a = e if os.path.isabs(e) else os.path.join(ROOT, e)
        a = os.path.abspath(a)
        if not a.startswith(os.path.abspath(ROOT)):
            sys.exit(f"--paths must be inside the repo: {e}")
        if os.path.isdir(a):
            for f in glob.glob(os.path.join(a, "**", "*"), recursive=True):
                if os.path.isfile(f):
                    files[os.path.relpath(f, ROOT).replace("\\", "/")] = f
        elif os.path.isfile(a):
            files[os.path.relpath(a, ROOT).replace("\\", "/")] = a
        else:
            sys.exit(f"--paths: not found {e}")
    return files


def build(name, cmd, paths, sources, internet, gpu, pin) -> str:
    sys.path.insert(0, HERE)
    from cloud import numeric_env
    out = os.path.join(tempfile.gettempdir(), "e6_kaggle_job", name)
    ds, kd = os.path.join(out, "dataset"), os.path.join(out, "kernel")
    os.makedirs(ds, exist_ok=True)
    os.makedirs(kd, exist_ok=True)
    files = _files(paths)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "cdd_oran", "scratchpad/e6_dev"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    text_ext = (".py", ".md", ".json", ".txt", ".sh", ".cfg", ".toml", ".yaml", ".yml")
    blobs = {}
    for rel, f in files.items():
        b = open(f, "rb").read()
        blobs[rel] = b.replace(b"\r\n", b"\n") if rel.endswith(text_ext) else b
    man = {"git_head": head, "dirty": bool(dirty), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cmd": cmd, "sources": sources, "sha256": {r: hashlib.sha256(b).hexdigest() for r, b in blobs.items()},
           "local_env": numeric_env()}
    with zipfile.ZipFile(os.path.join(ds, f"{name}_bundle.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for rel, b in blobs.items():                 # dot-dirs (.tmp/...) break Kaggle's dataset processing: encoded
            z.writestr("/".join(DOT + c[1:] if c.startswith(".") else c for c in rel.split("/")), b)
        z.writestr("MANIFEST.json", json.dumps(man, indent=1))
        z.writestr(f"_job_{name}.json", json.dumps({"name": name, "git_head": head}))
    json.dump({"title": f"{name}-code", "id": f"{OWNER}/{name}-code", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(ds, "dataset-metadata.json"), "w"))
    src = (KERNEL.replace("__NAME__", repr(name)).replace("__CMD__", repr(cmd)).replace("__PIN__", repr(pin))
           .replace("__SOURCES__", repr(list(sources))))
    open(os.path.join(kd, "kernel.py"), "w", newline="\n").write(src)
    json.dump({"id": f"{OWNER}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": bool(gpu),
               "enable_internet": bool(internet or pin == "match"), "dataset_sources": [f"{OWNER}/{name}-code"],
               "kernel_sources": list(sources)}, open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=1)
    print(json.dumps({"files": len(files), "bundle_mb": round(os.path.getsize(os.path.join(ds, f"{name}_bundle.zip"))
                                                               / 2 ** 20, 2), "git_head": head, "dirty": bool(dirty),
                      "sources": sources, "pin": pin, "out": out}))
    return out


def launch(a):
    sources = [s for s in (a.sources or "").split(",") if s]
    out = build(a.name, a.cmd, a.paths, sources, a.internet, a.gpu, a.pin)
    c = k("datasets", "create", "-p", "dataset", cwd=out)                  # relative -p: CLI mangles abs paths
    print(c.splitlines()[-1])
    if "error" in c.lower():
        sys.exit("dataset create failed (NAME must be new per launch)")
    t0 = time.time()
    while True:                                  # a new dataset answers 403 until it is ready
        s = k("datasets", "status", f"{OWNER}/{a.name}-code", timeout=120)
        if "ready" in s.lower():
            break
        if time.time() - t0 > 900:
            sys.exit(f"dataset not ready after 15 min: {s[-300:]}")
        time.sleep(10)
    print(f"dataset ready after {time.time() - t0:.0f} s")
    t0 = time.time()
    while True:                                  # queue behind the account's 5 concurrent CPU sessions
        s = k("kernels", "push", "-p", "kernel", cwd=out)
        if "maximum batch" not in s.lower():
            break
        if time.time() - t0 > 60 * a.wait:
            sys.exit(f"push still blocked after {a.wait} min: {s[-200:]}")
        print(f"{time.strftime('%H:%M:%S')} {s.splitlines()[-1]} - retry in 60 s", flush=True)
        time.sleep(60)
    print(s.splitlines()[-1])
    if "error" in s.lower():
        sys.exit("kernel push failed")
    dst = os.path.join(HERE, "runs", a.name)
    os.makedirs(dst, exist_ok=True)
    json.dump({"name": a.name, "cmd": a.cmd, "sources": sources, "paths": a.paths, "internet": a.internet,
               "gpu": a.gpu, "pin": a.pin, "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(os.path.join(dst, "launch.json"), "w"), indent=1)


def status(a):
    print(k("kernels", "status", f"{OWNER}/{a.name}", timeout=60))


def pull(a):
    dst = os.path.join(HERE, "runs", a.name)
    os.makedirs(dst, exist_ok=True)
    args = ["kernels", "output", f"{OWNER}/{a.name}", "-p", os.path.relpath(dst, HERE), "-o"]
    if not a.all:
        args += ["--file-pattern", a.pattern or DEFAULT_PULL]
    print(k(*args, cwd=HERE, timeout=3600)[-2000:])
    j = os.path.join(dst, "job.json")
    if os.path.exists(j):
        s = json.load(open(j))
        print({x: s.get(x) for x in ("rc", "wall_s", "versions_now", "cpus")})
    log = os.path.join(dst, "job.log")
    if os.path.exists(log):
        print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-25:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    la = sp.add_parser("launch")
    la.add_argument("name")
    la.add_argument("--cmd", required=True)
    la.add_argument("--paths", nargs="*", default=[])
    la.add_argument("--sources", default="")
    la.add_argument("--internet", action="store_true")
    la.add_argument("--gpu", action="store_true")
    la.add_argument("--pin", choices=("match", "off"), default="match")
    la.add_argument("--wait", type=int, default=240, help="minutes to retry a push blocked by the 5-session limit")
    st = sp.add_parser("status")
    st.add_argument("name")
    pu = sp.add_parser("pull")
    pu.add_argument("name")
    pu.add_argument("--all", action="store_true")
    pu.add_argument("--pattern", default=None)
    a = ap.parse_args(argv)
    {"launch": launch, "status": status, "pull": pull}[a.op](a)


if __name__ == "__main__":
    main()
