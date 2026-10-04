"""VPS lane (5th platform): run bundled jobs on the user's Linux VPS (root@vps.brihatech.com) inside a resource-capped
systemd scope. The VPS runs PRODUCTION docker services, so every rule below is a hard safety rule (user, 2026-10-04):

  * Touch nothing outside BASE = /opt/cdd-xm: no docker / containers / services / packages / firewall / users / cron.
    Every remote command runs with HOME, XDG_*, UV_* (cache, Python installs, tools) and TMPDIR inside BASE.
  * Only uv (static release binary, sha256-checked) and Python 3.12 via uv, both inside BASE.
  * Compute only as `systemd-run --scope --unit cdd-xm-<job> -p CPUQuota=700% -p MemoryMax=11G -p MemorySwapMax=0
    nice -n 19 ionice -c3 ...`, at most MAX_PROCS = 7 single-threaded processes, ONE active cdd-xm job at a time.
  * Before every launch: refuse if MemAvailable < 12 GB or the 1-min load from others (load minus our running
    processes) > 1.
  * Never reboot; `stop` only stops our own cdd-xm-* scope.

  .venv/Scripts/python.exe scratchpad/e6_dev/vps_run.py probe                      # read-only host check
  .venv/Scripts/python.exe scratchpad/e6_dev/vps_run.py setup [--uv-version V]    # BASE tree, uv, Python PYTHON (3.12.14)
  .venv/Scripts/python.exe scratchpad/e6_dev/vps_run.py push NAME --paths P1 P2 ... [--root WT] [--extra F ...] [--add LOCAL=DEST ...]
        # bundle: git-tracked files of worktree WT (default this one) + untracked inputs, sha256 in MANIFEST.json
  .venv/Scripts/python.exe scratchpad/e6_dev/vps_run.py venv NAME --lock-install L --lock L2  # pinned venv + pin check
  .venv/Scripts/python.exe scratchpad/e6_dev/vps_run.py launch NAME --cmd "CMD" [--procs 7] [--dry-run]
  .venv/Scripts/python.exe scratchpad/e6_dev/vps_run.py status NAME | pull NAME | stop NAME

CLI CONTRACT "vps-run CLI v1" (stable; aud1's EVAL dispatcher lane calls it; change only additively):
  exit 0 = done; non-zero = refused / failed with the reason on stderr ("refused: ..." for safety refusals: an
  active cdd-xm job, MemAvailable < 12 GB, load from others > 1, bundle != clean HEAD). `launch` returns once the
  scope is active (the job then runs detached); completion = BASE/jobs/NAME/exit_code (shown by `status`, pulled by
  `pull` to runs/NAME/exit_code); records in runs/NAME/out/ after `pull`. NAME: [a-z0-9][a-z0-9-]{1,48}, one
  launch per NAME (re-launch only after `stop` or completion). `--cmd` placeholders: {OUT}, {PROCS}.

Layout: BASE/bin (uv), BASE/python, BASE/cache, BASE/home, BASE/tmp, BASE/venvs/<lock sha12> (shared per lock),
BASE/jobs/NAME/{b (bundle root), out, run.sh, run.out, exit_code}. CMD runs from the bundle root with the venv's
python first on PATH, PYTHONPATH=., OMP / OPENBLAS / MKL / NUMEXPR / NUMBA threads = 1, XM_PLATFORM=vps,
XM_CODE_COMMIT / XM_CODE_DIRTY from the local git tree, `{OUT}` = the job's out dir and `{PROCS}` = --procs.
Local mirror: scratchpad/e6_dev/runs/NAME/ (launch.json, out/ pulled append-only: a local file never shrinks).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HOST = "root@vps.brihatech.com"
BASE = "/opt/cdd-xm"
MAX_PROCS = 7
LIMITS = ("-p", "CPUQuota=700%", "-p", "MemoryMax=11G", "-p", "MemorySwapMax=0")
MIN_MEM_AVAILABLE_KB = 12 * 1024 * 1024                      # 12 GB
MAX_OTHER_LOAD = 1.0
UNIT_PREFIX = "cdd-xm-"
PYTHON = "3.12.14"                                          # = the Kaggle / Colab sessions (user, 2026-10-04)
UV_TARGET = "uv-x86_64-unknown-linux-gnu"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")
ENV = (f"export HOME={BASE}/home XDG_CONFIG_HOME={BASE}/home/.config XDG_CACHE_HOME={BASE}/cache "
       f"XDG_DATA_HOME={BASE}/home/.local/share UV_CACHE_DIR={BASE}/cache/uv UV_PYTHON_INSTALL_DIR={BASE}/python "
       f"UV_PYTHON_BIN_DIR={BASE}/bin UV_TOOL_DIR={BASE}/tools UV_NO_CONFIG=1 UV_NO_MODIFY_PATH=1 "
       f"TMPDIR={BASE}/tmp PATH={BASE}/bin:$PATH")


def runs_dir(name: str) -> str:
    return os.path.join(HERE, "runs", name)


def check_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise SystemExit(f"bad job name {name!r} (lowercase letters, digits, '-')")
    return name


def ssh(script: str, *, timeout: int = 600, check: bool = True) -> str:
    """Run a bash script on the VPS (fed on stdin: no remote quoting). ``set -eu``; BASE-only environment."""
    body = f"set -eu\n{ENV}\n{script}\n"
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, "bash", "-s"],
                       input=body.encode(), capture_output=True, timeout=timeout)
    out = (r.stdout + r.stderr).decode(errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"remote rc {r.returncode}:\n{out[-3000:]}")
    return out


def ssh_put(data: bytes, remote_cmd: str, timeout: int = 1800) -> str:
    """Stream bytes into a remote command (e.g. tar extraction inside BASE)."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST,
                        f"set -eu; {ENV}; {remote_cmd}"], input=data, capture_output=True, timeout=timeout)
    out = (r.stdout + r.stderr).decode(errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"remote rc {r.returncode}:\n{out[-3000:]}")
    return out


def host_state() -> dict:
    """Read-only: load, memory, our running cdd-xm scopes and their process counts."""
    out = ssh(f"""
read l1 l5 l15 rest < /proc/loadavg
ma=$(awk '/MemAvailable/ {{print $2}}' /proc/meminfo)
units=$(systemctl list-units --type=scope --state=active --no-legend --plain '{UNIT_PREFIX}*' | awk '{{print $1}}')
ours=0
for u in $units; do
  cg=/sys/fs/cgroup$(systemctl show -p ControlGroup --value $u)/cgroup.procs
  if [ -f $cg ]; then ours=$((ours + $(wc -l < $cg))); fi
done
echo "{{\\"load1\\": $l1, \\"load5\\": $l5, \\"mem_available_kb\\": $ma, \\"our_procs\\": $ours, \\"units\\": \\"$(echo $units)\\"}}"
""", timeout=120)
    return json.loads(out.strip().splitlines()[-1])


def preflight() -> dict:
    st = host_state()
    other = st["load1"] - st["our_procs"]
    st["other_load"] = round(other, 2)
    if st["units"]:
        raise SystemExit(f"refused: a cdd-xm job is active ({st['units']}); one job at a time")
    if st["mem_available_kb"] < MIN_MEM_AVAILABLE_KB:
        raise SystemExit(f"refused: MemAvailable {st['mem_available_kb'] / 2**20:.1f} GB < 12 GB")
    if other > MAX_OTHER_LOAD:
        raise SystemExit(f"refused: 1-min load from others {other:.2f} > {MAX_OTHER_LOAD}")
    return st


# ================================================================================================ commands
def probe() -> None:
    print(json.dumps(host_state()))
    print(ssh(f"grep -m1 'model name' /proc/cpuinfo; nproc; df -h /opt | tail -1; "
              f"du -sh {BASE} 2>/dev/null || echo '{BASE} absent'", timeout=120))


def setup(uv_version: str | None = None) -> None:
    """BASE tree + uv (release binary, sha256-checked; ``uv_version`` or the local uv's) + Python PYTHON (exact
    patch), all in BASE."""
    ver = uv_version or subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.split()[1]
    url = f"https://github.com/astral-sh/uv/releases/download/{ver}/{UV_TARGET}.tar.gz"
    print(ssh(f"""
mkdir -p {BASE}/bin {BASE}/python {BASE}/cache {BASE}/home {BASE}/tmp {BASE}/venvs {BASE}/jobs {BASE}/tools
cd {BASE}/tmp
if ! {BASE}/bin/uv --version 2>/dev/null | grep -q " {ver}"; then
  curl -fsSL -o uv.tgz {url}
  curl -fsSL -o uv.tgz.sha256 {url}.sha256
  echo "$(cut -d' ' -f1 uv.tgz.sha256)  uv.tgz" | sha256sum -c -
  tar -xzf uv.tgz
  install -m 0755 {UV_TARGET}/uv {UV_TARGET}/uvx {BASE}/bin/
  rm -rf uv.tgz uv.tgz.sha256 {UV_TARGET}
fi
{BASE}/bin/uv --version
nice -n 19 ionice -c3 {BASE}/bin/uv python install {PYTHON}
{BASE}/bin/uv python find {PYTHON}
du -sh {BASE}
""", timeout=1800))


def _git(root: str = ROOT) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def bundle(paths: list[str], root: str = ROOT, extra: list[str] | None = None,
           add: list[str] | None = None) -> tuple[bytes, dict]:
    """tar.gz of the git-tracked files under ``paths`` of the worktree ``root`` (read-only) + the untracked
    ``extra`` files / dirs (inputs such as pulled kernel outputs; root-relative) + ``add`` entries LOCAL=DEST (a
    file from anywhere, placed at bundle path DEST, e.g. a generated lock export) + MANIFEST.json (commit, dirty,
    sha256 of every file; extra and add listed apart)."""
    files = subprocess.run(["git", "ls-files", "-z", "--", *paths], cwd=root, capture_output=True,
                           check=True).stdout.decode().split("\0")
    files = sorted(f for f in files if f and "__pycache__" not in f)
    ex = []
    for e in extra or []:
        pe = os.path.join(root, e)
        if os.path.isdir(pe):
            ex += sorted(os.path.relpath(os.path.join(d, f), root).replace("\\", "/")
                         for d, _, fs in os.walk(pe) for f in fs if "__pycache__" not in d)
        elif os.path.isfile(pe):
            ex.append(e.replace("\\", "/"))
        else:
            raise SystemExit(f"--extra not found: {e}")
    if not files:
        raise SystemExit("empty bundle")
    man = {**_git(root), "root": os.path.abspath(root), "files": {}, "extra": {}, "add": {}}
    adds = []
    for e in add or []:
        src, _, dest = e.partition("=")
        if not dest or not os.path.isfile(src) or dest.startswith("/") or ".." in dest:
            raise SystemExit(f"--add needs LOCAL_FILE=BUNDLE_REL_PATH, got {e!r}")
        adds.append((src, dest))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f, kind in [(f, "files") for f in files] + [(f, "extra") for f in ex if f not in files]:
            data = open(os.path.join(root, f), "rb").read()
            man[kind][f] = hashlib.sha256(data).hexdigest()
            ti = tarfile.TarInfo(f)
            ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(data))
        for src, dest in adds:
            data = open(src, "rb").read()
            man["add"][dest] = {"sha256": hashlib.sha256(data).hexdigest(), "from": os.path.abspath(src)}
            ti = tarfile.TarInfo(dest)
            ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(data))
        mb = json.dumps(man, indent=1).encode()
        ti = tarfile.TarInfo("MANIFEST.json")
        ti.size, ti.mtime = len(mb), int(time.time())
        tf.addfile(ti, io.BytesIO(mb))
    return buf.getvalue(), man


def push(name: str, paths: list[str], root: str = ROOT, extra: list[str] | None = None,
         add: list[str] | None = None) -> None:
    data, man = bundle(paths, root, extra, add)
    jd = f"{BASE}/jobs/{name}"
    print(ssh_put(data, f"rm -rf {jd}/b && mkdir -p {jd}/b {jd}/out && tar -xzf - -C {jd}/b && "
                        f"echo pushed $(find {jd}/b -type f | wc -l) files"))
    os.makedirs(runs_dir(name), exist_ok=True)
    json.dump({"name": name, "paths": paths, "root": man["root"], "commit": man["commit"], "dirty": man["dirty"],
               "n_files": len(man["files"]), "extra": man["extra"], "add": man["add"], "bundle_bytes": len(data)},
              open(os.path.join(runs_dir(name), "push.json"), "w"), indent=1)
    print(f"[vps] {name}: {len(man['files'])} files + {len(man['extra'])} extra, {len(data) / 2**20:.1f} MB, "
          f"commit {man['commit'][:7]}{' DIRTY' if man['dirty'] else ''} ({man['root']})")


def venv(name: str, lock_install: str, lock: str, pincheck_cmd: str | None) -> None:
    """Shared venv per lock (BASE/venvs/<sha12 of the install lock>): uv venv 3.12, hash-pinned --no-deps install,
    torch == the lock's public version from the CPU index; then the pin check into the job's out dir."""
    sha = hashlib.sha256(open(os.path.join(ROOT, lock_install), "rb").read()).hexdigest()[:12]
    pins = open(os.path.join(ROOT, lock), encoding="utf-8").read()
    m = re.search(r"^torch==([0-9.]+)", pins, re.M)
    torch_v = m.group(1) if m else None
    vd, jd = f"{BASE}/venvs/{sha}-py{PYTHON}", f"{BASE}/jobs/{name}"
    pyx = f"{BASE}/python/cpython-{PYTHON}-linux-x86_64-gnu/bin/python3.12"   # exact patch dir, not the 3.12 link
    torch = (f"""py -c "import importlib.metadata as m, sys; from packaging.version import Version; \
sys.exit(Version(m.version('torch')).public != '{torch_v}')" || nice -n 19 ionice -c3 uv pip install --python {vd}/bin/python \
--no-deps torch=={torch_v} --index-url https://download.pytorch.org/whl/cpu""" if torch_v else ":")
    print(ssh(f"""
py() {{ {vd}/bin/python "$@"; }}
cd {jd}/b
[ -x {pyx} ] || {{ echo "missing {pyx}: run setup"; exit 3; }}
if [ ! -f {vd}/.complete ]; then
  rm -rf {vd}
  uv venv --managed-python --python {PYTHON} {vd}      # patch request: uv pins home to the exact patch dir
  nice -n 19 ionice -c3 uv pip install --python {vd}/bin/python --require-hashes --no-deps -r {lock_install}
  {torch}
  touch {vd}/.complete
fi
ln -sfn {vd} {jd}/venv
v=$(py --version 2>&1); echo "$v"; [ "$v" = "Python {PYTHON}" ] || {{ echo "interpreter mismatch: $v != {PYTHON}"; exit 4; }}
grep '^home' {vd}/pyvenv.cfg
grep -q '^home = {BASE}/python/cpython-{PYTHON}-' {vd}/pyvenv.cfg || {{ echo "venv home is not the {PYTHON} patch dir \
(floating minor link?)"; exit 5; }}
{f"PYTHONPATH=. py {pincheck_cmd} > {jd}/out/pin_check.json 2>&1; cat {jd}/out/pin_check.json" if pincheck_cmd else ""}
du -sh {vd}
""", timeout=3600))


def launch(name: str, cmd: str, procs: int, dry: bool) -> None:
    if not 1 <= procs <= MAX_PROCS:
        raise SystemExit(f"--procs must be 1..{MAX_PROCS}")
    pj = os.path.join(runs_dir(name), "push.json")
    if not os.path.exists(pj):
        raise SystemExit("push first")
    pushed = json.load(open(pj))
    code = _git(pushed.get("root") or ROOT)
    if pushed["commit"] != code["commit"] or pushed["dirty"] or code["dirty"]:
        raise SystemExit("refused: the pushed bundle is not the clean local HEAD (commit, then push again)")
    jd = f"{BASE}/jobs/{name}"
    unit = UNIT_PREFIX + name
    full = cmd.replace("{OUT}", f"{jd}/out").replace("{PROCS}", str(procs))
    run_sh = "\n".join([
        "#!/bin/bash", "set -u", ENV, f"cd {jd}/b",
        f"export PATH={jd}/venv/bin:$PATH PYTHONPATH=. XM_PLATFORM=vps XM_CODE_COMMIT={code['commit']} "
        f"XM_CODE_DIRTY={int(code['dirty'])} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
        f"echo started $(date -u +%FT%TZ) >> {jd}/run.out", full, "rc=$?",
        f"echo $rc > {jd}/exit_code.tmp && mv {jd}/exit_code.tmp {jd}/exit_code",
        f"echo finished rc $rc $(date -u +%FT%TZ) >> {jd}/run.out", ""])
    st = preflight() if not dry else {}
    print(f"[vps] preflight ok: {json.dumps(st)}")
    remote = f"""
cat > {jd}/run.sh <<'XM_EOF'
{run_sh}
XM_EOF
chmod 0755 {jd}/run.sh
rm -f {jd}/exit_code
cd {jd}
nohup setsid systemd-run --scope --quiet --unit {unit} {' '.join(LIMITS)} nice -n 19 ionice -c3 {jd}/run.sh \
  >> {jd}/run.out 2>&1 < /dev/null &
sleep 5
systemctl is-active {unit}.scope || (echo NOT-ACTIVE; tail -20 {jd}/run.out; exit 1)
systemctl show {unit}.scope -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax
"""
    if dry:
        print(remote)
        return
    print(ssh(remote, timeout=300))
    json.dump({"name": name, "unit": unit, "cmd": full, "procs": procs, "commit": code["commit"],
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preflight": st},
              open(os.path.join(runs_dir(name), "launch.json"), "w"), indent=1)


def status(name: str) -> None:
    jd, unit = f"{BASE}/jobs/{check_name(name)}", UNIT_PREFIX + name
    print(json.dumps(host_state()))
    print(ssh(f"""
echo "scope: $(systemctl is-active {unit}.scope 2>/dev/null || true)"
[ -f {jd}/exit_code ] && echo "exit_code: $(cat {jd}/exit_code)" || echo "exit_code: none"
for f in {jd}/out/res_*.jsonl; do [ -f "$f" ] && echo "$(basename $f) $(wc -l < $f)"; done
tail -n 3 {jd}/run.out 2>/dev/null || true
for f in {jd}/out/log_*.txt; do [ -f "$f" ] && echo "-- $(basename $f): $(tail -n 1 $f)"; done
""", timeout=120, check=False))


def pull(name: str) -> None:
    """tar the job's out dir over ssh; merge into runs/NAME/out (a local file is replaced only by a larger one)."""
    jd = f"{BASE}/jobs/{check_name(name)}"
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST,
                        f"cd {jd} && tar -czf - out run.out $(ls exit_code 2>/dev/null)"],
                       capture_output=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode(errors="replace")[-2000:])
    dst = runs_dir(name)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.name.startswith("/") or ".." in m.name:
                continue
            p = os.path.join(dst, m.name)
            if os.path.exists(p) and os.path.getsize(p) > m.size:
                continue                                       # never shrink a local file
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(tf.extractfile(m).read())
            n += 1
    print(f"[vps] pulled {n} files -> {dst}")


def stop(name: str) -> None:
    unit = UNIT_PREFIX + check_name(name)
    print(ssh(f"systemctl stop {unit}.scope && echo stopped {unit} || echo 'not active: {unit}'", timeout=300,
              check=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("probe")
    sub.add_parser("setup").add_argument("--uv-version", default=None, help="uv release (default: the local uv)")
    p = sub.add_parser("push")
    p.add_argument("name")
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--root", default=ROOT, help="worktree to bundle from (read-only; default this one)")
    p.add_argument("--extra", nargs="*", default=None, help="untracked input files / dirs (root-relative)")
    p.add_argument("--add", nargs="*", default=None, help="LOCAL_FILE=BUNDLE_REL_PATH (e.g. a lock export)")
    p = sub.add_parser("venv")
    p.add_argument("name")
    p.add_argument("--lock-install", required=True, help="repo-relative hash-pinned lock minus torch")
    p.add_argument("--lock", required=True, help="repo-relative full lock (torch version)")
    p.add_argument("--pincheck", default=None, help="python args printing the pin check JSON, from the bundle root")
    p = sub.add_parser("launch")
    p.add_argument("name")
    p.add_argument("--cmd", required=True)
    p.add_argument("--procs", type=int, default=MAX_PROCS)
    p.add_argument("--dry-run", action="store_true")
    for c in ("status", "pull", "stop"):
        sub.add_parser(c).add_argument("name")
    a = ap.parse_args(argv)
    if hasattr(a, "name"):
        check_name(a.name)
    if a.action == "probe":
        probe()
    elif a.action == "setup":
        setup(a.uv_version)
    elif a.action == "push":
        push(a.name, a.paths, a.root, a.extra, a.add)
    elif a.action == "venv":
        venv(a.name, a.lock_install, a.lock, a.pincheck)
    elif a.action == "launch":
        launch(a.name, a.cmd, a.procs, a.dry_run)
    else:
        {"status": status, "pull": pull, "stop": stop}[a.action](a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
