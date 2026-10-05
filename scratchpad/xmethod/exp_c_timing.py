"""Experiment C timing runs (R-55): the paired T3 calibration block and the dedicated paper timing block.

Both blocks run one single-threaded process per vCPU (the session's cgroup CPU quota, never oversubscribed) and
stamp every unit with platform, CPU model, vCPU quota, load before / after (loadavg, running processes, concurrent
timing processes), code commit and package versions, in the campaign record format (key, job, cpu_s, wall_s,
peak_rss_mb, host, load), so `exp_c_runtime.py calib` / `table` read them directly.

  calib  T3 speed factors (R-55 (1)): 6 anchor arms spanning cheap / medium / expensive (pcorr_eq, rcot2_eq, shap_dag,
         pmrt_nl_eq, cdl, mscr_eq_min), n 500 / 1000 / 4000 (mscr n <= 1000, R-54), E2 R2 kappa .25 (the costliest or
         near-costliest DEV cell of every anchor), DEV seeds 3_000_000-002 (the T3 calibration seeds; R-55's "3
         repeats"). Run as its own job on Kaggle (reference) and Colab (CPU runtime; CPU model tagged).
  paper  Paper runtime table (R-55 (3)): every Study A arm (EVAL spec arms minus pdcor / cmi_knn, R-48 / R-49, plus
         pmrt_nl_eq and cdl, minus mscr_eq_min: R-58(5), R-60 X1) x n grid (mscr n <= 1000), E2 R2 (granger: E3 R2),
         seeds 3_000_000-002, on Kaggle only, one uncontended session. Launch after EVAL is underway (R-60).

  uv run python scratchpad/xmethod/exp_c_timing.py make-spec          # -> specs/exp_c/timing.json (from feat/v2 specs)
  uv run python scratchpad/xmethod/exp_c_timing.py lock               # -> _bundle_expc/requirements.{lock,cloud}.txt
  uv run python scratchpad/xmethod/exp_c_timing.py units --block calib [--part i/P]
  uv run python scratchpad/xmethod/exp_c_timing.py run --block calib --out-dir D [--procs P] [--max-units K]
  uv run python scratchpad/xmethod/exp_c_timing.py launch kaggle|colab --block calib --name NAME [--dry-run]

Cloud setup (as the campaign's, R-35): uv-installed Python 3.12 venv, the hash-pinned uv.lock export (no deps) minus
the torch build, torch == the lock's public version (image build kept if equal, else the CPU wheel), pin check ->
startup files in the out dir. Kaggle launch needs fewer than 4 RUNNING / QUEUED kernels on the account; Colab at
most 3 exp-c jobs at a time (R-55; the cdl dispatcher uses the rest). Never Lightning.
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata as md
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

VERSION = "xm-exp-c-timing/1"
SPEC_REL = "scratchpad/xmethod/specs/exp_c/timing.json"
BUNDLE_REL = "scratchpad/xmethod/_bundle_expc"
LOCK_REL = f"{BUNDLE_REL}/requirements.lock.txt"
LOCK_INSTALL_REL = f"{BUNDLE_REL}/requirements.cloud.txt"
SELF_REL = "scratchpad/xmethod/exp_c_timing.py"
DEV_RUNTIME_REL = "scratchpad/xmethod/results/exp_c/dev/exp_c_runtime.json"
EVAL_SPEC_REL = "scratchpad/xmethod/specs/eval/full.json"
BUNDLE_DATA = ("configs", "scripts")                          # read by cdd_oran (cdl yaml; two_tower)
PKGS = ("numpy", "scipy", "scikit-learn", "statsmodels", "causal-learn", "xgboost", "shap", "torch",
        "threadpoolctl", "momentchi2")
PYTHON = "3.12.14"                                            # exact patch: uv now defaults 3.12 to 3.12.15
KAGGLE = os.path.expanduser("~/.cloudtools/Scripts/kaggle.exe")
KAGGLE_RUNNING_MAX = 4                                        # launch only if fewer than this many run (R-55)
KAGGLE_ACCOUNT_CAP = 5                                        # --slot5 (orchestrator, freeze gate): the 5th slot
COLAB_MAX = 3
CALIB_ANCHORS = ("pcorr_eq", "rcot2_eq", "shap_dag", "pmrt_nl_eq", "cdl", "mscr_eq_min")
DROPPED = ("pdcor", "cmi_knn")                                # R-48, R-49 revised
EXTRA_ARMS = {                                                # arms not in feat/v2's eval spec (copied from dev specs)
    "pmrt_nl_eq": {"ref": "cdd_oran.xmethod.methods.pmrt_core:PmrtCore",
                   "config": {"covariates": "eq", "statistic": "gbm"},
                   "source": "xm/dev-runs specs/dev/pmrt_nl.json (R-42 / R-52 winner)"},
    "cdl": {"ref": "cdd_oran.xmethod.methods.cdl:CDLMethod", "config": {"arm": "native"},
            "source": "xm/dev-runs specs/dev/cdl.json (R-50)"},
}
MAX_N = {"mscr_eq": 1000, "mscr_native": 1000, "mscr_eq_min": 1000}   # R-54
PAPER_EXCLUDE = ("mscr_eq_min",)                              # R-58(5) / R-60 X1 (still a calib anchor)
N_GRID = [500, 1000, 4000, 8000, 24000]
SEEDS = [3_000_000, 3_000_001, 3_000_002]


# ================================================================================================ spec / units
def make_spec() -> dict:
    ev = json.load(open(os.path.join(ROOT, EVAL_SPEC_REL), encoding="utf-8"))
    arms = {a: {"ref": d["ref"], "config": d.get("config", {}), **({"worlds": d["worlds"]} if "worlds" in d else {})}
            for a, d in ev["arms"].items() if not a.startswith(DROPPED)}
    arms.update({a: {k: v for k, v in d.items()} for a, d in EXTRA_ARMS.items()})
    for a, m in MAX_N.items():
        if a in arms:
            arms[a]["max_n"] = m
    est = {}
    p = os.path.join(ROOT, DEV_RUNTIME_REL)
    if os.path.exists(p):                                     # DEV median per (arm, world, regime, n), ref seconds
        for r in json.load(open(p, encoding="utf-8"))["by_world_regime"]:
            est[f"{r['arm']}|{r['world']}|{r['regime']}|n{r['n']}"] = r["median"]
    return {"name": "exp_c_timing", "version": VERSION, "kappa": 0.25, "seeds": SEEDS,
            "source": f"arms from {EVAL_SPEC_REL} (feat/v2) minus {list(DROPPED)}, plus {sorted(EXTRA_ARMS)}",
            "arms": dict(sorted(arms.items())),
            "blocks": {"calib": {"arms": list(CALIB_ANCHORS), "ns": [500, 1000, 4000], "cell": ["E2", "R2"]},
                       "paper": {"arms": "all", "exclude": list(PAPER_EXCLUDE), "ns": N_GRID, "cell": ["E2", "R2"],
                                 "cell_E3_only": ["E3", "R2"]}},
            "est_cost_s": est}


def load_spec() -> dict:
    return json.load(open(os.path.join(ROOT, SPEC_REL), encoding="utf-8"))


def _key(arm, w, r, n, s, kappa) -> str:
    from cdd_oran.xmethod.runner import job_key
    return job_key(arm, w, r, None, n, s, kappa)


def est_cost(spec: dict, arm: str, w: str, r: str, n: int) -> float:
    e = spec.get("est_cost_s", {}).get(f"{arm}|{w}|{r}|n{n}")
    return float(e) if e is not None else 30.0 * n / 1000          # unknown (pmrt_nl_eq, cdl): 30 s per 1000 rows


def units(spec: dict, block: str) -> list[dict]:
    b = spec["blocks"][block]
    arms = [a for a in spec["arms"] if a not in b.get("exclude", [])] if b["arms"] == "all" else b["arms"]
    out = []
    for a in arms:
        d = spec["arms"][a]
        w, r = b["cell"]
        if d.get("worlds") and w not in d["worlds"]:
            w, r = b.get("cell_E3_only", [d["worlds"][0], r])
        for n in b["ns"]:
            if d.get("max_n") and n > d["max_n"]:
                continue
            for s in spec["seeds"]:
                out.append({"key": _key(a, w, r, n, s, spec["kappa"]), "arm": a, "world": w, "regime": r, "n": n,
                            "seed": s, "kappa": spec["kappa"], "est_cost_s": est_cost(spec, a, w, r, n)})
    return out


def assign(us: list[dict], procs: int) -> list[list[dict]]:
    """Longest-processing-time first: costliest unit to the least-loaded process (balances wall, keeps 1 per vCPU)."""
    bins: list[list[dict]] = [[] for _ in range(procs)]
    load = [0.0] * procs
    for u in sorted(us, key=lambda u: (-u["est_cost_s"], u["key"])):
        i = load.index(min(load))
        bins[i].append(u)
        load[i] += u["est_cost_s"]
    return bins


# ================================================================================================ host / load
def _own_cgroup_dirs(cgroup: str) -> list[str]:
    """cgroup v2 directories of this process, own first then every ancestor up to ``cgroup`` (a systemd-run scope
    such as the VPS lane's CPUQuota=600% puts cpu.max on the scope, not on the root)."""
    out = []
    try:
        for line in open("/proc/self/cgroup"):
            if line.startswith("0::"):
                rel = line.split("::", 1)[1].strip().strip("/")
                parts = rel.split("/") if rel else []
                out = [os.path.join(cgroup, *parts[:k]) for k in range(len(parts), 0, -1)]
    except OSError:
        pass
    return out + [cgroup]


def cpu_quota(cgroup: str = "/sys/fs/cgroup") -> int:
    """vCPUs this process may use: the tightest cgroup v2 cpu.max on its own cgroup path (else v1 cfs), capped by
    the affinity mask."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    q = None
    qs = []
    for d in _own_cgroup_dirs(cgroup):
        try:
            a, b = open(os.path.join(d, "cpu.max")).read().split()[:2]
            if a != "max":
                qs.append(float(a) / float(b))
        except (OSError, ValueError):
            continue
    if qs:
        q = min(qs)
    else:
        for d in ("cpu", "cpu,cpuacct"):
            try:
                a = float(open(os.path.join(cgroup, d, "cpu.cfs_quota_us")).read())
                b = float(open(os.path.join(cgroup, d, "cpu.cfs_period_us")).read())
                q = a / b if a > 0 else None
                break
            except (OSError, ValueError):
                continue
    return max(1, min(n, math.ceil(q - 1e-9))) if q else max(1, n)


def timing_procs(proc: str = "/proc") -> int | None:
    """Running `exp_c_timing.py worker` processes on this host; None without /proc."""
    if not os.path.isdir(proc):
        return None
    k = 0
    for pid in os.listdir(proc):
        if pid.isdigit():
            try:
                cmd = open(os.path.join(proc, pid, "cmdline"), "rb").read().split(b"\0")
            except OSError:
                continue
            k += any(c.endswith(b"exp_c_timing.py") for c in cmd) and b"worker" in cmd
    return k


def load_info() -> dict:
    out = {"loadavg": None, "procs_running": None, "procs_total": None, "timing_procs": timing_procs(),
           "cpu_quota": cpu_quota()}
    try:
        f = open("/proc/loadavg").read().split()
        r, t = f[3].split("/")
        out.update(loadavg=[float(x) for x in f[:3]], procs_running=int(r), procs_total=int(t))
    except (OSError, ValueError, IndexError):
        if hasattr(os, "getloadavg"):
            out["loadavg"] = list(os.getloadavg())
    return out


def host_info() -> dict:
    cpu = platform.processor() or None
    ident: dict = {}                                          # GCP labels several generations "Xeon @ 2.20GHz"
    try:
        for line in open("/proc/cpuinfo"):
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "model name" and "cpu_model_set" not in ident:   # /proc/cpuinfo wins over platform.processor
                cpu, ident["cpu_model_set"] = v, True
            elif k in ("cpu family", "model", "stepping", "microcode", "cache size") and k not in ident:
                ident[k] = v
            elif k == "flags" and "flags_sha" not in ident:
                import hashlib
                fl = v.split()
                ident.update(flags_sha=hashlib.sha256(" ".join(sorted(fl)).encode()).hexdigest()[:12],
                             avx512f="avx512f" in fl, amx="amx_tile" in fl)
            elif not line.strip() and ident:
                break                                         # first processor block only
    except OSError:
        pass
    ident.pop("cpu_model_set", None)
    return {"platform": os.environ.get("XM_PLATFORM", "local"), "cpu_model": cpu, "cpu_count": os.cpu_count(),
            "cpu_quota": cpu_quota(), "node": platform.node(), "gpu": None, "cpu_ident": ident or None}


def pkg_versions() -> dict:
    out = {}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


# ================================================================================================ run
def worker(spec: dict, us: list[dict], out: str, block: str) -> int:
    from cdd_oran.xmethod import runner as R
    from cdd_oran.xmethod.worlds import generate_dataset
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add(r["key"])
    host, pkgs = host_info(), pkg_versions()
    code = {"commit": os.environ.get("XM_CODE_COMMIT"), "dirty": os.environ.get("XM_CODE_DIRTY") == "1"}
    methods: dict = {}
    k = 0
    with open(out, "a", encoding="utf-8") as fh:
        for u in us:
            if u["key"] in done:
                continue
            a = spec["arms"][u["arm"]]
            cfg = dict(a.get("config", {}))
            load0 = load_info()
            t0 = time.process_time()
            ds, truth = generate_dataset(u["world"], u["regime"], u["n"], u["seed"], lam=1.0, kappa=u["kappa"])
            gen = time.process_time() - t0
            try:
                if u["arm"] not in methods:
                    methods[u["arm"]] = R.load_method(a["ref"])
                rec = R.run_one(methods[u["arm"]], ds, truth, cfg, gen, u["key"])
            except Exception:
                import traceback
                rec = {"key": u["key"], "error": traceback.format_exc()[-2000:], "config": cfg,
                       "job": {"world": u["world"], "regime": u["regime"], "lam": None, "n": u["n"],
                               "seed": u["seed"], "kappa": u["kappa"]}}
            rec.update(status="ok" if rec.get("error") is None else "error", arm=u["arm"], role="timing",
                       block=block, spec_name=f"exp_c_{block}", campaign=VERSION, host=host, pkgs=pkgs, code=code,
                       load={"start": load0, "end": load_info()})
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
            fh.flush()
            k += 1
            print(f"[exp-c] {u['key']} cpu {rec.get('cpu_s') or float('nan'):.1f}s wall "
                  f"{rec.get('wall_s') or float('nan'):.1f}s {rec['status']}", flush=True)
    return k


def run(spec: dict, block: str, out_dir: str, procs: int | None, max_units: int | None) -> int:
    us = units(spec, block)[:max_units] if max_units else units(spec, block)
    p = procs or cpu_quota()
    bins = assign(us, p)
    os.makedirs(out_dir, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "PYTHONPATH": ROOT}
    t0 = time.time()
    ps = []
    for i, b in enumerate(bins):
        json.dump(b, open(os.path.join(out_dir, f"units_{i}.json"), "w"), indent=0)
        ps.append(subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "worker", "--block", block,
                                    "--units", os.path.join(out_dir, f"units_{i}.json"),
                                    "--out", os.path.join(out_dir, f"res_{i}.jsonl")],
                                   env=env, stdout=open(os.path.join(out_dir, f"log_{i}.txt"), "a"),
                                   stderr=subprocess.STDOUT))
    rcs = [q.wait() for q in ps]
    info = {"version": VERSION, "block": block, "procs": p, "cpu_quota": cpu_quota(), "units": len(us),
            "wall_s": time.time() - t0, "rcs": rcs, "host": host_info(), "est_cost_s_per_proc":
            [sum(u["est_cost_s"] for u in b) for b in bins]}
    json.dump(info, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    return max(rcs, default=0)


# ================================================================================================ lock / pins
def is_torch_build(name: str) -> bool:
    return name in ("torch", "triton") or (name.startswith("nvidia-") and name.endswith("-cu12"))


def write_lock() -> None:
    """uv export of uv.lock (hashes; project deps + every non-dev group) and the same minus the torch build."""
    import tomllib
    pp = tomllib.load(open(os.path.join(ROOT, "pyproject.toml"), "rb"))
    groups = [g for g in pp.get("dependency-groups", {}) if g != "dev"]
    path = os.path.join(ROOT, *LOCK_REL.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--format", "requirements-txt",
                    "-o", path] + [x for g in groups for x in ("--group", g)], cwd=ROOT, check=True,
                   capture_output=True)
    blocks: list[str] = []
    for line in open(path, encoding="utf-8").read().splitlines(keepends=True):
        if line.startswith((" ", "\t")) and blocks:
            blocks[-1] += line
        else:
            blocks.append(line)
    keep = [b for b in blocks if not (b.strip() and not b.startswith(("#", "-"))
                                      and is_torch_build(b.split(";")[0].split("==")[0].strip().lower()))]
    open(os.path.join(ROOT, *LOCK_INSTALL_REL.split("/")), "w", encoding="utf-8", newline="\n").write("".join(keep))
    print(f"[exp-c] {LOCK_REL} + {LOCK_INSTALL_REL} (groups {groups})")


def lock_pins(path: str) -> dict[str, str]:
    from packaging.markers import Marker
    from packaging.utils import canonicalize_name
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith((" ", "\t", "#", "-")) or "==" not in line:
            continue
        req, _, marker = line.strip().rstrip("\\").strip().partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        name, _, ver = req.partition("==")
        out[canonicalize_name(name.strip())] = ver.strip()
    return out


def pin_check(path: str) -> dict:
    """Installed vs locked versions (torch: public version only, any build; triton / nvidia-*-cu12 exempt)."""
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    inst: dict[str, str] = {}
    for d in md.distributions():
        if d.metadata["Name"]:
            inst.setdefault(canonicalize_name(d.metadata["Name"]), d.version)
    bad = {}
    for n, v in lock_pins(path).items():
        have = inst.get(n)
        if n == "torch":
            if have is None or Version(have).public != Version(v).public:
                bad[n] = [v, have]
        elif not is_torch_build(n) and (have is None or Version(have) != Version(v)):
            bad[n] = [v, have]
    return {"lock": path, "mismatches": bad, "torch_build": inst.get("torch"), "python": platform.python_version()}


# ================================================================================================ cloud
def _git() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def cloud_cmd(block: str, out_dir: str, tag: str, code: dict, wall_s: int, torch_v: str) -> str:
    log = f"{out_dir}/install.log"
    pip = "uv pip install --python /tmp/xm_venv/bin/python"
    setup = (f"mkdir -p {out_dir}; (command -v uv || pip install uv) > {log} 2>&1; uv python install {PYTHON} >> {log}"
             f" 2>&1; uv venv --python {PYTHON} /tmp/xm_venv >> {log} 2>&1; source /tmp/xm_venv/bin/activate; "
             f"unset UV_SYSTEM_PYTHON; python --version >> {log} 2>&1; "
             f"{pip} --require-hashes --no-deps -r {LOCK_INSTALL_REL} >> {log} 2>&1; "
             f"(python -c \"import importlib.metadata as m, sys; from packaging.version import Version; "
             f"sys.exit(Version(m.version('torch')).public != '{torch_v}')\" || {pip} --no-deps torch=={torch_v} "
             f"--index-url https://download.pytorch.org/whl/cpu) >> {log} 2>&1; "
             f"python {SELF_REL} pincheck > {out_dir}/pin_check.json 2>&1; nproc >> {log}; "
             f"cat /sys/fs/cgroup/cpu.max >> {log} 2>&1; ")
    env = (f"XM_PLATFORM={tag} XM_CODE_COMMIT={code['commit']} XM_CODE_DIRTY={int(code['dirty'])} "
           "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1")
    return setup + f"{env} timeout {wall_s} python -u {SELF_REL} run --block {block} --out-dir {out_dir}"


def kaggle_running() -> int:
    out = subprocess.run([KAGGLE, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "20"],
                         capture_output=True, text=True, timeout=180).stdout
    n = 0
    for ln in out.splitlines()[2:]:
        if ln.strip():
            s = subprocess.run([KAGGLE, "kernels", "status", ln.split()[0]], capture_output=True, text=True,
                               timeout=120).stdout
            n += "RUNNING" in s or "QUEUED" in s
    return n


def colab_mine() -> int:
    n = 0
    for f in glob.glob(os.path.join(ROOT, "scratchpad", "e6_dev", "runs", "xm-expc-*", "colab.json")):
        m = json.load(open(f, encoding="utf-8"))
        if m.get("kind") == "job" and m.get("launched") and not m.get("finished_utc") and not m.get("lost_utc") \
                and not os.path.exists(os.path.join(os.path.dirname(f), "exit_code")):
            n += 1
    return n


def launch(platform_: str, block: str, name: str, dry: bool, wall_s: int | None, slot5: bool = False) -> int:
    code = _git()
    if code["dirty"] and not dry:
        raise SystemExit("refused: tracked files are dirty (commit first; records stamp the commit)")
    torch_v = lock_pins(os.path.join(ROOT, *LOCK_REL.split("/")))["torch"].split("+")[0]
    paths = ["cdd_oran", *BUNDLE_DATA, SPEC_REL, SELF_REL, LOCK_REL, LOCK_INSTALL_REL]
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if platform_ == "kaggle":
        k = kaggle_running() if not dry else 0
        cap = KAGGLE_ACCOUNT_CAP if slot5 else KAGGLE_RUNNING_MAX
        if k >= cap:
            raise SystemExit(f"refused: {k} Kaggle kernels RUNNING / QUEUED (launch only below {cap})")
        cmd = cloud_cmd(block, "$JOB_OUT", "kaggle", code, wall_s or 11 * 3600, torch_v)
        argv = [py, os.path.join(ROOT, "scratchpad", "e6_dev", "kaggle_job.py"), "launch", name, "--cmd", cmd,
                "--paths", *paths, "--internet", "--pin", "off"]
        env = os.environ
    else:
        k = colab_mine()
        if k >= COLAB_MAX:
            raise SystemExit(f"refused: {k} exp-c Colab jobs active (max {COLAB_MAX})")
        cmd = cloud_cmd(block, "xm_out", "colab", code, wall_s or 3 * 3600, torch_v)
        argv = [py, os.path.join(ROOT, "scratchpad", "e6_dev", "colab_run.py"), "job", name, "--paths", *paths,
                "--cmd", cmd, "--out-dir", "xm_out", "--threads", "1"]
        env = {**os.environ, "COLAB_ACCEL": "cpu"}             # plain CPU runtime, as the DEV campaign's Colab shards
    print(" ".join(shlex.quote(x) for x in argv), flush=True)
    if not dry:
        subprocess.run(argv, cwd=ROOT, check=True, env=env)
    return 0


# ================================================================================================ cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-spec")
    sub.add_parser("lock")
    sub.add_parser("pincheck")
    p = sub.add_parser("units")
    p.add_argument("--block", required=True)
    p.add_argument("--procs", type=int, default=4)
    p = sub.add_parser("run")
    p.add_argument("--block", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p = sub.add_parser("worker")
    p.add_argument("--block", required=True)
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("launch")
    p.add_argument("platform", choices=("kaggle", "colab"))
    p.add_argument("--block", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--wall-s", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--slot5", action="store_true", help="Kaggle: allow the account's 5th slot (orchestrator only)")
    a = ap.parse_args(argv)
    if a.cmd == "make-spec":
        sp = make_spec()
        path = os.path.join(ROOT, *SPEC_REL.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(sp, open(path, "w", encoding="utf-8", newline="\n"), indent=1)
        print(f"[exp-c] {SPEC_REL}: {len(sp['arms'])} arms; calib {len(units(sp, 'calib'))} units, "
              f"paper {len(units(sp, 'paper'))} units")
    elif a.cmd == "lock":
        write_lock()
    elif a.cmd == "pincheck":
        print(json.dumps(pin_check(os.path.join(ROOT, *LOCK_REL.split("/")))))
    elif a.cmd == "units":
        us = units(load_spec(), a.block)
        bins = assign(us, a.procs)
        tot = sum(u["est_cost_s"] for u in us)
        print(f"[exp-c] {a.block}: {len(us)} units, est. {tot / 3600:.2f} CPU-h (DEV medians; unknown 30 s/1000 rows),"
              f" wall at {a.procs} procs ~{max(sum(u['est_cost_s'] for u in b) for b in bins) / 3600:.2f} h")
        for arm in dict.fromkeys(u["arm"] for u in us):
            xs = [u for u in us if u["arm"] == arm]
            print(f"  {arm}: {len(xs)} units, n {sorted({u['n'] for u in xs})}, {xs[0]['world']} {xs[0]['regime']},"
                  f" est {sum(u['est_cost_s'] for u in xs):.0f} s")
    elif a.cmd == "run":
        return run(load_spec(), a.block, a.out_dir, a.procs, a.max_units)
    elif a.cmd == "worker":
        worker(load_spec(), json.load(open(a.units)), a.out, a.block)
    else:
        return launch(a.platform, a.block, a.name, a.dry_run, a.wall_s, a.slot5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
