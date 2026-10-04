"""DEV campaign driver of the cross-method study (brief dev-runs): validity, power and cost per
(method arm, world, regime, lam, n, kappa) cell, sharded so any platform can take any shard (ruling R-26).

    uv run python -m cdd_oran.xmethod.campaign list      --spec SPEC.json
    uv run python -m cdd_oran.xmethod.campaign run       --spec SPEC.json --part i/P --out res_i.jsonl [--budget S]
    uv run python -m cdd_oran.xmethod.campaign merge     --spec SPEC.json --inputs 'runs/x/**/*.jsonl' --out m.jsonl
    uv run python -m cdd_oran.xmethod.campaign aggregate --spec SPEC.json --merged m.jsonl --out agg.json
    uv run python -m cdd_oran.xmethod.campaign project   --agg PILOT_AGG.json --spec FULL.json --out proj.json
    uv run python -m cdd_oran.xmethod.campaign kaggle    --spec SPEC.json --name NAME [--parts 4] [--part-set i,j/P]

SPEC (JSON, repo-relative path so cloud bundles carry it)::

    {"name": "dev_pilot",
     "budget_cpu_s": 7200,                                   # R-13 per-(method, dataset) budget
     "arms": {"pmrt_eq": {"ref": "cdd_oran.xmethod.methods.pmrt_core:PmrtCore", "config": {"covariates": "eq"},
                          "declare": "by"},                  # "by" (p-values, pooled BY) | "tau" (placebo rule)
              "granger_eq": {..., "worlds": ["E3"]}, ...},   # optional "worlds" / "max_n" restrictions
     "blocks": [{"role": "measure" | "tune", "arms": "all" | [...], "worlds": [...], "regimes": [...],
                 "ns": [...], "kappas": [0.25], "seeds": [lo, hi] | [s1, s2, ...],
                 "e4_lams": {"R1": [1.0], "R2": [1.0], "R3": [0, .5, 1, 1.5], "R4": [0, .5, 1, 1.5]}}]}

A UNIT (= one shard, one JSON record) is (arm, world, regime, lam, n, kappa, seed); its key is the runner's job key
with the arm name as the method ("<arm>|<world>|<regime>|[lam<l>|]k<kappa>|n<n>|s<seed>"). Units sharing a dataset
go to the same part (the dataset is generated once); parts are balanced by a cost estimate (pilot cost table if
given). ``run`` appends one line per unit and skips units already in ``--out`` (resumable). On Linux each unit runs
in a forked child under RLIMIT_CPU = budget: a unit that exceeds it is recorded ``status: "infeasible"`` with the
measured CPU-s (R-13), and the same (arm, cell) at larger n in that part is recorded infeasible without running.
Records carry the code commit, package versions and the host (``XM_PLATFORM``).

``merge`` keeps one record per key (ok > infeasible > error; later beats earlier within a class), tolerant of
missing and duplicate shards. ``aggregate`` computes, per cell, on MEASUREMENT seeds only: truth-null primary-edge
rejection (BY / tau declaration and raw p <= .05), P_placebo / P_placebo_conf declaration and raw-p rates with
seed-cluster bootstrap CIs, per-seed recall / FDP / sign accuracy, cost. Score-only ("tau") arms declare
score > tau, tau = 2nd-largest P_placebo score over the cell's TUNE-seed records (score.placebo_tau; R-2); tune and
measurement seeds are disjoint by construction. Truth enters only here (scoring), never a method.

EVAL mode (PROTOCOL_A section 11, ruling R-35): a spec with any seed >= 3_100_000. Refused unless ``eval_authorised``:
the spec's CONSTANT ``protocol_sha256`` (the frozen PROTOCOL_A blob) is verified against a frozen protocol text (the
live file, the git blob at ``freeze_commit``, or the launcher's bundled frozen copy) and the live PROTOCOL_A says
'FROZEN: yes'. Measure blocks use EVAL seeds only, tune blocks DEV seeds (re-run). A run also needs a clean, known
code commit and every uv.lock pin installed (``pin_check``; launchers install the full lock export). Every record
carries ``integrity`` (code commit / dirty, protocol sha, spec sha, python, lock sha, isolation, budget, cost-table
sha, part index / count) and its ``limits``. EVAL budget = the spec's (no CLI override), safety cap 2x: a unit over it
is infeasible with its CPU-s, never re-run (PROTOCOL_A s.7, R-59); EVAL runs isolated only; feasibility comes from
the spec (arm "infeasible_n" = DEV T3 cost); a child killed below its CPU limit (OOM) is an error in both modes.
EVAL and DEV shard files never mix (``run`` refuses, ``merge`` and --skip-complete-from drop the other mode / spec);
merge keeps each dataset's arms on one platform where possible, reports the rest and the integrity sets, and
refuses an EVAL run mixing cost tables or part counts. A session's exit status is non-zero if any part failed.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

from cdd_oran.xmethod import api
from cdd_oran.xmethod import runner as R
from cdd_oran.xmethod.score import PLACEBO, PLACEBO_CONF, apply_threshold, placebo_tau, score
from cdd_oran.xmethod.worlds import REGIMES_OF, generate_dataset, truth_for

CAMPAIGN_VERSION = "xm-campaign/1"
ROLES = ("tune", "measure")
DECLARE = ("by", "tau")
DEFAULT_E4_LAMS = {"R1": [1.0], "R2": [1.0], "R3": [0.0, 0.5, 1.0, 1.5], "R4": [0.0, 0.5, 1.0, 1.5]}
PKGS = ("numpy", "scipy", "scikit-learn", "statsmodels", "causal-learn", "xgboost", "shap", "torch",
        "threadpoolctl", "tigramite", "dcor", "momentchi2")
ALPHA = 0.05
BOOT_REPS = 2000
BOOT_SEED = 20261002
MIN_FLAG_SEEDS = 10          # fewer measurement seeds: the cluster bootstrap is degenerate, no validity flag
# EVAL mode (PROTOCOL_A section 11, ruling R-35)
GIT_ROOT = R.ROOT
PROTOCOL_REL = "docs/xmethod/PROTOCOL_A.md"
PROTOCOL_PATH = os.path.join(R.ROOT, *PROTOCOL_REL.split("/"))
FROZEN_MARK = "FROZEN: yes"
BUNDLE_REL = "scratchpad/xmethod/_bundle"                    # launcher-written files carried by cloud bundles
LOCK_REL = f"{BUNDLE_REL}/requirements.lock.txt"            # uv export of uv.lock (hashes)
LOCK_PATH = os.path.join(R.ROOT, *LOCK_REL.split("/"))
LOCK_INSTALL_REL = f"{BUNDLE_REL}/requirements.cloud.txt"  # the lock minus the torch build (R-35 amendment)
LOCK_INSTALL_PATH = os.path.join(R.ROOT, *LOCK_INSTALL_REL.split("/"))
FROZEN_COPY_REL = f"{BUNDLE_REL}/PROTOCOL_A.frozen.md"      # the verified frozen protocol text (cloud guard)
FROZEN_COPY_PATH = os.path.join(R.ROOT, *FROZEN_COPY_REL.split("/"))
BUNDLE_DATA = ("configs", "scripts")                       # imported / read by cdd_oran (config yaml; two_tower)
EVAL_CAP_FACTOR = 2.0                                       # EVAL budget = safety cap (R-35)
EVAL_PYTHON = "3.12.14"                     # EVAL interpreter, exact patch (Q7, R-57: uv's "3.12" is 3.12.15 now)
DIRTY_SPEC_DIR = "scratchpad/xmethod/specs/"
LIGHTNING_PYTHON = os.path.expanduser("~/.cloudtools/Scripts/python.exe")  # interpreter with lightning_sdk
VPS_MAX_PROCS = 7                                           # user VPS: systemd scope CPUQuota=700% (vps_run.py)


# ================================================================================================ spec / units
@dataclasses.dataclass(frozen=True, order=True)
class Unit:
    world: str
    regime: str
    lam: float | None
    n: int
    kappa: float
    seed: int
    arm: str
    role: str

    @property
    def key(self) -> str:
        return R.job_key(self.arm, self.world, self.regime, self.lam, self.n, self.seed, self.kappa)

    @property
    def dataset(self) -> tuple:
        return (self.world, self.regime, self.lam, self.n, self.seed, self.kappa)

    @property
    def cell(self) -> str:
        """Cell key incl. the arm: '<arm>|<world>|<regime>|[lam<l>|]k<kappa>|n<n>'."""
        return f"{self.arm}|{R.cell_key(self.world, self.regime, self.lam, self.n, self.kappa)}"


def _sha_lf(data: bytes) -> str:
    """sha256 of ``data`` with CRLF normalised to LF (git blob content of a text file)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def protocol_sha256(path: str | None = None) -> str:
    """LF-normalised sha256 of the protocol file (the hash the freeze commit records)."""
    return _sha_lf(open(path or PROTOCOL_PATH, "rb").read())


def _is_frozen(data: bytes) -> bool:
    return any(line.startswith(FROZEN_MARK) for line in data.replace(b"\r\n", b"\n").decode("utf-8").split("\n"))


def _git_blob(commit: str, relpath: str) -> bytes | None:
    """Content of ``relpath`` at ``commit`` (git blob, LF as committed), or None without git / the commit."""
    try:
        p = subprocess.run(["git", "-C", GIT_ROOT, "show", f"{commit}:{relpath}"], capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def eval_authorised(spec: dict) -> tuple[bool, str]:
    """EVAL guard (ruling R-35). The spec carries the CONSTANT ``protocol_sha256`` = LF sha256 of PROTOCOL_A as
    frozen (the freeze-commit blob). Allowed only if (1) the live docs/xmethod/PROTOCOL_A.md has a line starting
    'FROZEN: yes' and (2) the constant is verified against the frozen text by one of: the live file itself (no
    amendment since the freeze); the git blob at ``spec['freeze_commit']``; the frozen copy the launcher bundles
    (FROZEN_COPY_PATH). Each verifying text must itself say 'FROZEN: yes'. Amendments appended to the live file
    after the freeze therefore do not lock out re-runs, and an unfrozen or unknown protocol always refuses."""
    want = str(spec.get("protocol_sha256") or "").lower()
    if not want:
        return False, "spec has no 'protocol_sha256'"
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        return False, f"spec 'protocol_sha256' is not a sha256 hex digest: {want!r}"
    if not os.path.exists(PROTOCOL_PATH):
        return False, f"protocol file missing: {PROTOCOL_PATH}"
    live = open(PROTOCOL_PATH, "rb").read()
    if not _is_frozen(live):
        return False, "protocol file is not frozen (no line starting 'FROZEN: yes')"
    sources = [("live file", live)]
    if spec.get("freeze_commit"):
        sources.append((f"git blob at {spec['freeze_commit']}", _git_blob(str(spec["freeze_commit"]),
                                                                          PROTOCOL_REL)))
    if os.path.exists(FROZEN_COPY_PATH):
        sources.append(("bundled frozen copy", open(FROZEN_COPY_PATH, "rb").read()))
    for name, data in sources:
        if data is not None and _sha_lf(data) == want and _is_frozen(data):
            return True, f"EVAL authorised: protocol sha256 {want} verified by the {name}"
    return False, (f"protocol sha256 constant {want} matches no frozen protocol text (checked: "
                   f"{', '.join(n for n, d in sources if d is not None)})")


def _seed_list(s, spec: dict | None = None) -> list[int]:
    """Seeds of a block. DEV block always allowed; seeds >= 3_100_000 (EVAL) only under ``eval_authorised``;
    anything else refused."""
    if isinstance(s, str) or any(isinstance(x, str) for x in s):
        raise ValueError(f"seeds must be numbers (a 'TBD' placeholder is filled at the freeze): {s!r}")
    seeds = list(range(int(s[0]), int(s[1]) + 1)) if (len(s) == 2 and s[1] > s[0] + 1) else [int(x) for x in s]
    ev = [x for x in seeds if x >= R.EVAL_FLOOR]
    bad = [x for x in seeds if x not in R.DEV_SEEDS and x < R.EVAL_FLOOR]
    if bad:
        raise ValueError(f"seeds outside the DEV block 3000000-3000199 (CONTRACT 6): {bad[:5]}")
    if ev:
        ok, why = eval_authorised(spec or {})
        if not ok:
            raise ValueError(f"EVAL seeds {ev[:3]} refused: {why}")
    return seeds


def is_eval(spec: dict) -> bool:
    """EVAL mode = any block seed >= 3_100_000 (unparseable seeds are left to ``_seed_list`` to refuse)."""
    for b in spec.get("blocks", []):
        try:
            if any(int(x) >= R.EVAL_FLOOR for x in b.get("seeds", [])):
                return True
        except (TypeError, ValueError):
            continue
    return False


def spec_sha256(spec: dict) -> str:
    """sha256 of the canonical JSON of the spec (independent of file formatting)."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_spec(spec: dict) -> None:
    arms = spec.get("arms")
    if not arms:
        raise ValueError("spec needs 'arms'")
    for a, d in arms.items():
        if "|" in a or not d.get("ref"):
            raise ValueError(f"arm {a!r}: needs a 'ref' and a name without '|'")
        if d.get("declare") not in DECLARE:
            raise ValueError(f"arm {a!r}: 'declare' must be one of {DECLARE}")
    for b in spec.get("blocks", []):
        if b.get("role") not in ROLES:
            raise ValueError(f"block role must be one of {ROLES}: {b}")
        R._kappas(b)                                       # kappas required, finite, >= 0 (R-24)
        sel = b.get("arms", "all")
        if sel != "all" and set(sel) - set(arms):
            raise ValueError(f"block names unknown arms: {sorted(set(sel) - set(arms))}")
    if is_eval(spec):                                  # EVAL: tune = DEV seeds (re-run, R-34), measure = EVAL
        for b in spec["blocks"]:
            sd = _seed_list(b["seeds"], spec)
            if b["role"] == "measure" and min(sd) < R.EVAL_FLOOR:
                raise ValueError("EVAL spec: measure blocks must use EVAL seeds (>= 3100000) only")
            if b["role"] == "tune" and max(sd) >= R.EVAL_FLOOR:
                raise ValueError("EVAL spec: tune blocks must use DEV seeds only")
    tune = {s for b in spec["blocks"] if b["role"] == "tune" for s in _seed_list(b["seeds"], spec)}
    meas = {s for b in spec["blocks"] if b["role"] == "measure" for s in _seed_list(b["seeds"], spec)}
    if tune & meas:
        raise ValueError(f"tune and measurement seeds overlap: {sorted(tune & meas)[:5]}")


def expand(spec: dict) -> list[Unit]:
    """All units of the spec, sorted (dataset-major), deduplicated across blocks."""
    validate_spec(spec)
    arms = spec["arms"]
    out: dict[str, Unit] = {}
    for b in spec["blocks"]:
        names = list(arms) if b.get("arms", "all") == "all" else list(b["arms"])
        lam_map = {**DEFAULT_E4_LAMS, **b.get("e4_lams", {})}
        for w in b["worlds"]:
            for r in b["regimes"]:
                if r not in REGIMES_OF[w]:
                    continue
                for lam in ([float(x) for x in lam_map[r]] if w == "E4" else [None]):
                    for kap in R._kappas(b):
                        for n in b["ns"]:
                            for s in _seed_list(b["seeds"], spec):
                                for a in names:
                                    d = arms[a]
                                    if "worlds" in d and w not in d["worlds"]:
                                        continue
                                    if "max_n" in d and int(n) > int(d["max_n"]):
                                        continue
                                    u = Unit(w, r, lam, int(n), float(kap), int(s), a, b["role"])
                                    out.setdefault(u.key, u)
    return sorted(out.values())


# ================================================================================================ partition
def unit_cost(u: Unit, cost_table: dict | None) -> float:
    """Estimated CPU-s of a unit: the pilot's mean for (arm, world, regime, n) if known, else proportional to n."""
    if cost_table:
        c = cost_table.get(f"{u.arm}|{u.world}|{u.regime}|n{u.n}")
        if c is not None:
            return float(c)
    return u.n / 1000.0


def partition(units: list[Unit], parts: int, cost_table: dict | None = None) -> list[list[Unit]]:
    """Whole dataset groups to parts, greedy longest-processing-time on the estimated cost. Deterministic. Within a
    part: ascending n, then dataset (each dataset generated once), so a session cut by its wall limit loses the
    largest units first."""
    groups: dict[tuple, list[Unit]] = defaultdict(list)
    for u in units:
        groups[u.dataset].append(u)
    order = sorted(groups, key=lambda g: (-sum(unit_cost(u, cost_table) for u in groups[g]), str(g)))
    load = [0.0] * parts
    out: list[list[Unit]] = [[] for _ in range(parts)]
    for g in order:
        i = int(np.argmin(load))
        out[i].extend(groups[g])
        load[i] += sum(unit_cost(u, cost_table) for u in groups[g])
    return [sorted(p, key=lambda u: (u.n, str(u.dataset), u.arm)) for p in out]   # small n first, dataset-major


# ================================================================================================ provenance
def dirty_paths(porcelain: str) -> list[str]:
    """Paths of ``git status --porcelain --untracked-files=all --ignored=matching`` output that make a tree dirty
    for a campaign record (R-35): any tracked change; untracked or ignored files only if they are code under
    cdd_oran/ or scratchpad/xmethod/ (*.py) or a spec (scratchpad/xmethod/specs/). Caches are never dirt."""
    out = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        flag, path = line[:2], line[3:].strip().strip('"').split(" -> ")[-1]
        if "__pycache__" in path or path.endswith((".pyc", ".pyo")):
            continue
        if flag in ("??", "!!"):
            code = ((path.endswith(".py") and path.startswith(("cdd_oran/", "scratchpad/xmethod/")))
                    or (path.endswith("/") and path.startswith("cdd_oran/")))     # a whole new package dir
            if not (code or path.startswith(DIRTY_SPEC_DIR)):
                continue
        out.append(path)
    return out


def code_info() -> dict:
    """Commit of the code that produced a record. Order: XM_CODE_COMMIT (+ XM_CODE_DIRTY "0"/"1", set by the
    launcher from its own git check), else git (dirty = ``dirty_paths``), else a cloud bundle manifest (its dirty
    flag covers cdd_oran/ and scratchpad/e6_dev/ only: scope recorded)."""
    if os.environ.get("XM_CODE_COMMIT"):
        d = os.environ.get("XM_CODE_DIRTY")
        return {"commit": os.environ["XM_CODE_COMMIT"], "dirty": None if d is None else d == "1",
                "source": "env (launcher git check)", "dirty_paths": None}
    try:
        head = subprocess.run(["git", "-C", GIT_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
        if head:
            st = subprocess.run(["git", "-C", GIT_ROOT, "status", "--porcelain", "--untracked-files=all",
                                 "--ignored=matching", "--", "."], capture_output=True, text=True, timeout=30).stdout
            dp = dirty_paths(st)
            return {"commit": head, "dirty": bool(dp), "source": "git", "dirty_paths": dp[:20]}
    except (OSError, subprocess.SubprocessError):
        pass
    for f in ("MANIFEST.json", "JOB_MANIFEST.json"):
        p = os.path.join(R.ROOT, f)
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            return {"commit": m.get("git_head"), "dirty": m.get("dirty", m.get("e6_dirty")),
                    "source": f"{f} (dirty scope cdd_oran/, scratchpad/e6_dev/)", "dirty_paths": None}
    return {"commit": None, "dirty": None, "source": None, "dirty_paths": None}


def lock_pins(path: str) -> dict[str, str]:
    """{normalised name: version} of the ``name==version`` lines of a ``uv export`` requirements file whose
    environment marker holds on this interpreter."""
    from packaging.markers import Marker
    from packaging.utils import canonicalize_name
    pins = {}
    text = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    for raw in text.replace("\\\n", " ").splitlines():
        line = raw.split("--hash")[0].strip()
        if not line or line.startswith(("#", "-")) or "==" not in line:
            continue
        req, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        name, _, ver = req.partition("==")
        pins[canonicalize_name(name.split("[")[0].strip())] = ver.strip()
    return pins


def is_torch_build(name: str) -> bool:
    """Packages of the torch BUILD (R-35 amendment, user 2026-10-03): torch itself, triton and the CUDA 12 runtime
    wheels torch+cu128 pulls. (nvidia-nccl-cu13 belongs to xgboost and stays pinned.)"""
    return name in ("torch", "triton") or (name.startswith("nvidia-") and name.endswith("-cu12"))


def pin_check(path: str | None = None, python: str | None = None) -> dict:
    """Installed versions vs the bundled uv.lock export (R-35). Every pin must match exactly, EXCEPT the torch
    build (amendment): torch must have the locked PUBLIC version (2.10.0) but any build (+cpu / +cu128) matching
    the session hardware; triton / nvidia-*-cu12 are not required (part of a CUDA build). ``mismatches`` maps
    name -> [locked, installed]; ``torch_build`` = the installed torch version string (with its build label).
    ``python`` (EVAL, R-57): the exact interpreter version required; another patch is a "python" mismatch."""
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    path = path or os.environ.get("XM_LOCK_FILE") or LOCK_PATH
    inst: dict[str, str] = {}
    for d in md.distributions():                        # sys.path order: the FIRST copy is the one imported
        if d.metadata["Name"]:                          # (cloud images carry stale duplicates, e.g. numpy 1.26)
            inst.setdefault(canonicalize_name(d.metadata["Name"]), d.version)
    py = platform.python_version()
    if not os.path.exists(path):
        return {"lock": None, "n_pins": 0, "mismatches": None, "torch_build": inst.get("torch"), "python": py}
    pins = lock_pins(path)
    bad = {}
    for n, v in pins.items():
        have = inst.get(n)
        if n == "torch":
            if have is None or Version(have).public != Version(v).public:
                bad[n] = [v, have]
        elif is_torch_build(n):
            continue
        elif have is None or Version(have) != Version(v):
            bad[n] = [v, have]
    if python is not None and py != python:
        bad["python"] = [python, py]
    try:
        rel = os.path.relpath(path, R.ROOT).replace("\\", "/")
    except ValueError:                                  # another drive (Windows)
        rel = path
    return {"lock": rel, "lock_sha256": _sha_lf(open(path, "rb").read()), "n_pins": len(pins), "mismatches": bad,
            "torch_build": inst.get("torch"), "python": py, "python_required": python,
            "torch_build_exempt": {n: inst.get(n) for n in pins if is_torch_build(n) and n != "torch"}}


def _req_blocks(text: str) -> list[str]:
    """Requirement entries of a ``uv export`` file (an entry = its line + indented ``--hash`` continuation lines);
    comments and blank lines are their own entries."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith((" ", "\t")) and out:
            out[-1] += line
        else:
            out.append(line)
    return out


def write_lock(path: str | None = None, install_path: str | None = None) -> str:
    """``uv export`` of uv.lock (hashes; project deps + every non-dev dependency group) -> ``path`` (the pin
    reference), and ``install_path`` = the same WITHOUT the torch build (installed as the image's matching build,
    R-35 amendment; installed with --no-deps since the export lists every dependency)."""
    import tomllib

    from packaging.utils import canonicalize_name
    path = path or LOCK_PATH
    install_path = install_path or LOCK_INSTALL_PATH
    pp = tomllib.load(open(os.path.join(GIT_ROOT, "pyproject.toml"), "rb"))
    groups = [g for g in pp.get("dependency-groups", {}) if g != "dev"]
    cmd = ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--format", "requirements-txt",
           "-o", path] + [x for g in groups for x in ("--group", g)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(cmd, cwd=GIT_ROOT, check=True, capture_output=True)
    keep = []
    for b in _req_blocks(open(path, encoding="utf-8").read()):
        head = b.split(";")[0].split("==")[0].strip()
        if head and not head.startswith(("#", "-")) and is_torch_build(canonicalize_name(head)):
            continue
        keep.append(b)
    open(install_path, "w", encoding="utf-8", newline="\n").write("".join(keep))
    return path


def lock_groups_and_indexes() -> tuple[list[str], list[str]]:
    import tomllib
    pp = tomllib.load(open(os.path.join(GIT_ROOT, "pyproject.toml"), "rb"))
    return ([g for g in pp.get("dependency-groups", {}) if g != "dev"],
            [i["url"] for i in pp.get("tool", {}).get("uv", {}).get("index", []) if i.get("url")])


def pkg_versions() -> dict:
    out = {}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def host_info() -> dict:
    cpu = platform.processor() or None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    try:                                                # GPU sessions only (R-41: GPU time reported apart)
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        gpu = [x.strip() for x in p.stdout.splitlines() if x.strip()] or None if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"platform": os.environ.get("XM_PLATFORM", "local"), "cpu_model": cpu, "cpu_count": os.cpu_count(),
            "node": platform.node(), "gpu": gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def cpu_quota(cgroup: str = "/sys/fs/cgroup", self_cgroup: str = "/proc/self/cgroup") -> int:
    """vCPUs this process may use (R-55): the tightest cgroup CPU quota on the path from this process's own cgroup
    (v2, from /proc/self/cgroup "0::<path>") up to the root (cpu.max; e.g. the VPS systemd scope CPUQuota=700% sits on
    the scope's cgroup, not the root), else the v1 cfs quota; capped by the affinity mask; the visible CPU count when
    there is no quota (e.g. Colab's TPU host shows 24 vCPUs with a 4-CPU quota)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    own = None
    try:
        for line in open(self_cgroup):
            if line.startswith("0::"):
                own = line.strip()[3:]
    except OSError:
        pass
    root = os.path.normpath(cgroup)
    dirs = [root]
    if own and own != "/":
        d = os.path.normpath(os.path.join(root, own.lstrip("/")))
        dirs = []
        while d.startswith(root) and len(d) >= len(root):
            dirs.append(d)
            if d == root:
                break
            d = os.path.dirname(d)
    qs = []
    for d in dirs:
        try:
            a, b = open(os.path.join(d, "cpu.max")).read().split()[:2]
            if a != "max":
                qs.append(float(a) / float(b))
        except (OSError, ValueError):
            continue
    if not qs:
        for d in ("cpu", "cpu,cpuacct"):
            try:
                a = float(open(os.path.join(root, d, "cpu.cfs_quota_us")).read())
                b = float(open(os.path.join(root, d, "cpu.cfs_period_us")).read())
                if a > 0:
                    qs.append(a / b)
                break
            except (OSError, ValueError):
                continue
    q = min(qs) if qs else None
    return max(1, min(n, math.ceil(q - 1e-9))) if q else max(1, n)


def campaign_procs(proc: str = "/proc") -> int | None:
    """Concurrent `campaign run` processes on this host (R-55) = distinct --out shard files among the processes whose
    command line runs it, so a part counts once whatever wraps it (timeout, forked unit children); None without /proc.
    (A parent-pid rule undercounted on Kaggle: 1 with 4 parts running, 2026-10-04.)"""
    if not os.path.isdir(proc):
        return None
    outs = set()
    for d in os.listdir(proc):
        if not d.isdigit():
            continue
        try:
            argv = open(os.path.join(proc, d, "cmdline"), "rb").read().split(bytes(1))
        except OSError:
            continue
        if b"cdd_oran.xmethod.campaign" not in argv or b"run" not in argv:
            continue
        key = argv[argv.index(b"--out") + 1] if b"--out" in argv[:-1] else d.encode()
        outs.add(key)
    return len(outs)


def proc_visible(proc: str = "/proc") -> int | None:
    """Processes listed in /proc (a restricted /proc would hide other parts' processes)."""
    return sum(1 for d in os.listdir(proc) if d.isdigit()) if os.path.isdir(proc) else None


def load_info(proc: str = "/proc") -> dict:
    """Host load around a unit (R-55): 1 / 5 / 15 min loadavg, running / total processes, concurrent campaign
    processes, vCPU quota."""
    out = {"loadavg": None, "procs_running": None, "procs_total": None, "campaign_procs": campaign_procs(proc),
           "proc_visible": proc_visible(proc), "cpu_quota": cpu_quota()}
    try:
        f = open(os.path.join(proc, "loadavg")).read().split()
        r, t = f[3].split("/")
        out.update(loadavg=[float(x) for x in f[:3]], procs_running=int(r), procs_total=int(t))
    except (OSError, ValueError, IndexError):
        if hasattr(os, "getloadavg"):
            out["loadavg"] = list(os.getloadavg())
    return out


def check_one_per_cpu(n_procs: int | None, quota: int) -> None:
    """EVAL (R-55): at most one campaign process per vCPU of the session."""
    if n_procs is not None and n_procs > quota:
        raise ValueError(f"EVAL refused: {n_procs} campaign processes on {quota} vCPU(s); R-55 allows one per vCPU")


# ================================================================================================ run
def _infeasible(u: Unit, cfg: dict, cpu_s: float | None, reason: str, rss_mb: float | None = None) -> dict:
    return {"key": u.key, "method": u.arm, "status": "infeasible", "error": None, "reason": reason,
            "job": {"world": u.world, "regime": u.regime, "lam": u.lam, "n": u.n, "seed": u.seed, "kappa": u.kappa},
            "cpu_s": cpu_s, "peak_rss_mb": rss_mb, "config": cfg}


def _failed(u: Unit, cfg: dict, cpu_s: float | None, error: str, rss_mb: float | None = None) -> dict:
    return {**_infeasible(u, cfg, cpu_s, "", rss_mb), "status": "error", "error": error, "reason": None}


def classify_kill(sig: int | None, cpu_s: float, limit_s: float | None, wall_hit: bool = False) -> str:
    """Why a child died: "wall_limit" (the parent killed it at the arm's wall budget, R-41), "cpu_limit" (SIGXCPU,
    or SIGKILL at the hard CPU limit) or "killed" (any other signal, e.g. the kernel OOM killer's SIGKILL: an
    ERROR, never "infeasible"; R-35)."""
    import signal
    if wall_hit:
        return "wall_limit"
    sigxcpu, sigkill = getattr(signal, "SIGXCPU", 24), getattr(signal, "SIGKILL", 9)   # Linux numbers
    if sig == sigxcpu:
        return "cpu_limit"
    if sig == sigkill and limit_s and cpu_s >= limit_s - 1.0:
        return "cpu_limit"
    return "killed"


def _run_isolated(method, ds, truth, cfg: dict, gen_s: float, key: str, limit_s: float | None,
                  wall_s: float | None = None):
    """Run one unit in a forked child under RLIMIT_CPU = limit_s (Linux); the parent kills it after ``wall_s``
    seconds of wall clock (GPU arms, R-41). Returns (record | None, child cpu_s, child maxrss MB, terminating
    signal | None, child wall s, wall limit hit)."""
    import resource
    import select
    import signal
    rfd, wfd = os.pipe()
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:                                                       # child
        code = 0
        try:
            os.close(rfd)
            if limit_s:
                resource.setrlimit(resource.RLIMIT_CPU, (int(math.ceil(limit_s)), int(math.ceil(limit_s)) + 10))
            rec = R.run_one(method, ds, truth, cfg, gen_s, key)
            data = json.dumps(rec, allow_nan=False).encode()
            with os.fdopen(wfd, "wb") as fh:
                fh.write(data)
        except BaseException:
            code = 1
        os._exit(code)
    os.close(wfd)
    chunks, hit = [], False
    with os.fdopen(rfd, "rb") as fh:
        while True:
            left = None if not wall_s else wall_s - (time.perf_counter() - t0)
            if left is not None and left <= 0:
                os.kill(pid, signal.SIGKILL)
                hit = True
                break
            ready, _, _ = select.select([fh], [], [], left)
            if not ready:
                continue
            b = fh.read1(1 << 20) if hasattr(fh, "read1") else fh.read(1 << 20)
            if not b:
                break
            chunks.append(b)
    _, status, ru = os.wait4(pid, 0)
    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
    rec = json.loads(b"".join(chunks)) if chunks and sig is None and not hit else None
    return rec, ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024.0, sig, time.perf_counter() - t0, hit


def _out_modes(path: str) -> set[tuple]:
    """{(mode, protocol_sha256)} of the records already in ``path`` (old DEV records: ("dev", None))."""
    out = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                rm = json.loads(line).get("run_mode") or {}
            except json.JSONDecodeError:
                continue
            out.add((rm.get("mode", "dev"), rm.get("protocol_sha256") if rm.get("mode") == "eval" else None))
    return out


def check_eval_preconditions(spec: dict, code: dict, pins: dict) -> None:
    """EVAL run preconditions (R-35, Q7): authorised protocol, Python 3.12, clean known code, every lock pin
    installed."""
    ok, why = eval_authorised(spec)
    if not ok:
        raise ValueError(f"EVAL refused: {why}")
    if platform.python_version() != EVAL_PYTHON:
        raise ValueError(f"EVAL refused: Python {platform.python_version()}, EVAL requires exactly {EVAL_PYTHON} "
                         "(Q7, R-57)")
    if not code.get("commit") or code.get("dirty") is not False:
        raise ValueError(f"EVAL refused: code commit unknown or not clean ({code})")
    if pins.get("mismatches") is None:
        raise ValueError(f"EVAL refused: no uv.lock export at {LOCK_PATH} (or XM_LOCK_FILE)")
    if pins["mismatches"]:
        raise ValueError(f"EVAL refused: installed packages differ from uv.lock: {dict(list(pins['mismatches'].items())[:8])}")


def arm_budgets(spec: dict, arm: str, budget: float | None) -> tuple[float | None, float | None]:
    """(CPU-s budget, wall-s budget) of an arm: arm "budget_cpu_s" overrides the run's budget (null = none) and
    "budget_wall_s" adds a wall-clock budget (GPU arms, R-41: 2 h wall per dataset on a T4)."""
    a = spec["arms"][arm]
    cpu = a["budget_cpu_s"] if "budget_cpu_s" in a else budget
    return (float(cpu) if cpu else None), (float(a["budget_wall_s"]) if a.get("budget_wall_s") else None)


def read_registry(path: str | None) -> dict[str, tuple[int, float, str]]:
    """Shared infeasibility registry (DEV): arm -> (smallest n over budget, measured cost, kind)."""
    out: dict[str, tuple[int, float, str]] = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["arm"] not in out or int(r["n"]) < out[r["arm"]][0]:
                out[r["arm"]] = (int(r["n"]), float(r["cost"]), r.get("kind", "CPU-s"))
    return out


def run_units(spec: dict, units: list[Unit], out: str, budget: float | None = None, isolate: bool | None = None,
              log=print, spec_file_sha256: str | None = None, skip_keys: set[str] | None = None,
              eval_rules: bool | None = None, registry: str | None = None, stamp: dict | None = None) -> int:
    """Run ``units`` (sorted dataset-major) appending one record each to ``out``; skips done keys and
    ``skip_keys``. Budgets per arm (``arm_budgets``): CPU-s via RLIMIT_CPU, wall-s (GPU arms, R-41) by the parent.
    DEV rules: a unit over its budget is "infeasible" with its measured cost; larger n of that (arm, cell) in this
    part are recorded infeasible unrun, and with ``registry`` (a file shared by the processes of a session) every
    unit of that ARM at n >= the infeasible n, in any cell, is recorded infeasible unrun (T3 is per (arm, n)).
    EVAL rules (R-35; ``eval_rules`` defaults to ``is_eval(spec)``): feasibility comes only from the spec (arm
    "infeasible_n" = DEV T3 cost), the budget is a safety cap (EVAL_CAP_FACTOR x budget): a unit over it is
    recorded infeasible with its CPU-s and never re-run (PROTOCOL_A s.7, R-59 F2), and nothing propagates. EVAL
    also refuses a budget other than the spec's (F6) and non-isolated runs (F7). Either way a child killed below
    its limits (OOM, other signals) is an error (re-run), and arm "infeasible_n" units are recorded unrun.
    ``stamp``: run-level integrity fields of the caller (run_part: cost-table sha, part index / count, F4); the
    isolation mode, budget and cap factor are stamped too, and each record carries its effective ``limits``."""
    ev = is_eval(spec) if eval_rules is None else eval_rules
    if ev:
        registry = None
    if isolate is None:
        isolate = hasattr(os, "fork") and sys.platform.startswith("linux")
    code, pkgs, host = code_info(), pkg_versions(), host_info()
    pins = pin_check(python=EVAL_PYTHON if is_eval(spec) else None)
    mode = {"mode": "eval" if is_eval(spec) else "dev", "protocol_sha256": spec.get("protocol_sha256")}
    if is_eval(spec):
        check_eval_preconditions(spec, code, pins)
        check_one_per_cpu(campaign_procs(), cpu_quota())
        budget = spec.get("budget_cpu_s") if budget is None else budget
        if budget != spec.get("budget_cpu_s"):                    # R-59 F6: the cap is the spec's, never a CLI's
            raise ValueError(f"EVAL refused: --budget {budget} differs from the spec budget_cpu_s "
                             f"{spec.get('budget_cpu_s')} (R-59 F6)")
        if not isolate:                                            # R-59 F7: RLIMIT_CPU + one dataset copy per arm
            raise ValueError("EVAL refused: units must run isolated (fork + RLIMIT_CPU); --no-isolate or no fork "
                             "on this platform (R-59 F7)")
    want = (mode["mode"], mode["protocol_sha256"] if mode["mode"] == "eval" else None)
    other = _out_modes(out) - {want}
    if other:
        raise ValueError(f"{out} already holds records of another run mode {sorted(other, key=str)}: EVAL and DEV "
                         f"outputs must be separate files (R-35)")
    integrity = {"code_commit": code.get("commit"), "code_dirty": code.get("dirty"),
                 "protocol_sha256": spec.get("protocol_sha256"), "spec_sha256": spec_sha256(spec),
                 "spec_file_sha256": spec_file_sha256, "python": platform.python_version(),
                 "lock_sha256": pins.get("lock_sha256"), "pin_mismatches": pins.get("mismatches"),
                 "torch_build": pins.get("torch_build"), "isolation": "fork+rlimit" if isolate else "in-process",
                 "budget_cpu_s": budget, "cap_factor": EVAL_CAP_FACTOR if ev else 1.0,
                 "cost_table_sha256": None, "part": None, "parts": None, **(stamp or {})}
    done = R._done_keys(out) | _infeasible_keys(out) | set(skip_keys or ())
    methods: dict[str, Any] = {}
    over: dict[tuple, tuple[int, float, str]] = {}                 # DEV: (arm, w, r, lam, kappa) -> (n, cost, kind)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    n_run = 0
    cur, ds, truth, gen = None, None, None, 0.0
    with open(out, "a", encoding="utf-8") as fh:
        for u in units:
            if u.key in done:
                continue
            arm = spec["arms"][u.arm]
            cfg = dict(arm.get("config", {}))
            cpu_b, wall_b = arm_budgets(spec, u.arm, budget)
            limit = (EVAL_CAP_FACTOR * cpu_b if ev else cpu_b) if cpu_b else None
            wlimit = (EVAL_CAP_FACTOR * wall_b if ev else wall_b) if wall_b else None
            fam = (u.arm, u.world, u.regime, u.lam, u.kappa)
            t3 = (arm.get("infeasible_n") or {}).get(str(u.n))
            reg = read_registry(registry).get(u.arm) if registry else None
            load0 = load_info()
            if t3 is not None:
                rec = _infeasible(u, cfg, None, f"infeasible at this n (DEV measured cost {float(t3):.0f} "
                                  f"{'wall-s' if wall_b else 'CPU-s'}, T3)")
            elif not ev and fam in over and u.n >= over[fam][0]:
                n0, c0, kind = over[fam]
                rec = _infeasible(u, cfg, None, f"not run: n {n0} exceeded the budget (measured {c0:.0f} {kind})")
            elif reg is not None and u.n >= reg[0]:
                rec = _infeasible(u, cfg, None, f"not run: arm infeasible at n {reg[0]} in this session "
                                  f"(measured {reg[1]:.0f} {reg[2]}; T3 per (arm, n))")
            else:
                if cur != u.dataset:
                    t0 = time.process_time()
                    ds, truth = generate_dataset(u.world, u.regime, u.n, u.seed,
                                                 lam=1.0 if u.lam is None else u.lam, kappa=u.kappa)
                    gen, cur = time.process_time() - t0, u.dataset
                if u.arm not in methods:
                    try:
                        methods[u.arm] = R.load_method(arm["ref"])
                    except Exception:                          # e.g. a missing dependency: error records, go on
                        import traceback
                        methods[u.arm] = traceback.format_exc()
                if isinstance(methods[u.arm], str):
                    rec = _failed(u, cfg, None, f"method load failed: {methods[u.arm][-1500:]}")
                elif isolate:
                    rec, ccpu, crss, sig, cwall, hit = _run_isolated(methods[u.arm], ds, truth, cfg, gen, u.key,
                                                                     limit, wlimit)
                    if rec is not None:
                        rec.update(child_cpu_s=ccpu, child_maxrss_mb=crss, child_wall_s=cwall)
                    else:
                        why = classify_kill(sig, ccpu, limit, hit) if (sig is not None or hit) else "no_record"
                        if why == "wall_limit":
                            rec = _over(u, cfg, cwall, crss, wall_b, wlimit, ev, "wall-s", cpu=ccpu)
                        elif why == "cpu_limit":
                            rec = _over(u, cfg, ccpu, crss, cpu_b, limit, ev, "CPU-s")
                        else:
                            rec = _failed(u, cfg, ccpu, f"child {'killed by signal ' + str(sig) if sig else 'exited without a record'}"
                                          f" at {ccpu:.0f} CPU-s, maxrss {crss:.0f} MB (out of memory?)", crss)
                else:
                    rec = R.run_one(methods[u.arm], ds, truth, cfg, gen, u.key)
                    if limit and rec.get("cpu_s") is not None and rec["cpu_s"] > limit:
                        rec = _over(u, cfg, rec["cpu_s"], rec.get("peak_rss_mb"), cpu_b, limit, ev, "CPU-s")
                    elif wlimit and rec.get("wall_s") is not None and rec["wall_s"] > wlimit:
                        rec = _over(u, cfg, rec["wall_s"], rec.get("peak_rss_mb"), wall_b, wlimit, ev, "wall-s",
                                    cpu=rec.get("cpu_s"))
                if rec.get("status") == "infeasible" and rec.get("cost") is not None:
                    over[fam] = (u.n, rec["cost"], rec["cost_kind"])
                    if registry:
                        with open(registry, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps({"arm": u.arm, "n": u.n, "cost": rec["cost"],
                                                 "kind": rec["cost_kind"], "key": u.key}) + "\n")
            rec.setdefault("status", "ok" if rec.get("error") is None else "error")
            rec.update(arm=u.arm, role=u.role, code=code, run_mode=mode, integrity=integrity, pkgs=pkgs, host=host,
                       limits={"cpu_s": limit, "wall_s": wlimit},
                       load={"start": load0, "end": load_info()}, campaign=CAMPAIGN_VERSION,
                       spec_name=spec.get("name"))
            fh.write(json.dumps(R._clean(rec), allow_nan=False) + "\n")
            fh.flush()
            n_run += 1
            log(f"[xm-c] {u.key} {rec['status']} cpu {rec.get('cpu_s') or float('nan'):.2f}s")
    return n_run


def _over(u: Unit, cfg: dict, cost: float, rss: float | None, budget: float | None, limit: float, ev: bool,
          kind: str, cpu: float | None = None) -> dict:
    """Record of a unit over its budget, infeasible with the measured cost (``cost`` in ``kind`` units: CPU-s, or
    wall-s for a GPU arm; ``cpu`` = its CPU-s then). EVAL: over the safety cap, never re-run, its cell infeasible
    (PROTOCOL_A s.7, R-59 F2); DEV: over the budget (T3)."""
    if ev:
        rec = _infeasible(u, cfg, cost if kind == "CPU-s" else cpu,
                          f"EVAL safety cap {limit:.0f} {kind} ({EVAL_CAP_FACTOR:g} x budget {budget}) exceeded "
                          f"(measured {cost:.0f}): infeasible, never re-run (PROTOCOL_A s.7)", rss)
    else:
        rec = _infeasible(u, cfg, cost if kind == "CPU-s" else cpu,
                          f"exceeded the budget {budget} {kind} (measured {cost:.0f})", rss)
    rec.update(cost=cost, cost_kind=kind)
    if kind == "wall-s":
        rec["wall_s"] = cost
    return rec


def _infeasible_keys(path: str) -> set[str]:
    out = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "infeasible":
                out.add(r["key"])
    return out


def complete_dataset_keys(units: list[Unit], paths: list[str], accept=None) -> set[str]:
    """Keys of the units whose WHOLE dataset (every arm of it in ``units``) has a final record (ok / infeasible)
    in ``paths``. A relaunch on another platform skips those and re-runs every arm of the incomplete datasets, so
    each dataset's arms come from one platform (R-35). ``accept`` (record -> bool; run_part: the merge's
    ``accept_rule``, R-59 F3) drops records of another run mode / protocol / spec before anything is skipped."""
    if accept is None:
        fin: set[str] = set()
        for p in paths:
            fin |= R._done_keys(p) | _infeasible_keys(p)
    else:
        fin = {r["key"] for p in paths for r in iter_jsonl(p)
               if r.get("key") and _status(r) in ("ok", "infeasible") and accept(r)}
    groups: dict[tuple, list[str]] = defaultdict(list)
    for u in units:
        groups[u.dataset].append(u.key)
    return {k for ks in groups.values() if all(k in fin for k in ks) for k in ks}


def run_part(spec: dict, part: int, parts: int, out: str, budget: float | None = None,
             cost_table: dict | None = None, isolate: bool | None = None, log=print,
             spec_file_sha256: str | None = None, skip_complete_from: list[str] | None = None,
             registry: str | None = None) -> int:
    units = partition(expand(spec), parts, cost_table)[part]
    skip = (complete_dataset_keys(units, skip_complete_from, accept_rule(spec)) if skip_complete_from else None)
    stamp = {"cost_table_sha256": spec_sha256(cost_table) if cost_table else None, "part": part, "parts": parts}
    return run_units(spec, units, out, budget if budget is not None else spec.get("budget_cpu_s"), isolate, log,
                     spec_file_sha256, skip, registry=registry, stamp=stamp)


def accept_rule(spec: dict):
    """The records a run of ``spec`` may use (merge, and --skip-complete-from, R-59 F3): EVAL = EVAL records of
    the spec's protocol AND of this spec (canonical sha); DEV = DEV records (R-35: the two never mix, e.g. on the
    shared tune-seed keys; DEV done-key files carry no stamps)."""
    if is_eval(spec):
        psha, ssha = spec.get("protocol_sha256"), spec_sha256(spec)

        def acc(r):
            rm = r.get("run_mode") or {}
            return (rm.get("mode") == "eval" and rm.get("protocol_sha256") == psha
                    and (r.get("integrity") or {}).get("spec_sha256") == ssha)
        return acc
    return lambda r: (r.get("run_mode") or {}).get("mode", "dev") == "dev"


# ================================================================================================ merge
_RANK = {"ok": 2, "infeasible": 1, "error": 0}


def _status(r: dict) -> str:
    return r.get("status") or ("ok" if r.get("error") is None else "error")


def _dataset_of(r: dict) -> tuple:
    j = r.get("job") or {}
    return (j.get("world"), j.get("regime"), j.get("lam"), j.get("n"), j.get("seed"), j.get("kappa"))


def _platform(r: dict) -> str | None:
    return (r.get("host") or {}).get("platform")


@dataclasses.dataclass
class _Entry:
    """Index entry of one record line (streaming merge: the record itself stays on disk)."""
    path: int
    offset: int
    length: int
    key: str
    status: str
    platform: str | None
    dataset: tuple
    sha: str | None
    dhash: int | None


def _index(paths: list[str], accept, stats: dict):
    """Yield an ``_Entry`` per parseable, accepted record line of ``paths`` (one record in memory at a time)."""
    for pi, path in enumerate(paths):
        off = 0
        with open(path, "rb") as fh:
            for raw in fh:
                start, off = off, off + len(raw)
                if not raw.strip():
                    continue
                stats["lines"] += 1
                try:
                    r = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    stats["bad_lines"] += 1
                    continue
                if not isinstance(r, dict) or r.get("key") is None:
                    stats["bad_lines"] += 1
                    continue
                if accept is not None and not accept(r):
                    stats["foreign"] += 1
                    continue
                dh = hash(tuple(bool(e.get("declared")) for e in r.get("edges", []))) if r.get("edges") else None
                yield _Entry(pi, start, len(raw), r["key"], _status(r), _platform(r), _dataset_of(r),
                             r.get("dataset_sha256"), dh)


def merge_index(paths: list[str], accept=None) -> tuple[dict[str, _Entry], dict]:
    """The merge rules on index entries only (memory: a few hundred bytes per record): one entry per key, ok >
    infeasible > error, later wins within a class; duplicates, conflicts (two ok records of one key with different
    declarations on the same dataset hash) and, per R-35, one platform per dataset where possible (else listed in
    ``mixed_platform_datasets``)."""
    best: dict[str, _Entry] = {}
    alts: dict[str, list[_Entry]] = defaultdict(list)
    stats = {"lines": 0, "bad_lines": 0, "duplicates": 0, "foreign": 0, "conflicts": [],
             "platform_switched_datasets": 0, "mixed_platform_datasets": []}
    for e in _index(paths, accept, stats):
        old = best.get(e.key)
        if old is None:
            best[e.key] = e
            continue
        stats["duplicates"] += 1
        if old.status == e.status == "ok" and old.sha == e.sha and old.dhash != e.dhash:
            stats["conflicts"].append(e.key)
        if _RANK[e.status] >= _RANK[old.status]:
            best[e.key] = e
            alts[e.key].append(old)
        else:
            alts[e.key].append(e)
    by_ds: dict[tuple, list[str]] = defaultdict(list)
    for k, e in best.items():
        by_ds[e.dataset].append(k)
    for dsk, keys in sorted(by_ds.items(), key=lambda t: str(t[0])):
        plats = {best[k].platform for k in keys}
        if len(plats) <= 1:
            continue
        cands = {k: [best[k]] + alts.get(k, []) for k in keys}
        counts = {p: sum(best[k].platform == p for k in keys) for p in plats}
        for p in sorted(plats, key=lambda q: (-counts[q], str(q))):
            pick = {k: next((x for x in cands[k] if x.platform == p and x.status != "error"), None) for k in keys}
            if all(pick.values()):
                best.update(pick)
                stats["platform_switched_datasets"] += 1
                break
        else:
            stats["mixed_platform_datasets"].append("|".join(str(x) for x in dsk))
    return best, stats


def _read_entry(fhs: list, e: _Entry) -> dict:
    fhs[e.path].seek(e.offset)
    return json.loads(fhs[e.path].read(e.length))


def merge_records(paths: list[str], accept=None) -> tuple[dict[str, dict], dict]:
    """``merge_index`` with the chosen records loaded (small inputs, tests)."""
    best, stats = merge_index(paths, accept)
    fhs = [open(p, "rb") for p in paths]
    try:
        return {k: _read_entry(fhs, e) for k, e in best.items()}, stats
    finally:
        for fh in fhs:
            fh.close()


MERGE_SETS = ("python", "lock_sha256", "spec_file_sha256", "isolation", "budget_cpu_s", "cost_table_sha256", "parts")


def merge(spec: dict, inputs: list[str], out: str) -> dict:
    """Merge shard files for ``spec`` into ``out`` (key-sorted; ``.gz`` = gzip), streaming: only the index is held
    in memory. EVAL specs keep only EVAL records of the spec's protocol; DEV specs drop EVAL records (R-35: the
    two never mix, e.g. on the shared tune-seed keys)."""
    import gzip
    ev = is_eval(spec)
    best, stats = merge_index(inputs, accept_rule(spec))
    want = {u.key for u in expand(spec)}
    by: dict[str, list[str]] = defaultdict(list)
    commits, dirty, plats, psha, ssha = set(), 0, set(), set(), set()
    seen = defaultdict(set)                                # R-59 F9 / F4: integrity values over the merged records
    fhs = [open(p, "rb") for p in inputs]
    op = gzip.open if out.endswith(".gz") else open
    try:
        with op(out, "wt", encoding="utf-8", newline="\n") as fo:
            for k in sorted(best):
                r = _read_entry(fhs, best[k])
                fo.write(json.dumps(r, allow_nan=False) + "\n")
                by[_status(r)].append(k)
                commits.add(str((r.get("code") or {}).get("commit")))
                dirty += (r.get("code") or {}).get("dirty") is not False
                plats.add(str(_platform(r)))
                psha.add(str((r.get("integrity") or {}).get("protocol_sha256")))
                ssha.add(str((r.get("integrity") or {}).get("spec_sha256")))
                for f in MERGE_SETS:
                    seen[f].add(str((r.get("integrity") or {}).get(f)))
    finally:
        for fh in fhs:
            fh.close()
    summary = {"mode": "eval" if ev else "dev", "n_expected": len(want), "n_records": len(best),
               "n_ok": len(by["ok"]), "missing": sorted(want - set(best)), "errors": sorted(by["error"]),
               "infeasible": sorted(by["infeasible"]), "unexpected": sorted(set(best) - want), **stats,
               "commits": sorted(commits), "dirty_records": dirty, "platforms": sorted(plats),
               "protocol_sha256": sorted(psha), "spec_sha256": sorted(ssha),
               **{f: sorted(seen[f]) for f in MERGE_SETS}}
    sp = (out[:-3] if out.endswith(".gz") else out) + ".summary.json"
    json.dump(summary, open(sp, "w", encoding="utf-8"), indent=1)
    mixed = [f for f in ("cost_table_sha256", "parts") if len(seen[f]) > 1]
    if ev and mixed:                                       # R-59 F4: one partition per EVAL run
        os.remove(out)
        raise ValueError(f"merge refused: EVAL records mix {', '.join(f'{f} {sorted(seen[f])}' for f in mixed)} "
                         f"(one cost table and part count per EVAL run; summary kept at {sp})")
    return summary


def iter_jsonl(path: str):
    """Records of a (possibly gzipped) JSONL file, one at a time; broken lines skipped."""
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_jsonl(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ================================================================================================ aggregate
def record_unit(r: dict, spec: dict) -> Unit:
    j = r["job"]
    return Unit(j["world"], j["regime"], None if j.get("lam") is None else float(j["lam"]), int(j["n"]),
                float(j.get("kappa") or 0.0), int(j["seed"]), r.get("arm") or r["key"].split("|")[0],
                r.get("role") or "measure")


def to_result(r: dict) -> api.Result:
    edges = tuple(api.EdgeResult(e["source"], e["target"], math.nan if e.get("score") is None else float(e["score"]),
                                 e.get("p"), int(e.get("sign") or 0), bool(e.get("declared")))
                  for e in r.get("edges", []))
    return api.Result(r.get("method") or "?", r.get("version") or "?", edges, float(r.get("method_cpu_s") or 0.0),
                      dict(r.get("config") or {}), dict(r.get("notes") or {}))


def cluster_ci(hits: np.ndarray, cnt: np.ndarray, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> list:
    """Seed-cluster bootstrap 95 % percentile CI of the ratio sum(hits) / sum(cnt) (seeds = clusters)."""
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    keep = cnt > 0
    hits, cnt = hits[keep], cnt[keep]
    if not len(cnt):
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cnt), (reps, len(cnt)))
    bs = hits[idx].sum(1) / cnt[idx].sum(1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def wilson_deff(hits: np.ndarray, cnt: np.ndarray, z: float = 1.959964) -> list:
    """Wilson 95 % interval on the design-effect-adjusted sample size (cluster = seed); defined at 0 hits."""
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    N = cnt.sum()
    if N <= 0:
        return [None, None]
    p = hits.sum() / N
    k = len(cnt)
    if 0 < p < 1 and k > 1:
        resid = hits - p * cnt
        v_clu = k / (k - 1) * float(np.sum(resid ** 2)) / N ** 2
        deff = max(1.0, v_clu / (p * (1 - p) / N))
    else:
        deff = 1.0
    ne = N / deff
    c = z * z / ne
    mid = (p + c / 2) / (1 + c)
    half = z * math.sqrt(p * (1 - p) / ne + c / (4 * ne)) / (1 + c)
    return [max(0.0, mid - half), min(1.0, mid + half)]


def _rate(hits, cnt) -> dict:
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    N = float(cnt.sum())
    rate = float(hits.sum() / N) if N else None
    ci = cluster_ci(hits, cnt)
    cw = wilson_deff(hits, cnt)
    from cdd_oran.xmethod.dev_power import design_effect
    return {"rate": rate, "hits": int(hits.sum()), "n": int(N), "n_seeds": int(np.sum(cnt > 0)), "ci": ci,
            "ci_wilson_deff": cw, "deff": design_effect(hits[cnt > 0], cnt[cnt > 0]),
            "above": bool(ci[0] is not None and ci[0] > ALPHA)}


def _is_primary(src: str) -> bool:
    return not src.startswith("K") and src not in (PLACEBO, PLACEBO_CONF)


def seed_row(r: dict, decl: api.Result, truth: api.Truth) -> dict:
    """Per-seed validity / power numbers of one measurement record under the arm's declaration ``decl``.
    Rate denominators (declaration and raw p) count the returned candidates of each kind MINUS the not-testable
    ones (R-22 / R-23; same rule as scratchpad/xmethod/eval_analysis.py, PROTOCOL_A section 8); raw-p rates also
    need a p-value."""
    sc = score(decl, truth)
    nt = set(sc.get("not_testable_edges", []))
    declared = set(sc["declared_edges"])
    nulls = {e for e in truth.null_edges if _is_primary(e[0])}
    cnt = {k: [0, 0] for k in ("null", "plac", "conf")}
    raw = {k: [0, 0] for k in ("null", "plac", "conf")}
    for e in r.get("edges", []):
        s, t, p = e["source"], e["target"], e.get("p")
        name = f"{s}->{t}"
        kind = ("plac" if s == PLACEBO else "conf" if s == PLACEBO_CONF
                else "null" if (s, t) in nulls else None)
        if kind is None or name in nt:
            continue
        cnt[kind][1] += 1
        cnt[kind][0] += int(name in declared)
        if p is not None:
            raw[kind][1] += 1
            raw[kind][0] += int(float(p) <= ALPHA)
    return {"seed": int(r["job"]["seed"]), "recall": sc["recall"], "fdp": sc["fdp"], "precision": sc["precision"],
            "sign_acc": sc["sign_acc"], "sign_n": sc["sign_n"], "n_declared": sc["n_declared"],
            "tp": sc["tp"], "n_true": sc["n_true"], "null_fp": cnt["null"][0], "n_null": cnt["null"][1],
            "raw_null": raw["null"], "plac_decl": cnt["plac"][0], "n_plac": cnt["plac"][1], "raw_plac": raw["plac"],
            "conf_decl": cnt["conf"][0], "n_conf": cnt["conf"][1], "raw_conf": raw["conf"],
            "nt_true": sc["n_not_testable_true"], "nt_null": sc["n_not_testable_null"],
            "declared": sc["declared_edges"], "cpu_s": r.get("cpu_s"), "peak_rss_mb": r.get("peak_rss_mb")}


def _mean(v) -> float | None:
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else None


def summarise_cell(rows: list[dict], has_p: bool) -> dict:
    out = {"n_seeds": len(rows),
           "null_decl": _rate([x["null_fp"] for x in rows], [x["n_null"] for x in rows]),
           "plac_decl": _rate([x["plac_decl"] for x in rows], [x["n_plac"] for x in rows])}
    if any(x["n_conf"] for x in rows):
        out["conf_decl"] = _rate([x["conf_decl"] for x in rows], [x["n_conf"] for x in rows])
    if has_p:
        out["null_raw05"] = _rate([x["raw_null"][0] for x in rows], [x["raw_null"][1] for x in rows])
        out["plac_raw05"] = _rate([x["raw_plac"][0] for x in rows], [x["raw_plac"][1] for x in rows])
        if any(x["n_conf"] for x in rows):
            out["conf_raw05"] = _rate([x["raw_conf"][0] for x in rows], [x["raw_conf"][1] for x in rows])
    keys = ("null_raw05", "plac_raw05") if has_p else ("null_decl", "plac_decl")
    if len(rows) < MIN_FLAG_SEEDS:
        out["valid_flag"] = "few_seeds"
    else:
        out["valid_flag"] = "ABOVE" if any(out[k]["above"] for k in (*keys, "null_decl", "plac_decl")) else "ok"
    rec = [x["recall"] for x in rows]
    out.update(recall=_mean(rec), recall_sd=float(np.nanstd([x for x in rec if x is not None], ddof=1))
               if sum(x is not None for x in rec) > 1 else None,
               fdp=_mean([x["fdp"] for x in rows]), sign_acc=_mean([x["sign_acc"] for x in rows]),
               n_declared=_mean([x["n_declared"] for x in rows]),
               nt_true=int(sum(x["nt_true"] for x in rows)), nt_null=int(sum(x["nt_null"] for x in rows)),
               per_seed={str(x["seed"]): {"recall": x["recall"], "fdp": x["fdp"], "sign_acc": x["sign_acc"]}
                         for x in rows})
    return out


def aggregate(records, spec: dict, stream_sorted: bool = False) -> dict:
    """Validity / power / cost per cell (module docstring). ``records`` = merged records (one per key), any
    iterable. ``stream_sorted``: the records come sorted by key (``merge`` output), so each cell's records are
    contiguous and a cell is finished as soon as the next one starts (memory: one cell)."""
    arms = spec["arms"]
    truths: dict[tuple, api.Truth] = {}
    cells, cost = {}, defaultdict(lambda: {"cpu": [], "wall": [], "rss": [], "gen": [], "infeasible": [], "errors": 0})

    def new_group():
        return {"tune": [], "measure": [], "infeasible": [], "error": []}

    def finish(c, u, g):
        mode = arms[u.arm]["declare"] if u.arm in arms else "by"
        truth = truths.setdefault((u.world, u.regime), truth_for(u.world, u.regime))
        tau = placebo_tau([to_result(r) for r in g["tune"]]) if g["tune"] else None
        rows, rows_tau = [], []
        for r in g["measure"]:
            res = to_result(r)
            if mode == "by":
                rows.append(seed_row(r, res, truth))
            if tau is not None:
                rows_tau.append(seed_row(r, apply_threshold(res, tau), truth))
        has_p = mode == "by"
        entry = {"arm": u.arm, "world": u.world, "regime": u.regime, "lam": u.lam, "n": u.n, "kappa": u.kappa,
                 "declare": mode, "tau": tau, "n_tune": len(g["tune"]), "n_measure": len(g["measure"]),
                 "n_infeasible": len(g["infeasible"]), "n_error": len(g["error"])}
        primary = rows if mode == "by" else rows_tau
        if primary:
            entry["primary"] = summarise_cell(primary, has_p)
        elif mode == "tau" and g["measure"] and tau is None:
            entry["primary"] = None
            entry["note"] = "untuned: no tune-seed records for this cell (tau undefined)"
        if mode == "by" and rows_tau:
            entry["secondary_tau"] = summarise_cell(rows_tau, False)
        cells[c] = entry
        ck = f"{u.arm}|{u.world}|{u.regime}|n{u.n}"
        for r in g["tune"] + g["measure"]:
            cost[ck]["cpu"].append(r.get("cpu_s"))
            cost[ck]["wall"].append(r.get("child_wall_s") or r.get("wall_s"))
            cost[ck]["rss"].append(r.get("peak_rss_mb"))
            cost[ck]["gen"].append(r.get("gen_cpu_s"))
        for r in g["infeasible"]:
            cost[ck]["infeasible"].append({"seed": r["job"]["seed"], "cpu_s": r.get("cpu_s"),
                                           "cost": r.get("cost"), "cost_kind": r.get("cost_kind"),
                                           "measured": r.get("cost") is not None, "reason": r.get("reason")})
        cost[ck]["errors"] += len(g["error"])
    if stream_sorted:
        done: set[str] = set()
        cur, cu, cg = None, None, None
        for r in records:
            u = record_unit(r, spec)
            if u.cell != cur:
                if cur is not None:
                    finish(cur, cu, cg)
                    done.add(cur)
                if u.cell in done:
                    raise ValueError(f"aggregate(stream_sorted=True): input not sorted by cell ({u.cell} again)")
                cur, cu, cg = u.cell, u, new_group()
            st = _status(r)
            cg[u.role if st == "ok" else st].append(r)
        if cur is not None:
            finish(cur, cu, cg)
    else:
        by_cell: dict[str, dict[str, list]] = defaultdict(new_group)
        units = {}
        for r in records:
            u = record_unit(r, spec)
            units[u.cell] = u
            st = _status(r)
            by_cell[u.cell][u.role if st == "ok" else st].append(r)
        for c in sorted(by_cell):
            finish(c, units[c], by_cell[c])
    cost_out = {}
    for k, v in sorted(cost.items()):
        cpu = [x for x in v["cpu"] if x is not None]
        wall = [x for x in v["wall"] if x is not None]
        cost_out[k] = {"n_runs": len(cpu), "cpu_s_mean": _mean(cpu), "cpu_s_max": max(cpu) if cpu else None,
                       "wall_s_mean": _mean(wall), "wall_s_max": max(wall) if wall else None,
                       "peak_rss_mb_max": max([x for x in v["rss"] if x is not None], default=None),
                       "gen_cpu_s_mean": _mean(v["gen"]), "infeasible": v["infeasible"], "errors": v["errors"]}
    return {"spec_name": spec.get("name"), "campaign": CAMPAIGN_VERSION, "alpha": ALPHA,
            "budget_cpu_s": spec.get("budget_cpu_s"), "cells": cells, "cost": cost_out}


# ================================================================================================ projection
def cost_table_from_agg(agg: dict, wall_arms: set[str] | frozenset = frozenset()) -> dict:
    """(arm|world|regime|n) -> mean cost per dataset: CPU-s, or wall-s for the GPU arms in ``wall_arms`` (R-41)."""
    out = {}
    for k, v in agg["cost"].items():
        x = v.get("wall_s_mean") if k.split("|", 1)[0] in wall_arms else v["cpu_s_mean"]
        if x is not None:
            out[k] = x
    return out


def measured_infeasible(agg: dict) -> dict[str, tuple[int, float, str]]:
    """arm -> (smallest n with a MEASURED over-budget unit, its cost, kind) from an aggregate's cost table."""
    out: dict[str, tuple[int, float, str]] = {}
    for k, v in agg["cost"].items():
        arm, n = k.split("|", 1)[0], int(k.rsplit("|n", 1)[1])
        for i in v.get("infeasible", []):
            c = i.get("cost") if i.get("cost") is not None else (i.get("cpu_s") if i.get("measured", True) else None)
            if c is not None and (arm not in out or n < out[arm][0]):
                out[arm] = (n, float(c), i.get("cost_kind") or "CPU-s")
    return out


def _fit_cost(table: dict, arm: str, world: str, regime: str, n: int) -> tuple[float | None, str]:
    """Measured mean if present, else a power-law fit in n over the arm's measured n in (world, regime), else over
    all worlds of the arm (scaled), else None."""
    k = f"{arm}|{world}|{regime}|n{n}"
    if k in table:
        return table[k], "measured"
    pts = [(int(kk.rsplit("|n", 1)[1]), v) for kk, v in table.items()
           if kk.startswith(f"{arm}|{world}|{regime}|n") and v and v > 0]
    if len(pts) < 2:
        pts = [(int(kk.rsplit("|n", 1)[1]), v) for kk, v in table.items() if kk.startswith(f"{arm}|") and v and v > 0]
    if len(pts) < 2:
        return None, "unknown"
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    b, a = np.polyfit(x, y, 1)
    return float(math.exp(a + b * math.log(n))), f"extrapolated (n^{b:.2f})"


def project(agg: dict, full: dict) -> dict:
    """Projected cost of ``full`` from a pilot aggregate: CPU-h for CPU arms, GPU wall-h for arms with a wall budget
    (R-41), per arm / n / role. Units at n >= an arm's MEASURED infeasible n are counted apart (not run, T3);
    extrapolations over the budget are capped at the budget and counted."""
    gpu_arms = {a for a, d in full["arms"].items() if d.get("budget_wall_s")}
    table = cost_table_from_agg(agg, gpu_arms)
    inf = measured_infeasible(agg)
    per_arm: dict[str, float] = defaultdict(float)
    per_n: dict[int, float] = defaultdict(float)
    per_role: dict[str, float] = defaultdict(float)
    gen = defaultdict(float)
    unknown, capped, infeasible = set(), defaultdict(int), defaultdict(int)
    how_used: dict[str, set] = defaultdict(set)
    seen_ds = set()
    units = expand(full)
    for u in units:
        if u.arm in inf and u.n >= inf[u.arm][0]:
            infeasible[f"{u.arm}|n{u.n}"] += 1
            continue
        c, how = _fit_cost(table, u.arm, u.world, u.regime, u.n)
        if c is None:
            unknown.add(u.arm)
            continue
        how_used[u.arm].add(how.split(" ")[0])
        b = arm_budgets(full, u.arm, full.get("budget_cpu_s"))
        cap = (b[1] if u.arm in gpu_arms else b[0]) or math.inf
        if c > cap:
            capped[f"{u.arm}|n{u.n}"] += 1
            c = cap
        per_arm[u.arm] += c
        if u.arm not in gpu_arms:
            per_n[u.n] += c
            per_role[u.role] += c
        if u.dataset not in seen_ds:
            seen_ds.add(u.dataset)
            g = [v["gen_cpu_s_mean"] for k, v in agg["cost"].items()
                 if k.endswith(f"|{u.world}|{u.regime}|n{u.n}") and v["gen_cpu_s_mean"]]
            gen["total"] += float(np.mean(g)) if g else 0.0
    cpu = sum(v for a, v in per_arm.items() if a not in gpu_arms) + gen["total"]
    return {"n_units": len(units), "n_datasets": len(seen_ds), "cpu_h_total": cpu / 3600,
            "gpu_wall_h_total": sum(v for a, v in per_arm.items() if a in gpu_arms) / 3600,
            "cpu_h_by_arm": {k: v / 3600 for k, v in sorted(per_arm.items()) if k not in gpu_arms},
            "gpu_wall_h_by_arm": {k: v / 3600 for k, v in sorted(per_arm.items()) if k in gpu_arms},
            "cpu_h_by_n": {str(k): v / 3600 for k, v in sorted(per_n.items())},
            "cpu_h_by_role": {k: v / 3600 for k, v in per_role.items()},
            "cpu_h_generation": gen["total"] / 3600, "arms_without_cost": sorted(unknown),
            "cost_basis_by_arm": {k: sorted(v) for k, v in sorted(how_used.items())},
            "measured_infeasible": {a: {"n": v[0], "cost": v[1], "kind": v[2]} for a, v in sorted(inf.items())},
            "units_infeasible_not_run": dict(infeasible), "units_capped_at_budget": dict(capped)}


# ================================================================================================ cloud
CITEST_REFS = ("mscr", "pcorr", "pdcor", "rcot2", "cmi_knn")


def _cloud_cmd(spec_rel: str, part_ids: list[int], parts: int, out_dir: str, platform_tag: str,
               cost_rel: str | None, wall_s: int, code: dict, prefix: str = "res", registry: bool = False,
               gpu: bool = False, skip_from: list[str] | None = None, one_per_cpu: bool = False) -> str:
    """P single-threaded processes, each under ``timeout wall_s`` so the session ends (and saves its outputs)
    before the platform's own limit; a cut part resumes on relaunch (records are appended per unit). The
    launcher's git check (commit, dirty) is passed as XM_CODE_COMMIT / XM_CODE_DIRTY (R-35). ``registry`` (DEV):
    the processes share {out_dir}/registry.jsonl (an arm over budget at n is not run at n' >= n). ``gpu``: process
    i sees GPU i mod (number of GPUs), and CUDA is not initialised before the fork (NVML device check).
    ``skip_from``: bundled shard files (globs) of an earlier session; datasets complete there are skipped (R-35).
    ``one_per_cpu`` (EVAL, R-55): at most NP = cpu_quota() processes at once (bash job semaphore); further parts
    start as earlier ones end, so a session should hold about NP parts of its wall budget."""
    thr = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    env = (f"XM_PLATFORM={platform_tag} XM_CODE_COMMIT={code['commit']} XM_CODE_DIRTY={int(bool(code['dirty']))} "
           f"XM_LOCK_FILE={LOCK_REL} {thr}")
    ct = f" --cost-table {cost_rel}" if cost_rel else ""
    reg = f" --registry {out_dir}/registry.jsonl" if registry else ""
    reg += (" --skip-complete-from " + " ".join(shlex.quote(x) for x in skip_from)) if skip_from else ""
    pre = ("NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ \"$NG\" -ge 1 ] || NG=1; nvidia-smi -L > "
           f"{out_dir}/gpus.txt 2>&1; " if gpu else "")
    gpu_env = (lambda i: f"PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=$(( {i} % NG )) ") if gpu else (
        lambda i: "")
    gate = "while [ $(jobs -rp | wc -l) -ge $NP ]; do wait -n; done; " if one_per_cpu else ""
    if one_per_cpu:
        pre += ('NP=$(python -c "from cdd_oran.xmethod.campaign import cpu_quota; print(cpu_quota())"); '
                f'echo "vCPU quota $NP" > {out_dir}/cpu_quota.txt; ')
    runs = " ".join(f"{gate}( {gpu_env(i)}{env} timeout {wall_s} python -u -m cdd_oran.xmethod.campaign run --spec "
                    f"{spec_rel} --part {i}/{parts} --out {out_dir}/{prefix}_{i}.jsonl{ct}{reg} > {out_dir}/log_{i}.txt "
                    f"2>&1; echo $? > {out_dir}/rc_{i}.txt ) &" for i in part_ids)
    # R-59 F5: each part's exit status in rc_<i>.txt (robust to the gate's `wait -n` reaping); the command's own
    # status (the session's / vps_run's exit_code) is non-zero if any part failed or left no status
    ids = " ".join(map(str, part_ids))
    check = (f"RC=0; for i in {ids}; do r=$(cat {out_dir}/rc_$i.txt 2>/dev/null || echo none); "
             f'[ "$r" = 0 ] || {{ echo "[xm-c] part $i exit $r"; RC=1; }}; done; [ $RC -eq 0 ]')
    return f"mkdir -p {out_dir} && {pre}{runs} wait; tail -n 3 {out_dir}/log_*.txt; {check}"


def _setup_cmd(out_dir: str, selftest: bool, torch_version: str | None, venv_python: str | None = None,
               gpu: bool = False) -> str:
    """Install every package of the bundled uv.lock export (hashes, --no-deps: the export lists every dependency;
    project deps + all non-dev groups, so the citests group when present) EXCEPT the torch build (R-16 / R-35 +
    amendment): the image's torch is kept when its public version equals the lock's, else the CPU build of that
    version is installed. ``venv_python`` (EVAL, Q7): first ``uv python install`` that version and activate a fresh
    venv (/tmp/xm_venv) for the rest of the session, so every later ``python`` is that interpreter. Then the pin
    check (pin_check.json); optionally the campaign tests (the fork / RLIMIT_CPU isolation test is Linux-only)."""
    log = f"{out_dir}/install.log"
    test = (f"python -m pytest -q -p no:cacheprovider tests/test_xmethod_campaign.py > {out_dir}/selftest.log 2>&1; "
            if selftest else "")
    check = ("python -c \"import json; from cdd_oran.xmethod.campaign import pin_check; "
             f"print(json.dumps(pin_check('{LOCK_REL}'"
             + (f", python='{EVAL_PYTHON}'" if venv_python == EVAL_PYTHON else "") + ")))\"")
    if venv_python:
        pre = (f"(command -v uv || pip install uv) > {log} 2>&1; uv python install {venv_python} >> {log} 2>&1; "
               f"uv venv --python {venv_python} /tmp/xm_venv >> {log} 2>&1; source /tmp/xm_venv/bin/activate; "
               f"unset UV_SYSTEM_PYTHON; python --version >> {log} 2>&1; ")   # Kaggle sets UV_SYSTEM_PYTHON
        pip = "uv pip install --python /tmp/xm_venv/bin/python"              # never the image's interpreter
        lock = f"{pip} --require-hashes --no-deps -r {LOCK_INSTALL_REL} >> {log} 2>&1; "
    else:
        pre = f": > {log}; "
        lock = (f"(command -v uv && uv pip install --system --require-hashes --no-deps -r {LOCK_INSTALL_REL} || "
                f"pip install --require-hashes --no-deps -r {LOCK_INSTALL_REL}) >> {log} 2>&1; ")
        pip = "pip install"
    torch = ""
    if torch_version:
        fallback = (f"{pip} torch=={torch_version} --index-url https://download.pytorch.org/whl/cu128" if gpu else
                    f"{pip} --no-deps torch=={torch_version} --index-url https://download.pytorch.org/whl/cpu")
        torch = (f"(python -c \"import importlib.metadata as m, sys; from packaging.version import Version; "
                 f"sys.exit(Version(m.version('torch')).public != '{torch_version}')\" || {fallback}) "
                 f">> {log} 2>&1; ")
    return (f"mkdir -p {out_dir}; {pre}{lock}{torch}{pip} pytest >> {log} 2>&1; "
            f"{check} > {out_dir}/pin_check.json 2>&1; nproc >> {log}; {test}")


def launch_checks(spec: dict, code: dict) -> list[str]:
    """Refusals before any cloud launch (R-35). EVAL: authorised protocol and a clean tree. Any mode: CI-test arms
    need the citests dependency group in pyproject (installed from the lock)."""
    errs = []
    groups, _ = lock_groups_and_indexes()
    if any(any(f"methods.{m}:" in d["ref"] for m in CITEST_REFS) for d in spec["arms"].values()) \
            and "citests" not in groups:
        errs.append("spec has CI-test arms but pyproject has no 'citests' dependency group")
    if is_eval(spec):
        ok, why = eval_authorised(spec)
        if not ok:
            errs.append(f"EVAL refused: {why}")
        if not code.get("commit") or code.get("dirty") is not False:
            errs.append(f"EVAL refused: tree not clean at launch ({code.get('dirty_paths')})")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cdd_oran.xmethod.campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("list", "run", "merge", "aggregate", "project", "kaggle", "colab", "lightning", "vps", "lock"):
        p = sub.add_parser(c)
        p.add_argument("--spec", required=c != "lock")
        if c == "run":
            p.add_argument("--part", default="0/1")
            p.add_argument("--out", required=True)
            p.add_argument("--budget", type=float, default=None)
            p.add_argument("--cost-table", default=None)
            p.add_argument("--no-isolate", action="store_true")
            p.add_argument("--registry", default=None, help="shared infeasibility registry file (DEV)")
            p.add_argument("--skip-complete-from", nargs="*", default=None,
                           help="shard files of an earlier session: skip datasets complete there, re-run every arm "
                                "of the others (one platform per dataset)")
        if c == "merge":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--out", required=True)
        if c == "aggregate":
            p.add_argument("--merged", required=True)
            p.add_argument("--out", required=True)
        if c == "project":
            p.add_argument("--agg", required=True)
            p.add_argument("--out", required=True)
        if c in ("list", "kaggle", "colab", "lightning", "vps"):
            p.add_argument("--parts", type=int, default=4)
            p.add_argument("--cost-table", default=None)
        if c in ("kaggle", "colab", "lightning", "vps"):
            p.add_argument("--name", required=True)
            p.add_argument("--part-set", default=None, help="'i,j,...' subset of the parts in this session")
            p.add_argument("--paths", nargs="*", default=[])
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--wall-s", type=int, default=None,
                           help="per-process timeout (default kaggle 11 h, colab 3 h, lightning 23 h)")
            p.add_argument("--selftest", action="store_true", help="run tests/test_xmethod_campaign.py first")
            p.add_argument("--venv-python", default=None,
                           help="run in a fresh uv venv of this Python (EVAL always uses EVAL_PYTHON, Q7)")
            p.add_argument("--gpu", action="store_true", help="GPU session (R-41 arms); torch CUDA build")
            p.add_argument("--skip-complete-from", nargs="*", default=None,
                           help="repo paths / globs of an earlier session's shards (bundled): skip its complete datasets")
            if c == "kaggle":
                p.add_argument("--pin", choices=("match", "off"), default="match")
            if c == "vps":
                p.add_argument("--procs", type=int, default=VPS_MAX_PROCS, help="processes (<= 7, systemd cap)")
            if c == "lightning":
                p.add_argument("--machine", default="CPU", help="lightning_sdk Machine name (CPU = 4 vCPU)")
                p.add_argument("--min-balance", type=float, default=None,
                               help="the studio's tick stops it (after a final pull) below this credit balance")
    a = ap.parse_args(argv)
    if a.cmd == "lock":
        print(f"[xm-c] uv.lock export -> {write_lock()}")
        return 0
    spec_bytes = open(a.spec, "rb").read()
    spec = json.loads(spec_bytes)
    ct = json.load(open(a.cost_table, encoding="utf-8")) if getattr(a, "cost_table", None) else None
    if ct is not None and "cost" in ct:
        ct = cost_table_from_agg(ct)
    if a.cmd == "list":
        units = expand(spec)
        parts = partition(units, a.parts, ct)
        print(json.dumps({"mode": "eval" if is_eval(spec) else "dev", "units": len(units),
                          "datasets": len({u.dataset for u in units}),
                          "by_role": {r: sum(u.role == r for u in units) for r in ROLES},
                          "by_arm": {k: sum(u.arm == k for u in units) for k in spec["arms"]},
                          "part_est_cpu_s": [round(sum(unit_cost(u, ct) for u in p)) for p in parts]}))
    elif a.cmd == "run":
        i, p = (int(x) for x in a.part.split("/"))
        skip = (sorted({f for pat in a.skip_complete_from for f in glob.glob(pat, recursive=True)})
                if a.skip_complete_from else None)
        n = run_part(spec, i, p, a.out, a.budget, ct, False if a.no_isolate else None,
                     spec_file_sha256=_sha_lf(spec_bytes), skip_complete_from=skip, registry=a.registry)
        print(f"[xm-c] part {i}/{p}: {n} units run")
    elif a.cmd == "merge":
        files = sorted({f for pat in a.inputs for f in glob.glob(pat, recursive=True)})
        s = merge(spec, files, a.out)
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()}))
    elif a.cmd == "aggregate":
        agg = aggregate(iter_jsonl(a.merged), spec, stream_sorted=True)
        json.dump(R._clean(agg), open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"[xm-c] {len(agg['cells'])} cells -> {a.out}")
    elif a.cmd == "project":
        pr = project(json.load(open(a.agg, encoding="utf-8")), spec)
        json.dump(pr, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps({k: v for k, v in pr.items() if k != "units_capped_at_budget"}, indent=1))
    else:
        code = code_info()
        errs = launch_checks(spec, code)
        if errs:
            raise SystemExit("[xm-c] launch refused:\n  " + "\n  ".join(errs))
        ev = is_eval(spec)
        spec_rel = os.path.relpath(os.path.abspath(a.spec), R.ROOT).replace("\\", "/")
        ids = [int(x) for x in a.part_set.split(",")] if a.part_set else list(range(a.parts))
        cost_rel = (os.path.relpath(os.path.abspath(a.cost_table), R.ROOT).replace("\\", "/")
                    if a.cost_table else None)
        write_lock()
        bundle = [LOCK_REL, LOCK_INSTALL_REL, *BUNDLE_DATA]
        torch_v = lock_pins(LOCK_PATH).get("torch")
        torch_v = torch_v.split("+")[0] if torch_v else None
        if ev:                                       # protocol + the verified frozen text for the cloud-side guard
            frozen = (_git_blob(str(spec["freeze_commit"]), PROTOCOL_REL) if spec.get("freeze_commit") else None)
            os.makedirs(os.path.dirname(FROZEN_COPY_PATH), exist_ok=True)
            open(FROZEN_COPY_PATH, "wb").write(frozen if frozen is not None else open(PROTOCOL_PATH, "rb").read())
            bundle += [PROTOCOL_REL, FROZEN_COPY_REL]
        skip_from = [x.replace("\\", "/") for x in a.skip_complete_from] if a.skip_complete_from else None
        extra = list(a.paths) + ([cost_rel] if cost_rel else []) + bundle + (
            ["tests/test_xmethod_campaign.py", PROTOCOL_REL, "scratchpad/xmethod/eval_analysis.py"]
            if a.selftest else [])                  # the self-test reads the protocol and eval_analysis.py

        if a.cmd == "kaggle":
            out_dir = "$JOB_OUT/eval" if ev else "$JOB_OUT"
            cmd = _setup_cmd(out_dir, a.selftest, torch_v, EVAL_PYTHON if ev else a.venv_python, a.gpu) + _cloud_cmd(
                spec_rel, ids, a.parts, out_dir, "kaggle", cost_rel, a.wall_s or 11 * 3600, code,
                "eval_res" if ev else "res", registry=not ev, gpu=a.gpu, skip_from=skip_from,
                one_per_cpu=ev)
            argv2 = [sys.executable, os.path.join(R.ROOT, "scratchpad", "e6_dev", "kaggle_job.py"), "launch",
                     a.name, "--cmd", cmd, "--paths", spec_rel, *extra, "--internet", "--pin", a.pin,
                     *(["--gpu"] if a.gpu else [])]
        elif a.cmd == "colab":
            out_dir = "xm_eval_out" if ev else "xm_out"
            cmd = _setup_cmd(out_dir, a.selftest, torch_v, EVAL_PYTHON if ev else a.venv_python, a.gpu) + _cloud_cmd(
                spec_rel, ids, a.parts, out_dir, "colab", cost_rel, a.wall_s or 3 * 3600, code,
                "eval_res" if ev else "res", registry=not ev, gpu=a.gpu, skip_from=skip_from,
                one_per_cpu=ev)
            argv2 = [sys.executable, os.path.join(R.ROOT, "scratchpad", "e6_dev", "colab_run.py"), "job", a.name,
                     "--paths", "cdd_oran", spec_rel, *extra, "--cmd", cmd, "--out-dir", out_dir, "--threads", "1"]
        elif a.cmd == "vps":                         # user VPS (xm-citests' vps_run.py, CLI v1): push, venv, launch
            if not 1 <= a.procs <= VPS_MAX_PROCS:
                raise SystemExit(f"--procs must be 1..{VPS_MAX_PROCS}")
            if len(ids) > a.procs and not ev:
                raise SystemExit(f"{len(ids)} parts > --procs {a.procs}: one process per part on the VPS")
            cmd = _cloud_cmd(spec_rel, ids, a.parts, "{OUT}", "vps", cost_rel, a.wall_s or 23 * 3600, code,
                             "eval_res" if ev else "res", registry=not ev, skip_from=skip_from, one_per_cpu=ev)
            vr = [sys.executable, os.path.join(R.ROOT, "scratchpad", "e6_dev", "vps_run.py")]
            tracked = ["cdd_oran", *BUNDLE_DATA, spec_rel, *([cost_rel] if cost_rel else []),
                       *(p for p in a.paths if not p.startswith("scratchpad/e6_dev/runs")),
                       *([PROTOCOL_REL] if ev else [])]
            untracked = sorted({os.path.dirname(g) for g in (skip_from or [])} |
                               {p for p in a.paths if p.startswith("scratchpad/e6_dev/runs")})
            adds = [f"{LOCK_PATH}={LOCK_REL}", f"{LOCK_INSTALL_PATH}={LOCK_INSTALL_REL}",
                    *([f"{FROZEN_COPY_PATH}={FROZEN_COPY_REL}"] if ev else [])]
            pin = ("-c \"import json; from cdd_oran.xmethod.campaign import pin_check; "
                   f"print(json.dumps(pin_check('{LOCK_REL}', python='{EVAL_PYTHON}')))\"")
            steps = [vr + ["push", a.name, "--paths", *tracked, *(["--extra", *untracked] if untracked else []),
                           "--add", *adds],
                     vr + ["venv", a.name, "--lock-install", LOCK_INSTALL_REL, "--lock", LOCK_REL, "--pincheck", pin],
                     vr + ["launch", a.name, "--cmd", cmd, "--procs", str(min(a.procs, len(ids)))]]
            for st_ in steps:
                print(" ".join(shlex.quote(x) for x in st_))
                if not a.dry_run:
                    subprocess.run(st_, cwd=R.ROOT, check=True)
            return 0
        else:                                        # Lightning studio (R-46): same bundle + commands, own runner
            out_dir = "xm_eval_out" if ev else "xm_out"
            cmd = _setup_cmd(out_dir, a.selftest, torch_v, EVAL_PYTHON if ev else (a.venv_python or EVAL_PYTHON),
                             a.gpu) + _cloud_cmd(spec_rel, ids, a.parts, out_dir, "lightning", cost_rel,
                                                 a.wall_s or 23 * 3600, code, "eval_res" if ev else "res",
                                                 registry=not ev, gpu=a.gpu, skip_from=skip_from,
                one_per_cpu=ev)
            argv2 = [LIGHTNING_PYTHON, os.path.join(R.ROOT, "scratchpad", "e6_dev", "xm_lightning.py"), "launch",
                     a.name, "--machine", a.machine, "--out-dir", out_dir, "--cmd", cmd,
                     *(["--min-balance", str(a.min_balance)] if a.min_balance is not None else []),
                     "--paths", "cdd_oran", spec_rel, *extra]
        if not a.dry_run:
            env = {**os.environ, **({"COLAB_ACCEL": "gpu:T4"} if a.cmd == "colab" and a.gpu else {})}
            subprocess.run(argv2, cwd=R.ROOT, check=True, env=env)
        print(" ".join(shlex.quote(x) for x in argv2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
