"""Job runner of the cross-method study: shard (world, regime, lam, n, seed, kappa, method) jobs, run, score, merge.

    uv run python -m cdd_oran.xmethod.runner list   --spec SPEC.json
    uv run python -m cdd_oran.xmethod.runner run    --spec SPEC.json --part i/P --out res_i.jsonl
    uv run python -m cdd_oran.xmethod.runner merge  --spec SPEC.json --inputs 'runs/x/**/*.jsonl' --out merged.jsonl
    uv run python -m cdd_oran.xmethod.runner kaggle --spec SPEC.json --name NAME [--parts 4] [--dry-run]
    uv run python -m cdd_oran.xmethod.runner colab  --spec SPEC.json --name NAME [--parts 4] [--dry-run]

SPEC (JSON; repo-relative path so the cloud bundles carry it)::

    {"methods": ["dummy", "pkg.module:Class", ...],      # registry name or "module:attr" (class or instance)
     "worlds": ["E1", ...], "regimes": ["R1", ...],        # regimes not defined for a world are skipped
     "ns": [500, 1000, 4000], "seeds": [3000000, 3000019], # [lo, hi] inclusive, or an explicit list
     "lams": [0.0, 0.5, 1.0, 1.5],                         # E4 only (other worlds: lam = null)
     "kappas": [0.3],                                      # REQUIRED (R-24 observation noise; 0 = noiseless);
                                                           # a spec without it is an error (no world default)
     "configs": {"<method>": {"default": {...}, "E1|R1|n500": {...}, "E1|R1|k0.3|n500": {...}}}}

Config lookup per cell: "default", then the kappa-free cell key, then the cell key with kappa ("k<kappa>|").
Job keys always carry the kappa: "<method>|<world>|<regime>|[lam<l>|]k<kappa>|n<n>|s<seed>".

Jobs are grouped by dataset (world, regime, lam, n, seed, kappa); a part gets whole groups (group index i mod P), so a
dataset is generated once per part and every method runs on the same arrays. ``run`` appends one JSON line per
job and skips keys already present without error in ``--out`` (resumable). Each record: job, dataset sha256,
the Result (edges with score / p / sign / declared), ``scores`` (``score.score``), ``gen_cpu_s``, ``cpu_s``
(process CPU seconds of ``method.run``, all threads), ``method_cpu_s`` (the Result's own figure), ``wall_s``,
``peak_rss_mb`` (+ ``rss_scope``: "job" on Linux via /proc clear_refs + VmHWM, "process" = process-lifetime peak
on Windows, null elsewhere), versions, and ``error`` (traceback text) if the job failed.
Seeds: the runner never generates seeds; the spec names them (DEV block 3_000_000-3_000_199 only, CONTRACT 6).
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from typing import Any

import numpy as np

from cdd_oran.xmethod import api
from cdd_oran.xmethod.score import score
from cdd_oran.xmethod.worlds import REGIMES_OF, dataset_hash, generate_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV_SEEDS = range(3_000_000, 3_000_200)
EVAL_FLOOR = 3_100_000
RUNNER_VERSION = "xm-runner/1"

REGISTRY: dict[str, str] = {
    "dummy": "cdd_oran.xmethod.runner:DummyMethod",
}


# ------------------------------------------------------------------------------------------------ dummy method
def _by_declare(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Yekutieli step-up at level q over all of ``p``."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    ok = p[order] <= q * np.arange(1, m + 1) / (m * c_m)
    k = int(np.max(np.nonzero(ok)[0])) + 1 if ok.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out


class DummyMethod:
    """End-to-end test method: i.i.d. uniform p-values (score = 1 - p), random signs, BY at q = .05.
    Reads nothing but the candidate list and the dataset identity (for its RNG)."""

    name = "dummy"
    version = "1"

    def tune(self, dev: list[api.Dataset], truth_free: bool = True) -> dict[str, Any]:
        return {"q": 0.05}

    def run(self, data: api.Dataset, config: dict[str, Any]) -> api.Result:
        t0 = time.process_time()
        lam = int(round(1000 * float(data.meta.get("lam", 0.0))))
        rng = np.random.default_rng(np.random.SeedSequence(
            [7899, int(data.seed), int(data.world[1]), int(data.regime[1]), lam, int(data.n)]))
        m = len(data.candidates)
        p = rng.uniform(size=m)
        sign = rng.choice([-1, 1], size=m)
        dec = _by_declare(p, float(config.get("q", 0.05)))
        edges = tuple(api.EdgeResult(s, t, float(1 - p[i]), float(p[i]), int(sign[i]), bool(dec[i]))
                      for i, (s, t) in enumerate(data.candidates))
        return api.Result(self.name, self.version, edges, time.process_time() - t0, dict(config))


# ------------------------------------------------------------------------------------------------ jobs
def _seeds(spec: dict) -> list[int]:
    s = spec["seeds"]
    seeds = list(range(int(s[0]), int(s[1]) + 1)) if (len(s) == 2 and s[1] > s[0] + 1) else [int(x) for x in s]
    bad = [x for x in seeds if x not in DEV_SEEDS]
    if bad:
        raise ValueError(f"seeds outside the DEV block 3000000-3000199 (CONTRACT 6): {bad[:5]}")
    return seeds


def _kappas(spec: dict) -> list[float]:
    """The spec's observation-noise levels (R-24). Required: no cell may silently use a world default."""
    if "kappas" not in spec:
        raise ValueError('spec must give "kappas" (R-24 observation noise, e.g. [0.3]; [0] = noiseless)')
    ks = spec["kappas"]
    if isinstance(ks, (str, bytes)) or not hasattr(ks, "__iter__") or not len(ks):
        raise ValueError(f'"kappas" must be a non-empty list of numbers, got {ks!r}')
    out = []
    for k in ks:
        if k is None or isinstance(k, bool) or not math.isfinite(float(k)) or float(k) < 0:
            raise ValueError(f'"kappas" entries must be finite numbers >= 0, got {k!r}')
        out.append(float(k))
    return out


def dataset_groups(spec: dict) -> list[tuple]:
    """Ordered dataset keys (world, regime, lam, n, seed, kappa)."""
    kappas = _kappas(spec)
    out = []
    for w in spec["worlds"]:
        for r in spec["regimes"]:
            if r not in REGIMES_OF[w]:
                continue
            lams = [float(x) for x in spec.get("lams", [1.0])] if w == "E4" else [None]
            for lam in lams:
                for kap in kappas:
                    for n in spec["ns"]:
                        for s in _seeds(spec):
                            out.append((w, r, lam, int(n), int(s), kap))
    return out


def _kpart(kappa: float | None) -> str:
    return "" if kappa is None else f"k{kappa:g}|"


def job_key(method: str, w: str, r: str, lam: float | None, n: int, s: int, kappa: float | None = None) -> str:
    return f"{method}|{w}|{r}|{'' if lam is None else f'lam{lam:g}|'}{_kpart(kappa)}n{n}|s{s}"


def cell_key(w: str, r: str, lam: float | None, n: int, kappa: float | None = None) -> str:
    return f"{w}|{r}|{'' if lam is None else f'lam{lam:g}|'}{_kpart(kappa)}n{n}"


def all_keys(spec: dict) -> list[str]:
    return [job_key(m, *g) for g in dataset_groups(spec) for m in spec["methods"]]


def method_config(spec: dict, method: str, w, r, lam, n, kappa: float | None = None) -> dict:
    c = spec.get("configs", {}).get(method, {})
    out = {**c.get("default", {}), **c.get(cell_key(w, r, lam, n), {})}
    if kappa is not None:
        out.update(c.get(cell_key(w, r, lam, n, kappa), {}))
    return out


def load_method(ref: str):
    target = REGISTRY.get(ref, ref)
    mod, _, attr = target.partition(":")
    obj = getattr(importlib.import_module(mod), attr)
    return obj() if isinstance(obj, type) else obj


# ------------------------------------------------------------------------------------------------ memory
class PeakRSS:
    """Peak resident memory over a block. Linux: reset the high-water mark (/proc/self/clear_refs "5") and read
    VmHWM after (scope "job"). Windows: process-lifetime PeakWorkingSetSize (scope "process"). Else None."""

    def __enter__(self):
        self.scope = None
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/clear_refs", "w") as f:
                    f.write("5")
                self.scope = "job"
            except OSError:
                self.scope = "process"
        elif sys.platform == "win32":
            self.scope = "process"
        return self

    def __exit__(self, *exc):
        self.mb = None
        try:
            if sys.platform.startswith("linux"):
                for line in open("/proc/self/status"):
                    if line.startswith("VmHWM:"):
                        self.mb = int(line.split()[1]) / 1024.0
            elif sys.platform == "win32":
                import ctypes
                from ctypes import wintypes

                class PMC(ctypes.Structure):
                    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                        (f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize",
                                                       "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                                       "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                       "PagefileUsage", "PeakPagefileUsage")]
                pmc = PMC()
                pmc.cb = ctypes.sizeof(PMC)
                k32 = ctypes.WinDLL("kernel32")
                k32.GetCurrentProcess.restype = wintypes.HANDLE
                psapi = ctypes.WinDLL("psapi")
                psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
                if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                    self.mb = pmc.PeakWorkingSetSize / 2 ** 20
        except Exception:                               # memory figure is diagnostic only
            self.mb = None
        return False


# ------------------------------------------------------------------------------------------------ run
def _clean(x):
    """JSON-safe copy: NaN / inf -> None, numpy scalars / arrays -> Python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "runner": RUNNER_VERSION}


def run_one(method, ds: api.Dataset, truth: api.Truth, config: dict, gen_cpu_s: float, key: str) -> dict:
    rec: dict[str, Any] = {"key": key, "method": getattr(method, "name", None),
                           "job": {"world": ds.world, "regime": ds.regime, "lam": ds.meta.get("lam"), "n": ds.n,
                                   "seed": ds.seed, "kappa": float((ds.meta.get("obs_noise") or {}).get("kappa", 0.0))},
                           "dataset_sha256": dataset_hash(ds), "gen_cpu_s": gen_cpu_s, "config": config,
                           "versions": _versions(), "error": None}
    try:
        with PeakRSS() as mem:
            c0, w0 = time.process_time(), time.perf_counter()
            res = method.run(ds, config)
            rec["cpu_s"], rec["wall_s"] = time.process_time() - c0, time.perf_counter() - w0
        rec.update(peak_rss_mb=mem.mb, rss_scope=mem.scope, version=res.version, method=res.method,
                   method_cpu_s=res.cpu_s, result_config=res.config, notes=res.notes,
                   edges=[{"source": e.source, "target": e.target, "score": e.score, "p": e.p, "sign": e.sign,
                           "declared": e.declared} for e in res.edges],
                   scores=score(res, truth, candidates=ds.candidates, kpi_sources=ds.kpi_names))
    except Exception:
        rec["error"] = traceback.format_exc()
    return _clean(rec)


def _done_keys(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:                # partial last line of an interrupted run
                continue
            if r.get("error") is None:
                done.add(r["key"])
    return done


def run_part(spec: dict, part: int, parts: int, out: str, log=print) -> int:
    groups = dataset_groups(spec)[part::parts]
    done = _done_keys(out)
    methods: dict[str, Any] = {}
    n_run = 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        for w, r, lam, n, s, kap in groups:
            todo = [m for m in spec["methods"] if job_key(m, w, r, lam, n, s, kap) not in done]
            if not todo:
                continue
            c0 = time.process_time()
            ds, truth = generate_dataset(w, r, n, s, lam=1.0 if lam is None else lam, kappa=kap)
            gen = time.process_time() - c0
            for m in todo:
                if m not in methods:
                    methods[m] = load_method(m)
                key = job_key(m, w, r, lam, n, s, kap)
                rec = run_one(methods[m], ds, truth, method_config(spec, m, w, r, lam, n, kap), gen, key)
                fh.write(json.dumps(rec, allow_nan=False) + "\n")
                fh.flush()
                n_run += 1
                log(f"[xm] {key} cpu {rec.get('cpu_s', float('nan')):.2f}s "
                    f"{'ERROR' if rec['error'] else 'ok'}")
    return n_run


# ------------------------------------------------------------------------------------------------ merge
def merge(spec: dict, inputs: list[str], out: str) -> dict:
    """Merge part files: one record per key (a clean record beats an error record; later beats earlier),
    sorted by key; writes ``out`` and ``<out>.summary.json`` (completeness + per-cell means)."""
    best: dict[str, dict] = {}
    for path in inputs:
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            old = best.get(r["key"])
            if old is None or old.get("error") is not None or r.get("error") is None:
                best[r["key"]] = r
    want = all_keys(spec)
    with open(out, "w", encoding="utf-8") as fh:
        for k in sorted(best):
            fh.write(json.dumps(best[k], allow_nan=False) + "\n")
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in best.values():
        if r.get("error") is None:
            j = r["job"]
            cells[f"{r['key'].split('|')[0]}|{cell_key(j['world'], j['regime'], j['lam'], j['n'], j.get('kappa'))}"
                  ].append(r)

    def mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None

    per_cell = {}
    for c, rs in sorted(cells.items()):
        sc = [r["scores"] for r in rs]
        per_cell[c] = {"n_jobs": len(rs), **{m: mean([s[m] for s in sc]) for m in
                                             ("precision", "recall", "f1", "fdp", "null_fpr", "sign_acc")},
                       "placebo_declared_total": int(sum(s["placebo_declared"] for s in sc)),
                       "placebo_conf_declared_total": int(sum(s.get("placebo_conf_declared", 0) for s in sc)),
                       "n_not_testable_true_total": int(sum(s.get("n_not_testable_true", 0) for s in sc)),
                       "n_not_testable_null_total": int(sum(s.get("n_not_testable_null", 0) for s in sc)),
                       "cpu_s_mean": mean([r["cpu_s"] for r in rs]),
                       "peak_rss_mb_max": max((r["peak_rss_mb"] or 0) for r in rs)}
    summary = {"n_expected": len(want), "n_records": len(best),
               "missing": sorted(set(want) - set(best)),
               "errors": sorted(k for k, r in best.items() if r.get("error") is not None),
               "unexpected": sorted(set(best) - set(want)), "per_cell": per_cell}
    json.dump(summary, open(out + ".summary.json", "w"), indent=1)
    return summary


# ------------------------------------------------------------------------------------------------ cloud
def _cloud_cmd(spec_rel: str, parts: int, out_dir: str) -> str:
    """Bash command run from the bundle root: P single-threaded processes, then wait. Single-quote-safe."""
    thr = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    runs = " ".join(f"{thr} python -u -m cdd_oran.xmethod.runner run --spec {spec_rel} --part {i}/{parts} "
                    f"--out {out_dir}/res_{i}.jsonl > {out_dir}/log_{i}.txt 2>&1 &" for i in range(parts))
    return f"mkdir -p {out_dir} && {runs} wait; tail -n 3 {out_dir}/log_*.txt"


def launch_kaggle(spec_rel: str, name: str, parts: int, extra_paths: list[str], dry_run: bool) -> list[str]:
    """One Kaggle session (scratchpad/e6_dev/kaggle_job.py), ``parts`` processes; outputs in $JOB_OUT.
    Pull with ``kaggle_job.py pull NAME`` (-> runs/NAME/), then ``merge``."""
    cmd = _cloud_cmd(spec_rel, parts, "$JOB_OUT")      # $JOB_OUT expands on Kaggle, never locally (argv list)
    argv = [sys.executable, os.path.join(ROOT, "scratchpad", "e6_dev", "kaggle_job.py"), "launch", name,
            "--cmd", cmd, "--paths", spec_rel, *extra_paths]
    if not dry_run:
        subprocess.run(argv, cwd=ROOT, check=True)
    return argv


def launch_colab(spec_rel: str, name: str, parts: int, extra_paths: list[str], dry_run: bool) -> list[str]:
    """One-shot Colab job (scratchpad/e6_dev/colab_run.py job); outputs under xm_out/ (mirrored to
    runs/NAME/xm_out by its scheduled tick), then ``merge``."""
    argv = [sys.executable, os.path.join(ROOT, "scratchpad", "e6_dev", "colab_run.py"), "job", name,
            "--paths", "cdd_oran", spec_rel, *extra_paths, "--cmd", _cloud_cmd(spec_rel, parts, "xm_out"),
            "--out-dir", "xm_out", "--threads", "1"]
    if not dry_run:
        subprocess.run(argv, cwd=ROOT, check=True)
    return argv


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cdd_oran.xmethod.runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("list", "run", "merge", "kaggle", "colab"):
        p = sub.add_parser(c)
        p.add_argument("--spec", required=True)
        if c == "run":
            p.add_argument("--part", default="0/1")
            p.add_argument("--out", required=True)
        if c == "merge":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--out", required=True)
        if c in ("kaggle", "colab"):
            p.add_argument("--name", required=True)
            p.add_argument("--parts", type=int, default=4)
            p.add_argument("--paths", nargs="*", default=[])
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    if a.cmd == "list":
        g = dataset_groups(spec)
        print(json.dumps({"datasets": len(g), "jobs": len(g) * len(spec["methods"])}))
    elif a.cmd == "run":
        i, p = (int(x) for x in a.part.split("/"))
        print(f"[xm] part {i}/{p}: {run_part(spec, i, p, a.out)} jobs run")
    elif a.cmd == "merge":
        files = sorted({f for pat in a.inputs for f in glob.glob(pat, recursive=True)})
        s = merge(spec, files, a.out)
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()}))
    else:
        spec_rel = os.path.relpath(os.path.abspath(a.spec), ROOT).replace("\\", "/")
        fn = launch_kaggle if a.cmd == "kaggle" else launch_colab
        print(" ".join(shlex.quote(x) for x in fn(spec_rel, a.name, a.parts, a.paths, a.dry_run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
