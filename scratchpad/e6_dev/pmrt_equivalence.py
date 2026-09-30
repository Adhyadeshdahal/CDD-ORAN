"""EQUIVALENCE proof for the label-only PMRT rename (docs/benchmark/METHOD_NAMES.md,
docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md): the renamed code + E6P_PMRT_V4.json must compute exactly
what the frozen code at commit 4fc2cd9 ("MSCR+": crt_units_plus / mscr_multi / mscr_integrate_bench /
mscr_plus_artifacts) + E6P_MSCRPLUS_V4_FROZEN.json computes.

  python scratchpad/e6_dev/pmrt_equivalence.py local --small DIR [--B 999] [--frozen-commit 4fc2cd9] [--out F.json]
  python scratchpad/e6_dev/pmrt_equivalence.py reports FROZEN.json RENAMED.json [--out F.json]

``local``: exports the frozen tree (``git archive``) to a temp dir and runs, in a separate interpreter per tree (no
module can leak across), the integrated statistic (``run_integrated``: every per-hypothesis z, beta, p2, p_plus,
p_minus, sign, status) and every declaration combination (``declare_all``: stat x layer, weights, directions, p used,
declared, sign) on the small caches placebo1 / plxc2 (split 9, the v4 placebo split) and dev1 (split 0). Each worker
writes one canonical JSON (sorted keys, floats by repr = exact); the proof is byte equality of the two, plus the max
abs difference over all numeric leaves (0 when equal).
``reports``: compares two v4 analysis reports (e6p_disc_analyze_v4.py analyze, frozen vs renamed code, same inputs)
leaf by leaf, ignoring only provenance / timing fields (``VOLATILE``); strings are compared after mapping the old
method label to the new one (``LABELS``: a rule text such as "count for MSCR+" reads "count for PMRT").
Exit status 0 iff equivalent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FROZEN_COMMIT = "4fc2cd9"
FROZEN_ARTIFACT = "docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json"
PMRT_ARTIFACT = "docs/benchmark/artifacts/E6P_PMRT_V4.json"
SMALL = (("placebo1", 9), ("plxc2", 9), ("dev1", 0))
EXPORT = ("cdd_oran", "scratchpad/e6_dev", "docs/benchmark/artifacts")
# report fields that legitimately differ between the two runs (provenance, file names, wall time, memory)
VOLATILE = {"argv", "artifact", "protocol", "timing", "timing_s", "wall_s", "peak_rss_mb", "rss_mb", "out", "cache_dir",
            "generated", "host", "elapsed_s"}
# label-only rename: string leaves are compared after mapping the old method label to the new one (reports mode only)
LABELS = (("MSCR+", "PMRT"),)


def _label(x):
    if isinstance(x, str):
        for old, new in LABELS:
            x = x.replace(old, new)
    return x


# ============================================================================================ worker (one tree)
def worker(a):
    """Runs INSIDE one tree (--root): imports only that tree's modules."""
    root = os.path.abspath(a.root)
    for p in (os.path.join(root, "scratchpad", "e6_dev"), root):
        sys.path.insert(0, p)
    if a.impl == "frozen":
        import mscr_integrate_bench as PB
        import mscr_plus_artifacts as PAR

        from cdd_oran.decision import crt_units_plus as PM
        load_pool, data, config = PM.load_plus_pool, PM.plus_data, PAR.plus_config
        art = FROZEN_ARTIFACT
    else:
        import pmrt_artifacts as PAR
        import pmrt_bench as PB

        from cdd_oran.decision import pmrt as PM
        load_pool, data, config = PM.load_pmrt_pool, PM.pmrt_data, PAR.pmrt_config
        art = PMRT_ARTIFACT
    for m in (PB, PAR, PM):
        assert os.path.abspath(m.__file__).startswith(root), (m.__name__, m.__file__, root)
    params, priors = PAR.load_artifact(os.path.join(root, *art.split("/")))
    cfg = config(params, a.B)
    out = {}
    for nm, split in SMALL:
        pool = load_pool([os.path.join(a.small, f"{nm}.npz")])
        pd = data(pool)
        run = PB.run_integrated(pd, params, cfg, split)
        decl = PB.declare_all(run, priors, int(pool.n_eps))
        out[nm] = {"split": split, "n_eps": int(pool.n_eps), "n_units": int(pd.n), "run": run,
                   "decl": {f"{s}+{lay}": {"|".join(h): v for h, v in d.items()} for (s, lay), d in decl.items()}}
    with open(a.out, "w", newline="\n") as fh:
        json.dump(out, fh, sort_keys=True, default=_default, allow_nan=True)


def _default(o):
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    return repr(o)


# ============================================================================================ comparison
def diff(x, y, path="", ignore=frozenset(), out=None, labels=False):
    """Leaf-by-leaf differences [(path, x, y)]; NaN == NaN; floats compared exactly; labels: strings compared after
    ``_label`` (old method label -> new)."""
    out = [] if out is None else out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in ignore:
                continue
            if k not in x or k not in y:
                out.append((f"{path}/{k}", x.get(k, "<missing>"), y.get(k, "<missing>")))
            else:
                diff(x[k], y[k], f"{path}/{k}", ignore, out, labels)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((f"{path}#len", len(x), len(y)))
        for i, (u, v) in enumerate(zip(x, y, strict=False)):
            diff(u, v, f"{path}[{i}]", ignore, out, labels)
    elif isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        pass
    elif labels and isinstance(x, str) and isinstance(y, str) and _label(x) == _label(y):
        pass
    elif type(x) is not type(y) or x != y:
        out.append((path, x, y))
    return out


def max_abs(d) -> float:
    m = 0.0
    for _, x, y in d:
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            m = max(m, abs(float(x) - float(y)))
        else:
            m = math.inf
    return m


def n_leaves(x) -> int:
    if isinstance(x, dict):
        return sum(n_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(n_leaves(v) for v in x)
    return 1


def summary(d, a, b, extra) -> dict:
    return {"equivalent": not d, "n_leaves": n_leaves(a), "n_diff": len(d), "max_abs_diff": max_abs(d),
            "first_diffs": [[p, repr(x)[:120], repr(y)[:120]] for p, x, y in d[:20]]} | extra


# ============================================================================================ commands
def cmd_local(a):
    tmp = tempfile.mkdtemp(prefix="pmrt_eq_")
    try:
        frozen = os.path.join(tmp, "frozen")
        os.makedirs(frozen)
        arc = subprocess.run(["git", "-C", ROOT, "archive", a.frozen_commit, *EXPORT], check=True,
                             capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", frozen], input=arc, check=True)
        res = {}
        for impl, root in (("frozen", frozen), ("renamed", ROOT)):
            f = os.path.join(tmp, f"{impl}.json")
            r = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "worker", "--impl", impl,
                                "--root", root, "--small", a.small, "--B", str(a.B), "--out", f],
                               capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"{impl} worker failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
            res[impl] = open(f, "rb").read()
        same_bytes = res["frozen"] == res["renamed"]
        A, B = (json.loads(res[k]) for k in ("frozen", "renamed"))
        d = diff(A, B)
        rep = summary(d, A, B, {"mode": "local", "frozen_commit": a.frozen_commit, "B": a.B,
                                "small": [n for n, _ in SMALL], "byte_identical": same_bytes,
                                "units": {n: A[n]["n_units"] for n, _ in SMALL},
                                "n_hyp": {n: len(A[n]["run"]) for n, _ in SMALL},
                                "n_declared_primary": {n: sum(bool(v.get("declared"))
                                                              for v in A[n]["decl"]["loadsp_c+wby1s"].values())
                                                       for n, _ in SMALL}})
        rep["equivalent"] = rep["equivalent"] and same_bytes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


def cmd_reports(a):
    A, B = (json.load(open(p)) for p in (a.frozen, a.renamed))
    d = diff(A, B, ignore=frozenset(VOLATILE), labels=True)
    return summary(d, A, B, {"mode": "reports", "frozen": a.frozen, "renamed": a.renamed,
                             "ignored": sorted(VOLATILE), "labels": [list(t) for t in LABELS]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    lo = sp.add_parser("local")
    lo.add_argument("--small", required=True, help="dir with placebo1.npz, plxc2.npz, dev1.npz")
    lo.add_argument("--B", type=int, default=999)
    lo.add_argument("--frozen-commit", dest="frozen_commit", default=FROZEN_COMMIT)
    lo.add_argument("--out", default="")
    rp = sp.add_parser("reports")
    rp.add_argument("frozen")
    rp.add_argument("renamed")
    rp.add_argument("--out", default="")
    wk = sp.add_parser("worker")
    wk.add_argument("--impl", choices=("frozen", "renamed"), required=True)
    wk.add_argument("--root", required=True)
    wk.add_argument("--small", required=True)
    wk.add_argument("--B", type=int, required=True)
    wk.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.op == "worker":
        return worker(a)
    rep = cmd_local(a) if a.op == "local" else cmd_reports(a)
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.out:
        open(a.out, "w", newline="\n").write(txt + "\n")
    sys.exit(0 if rep["equivalent"] else 1)


if __name__ == "__main__":
    main()
