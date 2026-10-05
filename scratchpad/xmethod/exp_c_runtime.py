"""Experiment C: per-method runtime table (Study A; docs/xmethod/PROTOCOL_A.md s.7, s.13 T3; R-13, R-35, R-41, R-51).

Reads campaign record files (merged.jsonl.gz or raw res_*.jsonl shards; DEV now, EVAL later, same format) and
writes, per (arm, n), CPU-s per unit (= per (method, dataset); record field ``cpu_s`` = process CPU time of
``method.run``, single thread, data generation excluded) as mean / median / max in Kaggle reference seconds, peak RSS
(MB, record ``peak_rss_mb``, not converted) and feasibility against the R-13 budget (7200 CPU-s per (method,
dataset)) by the T3 rule: the max over worlds, regimes, lambdas of the unit cost at n > budget makes n and every
larger n infeasible.

Host conversion (R-51, T3): a unit run on another host type is converted as cost x f, f = sum of Kaggle cost / sum of
host cost over the same calibration units run on both hosts (per (arm, host type)). Factor sources, in order:
  1. ``--factors`` file written by ``calib`` (paired units; with ``--plan`` exactly the T3 calibration units);
  2. otherwise a PROVISIONAL within-cell estimate (``observational``): the same (arm, world, regime, lam, kappa, n)
     cell has seeds on both hosts (platforms are assigned per dataset), f = sum_c n_H(c) mean_ref(c) /
     sum_c n_H(c) mean_H(c). Not the T3 factor; disabled with ``--no-observational`` (then unconverted units are
     flagged and their arm x n row is marked provisional).
A host type is "platform|cpu model" (+ "|gpu name" on GPU sessions); the reference is ``--ref-host``. GPU arms (spec
``budget_wall_s``, R-41) are read in wall-s (child wall clock when isolated) against a Kaggle T4, reported apart.

Subcommands
  table  --records F.. [--spec S..] [--factors J] --out-dir D   -> exp_c_runtime.json + EXP_C_RUNTIME.md
  plan   --records F.. --out P.json      the T3 calibration units (DEV seeds 3_000_000-002 in the arm's 3 costliest
                                         (world, regime) at its largest measured n) per arm and host type to run
  calib  --records F.. [--plan P.json] --out J.json   speed factors from units present on the reference and a host
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

VERSION = "xm-exp-c/1"
BUDGET_CPU_S = 7200.0                                       # R-13 / PROTOCOL_A s.7
REF_HOST = "kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz"          # Kaggle CPU session (T3 reference host)
N_GRID = (500, 1000, 4000, 8000, 24000)                     # PROTOCOL_A s.7
CALIB_SEEDS = (3_000_000, 3_000_001, 3_000_002)             # T3 calibration units
PRIMARY_KAPPA = 0.25
MULTI_THREAD_RATIO = 1.15                                   # cpu_s > 1.15 wall_s + .05: more than one busy thread
CONTENTION_RATIO = 1.3                                      # wall_s > 1.3 cpu_s + .05: CPU-starved (oversubscribed host)


# ================================================================================================ load
def host_type(r: dict) -> str:
    h = r.get("host") or {}
    parts = [str(h.get("platform") or "unknown"), str(h.get("cpu_model") or "unknown")]
    gpu = h.get("gpu")
    if gpu:
        parts.append(str(gpu[0] if isinstance(gpu, list) else gpu).split(",")[0].strip())
    return "|".join(parts)


def _arm(r: dict) -> str:
    return r.get("arm") or str(r.get("key", "?")).split("|", 1)[0]


def slim(r: dict) -> dict:
    """The fields Experiment C needs (records carry edges / notes; keep memory flat for EVAL-size inputs)."""
    j = r.get("job") or {}
    st = r.get("status") or ("ok" if r.get("error") is None else "error")
    wall = r.get("child_wall_s") if r.get("child_wall_s") is not None else r.get("wall_s")
    cost = None
    if st == "infeasible":                                   # campaign: measured over-budget unit (cost, cost_kind)
        cost = r.get("cost") if r.get("cost") is not None else r.get("cpu_s")
    return {"key": r.get("key"), "arm": _arm(r), "method": r.get("method"), "world": j.get("world"),
            "regime": j.get("regime"), "lam": j.get("lam"), "kappa": j.get("kappa"), "n": j.get("n"),
            "seed": j.get("seed"), "role": r.get("role"), "status": st, "cpu_s": r.get("cpu_s"), "wall_s": wall,
            "peak_rss_mb": r.get("peak_rss_mb"), "rss_scope": r.get("rss_scope"), "gen_cpu_s": r.get("gen_cpu_s"),
            "gpu_s": r.get("gpu_s"), "host": host_type(r), "cost": cost, "cost_kind": r.get("cost_kind"),
            "spec_name": r.get("spec_name"), "commit": (r.get("code") or {}).get("commit"),
            "dataset_sha256": r.get("dataset_sha256"), "procs_per_vcpu": procs_per_vcpu(r),
            "node": (r.get("host") or {}).get("node")}


def procs_per_vcpu(r: dict) -> float | None:
    """Concurrent campaign / timing processes per vCPU at the unit's start (R-55 load fields), else None."""
    ld = ((r.get("load") or {}).get("start")) or {}
    k = ld.get("campaign_procs") if ld.get("campaign_procs") is not None else ld.get("timing_procs")
    q = ld.get("cpu_quota")
    return float(k) / float(q) if k is not None and q else None


def iter_records(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:                     # partial last line of a live shard
                print(f"[exp-c] {path}:{i + 1}: unparsable line skipped", file=sys.stderr)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(paths: list[str]) -> tuple[list[dict], dict]:
    """One record per key (first ok record wins; an ok record replaces an earlier non-ok one). Every copy of a key
    is kept in ``info['copies']`` (key -> [slim, ...]) when it appears more than once (cross-host pairs)."""
    by_key: dict[str, dict] = {}
    copies: dict[str, list[dict]] = defaultdict(list)
    info = {"files": [], "lines": 0, "duplicates": 0, "conflicts": 0}
    for p in paths:
        n0 = info["lines"]
        for r in iter_records(p):
            info["lines"] += 1
            s = slim(r)
            k = s["key"]
            copies[k].append(s)
            if k not in by_key:
                by_key[k] = s
                continue
            info["duplicates"] += 1
            old = by_key[k]
            if old["status"] != "ok" and s["status"] == "ok":
                by_key[k] = s
            elif old["status"] == s["status"] == "ok" and old["host"] != s["host"]:
                info["conflicts"] += 1                       # same unit on two hosts: kept first, pair for calib
        info["files"].append({"path": os.path.abspath(p), "sha256": file_sha256(p), "lines": info["lines"] - n0})
    info["copies"] = {k: v for k, v in copies.items() if len(v) > 1}
    return list(by_key.values()), info


def load_specs(paths: list[str] | None) -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for p in paths or []:
        arms.update(json.load(open(p, encoding="utf-8")).get("arms", {}))
    return arms


def cost_field(arm_spec: dict | None) -> str:
    """'wall_s' for GPU arms (spec budget_wall_s, R-41), else 'cpu_s' (R-13); as eval_analysis.cost_field."""
    return "wall_s" if (arm_spec or {}).get("budget_wall_s") else "cpu_s"


def unit_cost(s: dict, field: str) -> float | None:
    if s["status"] == "infeasible":
        if s["cost"] is not None and (s["cost_kind"] is None or ("wall" in str(s["cost_kind"])) == (field == "wall_s")):
            return float(s["cost"])
        return None
    v = s.get(field)
    return None if v is None else float(v)


def _cell(s: dict) -> tuple:
    return (s["arm"], s["world"], s["regime"], s["lam"], s["kappa"], s["n"])


# ================================================================================================ factors
def observational_factors(recs: list[dict], ref: str, fields: dict[str, str]) -> dict[tuple, dict]:
    """(arm, host) -> provisional f from cells measured on both the reference and the host (see module doc)."""
    g: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in recs:
        if s["status"] == "ok":
            c = unit_cost(s, fields.get(s["arm"], "cpu_s"))
            if c is not None:
                g[_cell(s)][s["host"]].append(c)
    num: dict[tuple, float] = defaultdict(float)
    den: dict[tuple, float] = defaultdict(float)
    ratios: dict[tuple, list[float]] = defaultdict(list)
    units: dict[tuple, int] = defaultdict(int)
    for cell, hosts in g.items():
        if ref not in hosts:
            continue
        mref = float(np.mean(hosts[ref]))
        for h, xs in hosts.items():
            if h == ref:
                continue
            mh, k = float(np.mean(xs)), len(xs)
            num[(cell[0], h)] += k * mref
            den[(cell[0], h)] += k * mh
            units[(cell[0], h)] += k
            if mh > 0:
                ratios[(cell[0], h)].append(mref / mh)
    out = {}
    for key in num:
        if den[key] > 0:
            r = ratios[key]
            out[key] = {"f": num[key] / den[key], "f_hi": max([num[key] / den[key], *r]), "source": "observational",
                        "n_cells": len(r),
                        "n_host_units": units[key],
                        "cell_ratio_q25_q50_q75": [float(x) for x in np.percentile(r, [25, 50, 75])] if r else None}
    return out


def load_factor_file(path: str | None) -> dict[tuple, dict]:
    """``calib`` output: {"factors": {arm: {host: {"f": .., ...}}}}; arm "*" = every arm without its own factor."""
    if not path:
        return {}
    d = json.load(open(path, encoding="utf-8"))
    src = d.get("source", "calib")
    return {(a, h): {**v, "source": v.get("source", src)} for a, hs in d.get("factors", {}).items()
            for h, v in hs.items()}


def resolve_factor(arm: str, host: str, ref: str, explicit: dict, obs: dict) -> dict | None:
    if host == ref:
        return {"f": 1.0, "f_hi": 1.0, "source": "reference"}
    return explicit.get((arm, host)) or explicit.get(("*", host)) or obs.get((arm, host))


# ================================================================================================ table
def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": float(np.mean(xs)), "median": float(np.median(xs)), "min": float(min(xs)), "max": float(max(xs))}


def build(recs: list[dict], info: dict, *, ref: str = REF_HOST, budget: float = BUDGET_CPU_S,
          arm_specs: dict | None = None, factors: dict | None = None, observational: bool = True,
          n_grid: tuple = N_GRID) -> dict:
    arm_specs = arm_specs or {}
    arms = sorted({s["arm"] for s in recs})
    fields = {a: cost_field(arm_specs.get(a)) for a in arms}
    obs = observational_factors(recs, ref, fields) if observational else {}
    explicit = factors or {}

    used: dict[tuple, dict] = {}
    rows: dict[tuple, dict] = defaultdict(lambda: {"cost": [], "raw": [], "rss": [], "wall": [], "gen": [], "gpu": [],
                                                   "hosts": defaultdict(int), "unconverted": 0, "infeasible": [],
                                                   "errors": 0, "no_cost": 0, "multi_thread": 0, "scopes": set(), "fsrc": set(), "contended": 0, "wc": [], "hi": [], "ppv": [],
                                                   "by_role": defaultdict(list),
                                                   "max_key": None, "methods": set(), "roles": defaultdict(int)})
    wr: dict[tuple, list[float]] = defaultdict(list)
    for s in recs:
        a, n = s["arm"], s["n"]
        row = rows[(a, n)]
        row["methods"].add(s["method"] or "?")
        if s["status"] == "error":
            row["errors"] += 1
            continue
        field = fields[a]
        c = unit_cost(s, field)
        if c is None:
            row["no_cost"] += 1
            continue
        f = resolve_factor(a, s["host"], ref, explicit, obs)
        if f is not None:
            used[(a, s["host"])] = f
            cr = c * f["f"]
            chi = c * float(f.get("f_hi") or f["f"])
            if s["host"] != ref:
                row["fsrc"].add(f["source"])
        else:
            row["unconverted"] += 1
            cr = chi = c
        row["hosts"][s["host"]] += 1
        row["roles"][s["role"]] += 1
        if s["status"] == "infeasible":
            row["infeasible"].append({"key": s["key"], "cost_raw": c, "cost_ref": cr, "host": s["host"]})
        else:
            if s["peak_rss_mb"] is not None:
                row["rss"].append(float(s["peak_rss_mb"]))
                row["scopes"].add(s["rss_scope"])
            for src, dst in (("wall_s", "wall"), ("gen_cpu_s", "gen"), ("gpu_s", "gpu")):
                if s[src] is not None:
                    row[dst].append(float(s[src]))
            if s["cpu_s"] is not None and s["wall_s"]:
                row["multi_thread"] += s["cpu_s"] > MULTI_THREAD_RATIO * s["wall_s"] + .05
                row["contended"] += s["wall_s"] > CONTENTION_RATIO * s["cpu_s"] + .05
                if s["cpu_s"] > 0:
                    row["wc"].append(s["wall_s"] / s["cpu_s"])
        row["raw"].append(c)
        row["hi"].append(chi)
        row["by_role"][str(s["role"])].append(cr)
        if s["procs_per_vcpu"] is not None:
            row["ppv"].append(s["procs_per_vcpu"])
        if not row["cost"] or cr > max(row["cost"]):
            row["max_key"] = {"key": s["key"], "host": s["host"], "cost_raw": c}
        row["cost"].append(cr)
        wr[(a, s["world"], s["regime"], n)].append(cr)

    table = []
    for (a, n), v in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        nref = v["hosts"].get(ref, 0)
        tot = sum(v["hosts"].values())
        table.append({
            "arm": a, "method": "/".join(sorted(v["methods"])), "n": n, "cost_field": fields[a],
            "budget": float((arm_specs.get(a) or {}).get("budget_wall_s") or budget),
            "n_units": tot, "n_ok": tot - len(v["infeasible"]), "n_infeasible": len(v["infeasible"]),
            "n_errors": v["errors"], "n_no_cost": v["no_cost"], "roles": dict(v["roles"]),
            "share_non_ref": (1 - nref / tot) if tot else None, "hosts": dict(v["hosts"]),
            "n_unconverted": v["unconverted"], "factor_sources": sorted(v["fsrc"]),
            "cost_ref": _stats(v["cost"]), "cost_raw": _stats(v["raw"]), "max_unit": v["max_key"],
            "cost_ref_hi_max": max(v["hi"]) if v["hi"] else None,
            "cost_ref_by_role": {k: _stats(x) for k, x in sorted(v["by_role"].items())},
            "procs_per_vcpu": ({"median": float(np.median(v["ppv"])), "max": float(max(v["ppv"]))}
                               if v["ppv"] else None),
            "peak_rss_mb": {"median": float(np.median(v["rss"])) if v["rss"] else None,
                            "max": max(v["rss"]) if v["rss"] else None, "scope": sorted(map(str, v["scopes"]))},
            "wall_s_mean": float(np.mean(v["wall"])) if v["wall"] else None,
            "gen_cpu_s_mean": float(np.mean(v["gen"])) if v["gen"] else None,
            "gpu_s_mean": float(np.mean(v["gpu"])) if v["gpu"] else None,
            "n_multi_thread": v["multi_thread"], "n_contended": v["contended"],
            "wall_cpu_ratio_median": float(np.median(v["wc"])) if v["wc"] else None, "infeasible_units": v["infeasible"][:10]})

    feas = t3(table, n_grid, {a: (d or {}).get("max_n") for a, d in arm_specs.items()})
    for row in table:
        row["t3"] = feas[row["arm"]]["by_n"].get(str(row["n"]))
    by_wr = [{"arm": k[0], "world": k[1], "regime": k[2], "n": k[3], "n_units": len(v), **_stats(v)}
             for k, v in sorted(wr.items(), key=lambda kv: (kv[0][0], kv[0][3] or 0, str(kv[0][1]), str(kv[0][2])))]
    return {
        "version": VERSION, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ref_host": ref, "budget_cpu_s": budget, "n_grid": list(n_grid),
        "inputs": {k: v for k, v in info.items() if k != "copies"},
        "spec_names": sorted({str(s["spec_name"]) for s in recs}), "commits": sorted({str(s["commit"]) for s in recs}),
        "hosts": host_counts(recs),
        "factors": [{"arm": a, "host": h, **f} for (a, h), f in sorted(used.items()) if h != ref],
        "observational_factors_all": [{"arm": a, "host": h, **f} for (a, h), f in sorted(obs.items())],
        "table": table, "feasibility": feas, "by_world_regime": by_wr,
        "spec_t3_fields": spec_fields(feas)}


def host_counts(recs: list[dict]) -> list[dict]:
    c: dict[str, int] = defaultdict(int)
    for s in recs:
        c[s["host"]] += 1
    return [{"host": h, "n_units": k} for h, k in sorted(c.items(), key=lambda kv: -kv[1])]


def t3(table: list[dict], n_grid: tuple, max_n: dict | None = None) -> dict:
    """PROTOCOL_A T3 per arm: the first n (ascending) whose max unit cost (reference seconds) exceeds the budget, or
    that has a measured infeasible unit, makes n and every larger n infeasible. R-55: a unit within the speed
    factor's error of the budget counts as over (test on cost x f_hi). Grid n without DEV cost below that are "no
    DEV cost" (T3: a cost pilot decides); above the arm's spec max_n they are "not in grid" (pre-registered, R-58)."""
    max_n = max_n or {}
    out = {}
    for a in sorted({r["arm"] for r in table}):
        rs = {r["n"]: r for r in table if r["arm"] == a}
        first, first_cost = None, None
        for n in sorted(rs):
            r = rs[n]
            mx = r["cost_ref"]["max"]
            over = [u["cost_ref"] for u in r["infeasible_units"]]
            hi = r.get("cost_ref_hi_max")
            if (hi is not None and hi > r["budget"]) or r["n_infeasible"]:
                first, first_cost = n, max([x for x in [mx, *over] if x is not None], default=None)
                break
        by_n = {}
        for n in sorted(set(n_grid) | set(rs)):
            r = rs.get(n)
            if first is not None and n == first:
                st = "infeasible"
            elif first is not None and n > first:
                st = "infeasible (smaller n infeasible)"
            elif r is None and max_n.get(a) and n > max_n[a]:
                st = "not in grid"
            elif r is None or r["cost_ref"]["max"] is None:
                st = "no DEV cost"
            else:
                st = "feasible"
            e = {"status": st}
            if r is not None and r["cost_ref"]["max"] is not None:
                e["max_cost"] = r["cost_ref"]["max"]
                e["budget_share"] = r["cost_ref"]["max"] / r["budget"]
                e["max_cost_hi"] = r.get("cost_ref_hi_max")
                if st == "infeasible" and r["cost_ref"]["max"] <= r["budget"] and not r["n_infeasible"]:
                    e["status"] = st = "infeasible (within the speed factor's error, R-55)"
                e["provisional"] = r["n_unconverted"] > 0     # some unit has no speed factor
            by_n[str(n)] = e
        out[a] = {"first_infeasible_n": first, "first_infeasible_cost": first_cost, "by_n": by_n}
    return out


def spec_fields(feas: dict) -> dict:
    """What the freeze spec takes for a T3-infeasible arm (eval spec tbd.max_n): max_n and t3_cost_cpu_s {n: cost}."""
    out = {}
    for a, d in feas.items():
        if d["first_infeasible_n"] is None:
            continue
        ok = [int(n) for n, e in d["by_n"].items() if e["status"] == "feasible"]
        out[a] = {"max_n": max(ok) if ok else None,
                  "t3_cost_cpu_s": {str(d["first_infeasible_n"]): d["first_infeasible_cost"]}}
    return out


# ================================================================================================ markdown
def _f(x, nd=1) -> str:
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.{nd}f}"
    return f"{x:.2f}" if x >= .1 else f"{x:.3f}"


def render_md(out: dict, title: str) -> str:
    tab = out["table"]
    ns = sorted({r["n"] for r in tab})
    L = [f"# Experiment C: runtime per method ({title})", "",
         f"Generated {out['generated_utc']} by `scratchpad/xmethod/exp_c_runtime.py` ({out['version']}). Unit = one "
         f"(method, dataset): record `cpu_s` = process CPU time of `method.run`, single thread (data generation "
         f"excluded, shown apart). Costs in **Kaggle reference CPU-s** (`{out['ref_host']}`); other hosts x f "
         f"(R-51). Peak RSS = `peak_rss_mb` (scope per job on Linux; includes interpreter, imports and dataset), "
         f"not converted. Budget R-13: {out['budget_cpu_s']:,.0f} CPU-s per (method, dataset); T3: max over "
         f"worlds, regimes, lambdas at n > budget makes n and every larger n infeasible; R-55: tested on cost x f_hi "
         f"(within the speed factor's error of the budget = over). Rows pool tune + measure seeds, kappas and "
         f"lambdas (per role and per world / regime in the json).", "",
         "## Inputs", ""]
    for f in out["inputs"]["files"]:
        L.append(f"- `{f['path']}` ({f['lines']} lines, sha256 `{f['sha256'][:16]}`)")
    L += [f"- {out['inputs']['lines']} lines, {out['inputs']['duplicates']} duplicate keys "
          f"({out['inputs']['conflicts']} on two hosts); specs {', '.join(out['spec_names'])}; "
          f"commits {', '.join(c[:7] for c in out['commits'])}", "", "Hosts (units):", ""]
    L += [f"- `{h['host']}`: {h['n_units']}" for h in out["hosts"]]

    L += ["", "## Speed factors applied (cost_ref = cost x f)", ""]
    if out["factors"]:
        L += ["| arm | host | f | f_hi | source | cells | host units | cell ratio q25 / q50 / q75 |",
              "|---|---|---|---|---|---|---|---|"]
        for f in out["factors"]:
            q = f.get("cell_ratio_q25_q50_q75")
            L.append(f"| {f['arm']} | `{f['host']}` | {f['f']:.3f} | {_f(f.get('f_hi'))} | {f['source']} | "
                     f"{f.get('n_cells', '-')} | "
                     f"{f.get('n_host_units', f.get('n_units', '-'))} | "
                     f"{' / '.join(f'{x:.2f}' for x in q) if q else '-'} |")
        if any(f["source"] == "observational" for f in out["factors"]):
            L += ["", "**observational** = PROVISIONAL within-cell estimate (same cell, different seeds on each host), "
                  "not the T3 factor (paired calibration units, `calib`). Replace with `--factors` before the freeze."]
    else:
        L.append("None needed (every unit on the reference host) or none available.")
    unconv = sum(r["n_unconverted"] for r in tab)
    if unconv:
        L += ["", f"**{unconv} units unconverted** (no factor for their (arm, host)): their rows are provisional."]

    L += ["", "## Summary: median / max CPU-s per unit (Kaggle ref.) by n; T3 status", "",
          "| arm | " + " | ".join(f"n {n}" for n in ns) + " | T3 |", "|---|" + "---|" * (len(ns) + 1)]
    for a in sorted({r["arm"] for r in tab}):
        cells = []
        for n in ns:
            r = next((x for x in tab if x["arm"] == a and x["n"] == n), None)
            if r is None or r["cost_ref"]["max"] is None:
                cells.append("-")
            else:
                mark = "" if r["t3"]["status"] == "feasible" else " **INF**"
                prov = ("*" if r["n_unconverted"] else "") + ("+" if "observational" in r["factor_sources"] else "")
                cells.append(f"{_f(r['cost_ref']['median'])} / {_f(r['cost_ref']['max'])}{prov}{mark}")
        fe = out["feasibility"][a]
        ok = [n for n, e in fe["by_n"].items() if e["status"] == "feasible"]
        none = [n for n, e in fe["by_n"].items() if e["status"] == "no DEV cost"]
        t3s = (f"feasible n <= {max(map(int, ok))}" if ok else "no feasible n")
        if fe["first_infeasible_n"] is not None:
            t3s += f"; INFEASIBLE from n {fe['first_infeasible_n']} (DEV cost {_f(fe['first_infeasible_cost'])})"
        if none:
            t3s += f"; no DEV cost at n {', '.join(none)}"
        nig = [n for n, e in fe["by_n"].items() if e["status"] == "not in grid"]
        if nig:
            t3s += f"; not in grid at n {', '.join(nig)} (spec max_n)"
        L.append(f"| {a} | " + " | ".join(cells) + f" | {t3s} |")
    L += ["", "`-` = no record at that n (above the spec max_n: not in grid; else T3: a cost pilot decides if the n is "
          "in the arm's grid); * = some unit "
          "unconverted; + = some unit converted with a provisional (observational) factor. Pooled over worlds, "
          "regimes, lambdas, kappas (the kappa sweep is at n 1000) and roles (tune + measure)."]

    L += ["", "## Full table", "",
          "| arm | method | n | units (tune / measure) | non-ref | CPU-s mean | median | min | max | max x f_hi | "
          "max / budget | RSS med MB | RSS max MB | wall-s mean | wall / CPU | procs / vCPU | gen CPU-s | multi-thr "
          "| T3 |", "|---|" * 1 + "---|" * 18]
    for r in tab:
        c = r["cost_ref"]
        ro = r["roles"]
        roles = " / ".join(str(ro.get(k, 0)) for k in ("tune", "measure")) if set(ro) & {"tune", "measure"} else             ", ".join(f"{k} {v}" for k, v in ro.items())
        ppv = r.get("procs_per_vcpu")
        L.append(f"| {r['arm']} | {r['method']} | {r['n']} | {r['n_units']} ({roles}) | "
                 f"{'-' if r['share_non_ref'] is None else f'{100 * r['share_non_ref']:.0f} %'} | "
                 f"{_f(c['mean'])} | {_f(c['median'])} | {_f(c['min'])} | {_f(c['max'])} | "
                 f"{_f(r.get('cost_ref_hi_max'))} | "
                 f"{'-' if c['max'] is None else f'{100 * c['max'] / r['budget']:.2f} %'} | "
                 f"{_f(r['peak_rss_mb']['median'])} | {_f(r['peak_rss_mb']['max'])} | {_f(r['wall_s_mean'])} | "
                 f"{_f(r['wall_cpu_ratio_median'])} | {_f(ppv['median']) if ppv else '-'} | "
                 f"{_f(r['gen_cpu_s_mean'])} | {r['n_multi_thread']} | {r['t3']['status']} |")
    gpu = [r for r in tab if r["cost_field"] == "wall_s"]
    if gpu:
        L += ["", f"GPU arms (R-41; wall-s vs a Kaggle T4, never summed with CPU-s): "
                  f"{', '.join(sorted({r['arm'] for r in gpu}))}."]
    errs = [r for r in tab if r["n_errors"] or r["n_no_cost"] or r["n_infeasible"]]
    if errs:
        L += ["", "Units without a cost: " + "; ".join(
            f"{r['arm']} n {r['n']}: {r['n_errors']} error, {r['n_no_cost']} no cost, {r['n_infeasible']} infeasible"
            for r in errs)]
    mt = sum(r["n_multi_thread"] for r in tab)
    ct = [r for r in tab if r["n_contended"]]
    L += ["", f"Single-thread check: {mt} units with cpu_s > {MULTI_THREAD_RATIO} x wall_s + .05 s "
              "(more than one busy thread)."]
    L += ["", f"Contention check: {sum(r['n_contended'] for r in tab)} units with wall_s > {CONTENTION_RATIO} x "
              "cpu_s + .05 s (CPU-starved: more busy processes than cores; CPU-s on shared hyperthreads runs slower, "
              "so the speed factor must be measured under the same load)"
          + ("; rows: " + ", ".join(f"{r['arm']} n {r['n']} ({r['n_contended']})" for r in ct[:20]) if ct else "")
          + "."]
    if out["spec_t3_fields"]:
        L += ["", "## T3 spec fields (for the freeze spec)", "", "```json",
              json.dumps(out["spec_t3_fields"], indent=1), "```"]
    return "\n".join(L) + "\n"


# ================================================================================================ plan / calib
def calibration_plan(recs: list[dict], ref: str, fields: dict[str, str] | None = None) -> dict:
    """T3 calibration units per arm: seeds 3_000_000-002 in the arm's 3 costliest (world, regime) at its largest
    measured n (cost = mean on the reference host if measured there, else raw mean over all hosts); in each (world,
    regime) the kappa .25 cell (if any) with the costliest lambda. Hosts = every non-reference host type the arm ran
    on (each needs the units on that host and on the reference)."""
    fields = fields or {}
    keys = {(s["arm"], s["world"], s["regime"], s["lam"], s["kappa"], s["n"], s["seed"]): s["key"] for s in recs}
    out = {}
    for a in sorted({s["arm"] for s in recs}):
        rs = [s for s in recs if s["arm"] == a and s["status"] == "ok"
              and unit_cost(s, fields.get(a, "cpu_s")) is not None]
        if not rs:
            continue
        nmax = max(s["n"] for s in rs)
        at = [s for s in rs if s["n"] == nmax]
        refm = any(s["host"] == ref for s in at)
        cm: dict[tuple, list[float]] = defaultdict(list)
        for s in at:
            if s["host"] == ref or not refm:
                cm[(s["world"], s["regime"], s["lam"], s["kappa"])].append(unit_cost(s, fields.get(a, "cpu_s")))
        best: dict[tuple, tuple] = {}
        for (w, rg, lam, kap), xs in cm.items():
            m = float(np.mean(xs))
            prim = kap is not None and abs(kap - PRIMARY_KAPPA) < 1e-9
            cand = (prim, m, lam, kap)
            if (w, rg) not in best or cand[:2] > best[(w, rg)][:2]:
                best[(w, rg)] = cand
        top = sorted(best.items(), key=lambda kv: -kv[1][1])[:3]
        units = []
        for (w, rg), (_, m, lam, kap) in top:
            for sd in CALIB_SEEDS:
                k = keys.get((a, w, rg, lam, kap, nmax, sd))
                units.append({"key": k or _key(a, w, rg, lam, kap, nmax, sd), "world": w, "regime": rg, "lam": lam,
                              "kappa": kap, "n": nmax, "seed": sd, "in_records": k is not None,
                              "cell_mean_cost": m})
        hosts = sorted({s["host"] for s in recs if s["arm"] == a} - {ref})
        out[a] = {"n": nmax, "cost_from": "reference host" if refm else "all hosts (no reference record at n)",
                  "units": units, "hosts_needing_factor": hosts}
    return {"version": VERSION, "ref_host": ref, "seeds": list(CALIB_SEEDS), "arms": out}


def _key(a, w, rg, lam, kap, n, sd) -> str:
    return "|".join([a, w, rg, *([f"lam{lam:g}"] if lam is not None else []), f"k{kap:g}", f"n{n}", f"s{sd}"])


def _session_factors(us: list[dict]) -> dict[str, float]:
    """f per host session (node) over the units it ran, when a host type has more than one session (spread, R-55)."""
    nodes = sorted({nd for u in us for nd in u["host_costs"]})
    if len(nodes) < 2:
        return {}
    out = {}
    for nd in nodes:
        un = [u for u in us if nd in u["host_costs"]]
        hc = sum(u["host_costs"][nd] for u in un)
        if hc > 0:
            out[nd] = sum(u["ref_cost"] for u in un) / hc
    return out


def calibrate(recs_all: list[dict], ref: str, plan: dict | None = None, fields: dict[str, str] | None = None) -> dict:
    """Speed factors from units with an ok record on the reference and on a host (``recs_all`` keeps every copy of a
    key). Per (arm, host): f = sum ref cost / sum host cost (T3), f_lo / f_hi = min / max of f and the per-n ratios
    (its error, R-55), dataset-hash agreement of the pairs. Several sessions (host.node) of one host type: host cost
    per unit = mean over sessions, and each session's f widens f_lo / f_hi (session spread). Per host, arm "*" (every arm without its own factor):
    pooled f over all arms, f_lo / f_hi = min / max over the arms' f (the anchors' spread, R-55). With ``plan`` only
    its units count. Source: 'r55_calibration_block' when every paired record is from the exp-c calib block,
    't3_calibration' with a plan, else 'paired_units'."""
    fields = fields or {}
    want = None if plan is None else {u["key"] for d in plan["arms"].values() for u in d["units"]}
    by_key: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))   # key -> host -> node -> cost
    meta = {}
    for s in recs_all:
        if s["status"] != "ok" or (want is not None and s["key"] not in want):
            continue
        c = unit_cost(s, fields.get(s["arm"], "cpu_s"))
        if c is not None:
            by_key[s["key"]][s["host"]].setdefault(s.get("node"), (c, s["dataset_sha256"], s["spec_name"]))
            meta[s["key"]] = (s["arm"], s["n"])
    pairs: dict[tuple, list[dict]] = defaultdict(list)
    specs = set()
    for k, hs in by_key.items():
        if ref in hs:
            rc, rsha, rsp = next(iter(hs[ref].values()))     # reference: first session seen
            for h, sess in hs.items():
                if h != ref:                                 # host: mean over its sessions (nodes)
                    cs = {str(nd): v[0] for nd, v in sess.items()}
                    shas = {v[1] for v in sess.values()}
                    pairs[(meta[k][0], h)].append({"key": k, "n": meta[k][1], "ref_cost": rc,
                                                   "host_cost": sum(cs.values()) / len(cs), "host_costs": cs,
                                                   "same_dataset": shas == {rsha} if rsha and None not in shas else None})
                    specs |= {v[2] for v in sess.values()} | {rsp}
    src = ("t3_calibration" if plan is not None else
           "r55_calibration_block" if specs == {"exp_c_calib"} else "paired_units")
    fac: dict[str, dict] = defaultdict(dict)
    for (a, h), us in sorted(pairs.items()):
        sh = sum(u["host_cost"] for u in us)
        if sh <= 0:
            continue
        f = sum(u["ref_cost"] for u in us) / sh
        per_n = {}
        for n in sorted({u["n"] for u in us}):
            un = [u for u in us if u["n"] == n]
            hn = sum(u["host_cost"] for u in un)
            if hn > 0:
                per_n[str(n)] = sum(u["ref_cost"] for u in un) / hn
        per_s = _session_factors(us)
        fac[a][h] = {"f": f, "f_lo": min([f, *per_n.values(), *per_s.values()]),
                     "f_hi": max([f, *per_n.values(), *per_s.values()]), "f_by_n": per_n,
                     "f_by_session": per_s, "source": src, "n_units": len(us),
                     "n_same_dataset": sum(u["same_dataset"] is True for u in us),
                     "n_different_dataset": sum(u["same_dataset"] is False for u in us), "units": us}
    for h in sorted({h for (_, h) in pairs}):
        arms = {a: d[h] for a, d in fac.items() if h in d and a != "*"}
        if not arms:
            continue
        us = [u for (a, hh), x in pairs.items() if hh == h for u in x]
        sh = sum(u["host_cost"] for u in us)
        per_s = _session_factors(us)
        fs = [d["f"] for d in arms.values()] + list(per_s.values())
        fac["*"][h] = {"f": sum(u["ref_cost"] for u in us) / sh, "f_lo": min(fs), "f_hi": max(fs),
                       "f_by_session": per_s, "source": f"{src} (pooled anchors)", "n_units": len(us),
                       "arms": sorted(arms)}
    missing = []
    if plan is not None:
        for a, d in plan["arms"].items():
            for h in d["hosts_needing_factor"]:
                if h not in fac.get(a, {}):
                    missing.append({"arm": a, "host": h})
                elif fac[a][h]["n_units"] < len(d["units"]):
                    fac[a][h]["incomplete"] = f"{fac[a][h]['n_units']} of {len(d['units'])} plan units paired"
    return {"version": VERSION, "ref_host": ref, "source": src, "factors": dict(fac), "missing": missing}


# ================================================================================================ cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("table", "plan", "calib"):
        p = sub.add_parser(name)
        p.add_argument("--records", nargs="+", required=True)
        p.add_argument("--spec", nargs="*", default=[], help="spec(s) for arm budget fields (GPU arms: budget_wall_s)")
        p.add_argument("--ref-host", default=REF_HOST)
        if name == "table":
            p.add_argument("--factors", default=None, help="calib output (T3 speed factors)")
            p.add_argument("--no-observational", action="store_true",
                           help="no provisional within-cell factors (EVAL / freeze: T3 factors only)")
            p.add_argument("--budget", type=float, default=BUDGET_CPU_S)
            p.add_argument("--out-dir", required=True)
            p.add_argument("--title", default="DEV")
        elif name == "plan":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--plan", default=None)
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs, info = load(a.records)
    arm_specs = load_specs(a.spec)
    fields = {k: cost_field(v) for k, v in arm_specs.items()}
    if a.cmd == "table":
        out = build(recs, info, ref=a.ref_host, budget=a.budget, arm_specs=arm_specs,
                    factors=load_factor_file(a.factors), observational=not a.no_observational)
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "exp_c_runtime.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1, allow_nan=False)
        with open(os.path.join(a.out_dir, "EXP_C_RUNTIME.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_md(out, a.title))
        inf = [x for x, d in out["feasibility"].items() if d["first_infeasible_n"] is not None]
        print(f"[exp-c] {len(recs)} units, {len(out['table'])} arm x n rows, factors {len(out['factors'])}, "
              f"T3-infeasible arms: {inf or 'none'} -> {a.out_dir}")
    elif a.cmd == "plan":
        out = calibration_plan(recs, a.ref_host, fields)
        json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print(f"[exp-c] plan: {sum(len(d['units']) for d in out['arms'].values())} units over "
              f"{len(out['arms'])} arms -> {a.out}")
    else:
        allc = [c for k, cs in info["copies"].items() for c in cs] + [s for s in recs if s["key"] not in info["copies"]]
        plan = json.load(open(a.plan, encoding="utf-8")) if a.plan else None
        out = calibrate(allc, a.ref_host, plan, fields)
        json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print(f"[exp-c] calib ({out['source']}): "
              + (", ".join(f"{arm}@{h.split('|')[0]} {v['f']:.3f} ({v['n_units']})"
                           for arm, hs in out["factors"].items() for h, v in hs.items())
                 or "no unit with an ok record on both the reference and another host")
              + (f"; missing {len(out['missing'])}" if out["missing"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
