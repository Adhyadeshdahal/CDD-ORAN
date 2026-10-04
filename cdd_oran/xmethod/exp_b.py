"""Experiment B driver: the final Study A method set on the frozen E6-P v4 data, DESCRIPTIVE (docs/xmethod/EXP_B.md).

    uv run python -m cdd_oran.xmethod.exp_b prep     --eval SRC --placebo SRC [--dev SRC] --out DIR     (cloud)
    uv run python -m cdd_oran.xmethod.exp_b counts   --spec SPEC (--tables DIR | --v4-analysis analysis_v4.json) --out F
    uv run python -m cdd_oran.xmethod.exp_b list     --spec SPEC
    uv run python -m cdd_oran.xmethod.exp_b run      --spec SPEC --tables DIR --part i/P --out res_i.jsonl
    uv run python -m cdd_oran.xmethod.exp_b merge    --spec SPEC --inputs 'runs/x/**/*.jsonl' --out merged.jsonl.gz
    uv run python -m cdd_oran.xmethod.exp_b table    --spec SPEC --merged merged.jsonl.gz --gt GT.json --out DIR
    uv run python -m cdd_oran.xmethod.exp_b project  --spec SPEC --dev-costs costs.json --out proj.json
    uv run python -m cdd_oran.xmethod.exp_b kaggle | colab --spec SPEC --name NAME [--parts P] [--part-set i,j] [--dry-run]

Machinery reused from ``cdd_oran.xmethod.campaign`` (xm/dev-runs; NOT forked): its ``Unit`` / job keys, ``partition``,
``run_units`` (forked child per unit under RLIMIT_CPU = the R-13 budget, infeasible records, resumable, provenance
stamps), ``merge_index`` (ok > infeasible > error, one platform per dataset), the lock install / pin check of cloud
sessions (``write_lock``, ``_setup_cmd``) and the seed-cluster bootstrap. Only the dataset source differs: a unit is
(arm, family, slice) of the frozen v4 data (``e6_bridge``), handed to ``run_units`` through ``campaign``'s dataset hook
(the module-level ``generate_dataset`` name, rebound for the run; campaign has no provider parameter).

A UNIT key = campaign's: "<arm>|E6|<family>|k0|n<units>|s<dataset seed>", dataset seed = e6_bridge.dataset_seed(v4
split, family) (60 000 000 + 10 split + family index: pooled 0, s60_i 20 + i, s120_k 40 + k, s300_m 60 + m, placebo
logs 9). Role "measure" (eval slices) or "placebo_log". ``n`` = the dataset's unit count (from the counts file), so
campaign's T3 propagation (an arm over budget at n is not run at n' >= n) means rows, as in Study A.
Nothing here reads or writes Study A seeds; nothing frozen is modified (the v4 JSONL is read through the v4 loaders).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict

import numpy as np

from cdd_oran.xmethod import e6_bridge as B

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPB_VERSION = "xm-expb/1"
ROLE_OF_KIND = {"placebo": "placebo_log"}
ALPHA = 0.05
TAU_LEVEL = 0.05
V4_SOURCES = {"eval": "e6p-disc-v4ev-1-a,e6p-disc-v4ev-1-b", "placebo": "e6p-disc-v4plc-1-a",
              "dev": "e6p-disc-v4dev-1-a"}                   # <V4_OWNER>/<slug> Kaggle kernels (read-only sources)
V4_OWNER = "bishalpanta"                                     # kaggle_job --sources needs owner/slug; mounted at $JOB_SRC/<slug>


def _campaign():
    """cdd_oran.xmethod.campaign (xm/dev-runs). Until that file is in this branch (status exp-b.md Q1), a local run
    may point XM_CAMPAIGN_FILE at a checkout's campaign.py (loaded read-only under the same module name)."""
    try:
        from cdd_oran.xmethod import campaign as C
        return C
    except ImportError:
        path = os.environ.get("XM_CAMPAIGN_FILE")
        if not path:
            raise
    import importlib.util
    name = "cdd_oran.xmethod.campaign"
    if name not in sys.modules:
        sp = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
    return sys.modules[name]


# ================================================================================================ spec / units
def load_spec(path: str) -> dict:
    """A spec, or a sub-spec ``{"base": <repo path of the full spec>, "slice_names" | "exclude_slices": [...]}``
    whose other fields override the base (same arms, so the same unit keys)."""
    spec = json.load(open(path, encoding="utf-8"))
    if spec.get("base"):
        base = load_spec(os.path.join(ROOT, *spec["base"].split("/")))
        spec = {**base, **{k: v for k, v in spec.items() if k != "base"}, "base_spec": spec["base"]}
    for a, d in spec["arms"].items():
        if "|" in a or not d.get("ref") or d.get("declare") not in ("by", "tau"):
            raise ValueError(f"arm {a!r}: needs 'ref', 'declare' in (by, tau), no '|'")
    bad = set(spec.get("families", B.FAMILIES)) - set(B.FAMILIES)
    if bad:
        raise ValueError(f"unknown families {bad}")
    return spec


def campaign_spec(spec: dict) -> dict:
    """The dict campaign.run_units reads: arms, budget, name; no blocks (never EVAL mode)."""
    return {"name": spec["name"], "budget_cpu_s": spec.get("budget_cpu_s"), "arms": spec["arms"], "blocks": []}


def slices(spec: dict) -> list[dict]:
    """The spec's slices: kinds ``slices``, optionally only ``slice_names`` and minus ``exclude_slices`` (pilot /
    remainder specs share keys with the full spec, so their records merge into it)."""
    out = B.slice_plan(tuple(spec["slices"]))
    if spec.get("slice_names"):
        out = [s for s in out if s["name"] in set(spec["slice_names"])]
    return [s for s in out if s["name"] not in set(spec.get("exclude_slices", []))]


def seed_index(spec: dict) -> dict[int, tuple[dict, str]]:
    return {B.dataset_seed(s["split"], f): (s, f) for s in slices(spec) for f in spec.get("families", B.FAMILIES)}


def load_counts(path: str) -> dict[str, dict[str, int]]:
    return json.load(open(path, encoding="utf-8"))["counts"]


def expand(spec: dict, counts: dict) -> list:
    """Every unit of the spec (dataset-major, deduplicated). Arms restricted by "kinds" / "families" / "max_n"."""
    C = _campaign()
    out = {}
    for s in slices(spec):
        for f in spec.get("families", B.FAMILIES):
            n = int(counts[s["name"]][f])
            if n == 0:                                           # no unit of f in the slice (synthetic tests only)
                continue
            for a, d in spec["arms"].items():
                if "kinds" in d and s["kind"] not in d["kinds"]:
                    continue
                if "families" in d and f not in d["families"]:
                    continue
                if "max_n" in d and n > int(d["max_n"]):
                    continue
                u = C.Unit(B.WORLD, f, None, n, 0.0, B.dataset_seed(s["split"], f), a,
                           ROLE_OF_KIND.get(s["kind"], "measure"))
                out.setdefault(u.key, u)
    return sorted(out.values())


def counts_from_tables(spec: dict, tables: str) -> dict:
    tabs = {}
    out = {}
    for s in slices(spec):
        t = tabs.setdefault(s["stage"], B.Table(os.path.join(tables, f"e6units_{s['stage']}.npz")))
        out[s["name"]] = {f: int(len(t.rows(f, s["seeds"]))) for f in B.FAMILIES}
    return out


def counts_from_v4(spec: dict, analysis: str) -> dict:
    """Unit counts as the stored v4 analysis reports them (same H 90 / H_pre 90 unit set); planning only."""
    a = json.load(open(analysis, encoding="utf-8"))
    out = {n: dict(v["units"]["by_family"]) for n, v in a["slices_result"].items()}
    pu = a.get("placebo_units") or {}
    if isinstance(pu, dict) and "by_family" in pu:
        out["placebo"] = dict(pu["by_family"])
    want = {s["name"] for s in slices(spec)}
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"v4 analysis has no unit counts for {missing}")
    return {k: v for k, v in out.items() if k in want}


# ================================================================================================ run
class Loader:
    """campaign's dataset hook: (world, regime, n, seed, ...) -> (Dataset, Truth) of the frozen v4 data."""

    def __init__(self, spec: dict, tables: str, gt_path: str):
        self.idx = seed_index(spec)
        self.tables = tables
        self.gt = B.load_gt(gt_path)
        self._tabs: dict[str, B.Table] = {}

    def __call__(self, world, regime, n, seed, lam=None, kappa=None):
        s, f = self.idx[int(seed)]
        if world != B.WORLD or regime != f:
            raise ValueError(f"unit ({world}, {regime}) does not match dataset seed {seed} ({f})")
        t = self._tabs.get(s["stage"])
        if t is None:
            t = self._tabs[s["stage"]] = B.Table(os.path.join(self.tables, f"e6units_{s['stage']}.npz"))
        ds = B.make_dataset(t, f, s["seeds"], s["split"], s["name"])
        if ds.n != int(n):
            raise ValueError(f"{s['name']} {f}: {ds.n} units, the counts file says {n}")
        return ds, B.truth_from_gt(self.gt, f, placebo_log=s["kind"] == "placebo")


def run(spec: dict, tables: str, part: int, parts: int, out: str, budget=None, cost_table=None,
        registry=None, isolate=None, spec_file_sha256=None) -> int:
    C = _campaign()
    units = C.partition(expand(spec, load_counts(spec["counts"])), parts, cost_table)[part]
    loader = Loader(spec, tables, spec["gt"])
    old = C.generate_dataset
    C.generate_dataset = loader                                  # campaign's dataset hook (see module doc)
    try:
        return C.run_units(campaign_spec(spec), units, out,
                           budget if budget is not None else spec.get("budget_cpu_s"), isolate,
                           spec_file_sha256=spec_file_sha256, registry=registry)
    finally:
        C.generate_dataset = old


def merge(spec: dict, inputs: list[str], out: str) -> dict:
    """campaign.merge_index rules (ok > infeasible > error; one platform per dataset where possible), expected
    keys = this spec's units."""
    import gzip
    C = _campaign()
    best, stats = C.merge_index(inputs, lambda r: (r.get("run_mode") or {}).get("mode", "dev") == "dev")
    want = {u.key for u in expand(spec, load_counts(spec["counts"]))}
    by = defaultdict(list)
    commits, dirty = set(), 0
    fhs = [open(p, "rb") for p in inputs]
    op = gzip.open if out.endswith(".gz") else open
    try:
        with op(out, "wt", encoding="utf-8", newline="\n") as fo:
            for k in sorted(best):
                r = C._read_entry(fhs, best[k])
                fo.write(json.dumps(r, allow_nan=False) + "\n")
                by[C._status(r)].append(k)
                commits.add(str((r.get("code") or {}).get("commit")))
                dirty += (r.get("code") or {}).get("dirty") is not False
    finally:
        for fh in fhs:
            fh.close()
    summ = {"n_expected": len(want), "n_records": len(best), "n_ok": len(by["ok"]),
            "missing": sorted(want - set(best)), "errors": sorted(by["error"]), "infeasible": sorted(by["infeasible"]),
            "unexpected": sorted(set(best) - want), **stats, "commits": sorted(commits), "dirty_records": dirty}
    json.dump(summ, open((out[:-3] if out.endswith(".gz") else out) + ".summary.json", "w"), indent=1)
    return summ


# ================================================================================================ table
def conformal_tau(scores, level: float = TAU_LEVEL) -> float | None:
    """R-29 rule: the ceil((M+1)(1-level))-th smallest of M placebo scores (the largest if beyond M)."""
    v = np.sort(np.asarray([x for x in scores if x is not None and np.isfinite(x)], float))
    if not len(v):
        return None
    k = math.ceil((len(v) + 1) * (1 - level))
    return float(v[min(k, len(v)) - 1])


def _auroc(pos, neg) -> float | None:
    pos = [x for x in pos if x is not None and np.isfinite(x)]
    neg = [x for x in neg if x is not None and np.isfinite(x)]
    if not pos or not neg:
        return None
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _sharp_proxy(gt: dict) -> set:
    """Experiment D set B: GT-NULL cells whose GT 95 % CI contains 0 (closest available proxy for a true null)."""
    return {h for h, c in gt.items() if c["status"] == "NULL" and c["ci"][0] <= 0.0 <= c["ci"][1]}


def table(spec: dict, merged: str, gt_path: str, out_dir: str, v4_analysis: str | None = None) -> dict:
    """Descriptive tables (EXP_B.md section 5). Per (arm, slice kind), pooled over the 4 family datasets of a slice:
    recall on GT-TRUE (29 hyp), rejection rates on GT-NULL (21; equivalence label) and on set B (sharp-null proxy),
    the P_placebo column (60 per slice), sign accuracy, AUROC (score, TRUE vs NULL); every arm under each rule it
    has: "by" (the adapter's BY declaration, p arms), "raw" (p <= .05 per hypothesis, p arms), "tau" (score >
    conformal placebo tau: leave-one-slice-out over the other slices of the kind, same family, M = 15 x their count;
    pooled / placebo logs: the dataset's own 15 placebo scores, labelled "in-sample"). Placebo logs: every real
    hypothesis is an exact sharp null. CIs: cluster bootstrap over slices (campaign.cluster_ci)."""
    C = _campaign()
    gt = B.load_gt(gt_path)
    sharp = _sharp_proxy(gt)
    idx = seed_index(spec)
    recs = defaultdict(dict)                                     # (arm, slice name) -> {family: record}
    status = defaultdict(lambda: defaultdict(int))
    for r in C.iter_jsonl(merged):
        s, f = idx[int(r["job"]["seed"])]
        arm = r.get("arm") or r["key"].split("|")[0]
        status[arm][C._status(r)] += 1
        if C._status(r) == "ok":
            recs[(arm, s["name"])][f] = r
    kinds = defaultdict(list)
    for s in slices(spec):
        kinds[s["kind"]].append(s["name"])

    def hyp_of(e):
        if e["source"] == B.PLACEBO:
            return None
        r_, k_ = e["target"][2:].split("_", 1)
        return (e["source"][2:], r_, k_)

    def score(e):
        x = e.get("score")
        return float(x) if x is not None and np.isfinite(x) else None

    tau_of = {}                                                  # (arm, slice, family) -> (tau, how)
    for arm in spec["arms"]:
        for kind, names in kinds.items():
            for name in names:
                for f in spec.get("families", B.FAMILIES):
                    others = [n for n in names if n != name] if kind not in ("pooled", "placebo") else [name]
                    pl = [score(e) for n in others for e in recs.get((arm, n), {}).get(f, {}).get("edges", [])
                          if e["source"] == B.PLACEBO]
                    tau_of[(arm, name, f)] = (conformal_tau(pl), "loso" if others != [name] else "in-sample")
    rows = {}
    for arm, d in spec["arms"].items():
        rules = (["by", "raw", "tau"] if d["declare"] == "by" else ["tau"]) + (
            ["fixed"] if d.get("fixed_threshold") is not None else [])
        for kind, names in kinds.items():
            for rule in rules:
                per = {k: [[], []] for k in ("rec", "null", "sharp", "plac", "plac_log")}
                sign_ok = sign_n = 0
                pos, neg = [], []
                n_ds = 0
                for name in names:
                    h = {k: [0, 0] for k in per}
                    for f in spec.get("families", B.FAMILIES):
                        r = recs.get((arm, name), {}).get(f)
                        if r is None:
                            continue
                        n_ds += 1
                        nt = set(map(str, (r.get("notes") or {}).get("not_testable_edges", []) or []))
                        tau, _ = tau_of[(arm, name, f)]
                        for e in r.get("edges", []):
                            sc, p = score(e), e.get("p")
                            if f"{e['source']}->{e['target']}" in nt:
                                dec = False
                            elif rule == "by":
                                dec = bool(e.get("declared"))
                            elif rule == "raw":
                                dec = p is not None and float(p) <= ALPHA
                            elif rule == "fixed":
                                dec = sc is not None and sc >= float(d["fixed_threshold"])
                            else:
                                dec = sc is not None and tau is not None and sc > tau
                            hy = hyp_of(e)
                            if kind == "placebo":
                                h["plac_log" if hy else "plac"][0] += dec
                                h["plac_log" if hy else "plac"][1] += 1
                                continue
                            if hy is None:
                                h["plac"][0] += dec
                                h["plac"][1] += 1
                                continue
                            st = gt[hy]["status"]
                            if st == "TRUE":
                                h["rec"][0] += dec
                                h["rec"][1] += 1
                                pos.append(sc)
                                if dec and e.get("sign"):
                                    sign_n += 1
                                    sign_ok += int(np.sign(e["sign"]) == gt[hy]["sign"])
                            elif st == "NULL":
                                h["null"][0] += dec
                                h["null"][1] += 1
                                neg.append(sc)
                                if hy in sharp:
                                    h["sharp"][0] += dec
                                    h["sharp"][1] += 1
                    for k in per:
                        per[k][0].append(h[k][0])
                        per[k][1].append(h[k][1])
                cell = {"n_slices": len(names), "n_datasets": n_ds}
                for k, (hits, cnt) in per.items():
                    if sum(cnt):
                        cell[k] = {"rate": sum(hits) / sum(cnt), "hits": int(sum(hits)), "n": int(sum(cnt)),
                                   "ci": C.cluster_ci(np.array(hits), np.array(cnt)) if len(names) > 1 else None}
                cell["sign_acc"] = sign_ok / sign_n if sign_n else None
                cell["sign_n"] = sign_n
                if rule in ("tau", "by") and kind != "placebo":
                    cell["auroc_true_vs_null"] = _auroc(pos, neg)
                if rule == "tau":
                    cell["tau_rule"] = sorted({tau_of[(arm, n, f)][1] for n in names
                                               for f in spec.get("families", B.FAMILIES)})
                rows[f"{arm}|{kind}|{rule}"] = cell
    ref = v4_reference(v4_analysis, gt, kinds) if v4_analysis else None
    out = {"schema": EXPB_VERSION, "gt": {"counts": dict(_count(gt)), "sharp_proxy": sorted("|".join(h) for h in sharp)},
           "status": {a: dict(v) for a, v in status.items()}, "cells": rows, "v4_frozen_reference": ref}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "exp_b_tables.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(out_dir, "EXP_B_TABLES.md"))
    return out


def _count(gt: dict) -> dict:
    c = defaultdict(int)
    for v in gt.values():
        c[v["status"]] += 1
    return c


def v4_reference(path: str, gt: dict, kinds: dict) -> dict:
    """The frozen v4 primary (loadsp_c + wby1s) per slice kind, recomputed from the stored hyp tables with the same
    definitions (recall on GT-TRUE, rate on GT-NULL): a reference row, nothing re-run."""
    a = json.load(open(path, encoding="utf-8"))
    out = {}
    for kind, names in kinds.items():
        tp = nt = fp = nn = 0
        for name in names:
            hyp = (a["placebo_hyp"] if kind == "placebo" else a["slices_result"].get(name, {}).get("hyp", {}))
            for h, o in hyp.items():
                dec = bool((o.get("primary") or {}).get("declared"))
                st = "NULL" if kind == "placebo" else gt[tuple(h.split("|"))]["status"]
                if st == "TRUE":
                    tp, nt = tp + dec, nt + 1
                elif st == "NULL":
                    fp, nn = fp + dec, nn + 1
        out[kind] = {"recall": tp / nt if nt else None, "null_rate": fp / nn if nn else None, "n_true": nt,
                     "n_null": nn}
    return out


def write_md(t: dict, path: str) -> None:
    nf = len(B.FAMILIES)
    L = ["# Experiment B tables (descriptive; frozen E6 v4 data)", "",
         f"GT: {t['gt']['counts']}; sharp-null proxy (set B): {len(t['gt']['sharp_proxy'])} hyp.", "",
         "Each rate: value (hits / n) [cluster-bootstrap CI over slices; few slices give narrow or zero-width CIs].",
         f"`ds` = datasets with an ok record out of slices x {nf} families; `*` = partial coverage (mscr runs only at",
         "n <= 1000, i.e. sleep; cdl units over the 7200 s budget are infeasible), so that row is not comparable with",
         "full-coverage rows.", "",
         "| arm | kind | rule | ds | recall TRUE | GT-NULL rate | set-B rate | P_placebo rate | placebo-log rate "
         "| sign acc | AUROC |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(c, k):
        v = c.get(k)
        if not v:
            return ""
        ci = v.get("ci")
        return (f"{v['rate']:.3f} ({v['hits']}/{v['n']})"
                + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else ""))

    for key, c in t["cells"].items():
        arm, kind, rule = key.split("|")
        want = c["n_slices"] * nf
        ds = f"{c['n_datasets']}/{want}" + ("*" if c["n_datasets"] < want else "")
        L.append(f"| {arm} | {kind} | {rule} | {ds} | {f(c, 'rec')} | {f(c, 'null')} | {f(c, 'sharp')} | "
                 f"{f(c, 'plac')} | {f(c, 'plac_log')} | "
                 f"{'' if c.get('sign_acc') is None else round(c['sign_acc'], 3)} | "
                 f"{'' if c.get('auroc_true_vs_null') is None else round(c['auroc_true_vs_null'], 3)} |")
    if t.get("v4_frozen_reference"):
        L += ["", "Frozen v4 primary (loadsp_c + wby1s, stored tables): " + json.dumps(t["v4_frozen_reference"])]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


# ================================================================================================ costs
REF_PLATFORM = "kaggle"                                         # the cost reference (R-51)


def costs(spec: dict, merged: str, factors: str, out_dir: str) -> dict:
    """Measured CPU-s per arm in the Kaggle reference (EXP_B.md s.6). Kaggle records count as measured. Records of
    another host are multiplied by that host's factor: the arm's own `f` when the factor file has that exact arm name,
    else the pooled `*` factor; no transfer across arm names. An infeasible unit counts at the budget (R-13)."""
    C = _campaign()
    fac = json.load(open(factors, encoding="utf-8"))["factors"]
    budget = float(spec["budget_cpu_s"])
    arms = defaultdict(lambda: {"n": 0, "ok": 0, "infeasible": 0, "ref_cpu_s": 0.0, "max_ref_cpu_s": 0.0,
                                "raw_cpu_s": defaultdict(float), "factor": {}})
    for r in C.iter_jsonl(merged):
        arm, st = r.get("arm") or r["key"].split("|")[0], C._status(r)
        host = r.get("host") or {}
        a = arms[arm]
        a["n"] += 1
        a[st] = a.get(st, 0) + 1
        raw = float(r.get("cpu_s") or 0.0)
        a["raw_cpu_s"][host.get("platform")] += raw
        if host.get("platform") == REF_PLATFORM:
            ref = raw
        else:
            hk = f"{host.get('platform')}|{host.get('cpu_model')}"
            src = arm if hk in fac.get(arm, {}) else "*"
            f = float(fac[src][hk]["f"])
            a["factor"][hk] = {"from": src, "f": f}
            ref = raw * f
        if st == "infeasible":
            ref = budget
        a["ref_cpu_s"] += ref
        a["max_ref_cpu_s"] = max(a["max_ref_cpu_s"], ref)
    res = {"schema": "xm-expb-costs/1", "reference": REF_PLATFORM, "factors_file": factors.replace("\\", "/"),
           "factors_sha256": C._sha_lf(open(factors, "rb").read()), "budget_cpu_s": budget,
           "arms": {k: {**v, "raw_cpu_s": dict(v["raw_cpu_s"])} for k, v in sorted(arms.items())}}
    res["total_ref_core_h"] = sum(v["ref_cpu_s"] for v in arms.values()) / 3600
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "exp_b_costs.json"), "w"), indent=1)
    L = ["# Experiment B measured cost (Kaggle reference CPU-s)", "",
         f"Factors: `{res['factors_file']}` (sha256 {res['factors_sha256'][:12]}...); infeasible units at the "
         f"{budget:.0f} s budget. Total {res['total_ref_core_h']:.1f} ref core-h.", "",
         "| arm | units | ok | infeasible | ref core-h | max ref CPU-s / unit | non-ref factor |", "|---|---|---|---|---|---|---|"]
    for k, v in res["arms"].items():
        fs = "; ".join(f"{x['from']} {x['f']:.3f}" for x in v["factor"].values())
        L.append(f"| {k} | {v['n']} | {v['ok']} | {v['infeasible']} | {v['ref_cpu_s'] / 3600:.2f} | "
                 f"{v['max_ref_cpu_s']:.0f} | {fs} |")
    open(os.path.join(out_dir, "EXP_B_COSTS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return res


# ================================================================================================ projection
def project(spec: dict, dev_costs: str, out: str | None = None) -> dict:
    """Cost projection: per arm, the Study A DEV CPU-s per dataset (max over worlds of the cell mean) interpolated
    log-log in n and extrapolated past 24 000 with the last segment's measured slope (clipped to [0, 2]), times the arm's E6 size factor
    (spec "size_factor", default 1: E6 datasets have 30 candidates, ~62-68 conditioners, 15 targets). Returns the
    per-unit costs (a campaign cost table keyed "arm|E6|family|n<n>") and totals."""
    dc = json.load(open(dev_costs, encoding="utf-8"))
    units = expand(spec, load_counts(spec["counts"]))
    table_, per_arm = {}, defaultdict(lambda: [0.0, 0, 0.0])
    over = []
    for u in units:
        d = spec["arms"][u.arm]
        src = d.get("cost_from", u.arm)
        pts = sorted((int(n), max(v[0] for v in w.values())) for n, w in dc[src].items())
        x = np.log([p[0] for p in pts])
        y = np.log([max(p[1], 0.05) for p in pts])
        ln = math.log(u.n)
        if len(pts) == 1:
            c = math.exp(y[0]) * u.n / pts[0][0]
        elif ln <= x[-1]:
            c = math.exp(np.interp(ln, x, y))
        else:
            slope = min(max((y[-1] - y[-2]) / (x[-1] - x[-2]), 0.0), 2.0)   # measured last segment (cdl: capped steps)
            c = math.exp(y[-1] + slope * (ln - x[-1]))
        c *= float(d.get("size_factor", 1.0))
        table_[f"{u.arm}|{u.world}|{u.regime}|n{u.n}"] = c
        pa = per_arm[u.arm]
        pa[0] += c
        pa[1] += 1
        pa[2] = max(pa[2], c)
        if spec.get("budget_cpu_s") and c > float(spec["budget_cpu_s"]):
            over.append(u.key)
    res = {"n_units": len(units), "total_core_h": sum(v[0] for v in per_arm.values()) / 3600.0,
           "per_arm": {a: {"core_h": v[0] / 3600.0, "units": v[1], "max_unit_s": v[2]} for a, v in per_arm.items()},
           "projected_over_budget": over, "cost_table": table_}
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


# ================================================================================================ cloud
def _run_cmd(spec_rel: str, tables: str, ids, parts: int, out_dir: str, tag: str, cost_rel, wall_s: int,
             code: dict, lock_rel: str) -> str:
    thr = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    env = (f"XM_PLATFORM={tag} XM_CODE_COMMIT={code['commit']} XM_CODE_DIRTY={int(bool(code['dirty']))} "
           f"XM_LOCK_FILE={lock_rel} {thr}")
    ct = f" --cost-table {cost_rel}" if cost_rel else ""
    # R-55: at most one process per vCPU (campaign._cloud_cmd's semaphore); later parts start as earlier ones end
    pre = ('NP=$(python -c "from cdd_oran.xmethod.campaign import cpu_quota; print(cpu_quota())"); '
           f'echo "vCPU quota $NP" > {out_dir}/cpu_quota.txt; ')
    gate = "while [ $(jobs -rp | wc -l) -ge $NP ]; do wait -n; done; "
    runs = " ".join(f"{gate}{env} timeout {wall_s} python -u -m cdd_oran.xmethod.exp_b run --spec {spec_rel} --tables "
                    f"{tables} --part {i}/{parts} --out {out_dir}/res_{i}.jsonl{ct} --registry {out_dir}/registry.jsonl"
                    f" > {out_dir}/log_{i}.txt 2>&1 &" for i in ids)
    return f"mkdir -p {out_dir} && {pre}{runs} wait; tail -n 3 {out_dir}/log_*.txt"


def cloud(a, spec: dict) -> int:
    """Kaggle: the v4 collection kernels are mounted as sources; the session first builds the unit tables (prep,
    a few minutes), then runs its parts. Colab: the tables come from a finished Kaggle prep (``--tables`` = a repo
    path of the pulled npz files, bundled)."""
    C = _campaign()
    code = C.code_info()
    spec_rel = os.path.relpath(os.path.abspath(a.spec), ROOT).replace("\\", "/")
    ids = [int(x) for x in a.part_set.split(",")] if a.part_set else list(range(a.parts))
    cost_rel = os.path.relpath(os.path.abspath(a.cost_table), ROOT).replace("\\", "/") if a.cost_table else None
    C.write_lock()
    torch_v = C.lock_pins(C.LOCK_PATH).get("torch")
    torch_v = torch_v.split("+")[0] if torch_v else None
    bundle = ([C.LOCK_REL, C.LOCK_INSTALL_REL, *C.BUNDLE_DATA, spec["counts"], spec["gt"]]
              + ([spec["base_spec"]] if spec.get("base_spec") else []) + ([cost_rel] if cost_rel else []))
    if a.cmd == "kaggle":
        out_dir = "$JOB_OUT"
        src = {k: ",".join(f"$JOB_SRC/{s}" for s in v.split(",")) for k, v in V4_SOURCES.items()}
        prep = (f"python -u -m cdd_oran.xmethod.exp_b prep --eval {src['eval']} --placebo {src['placebo']} "
                f"--out {out_dir}/tables --workers 4 > {out_dir}/prep.log 2>&1; rm -rf {out_dir}/tables/cache; "
                f"python -u -m cdd_oran.xmethod.exp_b counts --spec {spec_rel} --tables {out_dir}/tables "
                f"--check {spec['counts']} --out {out_dir}/counts_check.json >> {out_dir}/prep.log 2>&1; ")
        cmd = (C._setup_cmd(out_dir, False, torch_v, a.venv_python, False) + prep
               + _run_cmd(spec_rel, f"{out_dir}/tables", ids, a.parts, out_dir, "kaggle", cost_rel,
                          a.wall_s or 11 * 3600, code, C.LOCK_REL))
        argv = [sys.executable, os.path.join(ROOT, "scratchpad", "e6_dev", "kaggle_job.py"), "launch", a.name,
                "--cmd", cmd, "--paths", spec_rel, *bundle, "--sources",
                ",".join(f"{V4_OWNER}/{s}" for v in V4_SOURCES.values() for s in v.split(",")
                         if s != "e6p-disc-v4dev-1-a"),
                "--internet", "--pin", "match"]
    else:
        if not a.tables:
            raise SystemExit("colab needs --tables <repo path of the pulled e6units_*.npz>")
        out_dir = "xm_out"
        tabs = a.tables.replace("\\", "/")
        cmd = C._setup_cmd(out_dir, False, torch_v, a.venv_python, False) + _run_cmd(
            spec_rel, tabs, ids, a.parts, out_dir, "colab", cost_rel, a.wall_s or 3 * 3600, code, C.LOCK_REL)
        argv = [sys.executable, os.path.join(ROOT, "scratchpad", "e6_dev", "colab_run.py"), "job", a.name,
                "--paths", "cdd_oran", spec_rel, *bundle, tabs, "--cmd", cmd, "--out-dir", out_dir, "--threads", "1"]
    if not a.dry_run:
        subprocess.run(argv, cwd=ROOT, check=True)
    print(" ".join(shlex.quote(x) for x in argv))
    return 0


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cdd_oran.xmethod.exp_b")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--eval", required=True)
    p.add_argument("--placebo", required=True)
    p.add_argument("--dev", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("counts")
    p.add_argument("--spec", required=True)
    p.add_argument("--tables", default=None)
    p.add_argument("--v4-analysis", dest="v4", default=None)
    p.add_argument("--check", default=None, help="compare with this counts file (exit 3 on a mismatch)")
    p.add_argument("--out", required=True)
    for c in ("list", "run", "merge", "table", "costs", "project", "kaggle", "colab"):
        p = sub.add_parser(c)
        p.add_argument("--spec", required=True)
        if c == "run":
            p.add_argument("--tables", required=True)
            p.add_argument("--part", default="0/1")
            p.add_argument("--out", required=True)
            p.add_argument("--budget", type=float, default=None)
            p.add_argument("--cost-table", default=None)
            p.add_argument("--registry", default=None)
            p.add_argument("--no-isolate", action="store_true")
        if c == "merge":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--out", required=True)
        if c == "table":
            p.add_argument("--merged", required=True)
            p.add_argument("--gt", default=None)
            p.add_argument("--v4-analysis", dest="v4", default=None)
            p.add_argument("--out", required=True)
        if c == "costs":
            p.add_argument("--merged", required=True)
            p.add_argument("--factors", required=True, help="non-reference host factors (EXP_B.md s.6)")
            p.add_argument("--out", required=True)
        if c == "project":
            p.add_argument("--dev-costs", required=True)
            p.add_argument("--out", default=None)
        if c in ("kaggle", "colab"):
            p.add_argument("--name", required=True)
            p.add_argument("--parts", type=int, default=4)
            p.add_argument("--part-set", default=None)
            p.add_argument("--cost-table", default=None)
            p.add_argument("--tables", default=None)
            p.add_argument("--wall-s", type=int, default=None)
            p.add_argument("--venv-python", default="3.12.14")  # = P0 / p1a (R-57: uv "3.12" is 3.12.15 now)
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        srcs = {"eval": a.eval, "placebo": a.placebo, **({"dev": a.dev} if a.dev else {})}
        print(json.dumps(B.build_table(srcs, a.out, a.workers), indent=1))
        return 0
    spec = load_spec(a.spec)
    if a.cmd == "counts":
        c = counts_from_tables(spec, a.tables) if a.tables else counts_from_v4(spec, a.v4)
        res = {"schema": "xm-expb-counts/1", "source": "tables" if a.tables else "v4 analysis", "counts": c}
        rc = 0
        if a.check:
            ref = load_counts(a.check)
            bad = {k: [ref.get(k), v] for k, v in c.items() if ref.get(k) != v}
            res["check"] = {"against": a.check, "mismatches": bad}
            rc = 3 if bad else 0
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res.get("check", {"n_slices": len(c)})))
        return rc
    if a.cmd == "list":
        units = expand(spec, load_counts(spec["counts"]))
        by = defaultdict(int)
        for u in units:
            by[u.arm] += 1
        print(json.dumps({"n_units": len(units), "n_datasets": len({u.dataset for u in units}), "per_arm": by}, indent=1))
        return 0
    if a.cmd == "run":
        i, n = (int(x) for x in a.part.split("/"))
        ct = json.load(open(a.cost_table))["cost_table"] if a.cost_table else None
        C = _campaign()
        sha = C._sha_lf(open(a.spec, "rb").read())
        run(spec, a.tables, i, n, a.out, a.budget, ct, a.registry, False if a.no_isolate else None, sha)
        return 0
    if a.cmd == "merge":
        files = sorted({f for g in a.inputs for f in glob.glob(g, recursive=True)})
        print(json.dumps({k: v for k, v in merge(spec, files, a.out).items() if not isinstance(v, list) or len(v) < 20},
                         indent=1))
        return 0
    if a.cmd == "table":
        table(spec, a.merged, a.gt or spec["gt"], a.out, a.v4)
        return 0
    if a.cmd == "costs":
        r = costs(spec, a.merged, a.factors, a.out)
        print(json.dumps({"total_ref_core_h": r["total_ref_core_h"], "factors_sha256": r["factors_sha256"]}))
        return 0
    if a.cmd == "project":
        r = project(spec, a.dev_costs, a.out)
        print(json.dumps({k: v for k, v in r.items() if k != "cost_table"}, indent=1))
        return 0
    return cloud(a, spec)


if __name__ == "__main__":
    sys.exit(main())
