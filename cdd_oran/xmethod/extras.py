"""R-60 supplementary experiments X2 / X3 of Study A: EXPLORATORY, they never change the C1-C3 verdicts or the E6
headline. Declared in scratchpad/xmethod/EXTRAS_PROTOCOL.md before launch. A wrapper over the frozen campaign driver
and generator: no frozen file is edited, the overrides live in this process only (campaign's forked unit children
inherit them).

    uv run python -m cdd_oran.xmethod.extras specs                     # (re)write the declared specs + DEV cost agg
    uv run python -m cdd_oran.xmethod.extras projection                # cost projection per spec (DEV cost table)
    uv run python -m cdd_oran.xmethod.extras <campaign cmd> --spec scratchpad/xmethod/specs/extras/<s>.json [...]
    uv run python -m cdd_oran.xmethod.extras analyse --spec S [S ...] --merged M [M ...] --out DIR

``<campaign cmd>`` is any of campaign's list / run / merge / aggregate / project / kaggle / colab / vps, same
arguments; cloud sessions run ``python -m cdd_oran.xmethod.extras run`` instead of campaign's ``run``.

X2 (dither dose-response, C1 mechanism): the spec's ``extras.design`` = {"delta", "n_blocks"} replaces
generate.DITHER_DELTA / DITHER_BLOCKS while campaign generates a dataset. delta .10 / 20 blocks is the frozen design:
those datasets equal the frozen generator's bit-for-bit (test_xmethod_extras).
X3 (design misspecification, C2a assumption): the data stay as generated; an arm's ``told`` changes the design TOLD
to the method (a wrapper around runner.run_one; the method, its ref and config are the frozen arm's):
  {"width": c}  every dither design's law U(-w, w) told as U(-c w, c w) (realised random / fixed parts unchanged);
  {"shift": f}  every dither design's setpoint switch times told s = floor(f L + .5) rows late (L = n / n_blocks rows
                per block): told setpoint[t] = setpoint[max(t - s, 0)], told random part = action - told setpoint
                (rows whose told setpoint is the true one keep the true random part exactly);
  {"lam": c}    E4 R3: the logged-policy propensity of every logged design recomputed with lambda x c (the observed
                context Z unchanged); the i.i.d. P_placebo design is not touched.
Seeds: 3_200_000-3_200_199 only (XMETHOD_EXTRAS in docs/benchmark/SEED_REGISTRY.json). For an extras spec campaign's
DEV / EVAL seed guard is replaced by this block's guard and the run is DEV-mode; records carry ``integrity.extras``
(experiment, design, protocol sha), told arms also ``told`` and ``dataset_sha256_generated`` (the data as generated).
``merge`` and --skip-complete-from accept only records of the same spec (canonical sha): designs share unit keys.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import gzip
import importlib.util
import json
import math
import os
import sys

import numpy as np

from cdd_oran.xmethod import api
from cdd_oran.xmethod import campaign as C
from cdd_oran.xmethod import runner as R
from cdd_oran.xmethod.worlds import e4_logged
from cdd_oran.xmethod.worlds import generate as G

EXTRAS_VERSION = "xm-extras/1"
SEEDS = range(3_200_000, 3_200_200)                 # XMETHOD_EXTRAS (R-60)
X2_TUNE = [3_200_000, 3_200_019]
X2_MEASURE = [3_200_020, 3_200_059]
X3_MEASURE = [3_200_060, 3_200_119]                 # 3_200_120-199 unassigned (a new use needs a new declaration)
PROTOCOL_REL = "scratchpad/xmethod/EXTRAS_PROTOCOL.md"
SPEC_DIR_REL = "scratchpad/xmethod/specs/extras"
FROZEN_SPEC_REL = "scratchpad/xmethod/specs/eval/full.json"
COST_AGG_REL = f"{SPEC_DIR_REL}/dev_cost_agg.json"
DEV_AGGS = ("scratchpad/xmethod/results/dev/full/agg.json", "scratchpad/xmethod/results/dev/ci_c/agg.json",
            "scratchpad/xmethod/results/dev/pmrt_nl/agg.json")
EVAL_ANALYSIS_REL = "scratchpad/xmethod/eval_analysis.py"
EXPERIMENTS = ("X2", "X3")
TOLD_KEYS = ("width", "shift", "lam")
KAPPA = 0.25

# X2: arms, cells and designs (R-60); tune seeds only for the score-only (tau) arms, the p arms declare by BY
X2_ARMS = ("corr", "pc_eq", "notears", "shap_dag", "pcorr_native", "pcorr_eq", "pmrt_eq", "pmrt_nl_eq")
X2_WORLDS, X2_NS = ("E1", "E2", "E3"), (1000, 4000)
X2_DESIGNS = {"x2_d02": (0.02, 20), "x2_d05": (0.05, 20), "x2_d10": (0.10, 20), "x2_d20": (0.20, 20),
              "x2_b05": (0.10, 5), "x2_b80": (0.10, 80)}
# X3: arms, told variants (R-60); the exact (frozen) arms run in both blocks
X3_ARMS = ("pmrt_nl_eq", "pmrt_eq", "pmrt_r3", "pcorr_eq")
X3_DITHER = {"w050": {"width": 0.5}, "w080": {"width": 0.8}, "w125": {"width": 1.25}, "w200": {"width": 2.0},
             "s02": {"shift": 0.02}, "s05": {"shift": 0.05}}
X3_LOGGED = {"l050": {"lam": 0.5}, "l080": {"lam": 0.8}, "l125": {"lam": 1.25}, "l200": {"lam": 2.0}}
X3_N = 1000


def _path(rel: str) -> str:
    return os.path.join(R.ROOT, *rel.split("/"))


def protocol_sha256() -> str | None:
    """LF sha256 of EXTRAS_PROTOCOL.md where present (cloud bundles carry it via --paths)."""
    p = _path(PROTOCOL_REL)
    return C._sha_lf(open(p, "rb").read()) if os.path.exists(p) else None


# ================================================================================================ spec checks
def is_extras(spec: dict | None) -> bool:
    return isinstance((spec or {}).get("extras"), dict)


def seed_list(s) -> list[int]:
    """Seeds of an extras block (campaign's [lo, hi] / list convention): XMETHOD_EXTRAS only, anything else refused
    (the DEV block, the EVAL block and every other seed)."""
    if isinstance(s, str) or any(isinstance(x, str) for x in s):
        raise ValueError(f"seeds must be numbers: {s!r}")
    seeds = list(range(int(s[0]), int(s[1]) + 1)) if (len(s) == 2 and s[1] > s[0] + 1) else [int(x) for x in s]
    bad = [x for x in seeds if x not in SEEDS]
    if bad:
        raise ValueError(f"extras specs use seeds 3200000-3200199 only (XMETHOD_EXTRAS, R-60): {bad[:5]}")
    return seeds


def check_design(d: dict) -> tuple[float, int]:
    delta, nb = float(d["delta"]), int(d["n_blocks"])
    if not (math.isfinite(delta) and delta > 0.0 and G.DITHER_ENVELOPE + delta < 0.5):
        raise ValueError(f"dither delta must be in (0, {0.5 - G.DITHER_ENVELOPE}): {delta}")
    if nb < 1 or nb != d["n_blocks"]:
        raise ValueError(f"n_blocks must be an integer >= 1: {d['n_blocks']!r}")
    return delta, nb


def check_told(told: dict) -> tuple[str, float]:
    if not isinstance(told, dict) or len(told) != 1 or next(iter(told)) not in TOLD_KEYS:
        raise ValueError(f"told must be one of {TOLD_KEYS} -> factor: {told!r}")
    k, v = next(iter(told.items()))
    v = float(v)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"told factor must be > 0: {told!r}")
    return k, v


def validate(spec: dict) -> None:
    """An extras spec: experiment X2 (a design, no told arms) or X3 (told arms, no design), seeds in the block."""
    if not is_extras(spec):
        raise ValueError("not an extras spec (no 'extras' block)")
    ex = spec["extras"]
    if ex.get("experiment") not in EXPERIMENTS:
        raise ValueError(f"extras.experiment must be one of {EXPERIMENTS}")
    told = {a: d["told"] for a, d in spec["arms"].items() if d.get("told") is not None}
    if ex["experiment"] == "X2":
        if not ex.get("design") or told:
            raise ValueError("X2 spec: needs extras.design and no told arms")
        check_design(ex["design"])
    else:
        if ex.get("design"):
            raise ValueError("X3 spec: the data stay as generated (no extras.design)")
        for t in told.values():
            check_told(t)
    for b in spec.get("blocks", []):
        seed_list(b["seeds"])


# ================================================================================================ X2 generation
@contextlib.contextmanager
def dither_constants(delta: float, n_blocks: int):
    """generate.DITHER_DELTA / DITHER_BLOCKS replaced inside the block (restored after, also on error)."""
    old = G.DITHER_DELTA, G.DITHER_BLOCKS
    G.DITHER_DELTA, G.DITHER_BLOCKS = float(delta), int(n_blocks)
    try:
        yield
    finally:
        G.DITHER_DELTA, G.DITHER_BLOCKS = old


def generate(world: str, regime: str, n: int, seed: int, lam: float = 1.0, kappa: float | None = None, *,
             design: dict | None = None) -> tuple[api.Dataset, api.Truth]:
    """The frozen generator, under the X2 ``design`` (delta, n_blocks) when given (only R2 reads the constants)."""
    if design is None:
        return G.generate_dataset(world, regime, n, seed, lam=lam, kappa=kappa)
    delta, nb = check_design(design)
    if regime == "R2" and int(n) < nb:
        raise ValueError(f"n {n} < n_blocks {nb}")
    with dither_constants(delta, nb):
        return G.generate_dataset(world, regime, n, seed, lam=lam, kappa=kappa)


# ================================================================================================ X3 told design
def shift_rows(frac: float, n: int, n_blocks: int) -> int:
    """Rows of a switch-time shift of ``frac`` of a block (L = n / n_blocks rows), rounded half up."""
    return int(math.floor(float(frac) * int(n) / int(n_blocks) + 0.5))


def tell(ds: api.Dataset, told: dict) -> api.Dataset:
    """``ds`` with the design TOLD to the method changed by ``told`` (module doc); data, truth and every other field
    unchanged. Refuses a variant that changes no design of ``ds``."""
    k, v = check_told(told)
    designs = list(ds.designs)
    A = np.asarray(ds.X_action, float)
    changed = 0
    for j, d in enumerate(designs):
        if k == "width" and d.kind == "dither":
            designs[j] = dataclasses.replace(d, dist={**d.dist, "lo": d.dist["lo"] * v, "hi": d.dist["hi"] * v})
        elif k == "shift" and d.kind == "dither":
            s = shift_rows(v, ds.n, ds.meta["dither"]["n_blocks"])
            f = np.asarray(d.fixed_part, float)
            ft = np.concatenate([np.repeat(f[:1], s), f[:len(f) - s]]) if s else f.copy()
            rt = np.where(ft == f, np.asarray(d.random_part, float), A[:, j] - ft)
            designs[j] = dataclasses.replace(d, fixed_part=ft, random_part=rt)
        elif k == "lam" and d.kind == "logged":
            lam_t = float(ds.meta["lam"]) * v
            table = e4_logged.policy_probs(np.asarray(ds.context, float)[:, 0], lam_t)
            designs[j] = dataclasses.replace(d, dist={**d.dist, "lam": lam_t}, propensity=table)
        else:
            continue
        changed += 1
    if not changed:
        raise ValueError(f"told {told} changes no design of {ds.world} {ds.regime}")
    return dataclasses.replace(ds, designs=tuple(designs))


def _told_run_one(orig, told_of: dict[str, dict]):
    """runner.run_one with the arm's told design (by the unit key's arm); the record also keeps the hash of the data
    as generated. Untold arms: unchanged."""
    def run_one(method, ds, truth, config, gen_cpu_s, key):
        told = told_of.get(key.split("|", 1)[0])
        if not told:
            return orig(method, ds, truth, config, gen_cpu_s, key)
        rec = orig(method, tell(ds, told), truth, config, gen_cpu_s, key)
        rec["told"] = dict(told)
        rec["dataset_sha256_generated"] = G.dataset_hash(ds)
        return rec
    return run_one


# ================================================================================================ campaign hooks
@contextlib.contextmanager
def installed(spec: dict):
    """campaign / runner hooks for an extras spec (restored after): seed guard = XMETHOD_EXTRAS and DEV mode, the X2
    design at generation, the X3 told designs at run_one, the ``extras`` integrity stamp, the same-spec accept rule,
    and cloud sessions running this module."""
    validate(spec)
    ex = spec["extras"]
    design = ex.get("design")
    told_of = {a: d["told"] for a, d in spec["arms"].items() if d.get("told") is not None}
    stamp = {"version": EXTRAS_VERSION, "experiment": ex["experiment"], "design": design,
             "seed_block": "XMETHOD_EXTRAS", "protocol_sha256": protocol_sha256()}
    names = ("is_eval", "_seed_list", "generate_dataset", "accept_rule", "run_units", "_cloud_cmd")
    saved = {k: getattr(C, k) for k in names}
    saved_run_one = R.run_one

    def is_eval(s):
        return False if is_extras(s) else saved["is_eval"](s)

    def _seed_list(s, sp=None):
        return seed_list(s) if is_extras(sp) else saved["_seed_list"](s, sp)

    def accept_rule(s):
        if not is_extras(s):
            return saved["accept_rule"](s)
        sha = C.spec_sha256(s)
        return lambda r: ((r.get("run_mode") or {}).get("mode", "dev") == "dev"
                          and (r.get("integrity") or {}).get("spec_sha256") == sha)

    def run_units(*a, stamp=None, **k):
        return saved["run_units"](*a, stamp={**(stamp or {}), "extras": stamp_x}, **k)

    def _cloud_cmd(*a, **k):
        cmd = saved["_cloud_cmd"](*a, **k)
        old, new = "-m cdd_oran.xmethod.campaign run", "-m cdd_oran.xmethod.extras run"
        if old not in cmd:
            raise AssertionError("campaign's cloud command changed: extras cannot route its run")
        return cmd.replace(old, new)

    stamp_x = stamp
    try:
        C.is_eval, C._seed_list, C.accept_rule = is_eval, _seed_list, accept_rule
        C.run_units, C._cloud_cmd = run_units, _cloud_cmd
        if design is not None:
            C.generate_dataset = functools.partial(generate, design=design)
        if told_of:
            R.run_one = _told_run_one(saved_run_one, told_of)
        yield stamp
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
        R.run_one = saved_run_one


# ================================================================================================ specs
def _frozen_arms() -> dict:
    return json.load(open(_path(FROZEN_SPEC_REL), encoding="utf-8"))["arms"]


def _arm(frozen: dict, name: str, told: dict | None = None) -> dict:
    keep = ("ref", "config", "declare", "analysis", "label", "worlds", "native_partner")
    d = {k: frozen[name][k] for k in keep if k in frozen[name]}
    if told is not None:
        d.update(base=name, told=told)
    return d


def _block(role: str, worlds, regimes, ns, seeds, arms, lams=None) -> dict:
    b = {"role": role, "worlds": list(worlds), "regimes": list(regimes), "ns": list(ns), "kappas": [KAPPA],
         "seeds": list(seeds), "arms": list(arms)}
    if lams is not None:
        b["e4_lams"] = lams
    return b


def build_specs() -> dict[str, dict]:
    """The declared specs (EXTRAS_PROTOCOL.md): arms copied from the frozen EVAL spec (ref, config, declare)."""
    fz = json.load(open(_path(FROZEN_SPEC_REL), encoding="utf-8"))
    arms, budget = fz["arms"], fz["budget_cpu_s"]
    out = {}
    tau_arms = [a for a in X2_ARMS if arms[a]["declare"] == "tau"]
    for name, (delta, nb) in X2_DESIGNS.items():
        out[name] = {
            "name": f"extras_{name}", "budget_cpu_s": budget,
            "note": (f"R-60 X2 (EXPLORATORY; does not change C1-C3): dither delta {delta:g}, {nb} setpoint blocks"
                     + (" = the frozen EVAL R2 design (reproduction cell)" if (delta, nb) == (0.10, 20) else "")
                     + f"; declared in {PROTOCOL_REL}"),
            "extras": {"experiment": "X2", "design": {"delta": delta, "n_blocks": nb}, "protocol": PROTOCOL_REL},
            "arms": {a: _arm(arms, a) for a in X2_ARMS},
            "blocks": [_block("tune", X2_WORLDS, ["R2"], X2_NS, X2_TUNE, tau_arms),
                       _block("measure", X2_WORLDS, ["R2"], X2_NS, X2_MEASURE, X2_ARMS)]}
    x3 = {a: _arm(arms, a) for a in X3_ARMS}
    dith, logd = list(X3_ARMS), list(X3_ARMS)
    for tag, told in X3_DITHER.items():
        for a in X3_ARMS:
            x3[f"{a}.{tag}"] = _arm(arms, a, told)
            dith.append(f"{a}.{tag}")
    for tag, told in X3_LOGGED.items():
        for a in X3_ARMS:
            x3[f"{a}.{tag}"] = _arm(arms, a, told)
            logd.append(f"{a}.{tag}")
    out["x3_told"] = {
        "name": "extras_x3_told", "budget_cpu_s": budget,
        "note": (f"R-60 X3 (EXPLORATORY; does not change C1-C3): data as generated (frozen design), the design TOLD "
                 f"to the arm misspecified (arm suffix: w = dither width x, s = switch-time shift, l = E4 R3 lambda "
                 f"x; no suffix = exact); a degradation curve, not evidence against C2a; declared in {PROTOCOL_REL}"),
        "extras": {"experiment": "X3", "design": None, "protocol": PROTOCOL_REL},
        "arms": x3,
        "blocks": [_block("measure", ["E1", "E2"], ["R2"], [X3_N], X3_MEASURE, dith),
                   _block("measure", ["E4"], ["R3"], [X3_N], X3_MEASURE, logd, lams={"R3": [1.0]})]}
    return out


def build_cost_agg(specs: dict[str, dict]) -> dict:
    """DEV cost table (campaign aggregate 'cost' entries, Kaggle CPU-s) of every arm in ``specs``: the frozen arms'
    DEV means; a told arm takes its base arm's entries (the told design costs nothing extra)."""
    cost = {}
    for rel in DEV_AGGS:
        cost.update(json.load(open(_path(rel), encoding="utf-8"))["cost"])
    out = {}
    for spec in specs.values():
        for a, d in spec["arms"].items():
            base = d.get("base", a)
            for k, v in cost.items():
                arm, _, rest = k.partition("|")
                if arm == base:
                    out[f"{a}|{rest}"] = v
    return {"about": f"R-60 extras DEV cost table: 'cost' of {', '.join(DEV_AGGS)}; told arms = their base arm",
            "cost": dict(sorted(out.items()))}


def write_specs() -> list[str]:
    specs = build_specs()
    os.makedirs(_path(SPEC_DIR_REL), exist_ok=True)
    paths = []
    for name, spec in specs.items():
        validate(spec)
        p = _path(f"{SPEC_DIR_REL}/{name}.json")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(spec, fh, indent=1)
            fh.write("\n")
        paths.append(p)
    with open(_path(COST_AGG_REL), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(build_cost_agg(specs), fh, indent=1)
        fh.write("\n")
    return paths


def load_spec(name_or_path: str) -> dict:
    p = name_or_path if os.path.exists(name_or_path) else _path(f"{SPEC_DIR_REL}/{name_or_path}.json")
    return json.load(open(p, encoding="utf-8"))


def projection(spec_names: list[str] | None = None) -> dict:
    """campaign.project of every declared spec on the DEV cost table (Kaggle-reference CPU-h, an upper estimate:
    the DEV CPU-s came from oversubscribed sessions)."""
    agg = json.load(open(_path(COST_AGG_REL), encoding="utf-8"))
    out = {}
    for name in spec_names or list(build_specs()):
        spec = load_spec(name)
        with installed(spec):
            out[name] = C.project(agg, spec)
    return out


def projection_md(pr: dict) -> str:
    lines = ["# R-60 extras: cost projection (DEV cost table)", "",
             "Exploratory supplementary runs (EXTRAS_PROTOCOL.md); does not change C1-C3. Kaggle-reference CPU-h from "
             f"`{COST_AGG_REL}` (DEV means; told arms = base arm; n without DEV data: campaign's power-law fit). "
             "Session wall ~ CPU-h / processes (one process per vCPU).", "",
             "| spec | units | datasets | CPU-h | gen CPU-h | top arms (CPU-h) | wall h at 4 procs | basis not measured |",
             "|---|---|---|---|---|---|---|---|"]
    tot = 0.0
    for name, p in pr.items():
        top = sorted(p["cpu_h_by_arm"].items(), key=lambda x: -x[1])[:3]
        est = sorted({a for a, b in p["cost_basis_by_arm"].items() if b != ["measured"]})
        lines.append(f"| {name} | {p['n_units']} | {p['n_datasets']} | {p['cpu_h_total']:.1f} | "
                     f"{p['cpu_h_generation']:.2f} | {', '.join(f'{a} {v:.1f}' for a, v in top)} | "
                     f"{p['cpu_h_total'] / 4:.1f} | {', '.join(est) or 'none'} |")
        tot += p["cpu_h_total"]
    lines += ["", f"Total {tot:.1f} CPU-h. Arms without any cost: "
              f"{sorted({a for p in pr.values() for a in p['arms_without_cost']}) or 'none'}."]
    return "\n".join(lines) + "\n"


# ================================================================================================ analysis
def _eval_analysis():
    """The frozen eval_analysis module (read only: its screening, cells, rates and CIs define the metrics)."""
    sp = importlib.util.spec_from_file_location("eval_analysis", _path(EVAL_ANALYSIS_REL))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def design_label(spec: dict, arm: str) -> str:
    ex = spec["extras"]
    if ex["experiment"] == "X2":
        return f"delta {ex['design']['delta']:g} / {ex['design']['n_blocks']} blocks"
    told = spec["arms"][arm].get("told")
    if not told:
        return "exact"
    k, v = check_told(told)
    return {"width": f"width x{v:g}", "shift": f"switch +{v * 100:g} % of a block", "lam": f"lambda x{v:g}"}[k]


def repro_check(records: list[dict], spec: dict) -> dict:
    """Data provenance: every distinct dataset of the records regenerated by the FROZEN generator. X2 delta .10 / 20
    blocks and X3: the records' generated-data hash must equal it; other X2 designs: must differ (R2 data change)."""
    ex = spec["extras"]
    want_equal = ex["experiment"] == "X3" or (float(ex["design"]["delta"]), int(ex["design"]["n_blocks"])) == (0.10, 20)
    seen, eq, ne = {}, 0, 0
    for r in records:
        if r.get("status") != "ok":
            continue
        j = r["job"]
        k = (j["world"], j["regime"], j.get("lam"), j["n"], j["seed"], j["kappa"])
        h = r.get("dataset_sha256_generated") or r.get("dataset_sha256")
        if k in seen:
            if seen[k][0] != h:
                return {"ok": False, "why": f"records of dataset {k} disagree on the generated data"}
            continue
        ds, _ = G.generate_dataset(j["world"], j["regime"], j["n"], j["seed"],
                                   lam=1.0 if j.get("lam") is None else j["lam"], kappa=j["kappa"])
        seen[k] = (h, G.dataset_hash(ds))
        eq, ne = eq + (h == seen[k][1]), ne + (h != seen[k][1])
    ok = (ne == 0) if want_equal else (eq == 0)
    return {"ok": ok, "datasets": len(seen), "equal_to_frozen": eq, "differ_from_frozen": ne,
            "expected": "equal" if want_equal else "differ"}


def _rate(x: dict | None) -> dict | None:
    return None if not x else {k: x.get(k) for k in ("rate", "hits", "n", "ci", "validity")}


def analyse(pairs: list[tuple[str, str]]) -> dict:
    """Tables of the declared metrics per (spec design, arm, cell): eval_analysis.screen + build_cells (primary
    declaration: BY for p arms, conformal tau over the cell's tune records for tau arms), raw p <= .05 rates, recall."""
    E = _eval_analysis()
    rows, checks, pooled = [], {}, []
    for spec_path, merged in pairs:
        spec = load_spec(spec_path)
        validate(spec)
        recs = E.load_records(merged)
        use, scr = E.screen(recs, spec)
        cells, seed_rows = E.build_cells(use, spec)
        checks[spec["name"]] = {"records": len(recs), "used": len(use), "duplicates": len(scr["duplicates"]),
                                "unexpected": len(scr["unexpected"]), "role_mismatch": len(scr["role_mismatch"]),
                                "provenance": repro_check(use, spec)}
        groups: dict[tuple, list] = {}
        for ck, e in sorted(cells.items()):
            prim = e.get("primary") or {}
            rows.append({"spec": spec["name"], "experiment": spec["extras"]["experiment"],
                         "design": design_label(spec, e["arm"]), "arm": e["arm"],
                         "base_arm": spec["arms"][e["arm"]].get("base", e["arm"]), "declare": e["declare"],
                         "world": e["world"], "regime": e["regime"], "lam": e["lam"], "n": e["n"],
                         "kappa": e["kappa"], "status": e["status"], "n_measure": e["n_measure"],
                         "planned_seeds": e["planned_seeds"], "n_infeasible": e["n_infeasible"],
                         "n_error": e["n_error"], "tau": e["tau"],
                         **{k: _rate(prim.get(k)) for k in E.RATE_KEYS}, "recall": prim.get("recall"),
                         "validity_label": prim.get("validity")})
            groups.setdefault((e["arm"], e["regime"]), []).extend(seed_rows.get(ck) or [])
        for (arm, regime), srows in sorted(groups.items()):
            pooled.append({"spec": spec["name"], "experiment": spec["extras"]["experiment"], "arm": arm,
                           "base_arm": spec["arms"][arm].get("base", arm), "declare": spec["arms"][arm]["declare"],
                           "regime": regime, "design": design_label(spec, arm), "design_key": _design_key(spec, arm),
                           **{k: _rate(E._pooled(srows, k)) if srows else None for k in E.RATE_KEYS},
                           "recall": E.mean_ci([x["recall"] for x in srows]) if srows else None})
    return {"extras_version": EXTRAS_VERSION, "analysis": getattr(E, "ANALYSIS_VERSION", None),
            "protocol_sha256": protocol_sha256(), "checks": checks, "rows": rows, "pooled": pooled,
            "x2_trend": x2_trend(pooled), "x3_curve": x3_curve(pooled), "wording": WORDING}


WORDING = ("These supplementary experiments are exploratory: they were declared before launch "
           f"({PROTOCOL_REL}) on a fresh seed block, use the frozen methods and procedures, and do not change C1-C3.")
TREND_KEYS = ("null_decl", "null_raw", "plac_raw")
X3_FAMILIES = {"width narrower": (("width", 0.8), ("width", 0.5)), "width wider": (("width", 1.25), ("width", 2.0)),
               "switch late": (("shift", 0.02), ("shift", 0.05)), "lambda lower": (("lam", 0.8), ("lam", 0.5)),
               "lambda higher": (("lam", 1.25), ("lam", 2.0))}


def _design_key(spec: dict, arm: str):
    ex = spec["extras"]
    if ex["experiment"] == "X2":
        return [float(ex["design"]["delta"]), int(ex["design"]["n_blocks"])]
    told = spec["arms"][arm].get("told")
    return list(check_told(told)) if told else ["exact", 1.0]


def trend_word(values: list[float | None]) -> str:
    """Declared descriptive rule (EXTRAS_PROTOCOL.md): point estimates in increasing delta; no test."""
    if any(v is None for v in values):
        return "not reported (a design has no rate)"
    d = np.diff(values)
    if np.all(d == 0):
        return "does not change"
    if np.all(d >= 0):
        return "rises"
    if np.all(d <= 0):
        return "falls"
    return "is not monotone"


def x2_trend(pooled: list[dict]) -> list[dict]:
    """Per X2 arm and rate: pooled rates over delta .02 / .05 / .10 / .20 (20 blocks), the trend word, and the
    5 / 80-block rates at delta .10."""
    out = []
    x2 = [p for p in pooled if p["experiment"] == "X2"]
    for arm in sorted({p["arm"] for p in x2}):
        by = {tuple(p["design_key"]): p for p in x2 if p["arm"] == arm}
        for k in TREND_KEYS:
            dl = [by.get((d, 20), {}).get(k) for d in (0.02, 0.05, 0.10, 0.20)]
            if all(x is None for x in dl):
                continue
            out.append({"arm": arm, "rate": k, "delta": [0.02, 0.05, 0.10, 0.20], "by_delta": dl,
                        "trend": trend_word([None if x is None else x["rate"] for x in dl]),
                        "blocks_5": by.get((0.10, 5), {}).get(k), "blocks_80": by.get((0.10, 80), {}).get(k)})
    return out


def x3_curve(pooled: list[dict]) -> list[dict]:
    """Per X3 base arm, regime and family (milder first): the R-30 class of the pooled truth-null and placebo raw /
    declared rates, and the first variant whose class is INVALID (declared descriptive rule)."""
    out = []
    x3 = [p for p in pooled if p["experiment"] == "X3"]
    for (base, regime) in sorted({(p["base_arm"], p["regime"]) for p in x3}):
        by = {tuple(p["design_key"]): p for p in x3 if p["base_arm"] == base and p["regime"] == regime}
        exact = by.get(("exact", 1.0))
        for fam, steps in X3_FAMILIES.items():
            got = [by.get(s) for s in steps]
            if not any(got):
                continue
            seq = [("exact", exact)] + [(f"{k} x{v:g}" if k != "shift" else f"switch +{v * 100:g} %", g)
                                        for (k, v), g in zip(steps, got, strict=True)]
            keys = ("null_raw", "plac_raw", "conf_raw", "null_decl", "plac_decl", "conf_decl")
            first = next((lab for lab, g in seq[1:] if g and any((g.get(k) or {}).get("validity") == "INVALID"
                                                                for k in keys)), None)
            out.append({"base_arm": base, "regime": regime, "family": fam,
                        "steps": [{"variant": lab, **{k: (g or {}).get(k) for k in keys}} for lab, g in seq],
                        "first_invalid": first or "none"})
    return out


def _txt(x: dict | None) -> str:
    if not x:
        return "NA"
    lo, hi = x["ci"]
    ci = f" [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
    return f"{x['rate']:.3f} ({x['hits']}/{x['n']}){ci}"


def analyse_md(out: dict) -> str:
    lines = ["# R-60 extras: tables", "", f"{out['wording']} Declared in `{PROTOCOL_REL}` (LF sha256 "
             f"{out['protocol_sha256']}). Rates: hits / testable candidates, seed-cluster 95 % bootstrap CI "
             "(eval_analysis). The R-30 class is a descriptive label here, not a verdict.", "", "## Checks", ""]
    for k, c in out["checks"].items():
        lines.append(f"- {k}: {json.dumps(c, sort_keys=True)}")
    lines += ["", "| spec | design | arm | world | regime | n | status | seeds | tau | truth-null decl | truth-null raw "
              "| P_placebo decl | P_placebo raw | P_placebo_conf raw | recall [CI] | label |",
              "|" + "---|" * 16]
    for r in out["rows"]:
        rc = r["recall"]
        rtxt = "NA" if not rc else (f"{rc['mean']:.3f}" + (f" [{rc['ci'][0]:.3f}, {rc['ci'][1]:.3f}]"
                                                             if rc["ci"][0] is not None else ""))
        tau = "" if r["tau"] is None else f"{r['tau']:.4g}"
        lam = "" if r["lam"] is None else f" lam {r['lam']:g}"
        lines.append(f"| {r['spec']} | {r['design']} | {r['arm']} | {r['world']} | {r['regime']}{lam} | "
                     f"{r['n']} | {r['status']} | "
                     f"{r['n_measure']}/{r['planned_seeds']} | {tau} | {_txt(r['null_decl'])} | {_txt(r['null_raw'])} | "
                     f"{_txt(r['plac_decl'])} | {_txt(r['plac_raw'])} | {_txt(r['conf_raw'])} | {rtxt} | "
                     f"{r['validity_label'] or ''} |")
    lines += ["", "## Pooled per arm (clusters = world x seed)", "",
              "| spec | design | arm | regime | truth-null decl | truth-null raw | P_placebo decl | P_placebo raw | "
              "P_placebo_conf raw | recall [CI] |", "|" + "---|" * 10]
    for p in out["pooled"]:
        rc = p["recall"]
        rtxt = "NA" if not rc else f"{rc['mean']:.3f}" + (f" [{rc['ci'][0]:.3f}, {rc['ci'][1]:.3f}]"
                                                           if rc["ci"][0] is not None else "")
        lines.append(f"| {p['spec']} | {p['design']} | {p['arm']} | {p['regime']} | {_txt(p['null_decl'])} | "
                     f"{_txt(p['null_raw'])} | {_txt(p['plac_decl'])} | {_txt(p['plac_raw'])} | "
                     f"{_txt(p['conf_raw'])} | {rtxt} |")
    if out["x2_trend"]:
        lines += ["", "## X2 dose-response (pooled; trend = declared rule on point estimates, no test)", ""]
        for t in out["x2_trend"]:
            lines.append(f"- {t['arm']} {t['rate']}: " + ", ".join(
                f"delta {d:g} {_txt(x)}" for d, x in zip(t["delta"], t["by_delta"], strict=True))
                + f"; {t['trend']} with delta; 5 blocks {_txt(t['blocks_5'])}, 80 blocks {_txt(t['blocks_80'])}")
    if out["x3_curve"]:
        lines += ["", "## X3 degradation curve (first variant with an INVALID pooled rate; descriptive)", ""]
        for c in out["x3_curve"]:
            lines.append(f"- {c['base_arm']} {c['regime']} {c['family']}: first INVALID at {c['first_invalid']}; "
                         + "; ".join(f"{s['variant']} null raw {_txt(s['null_raw'])}, P_placebo raw "
                                     f"{_txt(s['plac_raw'])}" for s in c["steps"]))
    return "\n".join(lines) + "\n"


# ================================================================================================ CLI
def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with (gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else
          open(path, "w", encoding="utf-8", newline="\n")) as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own = ("specs", "projection", "analyse")
    if argv and argv[0] in own:
        ap = argparse.ArgumentParser(prog="cdd_oran.xmethod.extras")
        sub = ap.add_subparsers(dest="cmd", required=True)
        sub.add_parser("specs")
        p = sub.add_parser("projection")
        p.add_argument("--specs", nargs="*", default=None)
        p.add_argument("--out", default=SPEC_DIR_REL)
        p = sub.add_parser("analyse")
        p.add_argument("--spec", nargs="+", required=True)
        p.add_argument("--merged", nargs="+", required=True)
        p.add_argument("--out", required=True)
        a = ap.parse_args(argv)
        if a.cmd == "specs":
            for p in write_specs():
                print(f"[xm-x] {os.path.relpath(p, R.ROOT)}")
        elif a.cmd == "projection":
            pr = projection(a.specs)
            _write(os.path.join(a.out, "projection.json"), json.dumps(pr, indent=1, sort_keys=True) + "\n")
            _write(os.path.join(a.out, "PROJECTION.md"), projection_md(pr))
            print(projection_md(pr))
        else:
            if len(a.spec) != len(a.merged):
                raise SystemExit("--spec and --merged pair up: give as many of each")
            out = analyse(list(zip(a.spec, a.merged, strict=True)))
            _write(os.path.join(a.out, "extras_tables.json"), json.dumps(R._clean(out), indent=1, sort_keys=True))
            _write(os.path.join(a.out, "EXTRAS_TABLES.md"), analyse_md(out))
            print(json.dumps(out["checks"], indent=1))
        return 0
    if "--spec" not in argv:
        raise SystemExit(f"usage: extras {{{','.join(own)}}} ... | extras <campaign cmd> --spec EXTRAS_SPEC ...")
    spec = json.load(open(argv[argv.index("--spec") + 1], encoding="utf-8"))
    if not is_extras(spec):
        raise SystemExit("extras refuses a spec without an 'extras' block (use cdd_oran.xmethod.campaign)")
    with installed(spec):
        return C.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
