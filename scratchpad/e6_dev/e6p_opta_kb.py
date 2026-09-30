"""E6-P option (a) kill test K-B, COLLECTION ("does confounding bite?"; scratchpad/e6_dev/decision/OPTION_A_PLAN.md
sections 1, 2, 6; DEV kill test, NOT frozen). Analysis: scratchpad/e6_dev/e6p_opta_kb_analyze.py (Kaggle job).

Plant: identical to e6p_discovery (P3 surge-L40 = ``e6p_screen.make_cfg("P3", 3, seed, lf)``, lf = e6p_state L40,
120 s warm-up + 600 s scored = 720 s; ``E6Env(cfg, log=False, wg3=True, trace=True)``; ``collect_p.run_collection``,
``units_p.UnitArbiter``, T = 60 s, open_rule "feasible"). Differences (option (a)):
  * logging policy = ``collect_p.IncumbentPolicy`` (context-dependent, accept / reject only, floor .15, draws tag 6622,
    per-unit row logged as unit["probs"]) or ``collect_p.PlaceboIncumbent`` (same draws / rows, accept applied);
  * the arbiter is active from t = 0 (``run_collection(arb_warmup_s=0.0)``): units open inside the warm-up (picos fall
    asleep at t 60-120); the plant's scoring is unchanged (warm-up unscored);
  * the KPI tap counts every second (``count_all=True``) so the privileged unit ``lab_kpi`` window sums are right for
    units opening in the warm-up (``lab_series`` records every second either way; the analysis reads lab_series).

Stages and seeds (every seed asserted in the K-B sub-block 186000-186079 of the registered "e6p_confounded" block
186000-187999 and outside every other E6 block; job order interleaves the stages):
  placebo  PlaceboIncumbent, seed = 186000 + j, j = 0..39; records stage "placebo", sub "kb", policy
           "placebo_incumbent"
  applied  IncumbentPolicy, seed = 186040 + j, j = 0..39; records stage "eval", sub "kb", fold = j // 10, policy
           "incumbent" (stage "eval" so the v2 analyzer / disc_bench readers accept the file)
  all      both (80 jobs; the cloud wrapper e6p_opta_kb_run.py default)

CLI (repo root, PYTHONPATH=.; in the cloud bundle the same file is e6dev/e6p_opta_kb.py):
  python scratchpad/e6_dev/e6p_opta_kb.py run [--stage placebo|applied|all] --part i/k --out F.jsonl [--smoke]
         [--short SCORED_S]
  python scratchpad/e6_dev/e6p_opta_kb_run.py run --part i/k --out F.jsonl          # cloud.py wrapper (stage all)
  python scratchpad/e6_dev/e6p_opta_kb.py summary --in FILES... [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_opta_kb.py list
--smoke: plumbing only; seed -> seed % 31, the FIRST job of each requested stage, records smoke=True; --short sets the
scored seconds (smoke only).

RECORD FORMAT: schema "e6p-disc-rec/1" exactly as scratchpad/e6_dev/e6p_discovery.py (its module docstring is the
contract), with: pi0_table = null (every unit carries its own row); per unit additionally ``probs`` ({"accept": pa,
"reject": 1 - pa}, the logged row; mode / p = the incumbent draw and its propensity = probs[mode]) and ``inc``
({cls, key, s, below}: request class, table cell, pressure, own_prot_below_frac); gt_labels = []; and episode keys
kb_stage ("placebo" | "applied"), incumbent (table, floor, s_hi, sg_raise_frac, tag), arb_warmup_s (0.0),
tap_count_all (true).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_discovery as DI  # noqa: E402  (record helpers: unit_record, _n_applied)
import e6p_screen as S  # noqa: E402  (sets ROOT / BUNDLE and sys.path)

from cdd_oran.decision import collect_p as CP  # noqa: E402
from cdd_oran.decision import features as F  # noqa: E402
from cdd_oran.decision import units_p as UP  # noqa: E402
from cdd_oran.envs.e6.env import E6Env  # noqa: E402

PLAN = "scratchpad/e6_dev/decision/OPTION_A_PLAN.md section 6 K-B (2026-09-30, not frozen)"
SCHEMA = DI.SCHEMA                       # "e6p-disc-rec/1"
REGISTRY_DOC = "docs/benchmark/SEED_REGISTRY.json"
PAIR, STRATUM = "P3", 3
SEED_BLOCK = (186000, 187999)            # registered "e6p_confounded" (E6.e6p_confounded_episodes)
KB_BLOCK = (186000, 186079)
FORBIDDEN = ((150000, 150399), (155000, 155399), (160000, 179999), (180000, 183999), (184000, 185999),
             (190000, 190399))
PLACEBO_BASE, APPLIED_BASE, N_EP = 186000, 186040, 40
N_FOLDS = 4
STAGES = ("placebo", "applied")
OPEN_RULE = "feasible"
ARB_WARMUP_S = 0.0
TAP_COUNT_ALL = True
STEP_S = DI.STEP_S
T = UP.T_UNIT
TAGS = {6622: "incumbent logging draws", 6623: "MapGate random map (K-A2)", 6624: "option (a) analysis bootstraps"}


# ---------------------------------------------------------------------------------------------- seeds / jobs
def check_seed(seed: int) -> int:
    seed = int(seed)
    assert KB_BLOCK[0] <= seed <= KB_BLOCK[1], f"seed {seed} outside the K-B block {KB_BLOCK}"
    assert SEED_BLOCK[0] <= seed <= SEED_BLOCK[1]
    for lo, hi in FORBIDDEN:
        assert not lo <= seed <= hi, f"seed {seed} inside another block [{lo}, {hi}]"
    return seed


def jobs(stage: str = "all") -> list:
    """[(kb_stage, seed, j)] in shard order (stages interleaved by j)."""
    stages = STAGES if stage == "all" else (stage,)
    if any(s not in STAGES for s in stages):
        raise SystemExit(f"--stage in {STAGES + ('all',)}")
    base = {"placebo": PLACEBO_BASE, "applied": APPLIED_BASE}
    return [(s, check_seed(base[s] + j), j) for j in range(N_EP) for s in stages]


def registry_check() -> dict | None:
    """{block, tags} registered in SEED_REGISTRY.json; None when the file is absent (cloud.py bundles it only for
    e6p_disc_* scripts; the hard-coded check_seed guard runs everywhere)."""
    path = os.path.join(S.ROOT, REGISTRY_DOC)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    blk = (d.get("E6", {}) or {}).get("e6p_confounded_episodes", [])
    return {"block_186000_187999": any(lo <= SEED_BLOCK[0] and hi >= SEED_BLOCK[1] for lo, hi in blk),
            "tags_6622_6624": all(t in d.get("rng_stream_tags", []) for t in TAGS)}


def make_policy(kb_stage: str, sd: int):
    if kb_stage == "placebo":
        return CP.PlaceboIncumbent(sd), "placebo_incumbent"
    if kb_stage == "applied":
        return CP.IncumbentPolicy(sd), "incumbent"
    raise SystemExit(f"stage in {STAGES}")


def incumbent_consts() -> dict:
    return {"table": CP.INCUMBENT_TABLE, "floor": CP.INC_FLOOR, "s_hi": CP.INC_S_HI,
            "sg_raise_frac": CP.INC_SG_RAISE_FRAC, "tag": CP.INC_TAG, "saving_dir": CP.INC_SAVING_DIR,
            "pressure": "max(ctx own_prb_util, ctx nbr_max_prb_util)"}


# ---------------------------------------------------------------------------------------------- one episode
def unit_record(i, u, n_app) -> dict:
    r = DI.unit_record(i, u, n_app)
    r["probs"] = {m: float(p) for m, p in u["probs"].items()}
    r["inc"] = u.get("inc")
    return r


def episode_job(kb_stage, seed, j, lf, smoke=False, short=None) -> dict:
    check_seed(seed)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    policy, pol_name = make_policy(kb_stage, sd)
    env = E6Env(cfg, log=False, wg3=True, trace=True)
    t_wall = time.time()
    res = CP.run_collection(cfg, policy, T=T, env=env, open_rule=OPEN_RULE, arb_warmup_s=ARB_WARMUP_S,
                            count_all=TAP_COUNT_ALL)
    t_p = time.process_time()
    trace = env.get_trace()
    panel = F.build_panel_p(trace, step_s=STEP_S, episode=seed, drop_degenerate=False)
    panel_cpu = time.process_time() - t_p
    units = res["units"]
    n_app = DI._n_applied(trace, units, T)
    urecs = [unit_record(i, u, n_app[i]) for i, u in enumerate(units)]
    by, by_cls = {}, {}
    for u in urecs:
        b = by.setdefault(u["knob"], {})
        b[u["mode"]] = b.get(u["mode"], 0) + 1
        c = by_cls.setdefault(f"{u['inc']['cls']}:{u['inc']['key']}", {})
        c[u["mode"]] = c.get(u["mode"], 0) + 1
    sl = [u for u in urecs if u["knob"] == "sleep"]
    cnt = {"units": len(urecs), "by_family_mode": by, "by_class_mode": by_cls, "sleep_units": len(sl),
           "sleep_rejects": sum(u["mode"] == "reject" for u in sl),
           "units_warmup": sum(u["t0"] < cfg.warmup_s for u in urecs), "labels": 0}
    ser = res["tap"].series()
    lay, pl = env.plant.lay, env.plant
    es = next((x for x in env.xapps if x.name == "ES"), None)
    st = env.static()
    stage = "placebo" if kb_stage == "placebo" else "eval"
    return {"kind": "episode", "schema": SCHEMA, "key": [stage, "kb", seed], "stage": stage, "sub": "kb",
            "kb_stage": kb_stage, "seed": seed, "cfg_seed": sd, "j": j,
            "fold": (j * N_FOLDS) // N_EP if kb_stage == "applied" else None, "smoke": smoke, "short": short,
            "policy": pol_name, "pi0_table": None, "incumbent": incumbent_consts(), "arb_warmup_s": ARB_WARMUP_S,
            "tap_count_all": TAP_COUNT_ALL, "T": T, "open_rule": OPEN_RULE, "load_factor": lf,
            "episode_s": env.total_s, "warmup_s": cfg.warmup_s, "n_cells": int(pl.nc), "step_s": STEP_S,
            "obs_static": {"is_macro": [bool(v) for v in st["is_macro"]],
                           "neighbours": [[int(v) for v in n] for n in st["neighbours"]]},
            "units": urecs, "panel": F.panel_to_rec(panel),
            "lab_series": {"fields": ser["fields"], "t_first": 1, "scored": CP.enc(ser["scored"]),
                           "data": CP.enc(ser["data"])},
            "lab_outcome": S.outcome(env),
            "gt_static": {"cand": {str(k): int(v) for k, v in (es.cand.items() if hasattr(es, "cand") else [])},
                          "cell_site": [int(v) for v in lay.cell_site],
                          "neighbours": [[int(v) for v in n] for n in lay.neighbours],
                          "is_macro": [bool(v) for v in lay.is_macro]},
            "gt_labels": [], "counts": cnt, "cpu_s": round(res["cpu_s"] + panel_cpu, 2),
            "collect_cpu_s": round(res["cpu_s"], 2), "label_cpu_s": 0.0, "panel_cpu_s": round(panel_cpu, 2),
            "n_roll": 0, "secs": round(time.time() - t_wall, 1), "rss_mb": _rss()}


def _rss():
    try:
        import e6p_step2_dev as D2
        return D2.peak_rss_mb()
    except Exception:                                                 # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------- run
def run(stage, part, out, smoke=False, short=None):
    stage = stage or "all"
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    reg = registry_check()
    if not smoke and reg is not None and not all(reg.values()) and not os.environ.get("E6P_KB_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block / RNG tags not registered in {REGISTRY_DOC}: {reg}")
    state = S.load_state()
    lf = S.lf_of(state, STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header("opta_kb", part, smoke, state)
    head.update(kind="header", schema=SCHEMA, plan=PLAN, driver="e6p_opta_kb", registry=reg,
                consts={"pair": PAIR, "stratum": STRATUM, "load_factor": lf, "T": T, "open_rule": OPEN_RULE,
                        "step_s": STEP_S, "arb_warmup_s": ARB_WARMUP_S, "tap_count_all": TAP_COUNT_ALL,
                        "incumbent": incumbent_consts(), "placebo_base": PLACEBO_BASE,
                        "applied_base": APPLIED_BASE, "n_ep": N_EP, "stage": stage, "tags": TAGS,
                        "series_fields": CP.SERIES_FIELDS, "smoke": {"short": short} if smoke else None})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "episode" and r.get("smoke") == smoke}
    t0, n, seen = time.time(), 0, set()
    J = jobs(stage)
    for u, (kb, seed, j) in enumerate(J):
        key = ("placebo" if kb == "placebo" else "eval", "kb", seed)
        if smoke:                                    # the first job of each requested stage, shard filter ignored
            if kb in seen:
                continue
            seen.add(kb)
        elif u % k != i or key in done:
            continue
        rec = episode_job(kb, seed, j, lf, smoke, short)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("kb_stage", "seed", "cfg_seed", "counts", "cpu_s", "secs",
                                                  "rss_mb")}), flush=True)
    S._append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_total": len(J),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def summary(paths, json_out=None, allow_smoke=False):
    recs = [r for p in paths for r in S._read(p) if r.get("kind") == "episode" and r.get("schema") == SCHEMA
            and r.get("kb_stage") in STAGES]
    recs = list({tuple(r["key"]) + (r.get("smoke"),): r for r in recs if allow_smoke or not r.get("smoke")}.values())
    report = {"plan": PLAN, "stages": {}}
    for st in STAGES:
        rs = [r for r in recs if r["kb_stage"] == st]
        if not rs:
            continue
        fam, cls = {}, {}
        for r in rs:
            for grp, dst in (("by_family_mode", fam), ("by_class_mode", cls)):
                for f, modes in r["counts"][grp].items():
                    d = dst.setdefault(f, {})
                    for m, v in modes.items():
                        d[m] = d.get(m, 0) + v
        ne = len(rs)
        rep = {"episodes": ne, "units_by_family_mode": fam, "units_by_class_mode": cls,
               "units_per_episode": {f: sum(d.values()) / ne for f, d in fam.items()},
               "units_warmup_per_episode": float(np.mean([r["counts"]["units_warmup"] for r in rs])),
               "sleep": {"units": sum(r["counts"]["sleep_units"] for r in rs),
                         "rejects": sum(r["counts"]["sleep_rejects"] for r in rs)},
               "psvr_mean": float(np.mean([r["lab_outcome"]["psvr"] for r in rs])),
               "energy_kwh_mean": float(np.mean([r["lab_outcome"]["energy_kwh"] for r in rs])),
               "cpu_s_per_episode": float(np.mean([r["cpu_s"] for r in rs])),
               "cpu_h_total": float(sum(r["cpu_s"] for r in rs) / 3600)}
        report["stages"][st] = rep
        print(f"== {st}: {ne} episodes, {rep['cpu_h_total']:.2f} CPU-h ({rep['cpu_s_per_episode']:.0f} s/ep); "
              f"units/ep { {f: round(v, 1) for f, v in rep['units_per_episode'].items()} }; warm-up units/ep "
              f"{rep['units_warmup_per_episode']:.1f}")
        print(f"   modes {fam}")
        print(f"   class x mode {cls}")
        print(f"   psvr {rep['psvr_mean']:.2f}  energy kWh {rep['energy_kwh_mean']:.3f}  sleep {rep['sleep']}")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs():
    print("plan:", PLAN)
    print("registry:", registry_check())
    for st in STAGES + ("all",):
        J = jobs(st)
        print(f"{st}: {len(J)} episodes; seeds {min(x[1] for x in J)}..{max(x[1] for x in J)}")


def cli(argv, stage=None):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        run(a.get("--stage") or stage, a["--part"], a["--out"], smoke="--smoke" in flags,
            short=float(a["--short"]) if a.get("--short") else None)
    elif cmd == "summary":
        files, js = [], None
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x != "--in":
                files.extend(f for f in x.split(",") if f)
        summary(files, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "list":
        list_jobs()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
