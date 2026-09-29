"""E6-P v2 O_tape falsifier driver. Implements docs/benchmark/E6P_V2_OTAPE_PROTOCOL.md, which inherits everything
from E6P_SCREEN_PROTOCOL.md (as run by e6p_screen.py, frozen hash in e6p_screen.FROZEN_SHA256) EXCEPT
(a) oracle guardrail admissibility, (b) fresh seeds 155000-155199, (c) scope = 3 cells.

This module imports e6p_screen and overrides ONLY what (a)-(c) need (see ``_install``):
  screen_seed (b) | units_for (c) | oracle_job + OracleG (a) | state files + write_state (separate v2 state) |
  PROTOCOL / FROZEN_SHA256 (run record) | summary (scope + the descriptive re-drawn-advantage report of doc sec. 4).
Everything else (arms, episodes, QACM, arm 9, analyse_pair_stratum, verdict, bootstrap) is e6p_screen's code.

CLI (repo root, PYTHONPATH=.; in the cloud bundle the same file is e6dev/e6p_v2.py):
  python scratchpad/e6_dev/e6p_v2_stage1.py run --part i/k --out FILE.jsonl [--smoke]   # stage 1 (fresh arms 1-8)
  python scratchpad/e6_dev/e6p_v2_stage2.py run --part i/k --out FILE.jsonl [--smoke]   # stage 2 (O_tape oracle)
  python scratchpad/e6_dev/e6p_v2_stage3.py run --part i/k --out FILE.jsonl [--smoke]   # stage 3 (arm 9, gated)
  python scratchpad/e6_dev/e6p_v2.py summary --in FILES... [--write-state] [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_v2.py init-state      # v2 state <- v1 load_factor (once; refuses to overwrite)
  python scratchpad/e6_dev/e6p_v2.py list | freeze-check | selftest
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_screen as S  # noqa: E402

E6Env = S.E6Env

# ---------------------------------------------------------------------------------------------- frozen record
PROTOCOL = "E6P_V2_OTAPE_PROTOCOL.md (E6P-v2 O_tape falsifier; inherits E6P_SCREEN_PROTOCOL.md Gate A v2)"
V2_DOC = "docs/benchmark/E6P_V2_OTAPE_PROTOCOL.md"
REGISTRY = "docs/benchmark/SEED_REGISTRY.json"
FROZEN_SHA256 = {"docs/benchmark/E6P_SPEC.md": S.FROZEN_SHA256["docs/benchmark/E6P_SPEC.md"],
                 "docs/benchmark/E6P_SCREEN_PROTOCOL.md": S.FROZEN_SHA256["docs/benchmark/E6P_SCREEN_PROTOCOL.md"],
                 V2_DOC: "a08aa3f85ecc911f15c48a508b05529d1768be3e1d43815bf68b5d29ae1ac457",
                 REGISTRY: "690292741a013d529e98679f900f46830f72ebfb1cfc2e795d1eca92a6da323e"}
CELLS = (("P1", 3), ("P3", 1), ("P3", 3))                 # (c) v1 criterion-1 cells
SEED_BASE = 155000                                         # (b)
SCREEN_RANGE, CONFIRM_RANGE = (155000, 155200), (155200, 155400)
GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
ZERO_SLACK = 1.0                                           # (a) bound when AA_k(t) == 0
ADV_TAG = 6612                                             # (d) descriptive bootstrap stream
_SD = os.environ.get("E6P_STATE_DIR", HERE)
STATE_JSON = os.path.join(_SD, "e6p_v2_state.json")
STATE_PY = os.path.join(_SD, "e6p_v2_state_json.py")
V1_STATE_JSON, V1_STATE_PY = os.path.join(HERE, "e6p_state.json"), os.path.join(HERE, "e6p_state_json.py")
V2_NOTES = [
    "v2 (a): guardrail traces = per-second cumulative svr (3600 viol_ue_s/ue_s), nonprot_embb_viol (embb_viol -"
    " prot_viol), ll_viol, rlf of the per-episode accept-all reference run (the v1 churn-cap run, now traced); trial"
    " admissible iff energy bound (unchanged) AND value_k(end of rollout) <= 1.10 AA_k(t) (1 if AA_k(t) == 0) for"
    " every k; redraw rollouts (rho_sign) keep the true-tape traces (their flag is unused, as in v1).",
    "v2 (b): seed = 155000 + 10*stratum + j, j = 0..7; asserted inside [155000, 155200).",
    "v2 (c): stage 1 = freeze + all stage-1 arms of the 3 cells; stage 2 runs on all 3 cells concurrently with stage 1"
    " (not gated); stage 3 gated by the v2 state's stage2_crit12.",
    "v2 (d): summed re-drawn advantage = sum over deviations of aa_redraw[0] - plan_redraw[0] (protected violated"
    " UE-s over H); seed bootstrap (10 000 x 8 seeds, default_rng([6612, pair_idx, stratum])), LB = 10th percentile;"
    " descriptive only.",
]


def screen_seed(s, j):
    return SEED_BASE + 10 * s + j


# ---------------------------------------------------------------------------------------------- state
def load_state():
    js = json.load(open(STATE_JSON)) if os.path.exists(STATE_JSON) else None
    py = None
    if os.path.exists(STATE_PY):
        ns = {}
        exec(open(STATE_PY).read(), ns)                                     # noqa: S102 (our own data mirror)
        py = json.loads(ns["STATE"])
    if js is not None and py is not None and js != py:
        raise SystemExit("e6p_v2_state.json and e6p_v2_state_json.py differ: re-run e6p_v2.py summary --write-state")
    st = js if js is not None else py
    if st is None:
        raise SystemExit("no v2 state: run `e6p_v2.py init-state` first")
    return st


def write_state(state):
    s = json.dumps(state, indent=1, sort_keys=True)
    open(STATE_JSON, "w", newline="\n").write(s + "\n")
    open(STATE_PY, "w", newline="\n").write('"""Mirror of e6p_v2_state.json (cloud.py bundles only *.py). Written by '
                                            'e6p_v2.py init-state / summary --write-state; do not edit."""\n'
                                            f"STATE = r'''{s}'''\n")


def init_state():
    if os.path.exists(STATE_JSON) or os.path.exists(STATE_PY):
        raise SystemExit(f"{STATE_JSON} exists; not overwritten")
    v1 = json.load(open(V1_STATE_JSON))
    ns = {}
    exec(open(V1_STATE_PY).read(), ns)                                     # noqa: S102 (our own data mirror)
    if json.loads(ns["STATE"]) != v1:
        raise SystemExit("v1 e6p_state.json and its mirror differ")
    st ={"inherited_from": "e6p_state.json (E6P-v1 stage 0a / 0b)", "load_factor": v1["load_factor"],
          "l_feasible": v1.get("l_feasible"), "not_screenable_pairs": v1.get("not_screenable_pairs", []),
          "v1_stage1_crit1": v1.get("stage1_crit1"), "cells": [list(c) for c in CELLS],
          "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    assert sorted(map(tuple, v1["stage1_crit1"])) == sorted(CELLS), v1["stage1_crit1"]
    write_state(st)
    print("v2 state ->", STATE_JSON, st)


# ---------------------------------------------------------------------------------------------- (a) oracle
def guard_values(sla):
    """Cumulative guardrail values from episode start (same definitions as e6p_screen.outcome)."""
    return {"svr": 3600.0 * float(sla["viol_ue_s"]) / max(float(sla["ue_s"]), 1e-9),
            "nonprot_embb_viol": float(sla["embb_viol"] - sla["prot_viol"]),
            "ll_viol": float(sla["ll_viol"]), "rlf": float(sla["rlf"])}


def guard_bound(aa_value):
    return S.GUARD * aa_value if aa_value > 0 else ZERO_SLACK


def guard_admissible(vals, aa_vals):
    """-> (ok, [violated keys]). vals / aa_vals: {key: cumulative value at the same second}."""
    bad = [k for k in GUARD_KEYS if vals[k] > guard_bound(aa_vals[k]) + 1e-9]
    return not bad, bad


def run_env_traces(cfg, arb, churn_cap=None):
    """e6p_screen.run_env (wg3=True, no log) that also records per-second cumulative guardrail traces
    {key: [value after t control seconds]}, index 0 = episode start (the E trace convention)."""
    env = E6Env(cfg, log=False, wg3=True, churn_cap=churn_cap)
    g0 = guard_values(env.plant.sla)
    G = {k: [g0[k]] for k in GUARD_KEYS}
    while env.sec < env.total_s:
        env.step(arb)
        g = guard_values(env.plant.sla)
        for k in GUARD_KEYS:
            G[k].append(g[k])
    return env, G


class OracleG(S.Oracle):
    """e6p_screen.Oracle with the v2 guardrail admissibility (doc sec. 1). ``G=None`` disables it (then the class is
    decision-identical to v1; checked by ``check_disabled_matches_v1``). Only ``roll`` differs; choose / act / run are
    inherited unchanged."""

    def __init__(self, env, xapps, Ef, EA, G=None, **kw):
        super().__init__(env, xapps, Ef, EA, **kw)
        self.G = G
        self.rej = {"energy_only": 0, "guard_only": 0, "both": 0}
        self.rej_keys = {k: 0 for k in GUARD_KEYS}

    def guard_at(self, sec):
        n = len(self.G[GUARD_KEYS[0]])
        sec = min(int(sec), n - 1)
        return {k: self.G[k][sec] for k in GUARD_KEYS}

    def roll(self, plan, obs, reseed=None):
        sim_ = self.env.copy(reseed=reseed)
        s0 = dict(sim_.plant.sla)
        hs = dict(self.hs)
        sim_.step_apply(S.decide(plan, obs, self.site, self.D, sim_.last_change, self.rb_at, True, hs))
        for _ in range(self.H - 1):
            if sim_.sec >= sim_.total_s:
                break
            o = sim_.step_propose()
            sim_.step_apply(S.decide(plan, o, self.site, self.D, sim_.last_change, self.rb_at, False, hs))
        self.n_roll += 1
        sla = sim_.plant.sla
        e_ok = bool(sla["energy_j"] <= self.bound(sim_.sec) + 1e-6)
        ok = e_ok
        if self.G is not None:
            g_ok, bad = guard_admissible(guard_values(sla), self.guard_at(sim_.sec))
            ok = e_ok and g_ok
            if reseed is None:                                   # diagnostics on true-tape trials only
                if not e_ok or not g_ok:
                    self.rej["both" if (not e_ok and not g_ok) else "guard_only" if e_ok else "energy_only"] += 1
                for k in bad:
                    self.rej_keys[k] += 1
        return (sla["prot_viol"] - s0["prot_viol"], sla["viol_ue_s"] - s0["viol_ue_s"], ok)


def oracle_job(pair, stratum, seed, lf, smoke, record_scores=True):
    """v1 oracle_job with the AA reference run traced (change (a)); references and churn cap as v1."""
    cfg = S.make_cfg(pair, stratum, seed, lf, smoke)
    t = time.time()
    ef_env, Ef = S.run_env(cfg, S.freeze, energy_trace=True)
    ea_env, EA = S.run_env(cfg, S.make_arbiter(S.a_arm(pair)), energy_trace=True)
    aa_env, G = run_env_traces(cfg, None)
    cap = aa_env.stats["changes"]
    t_ref = time.time() - t
    env = E6Env(cfg, log=False, wg3=True, churn_cap=cap)
    orc = OracleG(env, S.deployed(pair), Ef, EA, G=G, **S.ORACLE)
    t = time.time()
    orc.run()
    aa_end, or_end = guard_values(aa_env.plant.sla), guard_values(env.plant.sla)
    rec = {"churn_cap": cap, "n_roll": orc.n_roll, "picks": orc.picks, "n_epochs": len(orc.epochs),
           "epoch_secs": orc.epochs, "ref_secs": round(t_ref, 1), "oracle_secs": round(time.time() - t, 1),
           "v2_guard": {"rule": "value_k <= 1.10 AA_k(t), 1 if AA_k(t) == 0", "rejections": orc.rej,
                        "rejections_by_key": orc.rej_keys, "AA_end": aa_end, "oracle_end": or_end,
                        "end_ratio": {k: (or_end[k] / aa_end[k] if aa_end[k] > 0 else None) for k in GUARD_KEYS}}}
    if record_scores:
        rec.update(S.outcome(env), deviations=orc.deviations,
                   ref_check={"freeze_prot_viol": ef_env.plant.sla["prot_viol"],
                              "A_prot_viol": ea_env.plant.sla["prot_viol"],
                              "AA_prot_viol": aa_env.plant.sla["prot_viol"],
                              "AA_energy_j": aa_env.plant.sla["energy_j"]})
    else:
        rec["n_deviations"] = len(orc.deviations)
    return rec


# ---------------------------------------------------------------------------------------------- (b)/(c) units
def units_for(stage, state):
    U = []
    if stage == "1":
        U += [[("1", pair, s, screen_seed(s, j), "qacm") for j in range(S.N_SCREEN)] for pair, s in CELLS]
        for s in sorted({s for _, s in CELLS}):
            for j in range(S.N_SCREEN):
                seed = screen_seed(s, j)
                U.append([("1", "*", s, seed, "freeze")])
                U += [[("1", pair, s, seed, a)] for pair, s2 in CELLS if s2 == s for a in S.stage1_arms(pair)
                      if a != "qacm"]
    elif stage == "2":
        U = [[("2", pair, s, screen_seed(s, j), "oracle")] for pair, s in CELLS for j in range(S.N_SCREEN)]
    elif stage == "3":
        gate = [tuple(c) for c in state.get("stage2_crit12", [])]
        U = [[("3", pair, s, screen_seed(s, j), "hind")] for pair, s in gate if (pair, s) in CELLS
             for j in range(S.N_SCREEN)]
    else:
        raise SystemExit("E6P-v2 runs stages 1, 2, 3 only (stage 0 is inherited from v1)")
    for u in U:
        for key in u:
            assert SCREEN_RANGE[0] <= key[3] < SCREEN_RANGE[1], key        # never the v2 confirmation range
    return U


# ---------------------------------------------------------------------------------------------- summary
def redraw_advantage(recs, pair, s):
    """(d) descriptive: summed re-drawn (and true-tape) advantage of the oracle over AA on protected violated UE-s,
    per-seed sums bootstrapped over seeds."""
    per_r = np.array([sum(d["aa_redraw"][0] - d["plan_redraw"][0] for d in r.get("deviations", [])) for r in recs])
    per_t = np.array([sum(d["aa_true"][0] - d["plan_true"][0] for d in r.get("deviations", [])) for r in recs])
    rng = np.random.default_rng([ADV_TAG, S.PAIRS[pair]["idx"], s])
    idx = rng.integers(0, len(recs), (S.N_BOOT, len(recs)))
    b = per_r[idx].sum(1)
    return {"sum_redraw": float(per_r.sum()), "sum_redraw_lb90": float(np.quantile(b, S.LB_Q)),
            "sum_true": float(per_t.sum()), "per_seed_redraw": per_r.tolist(), "per_seed_true": per_t.tolist(),
            "n_deviations": int(sum(len(r.get("deviations", [])) for r in recs)), "descriptive_only": True}


def summary(paths, write=False, json_out=None, allow_smoke=False):
    recs = [r for p in paths for r in S._read(p)]
    recs = [r for r in recs if allow_smoke or not r.get("smoke")]
    state = load_state()
    new_state = dict(state)
    report = {"protocol": PROTOCOL, "inherits": S.PROTOCOL, "e6p_spec_commit": S.E6P_SPEC_COMMIT,
              "frozen_sha256": FROZEN_SHA256, "notes": S.IMPLEMENTATION_NOTES + V2_NOTES}
    heads = [r for r in recs if r.get("kind") == "header"]
    report["platforms"] = sorted({json.dumps({k: (h.get("numeric_env") or {}).get(k) for k in ("platform", "numpy")})
                                  for h in heads})
    R = {tuple(r["key"]): r for r in recs if r.get("kind") == "job"}
    rows = {}
    for pair, s in CELLS:
        r = S.analyse_pair_stratum(R, pair, s)
        orc = [R.get(("2", pair, s, screen_seed(s, j), "oracle")) for j in range(S.N_SCREEN)]
        if r is not None and all(orc):
            r["v2_redraw_advantage"] = redraw_advantage(orc, pair, s)
            aa = [R.get(("1", pair, s, screen_seed(s, j), "noarb")) for j in range(S.N_SCREEN)]
            if all(aa):
                Po, Pa = S._pool(orc), S._pool(aa)
                r["v2_oracle_guard_ratio"] = {k: (Po[k] / Pa[k] if Pa[k] > 0 else None) for k in GUARD_KEYS}
            rej = {}
            for o in orc:
                for k, v in o.get("v2_guard", {}).get("rejections", {}).items():
                    rej[k] = rej.get(k, 0) + v
            r["v2_trial_rejections"] = rej
            r["v2_oracle_n_roll"] = int(sum(o["n_roll"] for o in orc))
        rows[(pair, s)] = r
    pairs_out = {}
    for pair in dict.fromkeys(p for p, _ in CELLS):
        prow = {s: rows[(p, s)] for p, s in CELLS if p == pair}
        v = S.verdict(prow, False)
        pairs_out[pair] = {"rows": prow, "verdict": v}
        print(f"\n== v2 {pair} ({'+'.join(S.PAIRS[pair]['A'])} vs {'+'.join(S.PAIRS[pair]['B'])}): {v}")
        for s, r in prow.items():
            if r is None or not r.get("complete"):
                print(f"  stratum {s} {S.STRATA[s]}: incomplete {r.get('missing') if r else '(no freeze)'}")
                continue
            print(f"  stratum {s} {S.STRATA[s]}: screenable={r['screenable']} AA matched={r['AA_matched']} "
                  f"V_AA={r['V_AA']:.1f} V_ref={r['V_ref']:.1f} ({r['ref_arm']}) Lambda={r['Lambda']:.3f} "
                  f"LB90={r['lb90_diff']:.1f} c1={r['c1']} R_static={r['R_static']} ({r['R_static_arm']})"
                  + (f" R_or={r['R_or']:.3f} CI{r['R_or_ci90']} c2={r['c2']} rho={r['rho_sign']:.3f} "
                     f"(n={r['n_deviations']}) c4={r['c4']}" if "R_or" in r else "")
                  + (f" headroom={r['headroom']:.3f} c3={r['c3']}" if "c3" in r else ""))
            if "oracle" in r["arms"]:
                t = r["arms"]["oracle"]
                print(f"    O_tape: raw R={t['R']:.3f} matched={t['matched']} guard={t['guard']} "
                      f"eligible={t['eligible']} guard ratios={r.get('v2_oracle_guard_ratio')} "
                      f"trial rejections={r.get('v2_trial_rejections')}")
            if "v2_redraw_advantage" in r:
                a = r["v2_redraw_advantage"]
                print(f"    [descriptive] summed re-drawn advantage={a['sum_redraw']:.1f} "
                      f"(seed-bootstrap LB90={a['sum_redraw_lb90']:.1f}), true-tape={a['sum_true']:.1f}, "
                      f"n_dev={a['n_deviations']}")
            for a, t in r["arms"].items():
                print(f"    [{t['group']:>2}] {a:40s} V={t['V']:8.2f} R={t['R']:7.3f} ret={t['retention']:7.3f} "
                      f"matched={int(t['matched'])} guard={int(t['guard'])} elig={int(t['eligible'])} "
                      f"churn={t['changes']:.0f}")
    report["pairs"] = {p: {"verdict": v["verdict"], "rows": {str(s): r for s, r in v["rows"].items()}}
                       for p, v in pairs_out.items()}
    stage1_done = all(r is not None and r.get("complete") for r in rows.values())
    if stage1_done:
        crit1 = [[p, s] for (p, s), r in rows.items() if r["c1"]]
        new_state["stage1_crit1"] = crit1
        if all("R_or" in rows[(p, s)] for p, s in crit1):
            new_state["stage2_crit12"] = [[p, s] for p, s in crit1 if rows[(p, s)]["c2"]]
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    if write:
        new_state["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new_state["sources"] = sorted({os.path.abspath(f) for p in paths for f in glob.glob(p)})
        write_state(new_state)
        print("v2 state ->", STATE_JSON, {k: new_state.get(k) for k in ("stage1_crit1", "stage2_crit12")})
    return report


# ---------------------------------------------------------------------------------------------- checks
def _dev_cfg(scored=20.0, seed=5, pair="P3", stratum=3, warmup=20.0):
    """Smoke-length E6-P config on a DEV seed (0-30), v2 cell P3 surge-L40 by default."""
    import dataclasses
    assert 0 <= seed <= 30, "checks use E6 DEV seeds only"
    lf = S.lf_of(load_state(), stratum)
    return dataclasses.replace(S.make_cfg(pair, stratum, seed, lf, smoke=True), scored_s=float(scored),
                               warmup_s=float(warmup))


def check_guard_rejects(seed=5):
    """A trial whose rollout RLF exceeds the bound is rejected (constraint on); the same trial is admissible with the
    constraint off and with a non-binding RLF trace. The RLF excess is injected deterministically (+5 RLF on the
    rollout copy); DEV seed, smoke-length episode."""
    cfg = _dev_cfg(seed=seed)
    aa_env, G = run_env_traces(cfg, None)
    env = E6Env(cfg, log=False, wg3=True)
    while env.sec < int(cfg.warmup_s):
        env.step(None)
    obs = env.step_propose()
    orig = env.copy

    def copy_plus5(reseed=None):
        c = orig(reseed=reseed)
        c.plant.sla["rlf"] += 5
        return c

    env.copy = copy_plus5
    n = len(G["rlf"])
    big = {k: [1e18] * n for k in GUARD_KEYS}
    res = {}
    for name, g in (("off", None), ("nonbinding", big),
                    ("rlf_bound_2.2", dict(big, rlf=[2.0] * n)),          # bound 1.10 * 2 = 2.2 < injected >= 5
                    ("rlf_zero_aa", dict(big, rlf=[0.0] * n))):           # +1 slack: bound 1 < 5
        orc = OracleG(env, S.deployed("P3"), None, None, G=g, D=20, H=5, n_glob=1, n_loc=1)
        pv, vi, ok = orc.roll(orc.aa(), obs)
        res[name] = {"ok": ok, "rej": dict(orc.rej), "rej_keys": dict(orc.rej_keys)}
    env.copy = orig
    assert res["off"]["ok"] and res["nonbinding"]["ok"], res
    assert not res["rlf_bound_2.2"]["ok"] and res["rlf_bound_2.2"]["rej_keys"]["rlf"] == 1, res
    assert not res["rlf_zero_aa"]["ok"] and res["rlf_zero_aa"]["rej"]["guard_only"] == 1, res
    # pure-function edge cases
    z = {k: 0.0 for k in GUARD_KEYS}
    assert guard_admissible(dict(z, rlf=1.0), z)[0] and not guard_admissible(dict(z, rlf=2.0), z)[0]
    assert guard_admissible(dict(z, ll_viol=11.0), dict(z, ll_viol=10.0))[0]
    assert guard_admissible(dict(z, ll_viol=11.1), dict(z, ll_viol=10.0)) == (False, ["ll_viol"])
    return res


def check_disabled_matches_v1(seed=5, warmup=120.0, scored=20.0, H=15, n_glob=2, n_loc=1):
    """With the guardrail constraint disabled (G=None) the v2 oracle reproduces v1's per-second decisions, rollout
    count, picks, deviations and final plant counters exactly (DEV seed, smoke-length, reduced budget); also with a
    non-binding trace (constraint code active, never binding). Defaults give 2 decision epochs and 1 deviation from
    accept-all on DEV seed 5 (asserted, so the non-accept-all path is exercised)."""
    cfg = _dev_cfg(scored=scored, seed=seed, warmup=warmup)
    _, Ef = S.run_env(cfg, S.freeze, energy_trace=True)
    _, EA = S.run_env(cfg, S.make_arbiter(S.a_arm("P3")), energy_trace=True)
    n = len(Ef)
    kw = dict(D=20, H=H, n_glob=n_glob, n_loc=n_loc)

    def go(cls, **extra):
        env = E6Env(cfg, log=False, wg3=True)
        orc = cls(env, S.deployed("P3"), Ef, EA, **kw, **extra)
        log = []
        act = orc.act

        def rec(obs):
            d = act(obs)
            log.append(copy.deepcopy(d))
            return d

        orc.act = rec
        orc.run()
        return {"decisions": log, "n_roll": orc.n_roll, "picks": orc.picks, "deviations": orc.deviations,
                "sla": {k: float(v) for k, v in env.plant.sla.items() if np.isscalar(v)}, "plan": orc.plan}

    v1 = go(S.Oracle)
    off = go(OracleG, G=None)
    nb = go(OracleG, G={k: [1e18] * n for k in GUARD_KEYS})
    for name, v in (("G=None", off), ("nonbinding", nb)):
        for f in ("decisions", "n_roll", "picks", "deviations", "sla", "plan"):
            assert v[f] == v1[f], (name, f)
    assert v1["deviations"], "no deviation from accept-all: the check would not exercise a non-trivial plan"
    return {"n_roll": v1["n_roll"], "n_seconds": len(v1["decisions"]), "picks": v1["picks"],
            "n_deviations": len(v1["deviations"]), "non_accept_decisions":
                sum(dd != "accept" for d in v1["decisions"] for dd in d["decisions"])}


def selftest():
    t = time.time()
    a = check_guard_rejects()
    print(json.dumps({"check_guard_rejects": "PASS", "detail": a, "secs": round(time.time() - t, 1)}), flush=True)
    t = time.time()
    b = check_disabled_matches_v1()
    print(json.dumps({"check_disabled_matches_v1": "PASS", "detail": b, "secs": round(time.time() - t, 1)}), flush=True)


def list_jobs():
    state = load_state()
    print("v2 state:", {k: state.get(k) for k in ("load_factor", "cells", "stage1_crit1", "stage2_crit12")})
    for st in ("1", "2", "3"):
        U = units_for(st, state)
        arms = {}
        for u in U:
            for key in u:
                g = key[4].split(":")[0]
                arms[g] = arms.get(g, 0) + 1
        seeds = sorted({key[3] for u in U for key in u})
        print(f"{st}: {len(U)} units, {sum(len(u) for u in U)} jobs {arms} seeds "
              f"{seeds[:1]}..{seeds[-1:]} ({len(seeds)})")


# ---------------------------------------------------------------------------------------------- install + CLI
def _install():
    """The complete list of e6p_screen overrides (module globals looked up at call time by run / run_job / header /
    analyse_pair_stratum)."""
    S.PROTOCOL = PROTOCOL
    S.FROZEN_SHA256 = FROZEN_SHA256
    S.STATE_JSON, S.STATE_PY = STATE_JSON, STATE_PY
    S.load_state, S.write_state = load_state, write_state
    S.screen_seed = screen_seed
    S.units_for = units_for
    S.oracle_job = oracle_job


_install()


def cli(argv, stage=None):
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    flags = {x for x in rest if x in ("--smoke", "--write-state", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        st = a.get("--stage") or stage
        if st not in ("1", "2", "3"):
            raise SystemExit("--stage in 1/2/3 (or use the e6p_v2_stage<S>.py wrappers)")
        S.run(st, a["--part"], a["--out"], smoke="--smoke" in flags)
    elif cmd == "summary":
        files, js = [], None
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x != "--in":
                files.extend(f for f in x.split(",") if f)
        summary(files, write="--write-state" in flags, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "init-state":
        init_state()
    elif cmd == "list":
        list_jobs()
    elif cmd == "freeze-check":
        print(json.dumps({"docs": S.docs_check(), "protocol": PROTOCOL}))
        S.check_frozen(S.make_cfg("P3", 3, screen_seed(3, 0), 1.0))
        print("config == frozen values: OK")
    elif cmd == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
