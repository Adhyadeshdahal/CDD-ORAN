"""E6 DEV policy-data collection grid (cloud/local shards, resumable). One npz per episode + a JSON line.

  PYTHONPATH=. python scratchpad/decision_stack/collect_grid.py list [--regime v3,ref] [--split pilot|fit|diag|all]
  PYTHONPATH=. python scratchpad/decision_stack/collect_grid.py run --part i/k --out DIR_or_res_i.jsonl
                                                                   [--regime v3,ref] [--split pilot|fit|diag|all]
Contract: docs/benchmark/E6_COLLECTION_CONTRACT.md (e6-collect-v3). Run the PILOT first (--split pilot = fit seeds
j in {0, 1} of all six strata, v3 + ref) and check its predeclared failure criteria before any other split.
--out DIR           -> summaries appended to DIR/res_<i>.jsonl, episodes in DIR/<job>.npz
--out FILE.jsonl    -> summaries appended to FILE.jsonl, episodes in <dir of FILE>/npz/<job>.npz (Kaggle pattern)
A job is skipped if its summary line exists AND its npz loads.

Regimes (default "v3,ref"):
  v3  collector v3 (collect.collect_episode_v3: spaced slots, D = 20 treatment + 70 s network accept-all washout,
      slot_s = H = 90, DEFAULT_STEP_MIXTURE) on the registered policy seeds; every job records collect.audit_episode
      and, in the pilot, collect.continuation_check (TrueSimWM(accept_all) == realised, first 2 slots).
  v2  collector v2 (kept for reproducibility; its labels identify "plan, then logging policy", not the target) (collect.collect_episode_v2, DEFAULT_STEP_MIXTURE, D = 20, label_H = 90, quiet epochs off) on
      the registered E6 DEV policy seeds 100000-100179: 30 independent seeds per scenario x load stratum,
      seed = collect.dev_seed("policy", scenario, load, j), j < 20 fit / j >= 20 diagnostics (SEED_REGISTRY E6).
  ref paired accept-all reference on the SAME seeds (collect_episode_v2 with ACCEPT_ALL_MIXTURE; identical to no
      arbiter): collection-cost accounting only (excess SVR / violations / RLF / severe / energy / churn), never the
      inferential null.
  v1  the original pilot (collect.collect_episode, eps = 0.3) on DEV seeds 16-30; job names unchanged, so earlier
      v1 outputs stay resumable. Runs with half_rule="legacy" (the pilot's "half" semantics); v2/ref use the
      current "halve the slew rate" rule (plans.half_step).
Nothing here is tuned on these outcomes.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

from cdd_oran.decision import collect as CO
from cdd_oran.decision.trace import Trace
from cdd_oran.envs.e6 import config as C

SCENARIOS, LOADS = CO.DEV_SCENARIOS, CO.DEV_LOADS
V1_SEEDS, V1_EPS = range(16, 31), (0.3,)
WARMUP, SCORED, D, LABEL_H, SLOT_S = 120.0, 600.0, 20, 90, 90
REGIMES = ("v3", "v2", "ref", "v1")
PILOT_J = (0, 1)


def jobs(regimes=("v3", "ref"), split="all"):
    """Job = (regime, scenario, load, seed, eps); eps is only used by v1 (None otherwise)."""
    out = []
    for reg in regimes:
        if reg == "v1":
            out += [("v1", s, ld, sd, e) for s, ld, sd, e in itertools.product(SCENARIOS, LOADS, V1_SEEDS, V1_EPS)]
            continue
        if reg not in REGIMES:
            raise ValueError(f"unknown regime {reg!r}")
        for s, ld in itertools.product(SCENARIOS, LOADS):
            for j in range(CO.DEV_PER_STRATUM):
                if split == "all" or (split == "pilot" and j in PILOT_J) or                         (split in ("fit", "diag") and (split == "fit") == (j < CO.DEV_FIT)):
                    out.append((reg, s, ld, CO.dev_seed("policy", s, ld, j), None))
    return out


def name(job):
    reg, scn, load, seed, eps = job
    if reg == "v1":
        return f"{scn}_{load}_s{seed}_e{eps:g}"
    return f"{reg}_{scn}_{load}_s{seed}"


def cfg_of(job):
    _, scn, load, seed, _ = job
    return C.E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=WARMUP, scored_s=SCORED,
                      scenario=scn)


def run_job(job, npz_dir):
    t = time.time()
    reg = job[0]
    if reg == "v1":
        tr = CO.collect_episode(cfg_of(job), job[4], D=D, half_rule="legacy")   # reproduces the pilot
    elif reg == "v2":
        tr = CO.collect_episode_v2(cfg_of(job), mixture=CO.DEFAULT_STEP_MIXTURE, D=D, label_H=LABEL_H)
    else:
        mix = CO.DEFAULT_STEP_MIXTURE if reg == "v3" else CO.ACCEPT_ALL_MIXTURE
        tr = CO.collect_episode_v3(cfg_of(job), mixture=mix, D=D, slot_s=SLOT_S, H=LABEL_H)
    path = os.path.join(npz_dir, name(job) + ".npz")
    tr.to_npz(path)
    a, sc = tr.arrays, tr.meta["collector"]["score"]
    rec = {"job": name(job), "regime": reg, "scenario": job[1], "load": job[2], "seed": job[3], "eps": job[4],
           "npz": os.path.basename(path), "mb": round(os.path.getsize(path) / 1e6, 3),
           "secs": round(time.time() - t, 1),
           **{k: sc[k] for k in ("svr", "energy_kwh", "changes", "req", "severe", "ll_viol", "embb_viol",
                                 "outage_viol", "rlf_per_ue_h", "ho_per_ue_h", "pingpong")},
           "epochs": int(len(a["pol_t"])), "nondefault_share": float((a["pol_code"] != 0).mean()),
           "locks": int(a["pol_locks"].sum()), "rollbacks": int(a["pol_rb_applied"].sum())}
    if reg != "v1":
        rec["split"] = CO.dev_stratum(job[3])["split"]
        rec["changed_knob_s"] = changed_knob_seconds(tr)
        rec["elig_share_by_xapp"] = {x: float(a["pol_x_elig"][..., i].mean()) for i, x in enumerate(CO.P.XAPPS)}
        rec["audit"] = CO.audit_episode(tr)
    pilot = CO.dev_stratum(job[3])["j"] in PILOT_J if reg != "v1" else False
    if reg == "ref" and pilot and CO.dev_stratum(job[3])["j"] == PILOT_J[0]:     # contract F6: ref == no arbiter
        from cdd_oran.envs.e6.env import E6Env
        s0 = E6Env(cfg_of(job), log=False).run()
        rec["ref_equals_noarb"] = bool(all(abs(float(s0[k]) - float(sc[k])) < 1e-9
                                           for k in ("svr", "energy_kwh", "changes", "severe", "req")))
        rec["ref_no_deviation"] = bool((a["pol_code"] == 0).all() and (a["pol_prop"] == 1.0).all())
    if reg == "v3" and pilot:
        cc = CO.continuation_check(cfg_of(job), n_slots=2, D=D, slot_s=SLOT_S, H=LABEL_H)
        rec["continuation_check"] = [[float(x) for x in c] for c in cc]
        rec["continuation_ok"] = bool(len(cc) == 2 and all(o == r for _, o, r in cc))
    return rec


def changed_knob_seconds(tr):
    """Knob-seconds (scored window) during which a knob's value differs from its value at the end of warm-up."""
    a = tr.arrays
    t0 = int(tr.meta["cfg"]["warmup_s"])
    base = tr.config_at(t0 + 1)
    tot, cur = 0, base.copy()
    ts = a["cfg_d_t"]
    for t in range(t0 + 1, int(a["t"][-1]) + 1):
        m = ts == t
        cur[a["cfg_d_k"][m]] = a["cfg_d_v"][m]
        tot += int((abs(cur - base) > 1e-12).sum())
    return tot


def _paths(part, out):
    i = part.split("/")[0]
    if out.endswith(".jsonl"):
        return out, os.path.join(os.path.dirname(os.path.abspath(out)), "npz")
    return os.path.join(out, f"res_{i}.jsonl"), out


def _ok(path):
    try:
        Trace.from_npz(path)
        return True
    except Exception:  # noqa: BLE001 - any unreadable/partial file is redone
        return False


def run(part, out, regimes=("v3", "ref"), split="all"):
    i, k = map(int, part.split("/"))
    res, npz_dir = _paths(part, out)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(res)), exist_ok=True)
    done = set()
    if os.path.exists(res):
        for line in open(res):
            try:
                done.add(json.loads(line)["job"])
            except (json.JSONDecodeError, KeyError):
                continue
    for j, job in enumerate(jobs(regimes, split)):
        if j % k != i or (name(job) in done and _ok(os.path.join(npz_dir, name(job) + ".npz"))):
            continue
        rec = run_job(job, npz_dir)
        with open(res, "a", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({x: rec[x] for x in ("job", "svr", "changes", "nondefault_share", "mb", "secs")}),
              flush=True)


if __name__ == "__main__":
    cmd, kw = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2], strict=True))
    regs = tuple(kw.get("--regime", "v3,ref").split(","))
    spl = kw.get("--split", "all")
    if cmd == "list":
        js = jobs(regs, spl)
        print(len(js), "jobs", {r: sum(1 for x in js if x[0] == r) for r in regs})
        for jb in js[:5]:
            print(" ", name(jb))
    else:
        run(kw["--part"], kw["--out"], regs, spl)
