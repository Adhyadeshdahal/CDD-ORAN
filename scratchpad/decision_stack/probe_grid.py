"""DEV probe-campaign grid (operator DEV probes, NOT WG3 policy data; cloud/local shards, resumable).

  PYTHONPATH=. python scratchpad/decision_stack/probe_grid.py list [--mode main|tranche|audit]
  PYTHONPATH=. python scratchpad/decision_stack/probe_grid.py run --part i/k --out DIR_or_res_i.jsonl [--mode M]
--out DIR           -> summaries appended to DIR/res_<i>.jsonl, episodes in DIR/<job>_{probe,ref}.npz
--out FILE.jsonl    -> summaries appended to FILE.jsonl, episodes in <dir of FILE>/npz/ (Kaggle pattern)
A job = one probe episode + its paired same-seed accept-all no-probe reference; it is skipped if its summary line
exists AND both npz load. Jobs = 6 strata (scenario x load) x 30 probe seeds, seed = 110000 + 30*stratum + j
(SEED_REGISTRY E6 dev_reserved.probe_episodes; j < 20 fit, j >= 20 untouched diagnostics). mix M4, mobility mixed,
warm-up 120 s, scored probe.PROBE_SCORED_S = 1800 s (~6.6 blocks/episode). Probe config = ProbeConfig() defaults
(e6-probe/3, predeclared; abort table frozen from the audit references).
Modes:
  main     the 180 jobs above.
  tranche  CONDITIONAL probe-only tranche, 30 more seeds/stratum = 110180 + 30*stratum + j (run ONLY if
           probe.tranche_decision(first_stage(main fit episodes), simulated power) triggers; blinded inputs only).
  audit    carryover audit on fit seeds j < 5 (30 jobs): base probe episode + same-seed REPLAY with block 0's arm
           flipped (probe <-> sham) + no-probe reference, saved as *_base / *_replay / *_ref.npz; analyse with
           ``probe_audit.py analyze --dir <npz dir>`` (contamination, first stage, sham-placebo, lag-1 CRT).
           Replays never enter inference.
Analysis (not here): crt.build_block_data(probe traces) -> crt.run_crt; placebo null =
crt.build_block_data(reference traces, schedule_from=probe traces); cost = the jsonl "cost" field.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

from cdd_oran.decision import probe as P
from cdd_oran.decision.trace import Trace
from cdd_oran.envs.e6 import config as C

SCENARIOS = ("base", "surge", "mistune")
LOADS = ("medium", "high")
PER_STRATUM, FIT, BASE = 30, 20, 110000
WARMUP, SCORED = 120.0, P.PROBE_SCORED_S
AUDIT_J = 5


def seed_of(scenario, load, j, base=BASE):
    return base + PER_STRATUM * (2 * SCENARIOS.index(scenario) + LOADS.index(load)) + int(j)


try:                                                # the registry mapping lives in decision.collect (Worker J)
    from cdd_oran.decision.collect import dev_seed
    assert all(dev_seed("probe", s, ld, j) == seed_of(s, ld, j) for s in SCENARIOS for ld in LOADS for j in (0, 29))
except ImportError:
    pass


def jobs(mode="main"):
    base = P.TRANCHE_SEED_BASE if mode == "tranche" else BASE
    n = AUDIT_J if mode == "audit" else PER_STRATUM
    return [(s, ld, j, base, mode) for s in SCENARIOS for ld in LOADS for j in range(n)]


def name(job):
    s, ld, j, base, mode = job
    return f"{'audit' if mode == 'audit' else 'probe'}_{s}_{ld}_s{seed_of(s, ld, j, base)}"


def cfg_of(job):
    s, ld, j, base, _ = job
    return C.E6Config(seed=seed_of(s, ld, j, base), load=ld, mobility="mixed", mix="M4", warmup_s=WARMUP,
                      scored_s=SCORED, scenario=s)


def run_audit_job(job, npz_dir):
    import probe_audit as PA  # sibling script
    t = time.time()
    cfg = cfg_of(job)
    base, rep = PA.run_pair(cfg, 0)
    rep.meta["probe"]["flip"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = P.run_reference(cfg)
    stem = os.path.join(npz_dir, name(job))
    base.to_npz(stem + "_base.npz")
    rep.to_npz(stem + "_replay.npz")
    ref.to_npz(stem + "_ref.npz")
    c = PA.contamination(base, rep, 0)
    return {"job": name(job), "mode": "audit", "seed": cfg.seed, "secs": round(time.time() - t, 1),
            "npz": [os.path.basename(stem + x) for x in ("_base.npz", "_replay.npz", "_ref.npz")],
            "contamination": c}


def run_job(job, npz_dir):
    if job[4] == "audit":
        return run_audit_job(job, npz_dir)
    t = time.time()
    cfg = cfg_of(job)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tr = P.run_probe_episode(cfg)
        ref = P.run_reference(cfg)
    base = os.path.join(npz_dir, name(job))
    tr.to_npz(base + "_probe.npz")
    ref.to_npz(base + "_ref.npz")
    cost = P.collection_cost(tr, ref)
    a = tr.arrays
    return {"job": name(job), "mode": job[4], "scenario": job[0], "load": job[1], "seed": cfg.seed, "j": job[2],
            "split": "fit" if job[2] < FIT else "diag", "probe_version": P.PROBE_VERSION,
            "npz": [os.path.basename(base + "_probe.npz"), os.path.basename(base + "_ref.npz")],
            "secs": round(time.time() - t, 1), "blocks": int(len(a["blk_t0"])),
            "arms": [P.ARMS[i] for i in a["blk_arm"]], "cost": cost}


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


def _files(job, npz_dir):
    stem = os.path.join(npz_dir, name(job))
    return [stem + x for x in (("_base.npz", "_replay.npz", "_ref.npz") if job[4] == "audit"
                               else ("_probe.npz", "_ref.npz"))]


def run(part, out, mode="main"):
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
    for j, job in enumerate(jobs(mode)):
        if j % k != i or (name(job) in done and all(_ok(f) for f in _files(job, npz_dir))):
            continue
        rec = run_job(job, npz_dir)
        with open(res, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if mode == "audit":
            print(json.dumps({"job": rec["job"], "secs": rec["secs"]}), flush=True)
            continue
        c = rec["cost"]
        print(json.dumps({"job": rec["job"], "blocks": rec["blocks"], "excess_svr": round(c["excess_svr"], 2),
                          "aborts": c["aborts_treated"] + c["aborts_sham"], "secs": rec["secs"]}), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    a = dict(zip(sys.argv[2::2], sys.argv[3::2], strict=True))
    mode = a.get("--mode", "main")
    if sys.argv[1] == "list":
        js = jobs(mode)
        print(len(js), f"{mode} jobs", "(base + replay + reference each)" if mode == "audit" else
              "(probe + paired reference each)")
        for jb in js[:3] + js[-2:]:
            print(" ", name(jb))
    else:
        run(a["--part"], a["--out"], mode)
