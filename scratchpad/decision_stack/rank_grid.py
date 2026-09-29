"""Step-5 ranking-evaluation grid (cloud/local shards, resumable): oracle panels on the DIAGNOSTIC v3 policy seeds.

  # 1. locally, ONCE, from the pulled v3 fit npz (collect_grid.py output): fit every ablation model into a bundle
  PYTHONPATH=. python scratchpad/decision_stack/rank_grid.py fit --npz DIR [--selectors none,topology,all]
                      [--weights NAME=path.npy,...] [--members 5] [--out scratchpad/decision_stack/rank_models.pkl]
  # 2. ship the bundle next to this script (or point RANK_MODELS / --models at it) and run shards
  PYTHONPATH=. python scratchpad/decision_stack/rank_grid.py list
  PYTHONPATH=. python scratchpad/decision_stack/rank_grid.py run --part i/k --out FILE.jsonl [--models PATH] [--smoke]
  PYTHONPATH=. python scratchpad/decision_stack/rank_grid.py summary --in FILE.jsonl[,FILE2.jsonl...]
  PYTHONPATH=. python scratchpad/decision_stack/rank_grid.py estimate          # measured cost per job (DEV seed 16)

Jobs = diagnostic policy seeds (collect.dev_seed("policy", scn, load, j), j = 20..29) x 6 strata = 60 episodes.
Each job runs the episode under accept-all (the v3 paired-reference semantics) and, at the 6 labelled slot starts
(t = 120 + 90 k, k = 0..5), scores the declared candidate panel (rank_eval.CandidateSpec) with
TrueSimWM(continuation="accept_all"), D = 20, H = 90, and every bundled model (support gate = SupportRule(20, 20, 10)
on the model's fit counts, per stratum). No gate calibration here (calib/eval seeds are a separate contract step).
Output: one {"type": "slot"} line per slot, written together with a closing {"type": "job"} line after the job; a
job is skipped iff its "job" line exists (partial jobs are recomputed; ``summary`` ignores slots of unfinished jobs).
Nothing here is tuned on these outcomes.

Cost (TrueSim rollouts dominate): per job ~ n_cand x 6 slots x 89 s rollouts + the 570 s + 89 s main run.
Measured locally (``estimate``) ~ 0.025 s per simulated second -> ~2.3 s per 89 s rollout; with ~30 candidates
(21 fixed/random + up to 4 per model x 3 models, deduplicated): measured 7.1 min/job, i.e. ~7.1 CPU-h for
60 jobs (x ~1.5 on Kaggle CPUs; run k shards in parallel).
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys
import time

import numpy as np

from cdd_oran.decision import collect as CO
from cdd_oran.decision import rank_eval as RE
from cdd_oran.envs.e6 import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = os.path.join(HERE, "rank_models.pkl")
WARMUP, SCORED, D, H = 120.0, 600.0, 20, 90
SPEC = RE.CandidateSpec()                  # top_k 4, n_random 6, single modes reject/half/lock, freeze, rollback


def jobs():
    return [(s, ld, CO.dev_seed("policy", s, ld, j)) for s, ld in itertools.product(CO.DEV_SCENARIOS, CO.DEV_LOADS)
            for j in range(CO.DEV_FIT, CO.DEV_PER_STRATUM)]


def name(job):
    return f"rank_{job[0]}_{job[1]}_s{job[2]}"


def cfg_of(scn, load, seed, scored=SCORED):
    return C.E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=WARMUP, scored_s=scored,
                      scenario=scn)


# ------------------------------------------------------------------------------------------------ fit (local, once)
def fit(npz_dir, out, selectors=("none", "topology", "all"), weights=(), members=5):
    paths = sorted(p for p in glob.glob(os.path.join(npz_dir, "v3_*.npz"))
                   if CO.dev_stratum(int(os.path.basename(p).rsplit("_s", 1)[1][:-4]))["split"] == "fit")
    if not paths:
        raise SystemExit(f"no v3 fit npz in {npz_dir}")
    bundle = {}
    for sel in selectors:
        t = time.time()
        bundle[sel] = RE.fit_from_npz(paths, H=H, selector=sel, n_members=members)
        print(json.dumps({"model": sel, **{k: bundle[sel].manifest[k] for k in ("n_episodes", "n_rows", "strata")},
                          "secs": round(time.time() - t, 1)}), flush=True)
    for w in weights:                                        # NAME=path.npy: (R, R) graph / SHAP context weights
        nm, p = w.split("=", 1)
        bundle[nm] = RE.fit_from_npz(paths, H=H, selector="weights", weights=np.load(p), n_members=members)
        bundle[nm].manifest["weights_file"] = os.path.basename(p)
    RE.save_bundle(out, bundle)
    print(f"bundle -> {out} ({len(paths)} fit episodes, models {sorted(bundle)})")


# ------------------------------------------------------------------------------------------------ run (cloud)
def _done(out):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "job":
                done.add(r["job"])
    return done


def run(part, out, models_path, smoke=False):
    i, k = map(int, part.split("/"))
    models = RE.load_bundle(models_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = _done(out)
    for j, job in enumerate(jobs()):
        if j % k != i or name(job) in done:
            continue
        t = time.time()
        cfg = cfg_of(*job, scored=150.0 if smoke else SCORED)
        rows = RE.oracle_panel(cfg, None, SPEC, models, D=D, H=H)
        for r in rows:
            r["job"] = name(job)
        tail = {"type": "job", "job": name(job), "scenario": job[0], "load": job[1], "seed": job[2],
                "split": CO.dev_stratum(job[2])["split"], "n_slots": len(rows),
                "n_cand": [len(r["cands"]) for r in rows], "models": sorted(models),
                "bundle": os.path.basename(models_path), "smoke": smoke, "secs": round(time.time() - t, 1)}
        with open(out, "a", newline="\n") as f:
            for r in rows + [tail]:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({x: tail[x] for x in ("job", "n_slots", "n_cand", "secs")}), flush=True)
        if smoke:
            break


def summary(paths):
    rows, done = [], set()
    for p in paths:
        done |= _done(p)
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    rows = [r for r in rows if r.get("type") == "slot" and r.get("job") in done]
    s = RE.summarize(rows)
    for nm, v in s.items():
        p = v["pooled"]
        print(nm, {"slots": p["n_slots"], "episodes": p["n_episodes"],
                   **{m: [round(p[m][x], 3) for x in ("mean", "lo", "hi")] for m in RE.METRICS},
                   "deviate": p["deviate"], "support_abstain": p["support_abstain"], "ood_abstain": p["ood_abstain"],
                   "step5_criterion": p["step5_criterion"]})
    return s


def estimate(n_cand=30, n_slots=6):
    """Time the main run + one oracle rollout on DEV seed 16 (not a registered policy seed) -> cost per job."""
    from cdd_oran.decision import plans as P
    from cdd_oran.decision.adapters.e6 import region_map
    from cdd_oran.decision.world_model import DecisionContext, TrueSimWM
    from cdd_oran.envs.e6.env import E6Env
    env = E6Env(cfg_of("base", "high", 16), log=False, wg3=True)
    site = region_map(env)
    t = time.time()
    while env.sec < int(WARMUP):
        o = env.step_propose()
        env.step_apply({"decisions": ["accept"] * len(o["requests"]), "writes": [], "rollback": []})
    per_s = (time.time() - t) / WARMUP
    o = env.step_propose()
    regs = sorted({int(x) for x in site})
    ctx = DecisionContext(o, site, regs, H, float(D), RE.LAM_E, RE.W_LL, dict(env.last_change), {}, env, {})
    t = time.time()
    TrueSimWM("accept_all").score(ctx, [P.network(regs, P.uniform("reject"))])
    roll = time.time() - t
    main = per_s * (WARMUP + 90 * (n_slots - 1) + H)
    job = main + n_cand * n_slots * roll
    print(json.dumps({"sec_per_sim_s": round(per_s, 4), "rollout_s": round(roll, 2), "main_run_s": round(main, 1),
                      "n_cand": n_cand, "n_slots": n_slots, "job_min": round(job / 60, 1),
                      "grid_cpu_h": round(job * len(jobs()) / 3600, 1), "jobs": len(jobs())}))


if __name__ == "__main__":
    cmd, kw = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2], strict=False))
    if cmd == "list":
        js = jobs()
        print(len(js), "jobs", {f"{s}-{ld}": sum(1 for x in js if x[:2] == (s, ld))
                                for s, ld in itertools.product(CO.DEV_SCENARIOS, CO.DEV_LOADS)})
        for jb in js[:3]:
            print(" ", name(jb))
    elif cmd == "fit":
        fit(kw["--npz"], kw.get("--out", DEFAULT_MODELS), tuple(kw.get("--selectors", "none,topology,all").split(",")),
            tuple(w for w in kw.get("--weights", "").split(",") if w), int(kw.get("--members", 5)))
    elif cmd == "run":
        mp = kw.get("--models") or os.environ.get("RANK_MODELS") or DEFAULT_MODELS
        run(kw["--part"], kw["--out"], mp, smoke="--smoke" in sys.argv)
    elif cmd == "summary":
        summary(kw["--in"].split(","))
    elif cmd == "estimate":
        estimate()
    else:
        raise SystemExit(f"unknown command {cmd}")
