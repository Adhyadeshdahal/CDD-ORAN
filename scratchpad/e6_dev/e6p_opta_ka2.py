"""E6-P option (a) K-A2: MapGate v2 on DEV (after the K-A KILL; DEV only, NOT frozen).

Question: with a rule that reads the map's edges for EVERY judged KPI (pv SLA, e budget, v / rlf guardrail proxies;
cdd_oran/decision/mapgate.py MapGateV2, block comment "MapGate v2"), does the knockout-GT map give an ELIGIBLE referee
that beats wrong maps, a random map and a no-map blanket by a margin that would make a discovery comparison meaningful?

Plant, seeds, references: identical to e6p_opta_ka (P3 surge-L40, lf = e6p_state L40, 120 s warm-up + 600 s scored,
the referee acts from t = 0; DEV 184200-184239). REUSED records: the reference + context arms of runs/e6p-s2dev-1
(schema e6p-s2dev-rec/1) and the K-A arms never_sleep / GT / GT_own / GT_flip of runs/e6p-optaka-1 (schema
e6p-optaka-rec/1). ``noarb_repro`` re-runs noarb on REPRO_SEEDS here and must reproduce the reused noarb bit for bit.

New arms (all DirectionalUnitArbiter(MapGateV2(M, .05, k_conf = 1)), T = 60 s, open_rule "feasible", warmup_s = 0):
  GT2       M_GT = TRUE edges of gt_ext (e6p_opta_ka.M_GT)
  GT2_own   own_only(M_GT)
  GT2_flip  sign_flip(M_GT, carrier)
  GT2_rand  random_sized_map(M_GT, gt_ext cells, tag 6623): 28 random keys of the 60-cell universe, random signs,
            |beta| = that cell's |GT mean|
  blanket2  blanket_saving_map(): every energy-saving request (carrier off, sleep, ptx down) hurts own + nbr pv
Decision tables: ``list`` prints them (decision_table_v2).

Pre-declared K-A2 verdict (point estimates; 90 % CIs printed): CONTINUE iff GT2 is eligible, R(GT2) - R(c) >= .10 for
c in (GT2_own, GT2_flip, GT2_rand, blanket2), and R(GT2) >= R(never_sleep); else STOP.

CLI (repo root, PYTHONPATH=.; in the cloud bundle the same file is e6dev/e6p_opta_ka2.py):
  python scratchpad/e6_dev/e6p_opta_ka2.py run --part i/k --out F.jsonl [--smoke] [--short S] [--arms a,b]
  python scratchpad/e6_dev/e6p_opta_ka2_run.py run --part i/k --out F.jsonl          # cloud.py wrapper
  python scratchpad/e6_dev/e6p_opta_ka2.py summary --in F.jsonl [--ref REF.jsonl] [--ka KA.jsonl] [--json OUT]
  python scratchpad/e6_dev/e6p_opta_ka2.py list
Record schema "e6p-optaka2-rec/1": as e6p-optaka-rec/1 plus policy_counts["passed"] (opposite-direction requests
accepted inside a reject unit) and the MapGateV2 counts (defer_ / accept_ / release_<family>).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_opta_ka as KA  # noqa: E402
import e6p_screen as S  # noqa: E402  (sets ROOT / BUNDLE and sys.path)
import e6p_step2_dev as D  # noqa: E402

from cdd_oran.decision.mapgate import (  # noqa: E402
    K_CONF,
    THETA,
    blanket_saving_map,
    decision_table_v2,
    mapgate_v2_arbiter,
    own_only,
    random_sized_map,
    sign_flip,
)

PLAN = "MapGate v2 (cdd_oran/decision/mapgate.py) K-A2 on DEV 184200-239 after the K-A KILL; 2026-09-30, not frozen"
SCHEMA = "e6p-optaka2-rec/1"
SEEDS, REPRO_SEEDS = KA.SEEDS, KA.REPRO_SEEDS
REF_ARMS, CTX_ARMS = KA.REF_ARMS, KA.CTX_ARMS
KA_REUSE = ("never_sleep", "GT", "GT_own", "GT_flip")
NEW_ARMS = ("GT2", "GT2_own", "GT2_flip", "GT2_rand", "blanket2")
ARMS = NEW_ARMS + ("noarb_repro",)
RAND_TAG = 6623
MARGIN = 0.10
REF_DEFAULT = KA.REF_DEFAULT
KA_DEFAULT = os.path.join(HERE, "runs", "e6p-optaka-1", "all.jsonl")
N_BOOT = KA.N_BOOT

M_GT = KA.M_GT
MAPS = {"GT2": M_GT, "GT2_own": own_only(M_GT), "GT2_flip": sign_flip(M_GT, ("carrier",)),
        "GT2_rand": random_sized_map(M_GT, KA._cells(), RAND_TAG), "blanket2": blanket_saving_map()}


def tables() -> dict:
    return {a: {f"{k[0]}{'+' if k[1] > 0 else '-'}": v for k, v in decision_table_v2(m, THETA).items()}
            for a, m in MAPS.items()}


def make_arbiter(arm: str):
    if arm in MAPS:
        return KA.Counting(mapgate_v2_arbiter(MAPS[arm], THETA, K_CONF), is_unit=True)
    if arm == "noarb_repro":
        return None
    raise KeyError(arm)


def jobs() -> list:
    J = [(D.check_seed(sd), arm) for sd in SEEDS for arm in NEW_ARMS]
    J += [(D.check_seed(sd), "noarb_repro") for sd in REPRO_SEEDS]
    return J


def run_job(seed: int, arm: str, lf: float, smoke: bool = False, short=None) -> dict:
    import dataclasses
    D.check_seed(seed)
    if arm not in ARMS:
        raise KeyError(arm)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(KA.PAIR, KA.STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm)
    t_wall, t_cpu = time.time(), time.process_time()
    env, _ = S.run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    pc = {}
    if arb is not None:
        pc = dict(arb.n)
        pc["policy"] = dict(arb.inner.policy.n)
        pc["passed"] = {"all": int(arb.inner.passed)}
    return {"kind": "job", "schema": SCHEMA, "key": [seed, arm], "seed": seed, "cfg_seed": sd, "arm": arm,
            "smoke": smoke, "short": short, "episode_s": env.total_s, "warmup_s": cfg.warmup_s, "load_factor": lf,
            **S.outcome(env), "policy_counts": pc, "cpu_s": round(cpu, 2), "secs": round(time.time() - t_wall, 1),
            "rss_mb": D.peak_rss_mb()}


def run(part, out, smoke=False, short=None, arms=None):
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    reg = D.registry_check()
    if not smoke and reg is not None and not all(reg.values()) and not os.environ.get("E6P_S2_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block {D.REG_BLOCK} not registered in {D.REGISTRY_DOC}: {reg}")
    if arms:
        bad = [a for a in arms if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}; arms: {ARMS}")
    state = S.load_state()
    lf = S.lf_of(state, KA.STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header("opta_ka2", part, smoke, state)
    head.update(kind="header", schema=SCHEMA, plan=PLAN, driver="e6p_opta_ka2", registry=reg,
                gt_map_check=KA.check_gt_map(),
                consts={"pair": KA.PAIR, "stratum": KA.STRATUM, "load_factor": lf, "seeds": [SEEDS[0], SEEDS[-1]],
                        "repro_seeds": REPRO_SEEDS, "arms": ARMS, "theta": THETA, "k_conf": K_CONF, "T": 60.0,
                        "warmup_s_arbiter": 0.0, "open_rule": "feasible", "directional": True, "rand_tag": RAND_TAG,
                        "maps": {a: [[*kk, v] for kk, v in sorted(m.items())] for a, m in MAPS.items()},
                        "decision_tables": tables(), "smoke": {"short": short} if smoke else None,
                        "arms_filter": arms})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    t0, n, first_seed = time.time(), 0, None
    J = jobs()
    for u, (seed, arm) in enumerate(J):
        if u % k != i or (seed, arm) in done or (arms and arm not in arms):
            continue
        if smoke:
            first_seed = seed if first_seed is None else first_seed
            if seed != first_seed:
                break
        rec = run_job(seed, arm, lf, smoke, short)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("seed", "cfg_seed", "arm", "psvr", "energy_j", "policy_counts",
                                                  "cpu_s", "secs", "rss_mb")}), flush=True)
    S._append(out, {"kind": "close", "stage": "opta_ka2", "part": part, "n_jobs": n, "n_total": len(J),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def load_all(paths, ref_path, ka_path, allow_smoke=False):
    """({seed: {arm: rec}}, new headers, ref headers, K-A headers)."""
    R, hn, hr, hk = {}, [], [], []
    for r in S._read(ref_path):
        if r.get("kind") == "header" and r.get("schema") == D.SCHEMA:
            hr.append(r)
        if (r.get("kind") == "job" and r.get("schema") == D.SCHEMA and not r.get("smoke")
                and r["arm"] in REF_ARMS + CTX_ARMS):
            R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    if ka_path and os.path.exists(ka_path):
        for r in S._read(ka_path):
            if r.get("kind") == "header" and r.get("schema") == KA.SCHEMA:
                hk.append(r)
            if (r.get("kind") == "job" and r.get("schema") == KA.SCHEMA and not r.get("smoke")
                    and r["arm"] in KA_REUSE):
                R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    for p in paths:
        for r in S._read(p):
            if r.get("kind") == "header" and r.get("schema") == SCHEMA:
                hn.append(r)
            if r.get("kind") != "job" or r.get("schema") != SCHEMA or (r.get("smoke") and not allow_smoke):
                continue
            R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R, hn, hr, hk


def summary(paths, ref_path=REF_DEFAULT, ka_path=KA_DEFAULT, json_out=None, allow_smoke=False, n_boot=N_BOOT):
    R, hn, hr, hk = load_all(paths, ref_path, ka_path, allow_smoke)
    cm = KA.config_match(R, hn, hr)
    cm_ka = KA.config_match({}, hk, hr)
    f = D._f
    print(f"== E6-P option (a) K-A2 MapGate v2 (P3 surge-L40, DEV {SEEDS[0]}-{SEEDS[-1]}): {len(R)} seeds")
    print("   config match (new vs ref):", {k: v["ok"] for k, v in cm.items()})
    print("   config match (K-A vs ref):", {k: v["ok"] for k, v in cm_ka.items() if k not in ("noarb_repro_max_abs_diff", "jobs")})
    report = {"plan": PLAN, "schema": SCHEMA, "config_match": cm, "config_match_ka": cm_ka,
              "gt_map_check": KA.check_gt_map(), "decision_tables": tables(), "arms": {}}
    for a, t in report["decision_tables"].items():
        print(f"   {a:9s}", t)
    rows = [a for a in NEW_ARMS + KA_REUSE + REF_ARMS + CTX_ARMS if any(a in R[s] for s in R)]
    for arm in rows:
        comps = tuple(c for c in (("GT2",) if arm != "GT2" else ()) if any(c in R[s] for s in R))
        st = KA.arm_stats(R, arm, comps, n_boot)
        if st is None:
            continue
        for grp in ("defer", "defer_units", "units", "collateral", "policy", "passed"):
            st[grp] = KA._sum_counts(R, arm, grp)
        st["cpu_s_mean"] = float(np.mean([R[s][arm]["cpu_s"] for s in R if arm in R[s]]))
        report["arms"][arm] = st
    A = report["arms"]
    if A:
        r0 = next(iter(A.values()))
        print(f"   V_AA {r0['V_AA']:.2f}  V_ref {r0['V_ref']:.2f} ({r0['ref_arm']})  den {r0['den']:.2f}  n {r0['n_seeds']}")
        print(f"   {'arm':20s} {'n':>3} {'V':>7} {'R':>7} {'R 90% CI':>17} {'GT2-arm':>7} {'90% CI':>17} {'ret':>6} "
              f"{'svr':>5} {'nonp':>5} {'ll':>5} {'rlf':>5} elig  cpu_s")
        for arm, st in A.items():
            ci = st.get("R_ci90", [None, None])
            dR = st["dR"].get("GT2")
            dci = st.get("dR_ci90", {}).get("GT2", [None, None])
            print(f"   {arm:20s} {st['n_seeds']:>3} {st['V']:7.2f} {f(st['R']):>7} [{f(ci[0])},{f(ci[1])}] "
                  f"{f(dR):>7} [{f(dci[0])},{f(dci[1])}] {st['retention']:6.3f} "
                  + " ".join(f"{st['guard_ratio'][k]:5.2f}" for k in D.GUARD_KEYS)
                  + f"  {'Y' if st['eligible'] else 'n'}{'' if st['matched'] else 'E'}{'' if st['guard_ok'] else 'G'}"
                  f"  {st['cpu_s_mean']:6.1f}")
        print("   (GT2-arm = R(GT2) - R(arm); eligible = retention >= .90 and every guard ratio <= 1.10)")
        for arm in NEW_ARMS:
            if arm in A:
                print(f"   {arm:9s} rejects {A[arm]['defer']}  policy {A[arm]['policy']}  passed {A[arm]['passed']}")
        verdict, why = None, []
        if "GT2" in A:
            verdict = True
            if not A["GT2"]["eligible"]:
                verdict = False
                why.append("GT2 ineligible")
            for c in ("GT2_own", "GT2_flip", "GT2_rand", "blanket2", "never_sleep"):
                if c not in A:
                    why.append(f"{c} missing")
                    verdict = None if verdict else verdict
                    continue
                d = A[c]["dR"]["GT2"]
                ci = A[c].get("dR_ci90", {}).get("GT2", [None, None])
                need = 0.0 if c == "never_sleep" else MARGIN
                print(f"   R(GT2) - R({c}) = {d:+.3f}  90% CI [{f(ci[0])},{f(ci[1])}]  (need >= {need:.2f})")
                if d < need:
                    verdict = False if verdict is not None else None
                    why.append(f"R(GT2) - R({c}) < {need:.2f}")
        report["verdict"] = {"continue": verdict, "why": why}
        print(f"   K-A2 verdict: {'CONTINUE' if verdict else ('STOP' if verdict is False else 'INCOMPLETE')}"
              + (f"  ({'; '.join(why)})" if why else ""))
        cpu_new = [R[s][a]["cpu_s"] for s in R for a in R[s] if a in ARMS]
        report["cpu_h_new"] = float(sum(cpu_new) / 3600)
        print(f"   CPU (new arms) {report['cpu_h_new']:.2f} h, {np.mean(cpu_new) if cpu_new else float('nan'):.1f} s/job")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs():
    J = jobs()
    print("plan:", PLAN)
    print("registry:", D.registry_check(), " gt_map_check:", KA.check_gt_map())
    print(f"{len(J)} jobs = {len(SEEDS)} seeds x {len(NEW_ARMS)} arms {NEW_ARMS} + noarb_repro on {REPRO_SEEDS}")
    for a, t in tables().items():
        print(f"  {a:9s} {len(MAPS[a]):2d} edges  {t}")


def cli(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        run(a["--part"], a["--out"], smoke="--smoke" in flags,
            short=float(a["--short"]) if a.get("--short") else None,
            arms=[x for x in a["--arms"].split(",") if x] if a.get("--arms") else None)
    elif cmd == "summary":
        files, js, ref, ka = [], None, REF_DEFAULT, KA_DEFAULT
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x == "--ref":
                ref = next(it)
            elif x == "--ka":
                ka = next(it)
            elif x != "--in":
                files.extend(y for y in x.split(",") if y)
        summary(files, ref_path=ref, ka_path=ka, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "list":
        list_jobs()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
