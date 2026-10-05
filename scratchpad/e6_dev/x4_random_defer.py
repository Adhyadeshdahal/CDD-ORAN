"""X4 (scratchpad/xmethod/EXTRAS_PROTOCOL.md, Amendments 2026-10-05): random accept / defer referee on the Study 3
(E6-P certsafe) EVAL episodes. POST HOC, DESCRIPTIVE: no hypothesis, no criterion, no change to any Study 1-3 file.

Policy X4:rand@<p>: every xApp request reaching the referee is deferred ("reject") independently with probability p,
from t = 0, all families and directions alike; one uniform per request from default_rng([seed, 6640]) (shared by the
three p: common random numbers). p = .352 (MG:PMRT's request-level deferral rate in the certsafe EVAL, 37890 / 107525),
.25, .50. Plant, seeds (191100-191259) and record schema are those of the certsafe driver (e6p_certsafe.arm_job),
which this file imports and does not edit. The provenance jobs (noarb, never_sleep on 191100-191101) use the frozen
driver's own arbiters; ``analyse`` compares them with the stored Kaggle records field by field.

CLI (repo root, PYTHONPATH=.):
  python scratchpad/e6_dev/x4_random_defer.py run --part i/k --out F.jsonl [--smoke] [--short S] [--seeds a,b]
         [--arms a,b]
  python scratchpad/e6_dev/x4_random_defer.py analyse --stored DIR --x4 DIR [--out DIR] [--n-boot 10000]
--smoke: plumbing only (seed % 31 as in the driver, --short scored seconds); never a result.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataclasses  # noqa: E402

import e6p_certsafe as X  # noqa: E402  (frozen certsafe driver: plant, seeds, anchors, arbiters)
import e6p_conf as C  # noqa: E402
import e6p_opta_ka as KA  # noqa: E402
import e6p_screen as S  # noqa: E402
import e6p_step2_dev as D  # noqa: E402
import numpy as np  # noqa: E402

SUB = "x4"
TAG = 6640                                  # new RNG tag (EXTRAS_PROTOCOL X4; free in SEED_REGISTRY rng_stream_tags)
P_PMRT = 0.352                              # MG:PMRT certsafe EVAL request-level deferral rate 37890 / 107525
PS = (P_PMRT, 0.25, 0.50)
ARMS = tuple(f"X4:rand@{p:g}" for p in PS)  # X4:rand@0.352, X4:rand@0.25, X4:rand@0.5
P_OF = dict(zip(ARMS, PS, strict=True))
PROV_ARMS = ("noarb", "never_sleep")
PROV_SEEDS = (191100, 191101)
COMPARE = ("noarb", "never_sleep", "MG:PMRT", "CS:PMRT", "CS:GT")
SKIP = {"cpu_s", "secs", "rss_mb", "sub", "aliases", "signature", "alias_of", "x4"}   # not outcome fields


class RandomDefer:
    """Defer each request independently with probability p (u < p, u ~ U(0, 1) from default_rng([seed, TAG]), one
    draw per request in obs["requests"] order). Reads nothing else from obs."""

    def __init__(self, p: float, seed: int):
        if not 0.0 <= p <= 1.0:
            raise ValueError(p)
        self.p, self.rng = float(p), np.random.default_rng([int(seed), TAG])
        self.n = {"req": 0, "rej": 0}

    def __call__(self, obs):
        dec = []
        for _ in obs["requests"]:
            self.n["req"] += 1
            if self.rng.random() < self.p:
                dec.append("reject")
                self.n["rej"] += 1
            else:
                dec.append("accept")
        return {"decisions": dec, "writes": [], "rollback": []}


def make_arbiter(arm: str, sd: int, warmup_s: float):
    if arm in P_OF:
        return KA.Counting(RandomDefer(P_OF[arm], sd), is_unit=False)
    if arm in PROV_ARMS:
        return X.make_arbiter(arm, sd, warmup_s, None)          # the frozen driver's arbiter
    raise KeyError(arm)


def job(seed: int, arm: str, lf: float, smoke: bool = False, short=None) -> dict:
    """One episode; the record of e6p_certsafe.arm_job (same plant and fields), sub "x4"."""
    X.check_seed(seed, "eval")
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(C.PAIR, C.STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm, sd, cfg.warmup_s)
    t_wall, t_cpu = time.time(), time.process_time()
    env, _ = S.run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    pc = {}
    if isinstance(arb, KA.Counting):
        pc = dict(arb.n)
        pc["policy"] = dict(getattr(arb.inner, "n", None) or {})
    elif arb is not None:
        pc = {"policy": dict(getattr(arb, "n", {}) or {})}
    return {"kind": "job", "schema": X.SCHEMA_EVAL, "key": [seed, arm], "seed": seed, "cfg_seed": sd, "arm": arm,
            "sub": SUB, "cs_stage": "eval", "smoke": smoke, "short": short, "episode_s": env.total_s,
            "warmup_s": cfg.warmup_s, "load_factor": lf, **S.outcome(env), "policy_counts": pc,
            "x4": {"p": P_OF.get(arm), "tag": TAG}, "cpu_s": round(cpu, 2), "secs": round(time.time() - t_wall, 1),
            "rss_mb": D.peak_rss_mb()}


def jobs(seeds=None, arms=None) -> list:
    """[(seed, arm)]: the provenance jobs first, then seed-major X4 jobs."""
    seeds = list(seeds) if seeds else X.stage_seeds("eval")
    J = [(s, a) for s in PROV_SEEDS for a in PROV_ARMS if s in seeds]
    J += [(s, a) for s in seeds for a in ARMS]
    return [(X.check_seed(s, "eval"), a) for s, a in J if not arms or a in arms]


def run(part: str, out: str, smoke=False, short=None, seeds=None, arms=None):
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    state = S.load_state()
    lf = S.lf_of(state, C.STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header("x4:eval", part, smoke, state)
    try:
        import cloud
        nenv = cloud.numeric_env()
    except Exception as e:                                              # noqa: BLE001
        nenv = {"error": repr(e)}
    head.update(kind="header", schema=X.SCHEMA_EVAL, driver="x4_random_defer", sub=SUB, cs_stage="eval",
                platform=nenv, host=os.environ.get("XM_PLATFORM"),
                code_commit=os.environ.get("XM_CODE_COMMIT"), code_dirty=os.environ.get("XM_CODE_DIRTY"),
                maps=C.maps_status(), certsafe_freeze=X.freeze_status(),
                driver_sha256={f: X.sha_lf(os.path.join(ROOT, f)) for f in (
                    "scratchpad/e6_dev/x4_random_defer.py", "scratchpad/e6_dev/e6p_certsafe.py",
                    "scratchpad/e6_dev/e6p_opta_ka.py", "scratchpad/e6_dev/e6p_step2_dev.py",
                    "scratchpad/e6_dev/e6p_screen.py")},
                consts={"pair": C.PAIR, "stratum": C.STRATUM, "load_factor": lf, "tag": TAG, "p": dict(P_OF),
                        "prov": [list(PROV_SEEDS), list(PROV_ARMS)], "smoke": {"short": short} if smoke else None})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    J = jobs(seeds, arms)
    t0, n = time.time(), 0
    for u, (seed, arm) in enumerate(J):
        if u % k != i or (seed, arm) in done:
            continue
        rec = job(seed, arm, lf, smoke, short)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("seed", "arm", "psvr", "rlf", "energy_j", "st_req", "st_rej",
                                                   "cpu_s")}, default=S._js), flush=True)
    S._append(out, {"kind": "close", "part": part, "n_jobs": n, "n_total": len(J), "secs": round(time.time() - t0, 1)})


# ============================================================================================ analysis
def _load(files, want_sub, drop_smoke=True):
    R, heads, bad = {}, [], 0
    for path in files:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("kind") == "header":
                heads.append(r)
            if r.get("kind") != "job" or r.get("sub") != want_sub or r.get("cs_stage") != "eval":
                continue
            if drop_smoke and r.get("smoke"):
                continue
            R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R, heads, bad


def _ci(x):
    return C.D._ci(x)


def analyse(stored: str, x4: str, out: str, n_boot: int = 10_000):
    import e6p_conf_analyze as AN
    files_s = sorted(glob.glob(os.path.join(stored, "*", "res_*.jsonl")))
    files_x = sorted(glob.glob(os.path.join(x4, "**", "res_*.jsonl"), recursive=True))
    Rs, _, _ = _load(files_s, X.SUB)
    Rx, heads_x, bad_x = _load(files_x, SUB)
    art = json.load(open(os.path.join(ROOT, X.ARTIFACT_DOC)))
    for s in Rs:                                       # aliased CS arms from their simulated target (cmd_eval)
        for arm, tgt in art["alias_of"].items():
            if tgt != arm and tgt in Rs[s] and arm not in Rs[s]:
                Rs[s][arm] = dict(Rs[s][tgt], arm=arm, alias_of=tgt)
    stored_arms = [a for a in X.ALL_ARMS if any(a in Rs[s] for s in Rs)]
    # ---- provenance: VPS re-runs of noarb / never_sleep vs the stored Kaggle records
    prov = []
    for s in PROV_SEEDS:
        for a in PROV_ARMS:
            v, k = Rx.get(s, {}).get(a), Rs.get(s, {}).get(a)
            if v is None or k is None:
                prov.append({"seed": s, "arm": a, "present": False})
                continue
            keys = sorted((set(v) | set(k)) - SKIP - {"policy_counts", "key", "kind", "schema"})
            diff = {x: [k.get(x), v.get(x)] for x in keys if k.get(x) != v.get(x)}
            prov.append({"seed": s, "arm": a, "present": True, "identical": not diff and
                         k.get("policy_counts") == v.get("policy_counts"), "diff": diff,
                         "policy_equal": k.get("policy_counts") == v.get("policy_counts"),
                         "cpu_kaggle": k.get("cpu_s"), "cpu_vps": v.get("cpu_s")})
    # ---- one bootstrap over the stored arms + X4
    R = {s: dict(Rs[s]) for s in Rs}
    for s in R:
        for a in ARMS:
            if a in Rx.get(s, {}):
                R[s][a] = Rx[s][a]
    x4_arms = [a for a in ARMS if all(a in Rx.get(s, {}) for s in X.stage_seeds("eval"))]
    missing = {a: [s for s in X.stage_seeds("eval") if a not in Rx.get(s, {})] for a in ARMS}
    st = AN.arm_stats_multi(R, stored_arms + x4_arms, n_boot)
    P, B = st["point"], st["boot"]
    E = json.load(open(os.path.join(ROOT, "scratchpad", "e6_dev", "decision", "cs_eval.json")))
    repro = {"gate_a": all(np.isclose(st[k], E["gate_a"][k], rtol=1e-12, atol=0) for k in ("V_AA", "V_ref", "den"))
             and st["ref_arm"] == E["gate_a"]["ref_arm"] and st["n"] == E["gate_a"]["n"] == 160}
    bad_arms = []
    for a in stored_arms:
        e, p = E["arms"][a], P[a]
        ok = (all(np.isclose(p[k], e[k], rtol=1e-9, atol=1e-12) for k in ("V", "R", "Rstar", "retention"))
              and p["eligible"] == e["eligible"]
              and all(np.isclose(p["guard_ratio"][g], e["guard_ratio"][g], rtol=1e-9) for g in p["guard_ratio"])
              and np.allclose(_ci(B[a]["R"]), e["R_ci90"], rtol=1e-9)
              and all(np.allclose(_ci(B[a]["guard"][g]), e["guard_ratio_ci90"][g], rtol=1e-9) for g in p["guard_ratio"]))
        if not ok:
            bad_arms.append(a)
    repro["arms"] = not bad_arms
    repro["mismatch"] = bad_arms
    rows = {}
    for a in list(COMPARE) + x4_arms:
        if a not in P:
            continue
        seeds = st["seeds"]
        rec = [R[s][a] for s in seeds]
        req, rej = sum(r["st_req"] for r in rec), sum(r["st_rej"] for r in rec)
        fam = {}
        for r in rec:
            for k_, v_ in (r.get("policy_counts") or {}).get("defer", {}).items():
                fam[k_] = fam.get(k_, 0) + v_
        rows[a] = dict(P[a], R_ci90=_ci(B[a]["R"]), Rstar_ci90=_ci(B[a]["Rstar"]),
                       retention_ci90=_ci(B[a]["retention"]),
                       guard_ratio_ci90={g: _ci(v) for g, v in B[a]["guard"].items()},
                       requests=req, deferred=rej, defer_rate=rej / req if req else float("nan"),
                       defer_by_family_phase=dict(sorted(fam.items())))
    contrasts = {a: {c: {"dR": P[a]["R"] - P[c]["R"], "ci90": _ci(B[a]["R"] - B[c]["R"])} for c in COMPARE}
                 for a in x4_arms}
    cpu_vps = sum(Rx[s][a]["cpu_s"] for s in Rx for a in Rx[s] if Rx[s][a].get("sub") == SUB) / 3600
    cpu_x4 = sum(Rx[s][a]["cpu_s"] for s in Rx for a in Rx[s] if a in ARMS) / 3600
    fk = [p_["cpu_kaggle"] / p_["cpu_vps"] for p_ in prov if p_.get("present") and p_.get("cpu_vps")]
    fac = json.load(open(os.path.join(ROOT, "scratchpad", "xmethod", "results", "exp_c", "calib", "factors.json")))
    vps_key = next((k for k in fac["factors"]["*"] if k.startswith("vps|")), None)
    fstar = fac["factors"]["*"][vps_key] if vps_key else None
    plats = sorted({json.dumps(AN.platform_key(h.get("platform")), default=str) for h in heads_x
                    if h.get("driver") == "x4_random_defer" and not h.get("smoke")})
    commits = sorted({str(h.get("code_commit")) for h in heads_x if h.get("driver") == "x4_random_defer"
                      and not h.get("smoke")})
    rep = {"schema": "x4-random-defer/1", "declared": "scratchpad/xmethod/EXTRAS_PROTOCOL.md Amendments 2026-10-05",
           "status": "POST HOC, DESCRIPTIVE", "p": dict(P_OF), "tag": TAG,
           "n_seeds": st["n"], "seeds": [st["seeds"][0], st["seeds"][-1]] if st["seeds"] else None,
           "gate_a": {k: st[k] for k in ("V_AA", "V_ref", "ref_arm", "den")}, "repro_cs_eval": repro,
           "provenance": prov, "provenance_identical": all(p_.get("identical") for p_ in prov),
           "x4_missing": missing, "x4_bad_lines": bad_x, "x4_platforms": plats, "x4_code_commits": commits,
           "rows": rows, "contrasts": contrasts,
           "cpu": {"vps_cpu_h_all": cpu_vps, "vps_cpu_h_x4": cpu_x4,
                   "factor_star": fstar, "factor_star_host": vps_key,
                   "kaggle_ref_cpu_h_x4_star": cpu_x4 * fstar["f"] if fstar else None,
                   "kaggle_ref_cpu_h_x4_star_lo_hi": [cpu_x4 * fstar["f_lo"], cpu_x4 * fstar["f_hi"]] if fstar else None,
                   "factor_direct_prov": fk, "factor_direct_mean": float(np.mean(fk)) if fk else None,
                   "kaggle_ref_cpu_h_x4_direct": cpu_x4 * float(np.mean(fk)) if fk else None},
           "n_boot": n_boot}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "x4_tables.json"), "w", newline="\n") as fh:
        json.dump(rep, fh, indent=1, default=S._js)
    print(table_md(rep))
    return rep


def _f(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}" if nd == 3 else f"{x:.{nd}f}"


def table_md(rep) -> str:
    G = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
    L = ["| arm | defer rate | R [90 % CI] | R* | retention [90 % CI] | svr | nonprot eMBB | LL | RLF [90 % CI] | eligible |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in rep["rows"].items():
        g = r["guard_ratio"]
        L.append(f"| {a} | {r['defer_rate']:.3f} | {_f(r['R'])} [{_f(r['R_ci90'][0])}, {_f(r['R_ci90'][1])}] | "
                 f"{_f(r['Rstar'])} | {r['retention']:.3f} [{r['retention_ci90'][0]:.3f}, {r['retention_ci90'][1]:.3f}] | "
                 + " | ".join(f"{g[k]:.2f}" for k in G[:3])
                 + f" | {g['rlf']:.2f} [{r['guard_ratio_ci90']['rlf'][0]:.2f}, {r['guard_ratio_ci90']['rlf'][1]:.2f}] | "
                 f"{'yes' if r['eligible'] else 'no'} |")
    L += ["", "| X4 arm | " + " | ".join(f"dR vs {c} [90 % CI]" for c in COMPARE) + " |",
          "|---|" + "---|" * len(COMPARE)]
    for a, cs in rep["contrasts"].items():
        L.append(f"| {a} | " + " | ".join(f"{_f(v['dR'])} [{_f(v['ci90'][0])}, {_f(v['ci90'][1])}]"
                                          for v in cs.values()) + " |")
    return "\n".join(L)


def cli(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    smoke = "--smoke" in rest
    rest = [x for x in rest if x != "--smoke"]
    a = dict(zip(rest[::2], rest[1::2], strict=True))
    if cmd == "run":
        run(a["--part"], a["--out"], smoke=smoke, short=float(a["--short"]) if a.get("--short") else None,
            seeds=[int(x) for x in a["--seeds"].split(",")] if a.get("--seeds") else None,
            arms=a["--arms"].split(",") if a.get("--arms") else None)
    elif cmd == "analyse":
        analyse(a["--stored"], a["--x4"], a.get("--out", "scratchpad/xmethod/results/extras/x4"),
                int(a.get("--n-boot", 10_000)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
