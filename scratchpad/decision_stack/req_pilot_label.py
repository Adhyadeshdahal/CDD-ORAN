"""Request-level oracle labeller for the req pilot (design: req_pilot/PILOT_DESIGN.md; driver: req_pilot_grid.py).
run_ep appends one jsonl line per (episode, k) decision second; finished (seed, k) are skipped on resume."""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

from cdd_oran.decision import collect as CO
from cdd_oran.decision.world_model import objective
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import LIMITS
from cdd_oran.envs.e6.sim import LL

LAM_E, W_LL = 1e4, 5.0
HS = (10, 30, 90)
ANCHORS = (150, 250, 350)
MAXREQ = 8
N_EP = 12
QU = {"cio": 1.0, "hys": 0.5, "ll_ratio": 0.05, "sleep": 1.0, "carrier": 1.0}


def episodes():
    out = []
    for j in range(20, 30):
        for s, ld in itertools.product(CO.DEV_SCENARIOS, CO.DEV_LOADS):
            out.append((s, ld, CO.dev_seed("policy", s, ld, j)))
    return out[:N_EP]


def rollout(base, dec_list):
    """base = env between propose/apply. Apply dec_list, then accept-all 90 s; J at H = 10, 30, 90."""
    sim = base.copy()
    sla0 = dict(sim.plant.sla)
    sim.step_apply({"decisions": dec_list, "writes": [], "rollback": []})
    J = {}
    for i in range(1, max(HS) + 1):
        o = sim.step_propose()
        sim.step_apply({"decisions": ["accept"] * len(o["requests"]), "writes": [], "rollback": []})
        if i in HS:
            J[i] = objective(sla0, sim.plant.sla, LAM_E, W_LL)
    return J


def labels(base, n_req, idx):
    J0 = rollout(base, ["accept"] * n_req)
    out = []
    for i in idx:
        d = ["accept"] * n_req
        d[i] = "reject"
        Ji = rollout(base, d)
        out.append({h: Ji[h] - J0[h] for h in HS})
    return J0, out


def _f(x, default=-1.0):
    x = float(x)
    return default if not np.isfinite(x) else x


def features(env, obs, rep, reqs, i):
    r = reqs[i]
    k = r["knob"]
    typ = k[0]
    p = env.plant
    lay = p.lay
    c = int(k[1])
    n = int(k[2]) if typ == "cio" else c
    fast, thp = rep.get("fast"), rep.get("thp")
    now = obs["t"]
    d = r["prop"] - r["cur"]
    u = fast["prb_util"] if fast is not None else np.full(p.nc, np.nan)
    ll = (fast["ll_delay_p95"][c] / C.LL_DELAY_TARGET_S) if fast is not None else np.nan
    nb = lay.neighbours[c]
    f = {
        "xapp": r["xapp"], "ktype": typ, "dir": float(np.sign(d)), "mag_q": abs(d) / QU.get(typ, 1.0),
        "cur": float(r["cur"]),
        "is_rev_cio": float(typ == "cio" and (c, n) not in {(s, m) for s in range(lay.n_cells) for m in lay.neighbours[s]}),
        "u_c": _f(u[c]), "u_n": _f(u[n]), "u_diff": _f(u[c] - u[n], 0.0),
        "u_nbr": _f(np.nanmean(u[nb]) if len(nb) else np.nan),
        "ll_slice_c": _f(fast["prb_util_slice"][c][LL]) if fast is not None else -1.0,
        "rsv_idle_c": _f(fast["prb_rsv_idle"][c]) if fast is not None else -1.0,
        "ll_ratio_delay": _f(ll), "ll_nan": float(not np.isfinite(ll)),
        "act_ue_c": _f(np.sum(fast["act_ue"][c])) if fast is not None else -1.0,
        "carriers_c": float(p.n_car[c]), "ll_ratio_c": float(p.ll_ratio[c]),
        "embb_p5_c": _f(thp["embb_thp_p5"][c] / C.EMBB_THP_TARGET_BPS) if thp is not None else -1.0,
        "is_macro_c": float(lay.is_macro[c]), "asleep_c": float(p.asleep[c]),
        "since_change": float(min(now - env.last_change.get(k, -1e9), 600.0)),
        "dwell_ok": float(now - env.last_change.get(k, -1e9) >= LIMITS[typ][3]),
        "n_same_cell": float(sum(int(q["knob"][1]) == c for q in reqs)),
        "n_same_knob": float(sum(q["knob"] == k for q in reqs)), "n_pending": float(len(reqs)),
        "c": c, "n": n,
    }
    f["dir_x_ll"] = f["dir"] * f["ll_ratio_delay"]
    f["dir_x_u"] = f["dir"] * f["u_c"]
    return f


def run_ep(ei, ks, out):
    scn, ld, seed = episodes()[ei]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                q = json.loads(line)
            except json.JSONDecodeError:      # truncated last line of a killed run: recomputed
                continue
            done.add((q["seed"], q["k"]))
    ks = [k for k in ks if (seed, k) not in done]
    if not ks:
        print("skip", ei, seed)
        return
    cfg = C.E6Config(seed=seed, load=ld, mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0, scenario=scn)
    env = E6Env(cfg, log=False, wg3=True)
    rep = {}
    T = time.time()
    kmax = max(ks)
    while True:
        obs = env.step_propose()
        for rr in obs["new_reports"]:
            rep[rr["gran"]] = rr
        t = int(obs["t"])
        reqs = obs["requests"]
        hit = None
        for k in [k_ for k_ in ks if (seed, k_) not in done]:
            a = ANCHORS[k]
            if a <= t < a + 10 and any(q["xapp"] != "SLICE" for q in reqs):
                hit = k
            elif t >= a + 10 and reqs:          # fallback: first second >= t_k with any pending request
                hit = k
            if hit is not None:
                break
        if hit is not None and (seed, hit) not in done:
            k = hit
            rng = np.random.default_rng([seed, 4401, k])
            n = len(reqs)
            idx = list(range(n)) if n <= MAXREQ else sorted(rng.choice(n, MAXREQ, replace=False).tolist())
            t0 = time.time()
            feats = [features(env, obs, rep, reqs, i) for i in idx]      # before labels: fail fast
            J0, lab = labels(env, n, idx)
            rows = []
            for j, i in enumerate(idx):
                rows.append({"i": i, "xapp": reqs[i]["xapp"], "knob": list(reqs[i]["knob"]), "cur": reqs[i]["cur"],
                             "prop": reqs[i]["prop"], "dJ": lab[j], "feat": feats[j]})
            rec = {"scenario": scn, "load": ld, "seed": seed, "ep": ei, "k": k, "t": t, "n_pending": n,
                   "J0": J0, "rows": rows, "numpy": np.__version__}
            if k == 0:
                nud = env.copy()
                nud.plant.pos = nud.plant.pos + 1e-6
                J0n, labn = labels(nud, n, idx)
                rec["J0_nudged"] = J0n
                for j, rw in enumerate(rows):
                    rw["dJ_nudged"] = labn[j]
            rec["secs"] = round(time.time() - t0, 1)
            with open(out, "a", newline="\n") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            done.add((seed, k))
            print(json.dumps({"ep": ei, "seed": seed, "k": k, "t": t, "n_pending": n, "n_lab": len(idx),
                              "secs": rec["secs"], "total": round(time.time() - T, 1)}), flush=True)
            if k == kmax:
                return
        env.step_apply({"decisions": ["accept"] * len(reqs), "writes": [], "rollback": []})
        if env.sec >= env.total_s:
            return

