"""E6-P learned WG3 arbiter pilot: stage L0 (scratchpad/e6_dev/decision/ARBITER_DESIGN.md sections 1, 3, 4 / L0).

Stage L0 = per (pair, stratum): 10 randomized collection episodes (cdd_oran.decision.collect_p, pi0 high-eps on even
j, low-eps on odd j) on DEV seeds 160000 + 1000*pair_idx + 100*stratum + j, j = 0..9, with paired-rollout labels on
sampled units (cdd_oran.decision.labels_p) and G0b region probes. Kill decisions (design sec. 4, L0):
  G0a label reliability corr(k1, k2) >= 0.70 (primary: dPV on the exposure set);
  G0c CV-by-episode sign AUC on nonzero dPV labels >= 0.70 with 90 % lower bound >= 0.60, and contrast CV R^2 >= 0.05;
  G0b fixes the unit: fall back to region x xApp (D = 20) if cell-unit contrasts explain < 50 % of the region contrast
      or > 50 % of the headroom mass sits in unmodelled unit types (operationalisation: see G0B_NOTE).

CLI (repo root, PYTHONPATH=.; in the Kaggle bundle the same file is e6dev/e6p_arbiter_pilot.py):
  python scratchpad/e6_dev/e6p_arbiter_pilot.py run --part i/k --out FILE.jsonl [--smoke] [--cells P1:3,P3:1]
  python scratchpad/e6_dev/e6p_arb_L0.py run --part i/k --out FILE.jsonl        # cloud.py wrapper
  python scratchpad/e6_dev/e6p_arbiter_pilot.py summary --in FILES... [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_arbiter_pilot.py list [--cells ...]
(pair, stratum) cells: --cells, else env E6P_ARB_CELLS ("P1:3,P3:1"), else e6p_state.json "stage3_pass" (when a later
stage-3 summary writes it), else "stage2_crit12". JSONL, one record per episode, resumable (done keys are skipped).
--smoke: DEV plumbing only (seed % 31, 60 s warm-up + 120 s scored, T = H = 30 s, <= 4 labels, flagged smoke=True).
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
import e6p_screen as S  # noqa: E402  (sets up ROOT on sys.path)

from cdd_oran.decision import collect_p as CP  # noqa: E402
from cdd_oran.decision import labels_p as LP  # noqa: E402
from cdd_oran.decision import units_p as UP  # noqa: E402

DESIGN = "scratchpad/e6_dev/decision/ARBITER_DESIGN.md (draft 2026-09-29, not frozen)"
STAGES = ("L0",)
N_EP = {"L0": 10}
SEED_BASE, BOOT_TAG = 160000, 6615
T, H, KS = UP.T_UNIT, LP.H_LABEL, LP.KS
SMOKE = dict(warmup_s=60.0, scored_s=120.0, T=30.0, H=30, max_labels=4, max_probes=1)
REL_MIN, AUC_MIN, AUC_LB_MIN, R2_MIN = 0.70, 0.70, 0.60, 0.05
G0B_EXPLAIN_MIN, G0B_UNMODELLED_MAX = 0.50, 0.50
MODELLED_TYPES = {("ES", "carrier"), ("ES", "sleep"), ("PowerES", "ptx"), ("SliceGuarantee", "prot_min")}
LAM_LOGIT, LAM_RIDGE, N_BOOT = 1.0, 10.0, 2000
G0B_NOTE = ("(i) explained = identity R^2 of the region contrast dR by the sum of its cells' single-cell contrasts "
            "(network dPV, mean over k; region probes, mode reject, held T from t0); (ii) unmodelled = share of the "
            "headroom mass sum_u max(0, -min_m dPV_net) NOT visible on the unit's exposure set at the best mode, plus "
            "the whole headroom of unit types outside MODELLED_TYPES.")


def arb_seed(pair, s, j):
    seed = SEED_BASE + 1000 * S.PAIRS[pair]["idx"] + 100 * int(s) + int(j)
    assert 160000 <= seed < 170000, seed
    return seed


def parse_cells(txt):
    out = []
    for tok in (t for t in txt.split(",") if t.strip()):
        p, s = tok.strip().split(":")
        if p not in S.PAIRS or int(s) not in range(4):
            raise SystemExit(f"bad cell {tok!r} (want P1:3,P3:1)")
        out.append([p, int(s)])
    return out


def cells_from(arg, state):
    if arg:
        return parse_cells(arg)
    if os.environ.get("E6P_ARB_CELLS"):
        return parse_cells(os.environ["E6P_ARB_CELLS"])
    for key in ("stage3_pass", "stage2_crit12"):
        if state.get(key):
            return [[p, int(s)] for p, s in state[key]]
    raise SystemExit("no (pair, stratum) cells: pass --cells, set E6P_ARB_CELLS or finish stage 2 (stage2_crit12)")


def registry_check():
    """{seeds, tags} registered in docs/benchmark/SEED_REGISTRY.json (None in the Kaggle bundle)."""
    if S.BUNDLE:
        return None
    d = json.load(open(os.path.join(S.ROOT, "docs", "benchmark", "SEED_REGISTRY.json")))
    rngs = []

    def walk(o):
        if isinstance(o, list) and len(o) == 2 and all(isinstance(v, int) for v in o):
            rngs.append(o)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
    walk(d.get("E6", {}))
    return {"seeds_160000_179999": any(lo <= 160000 and hi >= 179999 for lo, hi in rngs),
            "tags_6612_6615": all(t in d.get("rng_stream_tags", []) for t in (6612, 6613, 6614, 6615))}


def units_for(stage, cells):
    return [(stage, p, int(s), arb_seed(p, s, j), j) for p, s in cells for j in range(N_EP[stage])]


# ---------------------------------------------------------------------------------------------- one episode
def episode_job(stage, pair, s, seed, j, lf, smoke=False, max_labels=None, max_probes=None):
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(pair, s, sd, lf, smoke)
    t_unit, h = T, H
    if smoke:
        cfg = dataclasses.replace(cfg, warmup_s=SMOKE["warmup_s"], scored_s=SMOKE["scored_s"])
        t_unit, h = SMOKE["T"], SMOKE["H"]
        max_labels = SMOKE["max_labels"] if max_labels is None else max_labels
        max_probes = SMOKE["max_probes"] if max_probes is None else max_probes
    regime = CP.regime_for(j)
    lab = LP.Labeller(ks=KS, H=h, T=t_unit)
    cnt = {"labels": 0, "probes": 0, "skipped_end": 0, "skipped_cap": 0}

    def hook(env, obs, snap, opened):
        for u in opened:
            if not LP.sampled(sd, u):
                continue
            if u["t0"] + h > env.total_s:
                u["label_skipped"] = "end"
                cnt["skipped_end"] += 1
                continue
            if max_labels is not None and cnt["labels"] >= max_labels:
                u["label_skipped"] = "cap"
                cnt["skipped_cap"] += 1
                continue
            u["label"] = lab.label(env, obs, snap, u)
            cnt["labels"] += 1
            if LP.region_sampled(sd, u) and (max_probes is None or cnt["probes"] < max_probes):
                pr = lab.region_probe(env, obs, snap, u, env.plant.lay.cell_site, "reject", u["label"])
                if pr is not None:
                    u["region_probe"] = pr
                    cnt["probes"] += 1

    t_wall = time.time()
    res = CP.run_collection(cfg, CP.RandomizedUnitPolicy(sd, regime), T=t_unit, labeller=hook)
    env = res["env"]
    units = [dict(u) for u in res["units"]]
    by_x = {}
    for u in units:
        b = by_x.setdefault(u["x"], {"units": 0, "labels": 0, "modes": {}})
        b["units"] += 1
        b["labels"] += int("label" in u)
        b["modes"][u["mode"]] = b["modes"].get(u["mode"], 0) + 1
    return {"kind": "episode", "key": [stage, pair, s, seed], "stage": stage, "pair": pair, "stratum": s,
            "seed": seed, "cfg_seed": sd, "j": j, "regime": regime, "load_factor": lf, "smoke": smoke,
            "T": t_unit, "H": h, "ks": list(KS), "episode_s": env.total_s, "outcome": S.outcome(env),
            "counts": cnt, "by_xapp": by_x, "units": units, "cpu_s": round(res["cpu_s"], 2),
            "label_cpu_s": round(lab.cpu_s, 2), "collect_cpu_s": round(res["cpu_s"] - lab.cpu_s, 2),
            "n_roll": lab.n_roll, "secs": round(time.time() - t_wall, 1)}


def run(stage, part, out, smoke=False, cells_arg=None):
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    state = S.load_state()
    cells = cells_from(cells_arg, state)
    reg = registry_check()
    if not smoke and reg is not None and not all(reg.values()) and not os.environ.get("E6P_ARB_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block / RNG tags not registered in SEED_REGISTRY.json: {reg} (design sec. 3); "
                         "register them first (or set E6P_ARB_ALLOW_UNREGISTERED=1 for a DEV dry run)")
    i, k = map(int, part.split("/"))
    head = S.header(stage, part, smoke, state)
    head.update(kind="header", design=DESIGN, pilot="e6p_arbiter_pilot", cells=cells, registry=reg,
                consts={"T": T, "H": H, "ks": list(KS), "pi0": CP.PI0, "label_rate": LP.LABEL_RATE,
                        "region_rate": LP.REGION_RATE, "seed_base": SEED_BASE, "smoke": SMOKE if smoke else None})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "episode" and r.get("smoke") == smoke}
    t0, n = time.time(), 0
    U = units_for(stage, cells)
    for u, (st, pair, s, seed, j) in enumerate(U):
        if u % k != i or (st, pair, s, seed) in done:
            continue
        rec = episode_job(st, pair, s, seed, j, S.lf_of(state, s), smoke)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("pair", "stratum", "seed", "regime", "counts", "cpu_s",
                                                  "label_cpu_s", "secs")}), flush=True)
        if smoke:
            break
    S._append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_units": len(U),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary helpers
def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _prep(Xtr, Xte):
    mu = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
    Xte = np.where(np.isfinite(Xte), Xte, mu)
    m, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (Xtr - m) / sd, (Xte - m) / sd


def logit_fit_predict(Xtr, ytr, Xte, lam=LAM_LOGIT, iters=50):
    Ztr, Zte = _prep(Xtr, Xte)
    Ztr = np.c_[np.ones(len(Ztr)), Ztr]
    Zte = np.c_[np.ones(len(Zte)), Zte]
    w = np.zeros(Ztr.shape[1])
    P = lam * np.eye(len(w))
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Ztr @ w, -30, 30)))
        g = Ztr.T @ (p - ytr) + P @ w
        Hs = Ztr.T @ (Ztr * (p * (1 - p))[:, None]) + P + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return Zte @ w


def ridge_fit_predict(Xtr, ytr, Xte, lam=LAM_RIDGE):
    Ztr, Zte = _prep(Xtr, Xte)
    ym = float(np.mean(ytr))
    coef = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]), Ztr.T @ (ytr - ym))
    return Zte @ coef + ym


def label_rows(recs):
    """One row per (labelled unit, non-accept mode)."""
    rows = []
    for r in recs:
        for u in r["units"]:
            L = u.get("label")
            if not L:
                continue
            for m in L["modes"]:
                if m == "accept":
                    continue
                rows.append({"ep": r["seed"], "x": u["x"], "knob": u["knob"], "mode": m, "ctx": u["ctx"],
                             "d": L["d"][m], "mean": L["mean"][m], "logged_mode": u["mode"], "unit": u})
    return rows


def features(rows):
    keys = sorted({k for r in rows for k, v in r["ctx"].items() if isinstance(v, (int, float))})
    knobs, modes, xs = ("carrier", "sleep", "ptx", "prot_min"), ("half", "reject", "accept+rb"), UP.XAPPS_P[:3]
    X = np.array([[float(r["ctx"].get(k, np.nan)) if r["ctx"].get(k) is not None else np.nan for k in keys]
                  + [float(r["knob"] == v) for v in knobs] + [float(r["mode"] == v) for v in modes]
                  + [float(r["x"] == v) for v in xs] for r in rows], float)
    return X, keys


def g0c(rows, rng, scope="exp"):
    """CV-by-episode (leave-one-episode-out) sign AUC on nonzero dPV (logistic) + contrast R^2 (ridge)."""
    if len(rows) < 10:
        return {"n": len(rows), "note": "too few labels"}
    X, _ = features(rows)
    y = np.array([r["mean"][scope]["pv"] for r in rows])
    ep = np.array([r["ep"] for r in rows])
    eps = np.unique(ep)
    out = {"n": len(rows), "n_eps": len(eps)}
    nz = np.abs(y) > 1e-9
    out["n_nonzero"] = int(nz.sum())
    out["n_pos"] = int((y > 1e-9).sum())
    s = np.full(len(y), np.nan)
    for e in eps:
        tr, te = nz & (ep != e), nz & (ep == e)
        if te.any() and len(np.unique(y[tr] > 0)) == 2:
            s[te] = logit_fit_predict(X[tr], (y[tr] > 0).astype(float), X[te])
    ok = nz & np.isfinite(s)
    if ok.sum() >= 4 and len(np.unique(y[ok] > 0)) == 2:
        a = auc(y[ok] > 0, s[ok])
        members = {e: np.nonzero(ok & (ep == e))[0] for e in eps}
        boot = []
        for _ in range(N_BOOT):
            idx = np.concatenate([members[e] for e in rng.choice(eps, len(eps))])
            yy = y[idx] > 0
            if 0 < yy.sum() < len(yy):
                boot.append(auc(yy, s[idx]))
        out.update(auc=a, auc_lb90=float(np.quantile(boot, 0.10)) if boot else None)
    yhat = np.full(len(y), np.nan)
    for e in eps:
        tr, te = ep != e, ep == e
        if tr.sum() >= 3:
            yhat[te] = ridge_fit_predict(X[tr], y[tr], X[te])
    okr = np.isfinite(yhat)
    sst = float(((y[okr] - y[okr].mean()) ** 2).sum())
    out["r2"] = float(1 - ((y[okr] - yhat[okr]) ** 2).sum() / sst) if sst > 0 else None
    out["pass"] = bool(out.get("auc") is not None and np.isfinite(out.get("auc", np.nan)) and out["auc"] >= AUC_MIN
                       and (out.get("auc_lb90") or 0) >= AUC_LB_MIN and (out["r2"] or -1) >= R2_MIN)
    return out


def g0a(rows):
    out = {}
    for sc in ("exp", "net"):
        for kpi in LP.KPIS:
            a = [r["d"][sc][kpi][0] for r in rows]
            b = [r["d"][sc][kpi][1] for r in rows]
            out[f"{sc}_{kpi}"] = _corr(a, b)
            nz = [i for i in range(len(a)) if abs(a[i]) > 1e-9 or abs(b[i]) > 1e-9]
            out[f"{sc}_{kpi}_nonzero"] = _corr([a[i] for i in nz], [b[i] for i in nz])
    out["n"] = len(rows)
    out["frac_zero_pv_exp"] = float(np.mean([abs(np.mean(r["d"]["exp"]["pv"])) <= 1e-9 for r in rows])) if rows \
        else None
    out["pass"] = out["exp_pv"] is not None and out["exp_pv"] >= REL_MIN
    return out


def g0b(recs):
    probes = [u["region_probe"] for r in recs for u in r["units"] if u.get("region_probe")]
    res = {"n_probes": len(probes)}
    if probes:
        dR = np.array([np.mean([p["dR"][i][0] for i in range(len(p["dR"]))]) for p in probes])
        dS = np.array([np.mean([p["sum_dC"][i][0] for i in range(len(p["sum_dC"]))]) for p in probes])
        sst = float(((dR - dR.mean()) ** 2).sum())
        res["explained_r2"] = float(1 - ((dR - dS) ** 2).sum() / sst) if sst > 0 else None
        res["proj_ratio"] = float((dR * dS).sum() / (dR ** 2).sum()) if (dR ** 2).sum() > 0 else None
    hm, un = 0.0, 0.0
    for r in recs:
        for u in r["units"]:
            L = u.get("label")
            if not L:
                continue
            ms = [m for m in L["modes"] if m != "accept"]
            best = min(ms, key=lambda m: L["mean"][m]["net"]["pv"])
            hu = max(0.0, -L["mean"][best]["net"]["pv"])
            if hu <= 0:
                continue
            hm += hu
            if (u["x"], u["knob"]) not in MODELLED_TYPES:
                un += hu
            else:
                un += hu - min(hu, max(0.0, -L["mean"][best]["exp"]["pv"]))
    res["headroom_mass"] = hm
    res["unmodelled_frac"] = un / hm if hm > 0 else None
    ex, uf = res.get("explained_r2"), res["unmodelled_frac"]
    res["switch_to_region"] = bool((ex is not None and ex < G0B_EXPLAIN_MIN) or (uf is not None and uf >
                                                                                  G0B_UNMODELLED_MAX))
    res["determined"] = ex is not None and uf is not None
    res["rule"] = G0B_NOTE
    return res


def summary(paths, json_out=None, allow_smoke=False):
    recs = [r for p in paths for r in S._read(p) if r.get("kind") == "episode"]
    recs = [r for r in recs if allow_smoke or not r.get("smoke")]
    uniq = {}
    for r in recs:
        uniq[tuple(r["key"])] = r
    recs = list(uniq.values())
    groups = {}
    for r in recs:
        groups.setdefault((r["pair"], r["stratum"]), []).append(r)
    report = {"design": DESIGN, "thresholds": {"REL_MIN": REL_MIN, "AUC_MIN": AUC_MIN, "AUC_LB_MIN": AUC_LB_MIN,
                                               "R2_MIN": R2_MIN, "G0B": [G0B_EXPLAIN_MIN, G0B_UNMODELLED_MAX]},
              "cells": {}}
    for (pair, s), rs in sorted(groups.items()):
        rows = label_rows(rs)
        rng = np.random.default_rng([BOOT_TAG, S.PAIRS[pair]["idx"], s])
        n_lab = sum(r["counts"]["labels"] for r in rs)
        cost = {"episodes": len(rs), "labels": n_lab, "probes": sum(r["counts"]["probes"] for r in rs),
                "collect_cpu_s_per_ep": float(np.mean([r["collect_cpu_s"] for r in rs])),
                "label_cpu_s_total": float(sum(r["label_cpu_s"] for r in rs)),
                "cpu_s_total": float(sum(r["cpu_s"] for r in rs)),
                "label_cpu_s_per_label": float(np.mean([u["label"]["cpu_s"] for r in rs for u in r["units"]
                                                        if u.get("label")])) if n_lab else None,
                "units_per_ep": {x: float(np.mean([r["by_xapp"].get(x, {}).get("units", 0) for r in rs]))
                                 for x in UP.XAPPS_P[:3]}}
        cell = {"episodes": sorted(r["j"] for r in rs), "cost": cost,
                "G0a": g0a(rows), "G0a_by_xapp": {x: g0a([q for q in rows if q["x"] == x])
                                                  for x in sorted({q["x"] for q in rows})},
                "G0b": g0b(rs), "G0c": g0c(rows, rng), "G0c_net": g0c(rows, rng, "net"),
                "G0c_by_xapp": {x: g0c([q for q in rows if q["x"] == x], rng) for x in sorted({q["x"] for q in rows})}}
        cell["L0_pass"] = bool(cell["G0a"]["pass"] and cell["G0c"].get("pass"))
        report["cells"][f"{pair}:{s}"] = cell
        a, c, b = cell["G0a"], cell["G0c"], cell["G0b"]
        print(f"== {pair} stratum {s} {S.STRATA[s]}: {len(rs)} episodes, {n_lab} labels, "
              f"{cost['cpu_s_total'] / 3600:.2f} CPU-h (label {cost['label_cpu_s_total'] / 3600:.2f})")
        print(f"  G0a reliability dPV_exp={a['exp_pv']} (nonzero {a['exp_pv_nonzero']}) dE={a['exp_e']} "
              f"dV={a['exp_v']} zero-share={a['frac_zero_pv_exp']} pass={a['pass']}")
        for x, v in cell["G0a_by_xapp"].items():
            print(f"    {x}: n={v['n']} rel dPV_exp={v['exp_pv']} dE={v['exp_e']} dV={v['exp_v']}")
        print(f"  G0b explained={b.get('explained_r2')} unmodelled={b['unmodelled_frac']} probes={b['n_probes']} "
              f"switch={b['switch_to_region']} determined={b['determined']}")
        print(f"  G0c AUC={c.get('auc')} LB90={c.get('auc_lb90')} R2={c.get('r2')} n_nonzero={c.get('n_nonzero')} "
              f"pass={c.get('pass')}")
        print(f"  L0 pass={cell['L0_pass']}  per-label CPU-s={cost['label_cpu_s_per_label']} "
              f"collect CPU-s/ep={cost['collect_cpu_s_per_ep']:.1f}")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs(cells_arg=None):
    state = S.load_state()
    cells = cells_from(cells_arg, state)
    U = units_for("L0", cells)
    print("cells:", cells, "registry:", registry_check())
    print(f"L0: {len(U)} episodes; seeds {[u[3] for u in U]}")


def cli(argv, stage=None):
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        run(a.get("--stage") or stage or "L0", a["--part"], a["--out"], smoke="--smoke" in flags,
            cells_arg=a.get("--cells"))
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
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        list_jobs(a.get("--cells"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
