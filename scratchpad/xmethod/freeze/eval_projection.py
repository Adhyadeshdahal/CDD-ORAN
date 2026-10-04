"""EVAL launch projection (freeze-prep): unit counts of specs/eval/full.json for S in 40..100 and the CPU-h / session-h
they cost, from the DEV cost tables. Estimate only; the T3 decision reads the DEV costs, not this file.

  uv run python scratchpad/xmethod/freeze/eval_projection.py --dev <results/dev dir> [--cdl-records GLOB]
      [--pmrt-nl-agg results/pmrt_nl/agg_valid_gbm.json] --out scratchpad/xmethod/freeze/eval_projection.json

Cost per unit (CPU-s, mean per dataset of the (arm, world, regime, n) cell):
- 10 classic / PMRT arms: <dev>/full/agg.json (Kaggle); CI arms: <dev>/ci_c/agg.json (Kaggle / Colab / Lightning).
- cdl: the cdl DEV run's records so far (--cdl-records, any host) by n; an n without records is scaled from the
  largest measured n by the training steps min(16000, ceil(130 n / 128)) (cdl.py budget); without records the
  cdl worker's table <dev>/cdl_cost_table.json.
- pmrt_nl_eq: the R-42 gbm validity run (--pmrt-nl-agg, mean by n); an n without data: power law in n fitted on
  the measured n (log-log least squares).
- dataset generation: mean gen_cpu_s of the (world, regime, n) cell, once per dataset.
Wall (R-55): EVAL runs 1 process per vCPU (Kaggle 4 per session, Colab 2 per job, VPS 7, R-57), so a process's
wall ~ its CPU-s. The DEV CPU-s came from oversubscribed sessions (8 / 4 processes on 4 / 2 vCPU): an upper
estimate. Host speed (--factors, the R-55 calibration factors.json of xm-citests): costs are Kaggle-reference CPU-s;
a host with factor f (per arm, else the pooled '*') runs a unit in cost / f, so its processes count as f Kaggle
processes for the workload mix (work-weighted harmonic factor). Wall columns use f_lo (slowest observed ratio,
conservative); the *_f columns use the central f. Without --factors every host counts as Kaggle.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import eval_analysis as E  # noqa: E402

SPEC = os.path.join(os.path.dirname(HERE), "specs", "eval", "full.json")
S_GRID = (40, 50, 60)                    # T6 cap 60 (R-56)
EVAL0 = 3_100_000


def fill(spec: dict, S: int) -> dict:
    out = json.loads(json.dumps(spec))
    for b in out["blocks"]:
        if b["seeds"] == "TBD":
            b["seeds"] = [EVAL0, EVAL0 + S - 1]
        elif b["seeds"] == "TBD_HALF":
            b["seeds"] = [EVAL0, EVAL0 + math.ceil(S / 2) - 1]
        elif isinstance(b["seeds"], str):
            raise ValueError(f"unfilled seed placeholder {b['seeds']!r}")
    return out


def steps(n: int) -> int:
    return min(16000, math.ceil(130 * n / 128))


N_GRID = (500, 1000, 4000, 8000, 24000)


def fill_per_n(cpu: dict, basis: dict, arms: set[str]) -> None:
    """For each arm with per-cell DEV costs, add 'arm|*|*|n' = mean over its measured cells at n (used for cells
    without a DEV cost, e.g. pmrt_nl_eq in E4); an n with no measured cell: power law in n fitted on those means."""
    for a in sorted(arms):
        per_n: dict[int, list] = defaultdict(list)
        for k, v in cpu.items():
            p = k.split("|")
            if p[0] == a and p[1] != "*":
                per_n[int(k.rsplit("|n", 1)[1])].append(v)
        if not per_n:
            continue
        m = {n: float(np.mean(v)) for n, v in per_n.items()}
        xs = sorted(m)
        fit = None
        if len(xs) >= 2:
            b, c = np.polyfit(np.log(xs), np.log([m[x] for x in xs]), 1)
            fit = (b, c)
        ext = []
        for n in N_GRID:
            if n in m:
                cpu.setdefault(f"{a}|*|*|n{n}", m[n])
            elif fit:
                cpu.setdefault(f"{a}|*|*|n{n}", float(math.exp(fit[1] + fit[0] * math.log(n))))
                ext.append(n)
        if ext:
            basis[a] += (f"; cells without a DEV cost: per-n mean, n {ext} extrapolated n^{fit[0]:.2f} (used only "
                         "where the arm has planned units)")


def grow_from_pilot(cpu: dict, basis: dict, arm: str, n_ref: int = 4000) -> None:
    """Cells of ``arm`` measured at ``n_ref`` but not at a larger n: cost at n_ref x the mean growth ratio
    cost(n) / cost(n_ref) of the (world, regime) cells measured at both (e.g. the pmrt_nl_eq cost pilot on E2)."""
    have = {tuple(k.split("|")[1:3]): None for k in cpu if k.startswith(f"{arm}|") and k.split("|")[1] != "*"}
    grown = []
    for n in [x for x in N_GRID if x > n_ref]:
        refs = [cpu[f"{arm}|{w}|{r}|n{n}"] / cpu[f"{arm}|{w}|{r}|n{n_ref}"] for w, r in have
                if f"{arm}|{w}|{r}|n{n}" in cpu and f"{arm}|{w}|{r}|n{n_ref}" in cpu]
        if not refs:
            continue
        g = float(np.mean(refs))
        for w, r in have:
            if f"{arm}|{w}|{r}|n{n_ref}" in cpu and f"{arm}|{w}|{r}|n{n}" not in cpu:
                cpu[f"{arm}|{w}|{r}|n{n}"] = cpu[f"{arm}|{w}|{r}|n{n_ref}"] * g
                grown.append(f"{w} {r} n{n} (x{g:.2f})")
    if grown:
        basis[arm] += f"; grown from n {n_ref} by the pilot's ratio: {', '.join(grown)}"


def apply_proxy(cpu: dict, basis: dict, arm: str, proxy: str) -> None:
    """Cells of ``arm`` without a DEV cost = arm's per-n mean x (proxy's cell cost / proxy's mean over the
    (world, regime) the arm was measured in, same n): e.g. pmrt_nl_eq in E4 R3 / R4 from the linear pmrt_eq."""
    meas = {tuple(k.split("|")[1:3]) for k in cpu if k.startswith(f"{arm}|") and k.split("|")[1] != "*"}
    if not meas:
        return
    done = 0
    for k, v in list(cpu.items()):
        p = k.split("|")
        if p[0] != proxy or p[1] == "*" or f"{arm}|{p[1]}|{p[2]}|{p[3]}" in cpu:
            continue
        ref = [cpu[f"{proxy}|{w}|{r}|{p[3]}"] for w, r in meas if f"{proxy}|{w}|{r}|{p[3]}" in cpu]
        base = cpu.get(f"{arm}|*|*|{p[3]}")
        if ref and base and np.mean(ref) > 0:
            cpu[f"{arm}|{p[1]}|{p[2]}|{p[3]}"] = base * v / float(np.mean(ref))
            done += 1
    basis[arm] += f"; {done} cells scaled by {proxy}'s per-cell / per-n cost ratio"


def cost_tables(dev: str, cdl_glob: str | None, nl_agg: str | None, extra: list[str] = (),
                cdl_kaggle: list[float] | None = None, proxies: dict | None = None,
                pilot: dict | None = None) -> tuple[dict, dict, dict, dict]:
    cpu, wall, gen, basis = {}, {}, {}, {}
    measured: set[str] = set()
    for f in ("full/agg.json", "ci_c/agg.json", *extra):
        agg = json.load(open(os.path.join(dev, f), encoding="utf-8"))
        for k, v in agg["cost"].items():
            cpu[k] = v["cpu_s_mean"]
            if v.get("wall_s_mean"):
                wall[k] = v["wall_s_mean"]
            if v.get("gen_cpu_s_mean"):
                gen.setdefault(k.split("|", 1)[1], []).append(v["gen_cpu_s_mean"])
            basis[k.split("|", 1)[0]] = f"DEV {f.rsplit('/', 1)[0]} (measured per cell)"
            measured.add(k.split("|", 1)[0])
    for k, v in ((pilot or {}).get("cost_mean_cpu_s") or {}).items():        # measured cost-pilot cells
        cpu[k] = float(v)
        basis[k.split("|", 1)[0]] += f"; cost pilot {k.split('|', 1)[1]} {v:g}"
    for a in {k.split("|", 1)[0] for k in ((pilot or {}).get("cost_mean_cpu_s") or {})}:
        grow_from_pilot(cpu, basis, a)
    fill_per_n(cpu, basis, measured)
    for arm, proxy in (proxies or {}).items():
        apply_proxy(cpu, basis, arm, proxy)
    if cdl_kaggle:                                  # per-n Kaggle CPU-s table (dev-runs status), all worlds
        for n, c in zip(N_GRID, cdl_kaggle, strict=True):
            cpu[f"cdl|*|*|n{n}"] = float(c)
        basis["cdl"] = "cdl Kaggle CPU-s per n (dev-runs status: " + " / ".join(f"{c:g}" for c in cdl_kaggle) + ")"
        cdl_glob = None
    # cdl
    recs = defaultdict(lambda: [[], []])
    for f in sorted(glob.glob(cdl_glob)) if cdl_glob else []:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("arm") == "cdl":
                x = recs[int(r["job"]["n"])]
                x[0].append(float(r["method_cpu_s"]))
                x[1].append(float(r.get("child_wall_s") or r.get("wall_s")))
    cdl_n = {n: (float(np.mean(c)), float(np.mean(w)), float(np.max(c)), len(c)) for n, (c, w) in recs.items()}
    if cdl_n:
        top = max(cdl_n)
        basis["cdl"] = (f"cdl DEV run records so far ({sum(v[3] for v in cdl_n.values())} units, n "
                        f"{sorted(cdl_n)}); other n scaled from n {top} by training steps")
    elif not cdl_kaggle:
        tab = json.load(open(os.path.join(dev, "cdl_cost_table.json"), encoding="utf-8"))
        basis["cdl"] = "cdl worker cost table (cdl_cost_table.json)"
    for n in () if cdl_kaggle else N_GRID:
        if cdl_n:
            c, w = (cdl_n[n][0], cdl_n[n][1]) if n in cdl_n else (cdl_n[top][0] * steps(n) / steps(top),
                                                                   cdl_n[top][1] * steps(n) / steps(top))
        else:
            c, w = tab[f"cdl|E1|R1|n{n}"], None
        cpu[f"cdl|*|*|n{n}"] = c
        if w:
            wall[f"cdl|*|*|n{n}"] = w
    # pmrt_nl_eq from the R-42 gbm validity run (only where no DEV campaign cost is given)
    if nl_agg and "pmrt_nl_eq" not in measured:
        nl = {int(k.split("|n")[1]): v["mean"] for k, v in json.load(open(nl_agg, encoding="utf-8"))["cost"].items()
              if k.startswith("gbm|")}
        xs = sorted(nl)
        b, a = np.polyfit(np.log(xs), np.log([nl[x] for x in xs]), 1)
        for n in (500, 1000, 4000, 8000, 24000):
            cpu[f"pmrt_nl_eq|*|*|n{n}"] = nl[n] if n in nl else float(math.exp(a + b * math.log(n)))
        basis["pmrt_nl_eq"] = (f"R-42 gbm validity run (mean by n, n {xs}); n 8000 / 24000 extrapolated n^{b:.2f}")
    return cpu, wall, {k: float(np.mean(v)) for k, v in gen.items()}, basis


def t3_max(dev: str, cdl_glob: str | None, nl_agg: str | None, extra: list[str] = (),
           cdl_kaggle: list[float] | None = None, pilot: dict | None = None) -> dict:
    """arm -> n -> max DEV CPU-s per unit (any host, NOT converted by a speed factor), for the T3 read (cdl: the
    per-n Kaggle table when given, a mean, so its T3 max comes from the cdl DEV merge)."""
    out: dict = defaultdict(dict)
    if cdl_kaggle:
        cdl_glob = None
    if any(f.startswith("pmrt_nl/") for f in extra):
        nl_agg = None
    for f in ("full/agg.json", "ci_c/agg.json", *extra):
        for k, v in json.load(open(os.path.join(dev, f), encoding="utf-8"))["cost"].items():
            arm, n = k.split("|", 1)[0], k.rsplit("|n", 1)[1]
            out[arm][n] = round(max(out[arm].get(n, 0.0), v["cpu_s_max"]), 1)
    for f in sorted(glob.glob(cdl_glob)) if cdl_glob else []:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("arm") == "cdl":
                n = str(r["job"]["n"])
                out["cdl"][n] = round(max(out["cdl"].get(n, 0.0), float(r["method_cpu_s"])), 1)
    if nl_agg:
        for k, v in json.load(open(nl_agg, encoding="utf-8"))["cost"].items():
            if k.startswith("gbm|"):
                out["pmrt_nl_eq"][k.split("|n")[1]] = v["max"]
    for k, v in ((pilot or {}).get("cost_max_cpu_s") or {}).items():
        arm, n = k.split("|", 1)[0], k.rsplit("|n", 1)[1]
        out[arm][n] = round(max(out[arm].get(n, 0.0), float(v)), 1)
    return {a: dict(sorted(d.items(), key=lambda x: int(x[0]))) for a, d in sorted(out.items())}


def unit_cost(tab: dict, arm: str, w: str, r: str, n: int) -> float | None:
    return tab.get(f"{arm}|{w}|{r}|n{n}", tab.get(f"{arm}|*|*|n{n}"))


def project(spec: dict, S: int, cpu: dict, gen: dict) -> dict:
    units = E.planned_units(fill(spec, S))
    by_arm, by_n, by_role = defaultdict(float), defaultdict(float), defaultdict(float)
    w_by_arm = defaultdict(float)
    w_by_arm_n = defaultdict(float)
    missing, ds = set(), set()
    gen_s = 0.0
    for u in units.values():
        c = unit_cost(cpu, u["arm"], u["world"], u["regime"], u["n"])
        if c is None:
            missing.add(f"{u['arm']}|n{u['n']}")
            continue
        by_arm[u["arm"]] += c
        by_n[u["n"]] += c
        by_role[u["role"]] += c
        w_by_arm[u["arm"]] += c                                  # 1 process per vCPU (R-55): wall ~ CPU-s
        w_by_arm_n[(u["arm"], u["n"])] += c
        d = (u["world"], u["regime"], u["lam"], u["n"], u["kappa"], u["seed"])
        if d not in ds:
            ds.add(d)
            gen_s += gen.get(f"{u['world']}|{u['regime']}|n{u['n']}", 0.0)
    cpu_h = (sum(by_arm.values()) + gen_s) / 3600
    return {"S": S, "units": len(units), "datasets": len(ds),
            "tune_units": sum(u["role"] == "tune" for u in units.values()),
            "cpu_h": round(cpu_h, 1), "process_wall_h": round((sum(w_by_arm.values()) + gen_s) / 3600, 1),
            "cpu_h_generation": round(gen_s / 3600, 1),
            "cpu_h_by_arm": {k: round(v / 3600, 1) for k, v in sorted(by_arm.items(), key=lambda x: -x[1])},
            "cpu_h_by_n": {str(k): round(v / 3600, 1) for k, v in sorted(by_n.items())},
            "cpu_h_by_role": {k: round(v / 3600, 1) for k, v in by_role.items()},
            "units_without_cost": sorted(missing), "_w_by_arm": dict(w_by_arm), "_gen_s": gen_s}


def host_factor(factors: dict | None, host: str | None, arm: str, which: str) -> float:
    """R-55 factor of `host` for `arm` (else the pooled '*'); 1.0 without factors (host counted as Kaggle)."""
    if not factors or not host:
        return 1.0
    for a in (arm, "*"):
        v = factors["factors"].get(a, {}).get(host)
        if v:
            return float(v[which])
    raise KeyError(f"no R-55 factor for {arm!r} or '*' on {host!r}")


def eff_factor(p: dict, factors: dict | None, host: str | None, which: str) -> float:
    """Work-weighted harmonic factor: Kaggle-ref process-s of the workload / host process-s of the same workload."""
    w = sum(p["_w_by_arm"].values()) + p["_gen_s"]
    h = sum(c / host_factor(factors, host, a, which) for a, c in p["_w_by_arm"].items())
    h += p["_gen_s"] / host_factor(factors, host, "*", which)
    return w / h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--cdl-records", default=None)
    ap.add_argument("--pmrt-nl-agg", default=None)
    ap.add_argument("--extra-agg", nargs="*", default=[], help="more DEV aggregates under --dev (e.g. pmrt_nl/agg.json)")
    ap.add_argument("--cdl-kaggle", default=None, help="cdl Kaggle CPU-s per n 500..24000, comma-separated")
    ap.add_argument("--proxy", nargs="*", default=[], help="ARM=PROXY: cost shape of ARM's unmeasured cells")
    ap.add_argument("--kaggle", type=int, default=4, help="concurrent Kaggle sessions")
    ap.add_argument("--colab", type=int, default=3, help="concurrent Colab CPU jobs")
    ap.add_argument("--kaggle-procs", type=int, default=4, help="processes per Kaggle session = vCPU (R-55)")
    ap.add_argument("--colab-procs", type=int, default=2, help="processes per Colab job = vCPU (R-55)")
    ap.add_argument("--vps-procs", type=int, default=0, help="processes on the VPS (R-57: 7)")
    ap.add_argument("--factors", default=None, help="R-55 calibration factors.json (xm-citests exp_c/calib)")
    ap.add_argument("--vps-host", default="vps|AMD EPYC-Rome Processor", help="host key of the VPS in --factors")
    ap.add_argument("--colab-host", default="colab|Intel(R) Xeon(R) CPU @ 2.20GHz",
                    help="host key of Colab CPU in --factors")
    ap.add_argument("--pilot", default=None, help="cost-pilot json (cost_mean_cpu_s / cost_max_cpu_s per cell)")
    ap.add_argument("--s-grid", default=",".join(map(str, S_GRID)), help="seed counts S to project (frozen spec: S)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(SPEC, encoding="utf-8"))
    s_grid = [int(x) for x in a.s_grid.split(",")]
    if not any(isinstance(b["seeds"], str) for b in spec["blocks"]) and len(s_grid) > 1:
        raise SystemExit("the spec's seeds are filled (frozen): pass --s-grid <its S>")
    ck = [float(x) for x in a.cdl_kaggle.split(",")] if a.cdl_kaggle else None
    pilot = json.load(open(a.pilot, encoding="utf-8")) if a.pilot else None
    cpu, _, gen, basis = cost_tables(a.dev, a.cdl_records, a.pmrt_nl_agg, a.extra_agg, ck,
                                     dict(x.split("=", 1) for x in a.proxy), pilot)
    kp, cp = a.kaggle_procs * a.kaggle, a.colab_procs * a.colab          # concurrent processes (1 per vCPU, R-55)
    vp = a.vps_procs
    factors = json.load(open(a.factors, encoding="utf-8")) if a.factors else None
    out = {"spec": "scratchpad/xmethod/specs/eval/full.json", "cost_basis": basis,
           "platforms": {"kaggle_sessions": a.kaggle, "kaggle_processes_per_session": a.kaggle_procs,
                         "colab_jobs": a.colab, "colab_processes_per_job": a.colab_procs, "vps_processes": vp,
                         "lightning": 0},
           "host_factors": {"source": a.factors, "ref_host": (factors or {}).get("ref_host", "kaggle"),
                            "vps_host": a.vps_host if factors else None,
                            "colab_host": a.colab_host if factors else None},
           "t3_max_cpu_s_by_arm_n": t3_max(a.dev, a.cdl_records, a.pmrt_nl_agg, a.extra_agg, ck, pilot),
           "by_S": {}}
    for S in s_grid:
        p = project(spec, S, cpu, gen)
        pw = p["process_wall_h"]                                          # Kaggle-reference process-h
        for which, sfx in (("f_lo", ""), ("f", "_f")):
            fv = eff_factor(p, factors, a.vps_host if factors else None, which)
            fc = eff_factor(p, factors, a.colab_host if factors else None, which)
            slots = kp + cp * fc + vp * fv                                # Kaggle-equivalent processes
            p[f"vps_factor_eff{sfx}"] = round(fv, 3)
            p[f"colab_factor_eff{sfx}"] = round(fc, 3)
            p[f"wall_h_all_platforms{sfx}"] = round(pw / slots, 1)
            p[f"wall_h_kaggle_vps{sfx}"] = round(pw / (kp + vp * fv), 1)  # Colab unavailable, VPS up
            if not sfx:
                p["kaggle_share_process_wall_h"] = round(pw * kp / slots, 1)
                p["colab_share_process_wall_h"] = round(pw * cp * fc / slots, 1)
                p["vps_share_process_wall_h"] = round(pw * vp * fv / slots, 1)
        p["wall_h_kaggle_only"] = round(pw / kp, 1)                       # Colab unavailable
        del p["_w_by_arm"], p["_gen_s"]
        out["by_S"][str(S)] = p
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for S, p in out["by_S"].items():
        print(S, p["units"], p["datasets"], p["tune_units"], p["cpu_h"], p["process_wall_h"], p["wall_h_all_platforms"],
              p["wall_h_kaggle_vps"], p["wall_h_kaggle_only"], "| central f:", p["wall_h_all_platforms_f"],
              p["wall_h_kaggle_vps_f"], "| f_eff vps", p["vps_factor_eff"], p["vps_factor_eff_f"],
              "colab", p["colab_factor_eff"], p["colab_factor_eff_f"], p["units_without_cost"][:5])
    print(json.dumps(basis, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
