"""E6-P MSCR causal discovery, step 1: COLLECTION + KNOCKOUT GROUND TRUTH (scratchpad/e6_dev/decision/
STEP1_MSCR_PLAN.md; not frozen). The analysis half (crt_units.py, baselines_disc.py, edge_score.py) reads the JSONL
records written here; the RECORD FORMAT below is its contract.

Plant: cell P3 surge-L40 = ``e6p_screen.make_cfg("P3", 3, seed, lf)`` (frozen E6P config, 120 s warm-up + 600 s
scored = 720 s; lf = the calibrated L40 load factor from e6p_state, 1.75546875). ``E6Env(cfg, log=False, wg3=True,
trace=True)`` + ``collect_p.run_collection`` (``units_p.UnitArbiter``, T = 60 s, open_rule "feasible", warm-up:
every request accepted, no unit opens).

CLI (repo root, PYTHONPATH=.; in the cloud bundle the same file is e6dev/e6p_discovery.py):
  python scratchpad/e6_dev/e6p_discovery.py run --stage {dev,eval,placebo,gt,prof} --part i/k --out F.jsonl
         [--smoke] [--short SCORED_S] [--max-labels N]
  python scratchpad/e6_dev/e6p_disc_<stage>.py run --part i/k --out F.jsonl      # cloud.py wrappers (baked stage)
  python scratchpad/e6_dev/e6p_discovery.py summary --in FILES... [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_discovery.py list
Jobs are dealt to shards by index (job u -> shard u % k); resumable: episodes whose key is already in --out (same
smoke flag) are skipped. --smoke: DEV plumbing only; the job's seed is remapped to seed % 31 (E6 DEV seeds 0-30),
one job per call, records flagged smoke=True (summary ignores them unless --allow-smoke); --short S sets the scored
seconds (smoke only; warm-up stays 120 s); --max-labels caps GT labels (smoke only).

Stages, policies and seeds (every protocol seed asserted in [180000, 183999] and outside 150200-150399,
155200-155399 and 160000-179999; block registered as E6 dev_reserved "e6p_discovery_episodes"):
  dev      pi0 = collect_p.PI0_HIGH_NO_RB (accept .5 / half .2 / reject .3 for ES, PowerES, SliceGuarantee)
           seed = 183300 + j, j = 0..19                                   (allowed before the freeze)
  eval     pi0, seed = 183300 + j, j = 20..79; fold = (j - 20) // 20 in {0, 1, 2}    (FROZEN protocol required)
  gt       pi0 base path + knockout labels (gt_p), seed = 183300 + j, j = 80..99      (FROZEN protocol required)
  placebo  collect_p.PlaceboPolicy (pi0 modes drawn + logged, accept applied), seed = 183400 + j, j = 0..19
  prof     PACIFISTA profiles: only xApp X's units accepted, every other xApp's unit rejected (warm-up all-accept as
           in every stage); X in PROFILES = (ES, PowerES, SliceGuarantee), seed = 183420 + 8 * profile_idx + e,
           e = 0..7                                                       (allowed before the freeze)
  Sub-block map: 183300-183319 dev | 183320-183379 eval | 183380-183399 gt | 183400-183419 placebo |
  183420-183443 prof | the rest of 180000-183999 unused. pi0 draws: collect_p (tag 6612, [seed, 6612, c, x_idx, t0]);
  GT label sampling: labels_p.sampled (tag 6613, rate gt_p.GT_RATE per knob family); GT bootstrap tag 6617.
Freeze: eval / gt (non-smoke) refuse to run unless docs/benchmark/E6P_DISCOVERY_PROTOCOL.md exists and its sha256
(LF-normalised) equals FROZEN_SHA256 (None until frozen). cloud.py bundles that doc (and SEED_REGISTRY.json) for
e6p_disc_* scripts, so the same check runs in the cloud bundle.

RECORD FORMAT (schema "e6p-disc-rec/1"). JSONL; line kinds: "header" (one per run call: plan, consts, freeze and
registry status, code / numeric env), "episode" (one per episode, below), "close". Arrays are ENC dicts
{"b64": base64(zlib(little-endian float32 C-order bytes)), "shape": [...], "dtype": "float32"}; decode with
``cdd_oran.decision.collect_p.dec`` (float32 values: integers < 2^24 exact). Episode keys:
  key [stage, sub, seed]; stage; sub ("" or the profile xApp); seed (protocol seed); cfg_seed (= seed, or seed % 31
  when smoke); j; fold (eval only, else null); smoke; short; policy ("pi0" | "placebo" | "profile:<X>");
  pi0_table ({xApp: {mode: p}}, the logged-propensity table; for placebo the table the logged modes were drawn from;
  for prof null); T (unit hold s); open_rule; load_factor; episode_s (720); warmup_s; n_cells; step_s (panel step).
  obs_static      {"is_macro": [C] bool, "neighbours": obs["static"]["neighbours"] lists}  (obs-only)
  units           list, one per opened unit, in opening order (index i = position):
                  i, c (cell), x (xApp), x_idx, knob (carrier | sleep | ptx | prot_min), t0 (s, the second the unit
                  opened; its decisions hold for t in [t0, t0 + T)), mode (LOGGED pi0 draw: accept | half | reject;
                  prof: accept / reject by profile), p (its logged propensity), applied_mode (what the arbiter
                  applied: = mode, except placebo = "accept"), cur / prop / step (opening request: current value,
                  proposed value, prop - cur), exp (exposure set N(c), obs-only), n_req (requests of x on c in the
                  unit), n_applied (of those, requests that changed the knob: trace out code "ok"), rb_issued,
                  ctx (obs-only context, units_p.UnitArbiter.context), lab_kpi (PRIVILEGED realised window KPIs of
                  collect_p: pv / e / v over N(c) and over c alone, seconds t0+1 .. t0+T, trunc flag).
  panel           features.panel_to_rec(build_panel_p(trace, step_s=10, drop_degenerate=False)): obs-only per
                  (10 s step, cell) panel, rows step-major then cell; {"schema": "e6p-panel/1", "columns", "kind",
                  "family", "data": {col: ENC (n_rows,)}, "t": ENC, "cell": ENC, "episode": ENC, "neighbours"
                  (CIO-linked), "ownership", "xapps", "dropped", "notes"}; NO cell_region (the site map is
                  privileged); rebuild with features.panel_from_rec, pool with concat_panels(...,
                  drop_degenerate=False) + features.drop_degenerate_p.
  lab_series      PRIVILEGED per-second per-cell plant series (collect_p.CellKPITap.series): {"fields": [pv, v, e,
                  rlf, prb_used, prb_cap, ue, prot_ue], "t_first": 1, "scored": ENC (T,) 0/1, "data": ENC
                  (T, F, C)}; row i = env second t = i + 1 = window (i, i+1] (warm-up rows included, scored = 0).
                  Unit target of the plan: KPI over seconds t0+1 .. t0+90 minus the pre-window, summed over the
                  relation's cells = rows t0 .. t0+89 (0-based) of data[:, f, cells].
  lab_outcome     PRIVILEGED episode totals (e6p_screen.outcome: sla counters, psvr, svr, energy, env stats).
  gt_static       PRIVILEGED layout truth, NEVER read by discovery: {"cand": {pico: ES candidate macro},
                  "cell_site": [C], "neighbours": plant layout neighbour lists, "is_macro": [C]}.
  gt_labels       PRIVILEGED (gt stage; [] otherwise): one per labelled unit: i (unit index), c, x, knob, t0, step,
                  exp, ks [1, 2, 3], H 90, kpis = labels_p.CELL_KPIS (pv, e, v, rlf, load, prb), delta = ENC
                  (len(ks), len(kpis), C) per-cell ACCEPT - REJECT contrast (CRN: same reseed k in both arms,
                  every other unit accepted), d_net_reject ({kpi: [per-k reject - accept, network]}), secs, cpu_s.
  counts          units per knob family and logged mode, sleep units / rejects, labels, skips.
  cpu_s, collect_cpu_s, label_cpu_s, panel_cpu_s, secs (wall), n_roll.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_screen as S  # noqa: E402  (sets ROOT / BUNDLE and sys.path)

from cdd_oran.decision import collect_p as CP  # noqa: E402
from cdd_oran.decision import features as F  # noqa: E402
from cdd_oran.decision import gt_p as G  # noqa: E402
from cdd_oran.decision import labels_p as LP  # noqa: E402
from cdd_oran.decision import units_p as UP  # noqa: E402
from cdd_oran.envs.e6.env import E6Env  # noqa: E402

PLAN = "scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md (2026-09-29, not frozen)"
SCHEMA = "e6p-disc-rec/1"
PROTOCOL_DOC = "docs/benchmark/E6P_DISCOVERY_PROTOCOL.md"
REGISTRY_DOC = "docs/benchmark/SEED_REGISTRY.json"
FROZEN_SHA256 = "f9d8774021fbdfed21f265efc0e00f07f7a626124914261ba9354b5132cccb75"  # E6P_DISCOVERY_PROTOCOL.md frozen 2026-09-29
PAIR, STRATUM = "P3", 3                 # P3 surge-L40
SEED_BLOCK = (180000, 183999)
FORBIDDEN = ((150200, 150399), (155200, 155399), (160000, 179999))
CELL_BASE, PB_BASE = 183300, 183400     # P3 s3 block; placebo + profiles block
STAGES = ("dev", "eval", "eval_ext", "placebo", "gt", "prof")
FROZEN_STAGES = ("eval", "eval_ext", "gt")
J_RANGE = {"dev": range(0, 20), "eval": range(20, 80), "gt": range(80, 100)}
N_PLACEBO, N_PROF = 20, 8
PROFILES = ("ES", "PowerES", "SliceGuarantee")
TABLE = CP.PI0_HIGH_NO_RB
OPEN_RULE = "feasible"
STEP_S = 10
T = UP.T_UNIT
TAGS = {6612: "pi0 draws", 6613: "label sampling", 6614: "region probes", 6615: "G0c bootstrap", 6616: "CRT draws",
        6617: "GT bootstrap"}
K1 = dict(min_sleep_units=60, min_sleep_rejects=15)


# ---------------------------------------------------------------------------------------------- seeds / jobs
def check_seed(seed: int) -> int:
    seed = int(seed)
    assert SEED_BLOCK[0] <= seed <= SEED_BLOCK[1], f"seed {seed} outside the e6p_discovery block {SEED_BLOCK}"
    for lo, hi in FORBIDDEN:
        assert not lo <= seed <= hi, f"seed {seed} inside a forbidden block [{lo}, {hi}]"
    return seed


def jobs(stage: str) -> list:
    """[(stage, sub, seed, j, fold)] in shard order."""
    if stage in J_RANGE:
        return [(stage, "", check_seed(CELL_BASE + j), j, (j - 20) // 20 if stage == "eval" else None)
                for j in J_RANGE[stage]]
    if stage == "eval_ext":   # K1 extension (protocol sec. 11.5): EVAL folds 3-5 on 183460-183519, records are "eval"
        return [("eval", "ext", check_seed(PB_BASE + N_PLACEBO + N_PROF * len(PROFILES) + 16 + j), j, 3 + j // 20)
                for j in range(60)]
    if stage == "placebo":
        return [(stage, "", check_seed(PB_BASE + j), j, None) for j in range(N_PLACEBO)]
    if stage == "prof":
        return [(stage, x, check_seed(PB_BASE + N_PLACEBO + N_PROF * pi + e), e, None)
                for pi, x in enumerate(PROFILES) for e in range(N_PROF)]
    raise SystemExit(f"--stage in {STAGES}")


def _sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def freeze_status() -> dict:
    path = os.path.join(S.ROOT, PROTOCOL_DOC)
    have = os.path.exists(path)
    sha = _sha_lf(path) if have else None
    return {"doc": PROTOCOL_DOC, "exists": have, "sha256": sha, "frozen_sha256": FROZEN_SHA256,
            "frozen": bool(have and FROZEN_SHA256 is not None and sha == FROZEN_SHA256)}


def registry_check() -> dict | None:
    """{block, tags} registered in SEED_REGISTRY.json (None when the file is absent, e.g. an older bundle)."""
    path = os.path.join(S.ROOT, REGISTRY_DOC)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
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
    return {"block_180000_183999": any(lo <= SEED_BLOCK[0] and hi >= SEED_BLOCK[1] for lo, hi in rngs),
            "tags_6612_6617": all(t in d.get("rng_stream_tags", []) for t in TAGS)}


# ---------------------------------------------------------------------------------------------- policies
class ProfilePolicy:
    """PACIFISTA profile: accept every unit of xApp ``keep``, reject every other xApp's unit (propensity 1)."""

    def __init__(self, keep: str):
        self.keep = keep

    def __call__(self, unit):
        return ("accept", 1.0) if unit["x"] == self.keep else ("reject", 1.0)


def make_policy(stage, sub, sd):
    if stage in ("dev", "eval", "gt"):
        return CP.RandomizedUnitPolicy(sd, tables=TABLE), "pi0", TABLE
    if stage == "placebo":
        return CP.PlaceboPolicy(sd, tables=TABLE), "placebo", TABLE
    if stage == "prof":
        return ProfilePolicy(sub), f"profile:{sub}", None
    raise SystemExit(f"--stage in {STAGES}")


# ---------------------------------------------------------------------------------------------- one episode
def _n_applied(trace, units, T_unit):
    a, meta = trace.arrays, trace.meta
    kidx = {tuple(k): i for i, k in enumerate(meta["knobs"])}
    xidx = {x: i for i, x in enumerate(meta["xapps"])}
    ok = meta["out_codes"].index("ok")
    t, xs, ks, out = a["rq_t"], a["rq_xapp"], a["rq_knob"], a["rq_out"]
    res = []
    for u in units:
        k = kidx.get((u["knob"], int(u["c"])), -1)
        m = (xs == xidx.get(u["x"], -1)) & (ks == k) & (t >= u["t0"]) & (t < u["t0"] + T_unit) & (out == ok)
        res.append(int(m.sum()))
    return res


def unit_record(i, u, n_app):
    ctx = u["ctx"]
    return {"i": i, "c": int(u["c"]), "x": u["x"], "x_idx": int(u["x_idx"]), "knob": u["knob"], "t0": float(u["t0"]),
            "mode": u["mode"], "p": float(u["p"]), "applied_mode": u.get("applied_mode", u["mode"]),
            "cur": ctx["cur"], "prop": ctx["prop"], "step": ctx["step"], "exp": [int(v) for v in u["exp"]],
            "n_req": int(u["n_req"]), "n_applied": n_app, "rb_issued": bool(u.get("rb_issued", False)),
            "ctx": ctx, "lab_kpi": u.get("kpi")}


def episode_job(stage, sub, seed, j, fold, lf, smoke=False, short=None, max_labels=None):
    check_seed(seed)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    policy, pol_name, table = make_policy(stage, sub, sd)
    env = E6Env(cfg, log=False, wg3=True, trace=True)
    lab = LP.Labeller(ks=G.GT_KS, H=G.GT_H, T=T, cells=True) if stage == "gt" else None
    cnt = {"labels": 0, "skipped_end": 0, "skipped_cap": 0}

    def hook(env_, obs, snap, opened):
        for u in opened:
            rate = G.GT_RATE.get(u["knob"])
            if rate is None or not LP.sampled(sd, u, rate):
                continue
            if u["t0"] + G.GT_H > env_.total_s:
                cnt["skipped_end"] += 1
                continue
            if max_labels is not None and cnt["labels"] >= max_labels:
                cnt["skipped_cap"] += 1
                continue
            u["_gt"] = lab.label(env_, obs, snap, u, modes=G.GT_MODES)
            cnt["labels"] += 1

    t_wall = time.time()
    res = CP.run_collection(cfg, policy, T=T, labeller=hook if lab is not None else None, env=env,
                            open_rule=OPEN_RULE)
    t_p = time.process_time()
    trace = env.get_trace()
    panel = F.build_panel_p(trace, step_s=STEP_S, episode=seed, drop_degenerate=False)
    panel_cpu = time.process_time() - t_p
    units = res["units"]
    n_app = _n_applied(trace, units, T)
    urecs = [unit_record(i, u, n_app[i]) for i, u in enumerate(units)]
    gt_labels = []
    for i, u in enumerate(units):
        L = u.get("_gt")
        if L is None:
            continue
        gt_labels.append({"i": i, "c": int(u["c"]), "x": u["x"], "knob": u["knob"], "t0": float(u["t0"]),
                          "step": u["ctx"]["step"], "exp": [int(v) for v in u["exp"]], "ks": L["ks"], "H": L["H"],
                          "kpis": L["cell_kpis"], "delta": CP.enc(G.unit_delta(L)),
                          "d_net_reject": L["d"]["reject"]["net"], "secs": [r["secs"] for r in L["raw"]["reject"]],
                          "cpu_s": L["cpu_s"]})
    by = {}
    for u in urecs:
        b = by.setdefault(u["knob"], {})
        b[u["mode"]] = b.get(u["mode"], 0) + 1
    sl = [u for u in urecs if u["knob"] == "sleep"]
    cnt.update(units=len(urecs), by_family_mode=by, sleep_units=len(sl),
               sleep_rejects=sum(u["mode"] == "reject" for u in sl))
    ser = res["tap"].series()
    lay, pl = env.plant.lay, env.plant
    es = next((x for x in env.xapps if x.name == "ES"), None)
    st = env.static()
    label_cpu = lab.cpu_s if lab is not None else 0.0
    return {"kind": "episode", "schema": SCHEMA, "key": [stage, sub, seed], "stage": stage, "sub": sub, "seed": seed,
            "cfg_seed": sd, "j": j, "fold": fold, "smoke": smoke, "short": short, "policy": pol_name,
            "pi0_table": table, "T": T, "open_rule": OPEN_RULE, "load_factor": lf, "episode_s": env.total_s,
            "warmup_s": cfg.warmup_s, "n_cells": int(pl.nc), "step_s": STEP_S,
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
            "gt_labels": gt_labels, "counts": cnt, "cpu_s": round(res["cpu_s"] + panel_cpu, 2),
            "collect_cpu_s": round(res["cpu_s"] - label_cpu, 2), "label_cpu_s": round(label_cpu, 2),
            "panel_cpu_s": round(panel_cpu, 2), "n_roll": lab.n_roll if lab is not None else 0,
            "secs": round(time.time() - t_wall, 1)}


# ---------------------------------------------------------------------------------------------- run
def run(stage, part, out, smoke=False, short=None, max_labels=None):
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    if (short or max_labels) and not smoke:
        raise SystemExit("--short / --max-labels are smoke-only")
    fz = freeze_status()
    if stage in FROZEN_STAGES and not smoke and not fz["frozen"]:
        raise SystemExit(f"stage {stage} needs the frozen protocol {PROTOCOL_DOC} with a matching FROZEN_SHA256: {fz}")
    reg = registry_check()
    if not smoke and not (reg is not None and all(reg.values())) and not os.environ.get("E6P_DISC_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block / RNG tags not registered in {REGISTRY_DOC}: {reg}")
    state = S.load_state()
    lf = S.lf_of(state, STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header(stage, part, smoke, state)
    head.update(kind="header", schema=SCHEMA, plan=PLAN, driver="e6p_discovery", freeze=fz, registry=reg,
                protocol=f"{PROTOCOL_DOC} (plant config: e6p_screen.make_cfg, frozen E6P values)",
                consts={"pair": PAIR, "stratum": STRATUM, "load_factor": lf, "T": T, "open_rule": OPEN_RULE,
                        "step_s": STEP_S, "pi0_table": TABLE, "profiles": PROFILES, "cell_base": CELL_BASE,
                        "pb_base": PB_BASE, "j_range": {s: [r.start, r.stop - 1] for s, r in J_RANGE.items()},
                        "gt": {"ks": G.GT_KS, "H": G.GT_H, "modes": G.GT_MODES, "rate": G.GT_RATE,
                               "kpis": LP.CELL_KPIS}, "tags": TAGS, "series_fields": CP.SERIES_FIELDS,
                        "smoke": {"short": short, "max_labels": max_labels} if smoke else None})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "episode" and r.get("smoke") == smoke}
    t0, n = time.time(), 0
    J = jobs(stage)
    for u, (st, sub, seed, j, fold) in enumerate(J):
        if u % k != i or (st, sub, seed) in done:
            continue
        rec = episode_job(st, sub, seed, j, fold, lf, smoke, short, max_labels)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("stage", "sub", "seed", "cfg_seed", "counts", "cpu_s",
                                                  "label_cpu_s", "secs")}), flush=True)
        if smoke:
            break
    S._append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_units": len(J),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def gt_episode(rec) -> dict:
    """gt_p input dict of one gt-stage episode record."""
    return {"seed": rec["seed"], "n_cells": rec["n_cells"], "gt_static": rec["gt_static"],
            "gt_labels": [dict(L, delta=CP.dec(L["delta"])) for L in rec["gt_labels"]]}


def _pv_sparsity(recs, H=G.GT_H):
    cell_zero, unit_zero, own_zero, n_units = [], 0, 0, 0
    for r in recs:
        ls = r["lab_series"]
        D = CP.dec(ls["data"])
        sc = CP.dec(ls["scored"]) > 0.5
        pv = D[:, ls["fields"].index("pv"), :]
        cell_zero.append(float((pv[sc] == 0).mean()) if sc.any() else float("nan"))
        for u in r["units"]:
            t0 = int(u["t0"])
            if t0 + H > len(pv):
                continue
            n_units += 1
            unit_zero += int(pv[t0:t0 + H][:, u["exp"]].sum() == 0)
            own_zero += int(pv[t0:t0 + H][:, u["c"]].sum() == 0)
    return {"cell_second_zero_frac": float(np.nanmean(cell_zero)) if cell_zero else None,
            "unit_exp_window_zero_frac": unit_zero / n_units if n_units else None,
            "unit_own_window_zero_frac": own_zero / n_units if n_units else None, "window_s": H}


def summary(paths, json_out=None, allow_smoke=False):
    recs = [r for p in paths for r in S._read(p) if r.get("kind") == "episode" and r.get("schema") == SCHEMA]
    recs = [r for r in recs if allow_smoke or not r.get("smoke")]
    uniq = {tuple(r["key"]) + (r.get("smoke"),): r for r in recs}
    recs = list(uniq.values())
    report = {"plan": PLAN, "freeze": freeze_status(), "stages": {}}
    for stage in STAGES:
        rs = [r for r in recs if r["stage"] == stage]
        if not rs:
            continue
        fam = {}
        for r in rs:
            for f, modes in r["counts"]["by_family_mode"].items():
                d = fam.setdefault(f, {})
                for m, v in modes.items():
                    d[m] = d.get(m, 0) + v
        ne = len(rs)
        sleep_u = sum(r["counts"]["sleep_units"] for r in rs)
        sleep_r = sum(r["counts"]["sleep_rejects"] for r in rs)
        rep = {"episodes": ne, "smoke": sorted({bool(r["smoke"]) for r in rs}),
               "units_by_family_mode": fam,
               "units_per_episode": {f: sum(d.values()) / ne for f, d in fam.items()},
               "sleep": {"units": sleep_u, "rejects": sleep_r, "per_episode": sleep_u / ne,
                         "per_episode_list": [r["counts"]["sleep_units"] for r in rs],
                         "K1_support": sleep_u >= K1["min_sleep_units"] and sleep_r >= K1["min_sleep_rejects"],
                         "K1_rule": K1},
               "pv_sparsity": _pv_sparsity(rs),
               "cpu_s_per_episode": float(np.mean([r["cpu_s"] for r in rs])),
               "label_cpu_s_per_episode": float(np.mean([r["label_cpu_s"] for r in rs])),
               "cpu_h_total": float(sum(r["cpu_s"] for r in rs) / 3600)}
        if stage == "gt":
            eps = [gt_episode(r) for r in rs]
            gs = G.summarize(eps)
            rep["gt"] = gs
            nl = sum(len(r["gt_labels"]) for r in rs)
            rep["label_cpu_s_per_label"] = float(sum(r["label_cpu_s"] for r in rs) / nl) if nl else None
        report["stages"][stage] = rep
        print(f"== {stage}: {ne} episodes, {rep['cpu_h_total']:.2f} CPU-h; units/episode "
              f"{ {f: round(v, 1) for f, v in rep['units_per_episode'].items()} }")
        print(f"   modes {fam}")
        print(f"   sleep units {sleep_u} (rejects {sleep_r}, {sleep_u / ne:.2f}/episode), K1 support "
              f"{rep['sleep']['K1_support']}; pv sparsity {rep['pv_sparsity']}")
        if stage == "gt":
            print(f"   GT labels {rep['gt']['n_labels']} {rep['gt']['labels_per_family']}; per-label CPU-s "
                  f"{rep['label_cpu_s_per_label']}")
            tab = rep["gt"]["tables"][G.PRIMARY_ORIENT]
            print(f"   GT ({tab['orient']}) counts {tab['counts']} delta {tab['delta']}")
            for c in tab["cells"]:
                if c["status"] != "INDET" or c["n"] >= G.GT_MIN_UNITS:
                    print(f"     {c['family']:>8} {c['relation']:>3} {c['kpi']:>4} {c['status']:>5}"
                          f"{'(+)' if c['sign'] > 0 else '(-)' if c['sign'] < 0 else '   '} mean {c['mean']:.3g} "
                          f"CI [{c['ci'][0]:.3g}, {c['ci'][1]:.3g}] n {c['n']} eps {c['n_eps']}")
            for s in rep["gt"]["receiving"]:
                print(f"     recv seed {s['seed']} pico {s['pico']}: n {s['n_units']} top1 {s['top1']} cand "
                      f"{s['cand']} share_cand {s['share_cand']}")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs():
    print("plan:", PLAN)
    print("freeze:", freeze_status())
    print("registry:", registry_check())
    for st in STAGES:
        J = jobs(st)
        print(f"{st}: {len(J)} episodes; seeds {J[0][2]}..{J[-1][2]}"
              + (f"; subs {sorted({x[1] for x in J})}" if st == "prof" else "")
              + ("; needs FROZEN protocol" if st in FROZEN_STAGES else ""))


def cli(argv, stage=None):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        run(a.get("--stage") or stage, a["--part"], a["--out"], smoke="--smoke" in flags,
            short=float(a["--short"]) if a.get("--short") else None,
            max_labels=int(a["--max-labels"]) if a.get("--max-labels") else None)
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
