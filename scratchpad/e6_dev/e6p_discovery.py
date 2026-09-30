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

Discovery v4 (docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md sections 2, 4, 9; added 2026-09-30). Fresh block
188000-189999 (V4_BLOCK; registered E6 dev_reserved "e6p_discovery_v4_episodes"; check_seed(seed, v4=True) accepts
ONLY this block for the v4 stages and asserts it is outside V4_FORBIDDEN; the v1-v3 stages keep the old block / check):
  dev_v4      188000-188019 (j 0-19)    records stage "dev",     sub "v4"          (allowed before the freeze)
  eval_v4     188100-188699 (j 0-599)   records stage "eval",    sub "v4", fold = j // 120, slice60 = j // 60,
                                        slice120 = j // 120                        (FROZEN v4 protocol required)
  placebo_v4  188700-188739 (j 0-39)    records stage "placebo", sub "v4"          (FROZEN v4 protocol required)
  gt_v4       188800-188839 (j 0-39)    records stage "gt",      sub "v4"          (FROZEN v4 protocol required)
  (188900-189999 reserve, unused). Policy collect_p.PI0_V4 (accept .5 / reject .5 for ES, PowerES, SliceGuarantee;
  placebo: PlaceboPolicy(tables=PI0_V4)); run_collection(arb_warmup_s=0.0, count_all=True): randomized from t = 0, the
  tap's cumulative arrays count warm-up seconds (both logged in the header consts and in every record). GT hook: units
  with t0 < V4_T0_MIN = 90 are not labelled (counts "skipped_early"). Freeze check per stage: the v4 frozen stages need
  docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md with LF-normalised sha256 == FROZEN_SHA256_V4 (None until frozen: refuse).
  Registry: v4 stages also require the v4 block to be registered. Extra record keys (v4 only): arb_warmup_s, count_all,
  slice60 / slice120 (eval_v4; else null); counts.skipped_early (gt_v4).

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
V4_STAGES = ("dev_v4", "eval_v4", "placebo_v4", "gt_v4")
STAGES = ("dev", "eval", "eval_ext", "eval_v2", "eval_v3", "placebo", "placebo_ext", "gt", "gt_ext", "prof") + V4_STAGES
FROZEN_STAGES_V4 = ("eval_v4", "gt_v4", "placebo_v4")
FROZEN_STAGES = ("eval", "eval_ext", "eval_v2", "eval_v3", "gt", "gt_ext") + FROZEN_STAGES_V4
GT_EXT_BASE = 183520                    # v2 re-test GT (added 2026-09-30, after the v1 verdict; records are "gt")
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

# ---- discovery v4 (module docstring, "Discovery v4")
PROTOCOL_DOC_V4 = "docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md"
FROZEN_SHA256_V4 = "6857466c91afa296a5cbaabd8fe0be9cb036dd35358736e855adfc540faf84ec"                 # set to the V4 doc's LF-normalised sha256 at the freeze (user)
V4_BLOCK = (188000, 189999)
V4_FORBIDDEN = ((150000, 150399), (155000, 155399), (160000, 179999), (180000, 187999), (190000, 190399))
V4_SUB = "v4"
V4_BASE = {"dev_v4": 188000, "eval_v4": 188100, "placebo_v4": 188700, "gt_v4": 188800}
V4_N = {"dev_v4": 20, "eval_v4": 600, "placebo_v4": 40, "gt_v4": 40}
V4_RECORD_STAGE = {"dev_v4": "dev", "eval_v4": "eval", "placebo_v4": "placebo", "gt_v4": "gt"}
V4_ARB_WARMUP_S = 0.0                   # arbiter randomizes from t = 0 (the plant warm-up stays 120 s)
V4_COUNT_ALL = True                     # tap cumulative arrays include warm-up seconds
V4_T0_MIN = 90                          # GT labels only for units with t0 >= 90 (the tested population, H_pre = 90)
TABLE_V4 = CP.PI0_V4
for _lo, _hi in V4_FORBIDDEN:           # the v4 block must be disjoint from every forbidden block
    assert V4_BLOCK[1] < _lo or V4_BLOCK[0] > _hi, f"V4_BLOCK {V4_BLOCK} overlaps forbidden [{_lo}, {_hi}]"
_v4r = sorted((V4_BASE[_s], V4_BASE[_s] + V4_N[_s] - 1) for _s in V4_STAGES)
assert all(V4_BLOCK[0] <= a <= b <= V4_BLOCK[1] for a, b in _v4r), _v4r          # every v4 stage inside the block
assert all(a[1] < b[0] for a, b in zip(_v4r, _v4r[1:], strict=False)), _v4r                 # stages pairwise disjoint


def is_v4(stage: str | None) -> bool:
    return stage in V4_STAGES


# ---------------------------------------------------------------------------------------------- seeds / jobs
def check_seed(seed: int, v4: bool = False) -> int:
    """v1-v3 stages: the 180000-183999 block minus FORBIDDEN (unchanged). v4 stages (``v4``): ONLY 188000-189999,
    never inside V4_FORBIDDEN."""
    seed = int(seed)
    if v4:
        assert V4_BLOCK[0] <= seed <= V4_BLOCK[1], f"seed {seed} outside the e6p_discovery_v4 block {V4_BLOCK}"
        for lo, hi in V4_FORBIDDEN:
            assert not lo <= seed <= hi, f"seed {seed} inside a forbidden block [{lo}, {hi}]"
        return seed
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
    if stage == "eval_v2":    # discovery v2 re-test EVAL (power-sized, 480 eps): the unused eval_ext seeds 183460-183519
        #                       + 183560-183979; records "eval", sub "v2", fold = j // 120 (4 folds)
        seeds = list(range(183460, 183520)) + list(range(183560, 183980))
        return [("eval", "v2", check_seed(sd), j, j // 120) for j, sd in enumerate(seeds)]
    if stage == "eval_v3":    # discovery v3 re-test EVAL (power-sized from v2, 1200 eps): 182000-183199 (unused part of
        #                       the registered block); records "eval", sub "v3", fold = j // 300 (4 folds)
        return [("eval", "v3", check_seed(182000 + j), j, j // 300) for j in range(1200)]
    if stage == "gt_ext":     # fresh knockout GT for the discovery v2 re-test: 183520-183539, records are "gt"
        return [("gt", "ext", check_seed(GT_EXT_BASE + j), j, None) for j in range(20)]
    if stage == "placebo_ext":  # fresh sharp-null calibration for the v2 re-test: 183540-183559, records "placebo"
        return [("placebo", "ext", check_seed(GT_EXT_BASE + 20 + j), j, None) for j in range(N_PLACEBO)]
    if stage == "placebo":
        return [(stage, "", check_seed(PB_BASE + j), j, None) for j in range(N_PLACEBO)]
    if stage == "prof":
        return [(stage, x, check_seed(PB_BASE + N_PLACEBO + N_PROF * pi + e), e, None)
                for pi, x in enumerate(PROFILES) for e in range(N_PROF)]
    if is_v4(stage):          # discovery v4 (protocol V4 section 4): records stage dev / eval / placebo / gt, sub "v4"
        rs = V4_RECORD_STAGE[stage]
        return [(rs, V4_SUB, check_seed(V4_BASE[stage] + j, v4=True), j, j // 120 if rs == "eval" else None)
                for j in range(V4_N[stage])]
    raise SystemExit(f"--stage in {STAGES}")


def _sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def freeze_status(stage: str | None = None) -> dict:
    """Freeze status of the protocol doc governing ``stage`` (v4 stages: PROTOCOL_DOC_V4 vs FROZEN_SHA256_V4; every
    other stage: PROTOCOL_DOC vs FROZEN_SHA256, unchanged)."""
    doc, want = (PROTOCOL_DOC_V4, FROZEN_SHA256_V4) if is_v4(stage) else (PROTOCOL_DOC, FROZEN_SHA256)
    path = os.path.join(S.ROOT, doc)
    have = os.path.exists(path)
    sha = _sha_lf(path) if have else None
    return {"doc": doc, "exists": have, "sha256": sha, "frozen_sha256": want,
            "frozen": bool(have and want is not None and sha == want)}


def registry_check(stage: str | None = None) -> dict | None:
    """{block, tags} registered in SEED_REGISTRY.json (None when the file is absent, e.g. an older bundle). v4 stages
    also need the v4 block (key "block_188000_189999")."""
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
    out = {"block_180000_183999": any(lo <= SEED_BLOCK[0] and hi >= SEED_BLOCK[1] for lo, hi in rngs),
           "tags_6612_6617": all(t in d.get("rng_stream_tags", []) for t in TAGS)}
    if is_v4(stage):
        out["block_188000_189999"] = any(lo <= V4_BLOCK[0] and hi >= V4_BLOCK[1] for lo, hi in rngs)
    return out


# ---------------------------------------------------------------------------------------------- policies
class ProfilePolicy:
    """PACIFISTA profile: accept every unit of xApp ``keep``, reject every other xApp's unit (propensity 1)."""

    def __init__(self, keep: str):
        self.keep = keep

    def __call__(self, unit):
        return ("accept", 1.0) if unit["x"] == self.keep else ("reject", 1.0)


def make_policy(stage, sub, sd):
    if sub == V4_SUB:                     # discovery v4: PI0_V4 (accept .5 / reject .5)
        if stage in ("dev", "eval", "gt"):
            return CP.RandomizedUnitPolicy(sd, tables=TABLE_V4), "pi0", TABLE_V4
        if stage == "placebo":
            return CP.PlaceboPolicy(sd, tables=TABLE_V4), "placebo", TABLE_V4
        raise SystemExit(f"no v4 policy for record stage {stage}")
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
    v4 = sub == V4_SUB
    check_seed(seed, v4=v4)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    policy, pol_name, table = make_policy(stage, sub, sd)
    env = E6Env(cfg, log=False, wg3=True, trace=True)
    lab = LP.Labeller(ks=G.GT_KS, H=G.GT_H, T=T, cells=True) if stage == "gt" else None
    cnt = {"labels": 0, "skipped_end": 0, "skipped_cap": 0}
    if v4:
        cnt["skipped_early"] = 0

    def hook(env_, obs, snap, opened):
        for u in opened:
            rate = G.GT_RATE.get(u["knob"])
            if rate is None or not LP.sampled(sd, u, rate):
                continue
            if v4 and u["t0"] < V4_T0_MIN:          # v4: untested (t0 < 90) units are not labelled
                cnt["skipped_early"] += 1
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
                            open_rule=OPEN_RULE, arb_warmup_s=V4_ARB_WARMUP_S if v4 else None,
                            count_all=V4_COUNT_ALL if v4 else False)
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
    extra = {}
    if v4:
        extra = {"arb_warmup_s": V4_ARB_WARMUP_S, "count_all": V4_COUNT_ALL,
                 "slice60": j // 60 if stage == "eval" else None, "slice120": j // 120 if stage == "eval" else None}
    return {**extra, "kind": "episode", "schema": SCHEMA, "key": [stage, sub, seed], "stage": stage, "sub": sub, "seed": seed,
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
    fz = freeze_status(stage)
    if stage in FROZEN_STAGES and not smoke and not fz["frozen"]:
        want = "FROZEN_SHA256_V4" if is_v4(stage) else "FROZEN_SHA256"
        raise SystemExit(f"stage {stage} needs the frozen protocol {fz['doc']} with a matching {want}: {fz}")
    reg = registry_check(stage)
    if not smoke and not (reg is not None and all(reg.values())) and not os.environ.get("E6P_DISC_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block / RNG tags not registered in {REGISTRY_DOC}: {reg}")
    state = S.load_state()
    lf = S.lf_of(state, STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header(stage, part, smoke, state)
    v4 = is_v4(stage)
    head.update(kind="header", schema=SCHEMA, plan=PLAN, driver="e6p_discovery", freeze=fz, registry=reg,
                protocol=f"{fz['doc']} (plant config: e6p_screen.make_cfg, frozen E6P values)",
                consts={"pair": PAIR, "stratum": STRATUM, "load_factor": lf, "T": T, "open_rule": OPEN_RULE,
                        "step_s": STEP_S, "pi0_table": TABLE_V4 if v4 else TABLE,
                        "arb_warmup_s": V4_ARB_WARMUP_S if v4 else None, "count_all": V4_COUNT_ALL if v4 else False,
                        "v4": {"block": V4_BLOCK, "forbidden": V4_FORBIDDEN, "base": V4_BASE, "n": V4_N,
                               "record_stage": V4_RECORD_STAGE, "sub": V4_SUB, "gt_t0_min": V4_T0_MIN}
                        if v4 else None,
                        "profiles": PROFILES, "cell_base": CELL_BASE,
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
    print("freeze v4:", freeze_status("eval_v4"))
    print("registry:", registry_check())
    print("registry v4:", registry_check("eval_v4"))
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
