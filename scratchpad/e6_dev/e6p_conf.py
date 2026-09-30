"""E6-P option (a) STUDY driver: causal maps from confounded incumbent logs + the WG3 referee they drive (protocol
docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md, sections 3, 5, 6, 9, 12 and the "Lead decisions"; analysis
scratchpad/e6_dev/e6p_conf_analyze.py).

Plant: identical to K-A / K-A2 / K-B / v4 (``e6p_screen.make_cfg("P3", 3, seed, lf)``, lf = e6p_state L40, 120 s
warm-up + 600 s scored = 720 s). Collection: ``E6Env(cfg, log=False, wg3=True, trace=True)`` +
``collect_p.run_collection(open_rule="feasible", arb_warmup_s=0.0, count_all=True,
arbiter_cls=mapgate.DirectionalUnitArbiter)``, T = 60 s: the DIRECTIONAL arbiter (lead decision 1: a reject unit defers
only its opening direction, as MapGateV2's; K-B's wrapper was not directional). Logging policy
``collect_p.IncumbentPolicy`` (per-unit row ``probs``, ``inc``; pi0_table null) or ``PlaceboIncumbent`` (placebo).

Stages (seed layout of protocol section 6; every seed asserted in its stage range, inside the registered block
186000-187999, outside the K-B block 186000-186079, the reserves and every other registered E6 / XTRUCE block):
  dev_conf  186080-186099 (20)  IncumbentPolicy          records stage "dev"      + repro arm jobs (below)  pre-freeze OK
  disc      186100-186699 (600) IncumbentPolicy          records stage "eval"     (FROZEN_SHA256_CONF required)
  placebo   186700-186899 (200) PlaceboIncumbent         records stage "placebo"  (FROZEN_SHA256_CONF required)
  gt        186900-186939 (40)  IncumbentPolicy base +   records stage "gt"       (FROZEN_SHA256_CONF required)
                                knockout labels (labels_p.Labeller ks 1-3, H 90, modes accept / reject, AA
                                continuation, GT_RATE, tag 6613; units with t0 < 90 NOT labelled: counts.skipped_early)
  eval      187000-187159 (160) every arm of section 5.2  schema e6p-optaka2-rec/1 (FROZEN_SHA256_CONF AND
                                MAPS_SHA256 = the LF sha256 of docs/benchmark/artifacts/E6P_CONF_MAPS.json required)
Collection records: schema "e6p-disc-rec/1" (the K-B format: e6p_opta_kb.unit_record -> ``probs``, ``inc``), sub
"conf", key [record stage, "conf", seed], conf_stage, directional true, arbiter "DirectionalUnitArbiter",
passed (opposite-direction requests accepted inside reject units), incumbent consts, arb_warmup_s 0, tap_count_all
true. dev_conf repro jobs (protocol section 9.1; seeds REPRO_SEEDS = 186080, 186081, arms REPRO_ARMS): "incumbent"
(the EVAL incumbent arm, must reproduce the dev_conf logging trajectory's lab_outcome), "MG:allaccept"
(MapGateV2({})) and "noarb" (bit-identity decides the noarb alias of an all-accept signature); conf_stage
"dev_repro", schema e6p-optaka2-rec/1.

EVAL arms (one job = one (seed, arm), seed-major, resumable): anchors freeze, sub:ES+PowerES, noarb, sub:ES,
sub:PowerES, sub:SliceGuarantee (exactly e6p_step2_dev); references incumbent (DirectionalUnitArbiter(IncumbentPolicy)
from t = 0, T 60, feasible: the dev_conf logging arbiter), never_sleep (e6p_opta_ka), B2 (e6p_step2_dev static
envelope); map arms blanket2, MG:PMRT, MG:GT, MG:rand and MG:<b> for the 13 associational maps: mapgate_v2_arbiter(M),
built from the frozen maps artifact, ONE job per distinct decision signature (``alias_table``; the artifact's "jobs"),
records carry ``signature`` and ``aliases``. An all-accept signature is aliased to noarb only when the artifact says
so (repro check).

CLI (repo root, PYTHONPATH=.; in the cloud bundle e6dev/e6p_conf.py):
  python scratchpad/e6_dev/e6p_conf.py run --stage S --part i/k --out F.jsonl [--smoke] [--short SCORED_S]
         [--max-labels N] [--arms a,b] [--maps FILE]
  python scratchpad/e6_dev/e6p_conf_{dev,disc,placebo,gt,eval}.py run --part i/k --out F.jsonl     # cloud wrappers
  python scratchpad/e6_dev/e6p_conf.py summary --in FILES... [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_conf.py list
--smoke: plumbing only (allowed before the freeze for every stage): seed -> seed % 31, only the jobs of the FIRST seed
of the stage (dev_conf: its collection episode + the 3 repro arms), --short sets the scored seconds, --max-labels caps
GT labels, eval smoke without a frozen maps artifact uses SMOKE maps (K-A's gt_ext map; never a result).
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
import e6p_opta_ka as KA  # noqa: E402
import e6p_opta_kb as KB  # noqa: E402  (unit_record, incumbent_consts)
import e6p_screen as S  # noqa: E402  (sets ROOT / BUNDLE and sys.path)
import e6p_step2_dev as D  # noqa: E402

from cdd_oran.decision import collect_p as CP  # noqa: E402
from cdd_oran.decision import features as F  # noqa: E402
from cdd_oran.decision import gt_p as G  # noqa: E402
from cdd_oran.decision import labels_p as LP  # noqa: E402
from cdd_oran.decision import mapgate as MG  # noqa: E402
from cdd_oran.decision import units_p as UP  # noqa: E402
from cdd_oran.envs.e6.env import E6Env  # noqa: E402

PROTOCOL_DOC = "docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md"
MAPS_DOC = "docs/benchmark/artifacts/E6P_CONF_MAPS.json"
REGISTRY_DOC = "docs/benchmark/SEED_REGISTRY.json"
FROZEN_SHA256_CONF = "f719f43085cf1f95cf7c487e0e2c015e4377f9d83a9d4daa5484f42a996c7a1b"  # freeze 1 (2026-10-01): the protocol doc's LF-normalised sha256
MAPS_SHA256 = None                      # freeze 2: the maps artifact's LF-normalised sha256 (after the disc analysis)
SCHEMA_COLL = "e6p-disc-rec/1"
SCHEMA_EVAL = "e6p-optaka2-rec/1"
SUB = "conf"
PAIR, STRATUM = "P3", 3
SEED_BLOCK = (186000, 187999)           # registered E6 "e6p_confounded_episodes"
KB_BLOCK = (186000, 186079)             # K-B (done): never re-used
LAYOUT = {"dev_conf": (186080, 20), "disc": (186100, 600), "placebo": (186700, 200), "gt": (186900, 40),
          "eval": (187000, 160)}
RESERVES = ((186940, 186999), (187160, 187999))
FORBIDDEN = ((150000, 150399), (155000, 155399), (160000, 179999), (180000, 183999), (184000, 185999),
             (188000, 189999), (190000, 190399), KB_BLOCK) + RESERVES
STAGES = ("dev_conf", "disc", "placebo", "gt", "eval")
FROZEN_STAGES = ("disc", "placebo", "gt", "eval")
RECORD_STAGE = {"dev_conf": "dev", "disc": "eval", "placebo": "placebo", "gt": "gt"}
REPRO_SEEDS = (186080, 186081)
REPRO_ARMS = ("incumbent", "MG:allaccept", "noarb")
GT_T0_MIN = 90
OPEN_RULE = "feasible"
ARB_WARMUP_S = 0.0
TAP_COUNT_ALL = True
T = UP.T_UNIT
STEP_S = 10
THETA, K_CONF = MG.THETA, MG.K_CONF
RAND_TAG, RAND_KEY = 6623, 3
TAGS = {6622: "incumbent logging draws", 6623: "random map default_rng([6623, 3])",
        6624: "study bootstraps: DISC [6624, 10, ...], EVAL [6624, 20, n_seeds]"}

ANCHORS = D.REF_ARMS                    # freeze, sub:ES+PowerES, noarb, sub:ES, sub:PowerES, sub:SliceGuarantee
REFS = ("incumbent", "never_sleep", "B2")
ASSOC = ("corr@dev", "corr@plc", "granger@dev", "granger@plc", "shap_gbdt@dev", "shap_gbdt@plc", "int@dev",
         "int@plc", "qacm@dev", "qacm@plc", "two_tower@dev", "two_tower@plc", "granger_by")
ASSOC_ARMS = tuple(f"MG:{b}" for b in ASSOC)                           # the associational map set A (13)
MAP_ARMS = ("MG:PMRT", "MG:GT", "MG:rand", "blanket2") + ASSOC_ARMS  # canonical order (alias targets = first)
ALL_ARMS = ANCHORS + REFS + MAP_ARMS                                   # the 26 named arms of section 5.2

# ---- layout self-checks (import time)
for _s, (_b, _n) in LAYOUT.items():
    assert SEED_BLOCK[0] <= _b and _b + _n - 1 <= SEED_BLOCK[1], _s
    for _lo, _hi in FORBIDDEN:
        assert _b + _n - 1 < _lo or _b > _hi, f"stage {_s} overlaps [{_lo}, {_hi}]"
_rng = sorted((b, b + n - 1) for b, n in LAYOUT.values())
assert all(a[1] < b[0] for a, b in zip(_rng, _rng[1:], strict=False)), _rng
assert len(ALL_ARMS) == 26 and len(ASSOC_ARMS) == 13


# ---------------------------------------------------------------------------------------------- seeds / status
def stage_seeds(stage: str) -> list:
    b, n = LAYOUT[stage]
    return list(range(b, b + n))


def check_seed(seed: int, stage: str) -> int:
    """Asserts the seed is in ``stage``'s range of the section 6 layout and outside every forbidden block."""
    seed = int(seed)
    if stage not in LAYOUT:
        raise SystemExit(f"stage in {STAGES}")
    b, n = LAYOUT[stage]
    assert b <= seed < b + n, f"seed {seed} outside the {stage} range [{b}, {b + n - 1}]"
    assert SEED_BLOCK[0] <= seed <= SEED_BLOCK[1]
    for lo, hi in FORBIDDEN:
        assert not lo <= seed <= hi, f"seed {seed} inside a forbidden block [{lo}, {hi}]"
    return seed


def sha_lf(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def freeze_status(root: str | None = None) -> dict:
    """Protocol doc: FROZEN line, LF sha256 vs FROZEN_SHA256_CONF. frozen = line "FROZEN: yes" and sha match."""
    path = os.path.join(root or S.ROOT, PROTOCOL_DOC)
    if not os.path.exists(path):
        return {"doc": PROTOCOL_DOC, "exists": False, "sha256": None, "want": FROZEN_SHA256_CONF, "frozen": False}
    txt = open(path, "rb").read().decode("utf-8", "replace")
    line = next((ln.strip() for ln in txt.splitlines() if ln.strip().startswith("FROZEN:")), "FROZEN: ?")
    sha = sha_lf(path)
    return {"doc": PROTOCOL_DOC, "exists": True, "line": line, "sha256": sha, "want": FROZEN_SHA256_CONF,
            "frozen": bool(FROZEN_SHA256_CONF is not None and sha == FROZEN_SHA256_CONF
                           and line.lower().startswith("frozen: yes"))}


def maps_status(path: str | None = None, root: str | None = None) -> dict:
    """Maps artifact: LF sha256 vs MAPS_SHA256 (freeze 2)."""
    path = path or os.path.join(root or S.ROOT, MAPS_DOC)
    if not os.path.exists(path):
        return {"path": path, "exists": False, "sha256": None, "want": MAPS_SHA256, "ok": False}
    sha = sha_lf(path)
    return {"path": path, "exists": True, "sha256": sha, "want": MAPS_SHA256,
            "ok": bool(MAPS_SHA256 is not None and sha == MAPS_SHA256)}


def registry_check(root: str | None = None) -> dict | None:
    """{block, tags, layout} in SEED_REGISTRY.json (None if the file is absent). layout = the confounded note names
    every stage range of section 6."""
    path = os.path.join(root or S.ROOT, REGISTRY_DOC)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    e6 = d.get("E6", {}) or {}
    blk = e6.get("e6p_confounded_episodes", [])
    note = str(e6.get("e6p_confounded_note", ""))
    return {"block_186000_187999": any(lo <= SEED_BLOCK[0] and hi >= SEED_BLOCK[1] for lo, hi in blk),
            "tags_6622_6624": all(t in d.get("rng_stream_tags", []) for t in TAGS),
            "layout": all(f"{b}-{b + n - 1}" in note for b, n in LAYOUT.values())}


def guard_run(stage: str, smoke: bool, maps_path: str | None = None, root: str | None = None) -> dict:
    """The refusals of protocol sections 6 / 9 (smoke runs are exempt): frozen protocol for disc / placebo / gt /
    eval, frozen maps artifact for eval, registered block / tags / layout for every stage, Linux numerics for the
    frozen stages. Returns the status dict; raises SystemExit on a refusal."""
    fz, reg = freeze_status(root), registry_check(root)
    ms = maps_status(maps_path, root) if stage == "eval" else None
    st = {"freeze": fz, "registry": reg, "maps": ms}
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    if smoke:
        return st
    if stage in FROZEN_STAGES and not fz["frozen"]:
        raise SystemExit(f"stage {stage} needs the frozen protocol {PROTOCOL_DOC} (FROZEN: yes, sha256 == "
                         f"FROZEN_SHA256_CONF): {fz}")
    if stage == "eval" and not ms["ok"]:
        raise SystemExit(f"stage eval needs the frozen maps artifact {MAPS_DOC} with sha256 == MAPS_SHA256: {ms}")
    if reg is None or not all(reg.values()):
        raise SystemExit(f"seed block / RNG tags / layout not registered in {REGISTRY_DOC}: {reg}")
    if stage in FROZEN_STAGES and not sys.platform.startswith("linux"):
        raise SystemExit(f"stage {stage}: Kaggle Linux numerics only (protocol section 3); platform {sys.platform}")
    return st


# ---------------------------------------------------------------------------------------------- maps / aliasing
def alias_table(maps: dict, theta: float = THETA, noarb_alias: bool = False) -> dict:
    """Arms whose maps share a decision signature are simulated once (protocol section 5.2). ``maps`` {arm: M} over
    MAP_ARMS (any subset). Canonical arm of a group = the first in MAP_ARMS order; an all-accept group is aliased to
    "noarb" iff ``noarb_alias``. Returns {"signature": {arm: key}, "alias_of": {arm: canonical | "noarb"},
    "groups": {key: [arms]}, "jobs": [canonical arms to simulate, MAP_ARMS order]}."""
    order = [a for a in MAP_ARMS if a in maps] + sorted(a for a in maps if a not in MAP_ARMS)
    sig = {a: MG.signature_key(MG.decision_signature(maps[a], theta)) for a in order}
    aa = MG.signature_key(MG.all_accept_signature(theta))
    groups, alias = {}, {}
    for a in order:
        groups.setdefault(sig[a], []).append(a)
    for k, arms in groups.items():
        tgt = "noarb" if (k == aa and noarb_alias) else arms[0]
        for a in arms:
            alias[a] = tgt
    jobs = [a for a in order if alias[a] == a]
    return {"signature": sig, "alias_of": alias, "groups": groups, "jobs": jobs, "all_accept": aa,
            "noarb_alias": bool(noarb_alias)}


def smoke_maps() -> dict:
    """Plumbing-only map set for eval smoke runs without a frozen artifact (K-A's gt_ext map and variants)."""
    M = KA.M_GT
    ms = {"MG:PMRT": MG.own_only(M), "MG:GT": M, "MG:rand": MG.random_sized_map(M, KA._cells(), RAND_TAG, RAND_KEY),
          "blanket2": MG.blanket_saving_map()}
    for i, b in enumerate(ASSOC_ARMS):
        ms[b] = MG.sign_flip(M, ("carrier",)) if i % 3 == 0 else ({} if i % 3 == 1 else M)
    return ms


def load_maps(path: str | None = None, smoke: bool = False) -> dict:
    """{"maps": {arm: M}, "alias": alias_table-like dict, "source", "sha256"} from the maps artifact; smoke without
    an artifact -> smoke_maps()."""
    p = path or os.path.join(S.ROOT, MAPS_DOC)
    if os.path.exists(p):
        a = json.load(open(p))
        maps = {arm: MG.map_from_json(v["map"]) for arm, v in a["arms"].items()}
        al = alias_table(maps, float(a["mapgate"]["theta"]), bool(a["noarb_alias"]))
        if al["jobs"] != a["jobs"] or al["alias_of"] != a["alias_of"]:
            raise SystemExit(f"maps artifact {p}: alias table does not reproduce (jobs {a['jobs']} vs {al['jobs']})")
        return {"maps": maps, "alias": al, "source": p, "sha256": sha_lf(p)}
    if not smoke:
        raise SystemExit(f"no maps artifact at {p}")
    maps = smoke_maps()
    return {"maps": maps, "alias": alias_table(maps), "source": "SMOKE maps (e6p_opta_ka.M_GT variants)",
            "sha256": None}


# ---------------------------------------------------------------------------------------------- collection episode
def make_policy(stage: str, sd: int):
    if stage == "placebo":
        return CP.PlaceboIncumbent(sd), "placebo_incumbent"
    if stage in ("dev_conf", "disc", "gt"):
        return CP.IncumbentPolicy(sd), "incumbent"
    raise SystemExit(f"no collection policy for stage {stage}")


def episode_job(stage: str, seed: int, j: int, lf: float, smoke: bool = False, short=None, max_labels=None) -> dict:
    check_seed(seed, stage)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    policy, pol_name = make_policy(stage, sd)
    env = E6Env(cfg, log=False, wg3=True, trace=True)
    lab = LP.Labeller(ks=G.GT_KS, H=G.GT_H, T=T, cells=True) if stage == "gt" else None
    cnt = {"labels": 0, "skipped_end": 0, "skipped_cap": 0, "skipped_early": 0}

    def hook(env_, obs, snap, opened):
        for u in opened:
            rate = G.GT_RATE.get(u["knob"])
            if rate is None or not LP.sampled(sd, u, rate):
                continue
            if u["t0"] < GT_T0_MIN:                     # untested units (t0 < 90) are not labelled (as v4)
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
                            open_rule=OPEN_RULE, arb_warmup_s=ARB_WARMUP_S, count_all=TAP_COUNT_ALL,
                            arbiter_cls=MG.DirectionalUnitArbiter)
    t_p = time.process_time()
    trace = env.get_trace()
    panel = F.build_panel_p(trace, step_s=STEP_S, episode=seed, drop_degenerate=False)
    panel_cpu = time.process_time() - t_p
    units = res["units"]
    import e6p_discovery as DI
    n_app = DI._n_applied(trace, units, T)
    urecs = [KB.unit_record(i, u, n_app[i]) for i, u in enumerate(units)]
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
    by, by_cls = {}, {}
    for u in urecs:
        b = by.setdefault(u["knob"], {})
        b[u["mode"]] = b.get(u["mode"], 0) + 1
        c = by_cls.setdefault(f"{u['inc']['cls']}:{u['inc']['key']}", {})
        c[u["mode"]] = c.get(u["mode"], 0) + 1
    T_ep = cfg.warmup_s + cfg.scored_s
    tested = [u for u in urecs if u["t0"] >= 90 and u["t0"] + 90 <= T_ep]
    sl = [u for u in urecs if u["knob"] == "sleep"]
    slt = [u for u in tested if u["knob"] == "sleep"]
    cnt.update(units=len(urecs), by_family_mode=by, by_class_mode=by_cls, sleep_units=len(sl),
               sleep_rejects=sum(u["mode"] == "reject" for u in sl), tested_units=len(tested),
               tested_sleep_units=len(slt), tested_sleep_rejects=sum(u["mode"] == "reject" for u in slt),
               units_warmup=sum(u["t0"] < cfg.warmup_s for u in urecs),
               missing_probs=sum(u.get("probs") is None for u in urecs))
    ser = res["tap"].series()
    lay, pl = env.plant.lay, env.plant
    es = next((x for x in env.xapps if x.name == "ES"), None)
    st = env.static()
    rs = RECORD_STAGE[stage]
    label_cpu = lab.cpu_s if lab is not None else 0.0
    j_ = int(j)
    return {"kind": "episode", "schema": SCHEMA_COLL, "key": [rs, SUB, seed], "stage": rs, "sub": SUB,
            "conf_stage": stage, "seed": seed, "cfg_seed": sd, "j": j_, "fold": j_ // 60 if stage == "disc" else None,
            "smoke": smoke, "short": short, "policy": pol_name, "pi0_table": None,
            "incumbent": KB.incumbent_consts(), "directional": True, "arbiter": "DirectionalUnitArbiter",
            "passed": int(res["arb"].passed), "arb_warmup_s": ARB_WARMUP_S, "tap_count_all": TAP_COUNT_ALL,
            "T": T, "open_rule": OPEN_RULE, "load_factor": lf, "episode_s": env.total_s, "warmup_s": cfg.warmup_s,
            "n_cells": int(pl.nc), "step_s": STEP_S,
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
            "secs": round(time.time() - t_wall, 1), "rss_mb": D.peak_rss_mb()}


# ---------------------------------------------------------------------------------------------- eval / repro arms
def make_arbiter(arm: str, sd: int, warmup_s: float, maps: dict | None):
    """Arbiter of an EVAL / repro arm (None = accept-all)."""
    if arm in ANCHORS:
        return D.make_arbiter(arm, warmup_s)
    if arm == "incumbent":
        return KA.Counting(MG.DirectionalUnitArbiter(CP.IncumbentPolicy(sd), T=T, warmup_s=0.0, record=False,
                                                     open_rule=OPEN_RULE), is_unit=True)
    if arm == "never_sleep":
        return KA.make_arbiter("never_sleep")
    if arm == "B2":
        return KA.Counting(D.make_arbiter("B2", warmup_s), is_unit=False)
    if arm == "MG:allaccept":
        return KA.Counting(MG.mapgate_v2_arbiter({}, THETA, K_CONF), is_unit=True)
    if maps is not None and arm in maps:
        return KA.Counting(MG.mapgate_v2_arbiter(maps[arm], THETA, K_CONF), is_unit=True)
    raise KeyError(arm)


def arm_job(seed: int, arm: str, lf: float, conf_stage: str, maps: dict | None = None, alias: dict | None = None,
            smoke: bool = False, short=None) -> dict:
    check_seed(seed, "dev_conf" if conf_stage == "dev_repro" else "eval")
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg("*" if arm == "freeze" else PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm, sd, cfg.warmup_s, maps)
    t_wall, t_cpu = time.time(), time.process_time()
    env, _ = S.run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    pc = {}
    if isinstance(arb, KA.Counting):
        pc = dict(arb.n)
        pol = getattr(arb.inner, "policy", None)
        pc["policy"] = dict(getattr(pol, "n", None) or getattr(arb.inner, "n", None) or {})
        if hasattr(arb.inner, "passed"):
            pc["passed"] = {"all": int(arb.inner.passed)}
    elif arb is not None:
        pc = {"policy": dict(getattr(arb, "n", {}) or {})}
    sig, aliases = None, []
    if alias is not None and arm in alias.get("signature", {}):
        sig = alias["signature"][arm]
        aliases = [a for a, t in alias["alias_of"].items() if t == arm and a != arm]
    return {"kind": "job", "schema": SCHEMA_EVAL, "key": [seed, arm], "seed": seed, "cfg_seed": sd, "arm": arm,
            "sub": SUB, "conf_stage": conf_stage, "smoke": smoke, "short": short, "episode_s": env.total_s,
            "warmup_s": cfg.warmup_s, "load_factor": lf, **S.outcome(env), "policy_counts": pc, "signature": sig,
            "aliases": aliases, "cpu_s": round(cpu, 2), "secs": round(time.time() - t_wall, 1),
            "rss_mb": D.peak_rss_mb()}


# ---------------------------------------------------------------------------------------------- jobs / run
def eval_arms(alias: dict) -> list:
    """Distinct EVAL arms: anchors + references + the artifact's map jobs (aliased arms excluded)."""
    return list(ANCHORS) + list(REFS) + [a for a in alias["jobs"] if a not in ANCHORS]


def jobs(stage: str, alias: dict | None = None) -> list:
    """[(kind, seed, j_or_arm)] in shard order (seed-major). kind "coll" (collection episode) or "arm"."""
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    seeds = stage_seeds(stage)
    if stage == "eval":
        if alias is None:
            raise ValueError("eval jobs need the alias table")
        return [("arm", check_seed(sd, stage), a) for sd in seeds for a in eval_arms(alias)]
    J = []
    for j, sd in enumerate(seeds):
        J.append(("coll", check_seed(sd, stage), j))
        if stage == "dev_conf" and sd in REPRO_SEEDS:
            J += [("arm", sd, a) for a in REPRO_ARMS]
    return J


def job_key(stage: str, kind: str, seed: int, x) -> tuple:
    return (RECORD_STAGE[stage], SUB, seed) if kind == "coll" else (seed, x)


def run(stage, part, out, smoke=False, short=None, max_labels=None, arms=None, maps_path=None):
    if (short or max_labels) and not smoke:
        raise SystemExit("--short / --max-labels are smoke-only")
    st = guard_run(stage, smoke, maps_path)
    M = load_maps(maps_path, smoke) if stage == "eval" else None
    alias = M["alias"] if M else None
    state = S.load_state()
    lf = S.lf_of(state, STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header(f"conf:{stage}", part, smoke, state)
    try:
        import cloud
        nenv = cloud.numeric_env()
    except Exception as e:                                              # noqa: BLE001
        nenv = {"error": repr(e)}
    head.update(kind="header", schema=SCHEMA_EVAL if stage == "eval" else SCHEMA_COLL, driver="e6p_conf",
                protocol=PROTOCOL_DOC, conf_stage=stage, freeze=st["freeze"], registry=st["registry"],
                maps=st["maps"], platform=nenv,
                maps_source={"source": M["source"], "sha256": M["sha256"], "jobs": alias["jobs"],
                             "alias_of": alias["alias_of"]} if M else None,
                mapgate_sha256=sha_lf(os.path.join(S.ROOT, "cdd_oran", "decision", "mapgate.py")),
                consts={"pair": PAIR, "stratum": STRATUM, "load_factor": lf, "T": T, "open_rule": OPEN_RULE,
                        "step_s": STEP_S, "arb_warmup_s": ARB_WARMUP_S, "tap_count_all": TAP_COUNT_ALL,
                        "directional": True, "incumbent": KB.incumbent_consts(), "layout": LAYOUT,
                        "repro_seeds": REPRO_SEEDS, "repro_arms": REPRO_ARMS, "theta": THETA, "k_conf": K_CONF,
                        "gt": {"ks": G.GT_KS, "H": G.GT_H, "modes": G.GT_MODES, "rate": G.GT_RATE,
                               "t0_min": GT_T0_MIN, "kpis": LP.CELL_KPIS}, "tags": TAGS,
                        "series_fields": CP.SERIES_FIELDS,
                        "smoke": {"short": short, "max_labels": max_labels} if smoke else None, "arms_filter": arms})
    S._append(out, head)
    done = set()
    for r in S._read(out):
        if r.get("kind") == "episode" and r.get("smoke") == smoke:
            done.add(tuple(r["key"]))
        elif r.get("kind") == "job" and r.get("smoke") == smoke:
            done.add(tuple(r["key"]))
    J = jobs(stage, alias)
    t0, n, first = time.time(), 0, None
    for u, (kind, seed, x) in enumerate(J):
        if smoke:
            first = seed if first is None else first
            if seed != first:
                break
        elif u % k != i:
            continue
        key = job_key(stage, kind, seed, x)
        if key in done or (arms and kind == "arm" and x not in arms):
            continue
        if kind == "coll":
            rec = episode_job(stage, seed, x, lf, smoke, short, max_labels)
            show = ("conf_stage", "seed", "cfg_seed", "counts", "passed", "cpu_s", "label_cpu_s", "secs", "rss_mb")
        else:
            rec = arm_job(seed, x, lf, "dev_repro" if stage == "dev_conf" else "eval", M["maps"] if M else None,
                          alias, smoke, short)
            show = ("conf_stage", "seed", "cfg_seed", "arm", "psvr", "energy_j", "policy_counts", "cpu_s", "secs",
                    "rss_mb")
        S._append(out, rec)
        n += 1
        print(json.dumps({x_: rec.get(x_) for x_ in show}, default=S._js), flush=True)
    S._append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_total": len(J),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary / repro
def repro_check(coll: dict, arms: dict) -> dict:
    """Protocol section 9.1 repro checks on REPRO_SEEDS. ``coll`` {seed: dev_conf collection record}; ``arms`` {seed:
    {arm: dev_repro job record}}. incumbent: its outcome vs the logging trajectory's lab_outcome (max abs diff over
    SUM_FIELDS); all-accept MapGateV2 vs noarb (bit identity decides the noarb alias)."""
    out = {"incumbent_vs_logging": {}, "allaccept_vs_noarb": {}}
    for s in REPRO_SEEDS:
        a = arms.get(s, {})
        if s in coll and "incumbent" in a:
            lo = coll[s]["lab_outcome"]
            out["incumbent_vs_logging"][s] = max(abs(float(lo[f]) - float(a["incumbent"][f])) for f in D.SUM_FIELDS)
        if "MG:allaccept" in a and "noarb" in a:
            out["allaccept_vs_noarb"][s] = max(abs(float(a["MG:allaccept"][f]) - float(a["noarb"][f]))
                                               for f in D.SUM_FIELDS)
    for k in ("incumbent_vs_logging", "allaccept_vs_noarb"):
        v = out[k]
        out[k] = {"per_seed": v, "ok": (len(v) == len(REPRO_SEEDS) and all(x == 0 for x in v.values())) if v else None}
    out["noarb_alias"] = bool(out["allaccept_vs_noarb"]["ok"])
    return out


def summary(paths, json_out=None, allow_smoke=False):
    recs, jobs_ = [], []
    for p in paths:
        for r in S._read(p):
            if r.get("smoke") and not allow_smoke:
                continue
            if r.get("kind") == "episode" and r.get("schema") == SCHEMA_COLL and r.get("sub") == SUB:
                r.pop("panel", None)
                r.pop("lab_series", None)
                recs.append(r)
            elif r.get("kind") == "job" and r.get("schema") == SCHEMA_EVAL and r.get("sub") == SUB:
                jobs_.append(r)
    recs = list({tuple(r["key"]) + (r.get("smoke"),): r for r in recs}.values())
    report = {"protocol": PROTOCOL_DOC, "freeze": freeze_status(), "registry": registry_check(), "stages": {}}
    for st in STAGES[:4]:
        rs = [r for r in recs if r.get("conf_stage") == st]
        if not rs:
            continue
        fam = {}
        for r in rs:
            for f, modes in r["counts"]["by_family_mode"].items():
                d = fam.setdefault(f, {})
                for m, v in modes.items():
                    d[m] = d.get(m, 0) + v
        ne = len(rs)
        ts = [r["counts"]["tested_sleep_units"] for r in rs]
        tr = [r["counts"]["tested_sleep_rejects"] for r in rs]
        rep = {"episodes": ne, "units_by_family_mode": fam,
               "units_per_episode": {f: sum(d.values()) / ne for f, d in fam.items()},
               "tested_sleep_units_per_episode": float(np.mean(ts)), "tested_sleep_rejects_per_episode":
                   float(np.mean(tr)), "K1_projection_disc600": {"sleep_units": 600 * float(np.mean(ts)),
                                                                 "sleep_rejects": 600 * float(np.mean(tr)),
                                                                 "rule": ">= 500 units, >= 75 rejects"},
               "missing_probs": int(sum(r["counts"]["missing_probs"] for r in rs)),
               "passed_per_episode": float(np.mean([r.get("passed", 0) for r in rs])),
               "labels": int(sum(r["counts"]["labels"] for r in rs)),
               "cpu_s_per_episode": float(np.mean([r["cpu_s"] for r in rs])),
               "cpu_h_total": float(sum(r["cpu_s"] for r in rs) / 3600)}
        report["stages"][st] = rep
        print(f"== {st}: {ne} episodes, {rep['cpu_h_total']:.2f} CPU-h ({rep['cpu_s_per_episode']:.0f} s/ep); units/ep "
              f"{ {f: round(v, 1) for f, v in rep['units_per_episode'].items()} }; tested sleep units/ep "
              f"{rep['tested_sleep_units_per_episode']:.2f} (rejects {rep['tested_sleep_rejects_per_episode']:.2f}); "
              f"K1 projection {rep['K1_projection_disc600']}; missing probs {rep['missing_probs']}")
    coll = {int(r["seed"]): r for r in recs if r.get("conf_stage") == "dev_conf"}
    arms = {}
    for r in jobs_:
        if r.get("conf_stage") == "dev_repro":
            arms.setdefault(int(r["seed"]), {})[r["arm"]] = r
    if arms:
        report["repro"] = repro_check(coll, arms)
        print("   repro:", json.dumps(report["repro"], default=S._js))
    ev = [r for r in jobs_ if r.get("conf_stage") == "eval"]
    if ev:
        byarm = {}
        for r in ev:
            byarm.setdefault(r["arm"], []).append(r["cpu_s"])
        report["eval"] = {"jobs": len(ev), "seeds": len({r["seed"] for r in ev}),
                          "cpu_h_total": float(sum(r["cpu_s"] for r in ev) / 3600),
                          "cpu_s_per_arm": {a: float(np.mean(v)) for a, v in byarm.items()}}
        print(f"== eval: {report['eval']['jobs']} jobs on {report['eval']['seeds']} seeds, "
              f"{report['eval']['cpu_h_total']:.2f} CPU-h")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs(maps_path=None):
    print("protocol:", freeze_status())
    print("maps:", maps_status(maps_path))
    print("registry:", registry_check())
    for st in STAGES:
        if st == "eval":
            M = load_maps(maps_path, smoke=not maps_status(maps_path)["exists"])
            J = jobs(st, M["alias"])
            print(f"eval: {len(J)} jobs = {len(stage_seeds(st))} seeds x {len(eval_arms(M['alias']))} distinct arms "
                  f"(maps: {M['source']}); aliases {M['alias']['alias_of']}")
        else:
            J = jobs(st)
            print(f"{st}: {len(J)} jobs; seeds {J[0][1]}..{J[-1][1]}"
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
            max_labels=int(a["--max-labels"]) if a.get("--max-labels") else None,
            arms=[x for x in a["--arms"].split(",") if x] if a.get("--arms") else None, maps_path=a.get("--maps"))
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
        list_jobs(a.get("--maps"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
