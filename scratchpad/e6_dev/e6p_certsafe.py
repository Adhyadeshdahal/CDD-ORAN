"""E6-P follow-up study 2b DRIVER: the certified-safe referee (protocol docs/benchmark/E6P_CERTSAFE_PROTOCOL.md;
analysis scratchpad/e6_dev/e6p_certsafe_analyze.py; gate cdd_oran/decision/certsafe.py).

Plant, referee constants and arm runner are those of option (a) (scratchpad/e6_dev/e6p_conf.py, frozen protocol
docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md): e6p_screen.make_cfg("P3", 3, seed, lf), lf = e6p_state L40, 120 s warm-up
+ 600 s scored; MapGateV2 theta .05, k_conf 1, T 60 s, open_rule "feasible", DirectionalUnitArbiter from t = 0. The
maps are the frozen option (a) maps artifact docs/benchmark/artifacts/E6P_CONF_MAPS.json (its LF sha256 must equal
e6p_conf.MAPS_SHA256); nothing is re-discovered.

Stages (seed block 191000-191999, registered E6 "e6p_certsafe_episodes"):
  dev   191000-191039 (40)   calibration: "CAL:allaccept" = DirectionalUnitArbiter(CertSafeMapGateV2({})) on every
                             seed (decides as accept-all; counts units per (family, direction)) -> G_k and N(f, d)
                             of tau; repro "noarb" on REPRO_SEEDS 191000-191001 (bit identity with CAL:allaccept
                             decides the noarb alias). Needs the FROZEN protocol (FROZEN_SHA256_CS).
  eval  191100-191259 (160)  anchors (e6p_step2_dev), incumbent, never_sleep, MG:PMRT (uncertified, descriptive
                             replication of option (a)), and the 17 certified-safe arms CS:<x> (ONE job per distinct
                             certsafe signature, the artifact's "jobs"). Needs FROZEN_SHA256_CS AND CERTSAFE_SHA256 =
                             the LF sha256 of docs/benchmark/artifacts/E6P_CERTSAFE.json (freeze 2).
Reserves 191040-191099 and 191260-191999 are unused. Records: schema "e6p-optaka2-rec/1", sub "certsafe",
cs_stage dev / eval.

CLI (repo root, PYTHONPATH=.; cloud wrappers e6p_certsafe_{dev,eval}.py):
  python scratchpad/e6_dev/e6p_certsafe.py run --stage dev|eval --part i/k --out F.jsonl [--smoke] [--short S]
         [--arms a,b] [--artifact FILE]
  python scratchpad/e6_dev/e6p_certsafe.py list [--artifact FILE]
--smoke: plumbing only (exempt from every guard): seed -> seed % 31, only the jobs of the FIRST seed, --short sets
the scored seconds; eval smoke without a frozen artifact uses ``smoke_artifact()`` (provisional tau and bounds,
never a result).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_conf as C  # noqa: E402  (option (a) driver: anchors, maps, arm runner pieces; unchanged)
import e6p_opta_ka as KA  # noqa: E402
import e6p_screen as S  # noqa: E402
import e6p_step2_dev as D  # noqa: E402

from cdd_oran.decision import certsafe as CS  # noqa: E402
from cdd_oran.decision import collect_p as CP  # noqa: E402
from cdd_oran.decision import mapgate as MG  # noqa: E402

PROTOCOL_DOC = "docs/benchmark/E6P_CERTSAFE_PROTOCOL.md"
ARTIFACT_DOC = "docs/benchmark/artifacts/E6P_CERTSAFE.json"
MAPS_DOC = C.MAPS_DOC
REGISTRY_DOC = C.REGISTRY_DOC
FROZEN_SHA256_CS = None       # freeze 1: the protocol doc's LF-normalised sha256 (before any dev / eval episode)
CERTSAFE_SHA256 = None        # freeze 2: the certsafe artifact's LF-normalised sha256 (before any eval episode)
SCHEMA_EVAL = C.SCHEMA_EVAL
SUB = "certsafe"
SEED_BLOCK = (191000, 191999)           # registered E6 "e6p_certsafe_episodes"
LAYOUT = {"dev": (191000, 40), "eval": (191100, 160)}
RESERVES = ((191040, 191099), (191260, 191999))
FORBIDDEN = C.FORBIDDEN + ((186000, 187999), (960000, 10 ** 9)) + RESERVES   # every other block, E6 TEST, reserves
STAGES = ("dev", "eval")
REPRO_SEEDS = (191000, 191001)
CAL_ARM = "CAL:allaccept"
THETA, K_CONF, T, OPEN_RULE = C.THETA, C.K_CONF, C.T, C.OPEN_RULE
ANCHORS = C.ANCHORS
REFS = ("incumbent", "never_sleep", "MG:PMRT")
CS_OF = {"CS:PMRT": "MG:PMRT", "CS:GT": "MG:GT", "CS:rand": "MG:rand", "CS:blanket2": "blanket2",
         **{f"CS:{b}": f"MG:{b}" for b in C.ASSOC}}
CS_ASSOC = tuple(f"CS:{b}" for b in C.ASSOC)                            # the associational set A under the gate
CS_ARMS = tuple(CS_OF)                                                  # canonical order (alias targets = first)
BOUND_METHOD = {"CS:PMRT": "pmrt", "CS:GT": "gt", "CS:rand": None, "CS:blanket2": None,
                **{a: "ols" for a in CS_ASSOC}}
ALL_ARMS = ANCHORS + REFS + CS_ARMS
PRIMARY_ARM = "CS:PMRT"

for _s, (_b, _n) in LAYOUT.items():
    assert SEED_BLOCK[0] <= _b and _b + _n - 1 <= SEED_BLOCK[1], _s
    for _lo, _hi in FORBIDDEN:
        assert _b + _n - 1 < _lo or _b > _hi, f"stage {_s} overlaps [{_lo}, {_hi}]"
assert LAYOUT["dev"][0] + LAYOUT["dev"][1] <= LAYOUT["eval"][0]
assert len(CS_ARMS) == 17 and len(CS_ASSOC) == 13 and len(ALL_ARMS) == 26 and CAL_ARM not in ALL_ARMS


# ---------------------------------------------------------------------------------------------- seeds / status
def stage_seeds(stage: str) -> list:
    b, n = LAYOUT[stage]
    return list(range(b, b + n))


def check_seed(seed: int, stage: str) -> int:
    seed = int(seed)
    if stage not in LAYOUT:
        raise SystemExit(f"stage in {STAGES}")
    b, n = LAYOUT[stage]
    assert b <= seed < b + n, f"seed {seed} outside the {stage} range [{b}, {b + n - 1}]"
    assert SEED_BLOCK[0] <= seed <= SEED_BLOCK[1]
    for lo, hi in FORBIDDEN:
        assert not lo <= seed <= hi, f"seed {seed} inside a forbidden block [{lo}, {hi}]"
    return seed


sha_lf = C.sha_lf


def freeze_status(root: str | None = None) -> dict:
    path = os.path.join(root or S.ROOT, PROTOCOL_DOC)
    if not os.path.exists(path):
        return {"doc": PROTOCOL_DOC, "exists": False, "sha256": None, "want": FROZEN_SHA256_CS, "frozen": False}
    txt = open(path, "rb").read().decode("utf-8", "replace")
    line = next((ln.strip() for ln in txt.splitlines() if ln.strip().startswith("FROZEN:")), "FROZEN: ?")
    sha = sha_lf(path)
    return {"doc": PROTOCOL_DOC, "exists": True, "line": line, "sha256": sha, "want": FROZEN_SHA256_CS,
            "frozen": bool(FROZEN_SHA256_CS is not None and sha == FROZEN_SHA256_CS
                           and line.lower().startswith("frozen: yes"))}


def artifact_status(path: str | None = None, root: str | None = None) -> dict:
    path = path or os.path.join(root or S.ROOT, ARTIFACT_DOC)
    if not os.path.exists(path):
        return {"path": path, "exists": False, "sha256": None, "want": CERTSAFE_SHA256, "ok": False}
    sha = sha_lf(path)
    return {"path": path, "exists": True, "sha256": sha, "want": CERTSAFE_SHA256,
            "ok": bool(CERTSAFE_SHA256 is not None and sha == CERTSAFE_SHA256)}


def registry_check(root: str | None = None) -> dict | None:
    path = os.path.join(root or S.ROOT, REGISTRY_DOC)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    e6 = d.get("E6", {}) or {}
    blk = e6.get("e6p_certsafe_episodes", [])
    note = str(e6.get("e6p_certsafe_note", ""))
    return {"block_191000_191999": any(lo <= SEED_BLOCK[0] and hi >= SEED_BLOCK[1] for lo, hi in blk),
            "tag_6624": 6624 in d.get("rng_stream_tags", []),
            "layout": all(f"{b}-{b + n - 1}" in note for b, n in LAYOUT.values())}


def guard_run(stage: str, smoke: bool, artifact_path: str | None = None, root: str | None = None,
              maps_path: str | None = None) -> dict:
    """Refusals (smoke runs are exempt): frozen protocol for dev and eval, the option (a) maps artifact unchanged
    (MAPS_SHA256), the certsafe artifact (CERTSAFE_SHA256) for eval, registered block / layout, Linux numerics."""
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    fz, reg = freeze_status(root), registry_check(root)
    ms = C.maps_status(maps_path, root)
    ar = artifact_status(artifact_path, root) if stage == "eval" else None
    st = {"freeze": fz, "registry": reg, "maps": ms, "artifact": ar}
    if smoke:
        return st
    if not fz["frozen"]:
        raise SystemExit(f"stage {stage} needs the frozen protocol {PROTOCOL_DOC} (FROZEN: yes, sha256 == "
                         f"FROZEN_SHA256_CS): {fz}")
    if not ms["ok"]:
        raise SystemExit(f"the option (a) maps artifact {MAPS_DOC} must match e6p_conf.MAPS_SHA256: {ms}")
    if stage == "eval" and not ar["ok"]:
        raise SystemExit(f"stage eval needs the frozen certsafe artifact {ARTIFACT_DOC} with sha256 == "
                         f"CERTSAFE_SHA256: {ar}")
    if reg is None or not all(reg.values()):
        raise SystemExit(f"seed block / tag / layout not registered in {REGISTRY_DOC}: {reg}")
    if not sys.platform.startswith("linux"):
        raise SystemExit(f"stage {stage}: Kaggle Linux numerics only; platform {sys.platform}")
    return st


# ---------------------------------------------------------------------------------------------- arms / aliasing
def cs_tables(maps: dict, bounds: dict, tau: dict, theta: float = THETA) -> dict:
    """{cs_arm: {"source", "map", "bound_method", "cert", "uncertified", "signature", "decision_table"}} for every
    CS arm whose source map is in ``maps`` ({source arm: M}); ``bounds`` {method: {(f, rel, k): bound} | None}."""
    out = {}
    for arm in CS_ARMS:
        src = CS_OF[arm]
        if src not in maps:
            continue
        M = maps[src]
        meth = BOUND_METHOD[arm]
        cert = CS.certify(M, bounds.get(meth) if meth else None, tau, theta)
        unc = [tuple(x) for x in cert["uncertified"]]
        dt = {}
        for (f, d), s in MG.decision_table_v2(M, theta).items():
            dt[f"{f}{'+' if d > 0 else '-'}"] = ("accept (uncertified: " + s + ")") if (f, d) in unc else s
        out[arm] = {"source": src, "map": M, "bound_method": meth, "cert": cert, "uncertified": unc,
                    "signature": MG.signature_key(CS.certsafe_signature(M, unc, theta)), "decision_table": dt}
    return out


def cs_alias_table(tables: dict, theta: float = THETA, noarb_alias: bool = False) -> dict:
    """As e6p_conf.alias_table over CS arms with the certsafe signature (equal signatures <=> identical policies)."""
    order = [a for a in CS_ARMS if a in tables]
    sig = {a: tables[a]["signature"] for a in order}
    aa = MG.signature_key(MG.all_accept_signature(theta))
    groups, alias = {}, {}
    for a in order:
        groups.setdefault(sig[a], []).append(a)
    for k, arms in groups.items():
        tgt = "noarb" if (k == aa and noarb_alias) else arms[0]
        for a in arms:
            alias[a] = tgt
    return {"signature": sig, "alias_of": alias, "groups": groups, "jobs": [a for a in order if alias[a] == a],
            "all_accept": aa, "noarb_alias": bool(noarb_alias)}


def maps_from_artifact(path: str | None = None) -> dict:
    p = path or os.path.join(S.ROOT, MAPS_DOC)
    a = json.load(open(p))
    return {arm: MG.map_from_json(v["map"]) for arm, v in a["arms"].items()}


def smoke_artifact(maps_path: str | None = None) -> dict:
    """Plumbing-only artifact (NEVER a result): the option (a) maps (or e6p_conf.smoke_maps without the artifact),
    provisional tau from fixed G / N, PMRT bounds = zero-width at 0 (every undeclared edge certified), GT / OLS
    bounds absent (every undeclared guard edge unresolved)."""
    p = maps_path or os.path.join(S.ROOT, MAPS_DOC)
    maps = maps_from_artifact(p) if os.path.exists(p) else C.smoke_maps()
    tau = CS.tau_table({"rlf": 40.0, "v": 20000.0}, {(f, d): 10.0 for f in MG.FAMILIES for d in CS.DIRS})
    zero = {(f, r, k): {"kind": "slope", "beta": 0.0, "se": 0.0} for f in MG.FAMILIES for r in MG.RELS_V2
            for k in CS.GUARD_KPIS}
    tables = cs_tables(maps, {"pmrt": zero, "gt": None, "ols": None}, tau)
    return {"tables": tables, "alias": cs_alias_table(tables), "maps": maps, "source": "SMOKE artifact",
            "sha256": None}


def load_artifact(path: str | None = None, smoke: bool = False, maps_path: str | None = None) -> dict:
    """{"tables": {cs_arm: {..., "map", "uncertified"}}, "alias", "maps" (source maps incl. MG:PMRT), "source",
    "sha256"} from the certsafe artifact; refuses one whose signatures / alias table do not reproduce."""
    p = path or os.path.join(S.ROOT, ARTIFACT_DOC)
    if os.path.exists(p):
        a = json.load(open(p))
        tables = {}
        for arm, v in a["arms"].items():
            M = MG.map_from_json(v["map"])
            unc = [tuple(x) for x in v["uncertified"]]
            sig = MG.signature_key(CS.certsafe_signature(M, unc, float(a["mapgate"]["theta"])))
            if sig != v["signature"]:
                raise SystemExit(f"certsafe artifact {p}: signature of {arm} does not reproduce")
            tables[arm] = {"source": v["source"], "map": M, "uncertified": unc, "signature": sig}
        al = cs_alias_table(tables, float(a["mapgate"]["theta"]), bool(a["noarb_alias"]))
        if al["jobs"] != a["jobs"] or al["alias_of"] != a["alias_of"]:
            raise SystemExit(f"certsafe artifact {p}: alias table does not reproduce ({a['jobs']} vs {al['jobs']})")
        maps = {v["source"]: tables[k]["map"] for k, v in a["arms"].items()}
        return {"tables": tables, "alias": al, "maps": maps, "source": p, "sha256": sha_lf(p)}
    if not smoke:
        raise SystemExit(f"no certsafe artifact at {p}")
    return smoke_artifact(maps_path)


# ---------------------------------------------------------------------------------------------- arm jobs
def make_arbiter(arm: str, sd: int, warmup_s: float, art: dict | None):
    if arm in ANCHORS:
        return D.make_arbiter(arm, warmup_s)
    if arm == "incumbent":
        return KA.Counting(MG.DirectionalUnitArbiter(CP.IncumbentPolicy(sd), T=T, warmup_s=0.0, record=False,
                                                     open_rule=OPEN_RULE), is_unit=True)
    if arm == "never_sleep":
        return KA.make_arbiter("never_sleep")
    if arm == CAL_ARM:
        return KA.Counting(CS.certsafe_arbiter({}, (), THETA, K_CONF, T, OPEN_RULE), is_unit=True)
    if arm == "MG:PMRT" and art is not None:
        return KA.Counting(MG.mapgate_v2_arbiter(art["maps"]["MG:PMRT"], THETA, K_CONF), is_unit=True)
    if art is not None and arm in art["tables"]:
        t = art["tables"][arm]
        return KA.Counting(CS.certsafe_arbiter(t["map"], t["uncertified"], THETA, K_CONF, T, OPEN_RULE),
                           is_unit=True)
    raise KeyError(arm)


def arm_job(seed: int, arm: str, lf: float, stage: str, art: dict | None = None, smoke: bool = False,
            short=None) -> dict:
    check_seed(seed, stage)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg("*" if arm == "freeze" else C.PAIR, C.STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm, sd, cfg.warmup_s, art)
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
    if art is not None and arm in art["alias"]["signature"]:
        sig = art["alias"]["signature"][arm]
        aliases = [a for a, t_ in art["alias"]["alias_of"].items() if t_ == arm and a != arm]
    return {"kind": "job", "schema": SCHEMA_EVAL, "key": [seed, arm], "seed": seed, "cfg_seed": sd, "arm": arm,
            "sub": SUB, "cs_stage": stage, "smoke": smoke, "short": short, "episode_s": env.total_s,
            "warmup_s": cfg.warmup_s, "load_factor": lf, **S.outcome(env), "policy_counts": pc, "signature": sig,
            "aliases": aliases, "cpu_s": round(cpu, 2), "secs": round(time.time() - t_wall, 1),
            "rss_mb": D.peak_rss_mb()}


def eval_arms(alias: dict) -> list:
    return list(ANCHORS) + list(REFS) + [a for a in alias["jobs"] if a not in ANCHORS]


def jobs(stage: str, alias: dict | None = None) -> list:
    """[(seed, arm)] in shard order (seed-major)."""
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    if stage == "dev":
        return [(check_seed(sd, "dev"), a) for sd in stage_seeds("dev")
                for a in ((CAL_ARM, "noarb") if sd in REPRO_SEEDS else (CAL_ARM,))]
    if alias is None:
        raise ValueError("eval jobs need the alias table")
    return [(check_seed(sd, "eval"), a) for sd in stage_seeds("eval") for a in eval_arms(alias)]


def run(stage, part, out, smoke=False, short=None, arms=None, artifact_path=None, maps_path=None):
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    st = guard_run(stage, smoke, artifact_path, maps_path=maps_path)
    art = load_artifact(artifact_path, smoke, maps_path) if stage == "eval" else None
    state = S.load_state()
    lf = S.lf_of(state, C.STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header(f"certsafe:{stage}", part, smoke, state)
    try:
        import cloud
        nenv = cloud.numeric_env()
    except Exception as e:                                              # noqa: BLE001
        nenv = {"error": repr(e)}
    root = S.ROOT
    head.update(kind="header", schema=SCHEMA_EVAL, driver="e6p_certsafe", protocol=PROTOCOL_DOC, cs_stage=stage,
                freeze=st["freeze"], registry=st["registry"], maps=st["maps"], artifact=st["artifact"],
                platform=nenv,
                artifact_source={"source": art["source"], "sha256": art["sha256"], "jobs": art["alias"]["jobs"],
                                 "alias_of": art["alias"]["alias_of"]} if art else None,
                mapgate_sha256=sha_lf(os.path.join(root, "cdd_oran", "decision", "mapgate.py")),
                certsafe_sha256=sha_lf(os.path.join(root, "cdd_oran", "decision", "certsafe.py")),
                consts={"pair": C.PAIR, "stratum": C.STRATUM, "load_factor": lf, "T": T, "open_rule": OPEN_RULE,
                        "theta": THETA, "k_conf": K_CONF, "rho": CS.RHO, "z90": CS.Z90,
                        "guard_kpis": CS.GUARD_KPIS, "layout": LAYOUT, "repro_seeds": REPRO_SEEDS,
                        "smoke": {"short": short} if smoke else None, "arms_filter": arms})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    J = jobs(stage, art["alias"] if art else None)
    t0, n, first = time.time(), 0, None
    for u, (seed, arm) in enumerate(J):
        if smoke:
            first = seed if first is None else first
            if seed != first:
                break
        elif u % k != i:
            continue
        if (seed, arm) in done or (arms and arm not in arms):
            continue
        rec = arm_job(seed, arm, lf, stage, art, smoke, short)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("cs_stage", "seed", "cfg_seed", "arm", "psvr", "rlf",
                                                   "energy_j", "policy_counts", "cpu_s", "rss_mb")},
                         default=S._js), flush=True)
    S._append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_total": len(J),
                    "secs": round(time.time() - t0, 1)})


def driver_sha(root: str | None = None) -> str:
    """LF sha256 of this driver with its CERTSAFE_SHA256 line normalised to None (the artifact pins the driver and
    the driver pins the artifact)."""
    path = os.path.join(root or S.ROOT, "scratchpad", "e6_dev", "e6p_certsafe.py")
    txt = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    txt = re.sub(r"^CERTSAFE_SHA256 = .*$", "CERTSAFE_SHA256 = None", txt, count=1, flags=re.M)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def list_jobs(artifact_path=None):
    print("protocol:", freeze_status())
    print("maps:", C.maps_status())
    print("artifact:", artifact_status(artifact_path))
    print("registry:", registry_check())
    print(f"dev: {len(jobs('dev'))} jobs; seeds {stage_seeds('dev')[0]}..{stage_seeds('dev')[-1]}")
    art = load_artifact(artifact_path, smoke=not artifact_status(artifact_path)["exists"])
    print(f"eval: {len(jobs('eval', art['alias']))} jobs = 160 seeds x {len(eval_arms(art['alias']))} distinct arms "
          f"({art['source']}); aliases {art['alias']['alias_of']}")


def cli(argv, stage=None):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x == "--smoke"}
    rest = [x for x in rest if x not in flags]
    a = dict(zip(rest[::2], rest[1::2], strict=True))
    if cmd == "run":
        run(a.get("--stage") or stage, a["--part"], a["--out"], smoke="--smoke" in flags,
            short=float(a["--short"]) if a.get("--short") else None,
            arms=[x for x in a["--arms"].split(",") if x] if a.get("--arms") else None,
            artifact_path=a.get("--artifact"), maps_path=a.get("--maps"))
    elif cmd == "list":
        list_jobs(a.get("--artifact"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
