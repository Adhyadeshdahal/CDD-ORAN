"""X7 (scratchpad/xmethod/EXTRAS_PROTOCOL.md, Amendments 2026-10-05): are the associational referees the static subset
rule sub:ES+PowerES? POST HOC, DESCRIPTIVE, on the Study 3 (E6-P certsafe) EVAL seeds 191100-191259; imports the frozen
certsafe driver and edits nothing.

Job kinds (record arm names; sub "x7"):
  X7:sub|shadow        sub:ES+PowerES applied; the four referees queried in shadow on a deep copy of every obs (A2 i)
  X7:<ref>|own         <ref> applied (the frozen arm, Counting wrapper as stored); the subset rule in shadow (A2 ii)
  X7:<ref>|pm          <ref>'s decisions with every non-prot_min deferral replaced by accept (A3)
  X7:<ref>|other       <ref>'s decisions with every prot_min deferral replaced by accept (A3)
Per-request tallies: record["agree"][shadow]["<family>|<dir>|<pre|post>"] = [aa, ar, ra, rr] (first letter = applied
decision, second = shadow decision; a = accept, r = reject).

CLI (repo root, PYTHONPATH=.):
  python scratchpad/e6_dev/x7_subset_check.py run --part i/k --out F.jsonl [--seeds a,b] [--kinds shadow,own,pm,other]
         [--smoke] [--short S]
  python scratchpad/e6_dev/x7_subset_check.py analyse --stored DIR --x7 DIR [--out DIR]
"""
from __future__ import annotations

import copy
import dataclasses
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

import e6p_certsafe as X  # noqa: E402  (frozen certsafe driver)
import e6p_conf as C  # noqa: E402
import e6p_opta_ka as KA  # noqa: E402
import e6p_screen as S  # noqa: E402
import e6p_step2_dev as D  # noqa: E402
import numpy as np  # noqa: E402
import x4_random_defer as X4  # noqa: E402  (loader / record helpers)

from cdd_oran.decision import certsafe as CS  # noqa: E402

SUB = "x7"
REF = "sub:ES+PowerES"
REFS = ("CS:granger@dev", "CS:granger_by", "CS:two_tower@dev", "CS:shap_gbdt@dev")
SHORT = {"CS:granger@dev": "granger", "CS:granger_by": "granger_by", "CS:two_tower@dev": "two_tower",
         "CS:shap_gbdt@dev": "shap"}
KINDS = ("shadow", "own", "pm", "other")
FAMS = ("prot_min", "ptx", "carrier", "sleep")
SHADOW_ARM = "X7:sub|shadow"
AGREE_BOOT, AGREE_KEY = 2000, (20261005, 7)
SKIP = X4.SKIP | {"agree", "x7"}


def arm_name(ref: str, kind: str) -> str:
    return SHADOW_ARM if kind == "shadow" else f"X7:{ref}|{kind}"


def _dir(r) -> int:
    return int((r["prop"] > r["cur"]) - (r["prop"] < r["cur"]))


def _key(r, t) -> str:
    return f"{r['knob'][0]}|{_dir(r):+d}|{'pre' if t < KA.WARM else 'post'}"


def _rej(d) -> int:
    return int(d == "reject")


class Shadowed:
    """Applies ``primary``'s decisions; queries every ``shadows[name]`` on a deep copy of the obs (taken before the
    primary sees it) and tallies the per-request decision pairs. Adds nothing to the applied decisions."""

    def __init__(self, primary, shadows: dict):
        self.primary, self.shadows = primary, shadows
        self.agree = {nm: {} for nm in shadows}

    def __call__(self, obs):
        copies = {nm: copy.deepcopy(obs) for nm in self.shadows}
        out = self.primary(obs)
        for nm, sh in self.shadows.items():
            sd = sh(copies[nm])["decisions"]
            tab = self.agree[nm]
            for r, a, b in zip(obs["requests"], out["decisions"], sd, strict=True):
                k = _key(r, obs["t"])
                c = tab.setdefault(k, [0, 0, 0, 0])
                c[2 * _rej(a) + _rej(b)] += 1
        return out


class Restrict:
    """``inner``'s decisions with every deferral ("reject") of a request outside ``keep`` families replaced by accept.
    The inner arbiter's state is untouched (output override only)."""

    def __init__(self, inner, keep: frozenset):
        self.inner, self.keep = inner, frozenset(keep)
        self.n = {"overridden": 0, "kept": 0}

    def __call__(self, obs):
        out = self.inner(obs)
        dec = list(out["decisions"])
        for i, (r, d) in enumerate(zip(obs["requests"], dec, strict=True)):
            if d == "reject":
                if r["knob"][0] in self.keep:
                    self.n["kept"] += 1
                else:
                    dec[i] = "accept"
                    self.n["overridden"] += 1
        out = dict(out)
        out["decisions"] = dec
        return out


def _raw_ref(ref: str, art: dict):
    t = art["tables"][ref]
    return CS.certsafe_arbiter(t["map"], t["uncertified"], X.THETA, X.K_CONF, X.T, X.OPEN_RULE)


def make(kind: str, ref: str | None, sd: int, warmup_s: float, art: dict):
    if kind == "shadow":
        return Shadowed(D.make_arbiter(REF, warmup_s), {r: _raw_ref(r, art) for r in REFS})
    if kind == "own":
        return Shadowed(X.make_arbiter(ref, sd, warmup_s, art), {REF: D.make_arbiter(REF, warmup_s)})
    keep = frozenset({"prot_min"}) if kind == "pm" else frozenset(FAMS) - {"prot_min"}
    return KA.Counting(Restrict(_raw_ref(ref, art), keep), is_unit=False)


def job(seed: int, kind: str, ref: str | None, lf: float, art: dict, smoke=False, short=None) -> dict:
    X.check_seed(seed, "eval")
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(C.PAIR, C.STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make(kind, ref, sd, cfg.warmup_s, art)
    t_wall, t_cpu = time.time(), time.process_time()
    env, _ = S.run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    pc, agree = {}, None
    if isinstance(arb, Shadowed):
        agree = arb.agree
        p = arb.primary
        if isinstance(p, KA.Counting):                    # stored referee record layout (e6p_certsafe.arm_job)
            pc = dict(p.n)
            pol = getattr(p.inner, "policy", None)
            pc["policy"] = dict(getattr(pol, "n", None) or getattr(p.inner, "n", None) or {})
            if hasattr(p.inner, "passed"):
                pc["passed"] = {"all": int(p.inner.passed)}
    else:
        pc = dict(arb.n)
        pol = getattr(arb.inner.inner, "policy", None)
        pc["policy"] = dict(getattr(pol, "n", None) or {})
        pc["restrict"] = dict(arb.inner.n)
    return {"kind": "job", "schema": X.SCHEMA_EVAL, "key": [seed, arm_name(ref, kind)], "seed": seed, "cfg_seed": sd,
            "arm": arm_name(ref, kind), "sub": SUB, "cs_stage": "eval", "smoke": smoke, "short": short,
            "episode_s": env.total_s, "warmup_s": cfg.warmup_s, "load_factor": lf, **S.outcome(env),
            "policy_counts": pc, "agree": agree, "x7": {"kind": kind, "ref": ref},
            "cpu_s": round(cpu, 2), "secs": round(time.time() - t_wall, 1), "rss_mb": D.peak_rss_mb()}


def jobs(seeds=None, kinds=KINDS) -> list:
    """[(seed, kind, ref)] seed-major: per seed the shadow job, then per referee own / pm / other."""
    seeds = list(seeds) if seeds else X.stage_seeds("eval")
    J = []
    for s in seeds:
        X.check_seed(s, "eval")
        if "shadow" in kinds:
            J.append((s, "shadow", None))
        J += [(s, k, r) for r in REFS for k in ("own", "pm", "other") if k in kinds]
    return J


def run(part, out, smoke=False, short=None, seeds=None, kinds=KINDS):
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    art = X.load_artifact()
    state = S.load_state()
    lf = S.lf_of(state, C.STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header("x7:eval", part, smoke, state)
    try:
        import cloud
        nenv = cloud.numeric_env()
    except Exception as e:                                              # noqa: BLE001
        nenv = {"error": repr(e)}
    head.update(kind="header", schema=X.SCHEMA_EVAL, driver="x7_subset_check", sub=SUB, cs_stage="eval",
                platform=nenv, host=os.environ.get("XM_PLATFORM"), code_commit=os.environ.get("XM_CODE_COMMIT"),
                code_dirty=os.environ.get("XM_CODE_DIRTY"), maps=C.maps_status(), artifact=X.artifact_status(),
                certsafe_freeze=X.freeze_status(),
                driver_sha256={f: X.sha_lf(os.path.join(ROOT, f)) for f in (
                    "scratchpad/e6_dev/x7_subset_check.py", "scratchpad/e6_dev/e6p_certsafe.py",
                    "cdd_oran/decision/certsafe.py", "cdd_oran/decision/mapgate.py",
                    "cdd_oran/envs/e6/baselines.py", "cdd_oran/envs/e6/env.py")},
                consts={"ref": REF, "refs": list(REFS), "kinds": list(kinds), "load_factor": lf,
                        "smoke": {"short": short} if smoke else None})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    J = jobs(seeds, kinds)
    t0, n = time.time(), 0
    for u, (seed, kind, ref) in enumerate(J):
        if u % k != i or (seed, arm_name(ref, kind)) in done:
            continue
        rec = job(seed, kind, ref, lf, art, smoke, short)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("seed", "arm", "psvr", "rlf", "st_req", "st_rej", "cpu_s")},
                         default=S._js), flush=True)
    S._append(out, {"kind": "close", "part": part, "n_jobs": n, "n_total": len(J), "secs": round(time.time() - t0, 1)})


# ============================================================================================ analysis
def _agree_sum(recs, shadow, sel=lambda key: True):
    """Sum of [aa, ar, ra, rr] over records (list) for keys passing ``sel``; also per record (for the bootstrap)."""
    per = np.zeros((len(recs), 4))
    for j, r in enumerate(recs):
        for key, c in (r.get("agree") or {}).get(shadow, {}).items():
            if sel(key):
                per[j] += c
    return per


def _agree_stats(per, rng_idx):
    tot = per.sum(0)
    n = tot.sum()
    agr = (tot[0] + tot[3]) / n if n else float("nan")
    b = per[rng_idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ab = (b[:, 0] + b[:, 3]) / b.sum(1)
    return {"n": int(n), "agree": float(agr), "ci90": D._ci(ab), "aa": int(tot[0]), "ar": int(tot[1]),
            "ra": int(tot[2]), "rr": int(tot[3])}


def agreement(recs, shadow, n_seeds):
    idx = np.random.default_rng(list(AGREE_KEY)).integers(0, n_seeds, (AGREE_BOOT, n_seeds))
    out = {}
    for win, wsel in (("all", lambda k: True), ("post", lambda k: k.endswith("|post"))):
        o = {"overall": _agree_stats(_agree_sum(recs, shadow, wsel), idx)}
        for f in FAMS:
            o[f] = _agree_stats(_agree_sum(recs, shadow, lambda k, f=f, wsel=wsel: wsel(k) and k.split("|")[0] == f), idx)
            for d in ("+1", "-1"):
                o[f"{f}|{d}"] = _agree_stats(_agree_sum(
                    recs, shadow, lambda k, f=f, d=d, wsel=wsel: wsel(k) and k.split("|")[0] == f and k.split("|")[1] == d), idx)
        out[win] = o
    return out


def _same(a: dict, b: dict) -> list:
    return [x for x in sorted((set(a) | set(b)) - SKIP - {"policy_counts", "key", "kind", "schema"})
            if a.get(x) != b.get(x)]


def analyse(stored: str, x7: str, out: str, n_boot: int = 10_000):
    import e6p_conf_analyze as AN
    files_s = sorted(glob.glob(os.path.join(stored, "*", "res_*.jsonl")))
    files_x = sorted(glob.glob(os.path.join(x7, "**", "res_*.jsonl"), recursive=True))
    Rs, _, _ = X4._load(files_s, X.SUB)
    Rx, heads_x, bad_x = X4._load(files_x, SUB)
    art = json.load(open(os.path.join(ROOT, X.ARTIFACT_DOC)))
    for s in Rs:
        for arm, tgt in art["alias_of"].items():
            if tgt != arm and tgt in Rs[s] and arm not in Rs[s]:
                Rs[s][arm] = dict(Rs[s][tgt], arm=arm, alias_of=tgt)
    seeds = X.stage_seeds("eval")
    expect = [arm_name(r, k) for k, r in [(k, r) for _, k, r in jobs(seeds[:1])]]
    missing = {a: [s for s in seeds if a not in Rx.get(s, {})] for a in expect}
    # ---- bit identity of the re-runs with the stored records
    ident = {}
    for a, src in [(SHADOW_ARM, REF)] + [(arm_name(r, "own"), r) for r in REFS]:
        nd, pol_eq, fields = 0, 0, {}
        for s in seeds:
            v, k = Rx.get(s, {}).get(a), Rs[s][src]
            if v is None:
                continue
            d = _same(v, k)
            nd += not d
            for x in d:
                fields[x] = fields.get(x, 0) + 1
            pol_eq += (k.get("policy_counts") == v.get("policy_counts")) if src != REF else 1
        ident[a] = {"source": src, "identical_outcome_seeds": nd, "policy_counts_equal_seeds": pol_eq,
                    "diff_fields": fields}
    # ---- A2 agreement
    agr = {"stream_ref": {}, "stream_own": {}}
    if all(SHADOW_ARM in Rx.get(s, {}) for s in seeds):
        recs = [Rx[s][SHADOW_ARM] for s in seeds]
        agr["stream_ref"] = {r: agreement(recs, r, len(seeds)) for r in REFS}
    for r in REFS:
        a = arm_name(r, "own")
        if all(a in Rx.get(s, {}) for s in seeds):
            agr["stream_own"][r] = agreement([Rx[s][a] for s in seeds], REF, len(seeds))
    # ---- A3 / A4: one bootstrap over stored + attribution arms
    R = {s: dict(Rs[s]) for s in Rs}
    a3 = [arm_name(r, k) for r in REFS for k in ("pm", "other")]
    a3 = [a for a in a3 if all(a in Rx.get(s, {}) for s in seeds)]
    for s in R:
        for a in a3:
            R[s][a] = Rx[s][a]
    stored_arms = [a for a in X.ALL_ARMS if any(a in Rs[s] for s in Rs)]
    st = AN.arm_stats_multi(R, stored_arms + a3, n_boot)
    P, B = st["point"], st["boot"]
    E = json.load(open(os.path.join(ROOT, "scratchpad", "e6_dev", "decision", "cs_eval.json")))
    bad = [a for a in stored_arms if not (np.isclose(P[a]["R"], E["arms"][a]["R"], rtol=1e-9)
                                         and np.allclose(D._ci(B[a]["R"]), E["arms"][a]["R_ci90"], rtol=1e-9))]
    n = st["n"]
    idx = np.random.default_rng([AN.EVAL_BOOT_TAG, AN.EVAL_BOOT_KEY, n]).integers(0, n, (n_boot, n))
    Vb = {a: D._pooled(D._arrays(R, st["seeds"], a), idx)["V"] for a in [REF] + list(REFS) + a3}
    rows, diffs = {}, {}
    for a in [REF, "noarb"] + list(REFS) + a3:
        rows[a] = dict(P[a], R_ci90=D._ci(B[a]["R"]), Rstar_ci90=D._ci(B[a]["Rstar"]),
                       retention_ci90=D._ci(B[a]["retention"]),
                       guard_ratio_ci90={g: D._ci(v) for g, v in B[a]["guard"].items()})
        if a in Vb and a != REF:
            dv_seed = np.array([R[s][a]["psvr"] - R[s][REF]["psvr"] for s in st["seeds"]])
            same_seed = sum(not _same(R[s][a], R[s][REF]) for s in st["seeds"])
            diffs[a] = {"dR": P[a]["R"] - P[REF]["R"], "dR_ci90": D._ci(B[a]["R"] - B[REF]["R"]),
                        "dV": P[a]["V"] - P[REF]["V"], "dV_ci90": D._ci(Vb[a] - Vb[REF]),
                        "seeds_bit_identical_to_ref": int(same_seed),
                        "per_seed_dpsvr": {"mean": float(dv_seed.mean()), "sd": float(dv_seed.std(ddof=1)),
                                           "zero": int((dv_seed == 0).sum()),
                                           "q05_q50_q95": [float(x) for x in np.quantile(dv_seed, [.05, .5, .95])]}}
    # share of the referee's R reproduced by the prot_min-only replay
    share = {r: {"R_full": P[r]["R"], "R_pm": P.get(arm_name(r, "pm"), {}).get("R"),
                 "R_other": P.get(arm_name(r, "other"), {}).get("R")} for r in REFS}
    cpu = sum(Rx[s][a]["cpu_s"] for s in Rx for a in Rx[s]) / 3600
    fac = json.load(open(os.path.join(ROOT, "scratchpad", "xmethod", "results", "exp_c", "calib", "factors.json")))
    vk = next((k for k in fac["factors"]["*"] if k.startswith("vps|")), None)
    plats = sorted({json.dumps(AN.platform_key(h.get("platform")), default=str) for h in heads_x
                    if h.get("driver") == "x7_subset_check" and not h.get("smoke")})
    rep = {"schema": "x7-subset-check/1", "declared": "scratchpad/xmethod/EXTRAS_PROTOCOL.md Amendments 2026-10-05 X7",
           "status": "POST HOC, DESCRIPTIVE", "ref": REF, "refs": list(REFS), "n_seeds": n,
           "missing": {a: v for a, v in missing.items() if v}, "bad_lines": bad_x, "platforms": plats,
           "code_commits": sorted({str(h.get("code_commit")) for h in heads_x if h.get("driver") == "x7_subset_check"}),
           "repro_cs_eval_mismatch": bad, "identity_reruns": ident, "agreement": agr, "rows": rows,
           "diff_vs_ref": diffs, "attribution": share,
           "cpu": {"vps_cpu_h": cpu, "factor_star": fac["factors"]["*"].get(vk) if vk else None,
                   "kaggle_ref_cpu_h": cpu * fac["factors"]["*"][vk]["f"] if vk else None}}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "x7_tables.json"), "w", newline="\n") as fh:
        json.dump(rep, fh, indent=1, default=S._js)
    return rep


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
            kinds=tuple(a["--kinds"].split(",")) if a.get("--kinds") else KINDS)
    elif cmd == "analyse":
        analyse(a["--stored"], a["--x7"], a.get("--out", "scratchpad/xmethod/results/extras/x7"),
                int(a.get("--n-boot", 10_000)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])
