"""ARTIFACT builder / loader for PMRT (Predictable Matched-Filter Randomization Test) in the v4 fresh-seed re-test
(agent I, 2026-09-30). Formerly ``mscr_plus_artifacts.py`` (method label "MSCR+"; docs/benchmark/METHOD_NAMES.md).

  python scratchpad/e6_dev/pmrt_artifacts.py build --params P.pkl --prior prior_ev2.json --out A.json
         [--supersedes OLD.json] [--src-root DIR]
  python scratchpad/e6_dev/pmrt_artifacts.py verify --artifact A.json --params P.pkl --small DIR [--prior F]
         [--legacy OLD.json]

``build`` writes ONE pure-data JSON file (no pickle, no sklearn) holding every learned piece the integrated PMRT
statistic and its declaration layers use, all fitted BEFORE any v4 record exists:
  statistic   (pmrt; fitted by agent S on ev2 sub "v2" 480 eps + DEV 20 eps, Kaggle kernel mscrplus-s-2):
              config (PmrtConfig), receiver table rho per family, training ctx_keys, and per hypothesis f|rel|kpi the
              ridge adjustment (fill, mx, sx, keep, B, lam), kernels K for the variants "plain" and "loadsp" (full /
              without the [90, 150) s bin), slot count S, training plain z and kernel signs (sdir), the running-centre
              intercepts (prior), the log-variance scale model h for the Huber clip (w, xm, ym, med, kappa);
  layer       (fdr_layer): the ev2 prior = signed effect z of the SAME statistic on ev2 per arm (plain_c, loadsp_c,
              max), n_prior, q, floor, cap, thr, and (for transparency) the resulting directions and the weights at the
              planned slice sizes.
Floats are written with Python's repr (exact round trip). The file's sha256 (LF bytes as written) is pinned by the
analyzers; ``verify`` reloads the artifact and checks that the statistic columns are BIT-IDENTICAL to the ones computed
from the original params pickle on the small local placebo tables, and that the layer inputs match.

Schemas: ``ARTIFACT_SCHEMA`` "e6p-pmrt/1" (E6P_PMRT_V4.json) and ``LEGACY_SCHEMA`` "e6p-mscrplus-frozen/1" (the frozen
v4 artifact E6P_MSCRPLUS_V4_FROZEN.json, same learned content under the old labels). ``load_artifact`` reads both. A
PMRT artifact built with ``--supersedes`` records the old artifact's sha256, the old code sha256s
(``legacy_code_sha256``, keyed by the OLD paths) and the file renames (``renamed_from``), and ``build`` refuses unless
every learned field is identical to the old artifact's (``LEARNED_KEYS``, ``LAYER_KEYS``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pmrt_bench as PB  # noqa: E402

from cdd_oran.decision import fdr_layer as FL  # noqa: E402
from cdd_oran.decision import pmrt as PM  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES  # noqa: E402

ARTIFACT_SCHEMA = "e6p-pmrt/1"
LEGACY_SCHEMA = "e6p-mscrplus-frozen/1"
SCHEMAS = (ARTIFACT_SCHEMA, LEGACY_SCHEMA)
METHOD = "PMRT (Predictable Matched-Filter Randomization Test)"
VARIANTS = ("plain", "loadsp")
CORE_KEYS = ("fill", "mx", "sx", "keep", "B", "lam", "S", "z_plain")
H_KEYS = ("w", "xm", "ym", "med", "kappa")
N_PLANNED = (60, 120, 600)
CODE = ("cdd_oran/decision/pmrt.py", "cdd_oran/decision/fdr_layer.py", "cdd_oran/decision/eprocess_units.py",
        "cdd_oran/decision/crt_units.py", "cdd_oran/decision/crt_units_v2.py", "cdd_oran/decision/disc_bench.py",
        "scratchpad/e6_dev/pmrt_bench.py", "scratchpad/e6_dev/pmrt_artifacts.py")
RENAMED_FROM = {"cdd_oran/decision/pmrt.py": "cdd_oran/decision/crt_units_plus.py",
                "cdd_oran/decision/fdr_layer.py": "cdd_oran/decision/mscr_multi.py",
                "scratchpad/e6_dev/pmrt_bench.py": "scratchpad/e6_dev/mscr_integrate_bench.py",
                "scratchpad/e6_dev/pmrt_artifacts.py": "scratchpad/e6_dev/mscr_plus_artifacts.py"}
LEARNED_KEYS = ("config", "n_train_units", "n_train_eps", "ctx_keys", "rho", "hyp", "statistic_arms", "max_of")
LAYER_KEYS = ("q", "floor", "thr", "cap", "n_prior", "z", "directions", "weights")


def _py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def sha_file(path: str) -> str:
    b = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def learned_diff(a: dict, b: dict) -> list:
    """Names of the learned fields (``LEARNED_KEYS``, layer ``LAYER_KEYS``) that differ between two artifacts (parsed
    JSON; exact equality, floats included)."""
    bad = [k for k in LEARNED_KEYS if a.get(k) != b.get(k)]
    la, lb = a.get("layer"), b.get("layer")
    if (la is None) != (lb is None):
        bad.append("layer")
    elif la is not None:
        bad += [f"layer.{k}" for k in LAYER_KEYS if la.get(k) != lb.get(k)]
    return bad


def build(params: dict, prior: dict | None, src: dict, legacy: dict | None = None,
          legacy_sha: str | None = None, legacy_name: str | None = None) -> dict:
    hyp = {}
    for h, hp in params["hyp"].items():
        core = hp["core"]
        c = {k: _py(core[k]) for k in CORE_KEYS}
        c["K"] = {v: [_py(core["K"][v][0]), _py(core["K"][v][1])] for v in VARIANTS}
        c["sdir"] = {v: float(core["sdir"][v]) for v in VARIANTS}
        c["prior"] = {v: {"1": float(core["prior"][v][1]), "-1": float(core["prior"][v][-1])} for v in VARIANTS}
        hyp[h] = {"core": c, "h": {v: {k: _py(hp["h"][v][k]) for k in H_KEYS} for v in VARIANTS}}
    art = {"schema": ARTIFACT_SCHEMA, "method": METHOD, "version": PM.PMRT_VERSION,
           "legacy_version": params["version"], "config": _py(params["config"]),
           "n_train_units": params["n_train_units"], "n_train_eps": params["n_train_eps"],
           "ctx_keys": list(params["ctx_keys"]), "rho": _py(params["rho"]), "hyp": hyp,
           "statistic_arms": list(PB.COL_ARMS), "max_of": list(PB.MAX_OF), "source": src,
           "code_sha256": {p: sha_file(os.path.join(ROOT, p)) for p in CODE if os.path.exists(os.path.join(ROOT, p))}}
    if prior is not None:
        lay = {"q": PB.Q, "floor": PB.FLOOR, "thr": PB.THR, "cap": FL.W_CAP, "n_prior": int(prior["n_prior"]),
               "layer_version": FL.FDR_LAYER_VERSION, "legacy_layer_version": FL.LEGACY_VERSION,
               "z": {}, "directions": {}, "weights": {}}
        for s in PB.STATS:
            zs = {h: (float(v["z"]) if v.get("status") == "tested" else None) for h, v in prior["z"][s].items()}
            lay["z"][s] = zs
            pr = FL.Prior.from_stats(prior["z"][s], n_prior=int(prior["n_prior"]))
            lay["directions"][s] = {"|".join(h): d for h, d in FL.prior_directions(pr, PB.THR).items() if d}
            lay["weights"][s] = {str(n): {"|".join(h): w for h, w in FL.prior_weights(pr, n, PB.Q, PB.FLOOR).items()}
                                 for n in N_PLANNED}
        art["layer"] = lay
    if legacy is not None:
        bad = learned_diff(art, legacy)
        if bad:
            raise ValueError(f"learned content differs from the superseded artifact: {bad}")
        art["supersedes"] = {"artifact": legacy_name, "sha256": legacy_sha}
        art["legacy_code_sha256"] = dict(legacy.get("code_sha256") or {})
        art["renamed_from"] = {k: v for k, v in RENAMED_FROM.items() if k in art["code_sha256"]}
        art["equivalence"] = ("label-only rename: statistic, p-values, weights, directions and declarations "
                              "bit-identical to the superseded artifact + the frozen code (scratchpad/e6_dev/"
                              "pmrt_equivalence.py; docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md)")
    return art


def load_artifact(path: str) -> tuple[dict, dict]:
    """-> (params usable by pmrt_bench.lean_columns / integrated_family, {stat: fdr_layer.Prior}). Reads the PMRT
    schema and the legacy (frozen v4, "MSCR+") schema: the learned content is the same."""
    a = json.load(open(path))
    if a.get("schema") not in SCHEMAS:
        raise ValueError(f"{path}: schema {a.get('schema')}")
    hyp = {}
    for h, hp in a["hyp"].items():
        c = hp["core"]
        core = {"fill": np.array(c["fill"], float), "mx": np.array(c["mx"], float), "sx": np.array(c["sx"], float),
                "keep": np.array(c["keep"], bool), "B": np.array(c["B"], float), "lam": c["lam"], "S": int(c["S"]),
                "z_plain": float(c["z_plain"]), "sdir": dict(c["sdir"]),
                "K": {v: (np.array(k[0], float), np.array(k[1], float)) for v, k in c["K"].items()},
                "prior": {v: {1: p["1"], -1: p["-1"]} for v, p in c["prior"].items()}}
        hm = {v: {"w": np.array(m["w"], float), "xm": np.array(m["xm"], float), "ym": float(m["ym"]),
                  "med": float(m["med"]), "kappa": float(m["kappa"])} for v, m in hp["h"].items()}
        hyp[h] = {"core": core, "h": hm, "gb": None}
    params = {"version": a["version"], "config": a["config"], "ctx_keys": a["ctx_keys"],
              "rho": {f: a["rho"][f] for f in a["rho"]}, "hyp": hyp}
    priors = {}
    if "layer" in a:
        L = a["layer"]
        for s in PB.STATS:
            d = {h: ({"z": z, "status": "tested"} if z is not None else {"status": "undetermined"})
                 for h, z in L["z"][s].items()}
            priors[s] = FL.Prior.from_stats(d, n_prior=L["n_prior"], source="frozen artifact")
    return params, priors


def pmrt_config(params: dict, B: int) -> PM.PmrtConfig:
    c = {k: v for k, v in params["config"].items() if k in PM.PmrtConfig.__dataclass_fields__}
    c["B"] = int(B)
    c["best_exclude"] = tuple(c.get("best_exclude", ()))
    return PM.PmrtConfig(**c)


def _rel(path: str, root: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")


def cmd_build(a):
    params = PB.load_params_nogb(a.params)
    prior = json.load(open(a.prior)) if a.prior else None
    root = a.src_root or ROOT
    src = {"params": _rel(a.params, root),
           "params_sha256": hashlib.sha256(open(a.params, "rb").read()).hexdigest(),
           "prior": _rel(a.prior, root) if a.prior else None,
           "prior_sha256": sha_file(a.prior) if a.prior else None,
           "fit": "agent S, Kaggle kernel bishalpanta/mscrplus-s-2 (train = ev2 sub v2 480 eps + DEV 20 eps)",
           "prior_run": "agent I, integrated statistic on ev2 (480 eps), split 0"}
    legacy = json.load(open(a.supersedes)) if a.supersedes else None
    art = build(params, prior, src, legacy, sha_file(a.supersedes) if a.supersedes else None,
                os.path.basename(a.supersedes) if a.supersedes else None)
    if legacy is not None:
        for k in ("params", "params_sha256", "prior", "prior_sha256"):
            if legacy.get("source", {}).get(k) != src[k]:
                raise ValueError(f"source {k}: {src[k]} != superseded {legacy.get('source', {}).get(k)}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="\n") as fh:
        json.dump(art, fh, indent=None, separators=(",", ":"))
    sha = sha_file(a.out)
    open(a.out + ".sha256", "w", newline="\n").write(f"{sha}  {os.path.basename(a.out)}\n")
    print(json.dumps({"out": a.out, "mb": round(os.path.getsize(a.out) / 2 ** 20, 2), "sha256": sha,
                      "hyp": len(art["hyp"]), "layer": "layer" in art, "supersedes": art.get("supersedes")}))


def cmd_verify(a):
    pk = PB.load_params_nogb(a.params)
    pa, priors = load_artifact(a.artifact)
    cfg_k, cfg_a = pmrt_config(pk, a.B), pmrt_config(pa, a.B)
    assert cfg_k == cfg_a, (cfg_k, cfg_a)
    for nm in ("placebo1", "plxc2", "dev1"):
        pdp = PM.pmrt_data(PM.load_pmrt_pool([os.path.join(a.small, f"{nm}.npz")]))
        for f in FAMILIES:
            if len(pdp.ud.rows_of(f)) < cfg_k.min_units:
                continue
            r1, W1, _, m1 = PB.lean_columns(pdp, f, pk, cfg_k)
            r2, W2, _, m2 = PB.lean_columns(pdp, f, pa, cfg_a)
            assert np.array_equal(r1, r2) and np.array_equal(W1, W2), (nm, f, float(np.nanmax(np.abs(W1 - W2))))
            assert m1 == m2, (nm, f)
        run1, run2 = PB.run_integrated(pdp, pk, cfg_k, 9), PB.run_integrated(pdp, pa, cfg_a, 9)
        assert json.dumps(run1, default=float, sort_keys=True) == json.dumps(run2, default=float, sort_keys=True)
        print(f"{nm}: artifact statistic bit-identical to the params pickle ({pdp.n} units)")
    if priors and a.prior:
        pj = json.load(open(a.prior))
        for s in PB.STATS:
            p0 = FL.Prior.from_stats(pj["z"][s], n_prior=pj["n_prior"])
            for n in N_PLANNED:
                w0, w1 = FL.prior_weights(p0, n, PB.Q, PB.FLOOR), FL.prior_weights(priors[s], n, PB.Q, PB.FLOOR)
                assert all(abs(w0[h] - w1[h]) < 1e-12 for h in w0), (s, n)
            assert FL.prior_directions(p0, PB.THR) == FL.prior_directions(priors[s], PB.THR)
        print("layer: weights / directions identical to the prior json")
    if a.legacy:
        new, old = json.load(open(a.artifact)), json.load(open(a.legacy))
        bad = learned_diff(new, old)
        assert not bad, bad
        sup = new.get("supersedes") or {}
        assert sup.get("sha256") == sha_file(a.legacy), (sup, sha_file(a.legacy))
        print(f"learned content identical to {os.path.basename(a.legacy)} (sha {sup['sha256']})")
    print(f"sha256 {sha_file(a.artifact)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    b = sp.add_parser("build")
    b.add_argument("--params", required=True)
    b.add_argument("--prior", default="")
    b.add_argument("--out", required=True)
    b.add_argument("--supersedes", default="", help="the artifact this one replaces (learned content must match)")
    b.add_argument("--src-root", dest="src_root", default="",
                   help="repo root the recorded params / prior paths are relative to (default: this checkout)")
    v = sp.add_parser("verify")
    v.add_argument("--artifact", required=True)
    v.add_argument("--params", required=True)
    v.add_argument("--prior", default="")
    v.add_argument("--small", required=True)
    v.add_argument("--B", type=int, default=199)
    v.add_argument("--legacy", default="", help="the superseded artifact: learned content + supersedes sha check")
    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    {"build": cmd_build, "verify": cmd_verify}[a.op](a)


if __name__ == "__main__":
    main()
