"""FROZEN-ARTIFACT builder for the MSCR+ v4 fresh-seed re-test (agent I, 2026-09-30; scratch until the v4 freeze).

  python scratchpad/e6_dev/mscr_plus_artifacts.py build --params P.pkl --prior prior_ev2.json --out A.json
  python scratchpad/e6_dev/mscr_plus_artifacts.py verify --artifact A.json --params P.pkl --small DIR

``build`` writes ONE pure-data JSON file (no pickle, no sklearn) holding every learned piece the integrated MSCR+
statistic and its declaration layers use, all fitted BEFORE any v4 record exists:
  statistic   (crt_units_plus; fitted by agent S on ev2 sub "v2" 480 eps + DEV 20 eps, Kaggle mscrplus-s-2):
              config (PlusConfig), receiver table rho per family, training ctx_keys, and per hypothesis f|rel|kpi the
              ridge adjustment (fill, mx, sx, keep, B, lam), kernels K for the variants "plain" and "loadsp" (full /
              without the [90, 150) s bin), slot count S, training plain z and kernel signs (sdir), the running-centre
              intercepts (prior), the log-variance scale model h for the Huber clip (w, xm, ym, med, kappa);
  layer       (mscr_multi): the ev2 prior = signed effect z of the SAME statistic on ev2 per arm (plain_c, loadsp_c,
              max), n_prior, q, floor, cap, thr, and (for transparency) the resulting directions and the weights at the
              planned slice sizes.
Floats are written with Python's repr (exact round trip). The file's sha256 (LF bytes as written) goes into the v4
protocol; ``verify`` reloads the artifact and checks that the statistic columns are BIT-IDENTICAL to the ones computed
from the original params pickle on the small local placebo tables, and that the layer inputs match.
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

import mscr_integrate_bench as IB  # noqa: E402
from cdd_oran.decision import crt_units_plus as CP  # noqa: E402
from cdd_oran.decision import mscr_multi as MM  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES  # noqa: E402

ARTIFACT_SCHEMA = "e6p-mscrplus-frozen/1"
VARIANTS = ("plain", "loadsp")
CORE_KEYS = ("fill", "mx", "sx", "keep", "B", "lam", "S", "z_plain")
H_KEYS = ("w", "xm", "ym", "med", "kappa")
N_PLANNED = (60, 120, 600)
CODE = ("cdd_oran/decision/crt_units_plus.py", "cdd_oran/decision/mscr_multi.py", "cdd_oran/decision/eprocess_units.py",
        "cdd_oran/decision/crt_units.py", "cdd_oran/decision/crt_units_v2.py", "cdd_oran/decision/disc_bench.py",
        "scratchpad/e6_dev/mscr_integrate_bench.py", "scratchpad/e6_dev/mscr_plus_artifacts.py")


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


def build(params: dict, prior: dict | None, src: dict) -> dict:
    hyp = {}
    for h, hp in params["hyp"].items():
        core = hp["core"]
        c = {k: _py(core[k]) for k in CORE_KEYS}
        c["K"] = {v: [_py(core["K"][v][0]), _py(core["K"][v][1])] for v in VARIANTS}
        c["sdir"] = {v: float(core["sdir"][v]) for v in VARIANTS}
        c["prior"] = {v: {"1": float(core["prior"][v][1]), "-1": float(core["prior"][v][-1])} for v in VARIANTS}
        hyp[h] = {"core": c, "h": {v: {k: _py(hp["h"][v][k]) for k in H_KEYS} for v in VARIANTS}}
    art = {"schema": ARTIFACT_SCHEMA, "version": params["version"], "config": _py(params["config"]),
           "n_train_units": params["n_train_units"], "n_train_eps": params["n_train_eps"],
           "ctx_keys": list(params["ctx_keys"]), "rho": _py(params["rho"]), "hyp": hyp,
           "statistic_arms": list(IB.COL_ARMS), "max_of": list(IB.MAX_OF), "source": src,
           "code_sha256": {p: sha_file(os.path.join(ROOT, p)) for p in CODE if os.path.exists(os.path.join(ROOT, p))}}
    if prior is not None:
        lay = {"q": IB.Q, "floor": IB.FLOOR, "thr": IB.THR, "cap": MM.W_CAP, "n_prior": int(prior["n_prior"]),
               "multi_version": MM.MULTI_VERSION, "z": {}, "directions": {}, "weights": {}}
        for s in IB.STATS:
            zs = {h: (float(v["z"]) if v.get("status") == "tested" else None) for h, v in prior["z"][s].items()}
            lay["z"][s] = zs
            pr = MM.Prior.from_stats(prior["z"][s], n_prior=int(prior["n_prior"]))
            lay["directions"][s] = {"|".join(h): d for h, d in MM.prior_directions(pr, IB.THR).items() if d}
            lay["weights"][s] = {str(n): {"|".join(h): w for h, w in MM.prior_weights(pr, n, IB.Q, IB.FLOOR).items()}
                                 for n in N_PLANNED}
        art["layer"] = lay
    return art


def load_artifact(path: str) -> tuple[dict, dict]:
    """-> (params usable by mscr_integrate_bench.lean_columns / integrated_family, {stat: mscr_multi.Prior})."""
    a = json.load(open(path))
    if a.get("schema") != ARTIFACT_SCHEMA:
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
        for s in IB.STATS:
            d = {h: ({"z": z, "status": "tested"} if z is not None else {"status": "undetermined"})
                 for h, z in L["z"][s].items()}
            priors[s] = MM.Prior.from_stats(d, n_prior=L["n_prior"], source="frozen artifact")
    return params, priors


def plus_config(params: dict, B: int) -> CP.PlusConfig:
    c = {k: v for k, v in params["config"].items() if k in CP.PlusConfig.__dataclass_fields__}
    c["B"] = int(B)
    c["best_exclude"] = tuple(c.get("best_exclude", ()))
    return CP.PlusConfig(**c)


def cmd_build(a):
    params = IB.load_params_nogb(a.params)
    prior = json.load(open(a.prior)) if a.prior else None
    src = {"params": os.path.relpath(os.path.abspath(a.params), ROOT).replace("\\", "/"),
           "params_sha256": hashlib.sha256(open(a.params, "rb").read()).hexdigest(),
           "prior": os.path.relpath(os.path.abspath(a.prior), ROOT).replace("\\", "/") if a.prior else None,
           "prior_sha256": sha_file(a.prior) if a.prior else None,
           "fit": "agent S, Kaggle kernel bishalpanta/mscrplus-s-2 (train = ev2 sub v2 480 eps + DEV 20 eps)",
           "prior_run": "agent I, integrated statistic on ev2 (480 eps), split 0"}
    art = build(params, prior, src)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="\n") as fh:
        json.dump(art, fh, indent=None, separators=(",", ":"))
    sha = sha_file(a.out)
    open(a.out + ".sha256", "w", newline="\n").write(f"{sha}  {os.path.basename(a.out)}\n")
    print(json.dumps({"out": a.out, "mb": round(os.path.getsize(a.out) / 2 ** 20, 2), "sha256": sha,
                      "hyp": len(art["hyp"]), "layer": "layer" in art}))


def cmd_verify(a):
    pk = IB.load_params_nogb(a.params)
    pa, priors = load_artifact(a.artifact)
    cfg_k, cfg_a = plus_config(pk, a.B), plus_config(pa, a.B)
    assert cfg_k == cfg_a, (cfg_k, cfg_a)
    for nm in ("placebo1", "plxc2", "dev1"):
        pdp = CP.plus_data(CP.load_plus_pool([os.path.join(a.small, f"{nm}.npz")]))
        for f in FAMILIES:
            if len(pdp.ud.rows_of(f)) < cfg_k.min_units:
                continue
            r1, W1, _, m1 = IB.lean_columns(pdp, f, pk, cfg_k)
            r2, W2, _, m2 = IB.lean_columns(pdp, f, pa, cfg_a)
            assert np.array_equal(r1, r2) and np.array_equal(W1, W2), (nm, f, float(np.nanmax(np.abs(W1 - W2))))
            assert m1 == m2, (nm, f)
        run1, run2 = IB.run_integrated(pdp, pk, cfg_k, 9), IB.run_integrated(pdp, pa, cfg_a, 9)
        assert json.dumps(run1, default=float, sort_keys=True) == json.dumps(run2, default=float, sort_keys=True)
        print(f"{nm}: artifact statistic bit-identical to the params pickle ({pdp.n} units)")
    if priors and a.prior:
        pj = json.load(open(a.prior))
        for s in IB.STATS:
            p0 = MM.Prior.from_stats(pj["z"][s], n_prior=pj["n_prior"])
            for n in N_PLANNED:
                w0, w1 = MM.prior_weights(p0, n, IB.Q, IB.FLOOR), MM.prior_weights(priors[s], n, IB.Q, IB.FLOOR)
                assert all(abs(w0[h] - w1[h]) < 1e-12 for h in w0), (s, n)
            assert MM.prior_directions(p0, IB.THR) == MM.prior_directions(priors[s], IB.THR)
        print("layer: weights / directions identical to the prior json")
    print(f"sha256 {sha_file(a.artifact)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    b = sp.add_parser("build")
    b.add_argument("--params", required=True)
    b.add_argument("--prior", default="")
    b.add_argument("--out", required=True)
    v = sp.add_parser("verify")
    v.add_argument("--artifact", required=True)
    v.add_argument("--params", required=True)
    v.add_argument("--prior", default="")
    v.add_argument("--small", required=True)
    v.add_argument("--B", type=int, default=199)
    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    {"build": cmd_build, "verify": cmd_verify}[a.op](a)


if __name__ == "__main__":
    main()
