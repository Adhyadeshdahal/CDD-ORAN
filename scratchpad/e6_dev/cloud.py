"""Build the Kaggle bundle + private script kernels for E6 DEV grids.

  .venv/Scripts/python.exe scratchpad/e6_dev/cloud.py OUTDIR NAME [SCRIPT=grid.py] [KERNELS=3] [PIN=match]
  .venv/Scripts/python.exe scratchpad/e6_dev/cloud.py fingerprint        # numeric fingerprint of THIS machine (JSON)
OUTDIR/dataset/: <NAME>_bundle.zip (cdd_oran/{__init__,envs/__init__,envs/e6/*,envs/xtruce/*,decision/**}.py,
                 e6dev/*.py, MANIFEST.json with git HEAD + sha256 + local_env + local_fp) + dataset-metadata.json;
                 SCRIPT e6p_disc_*.py / e6p_conf*.py also bundles docs/benchmark/{E6P_DISCOVERY_PROTOCOL.md,
                 E6P_DISCOVERY_PROTOCOL_V4.md,E6P_CONFOUNDED_PROTOCOL.md,SEED_REGISTRY.json} and
                 docs/benchmark/artifacts/* (v4 PMRT artifacts, option-(a) maps artifact)
OUTDIR/kernel_<tag>/: 4 shards per kernel, one per CPU; one process per shard, single native thread each.

NUMERIC REPRODUCIBILITY DECISION (2026-09-28; probe scratchpad/decision_stack/diag_rank/k_numpy_repro.py):
  Trigger: commit b00920b, same seed, local numpy 2.4.2 (Windows, AVX2) vs Kaggle numpy 2.0.2 / scipy 1.16.3
  (Linux) gave diverging E6 trajectories (accept-all SVR 872.1 vs 877.0). E6 is chaotic (argmax serving / HO,
  thresholds, Poisson draws), so ONE ulp anywhere forks the trajectory.
  Measured on the same Windows box: numpy 2.0.2 + scipy 1.16.3 vs 2.4.2 + 1.18.1 -> identical Generator streams
  (uniform / integers / normal / poisson / choice), identical transcendentals (log10 / log / log2 / pow / exp / sin /
  cos / arctan2), identical gain maps and a BIT-IDENTICAL full 720 s accept-all episode; only np.sum over a long
  array (reduction order) and scipy norm.cdf differed, neither reaching the trajectory.
  Kaggle probes (kernels e6-npfp-{match,off}, e6-npfp300-match; Linux glibc 2.35, AVX2, no AVX-512): Linux numpy
  2.0.2 vs 2.4.2 -> only np.sum differs, trajectory identical; Linux numpy 2.4.2 + scipy 1.18.1 (SAME versions as
  local) vs Windows -> RNG identical, but every transcendental (log10 / log / log2 / pow / exp / sin / cos /
  arctan2) and norm.cdf differ in the last bits and the trajectory forks at second 122 (just after warm-up).
  VERIFIED: the cause is the PLATFORM libm (glibc vs MSVC UCRT), NOT the numpy version; pinning numpy cannot make
  Linux clouds bit-identical to local Windows. Linux-vs-Linux with the same versions should match (check fp_match).
  Policy: (1) the local env is pinned (pyproject numpy==2.4.2, scipy==1.18.1; uv.lock); (2) every MANIFEST records
  local_env (versions, platform, SIMD) and local_fp (numeric fingerprint: sim primitives + a 300 s trajectory hash);
  (3) PIN=match (default) makes the kernel pip-install the local numpy / scipy versions when they differ (needs
  enable_internet, i.e. a phone-verified Kaggle account; set automatically), PIN=strict also aborts if numpy still
  differs, PIN=off keeps Kaggle's stack (no internet); (4) the kernel writes startup.json (env, install log,
  fingerprint, per-layer equality, fp_match) next to the results. fp_match False = results are NOT bit-comparable
  with local runs: compare within one platform only (oracle panels, main runs and re-scoring from one machine
  class), never mix platforms inside one comparison.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PINS = ("match", "strict", "off")
KERNEL = r'''
import glob, json, os, shutil, subprocess, sys, time, zipfile
import importlib.metadata as md
SHARDS, NSHARDS, W, PIN = __SHARDS__, __NSHARDS__, "/kaggle/working", "__PIN__"
root = os.path.join(W, "bundle")
src = glob.glob("/kaggle/input/**/__NAME___bundle.zip", recursive=True)
if src:
    zipfile.ZipFile(src[0]).extractall(root)
else:  # Kaggle unzipped the archive
    d = os.path.dirname(glob.glob("/kaggle/input/**/e6dev/__SCRIPT__", recursive=True)[0])
    shutil.copytree(os.path.dirname(d), root, dirs_exist_ok=True)
env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
man = json.load(open(os.path.join(root, "MANIFEST.json")))
want = man.get("local_env", {})
have = {p: md.version(p) for p in ("numpy", "scipy")}
inst = []
if PIN in ("match", "strict") and want and any(have[p] != want.get(p) for p in have):
    for pkgs in ([f"{p}=={want[p]}" for p in have], [f"numpy=={want['numpy']}"]):   # numpy alone if scipy can't
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
        inst.append({"pkgs": pkgs, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]})
        if r.returncode == 0:
            break
fp = subprocess.run([sys.executable, os.path.join(root, "e6dev", "cloud.py"), "fingerprint"], env=env, cwd=root,
                    capture_output=True, text=True)
try:
    fpj = json.loads(fp.stdout.strip().splitlines()[-1])
except Exception:
    fpj = {"error": (fp.stdout + fp.stderr)[-1500:]}
lfp = man.get("local_fp", {})
eq = {k: fpj.get(k) == lfp.get(k) for k in ("rng", "ufunc", "scipy_fp", "traj")} if lfp and "env" in fpj else {}
start = {"manifest": man["git_head"], "cpus": os.cpu_count(), "pin": PIN, "want": want, "install": inst,
         "env": fpj.get("env"), "fp_layers_equal": eq, "fp_match": bool(eq) and all(eq.values()),
         "fp_error": fpj.get("error")}
json.dump(dict(start, fingerprint=fpj), open(os.path.join(W, "startup.json"), "w"), indent=1)
print(json.dumps(start), flush=True)
if PIN == "strict" and (fpj.get("env") or {}).get("numpy") != want.get("numpy"):
    sys.exit("PIN=strict: numpy version still differs from the local one")
procs = []
for s in SHARDS:
    log = open(os.path.join(W, f"log_{s}.txt"), "a")
    procs.append(subprocess.Popen([sys.executable, "-u", os.path.join(root, "e6dev", "__SCRIPT__"), "run",
                                   "--part", f"{s}/{NSHARDS}", "--out", os.path.join(W, f"res_{s}.jsonl")],
                                  env=env, stdout=log, stderr=subprocess.STDOUT, cwd=root))
t0 = time.time()
while any(p.poll() is None for p in procs) and time.time() - t0 < 11 * 3600:
    time.sleep(60)
    print(f"{(time.time()-t0)/60:.0f} min, running {[p.poll() is None for p in procs]}", flush=True)
for p in procs:
    if p.poll() is None:
        p.kill()
print("exit codes", [p.poll() for p in procs], flush=True)
for s in SHARDS:
    print(open(os.path.join(W, f"log_{s}.txt")).read()[-3000:], flush=True)
'''


def sha(b):
    return hashlib.sha256(b).hexdigest()


def numeric_env() -> dict:
    """numpy / scipy / python versions, OS and the SIMD extensions numpy dispatches to on this CPU."""
    import platform

    import numpy
    import scipy
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as F
        simd = sorted(k for k in ("AVX2", "FMA3", "AVX512F", "AVX512_SKX", "AVX512_SPR") if F.get(k))
    except Exception:  # noqa: BLE001
        simd = None
    return {"numpy": numpy.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "simd": simd}


def fingerprint(seconds: int = 300) -> dict:
    """Hashes of the numeric primitives the E6 sim uses (Generator methods with sim._rng's seed shape,
    transcendentals on a bit-stable PCG64 raw stream, scipy gaussian_filter / norm.cdf) and a per-second
    ``plant.sla`` hash of a short accept-all E6 episode. Equal hashes = bit-identical numerics for that layer.
    Needs cdd_oran importable (repo root or bundle root on sys.path)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.stats import norm

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:12]

    def gen():
        return np.random.default_rng([7, 6600, 3, 11])

    rng = {"uniform": h(gen().uniform(0, 1, 5000)), "integers": h(gen().integers(0, 37, 5000)),
           "normal": h(gen().normal(0, 1, 5000)), "poisson": h(gen().poisson(np.linspace(0, 400, 5000))),
           "choice": h(gen().choice(500, 40, replace=False))}
    x = (np.random.PCG64(12345).random_raw(100_000) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    y = x * 200 - 100
    uf = {"log10": np.log10(x + 1e-3), "log": np.log(x + 1e-3), "log2": np.log2(1 + 10 ** (y / 10)),
          "pow": 10 ** (y / 10), "exp": np.exp(-y / 7), "sin": np.sin(y), "cos": np.cos(y),
          "arctan2": np.arctan2(y, x - 0.5), "sum": [np.sum(y), np.mean(y)]}
    z = np.random.default_rng(1).normal(size=(128, 128))
    sp = {"gaussian_filter": h(gaussian_filter(z, 4.0, mode="wrap")), "norm_cdf": h(norm.cdf(y / 30))}
    from cdd_oran.envs.e6 import config as C
    from cdd_oran.envs.e6.env import E6Env
    env = E6Env(C.E6Config(seed=100020, load="medium", mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0,
                           scenario="base"), log=False, wg3=True)
    traj = []
    while env.sec < seconds:
        obs = env.step_propose()
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
        traj.append(h([float(v) for _, v in sorted(env.plant.sla.items()) if np.isscalar(v)]))
    return {"env": numeric_env(), "rng": rng, "ufunc": {k: h(v) for k, v in uf.items()}, "scipy_fp": sp,
            "traj": traj}


def main(out, name, script="grid.py", kernels="3", pin="match"):
    if pin not in PINS:
        sys.exit(f"PIN in {PINS}")
    script = os.path.basename(script)      # bundle paths are e6dev/<basename>; a repo-relative path used to break both
    import warnings
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_fp = fingerprint()
    files = {"cdd_oran/__init__.py": os.path.join(ROOT, "cdd_oran", "__init__.py"),
             "cdd_oran/envs/__init__.py": os.path.join(ROOT, "cdd_oran", "envs", "__init__.py")}
    for f in glob.glob(os.path.join(HERE, "*.py")):
        files["e6dev/" + os.path.basename(f)] = f
    for f in glob.glob(os.path.join(ROOT, "scratchpad", "decision_stack", "*.py")) +             glob.glob(os.path.join(ROOT, "scratchpad", "decision_stack", "*.pkl")):   # decision-stack scripts + models
        files["e6dev/" + os.path.basename(f)] = f
    for f in glob.glob(os.path.join(ROOT, "cdd_oran", "decision", "**", "*.py"), recursive=True):
        files[os.path.relpath(f, ROOT).replace("\\", "/")] = f
    for f in glob.glob(os.path.join(ROOT, "cdd_oran", "envs", "e6", "*.py")):
        files["cdd_oran/envs/e6/" + os.path.basename(f)] = f
    for f in glob.glob(os.path.join(ROOT, "cdd_oran", "envs", "xtruce", "*.py")):   # xTRUCE plant (additive)
        files["cdd_oran/envs/xtruce/" + os.path.basename(f)] = f
    if script.startswith(("e6p_disc", "e6p_conf")):   # E6-P discovery / option (a): freeze, registry, maps checks
        for rel in ("docs/benchmark/E6P_DISCOVERY_PROTOCOL.md", "docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md",
                    "docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md", "docs/benchmark/SEED_REGISTRY.json"):
            if os.path.exists(os.path.join(ROOT, *rel.split("/"))):
                files[rel] = os.path.join(ROOT, *rel.split("/"))
        for f in glob.glob(os.path.join(ROOT, "docs", "benchmark", "artifacts", "*")):   # v4 frozen artifact (+ .sha256)
            if os.path.isfile(f):
                files["docs/benchmark/artifacts/" + os.path.basename(f)] = f
    blobs = {k: open(v, "rb").read() if v.endswith(".pkl") else open(v, "rb").read().replace(b"\r\n", b"\n")
             for k, v in files.items()}                                   # binary artifacts are never EOL-normalised
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "cdd_oran/envs/e6", "cdd_oran/decision"], cwd=ROOT,
                           capture_output=True,
                           text=True).stdout.strip()
    xdirty = subprocess.run(["git", "status", "--porcelain", "cdd_oran/envs/xtruce"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    man = {"git_head": head, "e6_dirty": bool(dirty), "xtruce_dirty": bool(xdirty),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sha256": {k: sha(b) for k, b in blobs.items()}, "local_env": local_fp["env"], "local_fp": local_fp}
    ds = os.path.join(out, "dataset")
    os.makedirs(ds, exist_ok=True)
    with zipfile.ZipFile(os.path.join(ds, f"{name}_bundle.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for rel, b in blobs.items():
            z.writestr(rel, b)
        z.writestr("MANIFEST.json", json.dumps(man, indent=1))
    json.dump({"title": name, "id": f"bishalpanta/{name}", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(ds, "dataset-metadata.json"), "w"))
    nk = int(kernels)
    for ki in range(nk):
        tag, shards = "abcdefgh"[ki], list(range(4 * ki, 4 * ki + 4))
        kd = os.path.join(out, f"kernel_{tag}")
        os.makedirs(kd, exist_ok=True)
        open(os.path.join(kd, "kernel.py"), "w").write(
            KERNEL.replace("__SHARDS__", str(shards)).replace("__NSHARDS__", str(4 * nk)).replace("__NAME__", name)
            .replace("__SCRIPT__", script).replace("__PIN__", pin))
        json.dump({"id": f"bishalpanta/{name}-{tag}", "title": f"{name}-{tag}", "code_file": "kernel.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
                   "enable_internet": pin != "off", "dataset_sources": [f"bishalpanta/{name}"]},
                  open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=1)
    print(json.dumps({"files": len(files), "git_head": head, "e6_dirty": bool(dirty), "script": script, "kernels": nk,
                      "pin": pin, "local_env": local_fp["env"], "out": out}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["fingerprint"]:
        import warnings
        warnings.simplefilter("ignore")
        if os.path.isdir(os.path.join(ROOT, "cdd_oran")) and ROOT not in sys.path:
            sys.path.insert(0, ROOT)                                  # repo checkout (the bundle uses PYTHONPATH)
        print(json.dumps(fingerprint(*[int(a) for a in sys.argv[2:]])))
    else:
        main(*sys.argv[1:])
