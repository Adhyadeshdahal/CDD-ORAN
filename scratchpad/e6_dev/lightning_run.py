"""Lightning AI Studio runner for e6dev grid bundles (DEV scratch). Rules: CPU machines only, never override auto-sleep,
stop the Studio after use.

  ~/.cloudtools/Scripts/python.exe scratchpad/e6_dev/lightning_run.py launch STUDIO BUNDLE_ZIP SCRIPT [MACHINE=CPU]
  ~/.cloudtools/Scripts/python.exe scratchpad/e6_dev/lightning_run.py status STUDIO
  ~/.cloudtools/Scripts/python.exe scratchpad/e6_dev/lightning_run.py pull STUDIO      # -> runs/<STUDIO>/all.jsonl
  ~/.cloudtools/Scripts/python.exe scratchpad/e6_dev/lightning_run.py stop STUDIO
The bundle is the one cloud.py builds (dataset/<name>_bundle.zip). One shard per vCPU, resumable, single-threaded.
Numeric pinning (cloud.py docstring): launch installs the bundle MANIFEST's local numpy / scipy versions when the
Studio's differ, then writes e6run/startup.json (env + numeric fingerprint) and prints which layers equal local.
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

from lightning_sdk import Machine, Studio

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "e6run"


def studio(name):
    return Studio(name=name, teamspace="default-project", user="bishalpanta01", create_ok=True)


def launch(name, bundle, script, machine="CPU"):
    s = studio(name)
    if str(s.status).lower().endswith("stopped") or "stop" in str(s.status).lower():
        s.start(getattr(Machine, machine))
    print("status", s.status, "machine", s.machine)
    man = json.loads(zipfile.ZipFile(bundle).read("MANIFEST.json"))
    want, lfp = man.get("local_env", {}), man.get("local_fp", {})
    s.upload_file(bundle, "e6run_bundle.zip")           # flat name: nested remote paths get Windows separators
    s.run(f"rm -rf {REMOTE} 'e6run\bundle.zip'; mkdir -p {REMOTE} && mv e6run_bundle.zip {REMOTE}/bundle.zip && "
          f"cd {REMOTE} && python3 -m zipfile -e bundle.zip b && (python3 -c 'import numpy, scipy' "
          f"|| pip install -q numpy scipy)")
    if want:
        pk = f"numpy=={want['numpy']} scipy=={want['scipy']}"
        print(s.run(f"python3 -c \"import numpy, scipy, sys; sys.exit(0 if (numpy.__version__, scipy.__version__) == "
                    f"('{want['numpy']}', '{want['scipy']}') else 1)\" || python3 -m pip install -q {pk} "
                    f"|| python3 -m pip install -q numpy=={want['numpy']}; echo pin-done").strip()[-400:])
    fp = json.loads(s.run(f"cd {REMOTE}/b && PYTHONPATH=. python3 e6dev/cloud.py fingerprint | tee ../startup.json"
                          ).strip().splitlines()[-1])
    eq = {k: fp.get(k) == lfp.get(k) for k in ("rng", "ufunc", "scipy_fp", "traj")} if lfp else {}
    print("env", fp["env"], "want", want, "fp_layers_equal", eq, "fp_match", bool(eq) and all(eq.values()))
    n = int(s.run("nproc").strip())
    cmd = (f"cd {REMOTE}/b && for i in $(seq 0 {n - 1}); do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
           f"MKL_NUM_THREADS=1 PYTHONPATH=. nohup python3 -u e6dev/{script} run --part $i/{n} "
           f"--out ../res_$i.jsonl > ../log_$i.txt 2>&1 & done; sleep 2; ps aux | grep -c '[e]6dev/'")
    print("shards", n, "running", s.run(cmd).strip())


def status(name):
    s = studio(name)
    print("status", s.status)
    if "running" in str(s.status).lower():
        print(s.run(f"cd {REMOTE} && ps aux | grep -c '[e]6dev/'; wc -l res_*.jsonl 2>/dev/null | tail -1; "
                    f"grep -l Traceback log_*.txt 2>/dev/null | head -3"))


def pull(name):
    s = studio(name)
    dst = os.path.join(HERE, "runs", name)
    os.makedirs(dst, exist_ok=True)
    s.run(f"cd {REMOTE} && cat res_*.jsonl > all.jsonl")
    s.run(f"cp {REMOTE}/all.jsonl e6run_all.jsonl")
    s.download_file("e6run_all.jsonl", os.path.join(dst, "all.jsonl"))
    n = sum(1 for _ in open(os.path.join(dst, "all.jsonl")))
    print(f"{n} records -> {os.path.join(dst, 'all.jsonl')}")


def stop(name):
    s = studio(name)
    s.stop()
    print("stopped", s.status)


if __name__ == "__main__":
    {"launch": launch, "status": status, "pull": pull, "stop": stop}[sys.argv[1]](*sys.argv[2:])
