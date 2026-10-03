"""Lightning AI Studio launcher for the cdl F4 shards (R-46). CPU studios only; stop after use; credits logged in
scratchpad/xmethod/status/cdl.md.

  ~/.cloudtools/Scripts/python.exe scratchpad/xmethod/cdl_lightning.py launch STUDIO --parts 4-7 --of 12 \
        --ns 500,4000 --reps 20 [--machine CPU]
  ~/.cloudtools/Scripts/python.exe scratchpad/xmethod/cdl_lightning.py status STUDIO
  ~/.cloudtools/Scripts/python.exe scratchpad/xmethod/cdl_lightning.py pull STUDIO   # -> results/cdl/f4/<STUDIO>.jsonl
  ~/.cloudtools/Scripts/python.exe scratchpad/xmethod/cdl_lightning.py stop STUDIO

Bundle: cdd_oran/**/*.py + scratchpad/xmethod/cdl_f4.py (repo-relative). Pins (R-35, R-35a): numpy 2.4.2, scipy 1.18.1,
torch 2.10.0 (CPU wheel), threadpoolctl; the installed versions are written to cdlrun/env.txt.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REMOTE = "cdlrun"
PINS = ("python3 -m pip install -q numpy==2.4.2 scipy==1.18.1 threadpoolctl && python3 -m pip install -q "
        "torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu")


def studio(name):
    from lightning_sdk import Studio
    return Studio(name=name, teamspace="default-project", user="bishalpanta01", create_ok=True)


def bundle() -> str:
    path = os.path.join(tempfile.gettempdir(), "cdl_lightning_bundle.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join(ROOT, "cdd_oran", "**", "*.py"), recursive=True):
            z.write(f, os.path.relpath(f, ROOT).replace(os.sep, "/"))
        z.write(os.path.join(HERE, "cdl_f4.py"), "scratchpad/xmethod/cdl_f4.py")
        z.write(os.path.join(ROOT, "scripts", "xm_classic_fidelity.py"), "scripts/xm_classic_fidelity.py")
    return path


def launch(a):
    from lightning_sdk import Machine
    s = studio(a.name)
    if "running" not in str(s.status).lower():
        s.start(getattr(Machine, a.machine))
    print("status", s.status, "machine", s.machine)
    s.upload_file(bundle(), "cdl_bundle.zip")
    print(s.run(f"rm -rf {REMOTE}; mkdir -p {REMOTE} && mv cdl_bundle.zip {REMOTE}/ && cd {REMOTE} && "
                f"python3 -m zipfile -e cdl_bundle.zip b && ({PINS}) > install.log 2>&1; "
                "python3 -c 'import sys, torch, numpy, scipy; print(sys.version, torch.__version__, numpy.__version__, "
                "scipy.__version__)' | tee env.txt; nproc").strip()[-400:])
    if a.cmd:   # one custom command (e.g. the sequential cost runs), output -> cdlrun/cmd.jsonl / cmd_log.txt
        print(s.run(f"cd {REMOTE}/b && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. "
                    f"nohup bash -c '{a.cmd}' > ../cmd.jsonl 2> ../cmd_log.txt & sleep 2; echo started"))
        return
    lo, hi = (int(x) for x in a.parts.split("-"))
    cmd = (f"cd {REMOTE}/b && for i in $(seq {lo} {hi}); do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
           f"MKL_NUM_THREADS=1 PYTHONPATH=. nohup python3 -u scratchpad/xmethod/cdl_f4.py run --ns {a.ns} "
           f"--reps {a.reps} --part $i/{a.of} --out ../f4_$i.jsonl > ../log_$i.txt 2>&1 & done; sleep 2; "
           "ps aux | grep -c '[c]dl_f4'")
    print("parts", a.parts, "of", a.of, "running", s.run(cmd).strip())


def status(a):
    s = studio(a.name)
    print("status", s.status)
    if "running" in str(s.status).lower():
        print(s.run(f"cd {REMOTE} && ps aux | grep -c '[c]dl_f4'; cat f4_*.jsonl 2>/dev/null | wc -l; "
                    "grep -l Traceback log_*.txt 2>/dev/null | head -3; tail -qn1 log_*.txt"))


def pull(a):
    s = studio(a.name)
    dst = os.path.join(HERE, "results", "cdl", "f4")
    os.makedirs(dst, exist_ok=True)
    s.run(f"cat {REMOTE}/f4_*.jsonl {REMOTE}/cmd.jsonl 2>/dev/null > cdl_f4_all.jsonl; cp {REMOTE}/env.txt cdl_env.txt")
    out = os.path.join(dst, f"{a.name}.jsonl")
    s.download_file("cdl_f4_all.jsonl", out)
    s.download_file("cdl_env.txt", os.path.join(dst, f"{a.name}_env.txt"))
    print(sum(1 for _ in open(out)), "records ->", out)


def stop(a):
    s = studio(a.name)
    s.stop()
    print("stopped", s.status)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("launch", "status", "pull", "stop"))
    ap.add_argument("name")
    ap.add_argument("--parts", default="0-3")
    ap.add_argument("--of", type=int, default=4)
    ap.add_argument("--ns", default="500,4000")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--machine", default="CPU")
    ap.add_argument("--cmd", default="", help="run this one bash command instead of the F4 parts")
    a = ap.parse_args(argv)
    {"launch": launch, "status": status, "pull": pull, "stop": stop}[a.what](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
