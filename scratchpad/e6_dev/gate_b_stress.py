"""Gate B, stress scenarios only (surge, mistune; E6-scn-v1 frozen at 1284869). Records as gate_b.py.

  PYTHONPATH=. python scratchpad/e6_dev/gate_b_stress.py list | run --part i/k --out res.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_b  # noqa: E402

gate_b.SCENARIOS = ("surge", "mistune")

if __name__ == "__main__":
    if sys.argv[1] == "list":
        print(len(gate_b.jobs()), "jobs")
    else:
        a = dict(zip(sys.argv[2::2], sys.argv[3::2], strict=True))
        gate_b.run(a["--part"], a["--out"])
