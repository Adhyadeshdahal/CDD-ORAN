"""Gate A, base scenario only (launched before the stress-scenario spec is frozen; same records as gate_a.py).

  PYTHONPATH=. python scratchpad/e6_dev/gate_a_base.py list | run --part i/k --out res.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_a  # noqa: E402

gate_a.SCENARIOS = ("base",)

if __name__ == "__main__":
    if sys.argv[1] == "list":
        print(len(gate_a.jobs()), "jobs")
    else:
        a = dict(zip(sys.argv[2::2], sys.argv[3::2], strict=True))
        gate_a.run(a["--part"], a["--out"])
