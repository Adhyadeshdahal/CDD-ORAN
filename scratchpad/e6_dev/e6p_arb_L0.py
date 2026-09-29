"""E6-P arbiter pilot, stage L0 (cloud.py kernels call `SCRIPT run --part i/k --out FILE`). Cells: env E6P_ARB_CELLS
or e6p_state.json (stage3_pass / stage2_crit12). See e6p_arbiter_pilot.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_arbiter_pilot  # noqa: E402

if __name__ == "__main__":
    e6p_arbiter_pilot.cli(sys.argv[1:], stage="L0")
