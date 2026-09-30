"""E6-P option (a) K-B collection, both stages (cloud.py kernels call `SCRIPT run --part i/k --out FILE`).
See e6p_opta_kb.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_opta_kb  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    e6p_opta_kb.cli(sys.argv[1:], stage="all")
