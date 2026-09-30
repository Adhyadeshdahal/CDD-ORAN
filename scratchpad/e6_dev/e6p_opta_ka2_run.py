"""E6-P option (a) K-A2 MapGate v2 arms (cloud.py kernels call `SCRIPT run --part i/k --out FILE`). See e6p_opta_ka2.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_opta_ka2  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    e6p_opta_ka2.cli(sys.argv[1:])
