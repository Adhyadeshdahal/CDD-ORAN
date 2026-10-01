"""E6-P follow-up 2b (certified-safe referee), stage dev (cloud.py kernels call `SCRIPT run --part i/k --out FILE`).
See e6p_certsafe.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_certsafe  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    e6p_certsafe.cli(sys.argv[1:], stage="dev")
