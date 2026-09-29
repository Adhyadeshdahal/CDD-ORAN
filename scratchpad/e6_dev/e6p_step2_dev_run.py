"""E6-P step-2 DEV arms, stage dev (cloud.py kernels call `SCRIPT run --part i/k --out FILE`). See e6p_step2_dev.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_step2_dev  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    e6p_step2_dev.cli(sys.argv[1:], stage="dev")
