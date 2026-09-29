"""E6-P v2 O_tape falsifier, stage 3 (cloud kernels / colab_run.py call `SCRIPT run --part i/k --out F`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_v2  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    e6p_v2.cli(sys.argv[1:], stage="3")
