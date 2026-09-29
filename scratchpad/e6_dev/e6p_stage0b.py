"""E6-P screen, stage 0b (cloud.py kernels call `SCRIPT run --part i/k --out FILE`). See e6p_screen.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_screen  # noqa: E402

if __name__ == "__main__":
    e6p_screen.cli(sys.argv[1:], stage="0b")
