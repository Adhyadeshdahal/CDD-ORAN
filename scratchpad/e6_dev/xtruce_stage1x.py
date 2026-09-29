"""xTRUCE screen, stage 1x (cloud.py kernels call `SCRIPT run --part i/k --out FILE`; the stage is baked in and beats
env XTRUCE_STAGE). See xtruce_screen.py and docs/benchmark/XTRUCE_SCREEN_PROTOCOL.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtruce_screen  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    xtruce_screen.cli(sys.argv[1:], stage="1x")
