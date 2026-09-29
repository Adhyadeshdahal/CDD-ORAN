"""xTRUCE screen wrapper selected by env XTRUCE_STAGE (local use only). cloud.py passes no env var: bundle the
per-stage wrappers xtruce_stage{0,0c,1,1x,2,3}.py instead. Without XTRUCE_STAGE (or --stage) it refuses to run.
See xtruce_screen.py and docs/benchmark/XTRUCE_SCREEN_PROTOCOL.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtruce_screen  # noqa: E402

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    xtruce_screen.cli(sys.argv[1:], stage=None)
